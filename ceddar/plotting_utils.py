
import torch
import os
import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Optional, Union, List, Dict, Tuple

from utils import _squeeze_geo_value
from variable_utils import (
    get_units,
    get_cmaps,
    get_cmap_for_variable,
    get_color_for_model,
    get_color_for_model_cycle,
)


# Set up logging
logger = logging.getLogger(__name__)


# --- Robust conversion for imshow ---
import numpy as _np
import torch as _torch
from pathlib import Path
from datetime import datetime


def _savefig(fig, out_path: Path, dpi: int = 300):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _nice():
    # lightweight, you can override with your global style later
    plt.rcParams.update({
        "figure.figsize": (5.5, 4.0),
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    })

def _to_date_safe(s: str) -> Optional[datetime]:
    s = s.strip()
    # accept "YYYY-MM-DD" and "YYYYMMDD"
    try:
        if len(s) == 8 and s.isdigit():
            return datetime.strptime(s, "%Y%m%d")
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _season_from_month(m: int) -> str:
    if m in (12, 1, 2):
        return "DJF"
    if m in (3, 4, 5):
        return "MAM"
    if m in (6, 7, 8):
        return "JJA"
    return "SON"



# ------------------------------
# DK outline via LSM (cached)
# ------------------------------
_DK_LSM_CACHE: Dict[Tuple[int, int, int, int], np.ndarray] = {}

def _load_dk_lsm_outline(
    bounds: tuple[int, int, int, int] = (200, 328, 380, 508),
    base: str = "/scratch/project_465001695/quistgaa/Data/Data_DiffMod",
    rel_path: str = "data_lsm/truth_fullDomain/lsm_full.npz",
    key_candidates: tuple[str, ...] = ("lsm_hr", "lsm", "mask", "roi", "lsm_full", "data", "arr_0"),
) -> np.ndarray | None:
    """Load and crop a land-sea mask and return a boolean [H,W] mask for Denmark.
    bounds is interpreted as (y0, y1, x0, x1) with y1/x1 exclusive; e.g., (200,328,380,508) → 128x128.
    """
    try:
        logger.info("[DEBUG] Loading DK LSM outline from %s/%s", base, rel_path)
        # base = os.environ.get(env_key, None)
        # if not base:
        #     return None
        p = Path(base) / rel_path
        if not p.exists():
            logger.warning("[DEBUG] DK LSM outline file not found: %s", str(p))
            return None
        d = np.load(p, allow_pickle=True)
        # Print the keys available in the npz file for debugging
        arr = None
        if hasattr(d, "files"):
            for k in key_candidates:
                if k in d.files:
                    arr = d[k]
                    break
        if arr is None:
            logger.warning("[DEBUG] DK LSM outline: no suitable key found in %s", str(p))
            return None
        a = np.asarray(arr)
        # normalize to [H,W]
        if a.ndim == 4 and a.shape[:2] == (1, 1):
            a = a.squeeze(0).squeeze(0)
        elif a.ndim == 3 and a.shape[0] == 1:
            a = a.squeeze(0)
        y0, y1, x0, x1 = bounds
        a = np.flipud(a)  # flip vertically if needed
        a = a[y0:y1, x0:x1]
        m = (a >= 0.5)
        m = np.flipud(m)  # flip back to original orientation

        logger.info("[DEBUG] DK LSM outline loaded with shape %s", str(m.shape))
        return m.astype(bool, copy=False)
    except Exception as e:
        logger.exception("[DEBUG] Exception while loading DK LSM outline: %s", str(e))
        return None

def get_dk_lsm_outline(
    bounds: tuple[int, int, int, int] = (200, 328, 380, 508),
) -> np.ndarray | None:
    """
    Return cached DK outline mask (boolean [H,W]) for the requested `bounds`.
    Caches per-bounds so different crops return correctly sized masks.
    """
    global _DK_LSM_CACHE
    try:
        key = (int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3]))
    except Exception:
        # Fallback to default if bounds malformed
        key = (200, 328, 380, 508)
    if key in _DK_LSM_CACHE:
        return _DK_LSM_CACHE[key]
    m = _load_dk_lsm_outline(bounds=key)
    if m is not None:
        _DK_LSM_CACHE[key] = m
    return m

def overlay_outline(ax, mask: np.ndarray | None, *, color: str = "black", linewidth: float = 0.8):
    """Overlay a contour outline (level 0.5) on the given axes if mask is provided."""
    if mask is None:
        return
    try:
        ax.contour(mask.astype(float, copy=False), levels=[0.5], colors=color, linewidths=linewidth)
    except Exception:
        pass


# === Centralized imshow for variables with DK outline ===
def imshow_variable(
    ax,
    img2d: np.ndarray,
    *,
    variable: str,
    bounds: tuple[int, int, int, int] = (200, 328, 380, 508),    
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | None = None,
    add_dk_outline: bool = True,
    outline_color: str = "darkgrey",
    outline_linewidth: float = 0.8,
    under_color: str | None = "#c2c2c2",
    under_threshold: float | None = 1e-6,
):
    """
    Centralized imshow for spatial maps that:
      - picks the correct colormap for `variable` (via get_cmap_for_variable) unless `cmap` is provided
      - inverts y-axis to be consistent with array conventions used elsewhere
      - optionally overlays a cached DK land/sea outline
      - values below `under_threshold` are drawn with `under_color` if provided
      - typical use: precipitation “no-rain” background set to gray
      
    Returns the image handle from imshow.
    """
    if img2d is None:
        raise ValueError("imshow_variable: img2d is None")
    arr = np.asarray(img2d)
    if arr.ndim != 2:
        # Squeeze simple singletons, otherwise pick first channel
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            arr = arr.reshape((-1, arr.shape[-2], arr.shape[-1]))[0]
    cm_in = cmap or get_cmap_for_variable(variable)
    try:
        # use matplotlib.cm.get_cmap (imported as mcm) to avoid relying on pyplot attribute
        if isinstance(cm_in, str):
            try:
                cm_obj = mcm.get_cmap(cm_in)
            except Exception:
                cm_obj = cm_in
        else:
            cm_obj = cm_in
    except Exception:
        cm_obj = cm_in

    # Optionally set an "under" color (values < vmin) for near-zero masks (e.g., no-rain as gray)
    if under_color is not None:
        # Try the modern `with_extremes` API if available, otherwise fall back to `set_under`
        w_ext = getattr(cm_obj, "with_extremes", None)
        if callable(w_ext):
            try:
                cm_obj = w_ext(under=under_color)
            except Exception:
                # ignore and fall back to set_under if present
                pass
        else:
            s_under = getattr(cm_obj, "set_under", None)
            if callable(s_under):
                try:
                    s_under(under_color)
                except Exception:
                    pass

    # If an under_threshold is specified and vmin not provided, use it
    if under_threshold is not None and vmin is None:
        vmin = float(under_threshold)

    im = ax.imshow(arr, cmap=cm_obj, vmin=vmin, vmax=vmax, interpolation="nearest", origin="lower")
    if add_dk_outline:
        mask = get_dk_lsm_outline(bounds)
        # flip upside down to match imshow orientation
        mask = np.flipud(mask)  # type: ignore
        overlay_outline(ax, mask, color=outline_color, linewidth=outline_linewidth)
    # ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])


    return im


# === Apply model color scheme to lines/markers ===
def apply_model_colors(
    ax,
    *,
    exclude_kind: str | None = None,
    assume_kind: str | None = None,
):
    """
    Recolors existing line artists in `ax` to follow model colors defined in variable_utils.get_color_for_model.
    - We infer the intended model from each legend label or Line2D label.
    - If exclude_kind == 'seasonal', we skip recoloring (keeps user's preferred seasonal palette).
    - If assume_kind provided, it's only metadata; current logic relies on labels.

    Typical usage right after plotting lines and before/after ax.legend().
    """
    if exclude_kind and exclude_kind.lower() == "seasonal":
        return

    # Collect handles and labels robustly
    handles, labels = ax.get_legend_handles_labels()
    if not labels:
        # Try to infer from lines if no legend exists yet
        lines = [l for l in ax.get_lines() if hasattr(l, "get_label")]
        handles = lines
        labels = [l.get_label() for l in lines]

    for h, lab in zip(handles, labels):
        key = None
        if lab is not None:
            s = lab.strip().lower()
            if any(k in s for k in ["hr", "danra", "truth", "high-res"]):
                key = "hr"
            elif "pmm" in s:
                key = "pmm"
            elif any(k in s for k in ["gen", "generated", "model", "ens"]):
                key = "generated"
            elif any(k in s for k in ["lr", "era5", "low-res"]):
                key = "lr"
        if key is None:
            continue
        try:
            color = get_color_for_model(key)
            # Set both face/edge colors as appropriate
            if hasattr(h, "set_color"):
                h.set_color(color)
            if hasattr(h, "set_markerfacecolor"):
                h.set_markerfacecolor(color)
            if hasattr(h, "set_markeredgecolor"):
                h.set_markeredgecolor(color)
            if hasattr(h, "set_facecolor"):
                h.set_facecolor(color)
            if hasattr(h, "set_edgecolor"):
                h.set_edgecolor(color)
        except Exception:
            # best-effort; keep going
            pass


# === Convenience wrapper for spatial panel plotting ===
def plot_spatial_panel(
    ax,
    img2d: np.ndarray,
    *,
    bounds: tuple[int, int, int, int] = (200, 328, 380, 508),
    variable: str,
    vmin: float | None = None,
    vmax: float | None = None,
    add_dk_outline: bool = True,
    outline_color: str = "darkgrey",
    outline_linewidth: float = 0.8,
    title: str | None = None,
    under_color: str | None = None,
    under_threshold: float | None = None,
):
    """
    Convenience wrapper used by spatial map routines to ensure:
      - correct variable colormap
      - DK outline overlay
      - tight axis cosmetics
    """
    im = imshow_variable(
        ax,
        img2d,
        variable=variable,
        vmin=vmin,
        vmax=vmax,
        add_dk_outline=add_dk_outline,
        outline_color=outline_color,
        outline_linewidth=outline_linewidth,
        under_color=under_color,
        under_threshold=under_threshold,
        bounds=bounds,
    )
    if title:
        ax.set_title(title, fontsize=10)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    ax.figure.colorbar(im, cax=cax, orientation="vertical")
    return im
# === IO-agnostic maybe_compute helper ===
def maybe_compute(cache_exists: bool, plot_only: bool, compute_fn, *args, **kwargs):
    """
    Calling-side helper for heavy steps:
      - If plot_only is True and cache_exists, SKIP compute_fn and return None.
      - Else, run compute_fn(*args, **kwargs) and return its result.

    This keeps plotting_utils IO-agnostic; the caller is responsible for checking
    the concrete cache path(s) and for loading cached results when plot_only=True.
    """
    if plot_only and cache_exists:
        logger.info("[plot_only] Skipping heavy computation because cache exists.")
        return None
    return compute_fn(*args, **kwargs)

def _to_imshow_image(arr, prefer_channel: int = 0):
    """
    Return (img, was_rgb) where `img` is suitable for plt.imshow.
      - 2D → as is
      - 3D (H,W,3|4) → RGB(A)
      - 3D (C,H,W) → pick channel `prefer_channel` (or squeeze if C==1)
      - torch.Tensor → to cpu().numpy()
    """
    if isinstance(arr, _torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = _np.asarray(arr)

    if arr.ndim == 2:
        return arr, False

    if arr.ndim == 3:
        H, W = arr.shape[-2], arr.shape[-1]
        # RGB(A) as (H,W,3|4)
        if arr.shape[0] == H and arr.shape[1] == W and arr.shape[-1] in (3, 4):
            return arr, True
        # Channel-first (C,H,W)
        if arr.shape[0] in (1, 2, 3, 4):
            C = arr.shape[0]
            if C == 1:
                return arr[0], False
            ch = max(0, min(prefer_channel, C - 1))
            return arr[ch], False
        # Generic fallback for (H,W,C)
        if arr.shape[-1] in (3, 4) and arr.shape[0] == H and arr.shape[1] == W:
            return arr, True

    # Last resort: squeeze singletons or take first slice
    squeezed = _np.squeeze(arr)
    if squeezed.ndim == 2:
        return squeezed, False
    view = squeezed.reshape((-1, squeezed.shape[-2], squeezed.shape[-1]))[0]
    return view, False

def plot_sample(sample, cfg, figsize=(15, 4)):
    """
    Plot a single sample in a consistent layout.
    - Dual-LR tensors [2,H,W] are expanded into two separate axes (ch0, ch1).
    - Optional lock of vmin/vmax across the dual pair via cfg['visualization']['dual_lr_lock_scale'].
    """
    # === cfg bits
    hr_model = cfg['highres']['model']
    hr_units, lr_units = get_units(cfg)
    lr_model = cfg['lowres']['model']
    var = cfg['highres']['variable']
    show_ocean = cfg['visualization'].get('show_ocean', True)
    force_matching_scale = cfg['visualization'].get('force_matching_scale', False)
    global_min = cfg['highres'].get('scaling_params', None)
    global_max = cfg['highres'].get('scaling_params', None)
    extra_keys = cfg.get('stationary_conditions', {}).get('geographic_conditions', {}).get('geo_variables', None)
    hr_cmap, lr_cmap_dict = get_cmaps(cfg)
    default_lr_cmap = 'inferno'
    extra_cmap_dict = {"topo": "terrain", "sdf": "coolwarm", "lsm": "binary"}

    # visualization options
    cfg_vis = cfg.get('visualization', {}) if isinstance(cfg, dict) else {}
    overlay_lsm_contour = bool(cfg_vis.get('overlay_lsm_contour', False))
    dual_lr_lock_scale = bool(cfg_vis.get('dual_lr_lock_scale', True))  # NEW

    # Build items
    hr_key = f"{var}_hr"
    lr_keys = sorted([k for k in sample.keys() if k.endswith('_lr')])

    # plot_items: (key, ch_idx) where ch_idx=None means single; 0/1 selects channel from [2,H,W]
    plot_items = []

    # HR
    plot_items.append((hr_key, None))

    # LR scaled keys: expand dual tensors
    for k in lr_keys:
        arr = sample.get(k)
        if torch.is_tensor(arr):
            is_dual = (arr.ndim == 3 and arr.shape[0] == 2)
        else:
            a = np.asarray(arr) if arr is not None else None
            is_dual = (a is not None and a.ndim == 3 and a.shape[0] == 2)
        if is_dual:
            plot_items.extend([(k, 0), (k, 1)])
        else:
            plot_items.append((k, None))

    # Originals
    for base_k in [hr_key] + lr_keys:
        orig_k = base_k + "_original"
        if orig_k in sample:
            arr = sample.get(orig_k)
            if torch.is_tensor(arr):
                is_dual = (arr.ndim == 3 and arr.shape[0] == 2)
            else:
                a = np.asarray(arr) if arr is not None else None
                is_dual = (a is not None and a.ndim == 3 and a.shape[0] == 2)
            if is_dual:
                plot_items.extend([(orig_k, 0), (orig_k, 1)])
            else:
                plot_items.append((orig_k, None))

    # Extras
    if extra_keys is not None:
        for ek in extra_keys:
            plot_items.append((ek, None))

    n = len(plot_items)
    fig, axs = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axs = np.array([axs])
    fig.suptitle(f"Sample from train dataset, {var} (HR: {hr_model}, LR: {lr_model})", fontsize=16)

    # Helper: compute joint vmin/vmax for a dual pair
    def _joint_limits_for_pair(key, ch_idx, img2d_cur):
        if not dual_lr_lock_scale:
            return None
        other = sample[key]
        other = other.detach().cpu().numpy() if torch.is_tensor(other) else np.asarray(other)
        if other.ndim != 3 or other.shape[0] < 2:
            return None
        other2d = other[1 - ch_idx].squeeze()
        # mask ocean for HR keys (not typical here, but harmless)
        return (float(np.nanmin([np.nanmin(img2d_cur), np.nanmin(other2d)])),
                float(np.nanmax([np.nanmax(img2d_cur), np.nanmax(other2d)])))

    for i, (key, ch_idx) in enumerate(plot_items):
        ax = axs[i]
        if key not in sample or sample[key] is None:
            ax.axis('off'); continue

        data = sample[key]
        arr = data.detach().cpu() if torch.is_tensor(data) else torch.as_tensor(data)  # torch for uniform ops
        if ch_idx is not None and arr.ndim == 3 and arr.shape[0] > ch_idx:
            arr = arr[ch_idx]
        img = arr.squeeze().numpy()

        # mask ocean for HR images
        if not show_ocean and (key.endswith("_hr") or key.endswith("_hr_original")) and ("lsm_hr" in sample):
            m = sample["lsm_hr"]
            m = m.squeeze().detach().cpu().numpy() if torch.is_tensor(m) else np.asarray(m).squeeze()
            img = np.where(m < 1, np.nan, img)

        # choose cmap
        if key.endswith('_hr') or key.endswith('_hr_original'):
            cmap = hr_cmap
        elif key.endswith('_lr') or key.endswith('_lr_original'):
            base = key[:-3] if key.endswith('_lr') else key[:-12]
            cmap = lr_cmap_dict.get(base, default_lr_cmap) if lr_cmap_dict is not None else default_lr_cmap
        else:
            cmap = extra_cmap_dict.get(key, 'viridis')

        # limits
        if force_matching_scale and isinstance(global_min, dict) and isinstance(global_max, dict):
            vmin = global_min.get(key, np.nanmin(img))
            vmax = global_max.get(key, np.nanmax(img))
        elif (key.endswith('_lr') or key.endswith('_lr_original')) and (ch_idx is not None):
            joint = _joint_limits_for_pair(key, ch_idx, img)
            if joint is not None:
                vmin, vmax = joint
            else:
                vmin, vmax = np.nanmin(img), np.nanmax(img)
        else:
            vmin, vmax = np.nanmin(img), np.nanmax(img)

        # title
        if key.endswith('_hr'):
            title = f"HR {hr_model} ({var})\nscaled"
        elif key.endswith('_hr_original'):
            title = f"HR {hr_model} ({var})\noriginal [{hr_units}]"
        elif key.endswith('_lr'):
            base = key[:-3]; suffix = f" (ch {ch_idx})" if ch_idx is not None else ""
            title = f"LR {lr_model} ({base})\nscaled{suffix}"
        elif key.endswith('_lr_original'):
            base = key[:-12]; suffix = f" (ch {ch_idx})" if ch_idx is not None else ""
            unit = lr_units[lr_keys.index(base)] if base in lr_keys else '—'
            title = f"LR {lr_model} ({base})\noriginal [{unit}]{suffix}"
        elif extra_keys is not None and key in extra_keys:
            title = "Topography" if key == "topo" else ("SDF" if key == "sdf" else ("Land/Sea Mask" if key == "lsm" else key))
        else:
            title = key

        # draw
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest', origin='lower')
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=10)

        # optional land/sea contour
        if overlay_lsm_contour and ((key.endswith('_hr') or key.endswith('_hr_original')) or
                                    (key.endswith('_lr') or key.endswith('_lr_original'))):
            if "lsm_hr" in sample and sample["lsm_hr"] is not None:
                m = sample["lsm_hr"]
                # m = m.squeeze().detach().cpu().numpy() if torch.is_tensor(m) else np.asarray(m).squeeze()
                try:
                    # ax.contour(lsm_data, levels=[0.5], colors='white', linewidths=0.5)
                    ax.contour(m.astype(float, copy=False), levels=[0.5], colors='darkgrey', linewidths=0.8)
                except Exception as e:
                    logger.warning(f"LSM contour failed on {key}: {e}")

        # colorbar + boxplot (same layout you had)
        divider = make_axes_locatable(ax)
        if key.endswith(('_hr', '_lr', '_hr_original', '_lr_original')):
            bax = divider.append_axes("right", size="10%", pad=0.1)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            vals = img[np.isfinite(img)].ravel()
            if vals.size > 0:
                flierprops = dict(marker='o', markerfacecolor='none', markersize=2,
                                  linestyle='None', markeredgecolor='darkgreen', alpha=0.4)
                medianprops = dict(linestyle='-', linewidth=2, color='black')
                meanprops = dict(marker='x', markerfacecolor='firebrick', markersize=5, markeredgecolor='firebrick')
                bax.boxplot(vals, vert=True, widths=2, showmeans=True,
                            meanprops=meanprops, flierprops=flierprops, medianprops=medianprops)
            bax.set_xticks([]); bax.set_yticks([]); bax.set_frame_on(False)
        else:
            cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im, cax=cax, orientation='vertical')

    fig.tight_layout()
    return fig, axs



def _finite_flat(arr):
    """Return finite values flattened (NaNs masked out)"""
    if arr is None:
        return np.empty((0,), dtype=float)
    # Ensure NumPy array (avoid torch boolean indexing deprecation)
    if torch.is_tensor(arr):
        arr = arr.detach().cpu().numpy()
    else:
        arr = np.asarray(arr)
    mask = np.isfinite(arr)
    return arr[mask].ravel()

def _add_colorbar_and_boxplot(fig, ax, im, img_data, *, boxplot=True, ylim=None):
    """
        Attach a boxplot (left) and a colorbar (right) to an image axis using axes_divider.
        The boxplot is vertical, minimal styling and hides ticks/frames.
    """
    divider = make_axes_locatable(ax)
    # order: [ax | boxplot | colorbar]
    bax = divider.append_axes("right", size="10%", pad=0.1) if boxplot else None
    cax = divider.append_axes("right", size="5%", pad=0.1)

    fig.colorbar(im, cax=cax, orientation='vertical')

    if boxplot and bax is not None:
        vals = _finite_flat(img_data)
        if vals.size:
            bax.boxplot(vals,
                        vert=True,
                        widths=0.9,
                        showmeans=True,
                        meanprops=dict(marker='x', markerfacecolor='firebrick', markersize=5, markeredgecolor='firebrick'),
                        flierprops=dict(marker='o', markerfacecolor='none', markersize=2, linestyle='None', markeredgecolor='darkgreen', alpha=0.4),
                        medianprops=dict(linestyle='-', linewidth=2, color='black'),
            )
            if ylim is not None:
                try:
                    y0, y1 = float(ylim[0]), float(ylim[1])
                    if np.isfinite([y0, y1]).all() and y1 > y0:
                        bax.set_ylim(y0, y1)
                except Exception as e:
                    logger.warning(f"Could not set boxplot ylim {ylim}: {e}")
                    pass
            # Cosmetic cleanup
            bax.set_xticks([])
            bax.set_yticks([])
            bax.set_frame_on(False)
        else:
            bax.axis('off')


def plot_samples_and_generated(
        samples,
        generated,
        cfg,
        *,
        dates: Optional[List[str]] = None,
        transform_back_bf_plot=False,
        back_transforms=None,
        n_samples_threshold=5,
        figsize=(15, 15),
):
    """
    Like ``plot_samples`` but adds an extra left-most column with “Generated”
    images (one per sample).

    Parameters
    ----------
    samples : dict | list[dict]
        The usual batch/list accepted by ``plot_samples``.
    generated : torch.Tensor | np.ndarray | list
        Shape (B,1,H,W) or list/tuple of length B with 2-D arrays.
    transform_back_bf_plot : bool, default False
        Apply inverse scaling before display.
    back_transforms : dict[str, Callable], optional
        Mapping *plot-key* → inverse-transform function.  Only used when
        *transform_back_bf_plot* is ``True``.
    """
    # Extract configuration for plotting
    hr_model = cfg['highres']['model']
    lr_model = cfg['lowres']['model']
    var = cfg['highres']['variable']
    hr_units, lr_units = get_units(cfg)
    hr_cmap, lr_cmap_dict = get_cmaps(cfg)
    default_lr_cmap = 'viridis'
    extra_cmap_dict = {"topo": "terrain", "lsm": "binary", "sdf": "coolwarm"}
    
    cfg_vis = cfg.get('visualization', {})
    show_ocean = cfg_vis.get('show_ocean', False)
    force_matching_scale = cfg_vis.get('force_matching_scale', True)
    global_min = cfg_vis.get('global_min', None)
    global_max = cfg_vis.get('global_max', None)
    extra_keys = cfg_vis.get('extra_keys', None)
    scaling = cfg_vis.get('scaling', True)
    add_boxplot_per_panel = bool(cfg_vis.get('add_boxplot_per_panel', True))
    add_boxplot_summary = bool(cfg_vis.get('add_boxplot_summary', False))
    summary_boxplot_keys = cfg_vis.get('summary_boxplot_keys', None)  # list of keys for summary boxplot column

    plot_dual_lr_channel = 0
    try: 
        plot_dual_lr_channel = int(cfg_vis.get('plot_dual_lr_channel', 0))
    except Exception:
        pass

    # ------------------------------------------------------------------ utils
    def to_numpy(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def maybe_inverse(k, arr, verbose=False):
        if transform_back_bf_plot and back_transforms and k in back_transforms:
            if verbose:
                logger.info(f"Applying inverse transformation for key: {k}")
                logger.info(f"Found inverse transformation for key: {k}")
            return back_transforms[k](arr)
        if verbose:
            if not transform_back_bf_plot:
                logger.info("transform_back_bf_plot is False, skipping inverse transform.")
            elif back_transforms is None:
                logger.info("No back_transforms provided, skipping inverse transform.")
            elif k not in back_transforms:
                logger.info(f"No inverse transformation found for key: {k}")
        return arr
    
    def _prep_for_limits(sample_dict, key):
        """Prep image like in plotting (inverse, mask, squeeze) for consistent vlim calc"""
        if key is None or key not in sample_dict or sample_dict[key] is None:
            return None
        arr = to_numpy(sample_dict[key]).squeeze()
        arr = _squeeze_geo_value(arr, key)
        arr = maybe_inverse(key, arr)
        if not show_ocean and key in {gen_key, hr_key, f"{hr_key}_original"}:
            if "lsm_hr" in sample_dict and sample_dict["lsm_hr"] is not None:
                mask = to_numpy(sample_dict["lsm_hr"]).squeeze()
                arr = np.where(mask < 1, np.nan, arr)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr.squeeze(axis=0)
        return arr
    
    def _finite_minmax(arrs):
        """Compute global min/max over a list of arrays, ignoring NaNs."""
        vals = []
        for a in arrs:
            if a is None:
                continue
            if torch.is_tensor(a):
                a = a.detach().cpu().numpy()
            else:
                a = np.asarray(a)
            af = a[np.isfinite(a)]
            if af.size:
                vals.append(af)
        if not vals:
            return None, None
        all_vals = np.concatenate(vals)
        return float(np.nanmin(all_vals)), float(np.nanmax(all_vals))


    # -------------------------------------------------------- unpack samples
    if isinstance(samples, dict):              # turn single batch-dict → list
        B = None
        for v in samples.values():
            if torch.is_tensor(v):
                B = v.shape[0]
                break
            if isinstance(v, list) and v and torch.is_tensor(v[0]):
                B = len(v)
                break
        if B is None:
            raise ValueError("Could not determine batch size (B) from samples dictionary.")
        sample_list = []
        for i in range(B):
            d = {}
            for k, v in samples.items():
                if torch.is_tensor(v):
                    d[k] = v[i]
                elif isinstance(v, (list, tuple)) and len(v) == B:
                    d[k] = v[i]
                else:
                    d[k] = v
            sample_list.append(d)
    else:
        sample_list = list(samples)

    sample_list = sample_list[:n_samples_threshold]

    # ------------------------------------------------------- generated batch
    # logger.info(f"Generated shape: {generated.shape}")
    gen_np = to_numpy(generated)
    if gen_np.ndim == 4:               # (B, 1, H, W), multiple samples with 1 channel
        gen_np = gen_np[:, 0, :, :]
    elif gen_np.ndim == 3:             # (B, H, W), multiple samples
        pass
    elif gen_np.ndim == 2:             # (H, W), single samples
        gen_np = np.expand_dims(gen_np, axis=0)
    else:
        raise ValueError(f"Unexpected shape for generated samples: {gen_np.shape}")

    gen_np = gen_np[:len(sample_list)]
    # logger.info(f"Generated shape after slicing: {gen_np.shape}")

    # inject into dicts
    gen_key = "generated"
    for d, im in zip(sample_list, gen_np):
        d[gen_key] = im

    # --------------------------------------------------- assemble key order
    hr_key = f"{var}_hr"
    lr_keys = sorted(k for k in sample_list[0] if k.endswith("_lr"))
    # Decide which LR key to use for matching (if any)
    matching_lr_key = f"{var}_lr" if f"{var}_lr" in lr_keys else None
    original_keys = [k + "_original"
                     for k in (hr_key, *lr_keys)
                     if k + "_original" in sample_list[0]]

    plot_keys = [gen_key, hr_key, *lr_keys, *original_keys]
    if extra_keys:
        plot_keys.extend(extra_keys)


    # -------------------------------------------------- Pooled colourlims (per sample)
    per_row_vlims = None
    if not (force_matching_scale and global_min is not None and global_max is not None):
        per_row_vlims = []
        for sd in sample_list:
            arrs = []
            arrs.append(_prep_for_limits(sd, gen_key))
            arrs.append(_prep_for_limits(sd, hr_key))
            if matching_lr_key is not None:
                arrs.append(_prep_for_limits(sd, matching_lr_key))
            vmin_row, vmax_row = _finite_minmax(arrs)
            # Fallback: if empty (all-NaN), compute from HR only
            if vmin_row is None or vmax_row is None:
                hr_only = _prep_for_limits(sd, hr_key)
                vmin_row, vmax_row = _finite_minmax([hr_only])
            per_row_vlims.append( (vmin_row, vmax_row) )

            
    # -------------------------------------------------------------- figure
    n_rows, n_cols = len(sample_list), len(plot_keys)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)

    # If requested, add a summary boxplot column, rebuild figure with +1 column to the right
    if add_boxplot_summary:
        plt.close(fig)
        n_rows, n_cols = len(sample_list), len(plot_keys) + 1
        fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
        summary_col_idx = n_cols - 1
    else:
        summary_col_idx = None

    # Ensure axs is always 2D
    if n_rows == 1 and n_cols == 1:
        axs = np.array([[axs]])
    elif n_rows == 1:
        axs = axs[np.newaxis, :]
    if n_cols == 1:
        axs = axs[:, np.newaxis]

    fig.suptitle(f"Generated vs. conditions – {var} (HR {hr_model} / LR {lr_model}) ")

    for r, sample in enumerate(sample_list):
        # For the summary column, collect distributions here:
        summary_vals = [] # list of (label, values)
        # If user provided explicit keys for the summary boxplot, use those; else default gen + HR + (matching LR)

        if summary_boxplot_keys is not None:
            row_summary_keys = [k for k in summary_boxplot_keys if k in sample]
        else:
            row_summary_keys = [gen_key, hr_key]
            if matching_lr_key is not None:
                row_summary_keys.append(matching_lr_key)
            row_summary_keys = [k for k in row_summary_keys if k in sample]

        for c, key in enumerate(plot_keys):
            ax = axs[r, c]
            if key not in sample or sample[key] is None:
                ax.axis('off')
                continue
            # Add date if provided as y-axis label on first column
            if c == 0 and dates is not None and r < len(dates):
                date_str = str(dates[r])
                if date_str:
                    ax.set_ylabel(date_str, fontsize=10)


            # ========= Retrieve image data =========
            img_data = to_numpy(sample[key]).squeeze()
            img_data = _squeeze_geo_value(img_data, key)
            img_data = maybe_inverse(key, img_data)

            # For HR images mask out ocean using lsm_hr if needed. TODO: Allow user to specify mask key?
            if not show_ocean and key in {gen_key, hr_key, f"{hr_key}_original"}:
                if "lsm_hr" in sample and sample["lsm_hr"] is not None:
                    mask = to_numpy(sample["lsm_hr"]).squeeze()
                    img_data = np.where(mask < 1, np.nan, img_data)
            # For matching LR image, also apply HR mask if needed. NOTE: Should full LR be shown, but only masked in boxplot?
            if (not show_ocean) and (matching_lr_key is not None) and (key == matching_lr_key):
                if "lsm_hr" in sample and sample["lsm_hr"] is not None:
                    mask = to_numpy(sample["lsm_hr"]).squeeze()
                    img_data = np.where(mask < 1, np.nan, img_data)

            # cmap selection
            if key in {gen_key, hr_key, f"{hr_key}_original"}:
                cmap = hr_cmap
            elif key.endswith('_lr') or key.endswith('_lr_original'):
                base = key.replace('_lr', '').replace('_lr_original', '')
                cmap = (lr_cmap_dict or {}).get(base, default_lr_cmap)
            else:
                cmap = (extra_cmap_dict or {}).get(key, 'viridis')

            # vmin/vmax 
            if force_matching_scale and global_min is not None and global_max is not None:
                vmin = global_min.get(key, np.nanmin(img_data)) if isinstance(global_min, dict) else global_min
                vmax = global_max.get(key, np.nanmax(img_data)) if isinstance(global_max, dict) else global_max
            else:
                use_row_pool = (key == gen_key) or (key == hr_key) or (matching_lr_key is not None and key == matching_lr_key)
                if use_row_pool and per_row_vlims is not None:
                    vmin, vmax = per_row_vlims[r]
                    # If degenerate or non-finite, fallback to per-image
                    if (vmin is None) or (vmax is None) or (not np.isfinite([vmin, vmax]).all()):
                        vmin, vmax = np.nanmin(img_data), np.nanmax(img_data)
                else:
                    vmin, vmax = np.nanmin(img_data), np.nanmax(img_data)
                

            # Ensure 2D
            if img_data.ndim == 3 and img_data.shape[0] == 1:
                img_data = img_data.squeeze(0)

            img2d, _ = _to_imshow_image(img_data, prefer_channel=plot_dual_lr_channel)
            im = ax.imshow(img2d, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])

            # ========= If LR conditions, add LSM contour =========
            # Specifically NOT the HR lsm, if we change LR geographical domain
            if key.endswith('_lr') and "lsm" in sample and sample["lsm"] is not None and bool(cfg_vis.get('overlay_lsm_contour', True)):
                lsm_data = to_numpy(sample["lsm"]).squeeze()
                ax.contour(lsm_data, levels=[0.5], colors='darkgrey', linewidths=0.5)

            # ========= column headers (title logic) =========
            if r == 0:
                if scaling:
                    if transform_back_bf_plot and back_transforms and key in back_transforms:
                        titles = {
                            gen_key: "Generated",
                            hr_key: f"HR {hr_model}, {var}\nback-transformed [{hr_units}]",
                            **{k: f"LR {lr_model} ({k[:-3]})\nback-transformed [{lr_units[lr_keys.index(k[:-3])] if k[:-3] in lr_keys else 'unknown'}]" for k in lr_keys},
                            **{k: f"LR {lr_model} ({k[:-12]})\nscaled" for k in original_keys},
                        }
                    else:
                        titles = {
                            gen_key: "Generated",
                            hr_key: f"HR {hr_model}, {var}\nscaled",
                            **{k: f"LR {lr_model} ({k[:-3]})\nscaled" for k in lr_keys},
                            **{k: f"LR {lr_model} ({k[:-12]})\noriginal [{lr_units[lr_keys.index(k[:-12])] if k[:-12] in lr_keys else 'unknown'}]" for k in original_keys},
                        }
                else:
                    titles = {
                        gen_key: "Generated",
                        hr_key: f"HR {hr_model}, {var}\nno scaling [{hr_units}]",
                        **{k: f"LR {lr_model} ({k[:-3]})\nno scaling [{lr_units[lr_keys.index(k[:-3])] if k[:-3] in lr_keys else 'unknown'}]" for k in lr_keys},
                        **{k: f"LR {lr_model}" for k in lr_keys},
                    }


                ax.set_title(titles.get(key, key), fontsize=9)
            
            # ========= Add per-panel boxplot if requested next to colorbar =========
            if key.endswith(("generated", "_hr", "_lr", "_hr_original", "_lr_original")) and add_boxplot_per_panel:
                _add_colorbar_and_boxplot(fig, ax, im, img2d, boxplot=True, ylim=(vmin, vmax))
            else:
                # Still add a colorbar but no boxplot for non-variable maps / extras
                divide = make_axes_locatable(ax)
                cax = divide.append_axes("right", size="5%", pad=0.1)
                fig.colorbar(im, cax=cax, orientation='vertical')
            
            # ========= Collect for the summary boxplot if requested =========
            if add_boxplot_summary and key in row_summary_keys:
                vals = _finite_flat(img2d)
                if vals.size:
                    if key == gen_key:
                        label = hr_key.replace('_hr', ' gen')
                    elif key == hr_key:
                        label = hr_key.replace('_hr', ' hr')
                    elif key.endswith('_lr'):
                        label = key.replace('_lr', ' lr')
                    else:
                        label = key
                    summary_vals.append((label, vals))
            # End of column loop 

        # ========= Draw the summary column for this row, if requested ========
        if add_boxplot_summary and summary_col_idx is not None:
            axd = axs[r, summary_col_idx]
            axd.clear()
            if summary_vals:
                labels, data = zip(*summary_vals)
                axd.boxplot(data, vert=True, widths=0.7, showmeans=True,
                            meanprops=dict(marker='x', markerfacecolor='firebrick', markersize=5, markeredgecolor='firebrick'),
                            flierprops=dict(marker='o', markerfacecolor='none', markersize=2, linestyle='None', markeredgecolor='darkgreen', alpha=0.35),
                            medianprops=dict(linestyle='-', linewidth=1.2, color='black'),
                )
                
                axd.tick_params(axis='y', labelsize=8)
                # Only add x-ticks  on last row
                if r == n_rows - 1:
                    axd.set_xticks(range(1, len(labels) + 1))
                    axd.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
                else:
                    axd.set_xticks([])
                # Only add title on first row
                if r == 0:
                    axd.set_title("Pixel distribution summary", fontsize=9)
                axd.set_frame_on(False)
                    
            else:
                axd.axis('off')
    fig.text(0.5, 0.01, f"Dual-LR plotting: channel {plot_dual_lr_channel} (if applicable) (ch0~HR z-space, ch1~LR z-space)", ha='center', fontsize=8, va='bottom', color='gray')
    # Tighten layout
    fig.tight_layout()

    return fig, axs


# ===============================
# Metrics plotting helpers
# ===============================

def _ensure_dir(path:str):
    os.makedirs(path, exist_ok=True)

def _safe_savefig(fig, save_dir: str, filename: str, dpi=300):
    _ensure_dir(save_dir)
    full = os.path.join(save_dir, filename)
    fig.savefig(full, dpi=dpi, bbox_inches='tight')
    logger.info(f"[plot] Saved figure to {full}")

def plot_live_training_metrics(
        steps: List[int],
        edm_cosine: List[float],
        hr_lr_corr: List[float],
        *,
        save_dir: str,
        n_samples: Optional[int] = None,
        filename: str = "live_metrics.png",
        show: bool = False,
        title: str | None = None,
        land_only: bool = False
):
    """
        Line plots for lightweight in-loop metrics collected over steps.
    """
    title = title or "Live Training Metrics"
    if land_only:
        title += " (land only)"

    if n_samples is not None:
        title = f"{title} (n={n_samples} samples)"

    fig, ax = plt.subplots(figsize=(8, 5))
    steps_np = np.asarray(steps, dtype=float)
    if len(steps_np) == 0:
        logger.warning("No steps provided for live training metrics plot.")
        return
    
    def _maybe_plot(y, label): 
        y = np.asarray(y, dtype=float)
        ok = np.isfinite(y)
        if ok.any():
            ax.plot(steps_np[ok], y[ok], label=label, lw=2)

    _maybe_plot(edm_cosine, "EDM Cosine Similarity")
    _maybe_plot(hr_lr_corr, "HR-LR Correlation")

    ax.set_xlabel("Global step")
    ax.set_ylabel("Metric value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    _safe_savefig(fig, save_dir, filename)
    if show:
        plt.show()
    plt.close(fig)


# ------------------------------
# FSS at multiple spatial scales
# ------------------------------
def plot_fss_epoch(
    fss: Dict[str, float],
    *,
    save_dir: str,
    filename: str = "fss_epoch.png",
    title: str = "FSS at scales",
    show: bool = False,
):
    """
    Bar plot for a single-epoch FSS dictionary, e.g. {'5km': 0.7, '10km': 0.8, ...}
    """
    if not fss:
        logger.warning("[plot] plot_fss_epoch: empty dict; skipping.")
        return
    # Sort by numeric km if possible
    def _km_key(k):
        try:
            return float(k.replace("km", ""))
        except Exception:
            return float("inf")
    keys = sorted(fss.keys(), key=_km_key)
    vals = [fss[k] for k in keys]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(keys, vals)
    ax.set_ylim(0, 1)
    ax.set_ylabel("FSS")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    _safe_savefig(fig, save_dir, filename)
    if show:
        plt.show()
    plt.close(fig)

def plot_fss_history(
    fss_hist: List[Dict[str, float]],
    epoch_list: Optional[List[int]] = None,
    *,
    save_dir: str,
    n_samples: Optional[int] = None,
    filename: str = "fss_history.png",
    title: str = "FSS over epochs",
    show: bool = False,
):
    """
    Line plot over epochs. Each scale gets its own line.
    fss_hist: list of dicts per epoch, e.g. [{'5km':..,'10km':..}, {...}, ...]
    """
    if not fss_hist:
        logger.warning("[plot] plot_fss_history: empty history; skipping.")
        return
    # Collect all scales
    scales = sorted({k for d in fss_hist for k in d.keys()},
                    key=lambda k: float(k.replace("km", "")) if "km" in k else float("inf"))

    if epoch_list is not None and len(epoch_list) == len(fss_hist):
        epochs = np.asarray(epoch_list, dtype=float)
    else:
        epochs = np.arange(1, len(fss_hist) + 1)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for s in scales:
        y = [d.get(s, np.nan) for d in fss_hist]
        y = np.asarray(y, dtype=float)
        ok = np.isfinite(y)
        if ok.any():
            ax.plot(epochs[ok], y[ok], label=s, lw=2)

    if n_samples is not None:
        title = f"{title} (n={n_samples} samples)"

    ax.set_xlabel("Epoch")
    ax.set_ylabel("FSS")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Scale")
    ax.set_title(title)
    fig.tight_layout()
    _safe_savefig(fig, save_dir, filename)
    if show:
        plt.show()
    plt.close(fig)

# ------------------------------
# PSD slope (β) comparisons
# ------------------------------
def plot_psd_slope_epoch(
    psd: Dict[str, float],
    *,
    save_dir: str,
    filename: str = "psd_slope_epoch.png",
    title: str = "PSD slope (log–log)",
    show: bool = False,
):
    """
    Bar plot comparing gen vs HR slopes (if available) with delta text.
    Expected keys: 'psd_slope_gen', optionally 'psd_slope_hr' and 'psd_slope_delta'
    """
    gen = psd.get("psd_slope_gen", np.nan)
    hr = psd.get("psd_slope_hr", np.nan)
    has_hr = np.isfinite(hr)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    labels = ["Gen"] + (["HR"] if has_hr else [])
    vals = [gen] + ([hr] if has_hr else [])
    ax.bar(labels, vals, color=["#4c72b0", "#55a868"][:len(labels)])
    ax.set_ylabel("Slope β")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.2)

    # annotate delta if both present
    if has_hr and np.isfinite(gen):
        delta = psd.get("psd_slope_delta", float(hr - gen))
        ax.text(0.5, max(vals) + 0.02, f"Δ (HR–Gen) ≈ {delta:.3f}",
                ha="center", va="bottom", transform=ax.get_xaxis_transform())

    fig.tight_layout()
    _safe_savefig(fig, save_dir, filename)
    if show:
        plt.show()
    plt.close(fig)

def plot_psd_slope_history(
    psd_hist: List[Dict[str, float]],
    epoch_list: Optional[List[int]] = None,
    *,
    save_dir: str,
    n_samples: Optional[int] = None,
    filename: str = "psd_slope_history.png",
    title: str = "PSD slope over epochs",
    show: bool = False,
):
    """
    Line plots of β_gen and β_hr over epochs (if HR available), plus Δ on a secondary axis.
    """
    if not psd_hist:
        logger.warning("[plot] plot_psd_slope_history: empty history; skipping.")
        return

    if epoch_list is not None and len(epoch_list) == len(psd_hist):
        epochs = np.asarray(epoch_list, dtype=float)
    else:
        epochs = np.arange(1, len(psd_hist) + 1, dtype=float)
    gen = np.array([d.get("psd_slope_gen", np.nan) for d in psd_hist], dtype=float)
    hr  = np.array([d.get("psd_slope_hr", np.nan) for d in psd_hist], dtype=float)
    delta = np.array([d.get("psd_slope_delta", np.nan) for d in psd_hist], dtype=float)

    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ln1 = ax1.plot(epochs, gen, label="β_gen", lw=2)
    ln2 = []
    if np.isfinite(hr).any():
        ln2 = ax1.plot(epochs, hr, label="β_hr", lw=2)

    if n_samples is not None:
        title = f"{title} (n={n_samples} samples)"

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Slope β")
    ax1.grid(True, alpha=0.3)
    ax1.set_title(title)

    ax2 = None
    if np.isfinite(delta).any():
        ax2 = ax1.twinx()
        ln3 = ax2.plot(epochs, delta, "--", label="Δ(HR–Gen)", lw=2)
        ax2.set_ylabel("Δ β")
        lines = ln1 + ln2 + ln3
    else:
        lines = ln1 + ln2

    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc="best")

    fig.tight_layout()
    _safe_savefig(fig, save_dir, filename)
    if show:
        plt.show()
    plt.close(fig)

# -----------------------------------------
# Quantiles & wet-day frequency comparisons
# -----------------------------------------
def plot_quantiles_wetday_epoch(
    q: Dict[str, float],
    *,
    save_dir: str,
    filename: str = "quantiles_wetday_epoch.png",
    title: str = "Quantiles and wet-day frequency",
    show: bool = False,
):
    """
    Grouped bar chart for P95, P99, wet-day freq (gen vs HR if available).
    Expected keys: 'gen_p95','gen_p99','gen_wet_freq' and optionally 'hr_*'
    """
    keys = [("p95", "gen_p95", "hr_p95"),
            ("p99", "gen_p99", "hr_p99"),
            ("wetfreq", "gen_wet_freq", "hr_wet_freq")]
    labels = []
    gen_vals, hr_vals = [], []
    for lab, gk, hk in keys:
        labels.append(lab.upper())
        gen_vals.append(q.get(gk, np.nan))
        hr_vals.append(q.get(hk, np.nan))

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width/2, gen_vals, width, label="Gen")
    if np.isfinite(hr_vals).any():
        ax.bar(x + width/2, hr_vals, width, label="HR")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    _safe_savefig(fig, save_dir, filename)
    if show:
        plt.show()
    plt.close(fig)

def plot_quantiles_wetday_history(
    q_hist: List[Dict[str, float]],
    epoch_list: Optional[List[int]] = None,
    *,
    save_dir: str,
    n_samples: Optional[int] = None,
    filename: str = "quantiles_wetday_history.png",
    title: str = "P95/P99/Wet-day over epochs",
    show: bool = False,
):
    """
    Line plots for P95/P99/wet-day across epochs (gen and HR where available).
    """
    if not q_hist:
        logger.warning("[plot] plot_quantiles_wetday_history: empty history; skipping.")
        return

    if epoch_list is not None and len(epoch_list) == len(q_hist):
        epochs = np.asarray(epoch_list, dtype=float)
    else:
        epochs = np.arange(1, len(q_hist) + 1, dtype=float)

    def _series(gk, hk):
        g = np.array([d.get(gk, np.nan) for d in q_hist], dtype=float)
        h = np.array([d.get(hk, np.nan) for d in q_hist], dtype=float)
        return g, h

    series = [
        ("P95", "gen_p95", "hr_p95"),
        ("P99", "gen_p99", "hr_p99"),
        ("Wet-day freq", "gen_wet_freq", "hr_wet_freq"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True)
    for ax, (name, gk, hk) in zip(axes, series):
        g, h = _series(gk, hk)
        okg = np.isfinite(g)
        if okg.any():
            ax.plot(epochs[okg], g[okg], label="Gen", lw=2)
        okh = np.isfinite(h)
        if okh.any():
            ax.plot(epochs[okh], h[okh], label="HR", lw=2)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        if name == "Wet-day freq":
            ax.set_ylim(0, 1)

    axes[0].set_xlabel("Epoch")
    axes[1].set_xlabel("Epoch")
    axes[2].set_xlabel("Epoch")
    axes[0].set_ylabel("Value")
    axes[0].legend(loc="best")

    if n_samples is not None:
        title = f"{title} (n={n_samples} samples)"

    fig.suptitle(title)
    fig.tight_layout()
    _safe_savefig(fig, save_dir, filename)
    if show:
        plt.show()
    plt.close(fig)






# def plot_sample_with_boxplot(
#         hr: Union[np.ndarray, dict], # Expecting a 2D array or dict with multiple days
#         lr: Optional[Union[np.ndarray, dict]] = None,
#         gen: Optional[Union[np.ndarray, dict]] = None,
#         variable: str = "Variable",
#         hr_model: str = "HR Model",
#         lr_model: Optional[str] = None,
#         gen_model: Optional[str] = None,
#         dates: Optional[Union[str, list]] = None,
#         save_path: Optional[str] = None,
#         show: bool = False,
#         cmap_default: str = "viridis",
#         combine_into_grid: bool = False,
#         n_rows_max: int = 5,
#     ):
#     """
#         Plots HR, LR and generated images side-by-side with boxplots adjacent to each image.
#         Accepts either single arrays or dicts + dates for multiple days.
#         If combine_into_grid is True, multiple dates will be plotted in a grid layout in a single figure.
#         - If hr is a dict and date is a list -> loop throuhg each date
#         - If hr is a dict and date is a single str -> lookup that date once
#         - If hr is a NumPy array, date is ignored
#     """

#     if save_path is None:
#         save_path = f"./comparison/{variable}/"

#     # Get cmap for variable if possible
#     try:
#         cmap_default = get_cmap_for_variable(variable)
#     except ValueError:
#         pass

#     # === MULTIPLE DAYS ===
#     if isinstance(dates, list):
#         # === IF COMBINING INTO GRID PLOT IN ONE FIGURE ===
#         if combine_into_grid:
#             dates = dates[:n_rows_max]
#             fields = [('Gen', gen), (f'{hr_model}', hr), (f'{lr_model}', lr)]
#             fields = [(name, f) for name, f in fields if f is not None]
#             n_fields = len(fields)
#             n_rows = len(dates)

#             fig = plt.figure(figsize=(5 * n_fields * 1.5, 3.5 * n_rows))
#             gs = GridSpec(n_rows, n_fields * 2, width_ratios=[4, 1] * n_fields, figure=fig)

#             for row_idx, d in enumerate(dates):
#                 row_data = []
#                 for label, dataset in fields:
#                     if isinstance(dataset, dict) and d in dataset:
#                         row_data.append((label, dataset[d]))
#                     else:
#                         row_data.append((label, None))

#                 vmin = min(np.min(x[1]) for x in row_data if x[1] is not None)
#                 vmax = max(np.max(x[1]) for x in row_data if x[1] is not None)

#                 for i, (label, data) in enumerate(row_data):
#                     ax_img = fig.add_subplot(gs[row_idx, i * 2])
#                     if data is not None:
#                         # Ensure data is a NumPy array before plotting
#                         if isinstance(data, dict):
#                             logger.warning(f"Cannot plot dictionary for label '{label}'. Skipping.")
#                             ax_img.set_title(f"{label} (invalid data type)")
#                             ax_img.axis('off')
#                             continue
#                         if not isinstance(data, np.ndarray):
#                             data = np.array(data)
#                         # Set date title to only be date (not time)
#                         try:
#                             d_title = d.split(' ')[0] if ' ' in d else d
#                         except Exception as e:
#                             logger.warning(f"Error extracting date title from '{d}': {e}")
#                             d_title = d
                        
#                         im = ax_img.imshow(data, cmap=cmap_default, vmin=vmin, vmax=vmax)
#                         ax_img.set_title(f"{label} ({d_title})", fontsize=10)
#                         ax_img.axis('off')
#                         ax_img.invert_yaxis()  # Invert y-axis to match the original image orientation
#                         plt.colorbar(im, ax=ax_img, shrink=0.8)
#                     else:
#                         ax_img.set_title(f"{label} (missing)")
#                         ax_img.axis('off')

#                     ax_box = fig.add_subplot(gs[row_idx, i * 2 + 1])
#                     if data is not None:
#                         ax_box.boxplot(
#                                 data.flatten(),
#                                 vert=True,
#                                 widths=1,
#                                 showmeans=True,
#                                 meanprops=dict(marker='x', markerfacecolor='firebrick', markersize=5, markeredgecolor='firebrick'),
#                                 flierprops=dict(marker='o', markerfacecolor='none', markersize=2, linestyle='None', markeredgecolor='darkgreen', alpha=0.4),
#                                 medianprops=dict(linestyle='-', linewidth=2, color='black'),
#                                 patch_artist=True,
#                                 )

#                     # ax_box.set_title("Box", fontsize=8)
#                     ax_box.set_xticks([])
#                     ax_box.tick_params(axis='y', labelsize=6)
#                     ax_box.set_frame_on(False)


#             fig.suptitle(f"{variable} | Multiple Dates", fontsize=16)
#             fig.tight_layout()
#             if save_path:
#                 os.makedirs(os.path.dirname(save_path), exist_ok=True)
#                 path = os.path.join(save_path, f"{variable}_{hr_model}_vs_{lr_model}_boxplot__qualitative_visual.png")
#                 plt.savefig(path, dpi=300, bbox_inches='tight')
#             if show:
#                 plt.show()
#             plt.close()
#             return  # Don't fall through to single plot

#         # === IF NOT COMBINING, PLOT EACH DATE SEPARATELY IN MULTIPLE FIGURES ===
#         for d in dates:
#             plot_sample_with_boxplot(
#                 hr=hr, lr=lr, gen=gen,
#                 variable=variable,
#                 hr_model=hr_model,
#                 lr_model=lr_model,
#                 gen_model=gen_model,
#                 dates=d,
#                 save_path=os.path.join(save_path, f"{variable}_{d}_boxplot__qualitative_visual.png") if save_path else None,
#                 show=show,
#                 cmap_default=cmap_default
#             )
#         return 

#     # === DICTIONARY LOOKUP ===
#     if isinstance(hr, dict):
#         if dates not in hr:
#             logger.warning(f"Date '{dates}' not found in HR data dictionary. Skipping plot.")
#             return
#         hr = hr[dates]
#         lr = lr.get(dates) if lr and isinstance(lr, dict) else None
#         gen = gen.get(dates) if gen and isinstance(gen, dict) else None

#     # === SINGLE PLOT ===

#     fields = [('HR', hr, hr_model)]
#     if gen is not None:
#         fields.insert(0, ('Generated', gen, gen_model if gen_model else "Gen Model"))
#     if lr is not None:
#         fields.append(('LR', lr, lr_model if lr_model else "LR Model"))

#     n_fields = len(fields)
#     fig = plt.figure(figsize=(5 * n_fields * 1.5, 5)) # 5 for each image, 1.5 for boxplot
#     gs = GridSpec(1, n_fields * 2, width_ratios=[4, 1] * n_fields, figure=fig)

#     vmin = min(np.nanmin(f[1]) for f in fields if isinstance(f[1], np.ndarray))
#     vmax = max(np.nanmax(f[1]) for f in fields if isinstance(f[1], np.ndarray))


#     for i, (label, data, model) in enumerate(fields):
#         if data is None:
#             continue
#         if not isinstance(data, np.ndarray):
#             data = np.array(data)
#         ax_img = fig.add_subplot(gs[0, i * 2])
#         im = ax_img.imshow(data, cmap=cmap_default, vmin=vmin, vmax=vmax)
#         ax_img.set_title(f"{label} ({model})", fontsize=14)
#         ax_img.axis('off')
#         ax_img.invert_yaxis()  # Invert y-axis to match the original image orientation

#         cbar = plt.colorbar(im, ax=ax_img, shrink=0.8)
#         cbar.ax.tick_params(labelsize=8)

#         ax_box = fig.add_subplot(gs[0, i * 2 + 1])
#         ax_box.boxplot(data.flatten(), vert=True, patch_artist=True,
#                           boxprops=dict(facecolor='lightblue', color='blue'),
#                           medianprops=dict(color='red'),
#                           flierprops=dict(marker='o', markerfacecolor='none', markersize=5, markeredgecolor='blue', alpha=0.5))
#         ax_box.set_xticks([])
#         ax_box.tick_params(axis='y', labelsize=8)


#     suptitle = f"{variable} | {dates}" if dates else variable
#     fig.suptitle(suptitle, fontsize=16)

#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         logger.info(f"Plot saved to {save_path}")
#     if show:
#         plt.show()
#     plt.close()
    
#     return 


# def plot_samples(samples, cfg, n_samples_threshold=3, figsize=(15, 8)):
#     """
#     Plot a batch of samples (provided as a list of sample dictionaries) in a grid where each row is a sample and
#     each column corresponds to a particular key (e.g., HR, LR, originals, geo).
    
#     If the number of samples exceeds n_samples_threshold, only the first n_samples_threshold will be plotted.
    
#     Parameters:
#       - sample_list: List of sample dictionaries.
#       - cfg: Configuration dictionary containing model and variable information.
#       - figsize: Overall figure size.
      
#     Returns:
#       - fig: The matplotlib Figure object.
#     """
#     from mpl_toolkits.axes_grid1 import make_axes_locatable

#     # Extract configuration for plotting
#     hr_model = cfg['highres']['model']
#     lr_model = cfg['lowres']['model']
#     var = cfg['highres']['variable']
#     hr_units, lr_units = get_units(cfg)
#     hr_cmap, lr_cmap_dict = get_cmaps(cfg)
#     default_lr_cmap = 'viridis'
#     extra_cmap_dict = {"topo": "terrain", "lsm": "binary", "sdf": "coolwarm"}
#     show_ocean = cfg.get('visualization', {}).get('show_ocean', False)
#     force_matching_scale = cfg.get('visualization', {}).get('force_matching_scale', True)
#     global_min = cfg.get('visualization', {}).get('global_min', None)
#     global_max = cfg.get('visualization', {}).get('global_max', None)
#     extra_keys = cfg.get('visualization', {}).get('extra_keys', None)


#     # If single batch dict is passed, unpack it to a list
#     if isinstance(samples, dict):
#         # Figure out batch size from first tensor we find:
#         batch_size = None
#         for v in samples.values():
#             if torch.is_tensor(v):
#                 batch_size = v.shape[0]
#                 break
#             if isinstance(v, list) and all(torch.is_tensor(x) for x in v):
#                 batch_size = len(v)
#                 break
#         if batch_size is None:
#             raise ValueError("No tensor found in the sample dictionary to determine batch size.")
        
#         sample_list = []
#         for i in range(batch_size):
#             single = {}
#             for k, v in samples.items():
#                 if torch.is_tensor(v):
#                     # Slice tensor on batch dim
#                     single[k] = v[i]
#                 elif isinstance(v, (list, tuple)) and len(v) == batch_size:
#                     # Truly per-sample list
#                     single[k] = v[i]
#                 else:
#                     # Some constant list or metadata: leave as-is
#                     single[k] = v
#             sample_list.append(single)
#     else:
#         sample_list = samples

#     # logger.info(f"Plotting first {n_samples_threshold} samples out of {len(sample_list)} provided.")
#     sample_list = sample_list[:n_samples_threshold]
    
#     # Construct the keys:
#     # HR key is "var_hr" (e.g., "prcp_hr")
#     hr_key = f"{var}_hr"
#     # Assume LR keys end with '_lr'
#     lr_keys = sorted([key for key in sample_list[0].keys() if key.endswith('_lr')])
#     scaled_keys = [hr_key] + lr_keys

#     # Determine original keys if available.
#     original_keys = []
#     for key in scaled_keys:
#         orig_key = key + "_original"
#         if orig_key in sample_list[0]:
#             original_keys.append(orig_key)
    
#     # Build final list of keys. Append extra keys if provided.
#     plot_keys = scaled_keys + original_keys
#     if extra_keys is not None:
#         plot_keys += extra_keys

#     num_samples = len(sample_list)
#     num_keys = len(plot_keys)

#     # Create a grid with rows = number of samples and columns = number of keys
#     fig, axs = plt.subplots(num_samples, num_keys, figsize=figsize)
#     # Set figure title 
#     fig.suptitle(f"Sample images for {var} (HR: {hr_model} and LR: {lr_model})", fontsize=16)
#     if num_samples == 1:
#         axs = np.expand_dims(axs, axis=0)
#     if num_keys == 1:
#         axs = np.expand_dims(axs, axis=1)

#     for row, sample in enumerate(sample_list):
#         for col, key in enumerate(plot_keys):
#             ax = axs[row, col]
#             if key not in sample or sample[key] is None:
#                 ax.axis('off')
#                 continue
#             # Retrieve image data
#             img_data = sample[key]
#             if torch.is_tensor(img_data):
#                 img_data = img_data.squeeze().cpu().numpy()
#             img_data = _squeeze_geo_value(img_data, key)
#             # For HR images mask out ocean using lsm_hr if needed.
#             if not show_ocean and (key.endswith('_hr') or key.endswith('_hr_original')):
#                 if "lsm_hr" in sample and sample["lsm_hr"] is not None:
#                     mask = sample["lsm_hr"].squeeze().cpu().numpy()
#                     img_data = np.where(mask < 1, np.nan, img_data)
#             # Determine color limits.
#             if force_matching_scale and global_min is not None and global_max is not None:
#                 vmin = global_min.get(key, np.nanmin(img_data))
#                 vmax = global_max.get(key, np.nanmax(img_data))
#             else:
#                 vmin, vmax = np.nanmin(img_data), np.nanmax(img_data)
#             # Choose colormap:
#             if key.endswith('_hr') or key.endswith('_hr_original'):
#                 cmap = hr_cmap
#             elif key.endswith('_lr') or key.endswith('_lr_original'):
#                 if key.endswith('_lr'):
#                     base = key[:-3]
#                 else:
#                     base = key[:-12]
#                 if lr_cmap_dict is not None and base in lr_cmap_dict:
#                     cmap = lr_cmap_dict[base]
#                 else:
#                     cmap = default_lr_cmap
#             else:
#                 if extra_cmap_dict is not None and key in extra_cmap_dict:
#                     cmap = extra_cmap_dict[key]
#                 else:
#                     cmap = 'viridis'
#             im = ax.imshow(img_data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
#             ax.invert_yaxis()
#             ax.set_xticks([])
#             ax.set_yticks([])
#             divider = make_axes_locatable(ax)
#             # For keys that correspond to variable fields, add a boxplot next to the colorbar.
#             if key.endswith('_hr') or key.endswith('_lr') or key.endswith('_hr_original') or key.endswith('_lr_original'):
#                 bax = divider.append_axes("right", size="10%", pad=0.1)
#                 cax = divider.append_axes("right", size="5%", pad=0.1)
#                 flierprops = dict(marker='o', markerfacecolor='none', markersize=2,
#                                   linestyle='none', markeredgecolor='darkgreen', alpha=0.4)
#                 medianprops = dict(linestyle='-', linewidth=2, color='black')
#                 meanpointprops = dict(marker='x', markerfacecolor='firebrick', markersize=5, markeredgecolor='firebrick')
#                 img_flat = img_data[~np.isnan(img_data)].flatten()
#                 if len(img_flat) > 0:
#                     bax.boxplot(img_flat,
#                                 vert=True,
#                                 widths=2,
#                                 patch_artist=True,
#                                 showmeans=True,
#                                 meanprops=meanpointprops,
#                                 medianprops=medianprops,
#                                 flierprops=flierprops)
#                 bax.set_xticks([])
#                 bax.set_yticks([])
#                 bax.set_frame_on(False)
#             else:
#                 cax = divider.append_axes("right", size="5%", pad=0.1)
#             fig.colorbar(im, cax=cax)

#             base = None


#             # Set column title (only for top row)
#             if row == 0:
#                 if key.endswith('_hr'):
#                     title = f"HR {hr_model} ({var})\nscaled"
#                 elif key.endswith('_hr_original'):
#                     title = f"HR {hr_model} ({var})\noriginal [{hr_units}]"
#                 elif key.endswith('_lr'):
#                     base = key[:-3]
#                     title = f"LR {lr_model} ({base})\nscaled"
#                 elif key.endswith('_lr_original'):
#                     base = key[:-12]
#                     title = f"LR {lr_model} ({base})\noriginal [{lr_units[lr_keys.index(base)]}]"
#                 elif extra_keys is not None and key in extra_keys:
#                     if key == "topo":
#                         title = f"Topography"
#                     elif key == "sdf":
#                         title = f"SDF"
#                     elif key == "lsm":
#                         title = f"Land/Sea Mask"
#                     else:
#                         title = f"{key}"
#                 else:
#                     title = f"{key}"
#                 ax.set_title(title, fontsize=10)
#     fig.tight_layout()
#     return fig, axs