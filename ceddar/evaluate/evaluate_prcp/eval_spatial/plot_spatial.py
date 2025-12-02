from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, List
import logging
import numpy as np
import matplotlib.pyplot as plt

from scor_dm.variable_utils import get_cmap_for_variable
from evaluate.evaluate_prcp.plot_utils import _nice, _savefig, _ensure_dir, get_dk_lsm_outline, overlay_outline
logger = logging.getLogger(__name__)

SET_DPI = 300

# Styling helper
def _get_var_style(var: str):
    """
    Returns (cmap, vmin, vmax, cbar_label) for the value row.
    """
    if var in {"mean", "p95", "p99"}:
        return (get_cmap_for_variable("prcp"), 0.0, None, "mm/day")
    if var in {"sum", "rx1", "rx5"}:
        return (get_cmap_for_variable("prcp"), 0.0, None, "mm")
    if var == "wetfreq":
        return ("Blues", 0.0, 1.0, "fraction of days")
    # generic fallback
    return ("viridis", 0.0, None, "")

def _load_npz(tables_dir: Path, tag: str):
    p = tables_dir / f"{tag}.npz"
    if not p.exists():
        return None
    return np.load(p, allow_pickle=True)

def _draw_single(
    ax,
    data,
    title: str,
    cmap="viridis",
    vmin=None,
    vmax=None,
    cbar_label="",
    *,
    dk_mask=None,
    add_stats: bool = False,
):
    if data is None:
        ax.axis("off")
        return None

    arr = np.asarray(data)

    if dk_mask is not None and dk_mask.shape == arr.shape:
        arr_plot = np.flipud(arr)
    else:
        arr_plot = arr

    im = ax.imshow(arr_plot, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")

    # Keep this so all spatial plots have the same orientation as elsewhere
    # ax.invert_yaxis()

    if dk_mask is not None:
        overlay_outline(ax, dk_mask)

    if add_stats:
        flat = arr.ravel()  # stats on original data (orientation doesn’t matter)
        flat = flat[np.isfinite(flat)]
        if flat.size > 0:
            mu = float(np.nanmean(flat))
            sd = float(np.nanstd(flat))
            title = f"{title}  |  {mu:.2f} ± {sd:.2f}"

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    if cbar_label:
        cb.set_label(cbar_label)

    return im

def plot_spatial_maps(eval_root: str | Path) -> None:
    """
    Compose multi-panel figures per group and variable:

      | Row 0: value maps for HR, Ensemble mean, Ensemble spread (std), and LR (where available).
      | Row 1: ratio maps vs HR for Ensemble mean and LR.
      | Row 2: bias maps vs HR for Ensemble mean and LR.

    Notes:
      - PMM/Generated columns are intentionally omitted.
      - Value colorbars start at 0 to avoid misleading grays on precipitation.
    """
    eval_root = Path(eval_root)
    tables = eval_root / "tables"
    figs   = _ensure_dir(eval_root / "figures")

    # Discover groups by NPZ names
    tags = [p.stem for p in tables.glob("spatial_*.npz")]
    # Build mapping group -> dict(source->npz)
    buckets: Dict[str, Dict[str, Path]] = {}
    for stem in tags:
        parts = stem.split("_", 2)  # ["spatial", src, group]
        if len(parts) != 3:
            continue
        _, src, group = parts
        buckets.setdefault(group, {})[src] = tables / f"{stem}.npz"

    if not buckets:
        logger.warning("[plot_spatial] No spatial_* NPZ files found under %s", str(tables))
        return

    variables = ["mean","sum","rx1","rx5","p95","p99","wetfreq"]
    dk_mask = get_dk_lsm_outline()

    for group, src_map in sorted(buckets.items()):
        # Load available sources (HR, EnsMean, EnsStd, LR)
        npz_hr      = _load_npz(tables, f"spatial_hr_{group}")        if "hr"  in src_map else None
        npz_ensmean = _load_npz(tables, f"spatial_ensmean_{group}")   if "ensmean" in src_map or "ens" in src_map else _load_npz(tables, f"spatial_ensmean_{group}")
        npz_ensstd  = _load_npz(tables, f"spatial_ensstd_{group}")    if "ensstd"  in src_map or "ens" in src_map else _load_npz(tables, f"spatial_ensstd_{group}")
        npz_lr      = _load_npz(tables, f"spatial_lr_{group}")        if "lr"  in src_map else None

        for var in variables:
            arrs: List[np.ndarray] = []
            titles: List[str] = []
            idx_map = {}  # keys: "hr", "ens", "ensstd", "lr"
            cmap, vmin, vmax, clabel = _get_var_style(var)

            if npz_hr is not None and var in npz_hr:
                idx_map["hr"] = len(arrs); arrs.append(npz_hr[var]); titles.append(f"HR | {var}")
            if npz_ensmean is not None and var in npz_ensmean:
                idx_map["ens"] = len(arrs); arrs.append(npz_ensmean[var]); titles.append(f"Ensemble mean | {var}")
            if npz_ensstd is not None and var in npz_ensstd:
                idx_map["ensstd"] = len(arrs); arrs.append(npz_ensstd[var]); titles.append(f"Ensemble spread (std) | {var}")
            if npz_lr is not None and var in npz_lr:
                idx_map["lr"] = len(arrs); arrs.append(npz_lr[var]); titles.append(f"LR | {var}")

            if not arrs:
                continue

            # robust vmin/vmax across value arrays (1–99th), then force vmin >= 0
            stack_vals = np.concatenate([a.reshape(-1) for a in arrs if a is not None])
            stack_vals = stack_vals[np.isfinite(stack_vals)]
            if stack_vals.size > 0:
                if vmin is None:
                    vmin = float(np.nanpercentile(stack_vals, 1.0))
                if vmax is None:
                    vmax = float(np.nanpercentile(stack_vals, 99.0))
                if np.isfinite(vmin) and vmin < 0.0:
                    vmin = 0.0
                if np.isfinite(vmin) and np.isfinite(vmax) and vmin >= vmax:
                    vmin, vmax = float(np.nanmin(stack_vals)), float(np.nanmax(stack_vals))
                    if vmin < 0.0:
                        vmin = 0.0

            _nice()

            # helpers for ratio/bias vs HR
            def _safe_ratio(num, den):
                if num is None or den is None:
                    return None
                with np.errstate(divide="ignore", invalid="ignore"):
                    r = num / den
                r[~np.isfinite(r)] = np.nan
                return r

            def _safe_bias(num, den):
                if num is None or den is None:
                    return None
                with np.errstate(invalid="ignore"):
                    b = num - den
                b[~np.isfinite(b)] = np.nan
                return b

            hr_arr   = npz_hr[var]       if (npz_hr is not None and var in npz_hr) else None
            ens_arr  = npz_ensmean[var]  if (npz_ensmean is not None and var in npz_ensmean) else None
            lr_arr   = npz_lr[var]       if (npz_lr is not None and var in npz_lr) else None

            rat_ens  = _safe_ratio(ens_arr, hr_arr)
            rat_lr   = _safe_ratio(lr_arr,  hr_arr)
            bias_ens = _safe_bias(ens_arr, hr_arr)
            bias_lr  = _safe_bias(lr_arr,  hr_arr)

            # Layout: row 0 = values, row 1 = ratios, row 2 = biases
            ncols = len(arrs)
            fig, axs = plt.subplots(3, ncols, figsize=(4.0*ncols, 10.5), squeeze=False)

            # Row 0: values (μ ± σ in titles)
            for j, a in enumerate(arrs):
                # For EnsStd column, keep vmin=0 explicitly
                is_spread = (j == idx_map.get("ensstd", -1))
                _draw_single(
                    axs[0, j], a, titles[j], cmap=cmap, vmin=(0.0 if is_spread else vmin), vmax=vmax,
                    cbar_label=clabel, dk_mask=dk_mask, add_stats=True
                )

            # Initialize rows 1–2 as empty
            for j in range(ncols):
                axs[1, j].axis("off")
                axs[2, j].axis("off")

            # Row 1: ratios
            if "ens" in idx_map and rat_ens is not None:
                _draw_single(
                    axs[1, idx_map["ens"]], rat_ens, f"EnsMean/HR | {var}",
                    cmap=get_cmap_for_variable("prcp_bias"), vmin=0.5, vmax=1.5,
                    cbar_label="ratio", dk_mask=dk_mask, add_stats=True
                )
                axs[1, idx_map["ens"]].axis("on")
            if "lr" in idx_map and rat_lr is not None:
                _draw_single(
                    axs[1, idx_map["lr"]], rat_lr, f"LR/HR | {var}",
                    cmap=get_cmap_for_variable("prcp_bias"), vmin=0.5, vmax=1.5,
                    cbar_label="ratio", dk_mask=dk_mask, add_stats=True
                )
                axs[1, idx_map["lr"]].axis("on")

            # Row 2: biases (symmetric limits)
            bias_vals = []
            if bias_ens is not None: bias_vals.append(np.abs(bias_ens).ravel())
            if bias_lr  is not None: bias_vals.append(np.abs(bias_lr).ravel())
            bmax = float(np.nanpercentile(np.concatenate(bias_vals), 99.0)) if bias_vals else (float(vmax) if vmax is not None else 1.0)

            if "ens" in idx_map and bias_ens is not None:
                _draw_single(
                    axs[2, idx_map["ens"]], bias_ens, f"EnsMean-HR | {var}",
                    cmap=get_cmap_for_variable("prcp_bias"), vmin=-bmax, vmax=bmax,
                    cbar_label="bias", dk_mask=dk_mask, add_stats=True
                )
                axs[2, idx_map["ens"]].axis("on")
            if "lr" in idx_map and bias_lr is not None:
                _draw_single(
                    axs[2, idx_map["lr"]], bias_lr, f"LR-HR | {var}",
                    cmap=get_cmap_for_variable("prcp_bias"), vmin=-bmax, vmax=bmax,
                    cbar_label="bias", dk_mask=dk_mask, add_stats=True
                )
                axs[2, idx_map["lr"]].axis("on")

            fig.suptitle(f"{group}: {var}")
            _savefig(fig, figs / f"spatial_{group}_{var}.png", dpi=SET_DPI)