import os
import json
import logging
from pathlib import Path
from venv import logger
import numpy as np
import torch
import matplotlib.pyplot as plt

from scor_dm.training_utils import get_model
from generate.generation_main import get_final_gen_dataloader
from generate.generation import GenerationRunner, GenerationConfig
from scor_dm.utils import get_model_string
from scor_dm.variable_utils import get_cmap_for_variable

logger = logging.getLogger(__name__)

def _outdir_quicklook(cfg) -> Path:
    model_str = get_model_string(cfg)
    out_dir = Path(cfg["paths"]["sample_dir"]) / "quicklook" / model_str
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def _imshow(ax, arr2d, title, vmin=None, vmax=None, cmap=None):
    """Draw a 2D array with provided vmin/vmax (can be shared across panels)."""
    if vmin is None and vmax is None:
        im = ax.imshow(arr2d, origin="upper", cmap=cmap)
    else:
        im = ax.imshow(arr2d, origin="upper", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    return im

def _pick_dates_from_loader(gen_loader, n_dates, specific_dates=None):
    """Iterate loader once; collect sample dicts for requested dates."""
    picked = []
    want = set(specific_dates) if specific_dates else None
    for batch in gen_loader:
        # Expect the dataset to emit a dict with 'date'
        dates = batch.get('date', None)
        date0 = dates[0] if isinstance(dates, (list, tuple)) else (str(dates) if dates is not None else None)

        if want is not None:
            if date0 in want:
                picked.append(batch)
        else:
            picked.append(batch)

        if len(picked) >= n_dates and want is None:
            break
        if want is not None and len(picked) >= len(want):
            break
    return picked

class _ListLoader:
    """Minimal iterable to feed a list of already-materialized samples to GenerationRunner.run()."""
    def __init__(self, items): self.items = items
    def __iter__(self): 
        for it in self.items:
            yield it
    def __len__(self): return len(self.items)


@torch.no_grad()
def quicklook_from_runner(cfg):
    """
    Generate small quicklook panels using GenerationRunner (no NPZ writes).
    Uses cfg['quicklook'] keys:
      - n_dates, members_to_show, sampler_steps, seed, specific_dates, vmax_mm, save_png
    """
    ql = cfg.get("quicklook", {})
    n_dates         = int(ql.get("n_dates", 6))
    members_to_show = int(ql.get("members_to_show", 3))
    sampler_steps   = ql.get("sampler_steps", None)
    seed            = ql.get("seed", None)
    specific_dates  = ql.get("specific_dates", None)
    vmax_mm         = ql.get("vmax_mm", None)
    save_png        = bool(ql.get("save_png", True))

    # Seeding
    if seed is None:
        seed = np.random.randint(0, 1_000_000)
    torch.manual_seed(int(seed)); torch.cuda.manual_seed(int(seed)); np.random.seed(int(seed))

    device = torch.device(cfg["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # --- Model ---
    model, ckpt_dir, ckpt_name = get_model(cfg)
    ckpt = torch.load(os.path.join(ckpt_dir, ckpt_name), map_location=device)
    model.load_state_dict(ckpt["network_params"])
    model.eval()

    # --- Data (full loader, then subselect) ---
    gen_loader_full = get_final_gen_dataloader(cfg, verbose=False)
    picked_samples = _pick_dates_from_loader(gen_loader_full, n_dates=n_dates, specific_dates=specific_dates)
    if len(picked_samples) == 0:
        raise RuntimeError("quicklook: no samples matched the request (check specific_dates or dataset).")
    mini_loader = _ListLoader(picked_samples)

    # --- GenerationRunner config ---
    edm = cfg.get("edm", {})
    gen_cfg = GenerationConfig(
        output_root      = str(_outdir_quicklook(cfg)),  # we only save PNGs here; run(save=False)
        ensemble_size    = members_to_show,
        max_dates        = len(picked_samples),
        sampler_steps    = int(sampler_steps) if sampler_steps is not None else int(edm.get("sampling_steps", 40)),
        seed             = int(seed),
        use_edm          = bool(edm.get("enabled", True)),
        sigma_min        = float(edm.get("sigma_min", 0.002)),
        sigma_max        = float(edm.get("sigma_max", 80.0)),
        rho              = float(edm.get("rho", 7.0)),
        S_churn          = float(edm.get("S_churn", 0.0)),
        S_min            = float(edm.get("S_min", 0.0)),
        S_max            = float(edm.get("S_max", float("inf"))),
        S_noise          = float(edm.get("S_noise", 1.0)),
        predict_residual = bool(edm.get("predict_residual", False)),
        save_space       = "physical",   # so we get physical tensors back in results for nice plots
        physical_dtype   = "float32",
        log_every        = 25,
    )


    runner = GenerationRunner(model=model, cfg=cfg, device=str(device), out_root=_outdir_quicklook(cfg), gen_config=gen_cfg, quicklook=True)

    # Run once, with in-memory return (no NPZ writing)
    ret = runner.run(mini_loader, save=False, output_results=True)
    if ret is None or "results" not in ret:
        raise RuntimeError("quicklook: runner.run() returned None or missing 'results'.")
    results = ret["results"]

    out_dir = _outdir_quicklook(cfg)
    vmax = None if vmax_mm is None else float(vmax_mm)

    # --- Resolve colormap: quicklook override or variable-specific default ---
    try:
        hr_var = cfg["highres"]["variable"]
    except Exception:
        hr_var = None
    cmap_name = ql.get("cmap", None)
    if cmap_name is None:
        # Fallback to registry based on the HR variable (e.g., "prcp" -> "inferno")
        cmap_name = get_cmap_for_variable(hr_var) if hr_var is not None else "viridis"

    # --- Render figures ---
    for item in results:
        date = item["date"]
        # Prefer physical space for plotting; fall back to model space
        lr = item.get("lr_phys", None)
        hr = item.get("hr_phys", None)
        pmm = item.get("pmm_phys", None)
        ens = item.get("ensemble_phys", None)

        if lr is None:  lr = item.get("lr_model", None)
        if hr is None:  hr = item.get("hr_model", None)
        if pmm is None: pmm = item.get("pmm_model", None)
        if ens is None: ens = item.get("ensemble_model", None)

        # Convert to numpy 2D
        def _to2d(x):
            if x is None: return None
            if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
            # accepted shapes: [1,1,H,W], [M,1,H,W]
            if x.ndim == 4:    # [*,1,H,W] -> [H,W]
                return x[0, 0]
            elif x.ndim == 3:  # [M,H,W] -> [H,W]
                return x[0]
            elif x.ndim == 2:  # [H,W]
                return x
            else:
                raise ValueError(f"Unexpected array shape for quicklook: {x.shape}")

        # pick K member maps
        member_maps = []
        if isinstance(ens, torch.Tensor):
            ens_np = ens.detach().cpu().numpy()
        else:
            ens_np = ens
        if ens_np is not None:
            # ens_np: [M,1,H,W] or [M,H,W]
            if ens_np.ndim == 4 and ens_np.shape[1] == 1:
                for k in range(min(members_to_show, ens_np.shape[0])):
                    member_maps.append(ens_np[k, 0])
            elif ens_np.ndim == 3:
                for k in range(min(members_to_show, ens_np.shape[0])):
                    member_maps.append(ens_np[k])

        # --- Determine shared color scale across all panels ---
        def _finite_minmax(arrs):
            vals = []
            for a in arrs:
                if a is None:
                    continue
                an = np.asarray(a, dtype=float)
                m = np.isfinite(an)
                if m.any():
                    vals.append(an[m])
            if not vals:
                return None, None
            cat = np.concatenate(vals)
            return float(np.min(cat)), float(np.max(cat))

        # Assemble all 2D arrays we are going to plot for this date
        arrays_all = []
        # Temporarily convert tensors to numpy for min/max if needed
        def _as2d_np(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            if x.ndim == 4 and x.shape[1] == 1:
                return x[0, 0]
            if x.ndim == 3:
                return x[0]
            if x.ndim == 2:
                return x
            return None

        arrays_all.extend([_as2d_np(lr), _as2d_np(hr), _as2d_np(pmm)])
        arrays_all.extend(member_maps)

        # Priority of explicit limits from cfg:
        # 1) quicklook['vmin'], quicklook['vmax'] if both provided
        # 2) quicklook['vmax_mm'] with vmin=0 (useful for precipitation)
        # 3) data-driven finite min/max
        vmin_cfg = ql.get("vmin", None)
        vmax_cfg = ql.get("vmax", None)
        if vmin_cfg is not None and vmax_cfg is not None:
            shared_vmin = float(vmin_cfg)
            shared_vmax = float(vmax_cfg)
        elif vmax is not None:
            shared_vmin = 0.0
            shared_vmax = float(vmax)
        else:
            mn, mx = _finite_minmax(arrays_all)
            shared_vmin, shared_vmax = mn, mx


        # Build figure: all panels share the same vmin/vmax and a single colorbar
        cols = [("LR", _to2d(lr)), ("HR", _to2d(hr)), ("PMM", _to2d(pmm))]
        for i, arr in enumerate(member_maps):
            cols.append((f"m{i+1}", arr))

        ncols = len(cols)
        fig, axes = plt.subplots(1, ncols, figsize=(3.0*ncols, 3.2), constrained_layout=True)
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]

        ims = []
        for ax, (title, arr2d) in zip(axes, cols):
            if arr2d is None:
                ax.axis("off"); ax.set_title(f"{title} (n/a)", fontsize=10)
            else:
                ims.append(_imshow(ax, arr2d, title, vmin=shared_vmin, vmax=shared_vmax, cmap=cmap_name))
        # Shared colorbar spanning all axes (use the last image handle)
        if ims:
            cbar = fig.colorbar(ims[-1], ax=axes, location="right", pad=0.02)
            # optional label from cfg
            if "cbar_label" in ql:
                cbar.set_label(str(ql["cbar_label"]))

        fig.suptitle(f"{date}", fontsize=11)
        # With constrained_layout=True, avoid tight_layout to prevent engine conflicts with colorbar.
        # Adjust top margin to make space for the suptitle.
        fig.subplots_adjust(top=0.90)

        if bool(save_png):
            fig.savefig(str(out_dir / f"{date}_quicklook.png"), dpi=150)
        plt.close(fig)
    logger.info(f"[quicklook] Done. Quicklook PNGs (if enabled) at: {out_dir}")
    return out_dir