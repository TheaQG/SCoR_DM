"""
Plot sigma*-dependent evaluation metrics: correlation, PSD slope, and CRPS.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging 

logger = logging.getLogger(__name__)

# Additional imports for color and data utilities
from scor_dm.variable_utils import get_color_for_model, get_cmap_for_variable
from scor_dm.evaluate.data_resolver import EvalDataResolver

# New imports for plotting utilities and DK outline
from scor_dm.evaluate.evaluate_prcp.plot_utils import _ensure_dir, _nice, _savefig, get_dk_lsm_outline, overlay_outline
from scor_dm.plotting_utils import _add_colorbar_and_boxplot

SET_DPI = 300

def plot_sigma_control(summary_csv, figures_dir, combined: bool = False):
    """
    Render sigma*-dependent summary plots.

    Parameters
    ----------
    summary_csv : str or Path
        Path to tables/agg_summary.csv
    figures_dir : str or Path
        Directory to write figures (created if missing).
    combined : bool
        If True, draw all three panels side-by-side in a single figure and also
        save individual panels. If False, save only individual panels.
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    data = np.genfromtxt(summary_csv, delimiter=',', names=True, dtype=None, encoding='utf-8')

    if isinstance(data, np.ndarray) and data.size == 0:
        logger.warning("[sigma_control.plot] Empty summary CSV: %s", str(summary_csv))
        return {}

    # Handle empty summary gracefully
    if isinstance(data, np.ndarray) and data.size == 0:
        return {}

    sigma = np.asarray(data['sigma_star'], dtype=float)
    r_lp_mean = np.asarray(data['r_lp_mean'], dtype=float)
    r_lp_std = np.asarray(data['r_lp_std'], dtype=float)
    slope_gen_mean = np.asarray(data['slope_gen_mean'], dtype=float)
    slope_gen_std = np.asarray(data['slope_gen_std'], dtype=float)
    slope_hr_mean = np.asarray(data['slope_hr_mean'], dtype=float)
    slope_hr_std = np.asarray(data['slope_hr_std'], dtype=float)
    crps_mean = np.asarray(data['crps_mean'], dtype=float)
    crps_std = np.asarray(data['crps_std'], dtype=float)

    # Optional: high-k gain (may be absent in old runs)
    hk_gain_mean = np.asarray(data['hk_gain_mean'], dtype=float) if (data.dtype.names is not None and 'hk_gain_mean' in data.dtype.names) else None
    hk_gain_std  = np.asarray(data['hk_gain_std'], dtype=float)  if (data.dtype.names is not None and 'hk_gain_std' in data.dtype.names) else None

    # Colors and styling
    color_gen = get_color_for_model("pmm")
    color_hr = get_color_for_model("hr")
    color_ens = get_color_for_model("ens")

    # Style
    _nice()
    marker_style = dict(marker="o", markersize=5, lw=1.8, capsize=3)

    # Utility for x padding
    def _xpad(x):
        if x.size == 0:
            return (0, 1)
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
        pad = 0.03 * max(1e-6, hi - lo)
        return lo - pad, hi + pad

    figpaths = {}

    # --- Combined figure (optional) ---
    if combined:
        import matplotlib.pyplot as plt
        ncols = 4 if hk_gain_mean is not None else 3
        fig, axes = plt.subplots(1, ncols, figsize=(4.6*ncols, 4.5), sharex=False)
        # 1) Correlation
        ax = axes[0]
        ax.errorbar(sigma, r_lp_mean, yerr=r_lp_std, color=color_gen, **marker_style)
        ax.set_xlabel(r"$\sigma^*$")
        ax.set_ylabel("LR-GEN correlation\n(LP ≤ LR Nyquist)")
        ax.set_xlim(*_xpad(sigma))
        ax.set_ylim(0.4, min(1.0, max(0.02, float(np.nanmax(r_lp_mean + r_lp_std)) + 0.02)))
        ax.set_title("Scale-aware correlation")
        # 2) PSD slope
        ax = axes[1]
        ax.errorbar(sigma, slope_gen_mean, yerr=slope_gen_std, color=color_gen, label="Generated", **marker_style)
        if slope_hr_mean.size > 0 and np.isfinite(slope_hr_mean).any():
            hr_ref = float(np.nanmean(slope_hr_mean))
            ax.axhline(hr_ref, color=color_hr, ls="--", lw=1.5, label="DANRA")
        ax.set_xlabel(r"$\sigma^*$")
        ax.set_ylabel("PSD slope (5-20 km)")
        ax.set_xlim(*_xpad(sigma))
        ax.legend(frameon=False)
        ax.set_title("Mesoscale PSD slope")
        # 3) CRPS
        ax = axes[2]
        ax.errorbar(sigma, crps_mean, yerr=crps_std, color=color_ens, **marker_style)
        ax.set_xlabel(r"$\sigma^*$")
        ax.set_ylabel("CRPS")
        ax.set_xlim(*_xpad(sigma))
        ax.set_title(r"Probabilistic skill vs $\sigma^*$")
        # 4) High-k gain panel if available
        if hk_gain_mean is not None:
            ax = axes[3]
            ax.errorbar(sigma, hk_gain_mean, yerr=hk_gain_std, color=color_gen, **marker_style)
            ax.set_xlabel(r"$\sigma^*$")
            ax.set_ylabel(r"$G_\mathrm{high}$  (P_GEN / P_HR, k > k_\mathrm{Nyq}^{LR})")
            ax.set_xlim(*_xpad(sigma))
            ax.set_title("High‑k power gain")
        fig.tight_layout()
        out_all = figures_dir / "sigma_control_overview.png"
        _savefig(fig, out_all, dpi=300)
        figpaths["overview"] = str(out_all)

    # --- Individual panels ---
    import matplotlib.pyplot as plt
    # 1) Correlation
    fig = plt.figure()
    ax = plt.gca()
    ax.errorbar(sigma, r_lp_mean, yerr=r_lp_std, color=color_gen, **marker_style)
    ax.set_xlabel(r"$\sigma^*$")
    ax.set_ylabel("LR-GEN correlation (LP ≤ LR Nyquist)")
    ax.set_xlim(*_xpad(sigma))
    ax.set_ylim(0.0, min(1.0, max(0.02, float(np.nanmax(r_lp_mean + r_lp_std)) + 0.02)))
    ax.set_title(r"Scale-aware correlation vs $\sigma^*$")
    figpaths["corr"] = str(figures_dir / "hr_lr_corr_vs_sigma.png")
    _savefig(fig, Path(figpaths["corr"]), dpi=SET_DPI)

    # 2) PSD slope
    fig = plt.figure()
    ax = plt.gca()
    ax.errorbar(sigma, slope_gen_mean, yerr=slope_gen_std, color=color_gen, label="Generated", **marker_style)
    if slope_hr_mean.size > 0 and np.isfinite(slope_hr_mean).any():
        hr_ref = float(np.nanmean(slope_hr_mean))
        ax.axhline(hr_ref, color=color_hr, ls="--", lw=1.5, label="DANRA")
    ax.set_xlabel(r"$\sigma^*$")
    ax.set_ylabel("PSD slope (5-20 km)")
    ax.set_xlim(*_xpad(sigma))
    ax.set_title(r"PSD slope vs $\sigma^*$")
    ax.legend(frameon=False)
    figpaths["slope"] = str(figures_dir / "psd_slope_vs_sigma.png")
    _savefig(fig, Path(figpaths["slope"]), dpi=SET_DPI)

    # 3) CRPS
    fig = plt.figure()
    ax = plt.gca()
    ax.errorbar(sigma, crps_mean, yerr=crps_std, color=color_ens, **marker_style)
    ax.set_xlabel(r"$\sigma^*$")
    ax.set_ylabel("CRPS")
    ax.set_xlim(*_xpad(sigma))
    ax.set_title(r"Probabilistic skill vs $\sigma^*$")
    figpaths["crps"] = str(figures_dir / "crps_vs_sigma.png")
    _savefig(fig, Path(figpaths["crps"]), dpi=SET_DPI)

    # 4) High-k gain (if present)
    if hk_gain_mean is not None:
        fig = plt.figure()
        ax = plt.gca()
        ax.errorbar(sigma, hk_gain_mean, yerr=hk_gain_std, color=color_gen, **marker_style)
        ax.set_xlabel(r"$\sigma^*$")
        ax.set_ylabel(r"$G_\mathrm{high}$  (P_GEN / P_HR, k > k_\mathrm{Nyq}^{LR})")
        ax.set_xlim(*_xpad(sigma))
        ax.axhline(1.0, color="0.3", ls="--", lw=1.2)  # reference where GEN matches HR power
        ax.set_title(r"High‑k power gain vs $\sigma^*$")
        figpaths["hk_gain"] = str(figures_dir / "high_k_gain_vs_sigma.png")
        _savefig(fig, Path(figpaths["hk_gain"]), dpi=SET_DPI)

    logger.info("[sigma_control.plot] Wrote figures: %s", figpaths)
    return figpaths




# --------------------------------------------------------------------------
# Qualitative example montage function for sigma* control
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Helper functions for squeezing and masking arrays
# --------------------------------------------------------------------------
def _squeeze2d_arr(a):
    import numpy as _np
    import torch as _torch
    if a is None:
        return None
    if isinstance(a, _torch.Tensor):
        a = a.detach().cpu().numpy()
    x = _np.asarray(a)
    while x.ndim > 2 and 1 in x.shape:
        x = _np.squeeze(x)
    if x.ndim == 3:
        if x.shape[0] <= 8:
            x = x[0]
        else:
            x = x[..., 0]
    return x if x.ndim == 2 else None

def _apply_mask_arr(x, mask):
    import numpy as _np
    if x is None:
        return None
    if mask is None:
        return x
    out = x.copy()
    m = mask
    if m.shape != out.shape:
        m = _np.broadcast_to(m, out.shape)
    out[~m] = _np.nan
    return out

# --------------------------------------------------------------------------
# New function: plot_sigma_control_examples_grid
# --------------------------------------------------------------------------
def plot_sigma_control_examples_grid(
    cfg,
    sigma_star_grid,
    gen_base_dir,
    out_dir,
    *,
    sigma_star_subset=None,
    n_members=3,
    date: str | None = None,
    land_only: bool = True,
    percentile: float = 99.5,
    fname: str = "examples_sigma_grid.png",
):
    gen_base_dir = Path(gen_base_dir)
    figs_root = Path(out_dir) / "figures" / "examples"
    _ensure_dir(figs_root)

    # If a subset of sigma* values is provided, intersect with the available grid
    if sigma_star_subset is not None:
        try:
            sigma_star_subset = [float(s) for s in sigma_star_subset]
        except TypeError:
            sigma_star_subset = [float(sigma_star_subset)]
        # retain only values that are in the provided grid (within a small tolerance)
        grid_vals = [float(s) for s in sigma_star_grid]
        subset_filtered = []
        for s in sigma_star_subset:
            for g in grid_vals:
                if abs(s - g) < 1e-6:
                    subset_filtered.append(g)
                    break
        if subset_filtered:
            sigma_star_grid = subset_filtered

    cmap = get_cmap_for_variable("prcp")
    dk_outline = get_dk_lsm_outline()
    if dk_outline is not None:
        dk_outline = np.flipud(dk_outline)

    resolvers = []
    for sstar in sigma_star_grid:
        subdir = gen_base_dir / f"sigma_star={float(sstar):.2f}"
        if not subdir.exists():
            continue
        resolvers.append((sstar, EvalDataResolver(gen_root=subdir, eval_land_only=land_only, lr_phys_key="lr")))
    if not resolvers:
        return []

    if date is None:
        common = None
        for _, R in resolvers:
            ds = set(R.list_dates())
            common = ds if common is None else (common & ds)
        if common and len(common) > 0:
            date = sorted(list(common))[0]
        else:
            date = resolvers[0][1].list_dates()[0]

    rows = []
    max_members = 0
    for sstar, R in resolvers:
        mask = None
        try:
            if land_only:
                m = _squeeze2d_arr(R.load_mask(date))
                if m is not None:
                    mask = (m > 0.5)
        except Exception:
            mask = None

        hr = _apply_mask_arr(_squeeze2d_arr(R.load_obs(date)), mask)
        lr = _apply_mask_arr(_squeeze2d_arr(R.load_lr(date)), mask)
        pmm = _apply_mask_arr(_squeeze2d_arr(R.load_pmm(date)), mask)
        ens = R.load_ens(date)

        members = []
        if ens is not None:
            A = ens.detach().cpu().numpy() if hasattr(ens, "detach") else np.asarray(ens)
            if A.ndim == 4 and A.shape[1] == 1:
                A = A[:,0]
            if A.ndim == 3 and A.shape[0] > 0:
                m = min(n_members, A.shape[0])
                for i in range(m):
                    members.append(_apply_mask_arr(_squeeze2d_arr(A[i]), mask))
        max_members = max(max_members, len(members))
        
        rows.append({"sigma": sstar, "hr": hr, "lr": lr, "pmm": pmm, "members": members})

    Rn = len(rows)
    Cn = 2 + max_members + 1

    # Global color scale across all σ* rows
    all_vals = []
    for row in rows:
        for arr in [row["hr"], row["lr"], row["pmm"], *row["members"]]:
            if arr is not None:
                vals = np.asarray(arr)
                vals = vals[np.isfinite(vals)]
                if vals.size > 0:
                    all_vals.append(vals)
    if all_vals:
        flat_all = np.concatenate(all_vals)
        vmin_global = 0.0
        vmax_global = float(np.nanpercentile(flat_all, percentile)) if flat_all.size > 0 else 1.0
        vmax_global = max(vmin_global + 1e-6, vmax_global)
    else:
        vmin_global, vmax_global = 0.0, 1.0

    _nice()
    fig_w = 3.2 * Cn
    fig_h = 3.2 * Rn
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(Rn, Cn, figsize=(fig_w, fig_h))
    if Rn == 1: axs = axs[np.newaxis, :]
    if Cn == 1: axs = axs[:, np.newaxis]

    titles = ["HR (DANRA)", "LR (ERA5)"] + [f"Ens-{i+1}" for i in range(max_members)] + ["PMM (gen)"]

    for r, row in enumerate(rows):
        hr = row["hr"]; lr = row["lr"]; pmm = row["pmm"]; members = row["members"]

        c = 0
        def draw(ax, img, title):
            if img is None:
                ax.axis("off"); return
            im = ax.imshow(img, origin="lower", vmin=vmin_global, vmax=vmax_global, cmap=cmap)
            overlay_outline(ax, dk_outline)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=14)
            _add_colorbar_and_boxplot(fig, ax, im, img, boxplot=True, ylim=(vmin_global, vmax_global))

        draw(axs[r, c], hr, titles[c]); c += 1
        draw(axs[r, c], lr, titles[c]); c += 1
        for j in range(max_members):
            ax = axs[r, c + j]
            if j < len(members):
                draw(ax, members[j], titles[c + j])
            else:
                ax.axis("off")
        c += max_members
        draw(axs[r, c], pmm, titles[-1])
        axs[r, 0].set_ylabel(f"σ*={row['sigma']:.2f}\n{date}", fontsize=14)

    fig.text(0.5, 0.02, "Precipitation [mm/day]", ha="center", fontsize=16)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    out_path = figs_root / fname
    _savefig(fig, out_path, dpi=300)
    return [str(out_path)]

def plot_sigma_control_psd_curves(out_dir: str | Path, sigma_subset=None) -> str | None:
    """
    Read <out_dir>/tables/sigma_psd_curves.npz and make a PSD vs wavelength plot
    showing HR, LR, and one curve per sigma* (mean +/- std shading), with:
      - green color scale across sigma*
      - shaded band for the slope window (e.g., 5-20 km)
      - textbox listing slopes per sigma* computed from the mean GEN PSD
    Saves to <out_dir>/figures/sigma_psd_curves.png
    """
    out_dir = Path(out_dir)
    tables = out_dir / "tables"
    figs = out_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    npz_path = tables / "sigma_psd_curves.npz"
    if not npz_path.exists():
        return None

    with np.load(npz_path) as d:
        k = d["k"]                       # [K]
        sigma_vals = d["sigma_vals"]     # [S]
        psd_hr_mean = d["psd_hr_mean"]   # [K]
        psd_hr_std  = d["psd_hr_std"]    # [K]
        psd_lr_mean = d["psd_lr_mean"]   # [K]
        psd_lr_std  = d["psd_lr_std"]    # [K]
        psd_gen_mean = d["psd_gen_mean"] # [S,K]
        psd_gen_std  = d["psd_gen_std"]  # [S,K]
        lr_nyq = float(d["lr_nyquist"]) if "lr_nyquist" in d.files else 0.0
        psd_band = tuple(d["psd_band_km"]) if "psd_band_km" in d.files else (5.0, 20.0)
        psd_hr_logmu  = d["psd_hr_logmu"]  if "psd_hr_logmu"  in d.files else None
        psd_hr_logstd = d["psd_hr_logstd"] if "psd_hr_logstd" in d.files else None
        psd_lr_logmu  = d["psd_lr_logmu"]  if "psd_lr_logmu"  in d.files else None
        psd_lr_logstd = d["psd_lr_logstd"] if "psd_lr_logstd" in d.files else None
        psd_gen_logmu  = d["psd_gen_logmu"]  if "psd_gen_logmu"  in d.files else None
        psd_gen_logstd = d["psd_gen_logstd"] if "psd_gen_logstd" in d.files else None

    # Optionally restrict to a subset of sigma* values (e.g. for clearer PSD plots)
    if sigma_subset is not None:
        try:
            sigma_subset = np.array([float(s) for s in sigma_subset], dtype=float)
        except TypeError:
            sigma_subset = np.array([float(sigma_subset)], dtype=float)
        mask = np.isin(sigma_vals, sigma_subset)
        if mask.any():
            sigma_vals = sigma_vals[mask]
            psd_gen_mean = psd_gen_mean[mask]
            psd_gen_std = psd_gen_std[mask]
            if psd_gen_logmu is not None and psd_gen_logstd is not None:
                psd_gen_logmu = psd_gen_logmu[mask]
                psd_gen_logstd = psd_gen_logstd[mask]

    # brief sanity
    if k.ndim != 1 or psd_gen_mean.ndim != 2:
        logger.warning("[sigma_psd] Unexpected shapes: k=%s, gen_mean=%s", k.shape, psd_gen_mean.shape)

    # Optional: read ramp/meta info
    ramp_info = None
    meta_path = (Path(out_dir) / "sigma_control_meta.json")
    if meta_path.exists():
        try:
            import json
            with open(meta_path, "r") as f:
                meta = json.load(f)
                ramp_info = meta.get("ramp", None)
        except Exception:
            ramp_info = None

    # Convert to wavelength (km), positive k only
    mpos = k > 0
    k_pos = k[mpos]
    lam = 1.0 / k_pos
    order = np.argsort(lam)[::-1] # Descending wavelength (reads large-scale to small-scale)
    lam = lam[order]

    # Prepare style
    _nice()
    fig, ax = plt.subplots(figsize=(9.0, 6.0))

    # HR and LR references
    hr_lin = np.maximum(psd_hr_mean[mpos][order], 1e-12)
    lr_lin = np.maximum(psd_lr_mean[mpos][order], 1e-12)
    ax.plot(lam, hr_lin, color=get_color_for_model("hr"), lw=2.0, label="HR (DANRA)")
    ax.plot(lam, lr_lin, color=get_color_for_model("lr"), lw=1.6, ls="--", label="LR (ERA5↑)")
    z = 1.0
    # Only fill bands if both log-mean and log-std are available
    # if (psd_hr_logmu is not None) and (psd_hr_logstd is not None):
    #     mu = psd_hr_logmu[mpos][order]; sd = psd_hr_logstd[mpos][order]
    #     ax.fill_between(lam, 10**(mu - z*sd), 10**(mu + z*sd), color=get_color_for_model("hr"), alpha=0.15)
    # if (psd_lr_logmu is not None) and (psd_lr_logstd is not None):
    #     mu = psd_lr_logmu[mpos][order]; sd = psd_lr_logstd[mpos][order]
    #     ax.fill_between(lam, 10**(mu - z*sd), 10**(mu + z*sd), color=get_color_for_model("lr"), alpha=0.12)

    # Shaded slope band (in λ)
    lam_lo, lam_hi = float(psd_band[0]), float(psd_band[1])  # e.g., 5–20 km
    left, right = min(lam_lo, lam_hi), max(lam_lo, lam_hi)
    ax.axvspan(left, right, color="0.85", alpha=0.35, zorder=0)
    # Optional: visualize the "controlled" high-k region (λ <= right) lightly
    ax.axvspan(0.0, right, color="0.9", alpha=0.15, hatch='///', linewidth=0, label="σ* control (late)") 

    # σ* curves with shaded ±1σ and green colormap
    import matplotlib.cm as cm
    cmap = cm.get_cmap("Greens")
    S = len(sigma_vals)
    slopes_txt = []
    
    for i, s in enumerate(sigma_vals):
        c = cmap(0.15 + 0.75 * (1 - i / max(1, S-1)))  # darkest for smallest σ*
        mean_i = np.maximum(psd_gen_mean[i][mpos][order], 1e-12)
        ax.plot(lam, mean_i, lw=1.8, color=c, label=fr"Gen (σ*={s:.2f})")
        z = 1.0
        # # Use log-mean/log-std band if both are available, otherwise fall back to linear std
        # if (psd_gen_logmu is not None) and (psd_gen_logstd is not None):
        #     mu = psd_gen_logmu[i][mpos][order]
        #     sd = psd_gen_logstd[i][mpos][order]
        #     lower = 10**(mu - z*sd)
        #     upper = 10**(mu + z*sd)
        # else:
        #     std_i = np.maximum(psd_gen_std[i][mpos][order], 0.0)
        #     lower = np.clip(mean_i - std_i, 1e-14, None)
        #     upper = mean_i + std_i
        # ax.fill_between(lam, lower, upper, color=c, alpha=0.22)

        # Compute slope in the band using log10 fit on k (unsorted k_pos domain)
        eps = 1e-20
        k_lo = 1.0 / right
        k_hi = 1.0 / left
        mband = (k_pos > k_lo) & (k_pos < k_hi)

        if mband.any():
            xk = np.log10(k_pos[mband])
            yk_gen = np.log10(np.clip(psd_gen_mean[i][mpos][mband], eps, None))
            coef = np.polyfit(xk, yk_gen, 1)
            slopes_txt.append(fr"σ*={s:.2f}: {coef[0]:.2f}")
        else:
            slopes_txt.append(fr"σ*={s:.2f}: n/a")

    # Compute HR and LR slopes in the same mesoscale band (use mean PSD curves)
    eps = 1e-20
    mband_ref = (k_pos > (1.0 / right)) & (k_pos < (1.0 / left))
    hr_slope = np.nan
    lr_slope = np.nan
    if mband_ref.any():
        xk = np.log10(k_pos[mband_ref])
        yk_hr = np.log10(np.clip(psd_hr_mean[mpos][mband_ref], eps, None))
        hr_slope = np.polyfit(xk, yk_hr, 1)[0]
        yk_lr = np.log10(np.clip(psd_lr_mean[mpos][mband_ref], eps, None))
        lr_slope = np.polyfit(xk, yk_lr, 1)[0]
    slopes_txt.append(fr"HR: {hr_slope:.2f}")
    slopes_txt.append(fr"LR: {lr_slope:.2f}")

    # LR Nyquist as vertical line
    if lr_nyq > 0.0:
        lam_nyq = 1.0 / lr_nyq
        ax.axvline(lam_nyq, color="0.2", lw=1.0, ls="--", label="LR Nyq")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-8, 1e5)
    # Cut off above LR Nyquist
    ax.invert_xaxis()
    ax.set_xlim(lam.max()*1.02, min(lam.min()*0.98, 1.0 / lr_nyq if lr_nyq > 0.0 else lam.min()*0.98)) 
    ax.set_xlabel("Wavelength λ (km)")
    ax.set_ylabel("Spectral power")
    ax.set_title(r"Mean ensemble PSDs vs wavelength across $\sigma^*$")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="upper right", fontsize=9, frameon=False)

    # Add slope textbox
    text_lines = ["PSD slope in band:"]
    text_lines.extend(slopes_txt)
    if ramp_info is not None:
        mode = str(ramp_info.get("mode", "global"))
        sf = ramp_info.get("start_frac", None)
        ef = ramp_info.get("end_frac", None)
        ss = ramp_info.get("start_sigma", None)
        es = ramp_info.get("end_sigma", None)
        text_lines.append("")
        text_lines.append("Ramp:")
        text_lines.append(f"mode: {mode}")
        if (sf is not None) and (ef is not None):
            text_lines.append(f"frac: {float(sf):.2f}→{float(ef):.2f}")
        if (ss is not None) and (es is not None):
            text_lines.append(f"σ gate: ≤{ss} → ≤{es}")
    textstr = "\n".join(text_lines)
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment="bottom", horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="0.6"))

    out_path = figs / "sigma_psd_curves.png"
    _savefig(fig, out_path, dpi=SET_DPI)
    logger.info("[sigma_psd] Saved PSD figure: %s", str(out_path))
    return str(out_path)