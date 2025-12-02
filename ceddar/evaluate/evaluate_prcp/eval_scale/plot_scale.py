from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Sequence, Any, List

import numpy as np
import matplotlib.pyplot as plt
import logging

from scor_dm.evaluate.evaluate_prcp.plot_utils import _ensure_dir, _savefig, _nice, _to_date_safe, _season_from_month
from scor_dm.variable_utils import get_units, get_color_for_model, get_cmap_for_variable
from scor_dm.evaluate.evaluate_prcp.overlay_utils import resolve_baseline_dirs

logger = logging.getLogger(__name__)

SET_DPI = 300
# Get colors and set them
COL_HR = get_color_for_model("HR")
COL_PMM = get_color_for_model("PMM")
COL_LR = get_color_for_model("LR")
COL_ENS = get_color_for_model("ensemble")
COL_QM = get_color_for_model("QM")
COL_UNET = get_color_for_model("unet")

# Preferred drawing order (larger zorder = on top)
ZORDER_HR = 20          # HR DANRA
ZORDER_PMM = 18         # PMM (gen)
ZORDER_ENS = 25         # GEN ensemble mean
ZORDER_LR = 10          # LR
ZORDER_BASELINE = 5     # extra baselines (QM, UNet-SR, etc.)
ZORDER_ANNOT = 2        # vertical lines, text, etc.

# ================================================================================
# 1. PSD curves
# ================================================================================

def plot_scale_psd(scale_root: Path, eval_cfg: Any | None = None) -> None:
    """
    Read scale_psd_curves.npz and make a single log-log PSD plot:
      - HR: thick, solid
      - GEN: solid
      - LR: full curve, but 'ghosted' (low alpha) for k < lr_nyquist and solid for k >= lr_nyquist
      - x-axis in wavelength lambda (km), log-scaled, inverted (large -> small)
      - HR: thick, black, with +/- 1 sigma shading
      - GEN/PMM: blue, with +/- 1 sigma shading
      - LR: pink
          - solid for λ >= λ_nyq   (i.e. k <= k_nyq)
          - faint/dashed for λ < λ_nyq (i.e. k > k_nyq)
      - vertical line at LR Nyquist
      - optional amplitude alignment for LR so it sits in the right ballpark
        compared to HR on the trusted (k <= k_nyq) range
    """
    tables = scale_root / "tables"
    figs = _ensure_dir(scale_root / "figures")

    npz_path = tables / "scale_psd_curves.npz"
    if not npz_path.exists():
        logger.warning(f"[plot_scale_psd] Did not find {npz_path} - skipping PSD plot.")
        return

    # Set colors
    col_hr = get_color_for_model("HR")
    col_gen = get_color_for_model("PMM")
    col_lr = get_color_for_model("LR")
    col_gen_ens = get_color_for_model("ensemble")

    with np.load(npz_path) as data:
        files = set(data.files)
        def _opt(key):
            return data[key] if key in files else None
        k = data["k"]               # [K]
        psd_hr = data["psd_hr"]     # [N, K]
        psd_gen = data["psd_gen"]   # [N, K]
        psd_lr = data["psd_lr"]     # [N, K]
        psd_lr_hr = data["psd_lr_hr"]  # [N, K]
        dates = data["dates"]          # [N]
        psd_hr_ci_lo = _opt("psd_hr_ci_lo")
        psd_hr_ci_hi = _opt("psd_hr_ci_hi")
        psd_gen_ci_lo = _opt("psd_gen_ci_lo")
        psd_gen_ci_hi = _opt("psd_gen_ci_hi")
        lr_nyquist_arr = _opt("lr_nyquist")
        lr_nyquist = float(lr_nyquist_arr) if lr_nyquist_arr is not None else 0.0
        # default low-k upper bound (k <= 1/200 km), but allow extending up to LR Nyquist if available
        low_k_max_default = 1.0 / 200.0
        low_k_max_eff = lr_nyquist if lr_nyquist > 0.0 else low_k_max_default        
        psd_gen_ens_mean = _opt("psd_gen_ens_mean")
        psd_gen_ens_ci_lo = _opt("psd_gen_ens_ci_lo")
        psd_gen_ens_ci_hi = _opt("psd_gen_ens_ci_hi")

    if psd_gen_ens_mean is None:
        logger.info("[plot_scale_psd] No ensemble PSD arrays found in NPZ -> only PMM will be plotted.")

    # mean over dates
    eps = 1e-12
    hr_mean = psd_hr.mean(axis=0)
    hr_std = psd_hr.std(axis=0)
    gen_mean = psd_gen.mean(axis=0)
    gen_std = psd_gen.std(axis=0)
    lr_mean = psd_lr.mean(axis=0)

    lr_hr_mean = None
    if psd_lr_hr is not None:
        lr_hr_mean = psd_lr_hr.mean(axis=0)

    hr_mean = np.maximum(hr_mean, eps)
    gen_mean = np.maximum(gen_mean, eps)
    lr_mean = np.maximum(lr_mean, eps)


    if lr_nyquist > 0.0:
        lr_mask_lo = k <= lr_nyquist * 1.0001
        lr_mask_hi = k > lr_nyquist * 1.0001
    else:
        # no LR Nyquist info → just plot as one line
        lr_mask_lo = np.ones_like(k, dtype=bool)
        lr_mask_hi = np.zeros_like(k, dtype=bool)

    # Convert to wavelength (km)
    mask_pos = k > 0.0
    k_pos = k[mask_pos]
    lam = 1.0 / k_pos
    # Sort from large to small wavelength so line is monotonic on x
    order = np.argsort(lam)[::-1]
    lam = lam[order]
    hr_mean = hr_mean[mask_pos][order]
    gen_mean = gen_mean[mask_pos][order]
    hr_std = hr_std[mask_pos][order]
    gen_std = gen_std[mask_pos][order]
    lr_mean = lr_mean[mask_pos][order]
    if lr_hr_mean is not None:
        lr_hr_mean = np.maximum(lr_hr_mean, eps)
        lr_hr_mean = lr_hr_mean[mask_pos][order]

    gen_ens_mean = None
    gen_ens_ci_lo = None
    gen_ens_ci_hi = None
    if psd_gen_ens_mean is not None:
        arr = np.maximum(np.asarray(psd_gen_ens_mean), eps)[mask_pos][order]
        gen_ens_mean = arr
        if psd_gen_ens_ci_lo is not None and psd_gen_ens_ci_hi is not None:
            gen_ens_ci_lo = np.maximum(np.asarray(psd_gen_ens_ci_lo), eps)[mask_pos][order]
            gen_ens_ci_hi = np.maximum(np.asarray(psd_gen_ens_ci_hi), eps)[mask_pos][order]

    # --- Compute band powers and ratios from mean PSDs (as plotted) ---
    # 1. k-array in plotted order
    k_plot = k_pos[order]
    # 2. Band definitions
    # low-k goes all the way up to LR Nyquist if we know it; otherwise fall back to 1/200 km
    low_k_max = float(low_k_max_eff)              # k <= ...  -> λ >= ...
    high_k_min = 1.0 / 20.0                       # k >= 5.000e-02 -> λ <= 20 km
    # 3. Helper for band integration
    def _band_int(k_arr: np.ndarray, p_arr: np.ndarray, kmin: float, kmax: float) -> float:
        m = (k_arr >= kmin) & (k_arr <= kmax)
        if np.any(m):
            return float(np.trapz(p_arr[m], k_arr[m]))

        # Fallback 1: requested band is entirely *below* available k -> use lowest bin
        if kmax < k_arr.min():  # e.g. low-k band but k_arr starts higher
            m = (k_arr <= k_arr.min() * 1.01)
            return float(np.trapz(p_arr[m], k_arr[m]))

        # Fallback 2: requested band is entirely *above* available k -> use highest bin
        if kmin > k_arr.max():  # e.g. very-high-k band
            m = (k_arr >= k_arr.max() * 0.99)
            return float(np.trapz(p_arr[m], k_arr[m]))

        # Final fallback - should rarely happen
        return float(np.trapz(p_arr, k_arr))

    # 4. Compute band powers for HR, GEN, LR-on-HR-grid (or native LR)
    hr_low  = _band_int(k_plot, hr_mean, 0.0, low_k_max)
    hr_high = _band_int(k_plot, hr_mean, high_k_min, k_plot.max())
    gen_low  = _band_int(k_plot, gen_mean, 0.0, low_k_max)
    gen_high = _band_int(k_plot, gen_mean, high_k_min, k_plot.max())
    if lr_hr_mean is not None:
        lr_used = lr_hr_mean
    else:
        lr_used = lr_mean
    lr_low  = _band_int(k_plot, lr_used, 0.0, low_k_max)
    lr_high = _band_int(k_plot, lr_used, high_k_min, k_plot.max())
    # 5. Compute ratios (guard against zero/NaN)
    def _safe_ratio(num: float, den: float, eps_d: float = 1e-12) -> float:
        if num is None or not np.isfinite(num) or num < 0.0:
            return float("nan")
        den_eff = den if (den is not None and np.isfinite(den) and den > 0.0) else eps_d
        return float(num / den_eff)
    gen_hr_low_ratio  = _safe_ratio(gen_low, hr_low)
    gen_hr_high_ratio = _safe_ratio(gen_high, hr_high)
    lr_hr_low_ratio   = _safe_ratio(lr_low, hr_low)
    lr_hr_high_ratio  = _safe_ratio(lr_high, hr_high)
    # 6. Write to CSV
    ratios_path = tables / "scale_psd_band_ratios_avg.csv"
    with open(ratios_path, "w") as f:
        f.write("series,band,power,ratio_to_hr\n")
        f.write(f"HR,low-k,{hr_low:.6e},1.0\n")
        f.write(f"HR,high-k,{hr_high:.6e},1.0\n")
        f.write(f"GEN,low-k,{gen_low:.6e},{gen_hr_low_ratio:.4f}\n")
        f.write(f"GEN,high-k,{gen_high:.6e},{gen_hr_high_ratio:.4f}\n")
        if np.isfinite(lr_low) or np.isfinite(lr_high):
            f.write(f"LR,low-k,{lr_low:.6e},{lr_hr_low_ratio:.4f}\n")
            f.write(f"LR,high-k,{lr_high:.6e},{lr_hr_high_ratio:.4f}\n")

    # --- slope fits in log10(k) vs log10(P) ---
    def _fit_slope(k_arr: np.ndarray, p_arr: np.ndarray, mask: np.ndarray) -> tuple[float, float, float, float, float]:
        xs = np.log10(k_arr[mask])
        ys = np.log10(np.maximum(p_arr[mask], eps))
        if xs.size < 2:
            return float("nan"), float("nan"), 0.0, float(np.nan), float(np.nan)
        b, a = np.polyfit(xs, ys, 1)  # y = b*x + a
        yhat = b * xs + a
        ss_res = float(np.sum((ys - yhat) ** 2))
        ss_tot = float(np.sum((ys - np.mean(ys)) ** 2)) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        return float(b), float(a), float(r2), float(k_arr[mask].min()), float(k_arr[mask].max())

    def _collect_slopes(name: str, p: np.ndarray) -> list[tuple[str, str, float, float, float, float, float]]:
        out_rows = []
        masks = {
            "all": np.isfinite(k_plot) & np.isfinite(p) & (p > 0),
            "low-k": (k_plot <= low_k_max) & np.isfinite(p) & (p > 0),
            "high-k": (k_plot >= high_k_min) & np.isfinite(p) & (p > 0),
        }
        for rng, m in masks.items():
            sl, ic, r2, kmin, kmax = _fit_slope(k_plot, p, m)
            out_rows.append((name, rng, sl, ic, r2, kmin, kmax))
        return out_rows

    slope_rows: list[tuple[str, str, float, float, float, float, float]] = []
    slope_rows += _collect_slopes("HR", hr_mean)
    slope_rows += _collect_slopes("GEN", gen_mean)  # PMM
    if gen_ens_mean is not None:
        slope_rows += _collect_slopes("GEN_ens", gen_ens_mean)  # ensemble mean
    if lr_hr_mean is not None:
        slope_rows += _collect_slopes("LR", lr_hr_mean)
    else:
        slope_rows += _collect_slopes("LR", lr_mean)

    # write slopes to CSV
    slopes_path = tables / "scale_psd_slopes_avg.csv"
    with open(slopes_path, "w") as f:
        f.write("series,range,slope,intercept,r2,kmin,kmax\n")
        for s, rng, sl, ic, r2, kmin, kmax in slope_rows:
            f.write(f"{s},{rng},{sl:.6f},{ic:.6f},{r2:.4f},{kmin:.6e},{kmax:.6e}\n")


    # --- Find HR-LR intersection in log space ---
    cross_info = None
    if lr_hr_mean is not None:
        lr_for_x = lr_hr_mean.copy()
    else:
        lr_for_x = lr_mean.copy()
    try:
        diff = np.abs(np.log10(hr_mean) - np.log10(lr_for_x))
        ix = int(np.argmin(diff))
        lam_cross = float(lam[ix])
        k_cross = float(k_pos[order][ix])
        cross_info = (lam_cross, k_cross)
    except Exception:
        cross_info = None

    # Nyquist as wavelength
    lam_nyq = None
    if lr_nyquist > 0.0:
        lam_nyq = 1.0 / lr_nyquist

    _nice()
    fig, ax = plt.subplots(figsize=(6.5, 5))

    # HR
    ax.plot(
        lam, hr_mean,
        color=col_hr, lw=1.6, label="HR (DANRA)",
        zorder=ZORDER_HR,
    )
    if psd_hr_ci_lo is not None and psd_hr_ci_hi is not None:
        ci_lo = np.asarray(psd_hr_ci_lo)[mask_pos][order]
        ci_hi = np.asarray(psd_hr_ci_hi)[mask_pos][order]
        ax.fill_between(
            lam,
            np.maximum(ci_lo, eps), np.maximum(ci_hi, eps),  # type: ignore
            color=col_hr, alpha=0.15,
            zorder=ZORDER_HR - 1,
        )
    else:
        ax.fill_between(
            lam,
            np.maximum(hr_mean - hr_std, eps),
            hr_mean + hr_std,
            color=col_hr, alpha=0.15,
            zorder=ZORDER_HR - 1,
        )

    # GEN / PMM
    ax.plot(
        lam, gen_mean,
        color=col_gen, lw=1.4, label="PMM (gen)", ls='-.',
        zorder=ZORDER_PMM,
    )
    if psd_gen_ci_lo is not None and psd_gen_ci_hi is not None:
        ci_lo = np.asarray(psd_gen_ci_lo)[mask_pos][order]
        ci_hi = np.asarray(psd_gen_ci_hi)[mask_pos][order]
        ax.fill_between(
            lam,
            np.maximum(ci_lo, eps), np.maximum(ci_hi, eps),  # type: ignore
            color=col_gen, alpha=0.12,
            zorder=ZORDER_PMM - 1,
        )
    else:
        ax.fill_between(
            lam,
            np.maximum(gen_mean - gen_std, eps),
            gen_mean + gen_std,
            color=col_gen, alpha=0.12,
            zorder=ZORDER_PMM - 1,
        )

    # GEN ensemble mean + CI band (if available)
    if gen_ens_mean is not None:
        ax.plot(
            lam, gen_ens_mean,
            color=col_gen_ens, lw=1.2, label="GEN (ens mean)",
            zorder=ZORDER_ENS,
        )
        if gen_ens_ci_lo is not None and gen_ens_ci_hi is not None:
            ax.fill_between(
                lam, gen_ens_ci_lo, gen_ens_ci_hi, # type: ignore
                color=col_gen_ens, alpha=0.08,  
                zorder=ZORDER_ENS - 1,
            )

    # LR
    if lr_nyquist > 0.0 and lam_nyq is not None:
        trusted = lam >= (lam_nyq * 0.999)
        ghost   = lam <  (lam_nyq * 0.999)

        # "Ghost" part
        if lr_hr_mean is not None:
            ax.plot(
                lam, lr_hr_mean,
                color=col_lr, lw=0.9, linestyle="--", alpha=0.35,
                label="LR (ERA5, > Nyq)",
                zorder=ZORDER_LR - 1,
            )
        else:
            ax.plot(
                lam, lr_mean,
                color=col_lr, lw=0.9, linestyle="--", alpha=0.35,
                label="LR (ERA5, > Nyq)",
                zorder=ZORDER_LR - 1,
            )

        ax.axvline(
            x=lam_nyq, color="black", lw=0.6, linestyle="--",
            label="LR Nyq", zorder=ZORDER_ANNOT,
        )

        # Solid, trusted LR (native spacing) on top of its ghost
        if np.any(trusted):
            if lr_hr_mean is not None:
                ax.plot(
                    lam[trusted], lr_hr_mean[trusted],
                    color=col_lr, lw=1.2, label="LR (ERA5 <= Nyq)",
                    zorder=ZORDER_LR,
                )
            else:
                ax.plot(
                    lam[trusted], lr_mean[trusted],
                    color=col_lr, lw=1.2, label="LR (ERA5 <= Nyq)",
                    zorder=ZORDER_LR,
                )
    else:
        # no Nyquist info → single line
        ax.plot(
            lam, lr_mean,
            color=col_lr, lw=1.2, label="LR (ERA5)",
            zorder=ZORDER_LR,
        )

    # # --- Add HR-LR crossing line and annotation ---
    # if cross_info is not None:
    #     lam_cross, k_cross = cross_info
    #     ax.axvline(x=lam_cross, color="magenta", lw=0.7, ls=":", label="HR-LR intersextion")
    #     y_max = ax.get_ylim()[1]
    #     ax.text(
    #         lam_cross,
    #         y_max * 0.45,
    #         f"k={k_cross:.3e}\nλ={lam_cross:.0f} km",
    #         color="magenta",
    #         ha="right",
    #         va="center",
    #         fontsize=7,
    #         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.55),
    #         rotation=90,
    #     )

    # --- mark low-k and high-k limits ---
    ax.axvline(1.0 / low_k_max, color="gray", linestyle="--",
               linewidth=0.8, alpha=0.7, zorder=ZORDER_ANNOT)
    # move slightly to the left to avoid overlap with high-k line
    x1 = 1.0 / low_k_max * 1.08
    y1 = 1e2 # 0.5
    ax.text(x1, y1, f"low-k λ={1.0/low_k_max:.0f} km",
        rotation=90, color="gray", fontsize=6.5, ha="center", va="bottom",)

    ax.axvline(1.0 / high_k_min, color="gray", linestyle="--",
               linewidth=0.8, alpha=0.7, zorder=ZORDER_ANNOT)
    x2 = 1.0 / high_k_min * 1.08
    y2 = 1e2 # 0.5
    ax.text(x2, y2, f"high-k λ={1.0/high_k_min:.0f} km",
        rotation=90, color="gray", fontsize=6.5, ha="center", va="bottom",)


    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Wavelength λ (km)")
    ax.set_ylabel("Spectral power")
    ax.set_title("Isotropic Power Spectral Density (PSD)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    # Put legend outside to the right
    fig.subplots_adjust(right=0.78)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=7, frameon=True, borderaxespad=0.0)

    # Never go below 1e-8 on the y-axis
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(bottom=5e-6, top=ymax)

    # --- annotate band ratios (more precision) ---
    lines = [
        f"low\u2011k (k ≤ {low_k_max:.3e}, λ ≥ {1.0/low_k_max:.0f} km)",
        f"  GEN / HR = {gen_hr_low_ratio:.6f}",
    ]
    if np.isfinite(lr_hr_low_ratio):
        lines.append(f"  LR  / HR = {lr_hr_low_ratio:.6f}")
    lines += [
        "",  # blank line
        f"high\u2011k (k ≥ {high_k_min:.3e}, λ ≤ {1.0/high_k_min:.0f} km)",
        f"  GEN / HR = {gen_hr_high_ratio:.6f}",
    ]
    if np.isfinite(lr_hr_high_ratio):
        lines.append(f"  LR  / HR = {lr_hr_high_ratio:.6f}")

    # --- slopes for HR, LR, PMM, and GEN (ensemble mean if available) ---
    def _get_slope(series: str, rng: str) -> float | None:
        vals = [r[2] for r in slope_rows if r[0] == series and r[1] == rng]
        return vals[0] if vals else None

    lines += ["", "slopes (log10 k – log10 P):"]
    sl_hr_hi = _get_slope("HR", "high-k")
    if sl_hr_hi is not None: lines += [f"  HR   high-k: {sl_hr_hi:.2f}"]
    sl_lr_hi = _get_slope("LR", "high-k")
    if sl_lr_hi is not None: lines += [f"  LR   high-k: {sl_lr_hi:.2f}"]
    sl_gen_ens_hi = _get_slope("GEN_ens", "high-k")
    if sl_gen_ens_hi is not None:
        lines += [f"  GEN  high-k: {sl_gen_ens_hi:.2f}"]   # GEN = ensemble mean
    sl_gen_pmm_hi = _get_slope("GEN", "high-k")
    if sl_gen_pmm_hi is not None:
        lines += [f"  PMM  high-k: {sl_gen_pmm_hi:.2f}"]
    ax.text(
        0.01,
        0.01,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.7,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.7),
    )

    # ---- Baseline overlays (PSD) ----
    bo = getattr(eval_cfg, "baselines_overlay", None) if eval_cfg is not None else None
    if bo and bo.get("enabled", False):
        try:
            dirs = resolve_baseline_dirs(
                sample_root=bo["sample_root"],
                types=tuple(bo.get("types", ())),
                split=str(bo.get("split", "test")),
                eval_type="scale",
            )
        except Exception as e:
            logger.warning(f"[plot_scale_psd] resolve_baseline_dirs failed: {e}")
            dirs = {}
        labels = bo.get("labels", {})
        styles = bo.get("styles", {})
        for t, d in dirs.items():
            b_npz = d / "scale_psd_curves.npz"
            if not b_npz.exists():
                continue
            try:
                with np.load(b_npz) as bdat:
                    bk = bdat["k"]
                    # prefer generated curve if present (baseline “forecast”)
                    if "psd_gen" in bdat.files:
                        bP = bdat["psd_gen"].mean(axis=0)
                    elif "psd_lr" in bdat.files:
                        bP = bdat["psd_lr"].mean(axis=0)
                    elif "psd_lr_hr" in bdat.files:
                        bP = bdat["psd_lr_hr"].mean(axis=0)
                    else:
                        continue
                # convert to wavelength and sort like the main plot
                mask_pos_b = bk > 0.0
                lam_b = 1.0 / bk[mask_pos_b]
                ord_b = np.argsort(lam_b)[::-1]
                lam_b = lam_b[ord_b]
                bP = np.maximum(bP[mask_pos_b][ord_b], eps)
                label = labels.get(t, t)
                style = dict(styles.get(t, {}))
                style = dict(style)
                style.setdefault("zorder", ZORDER_BASELINE)
                ax.plot(lam_b, bP, label=label, **style)
            except Exception as e:
                logger.warning(f"[plot_scale_psd] Failed to overlay baseline '{t}': {e}")
        # ensure legend includes baselines
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels,
                loc="upper right", bbox_to_anchor=(0.98, 0.98),
                fontsize=8, frameon=True)
        # never go below 1e-8 on y
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(bottom=5e-6, top=max(ymax, 1e-7))        
    _savefig(fig, figs / "scale_psd.png", dpi=SET_DPI)





# ================================================================================
# 2. PSD low/high band ratio diag plot
# ================================================================================

# in sbgm/evaluate/evaluate_prcp/eval_scale/plot_scale.py

def plot_psd_lowhigh_diag(scale_root: Path, eval_cfg: Any | None = None) -> None:
    """
    Summarize PSD low/high band ratios across *all* days.

    We read scale_psd_summary.csv (written by evaluate_scale.py) and build boxplots for:
      - GEN / HR (low-k)
      - GEN / LR (low-k)  [only if present]
      - GEN / HR (high-k)

    This is much more stable than plotting vs date index, because a single day with
    ~zero HR high-k power can blow up the ratio.
    """
    tables = scale_root / "tables"
    figs = _ensure_dir(scale_root / "figures")
    csv_path = tables / "scale_psd_summary.csv"
    if not csv_path.exists():
        logger.warning(f"[plot_psd_lowhigh_diag] Did not find {csv_path} - skipping.")
        return

    # read rows (no pandas)
    gen_low_hr: list[float] = []
    gen_low_lr: list[float] = []
    gen_high_hr: list[float] = []

    with open(csv_path, "r") as f:
        header = f.readline().strip().split(",")
        # expected (from evaluate_scale.py):
        # date,hr_lowk,gen_lowk,gen_lowk_vs_hr,gen_lowk_vs_lr,hr_highk,gen_highk,gen_highk_vs_hr
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue

            def _to_float(s: str) -> float | None:
                s = s.strip()
                if s == "":
                    return None
                try:
                    return float(s)
                except Exception:
                    return None

            hr_lowk = _to_float(parts[1])
            gen_lowk = _to_float(parts[2])
            hr_highk = _to_float(parts[5])
            gen_highk = _to_float(parts[6])
            gl_lr_csv = _to_float(parts[4])  # optional
            gl_hr_csv = _to_float(parts[3])
            gh_hr_csv = _to_float(parts[7])

            # prefer recomputation from powers, fall back to CSV if needed
            gl_hr = None
            if (hr_lowk is not None and gen_lowk is not None and
                    np.isfinite(hr_lowk) and np.isfinite(gen_lowk) and hr_lowk > 0.0):
                gl_hr = float(gen_lowk / max(hr_lowk, 1e-20))
            elif gl_hr_csv is not None and np.isfinite(gl_hr_csv):
                gl_hr = float(gl_hr_csv)

            gl_lr = gl_lr_csv if (gl_lr_csv is not None and np.isfinite(gl_lr_csv)) else None

            gh_hr = None
            if (hr_highk is not None and gen_highk is not None and
                    np.isfinite(hr_highk) and np.isfinite(gen_highk) and hr_highk > 0.0):
                gh_hr = float(gen_highk / max(hr_highk, 1e-20))
            elif gh_hr_csv is not None and np.isfinite(gh_hr_csv):
                gh_hr = float(gh_hr_csv)

            if gl_hr is not None and np.isfinite(gl_hr):
                gen_low_hr.append(float(gl_hr))
            if gl_lr is not None and np.isfinite(gl_lr):
                gen_low_lr.append(float(gl_lr))
            if gh_hr is not None and np.isfinite(gh_hr):
                gen_high_hr.append(float(gh_hr))

    if not gen_low_hr and not gen_low_lr and not gen_high_hr:
        logger.warning("[plot_psd_lowhigh_diag] No valid ratios found - skipping plot.")
        return

    # build the boxplot data
    labels = []
    data = []

    if gen_low_hr:
        labels.append("GEN / HR (low-k)")
        data.append(gen_low_hr)
    if gen_low_lr:
        labels.append("GEN / LR (low-k)")
        data.append(gen_low_lr)
    if gen_high_hr:
        labels.append("GEN / HR (high-k)")
        data.append(gen_high_hr)


    _nice()
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    # clip ratios to [0, 5] only for display
    def _clip(a): return np.clip(np.asarray(a, dtype=float), 0, 5)
    data = [ _clip(a) for a in data ]

    fig.subplots_adjust(left=0.28)  # avoid y-label cutoff on narrow canvases
    bp = ax.boxplot(
        data,
        vert=False,
        labels=labels,
        showfliers=True,
        patch_artist=True,
    )

    # light fill like you used for PSD plot
    for patch in bp["boxes"]:
        p: Any = patch
        try:
            p.set_facecolor("0.9")
        except Exception:
            # Some backends/types may return Line2D objects without set_facecolor;
            # try a more generic set_color fallback and ignore failures.
            try:
                p.set_color("0.9")
            except Exception:
                pass

    # annotate means
    for y, arr in enumerate(data, start=1):
        arr_np = np.asarray(arr, dtype=float)
        mean_val = float(arr_np.mean())
        ax.text(
            mean_val,
            y,
            f"  μ={mean_val:.2f}",
            va="center",
            ha="left",
            fontsize=9,
        )

    # --- set informative x-limits and note if clipped ---
    xmax = max([max(a) for a in data]) if data else 2.0
    if xmax > 5.0:
        ax.set_xlim(0, 5.0)
        ax.text(0.99, 0.02, "(values >5 clipped)", transform=ax.transAxes, ha="right", va="bottom", fontsize=7, alpha=0.5)
    else:
        ax.set_xlim(0, max(2.0, xmax * 1.05))

    # --- subtitle with bands in k and λ ---
    low_k_max = 1.0 / 200.0  # cycles/km → λ = 200 km
    high_k_min = 1.0 / 20.0  # cycles/km → λ = 20 km

    ax.set_xlabel("Ratio")
    ax.set_title("PSD band ratios (all days)")
    ax.text(
        0.01,
        1.02,
        f"low-k: k ≤ {low_k_max:.3e} (λ ≥ {1.0/low_k_max:.0f} km)\n"
        f"high-k: k ≥ {high_k_min:.3e} (λ ≤ {1.0/high_k_min:.0f} km)",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
    )
    ax.grid(True, axis="x", ls=":", alpha=0.5)

    _savefig(fig, figs / "scale_psd_lowhigh.png", dpi=SET_DPI)




# ================================================================================
# 3. FSS vs scale curves
# ================================================================================

def plot_fss_curves(scale_root: Path, eval_cfg: Any | None = None) -> None:
    """
    Read FSS outputs and make **one** multi-panel figure:
      - 1 subplot per base threshold (gen + LR if present)
      - 1 final subplot with all thresholds together
    Layout: 2 rows x 3 columns (covers up to 5 thresholds + 1 overview).
    If there are more than 5 thresholds, extra ones are added to the last row.
    
    Primary source:  <scale_root>/tables/scale_fss_summary.csv
    Optional source: <scale_root>/tables/scale_fss_daily.csv  (used ONLY to recover
    LR baselines, because the summary file currently does not contain LR rows.)

    """
    tables = scale_root / "tables"
    figs = _ensure_dir(scale_root / "figures")
    summary_path = tables / "scale_fss_summary.csv"
    if not summary_path.exists():
        logger.warning(f"[plot_fss_curves] Did not find {summary_path} - skipping FSS plot.")
        return

    ens_summary_path = tables / "scale_fss_ens_summary.csv"
    by_thr_ens: dict[str, list[tuple[float, float]]] = {}
    if ens_summary_path.exists():
        with open(ens_summary_path, "r") as f:
            lines2 = [l.strip() for l in f.readlines() if l.strip()]
        if len(lines2) > 1:
            header2 = lines2[0].split(",")
            rows2 = [l.split(",") for l in lines2[1:]]
            fss_cols2 = [(i, col) for i, col in enumerate(header2) if col.lower().startswith("fss_")]
            for r in rows2:
                base_thr = r[0].strip()
                if base_thr == "":
                    continue
                for idx, col in fss_cols2:
                    try:
                        scale_km = float(col.split("_")[1].replace("km", ""))
                    except Exception:
                        continue
                    v = r[idx].strip()
                    if v == "":
                        continue
                    by_thr_ens.setdefault(base_thr, []).append((scale_km, float(v)))

    # ------------------------------------------------------------
    # 1) Read summary (always available) - gives us GEN per threshold
    # ------------------------------------------------------------
    with open(summary_path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    if not lines:
        logger.warning("[plot_fss_curves] scale_fss_summary.csv is empty.")
        return

    header = lines[0].split(",")
    rows = [l.split(",") for l in lines[1:]]

    # find FSS columns
    fss_cols = [(i, col) for i, col in enumerate(header) if col.lower().startswith("fss_")]
    if not fss_cols:
        logger.warning("[plot_fss_curves] No FSS_* columns found.")
        return

    # by_thr["1.00"] = {"gen": [(5,0.5),...], "lr": [(5,0.4), ...]}
    by_thr: dict[str, dict[str, list[tuple[float, float]]]] = {}

    for r in rows:
        base_thr = r[0].strip()  # e.g. "1.00"
        if base_thr == "":
            continue
        
        if base_thr not in by_thr:
            by_thr[base_thr] = {"gen": [], "lr": []}

        for idx, col in fss_cols:
            # col looks like "fss_5km"
            try:
                scale_km = float(col.split("_")[1].replace("km", ""))
            except Exception:
                continue
            val = r[idx].strip()
            if val == "":
                continue
            by_thr[base_thr]["gen"].append((scale_km, float(val)))

    # ------------------------------------------------------------
    # 2) Try to recover LR baselines from the *daily* CSV
    #    (evaluate_scale.py currently only writes LR lines there)
    # ------------------------------------------------------------
    daily_path = tables / "scale_fss_daily.csv"
    if daily_path.exists():
        with open(daily_path, "r") as f:
            d_lines = [l.strip() for l in f.readlines() if l.strip()]
        if len(d_lines) > 1:
            d_header = d_lines[0].split(",")  # ["date", "thr_mm", "fss_5km", ...]
            d_rows = [l.split(",") for l in d_lines[1:]]
            d_fss_cols = [(i, col) for i, col in enumerate(d_header) if col.lower().startswith("fss_")]
            # tmp: (base_thr -> scale_km -> list of values)
            lr_acc: dict[str, dict[float, list[float]]] = {}
            for r in d_rows:
                thr_str = r[1].strip()  # e.g. "1.00" or "1.00_LR"
                if not thr_str.endswith("_LR"):
                    continue  # only interested in LR here
                base_thr = thr_str.replace("_LR", "")
                for idx, col in d_fss_cols:
                    try:
                        scale_km = float(col.split("_")[1].replace("km", ""))
                    except Exception:
                        continue
                    val = r[idx].strip()
                    if val == "":
                        continue
                    v = float(val)
                    lr_acc.setdefault(base_thr, {}).setdefault(scale_km, []).append(v)

            # turn accumulators into means, append to by_thr
            for base_thr, per_scale in lr_acc.items():
                if base_thr not in by_thr:
                    by_thr[base_thr] = {"gen": [], "lr": []}
                for scale_km, vals in per_scale.items():
                    m = float(np.mean(vals))
                    by_thr[base_thr]["lr"].append((scale_km, m))
    else:
        logger.debug("[plot_fss_curves] No scale_fss_daily.csv - LR baselines will not be shown.")

    # ------------------------------------------------------------
    # 3) Build figure with subplots
    # ------------------------------------------------------------
    # Sort thresholds numerically if possible
    def _thr_key(x: str) -> float:
        try:
            return float(x)
        except Exception:
            return 9e9

    thr_list = sorted(by_thr.keys(), key=_thr_key)
    n_thr = len(thr_list)
    n_panels = n_thr + 1  # last panel = all thresholds

    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))
    _nice()
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols * 0.6, 4 * nrows * 0.65), sharex=True, sharey=True)
    axs = np.atleast_2d(axs)

    # color cycle for per-threshold lines (will re-use in the last panel)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3", "C4", "C5"])

    # Baseline overlay config (optional)
    bo = getattr(eval_cfg, "baselines_overlay", None) if eval_cfg is not None else None
    if bo and bo.get("enabled", False):
        try:
            from scor_dm.evaluate.evaluate_prcp.overlay_utils import resolve_baseline_dirs
            _baseline_dirs_fss = resolve_baseline_dirs(
                sample_root=bo["sample_root"],
                types=tuple(bo.get("types", ())),
                split=str(bo.get("split", "test")),
                eval_type="scale",
            )
        except Exception as e:
            logger.warning(f"[plot_fss_curves] resolve_baseline_dirs failed: {e}")
            _baseline_dirs_fss = {}
        _baseline_labels = bo.get("labels", {})
        _baseline_styles = bo.get("styles", {})
    else:
        _baseline_dirs_fss = {}
        _baseline_labels = {}
        _baseline_styles = {}

    # -- 3a) individual panels --
    for i, thr in enumerate(thr_list):
        row = i // ncols
        col = i % ncols
        ax = axs[row, col]
        data = by_thr[thr]
        gen_pairs = sorted(data["gen"], key=lambda t: t[0])
        lr_pairs = sorted(data["lr"], key=lambda t: t[0]) if data["lr"] else []
        # color chosen per threshold
        color_thr = colors[i % len(colors)]

        # Ensemble mean (if available): solid colored line
        ens_pairs = sorted(by_thr_ens.get(thr, []), key=lambda t: t[0])
        if ens_pairs:
            ax.plot([p[0] for p in ens_pairs], [p[1] for p in ens_pairs],
                    linestyle="-", linewidth=1.8, marker="o", markersize=3.5,
                    color=COL_ENS, label="Gen (ens)")

        # PMM (formerly "gen"): same color, dashed
        gen_pairs = sorted(data["gen"], key=lambda t: t[0])
        if gen_pairs:
            ax.plot([p[0] for p in gen_pairs], [p[1] for p in gen_pairs],
                    linestyle="-.", linewidth=1.4, marker=".", color=COL_PMM, markersize=2.5,
                    label="PMM")

        # LR baseline (grey), if we reconstructed it
        lr_pairs = sorted(data["lr"], key=lambda t: t[0]) if data["lr"] else []
        if lr_pairs:
            ax.plot([p[0] for p in lr_pairs], [p[1] for p in lr_pairs],
                    marker="x", linestyle="--", linewidth=1.0, color=COL_LR, markersize=2,
                    label="LR")

        # ---- Baseline overlays: per-threshold panel only ----
        if _baseline_dirs_fss:
            for t, d in _baseline_dirs_fss.items():
                sp_b = d / "scale_fss_summary.csv"
                if not sp_b.exists():
                    continue
                try:
                    with open(sp_b, "r") as fb:
                        lines_b = [l.strip() for l in fb.readlines() if l.strip()]
                    if not lines_b:
                        continue
                    header_b = lines_b[0].split(",")
                    rows_b = [l.split(",") for l in lines_b[1:]]
                    fss_cols_b = [(idx, col) for idx, col in enumerate(header_b) if col.lower().startswith("fss_")]
                    # collect points for the current threshold only
                    pts = []
                    for r in rows_b:
                        base_thr_b = r[0].strip()
                        if base_thr_b != thr:
                            continue
                        for idx, col in fss_cols_b:
                            try:
                                scale_km = float(col.split("_")[1].replace("km", ""))
                            except Exception:
                                continue
                            v = r[idx].strip()
                            if v == "":
                                continue
                            pts.append((scale_km, float(v)))
                    if not pts:
                        continue
                    pts = sorted(pts, key=lambda p: p[0])
                    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                    label = _baseline_labels.get(t, t)
                    style = dict(_baseline_styles.get(t, {}))
                    ax.plot(xs, ys, label=label, **style)
                except Exception as e:
                    logger.warning(f"[plot_fss_curves] Failed to overlay FSS baseline '{t}': {e}")

        # Only set labels on leftmost and bottom plots
        if col == 0:
            ax.set_ylabel("FSS")
        if row == nrows - 1:
            ax.set_xlabel("Neighborhood scale (km)")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"FSS vs scale. Thr ≥ {float(thr):.0f} mm/day")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, ls=":", alpha=0.4)

     # -- 3b) combined panel (ensemble-only, one line per threshold) --
    last_row = (n_panels - 1) // ncols
    last_col = (n_panels - 1) % ncols
    ax_all = axs[last_row, last_col]
    ax_all.set_title("FSS vs scale (ensemble only)")

    any_ens = False
    for i, thr in enumerate(thr_list):
        ens_pairs = sorted(by_thr_ens.get(thr, []), key=lambda t: t[0])
        if not ens_pairs:
            continue
        xs = [p[0] for p in ens_pairs]
        ys = [p[1] for p in ens_pairs]
        # one line per threshold (ensemble mean only)
        ax_all.plot(
            xs, ys,
            linewidth=1.6,
            marker=".",
            label=f"≥ {float(thr):.0f} mm"
        )
        any_ens = True

    ax_all.set_xlabel("Neighborhood scale (km)")
    ax_all.set_ylabel("FSS")
    ax_all.set_ylim(0.0, 1.0)
    ax_all.grid(True, ls=":", alpha=0.4)

    # compact legend outside the axis
    if any_ens:
        h, l = ax_all.get_legend_handles_labels()
        ax_all.legend(h, l, ncol=1, fontsize=8, frameon=True,
                      loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    else:
        ax_all.text(0.5, 0.5, "No ensemble summary available",
                    ha="center", va="center", transform=ax_all.transAxes, fontsize=9, alpha=0.7)

    # turn off any unused axes (if grid has extra cells)
    for j in range(n_panels, nrows * ncols):
        r, c = divmod(j, ncols)
        axs[r, c].axis("off")

    _savefig(fig, figs / "scale_fss.png", dpi=SET_DPI)




# ================================================================================
# 4. ISS at scales
# ================================================================================

def plot_iss_curves(scale_root: Path, eval_cfg: Any | None = None) -> None:
    """
    ISS vs scale, same layout as FSS:
      - 1 panel per threshold (GEN + LR if present)
      - 1 final panel with all thresholds together
    Reads:
      - <scale_root>/tables/scale_iss_summary.csv  (always)
      - <scale_root>/tables/scale_iss_daily.csv    (to recover LR baselines)
      - <scale_root>/tables/scale_iss_ens_summary.csv (optional ensemble-mean)      
    """
    tables = scale_root / "tables"
    figs = _ensure_dir(scale_root / "figures")
    summary_path = tables / "scale_iss_summary.csv"

    # --- read ensemble ISS summary (optional) ---
    ens_summary_path = tables / "scale_iss_ens_summary.csv"
    by_thr_ens: dict[str, list[tuple[float, float]]] = {}
    if ens_summary_path.exists():
        with open(ens_summary_path, "r") as f:
            lines2 = [l.strip() for l in f.readlines() if l.strip()]
        if len(lines2) > 1:
            header2 = lines2[0].split(",")
            rows2 = [l.split(",") for l in lines2[1:]]
            iss_cols2 = [(i, col) for i, col in enumerate(header2) if col.lower().startswith("iss_")]
            for r in rows2:
                base_thr = r[0].strip()
                if base_thr == "":
                    continue
                for idx, col in iss_cols2:
                    try:
                        scale_km = float(col.split("_")[1].replace("km", ""))
                    except Exception:
                        continue
                    v = r[idx].strip()
                    if v == "":
                        continue
                    by_thr_ens.setdefault(base_thr, []).append((scale_km, float(v)))
    else:
        logger.debug("[plot_iss_curves] No scale_iss_ens_summary.csv - ensemble means will not be shown.")

    if not summary_path.exists():
        logger.warning(f"[plot_iss_curves] Did not find {summary_path} - skipping ISS plot.")
        return

    with open(summary_path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    if not lines:
        logger.warning("[plot_iss_curves] scale_iss_summary.csv is empty - skipping.")
        return

    header = lines[0].split(",")
    rows = [l.split(",") for l in lines[1:]]

    iss_cols = [(i, col) for i, col in enumerate(header) if col.lower().startswith("iss_")]
    if not iss_cols:
        logger.warning("[plot_iss_curves] No ISS_* columns found - skipping.")
        return

    # by_thr["1.00"] = {"gen": [(5,0.7), ...], "lr": [(5,0.8), ...]}
    by_thr: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for r in rows:
        thr_id = r[0].strip()
        if not thr_id:
            continue
        by_thr.setdefault(thr_id, {"gen": [], "lr": []})
        for idx, col in iss_cols:
            try:
                scale_km = float(col.split("_")[1].replace("km", ""))
            except Exception:
                continue
            val = r[idx].strip()
            if val == "":
                continue
            by_thr[thr_id]["gen"].append((scale_km, float(val)))

    # try to get LR baselines from daily CSV
    daily_path = tables / "scale_iss_daily.csv"
    if daily_path.exists():
        with open(daily_path, "r") as f:
            d_lines = [l.strip() for l in f.readlines() if l.strip()]
        if len(d_lines) > 1:
            d_header = d_lines[0].split(",")
            d_rows = [l.split(",") for l in d_lines[1:]]
            d_iss_cols = [(i, col) for i, col in enumerate(d_header) if col.lower().startswith("iss_")]
            lr_acc: dict[str, dict[float, list[float]]] = {}
            for r in d_rows:
                thr_raw = r[1].strip()  # e.g. "1.00" or "1.00_LR"
                is_lr = thr_raw.endswith("_LR")
                base_thr = thr_raw.replace("_LR", "")
                if base_thr not in by_thr:
                    continue
                for idx, col in d_iss_cols:
                    try:
                        scale_km = float(col.split("_")[1].replace("km", ""))
                    except Exception:
                        continue
                    v = r[idx].strip()
                    if v == "":
                        continue
                    if is_lr:
                        lr_acc.setdefault(base_thr, {}).setdefault(scale_km, []).append(float(v))
            # push averaged LR back into by_thr
            for thr, scales in lr_acc.items():
                for scale_km, arr in scales.items():
                    mean_lr = float(np.mean(arr))
                    by_thr[thr]["lr"].append((scale_km, mean_lr))

    thrs_sorted = sorted(by_thr.keys(), key=lambda s: float(s))
    n_thr = len(thrs_sorted)
    n_panels = n_thr + 1  # overview
    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))

    _nice()
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols * 0.6, 4 * nrows * 0.65), sharex=True, sharey=True)
    axs = np.atleast_2d(axs)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3", "C4", "C5"])

    # Baseline overlay config (optional)
    bo = getattr(eval_cfg, "baselines_overlay", None) if eval_cfg is not None else None
    if bo and bo.get("enabled", False):
        try:
            from scor_dm.evaluate.evaluate_prcp.overlay_utils import resolve_baseline_dirs
            _baseline_dirs_iss = resolve_baseline_dirs(
                sample_root=bo["sample_root"],
                types=tuple(bo.get("types", ())),
                split=str(bo.get("split", "test")),
                eval_type="scale",
            )
        except Exception as e:
            logger.warning(f"[plot_iss_curves] resolve_baseline_dirs failed: {e}")
            _baseline_dirs_iss = {}
        _baseline_labels = bo.get("labels", {})
        _baseline_styles = bo.get("styles", {})
    else:
        _baseline_dirs_iss = {}
        _baseline_labels = {}
        _baseline_styles = {}

    for i, thr in enumerate(thrs_sorted):
        row = i // ncols
        col = i % ncols
        # leave the very last cell for the overview
        ax = axs[row, col]

        # ensemble mean curve (if present)
        ens_pairs = sorted(by_thr_ens.get(thr, []), key=lambda t: t[0])
        x = [p[0] for p in ens_pairs]
        y = [p[1] for p in ens_pairs]
        if ens_pairs:
            ax.plot(x, y, marker=".", linewidth=1.4, color=COL_ENS, label="Gen (ens)")

        # PMM / GEN curve
        gen_pts = sorted(by_thr[thr]["gen"], key=lambda p: p[0])
        x = [p[0] for p in gen_pts]
        y = [p[1] for p in gen_pts]
        colr = COL_PMM
        ax.plot(x, y, marker=".", linewidth=1.4, color=colr, label=f"PMM")

        lr_pts = sorted(by_thr[thr]["lr"], key=lambda p: p[0]) if by_thr[thr]["lr"] else []
        if lr_pts:
            ax.plot([p[0] for p in lr_pts], [p[1] for p in lr_pts],
                    linestyle="--", marker="x", linewidth=1.0, color=COL_LR,
                    label=f"LR")


        # ---- Baseline overlays: per-threshold panel only ----
        if _baseline_dirs_iss:
            for t, d in _baseline_dirs_iss.items():
                sp_b = d / "scale_iss_summary.csv"
                if not sp_b.exists():
                    continue
                try:
                    with open(sp_b, "r") as fb:
                        lines_b = [l.strip() for l in fb.readlines() if l.strip()]
                    if not lines_b:
                        continue
                    header_b = lines_b[0].split(",")
                    rows_b = [l.split(",") for l in lines_b[1:]]
                    iss_cols_b = [(idx, col) for idx, col in enumerate(header_b) if col.lower().startswith("iss_")]
                    pts = []
                    for r in rows_b:
                        thr_b = r[0].strip()
                        if thr_b != thr:
                            continue
                        for idx, col in iss_cols_b:
                            try:
                                scale_km = float(col.split("_")[1].replace("km", ""))
                            except Exception:
                                continue
                            v = r[idx].strip()
                            if v == "":
                                continue
                            pts.append((scale_km, float(v)))
                    if not pts:
                        continue
                    pts = sorted(pts, key=lambda p: p[0])
                    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                    label = _baseline_labels.get(t, t)
                    style = dict(_baseline_styles.get(t, {}))
                    ax.plot(xs, ys, label=label, **style)
                except Exception as e:
                    logger.warning(f"[plot_iss_curves] Failed to overlay ISS baseline '{t}': {e}")

        ax.set_ylim(0.0, 1.05)
        # Set title as per-threshold
        ax.set_title(f"ISS vs scale. Thr ≥ {float(thr):.0f} mm")
        
        # Only plot xlabel on bottom row
        if row == nrows - 1:
            ax.set_xlabel("Neighborhood scale (km)")
        # only plot ylabel on leftmost column
        if col == 0:
            ax.set_ylabel(f"ISS vs scale.")
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(fontsize=9, loc="lower right")

    # overview panel in the last slot
    last_idx = n_panels - 1
    ov_row = last_idx // ncols
    ov_col = last_idx % ncols
    ax_all = axs[ov_row, ov_col]
    for i, thr in enumerate(thrs_sorted):
        gen_pts = sorted(by_thr[thr]["gen"], key=lambda p: p[0])
        if not gen_pts:
            continue
        x = [p[0] for p in gen_pts]
        y = [p[1] for p in gen_pts]
        colr = colors[i % len(colors)]
        ax_all.plot(x, y, marker=".", linewidth=1.2, color=colr)
        ax_all.text(x[-1] * 1.01, y[-1], f"≥ {float(thr):.0f} mm", color=colr, fontsize=4, va="center")

        # lr_pts = sorted(by_thr[thr]["lr"], key=lambda p: p[0]) if by_thr[thr]["lr"] else []
        # if lr_pts:
        #     ax_all.plot([p[0] for p in lr_pts], [p[1] for p in lr_pts],
        #                 linestyle="--", linewidth=0.8, color="0.5", alpha=0.7)

    ax_all.set_ylim(0.3, 1.01)
    ax_all.set_xlabel("Neighborhood scale (km)")
    ax_all.set_ylabel("ISS")
    ax_all.set_title("ISS vs scale (all thresholds)")
    ax_all.grid(True, ls=":", alpha=0.5)

    # turn off empty axes (if any)
    for j in range(n_panels, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axs[r, c].axis("off")

    fig.tight_layout()
    _savefig(fig, figs / "scale_iss_curves.png", dpi=SET_DPI)



# ================================================================================
# Master entry point
# ================================================================================

def plot_scale(eval_root: str | Path, eval_cfg: Any | None = None) -> None:
    """
    Master entry point - call this from evaluate_scale.py

    baseline_eval_dirs is kept here for symmetry with your old plotting module,
    but we don’t actually use it yet (easy to add later).
    """
    scale_root = Path(eval_root)
    if not scale_root.exists():
        logger.warning(f"[plot_scale] {scale_root} does not exist.")
        return

    plot_scale_psd(scale_root, eval_cfg=eval_cfg)
    plot_fss_curves(scale_root, eval_cfg=eval_cfg)
    plot_iss_curves(scale_root, eval_cfg=eval_cfg)
    plot_psd_lowhigh_diag(scale_root, eval_cfg=eval_cfg)
    logger.info(f"[plot_scale] Plots written to {scale_root / 'figures'}")