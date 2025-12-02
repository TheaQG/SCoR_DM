"""
    Plotting utilities for the precipitation probabilistic evaluation block.

    Reads the artefacts that "sbgm.evaluate.evaluate_prcp.eval_probabilistic" writes to:
        <eval_root>/tables/
            - prob_crps_daily.csv
            - prob_reliability_{thr}.csv
            - prob_spread_skill.csv
            - prob_rank_histogram.npz
            - prob_pit_values.npz

    and creates plots in:
        <eval_root>/figures/
            - prob_crps_summary.png
            - prob_pit_hist.png
            - prob_rank_hist.png
            - prob_reliability_{thr}.png
            - prob_spread_skill.png
"""


from __future__ import annotations
from pathlib import Path
from typing import Sequence, Optional, Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from datetime import datetime
import logging

from mpl_toolkits.axes_grid1 import make_axes_locatable

from evaluate.evaluate_prcp.plot_utils import (
    _ensure_dir, _savefig, _nice, _to_date_safe, _season_from_month, overlay_outline, get_dk_lsm_outline
)
from scor_dm.variable_utils import get_cmap_for_variable, get_color_for_model

logger = logging.getLogger(__name__)

SET_DPI = 300

# Set colors
col_hr = get_color_for_model("HR")
col_gen = get_color_for_model("gen")
col_lr = get_color_for_model("LR")

# Set colormap for variables
cmap_precip = get_cmap_for_variable("prcp")

# ================================================================================
# 1. PIT
# ================================================================================

def plot_pit(eval_root: str | Path, *, bins: int = 20):
    eval_root = Path(eval_root)
    tables_dir = eval_root / "tables"
    figs_dir = _ensure_dir(eval_root / "figures")

    pit_file = tables_dir / "prob_pit_values.npz"
    if not pit_file.exists():
        # nothing to do
        return

    arr = np.load(pit_file)
    pit = arr["pit"]
    # guard
    if pit is None or len(pit) == 0:
        return

    n = int(len(pit))
    k = int(max(2, bins))  # at least 2 bins for the CI math


    # 1x2: histogram + ECDF (more sensitive, bin-free check)
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6))
    _nice()
    ax = axs[0]

    # --- Histogram ---
    # density=True makes the uniform reference equal to 1
    ax.hist(pit, bins=k, range=(0, 1), density=True, edgecolor="white", linewidth=0.5, color=col_gen)

    # 95% sampling-variability band for a uniform[0,1] null (normal approx)
    # For density heights with bin width w=1/k, sd ≈ sqrt(p(1-p)/n)/w with p=1/k,
    # which simplifies to sd ≈ sqrt((k-1)/n). See standard binomial variance.
    z = 1.96
    sd = np.sqrt(max(0.0, (k - 1) / n))
    lo, hi = max(0.0, 1.0 - z * sd), 1.0 + z * sd
    ax.fill_between([0, 1], lo, hi, color="0.7", alpha=0.25, label="95% sampling band")

    # uniform reference
    ax.hlines(1.0, 0, 1, linestyles="dashed", colors=["0.3"], linewidth=1.0, label="Uniform")

    ax.set_xlim(0, 1)
    ax.set_xlabel("PIT")
    ax.set_ylabel("Density")
    ax.set_title(f"PIT histogram (N={n:,}, bins={k})")
    ax.legend(loc="upper right", frameon=False)

    # --- ECDF with KS distance ---
    ax2 = axs[1]
    x = np.sort(np.asarray(pit).ravel())
    y = np.arange(1, n + 1) / n
    # steps-post to show the empirical CDF cleanly
    ax2.step(x, y, where="post", lw=1.2, label="ECDF", color=col_gen)
    ax2.plot([0, 1], [0, 1], "k--", lw=1.0, label="Uniform CDF")

    # One-sample KS statistic against U(0,1) without scipy
    y_prev = np.arange(0, n) / n
    D_plus = np.max(y - x)
    D_minus = np.max(x - y_prev)
    D = float(max(D_plus, D_minus))

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("PIT")
    ax2.set_ylabel("Cumulative prob.")
    ax2.set_title(f"ECDF vs uniform (KS D = {D:.3f})")
    ax2.legend(loc="lower right", frameon=False)
    ax2.grid(True, ls=":", alpha=0.4)

    fig.tight_layout()
    _savefig(fig, figs_dir / "prob_pit_hist.png", dpi=SET_DPI)



# ================================================================================
# 2. Rank histogram
# ================================================================================

def plot_rank(eval_root: str | Path):
    eval_root = Path(eval_root)
    tables_dir = eval_root / "tables"
    figs_dir = _ensure_dir(eval_root / "figures")

    rh_file = tables_dir / "prob_rank_histogram.npz"
    if not rh_file.exists():
        return

    arr = np.load(rh_file)
    counts = np.asarray(arr["rank_hist"], dtype=float)

    K = int(len(counts))          # number of rank bins = M+1
    if K == 0:
        return
    N = float(np.sum(counts))     # total verifications
    M = K - 1                     # ensemble size

    # expected count per bin and 95% band under perfect reliability
    p = 1.0 / K
    exp = N * p
    sd = np.sqrt(N * p * (1.0 - p))  # multinomial -> binomial per bin
    z = 1.96
    lo, hi = exp - z * sd, exp + z * sd


    _nice()
    fig, ax = plt.subplots()

    x = np.arange(K)
    ax.bar(x, counts, width=0.85, edgecolor="white", linewidth=0.5, label="Observed", color=col_gen)

    # uniform reference and CI band (drawn across the full span of bars)
    ax.fill_between([-0.5, K - 0.5], [lo, lo], [hi, hi], color="0.7", alpha=0.25, # type: ignore
                    label="95% sampling band")
    ax.hlines(exp, -0.5, K - 0.5, linestyles="dashed", colors=["0.3"], linewidth=1.0,
              label="Uniform expected")

    # x ticks at key ranks
    tick_locs = [0, M // 2, M]
    tick_locs = sorted(set(int(t) for t in tick_locs))
    ax.set_xticks(tick_locs)
    ax.set_xticklabels([str(t) for t in tick_locs])

    ax.set_xlim(-0.5, K - 0.5)
    ax.set_xlabel("Rank (0..M)")
    ax.set_ylabel("Count")

    # secondary y-axis in proportions
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim()[0] / N, ax.get_ylim()[1] / N)
    ax2.set_ylabel("Proportion")

    # annotate a simple diagnostic: max absolute standardized deviation
    z_scores = (counts - exp) / (sd if sd > 0 else 1.0)
    max_abs = float(np.max(np.abs(z_scores))) if N > 0 else 0.0

    ax.set_title(f"Rank histogram (M={M}, N={int(N):,})\nmax |z| per bin = {max_abs:.2f}")
    ax.legend(loc="upper left", frameon=False)

    _savefig(fig, figs_dir / "prob_rank_hist.png", dpi=SET_DPI)




# ================================================================================
# 3. Reliability diagrams
# ================================================================================

def plot_reliability(
    eval_root: str | Path,
    *,
    thresholds: Sequence[float] = (1.0, 5.0, 10.0),
    min_count_to_show: int = 150,
):
    """
    Expects files like:
        <tables>/prob_reliability_1.0mm.csv
        <tables>/prob_reliability_5.0mm.csv
    written by evaluate_probabilistic.py.

    Produces a single grid figure combining all available thresholds.    
    """
    eval_root = Path(eval_root)
    tables_dir = eval_root / "tables"
    figs_dir = _ensure_dir(eval_root / "figures")
    
    # Load all available thresholds
    panels = []  # list of dicts with keys: thr, pf, po, cnt
    for thr in thresholds:
        fpath = tables_dir / f"prob_reliability_{thr:.1f}mm.csv"
        if not fpath.exists():
            continue

        lines = fpath.read_text().strip().splitlines()
        if len(lines) <= 1:
            continue

        bc, pf, po, cnt = [], [], [], []
        for ln in lines[1:]:
            s = ln.split(",")
            if len(s) < 4:
                continue
            bc.append(float(s[0]))
            pf.append(float(s[1]))
            po.append(float(s[2]))
            cnt.append(int(float(s[3])))

        if not pf:
            continue

        bc = np.array(bc, dtype=float)
        pf = np.array(pf, dtype=float)
        po = np.array(po, dtype=float)
        cnt = np.array(cnt, dtype=int)

        # sort by forecast prob so lines don't zig-zag
        order = np.argsort(pf)
        panels.append({
            "thr": thr,
            "pf": pf[order],
            "po": po[order],
            "cnt": cnt[order],
        })

    if not panels:
        return

    # ------------------------------
    # Make a grid (adaptive columns)
    # ------------------------------
    n = len(panels)
    ncols = 1 if n == 1 else (2 if n <= 4 else 3)
    nrows = int(np.ceil(n / ncols))

    _nice()
    fig, axs = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows))
    _nice()

    if nrows == 1 and ncols == 1:
        axs = np.array([[axs]])
    elif nrows == 1:
        axs = np.array([axs])

    # ------------------------------
    # Plot each threshold in its panel
    # ------------------------------
    for i, panel in enumerate(panels):
        r, c = divmod(i, ncols)
        ax = axs[r, c]
        thr = panel["thr"]
        pf = panel["pf"]
        po = panel["po"]
        cnt = panel["cnt"]

        ax.plot([0, 1], [0, 1], "k--", lw=1.0, label="Perfect")

        # draw high-count bins as line+markers, low-count as faint dots
        high = cnt >= min_count_to_show
        low = ~high

        if np.any(high):
            ax.plot(pf[high], po[high], ".-", linewidth=1.0, label=f"≥ {thr:.1f} mm", color=col_gen)

        if np.any(low):
            ax.scatter(pf[low], po[low], s=24, alpha=0.35, edgecolors="none", color=col_gen,)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Forecast probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(f"Reliability - {thr:.1f} mm/day")
        ax.legend(loc="lower right")

        # add count axis (optional)
        ax2 = ax.twinx()
        ax2.bar(pf, cnt, width=0.025, alpha=0.18, color=col_gen)
        ax2.set_ylabel("Bin count")

    # Hide any unused axes in the grid
    total_axes = nrows * ncols
    for j in range(n, total_axes):
        r, c = divmod(j, ncols)
        axs[r, c].axis("off")
    fig.suptitle("Reliability diagrams", y=0.995)
    fig.tight_layout()
    _savefig(fig, figs_dir / "prob_reliability_grid.png", dpi=SET_DPI)



# ================================================================================
# 4. Spread-skill
# ================================================================================


def plot_spread_skill(
    eval_root: str | Path,
    *,
    min_count_to_show: int = 500,
    add_fit: bool = True,
    fit_through_origin: bool = True,
    show_binned_curve: bool = False,
):
    """
    Reads:
        <tables>/prob_spread_skill.csv
    with rows:
        date,spread_mean,skill_mean
    and plots:
      1) time series with sparse date labels and mean lines
      2) spread vs skill scatter
    """
    eval_root = Path(eval_root)
    tables_dir = eval_root / "tables"
    figs_dir = _ensure_dir(eval_root / "figures")

    csv_path = tables_dir / "prob_spread_skill.csv"
    if not csv_path.exists():
        return

    lines = csv_path.read_text().strip().splitlines()
    if len(lines) <= 1:
        return

    dates, spreads, skills = [], [], []
    for ln in lines[1:]:
        s = ln.split(",")
        if len(s) < 3:
            continue
        dates.append(s[0].strip())
        try:
            spreads.append(float(s[1]))
        except Exception:
            spreads.append(np.nan)
        try:
            skills.append(float(s[2]))
        except Exception:
            skills.append(np.nan)

    x = np.arange(len(dates))

    # --- time series ---
    _nice()
    fig, ax = plt.subplots()
    ax.plot(x, spreads, label="spread (mean)", linewidth=1.2, alpha=0.9)
    ax.plot(x, skills, label="skill (mean)", linewidth=1.2, alpha=0.9)

    # mean lines
    if len(spreads):
        m_spread = float(np.nanmean(spreads))
        ax.axhline(m_spread, color="0.5", linestyle="--", linewidth=1.0,
                   label=f"spread mean={m_spread:.2f}")
    if len(skills):
        m_skill = float(np.nanmean(skills))
        ax.axhline(m_skill, color="0.7", linestyle=":", linewidth=1.0,
                   label=f"skill mean={m_skill:.2f}")

    # sparse date labels
    n_labels = min(10, len(dates))
    if n_labels > 0:
        idxs = np.linspace(0, len(dates) - 1, n_labels, dtype=int)
        ax.set_xticks(idxs)
        ax.set_xticklabels([dates[i] for i in idxs], rotation=30, ha="right")

    ax.set_xlabel("sample index (date order)")
    ax.set_ylabel("mm/day")
    ax.set_title("Spread-skill diagnostics (per date)")
    ax.legend()
    _savefig(fig, figs_dir / "prob_spread_skill.png", dpi=SET_DPI)

    # --- spread vs skill scatter ---
    col_gen = get_color_for_model("gen")
    _nice()
    fig, ax = plt.subplots()
    ax.scatter(spreads, skills, s=14, alpha=0.6, edgecolors="none", c=col_gen)
    maxv = float(max(
        np.nanmax(spreads) if len(spreads) else 0.0,
        np.nanmax(skills) if len(skills) else 0.0,
        1.0
    ))

    # Fit line and/or binned mean curve
    spreads_arr = np.asarray(spreads)
    skills_arr = np.asarray(skills)
    mask = np.isfinite(spreads_arr) & np.isfinite(skills_arr)
    x_fit = np.asarray(spreads_arr)[mask].astype(float)
    y_fit = np.asarray(skills_arr)[mask].astype(float)

    if add_fit:
        if fit_through_origin:
            denom = np.sum(x_fit * x_fit)
            slope = float(np.sum(x_fit * y_fit) / denom) if denom != 0 else 0.0
            intercept = 0.0
        else:
            slope, intercept = np.polyfit(x_fit, y_fit, 1)
        # Pearson r
        r = float(np.corrcoef(x_fit, y_fit)[0, 1]) if len(x_fit) > 1 else np.nan
        # Spearman rho (rank correlation) without scipy
        rx = np.argsort(np.argsort(x_fit))
        ry = np.argsort(np.argsort(y_fit))
        mx = rx.astype(float)
        my = ry.astype(float)
        mx -= np.mean(mx)
        my -= np.mean(my)
        denom_rho = np.sqrt(np.sum(mx ** 2)) * np.sqrt(np.sum(my ** 2))
        rho = float(np.sum(mx * my) / denom_rho) if denom_rho != 0 else np.nan
        ax.plot(
            [0, maxv],
            [slope * 0 + intercept, slope * maxv + intercept],
            lw=1.2,
            label=f"fit: y={slope:.2f}x{intercept:+.2f} (r={r:.2f}, ρ={rho:.2f})",
            color=col_gen,
        )

    if show_binned_curve:
        # 12 quantile bins of x_fit
        if len(x_fit) >= 2:
            quantiles = np.linspace(0, 1, 13)
            bins = np.quantile(x_fit, quantiles)
            # make bins monotonic (in case of ties)
            bins = np.unique(bins)
            digitized = np.digitize(x_fit, bins, right=True)
            mx, my = [], []
            for b in range(1, len(bins)):
                sel = digitized == b
                if np.any(sel):
                    mx.append(np.mean(x_fit[sel]))
                    my.append(np.mean(y_fit[sel]))
            if len(mx) > 1:
                ax.plot(mx, my, "-o", lw=1.0, ms=3, alpha=0.9, label="binned mean")

    ax.plot([0, maxv], [0, maxv], "k--", lw=1.0, label="spread = skill")
    ax.set_xlim(0, maxv)
    ax.set_ylim(0, maxv)
    ax.set_xlabel("Spread (mm/day)")
    ax.set_ylabel("Skill (mm/day)")
    ax.set_title("Spread vs skill")
    ax.legend()
    _savefig(fig, figs_dir / "prob_spread_vs_skill.png", dpi=SET_DPI)


# ================================================================================
# 4b. Energy score & Variogram score
# ================================================================================

def plot_energy_variogram(eval_root: str | Path):
    eval_root = Path(eval_root)
    tables_dir = eval_root / "tables"
    figs_dir = _ensure_dir(eval_root / "figures")

    es_path = tables_dir / "prob_energy_daily.csv"
    vs_path = tables_dir / "prob_variogram_daily.csv"

    if (not es_path.exists()) and (not vs_path.exists()):
        return

    dates_es, es_vals = [], []
    if es_path.exists():
        lines = es_path.read_text().strip().splitlines()
        for ln in lines[1:]:
            s = ln.split(",")
            if len(s) < 2:
                continue
            dates_es.append(s[0].strip())
            try:
                es_vals.append(float(s[1]))
            except Exception:
                es_vals.append(np.nan)

    dates_vs, vs_vals = [], []
    if vs_path.exists():
        lines = vs_path.read_text().strip().splitlines()
        for ln in lines[1:]:
            s = ln.split(",")
            if len(s) < 2:
                continue
            dates_vs.append(s[0].strip())
            try:
                vs_vals.append(float(s[1]))
            except Exception:
                vs_vals.append(np.nan)

    _nice()
    fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.0))
    ax1, ax2 = axs

    # --- ES ---
    if es_vals:
        x = np.arange(len(es_vals))
        ax1.plot(x, es_vals, lw=1.0, label="ES", color=col_gen)
        m_es = float(np.nanmean(es_vals))
        ax1.axhline(m_es, color="0.5", ls="--", lw=0.8, label=f"mean={m_es:.3f}")
        ax1.set_title("Energy score (daily)")
        ax1.set_xlabel("sample index")
        ax1.set_ylabel("ES")
        ax1.legend()
        ax1.grid(True, ls=":", alpha=0.4)

    # --- VS ---
    if vs_vals:
        x = np.arange(len(vs_vals))
        ax2.plot(x, vs_vals, lw=1.0, label="VS", color=col_gen)
        m_vs = float(np.nanmean(vs_vals))
        ax2.axhline(m_vs, color="0.5", ls="--", lw=0.8, label=f"mean={m_vs:.3f}")
        ax2.set_title("Variogram score (daily)")
        ax2.set_xlabel("sample index")
        ax2.set_ylabel("VS")
        ax2.legend()
        ax2.grid(True, ls=":", alpha=0.4)

    fig.tight_layout()
    _savefig(fig, figs_dir / "prob_energy_variogram.png", dpi=SET_DPI)


# ================================================================================
# 5. CRPS, examples and timeseries
# ================================================================================

def plot_crps_examples(
        eval_root: str | Path,
        *,
        gen_root: Optional[str | Path] = None,
        n_examples: int = 6,  # selection done when NPZ is created
):
    eval_root = Path(eval_root)
    tables_dir = eval_root / "tables"
    figs_dir = _ensure_dir(eval_root / "figures")

    npz_path = tables_dir / "prob_crps_examples_members.npz"
    if not npz_path.exists():
        return

    data = np.load(npz_path, allow_pickle=True)
    dates = [str(s) for s in data["dates"]]
    T = len(dates)
    if T == 0:
        return

    # Count saved members by probing keys
    def members_saved_for(d: str) -> int:
        j = 0
        while f"MEM_{j}_{d}" in data:
            j += 1
        return j
    n_mem = max(members_saved_for(d) for d in dates)
    if n_mem == 0:
        return

    # Will there be a PMM row?
    has_pmm = any((f"PMM_{d}" in data) for d in dates)
    nrows = 1 + n_mem + (1 if has_pmm else 0)   # HR + members + optional PMM
    ncols = T

    # Mask-aware cmap
    cmap = get_cmap_for_variable("prcp")
    # try:
    #     cmap = cmap.copy() 
    #     cmap.set_bad("#c2c2c2")
    #     cmap.set_under("#c2c2c2")
    # except Exception:
    #     pass

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows))
    if nrows == 1: axes = np.array([axes])
    if ncols == 1: axes = axes[:, None]
    dk_mask = get_dk_lsm_outline()
    # Flip
    if dk_mask is not None:
        dk_mask = np.flipud(dk_mask)

    for t, d in enumerate(dates):
        hr = data[f"HR_{d}"]

        # Stack values for dynamic vmax: HR + all members (+ PMM if present)
        stacks = [hr]
        for j in range(n_mem):
            k_mem = f"MEM_{j}_{d}"
            if k_mem in data:
                stacks.append(data[k_mem])
        if has_pmm and (f"PMM_{d}" in data):
            stacks.append(data[f"PMM_{d}"])

        vals = np.concatenate([a[~np.isnan(a)].ravel() for a in stacks if a is not None]) if stacks else np.array([1.0])
        vmax = float(np.percentile(vals, 99.5)) if vals.size else 1.0
        vmax = max(vmax, 1.0)
        vmin = 0.0

        # --- Row 0: HR with bold CRPS in the title
        ax_hr = axes[0, t]
        im = ax_hr.imshow(hr, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
        overlay_outline(ax_hr, dk_mask, color="black", linewidth=0.6)        
        crps_val = data.get(f"CRPS_ENS_{d}", np.nan)
        ax_hr.set_title(f"{d}\nCRPS={crps_val:.3f}", fontweight="bold")
        ax_hr.set_xticks([]); ax_hr.set_yticks([])
        if t == 0:
            ax_hr.set_ylabel("HR")

        # Attach a slim colorbar to the HR axis (per column)
        divider = make_axes_locatable(ax_hr)
        cax = divider.append_axes("right", size="4%", pad=0.05)
        fig.colorbar(im, cax=cax, label="mm/day")

        # --- Member rows with per-member MAE
        for j in range(n_mem):
            ax = axes[1 + j, t]
            kf = f"MEM_{j}_{d}"
            if kf in data:
                mem = data[kf]
                ax.imshow(mem, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
                overlay_outline(ax, dk_mask, color="black", linewidth=0.6)                
                mae = data.get(f"MAE_MEM_{j}_{d}", np.nan)
                ax.set_title(f"m{j}  |  MAE={mae:.3f}")
            else:
                ax.text(0.5, 0.5, "—", ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([])
            if t == 0:
                ax.set_ylabel("Ens member")

        # --- Last row: PMM (if present)
        if has_pmm:
            ax_pmm = axes[-1, t]
            if f"PMM_{d}" in data:
                pmm = data[f"PMM_{d}"]
                ax_pmm.imshow(pmm, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
                overlay_outline(ax_pmm, dk_mask, color="black", linewidth=0.6)                
                ax_pmm.set_title("PMM")
            else:
                ax_pmm.text(0.5, 0.5, "—", ha="center", va="center")
            ax_pmm.set_xticks([]); ax_pmm.set_yticks([])
            if t == 0:
                ax_pmm.set_ylabel("PMM")

    fig.suptitle("CRPS examples: HR (title shows ensemble CRPS), ensemble members (titles show MAE), PMM (bottom row)", y=0.995)
    _savefig(fig, figs_dir / "prob_crps_examples.png", dpi=SET_DPI)


def plot_crps_timeseries(
        eval_root: str | Path,
):
    eval_root = Path(eval_root)
    tables_dir = eval_root / "tables"
    figs_dir = _ensure_dir(eval_root / "figures")

    crps_path = tables_dir / "prob_crps_daily.csv"
    if not crps_path.exists():
        logger.info(f"CRPS timeseries file not found: {crps_path}")
        return

    lines = crps_path.read_text().strip().splitlines()
    if len(lines) <= 1:
        return

    dates_dt: list[datetime] = []
    dates_raw: list[str] = []
    crps_vals: list[float] = []
    seasons: list[str] = []

    for ln in lines[1:]:
        s = ln.split(",")
        if len(s) < 2:
            continue
        d_raw = s[0].strip()
        d_dt = _to_date_safe(d_raw)
        try:
            v = float(s[1])
        except Exception:
            continue
        if d_dt is None:
            continue
        dates_dt.append(d_dt)
        dates_raw.append(d_raw)
        crps_vals.append(v)
        seasons.append(_season_from_month(d_dt.month))

    if not crps_vals:
        return

    # sort by date
    order = np.argsort(np.array(dates_dt, dtype="datetime64[s]"))
    dates_dt = [dates_dt[i] for i in order]
    dates_raw = [dates_raw[i] for i in order]
    crps_vals = [crps_vals[i] for i in order]
    seasons = [seasons[i] for i in order]

    # --- figure 1: daily timeseries ---
    _nice()
    fig, ax = plt.subplots()
    x = np.arange(len(crps_vals))
    ax.plot(x, crps_vals, "-", lw=0.9, label="daily CRPS", color=col_gen)

    # 7d smoother
    if len(crps_vals) >= 7:
        window = 7
        sm = np.convolve(crps_vals, np.ones(window)/window, mode="valid")
        ax.plot(np.arange(window-1, window-1+len(sm)), sm, lw=1.5, label="7d mean")

    # horizontal mean line
    mean_crps = float(np.mean(crps_vals))
    ax.axhline(mean_crps, color="0.5", linestyle="--", linewidth=1.0,
               label=f"mean={mean_crps:.2f}")

    # sparse date labels (max 10)
    n_labels = min(10, len(dates_raw))
    idxs = np.linspace(0, len(dates_raw) - 1, n_labels, dtype=int)
    ax.set_xticks(idxs)
    ax.set_xticklabels([dates_raw[i] for i in idxs], rotation=30, ha="right")

    ax.set_xlabel("time (sorted by date)")
    ax.set_ylabel("CRPS")
    ax.set_title("Daily CRPS")
    ax.legend()
    _savefig(fig, figs_dir / "prob_crps_timeseries.png", dpi=SET_DPI)

    # --- figure 2: seasonal + ALL boxplot ---
    by_season: Dict[str, list[float]] = {"DJF": [], "MAM": [], "JJA": [], "SON": []}
    for v, s in zip(crps_vals, seasons):
        by_season[s].append(v)

    _nice()
    fig, ax = plt.subplots()
    data_all = crps_vals
    data = [data_all,
            by_season["DJF"],
            by_season["MAM"],
            by_season["JJA"],
            by_season["SON"]]
    labels = ["ALL", "DJF", "MAM", "JJA", "SON"]
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylabel("CRPS")
    ax.set_title("Seasonal CRPS distribution")

    # # tiny legend explaining the green mean symbol
    # legend_elems = [
    #     Patch(facecolor="white", edgecolor="black", label="IQR + whiskers"),
    #     Patch(facecolor="white", edgecolor="none", label="● mean (green)")
    # ]
    # ax.legend(handles=legend_elems, loc="upper right")

    _savefig(fig, figs_dir / "prob_crps_seasonal.png", dpi=SET_DPI)

def plot_crps_spatial(eval_root: str | Path):
    eval_root = Path(eval_root)
    tables_dir = eval_root / "tables"
    figs_dir = _ensure_dir(eval_root / "figures")
    fpath = tables_dir / "prob_crps_mean_map.npz"
    if not fpath.exists():
        return
    data = np.load(fpath)
    crps_map = data["crps_mean_map"]
    mean_crps = float(np.nanmean(crps_map))

    _nice()
    cmap = get_cmap_for_variable("prcp")
    fig, ax = plt.subplots()
    im = ax.imshow(crps_map, origin="lower", cmap=cmap)
    dk_mask = get_dk_lsm_outline()
    if dk_mask is not None:
        dk_mask = np.flipud(dk_mask)
    overlay_outline(ax, dk_mask, color="black", linewidth=0.6)

    # Try to compute overall std from the *daily* domain-mean CRPS values
    crps_daily_csv = tables_dir / "prob_crps_daily.csv"
    overall_std = np.nan
    try:
        if crps_daily_csv.exists():
            lines = crps_daily_csv.read_text().strip().splitlines()
            vals = []
            for ln in lines[1:]:
                parts = ln.split(",")
                if len(parts) >= 2:
                    try:
                        vals.append(float(parts[1]))
                    except Exception:
                        pass
            if len(vals) > 0:
                overall_std = float(np.nanstd(np.asarray(vals, dtype=float), ddof=0))
    except Exception:
        pass

    title = f"Mean CRPS (per pixel, over time)\nOverall mean: {mean_crps:.2f}"
    if np.isfinite(overall_std):
        title = f"Mean CRPS (per pixel, over time)\nOverall mean ± std: {mean_crps:.2f} ± {overall_std:.2f}"
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, label="CRPS")
    _savefig(fig, figs_dir / "prob_crps_mean_map.png", dpi=SET_DPI)



# ================================================================================
# Master plotting function
# ================================================================================

def plot_probabilistic(
        eval_root: str | Path,
        *,
        gen_root: Optional[str | Path] = None,
        thresholds: Sequence[float] = (1.0, 5.0, 10.0),
        pit_bins: int = 20,
):
    """
        One master function to plot all probabilistic evaluation plots.
        To be called from evaluate_probabilistic.py after all tables have been written.
    """
    plot_pit(eval_root, bins=pit_bins)
    plot_rank(eval_root)
    plot_reliability(eval_root, thresholds=thresholds)
    plot_spread_skill(eval_root)
    plot_energy_variogram(eval_root)
    plot_crps_examples(eval_root, gen_root=gen_root)
    plot_crps_timeseries(eval_root)
    plot_crps_spatial(eval_root)