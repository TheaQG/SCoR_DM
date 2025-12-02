# sbgm/evaluate/evaluate_prcp/eval_distributional/plot_distributional.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from evaluate.evaluate_prcp.plot_utils import _ensure_dir, _nice, _savefig
from scor_dm.variable_utils import get_color_for_model

# Baseline overlays
from evaluate.evaluate_prcp.overlay_utils import resolve_baseline_dirs, load_csv_if_exists

# Helper to sanitize arrays for plotting (for log plots)
def _finite(arr: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """Replace non-finite values with small positive floor (for log plots)."""
    return np.where(np.isfinite(arr), arr, floor)

logger = logging.getLogger(__name__)



SET_DPI = 300

# Z-order scheme so ensemble is on top, then HR, PMM, LR, then baselines
ZORDER_ENS = 25
ZORDER_HR = 20
ZORDER_PMM = 18
ZORDER_LR = 10
ZORDER_BASELINE = 5

# Default visualization toggles (can be overridden from eval_cfg)
DEFAULT_SHOW_CI = False    # Turn CI bands off by default
DEFAULT_SHOW_INSET = True  # Show tail-zoom inset by default
Y_FLOOR = 1e-7             # Log-axis floor to avoid collapsing


def plot_distributional(dist_root: str | Path, eval_cfg: Any | None = None) -> None:
    dist_root = Path(dist_root)
    tables = dist_root / "tables"
    figs = _ensure_dir(dist_root / "figures")

    # Set colors
    col_hr = get_color_for_model("hr")
    col_pmm = get_color_for_model("pmm")
    col_ens = get_color_for_model("ensemble")
    col_lr  = get_color_for_model("lr")

    bins_path = tables / "dist_bins.csv"
    if not bins_path.exists():
        logger.warning("[plot_distributional] No dist_bins.csv – skipping.")
        return
    bins = np.loadtxt(bins_path, delimiter=",", skiprows=1) if bins_path.read_text().startswith("bin_edge") else np.loadtxt(bins_path, delimiter=",")
    # If 1D
    if bins.ndim > 1:
        bins = bins[:, 0]

    def _read_hist(name: str) -> Optional[np.ndarray]:
        p = tables / f"dist_{name}.csv"
        if not p.exists():
            return None
        xs, cs = [], []
        with open(p, "r") as f:
            next(f)  # header
            for ln in f:
                s = ln.strip().split(",")
                if len(s) != 2:
                    continue
                xs.append(int(s[0])); cs.append(int(float(s[1])))
        return np.array(cs, dtype=float)

    hr = _read_hist("hr")
    gen = _read_hist("gen")
    lr  = _read_hist("lr")

    # Try to read ensemble artifacts
    ens_mode = None
    gen_ens_pool = None
    gen_ens_mean = None
    ens_npz = tables / "dist_member_histograms.npz"
    if (tables / "dist_gen_ens_pool.csv").exists():
        xs, cs = [], []
        with open(tables / "dist_gen_ens_pool.csv", "r") as f:
            next(f)
            for ln in f:
                s = ln.strip().split(",")
                if len(s) == 2:
                    xs.append(int(s[0])); cs.append(float(s[1]))
        gen_ens_pool = np.array(cs, dtype=float)
        ens_mode = "pool"
    if (tables / "dist_gen_ens_mean.csv").exists():
        xs, ps = [], []
        with open(tables / "dist_gen_ens_mean.csv", "r") as f:
            next(f)
            for ln in f:
                s = ln.strip().split(",")
                if len(s) == 2:
                    xs.append(int(s[0])); ps.append(float(s[1]))
        gen_ens_mean = np.array(ps, dtype=float)
        ens_mode = "member_mean"
    q10 = q50 = q90 = None
    if ens_npz.exists():
        try:
            d = np.load(ens_npz)
            q10 = d.get("pdf_q10", None)
            q50 = d.get("pdf_q50", None)
            q90 = d.get("pdf_q90", None)
            if ens_mode is None and "mode" in d:
                try:
                    ens_mode = str(d["mode"])  # may be 0-d array
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"[plot_distributional] Could not load ensemble NPZ: {e}")

    metrics_path = tables / "dist_metrics.csv"
    gen_text = None
    lr_text = None
    if metrics_path.exists():
        lines = metrics_path.read_text().strip().splitlines()
        # header: ref,comp,wasserstein,ks_stat,ks_p,kl_hr_to_x
        rows = []
        for ln in lines[1:]:
            ref, comp, w1, ks_s, ks_p, kl = ln.split(",")
            rows.append((ref, comp, float(w1), float(ks_s), float(ks_p), float(kl)))
        # Separate GEN and LR metrics
        gen_parts = []
        lr_parts = []
        for (ref, comp, w1, kss, ksp, kl) in rows:
            if comp.lower() in ("gen_ens_pool", "gen_ens_mean", "gen_pmm"):
                txt = (
                    f"{comp.upper()} vs {ref.upper()}:\n"
                    f"  W1  = {w1:.3f}\n"
                    f"  KS  = {kss:.3f} (p={ksp:.2f})\n"
                    f"  KL  = {kl:.3f}"
                )
                gen_parts.append(txt)
        gen_text = "\n".join(gen_parts).strip() if gen_parts else None

        for (ref, comp, w1, kss, ksp, kl) in rows:
            if comp.lower() == "lr":
                txt = (
                    f"{comp.upper()} vs {ref.upper()}:\n"
                    f"  W1  = {w1:.3f}\n"
                    f"  KS  = {kss:.3f} (p={ksp:.2f})\n"
                    f"  KL  = {kl:.3f}"
                )
                lr_parts.append(txt)
        lr_text = "\n".join(lr_parts).strip() if lr_parts else None

    # Plot
    _nice()
    fig, ax = plt.subplots(figsize=(7,5.5))

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    eps = 1e-12
    y_floor = Y_FLOOR  # use module-level constant
    # normalize to PDF shape (so the area is comparable)
    def _norm(h: np.ndarray | None) -> np.ndarray | None:
        if h is None:
            return None
        s = h.sum()
        if s <= 0:
            return h
        return h / s

    hr_n = _norm(hr)
    gen_n = _norm(gen)
    lr_n  = _norm(lr)

    # Optional toggles from eval config
    show_ci = DEFAULT_SHOW_CI
    show_inset = DEFAULT_SHOW_INSET
    if eval_cfg is not None:
        try:
            show_ci = bool(getattr(eval_cfg, "dist_show_ci", DEFAULT_SHOW_CI))
            show_inset = bool(getattr(eval_cfg, "dist_show_inset", DEFAULT_SHOW_INSET))
        except Exception:
            pass

    # Optional daily uncertainty bands
    daily_npz = tables / "dist_daily.npz"
    ci = {}
    if daily_npz.exists():
        try:
            npz = np.load(daily_npz)
            def _series_ci(counts_key: str, n_key: str):
                # np.load(...) returns an NpzFile which exposes 'files'; fall back to dict-like keys() if needed
                keys = getattr(npz, "files", None)
                if keys is None:
                    # avoid direct attribute access that static analyzers may confuse with Path
                    keys_attr = getattr(npz, "keys", None)
                    if callable(keys_attr):
                        try:
                            k = keys_attr()
                            # Only convert to list if the returned object is actually iterable
                            try:
                                from collections.abc import Iterable  # local import to avoid unused-top-level import
                                if isinstance(k, Iterable):
                                    keys = list(k)
                                else:
                                    keys = None
                            except Exception:
                                # Fallback: attempt a best-effort conversion
                                try:
                                    # k may be typed as 'object' by static analyzers; silence that specific type error
                                    keys = list(k)  # type: ignore[arg-type]
                                except Exception:
                                    keys = None
                        except Exception:
                            keys = None
                    else:
                        keys = None
                if keys is None or (counts_key not in keys) or (n_key not in keys):
                    return None
                C = npz[counts_key]      # [D,B]
                n = npz[n_key]           # [D]
                if C.size == 0 or n.size == 0:
                    return None
                # avoid division by zero
                n = np.maximum(n.astype(float), 1.0)
                pdf = (C.astype(float).T / n).T  # [D,B]
                lo = np.percentile(pdf, 5, axis=0)
                hi = np.percentile(pdf, 95, axis=0)
                med = np.percentile(pdf, 50, axis=0)
                return lo, hi, med
            ci["hr"]  = _series_ci("counts_hr",  "n_hr")
            ci["gen"] = _series_ci("counts_gen", "n_gen")
            if "counts_lr" in npz and "n_lr" in npz:
                ci["lr"] = _series_ci("counts_lr", "n_lr")
        except Exception as e:
            logger.warning(f"[plot_distributional] Failed to parse dist_daily.npz for CI shading: {e}")
    # Plot CI bands (shading) before lines, only if show_ci is True
    if show_ci:
        val = ci.get("hr")
        if val is not None:
            lo, hi, _ = val
            ax.fill_between(
                bin_centers,
                np.maximum(_finite(lo, eps), y_floor).tolist(),
                np.maximum(_finite(hi, eps), y_floor).tolist(),
                color=col_hr, alpha=0.10, linewidth=0
            )
        val = ci.get("gen")
        if val is not None:
            lo, hi, _ = val
            ax.fill_between(
                bin_centers,
                np.maximum(_finite(lo, eps), y_floor).tolist(),
                np.maximum(_finite(hi, eps), y_floor).tolist(),
                color=col_pmm, alpha=0.08, linewidth=0
            )
        val = ci.get("lr")
        if val is not None and lr_n is not None:
            lo, hi, _ = val
            ax.fill_between(
                bin_centers,
                np.maximum(_finite(lo, eps), y_floor).tolist(),
                np.maximum(_finite(hi, eps), y_floor).tolist(),
                color=col_lr, alpha=0.07, linewidth=0
            )

    # Helper: percentile from histogram
    def _percentile_from_hist(counts: np.ndarray | None, bins_arr: np.ndarray, p: float) -> Optional[float]:
        if counts is None or counts.size == 0:
            return None
        c = np.cumsum(counts.astype(float))
        c /= max(c[-1], 1.0)
        idx = np.searchsorted(c, p)
        idx = int(np.clip(idx, 0, len(bins_arr)-2))
        return float(0.5 * (bins_arr[idx] + bins_arr[idx+1]))

    # Base curves: LR in the back, then HR, then PMM; ensemble will sit on top via ZORDER_ENS
    if lr_n is not None:
        ax.plot(
            bin_centers,
            lr_n,
            color=col_lr,
            lw=1.0,
            ls="--",
            label="LR",
            zorder=ZORDER_LR,
        )
    if hr_n is not None:
        ax.plot(
            bin_centers,
            hr_n,
            color=col_hr,
            lw=1.5,
            label="HR",
            zorder=ZORDER_HR,
        )
    if gen_n is not None:
        ax.plot(
            bin_centers,
            gen_n,
            color=col_pmm,
            ls='-.',
            lw=1.2,
            label="PMM",
            zorder=ZORDER_PMM,
        )

    # Build reusable ensemble curve (pool preferred, else member mean)
    ens_curve = None
    if gen_ens_pool is not None:
        s = float(np.sum(gen_ens_pool))
        if s > 0:
            ens_curve = gen_ens_pool / s
    elif gen_ens_mean is not None:
        ens_curve = np.asarray(gen_ens_mean, dtype=float)

    # Reference wet-day threshold
    wet_thr = float(getattr(eval_cfg, "wet_threshold_mm", 1.0)) if eval_cfg is not None else 1.0
    # ax.axvline(wet_thr, ls=":", lw=0.8, color="0.3", alpha=0.6)

    # HR percentiles from histogram
    p95 = _percentile_from_hist(hr, bins, 0.95)
    p99 = _percentile_from_hist(hr, bins, 0.99)
    if p95 is not None:
        ax.axvline(p95, color="0.2", lw=0.8, ls="--", alpha=0.6)
        ax.text(p95, ax.get_ylim()[1]*0.6, "P95", rotation=90, va="top", ha="right", fontsize=8, color="0.25")
    if p99 is not None:
        ax.axvline(p99, color="0.2", lw=0.8, ls="--", alpha=0.6)
        ax.text(p99, ax.get_ylim()[1]*0.6, "P99", rotation=90, va="top", ha="right", fontsize=8, color="0.25")

    # Tail inset (fixed window 20–80 mm/day), optional and placed fully outside the axes
    try:
        if show_inset:
            x_min = 20.0
            x_max = min(80.0, float(bins.max()))
            if x_max > x_min:
                ax_ins = inset_axes(
                    ax,
                    width="36%", height="56%",
                    loc="upper right",
                    bbox_to_anchor=(-0.22, 1.0),   # outside, to the LEFT of the axes to avoid legend
                    bbox_transform=ax.transAxes,
                    borderpad=0.0
                )
                if hr_n is not None:
                    ax_ins.plot(bin_centers, hr_n, color=col_hr, lw=1.2)
                if gen_n is not None:
                    ax_ins.plot(bin_centers, gen_n, color=col_pmm, lw=1.0)
                if lr_n is not None:
                    ax_ins.plot(bin_centers, lr_n, color=col_lr, lw=0.9, ls="--")
                if ens_curve is not None:
                    ax_ins.plot(bin_centers, np.maximum(ens_curve, eps), lw=1.0, color=col_ens)
                ax_ins.set_xlim(x_min, x_max)
                ax_ins.set_yscale("log")
                ax_ins.set_ylim(max(ax.get_ylim()[0], y_floor), ax.get_ylim()[1])
                ax_ins.tick_params(labelsize=7)
                ax_ins.grid(True, ls=":", alpha=0.3)
    except Exception as e:
        logger.info(f"[plot_distributional] Tail inset skipped: {e}")

    # Plot ensemble curve(s)
    if ens_curve is not None:
        ax.plot(
            bin_centers,
            np.maximum(ens_curve, eps),
            lw=1.6,
            label="GEN (ensemble)",
            color=col_ens,
            zorder=ZORDER_ENS,
        )
    if q10 is not None and q90 is not None:
        ax.fill_between(bin_centers, np.maximum(q10, eps).tolist(), np.maximum(q90, eps).tolist(), alpha=0.10, linewidth=0, label="Ens spread (10–90%)")

    # === Baseline overlays ===
    bo = getattr(eval_cfg, "baselines_overlay", None) if eval_cfg is not None else None
    if bo:
        try:
            dirs = resolve_baseline_dirs(
                sample_root=bo["sample_root"],
                types=tuple(bo.get("types", ())),
                split=str(bo.get("split", "test")),
                eval_type="distributional"
            )
        except Exception as e:
            logger.warning(f"[plot_distributional] Failed to resolve baseline dirs: {e}")
            dirs = {}
        for t, d in dirs.items():
            try:
                # Try to read baseline's dist_bins.csv
                bins_arr = None
                bins_path = d / "dist_bins.csv"
                if bins_path.exists():
                    bins_arr = np.loadtxt(bins_path, delimiter=",", skiprows=1) if bins_path.read_text().startswith("bin_edge") else np.loadtxt(bins_path, delimiter=",")
                    if bins_arr.ndim > 1:
                        bins_arr = bins_arr[:, 0]
                else:
                    logger.info(f"[plot_distributional] Baseline {t}: missing dist_bins.csv at {bins_path}")
                    continue
                bin_centers_b = 0.5 * (bins_arr[:-1] + bins_arr[1:])
                # Try dist_gen.csv first, else dist_lr.csv
                arr = load_csv_if_exists(d, "dist_gen")
                if arr is None:
                    arr = load_csv_if_exists(d, "dist_lr")
                    if arr is None:
                        logger.info(f"[plot_distributional] Baseline {t}: missing both dist_gen.csv and dist_lr.csv in {d}")
                        continue
                # arr: structured array with fields 'bin_idx' and 'count'
                try:
                    counts = np.asarray(arr["count"], dtype=float)
                except Exception:
                    # fallback for 2-column shape
                    counts = np.asarray(arr[:, 1], dtype=float)
                pdf = counts / (np.sum(counts) + eps)
                label = bo.get("labels", {}).get(t, t)
                # Start from configured style (if any) and enforce a sensible default z-order
                style = dict(bo.get("styles", {}).get(t, {}))
                style.setdefault("zorder", ZORDER_BASELINE)
                ax.plot(bin_centers_b, pdf, label=label, **style)
            except Exception as e:
                logger.info(f"[plot_distributional] Baseline overlay for {t} failed: {e}")
                continue


    ax.set_xlabel("Precipitation (mm/day)")
    ax.set_yscale("log")
    ax.set_ylabel("Probability")
    # Title + subtitle with pooling info
    ens_label = " (ensemble)" if (gen_ens_pool is not None or gen_ens_mean is not None) else ""
    ax.set_title("Pooled pixel distributions", fontsize=15, pad=10)
    # Compact annotation for sample size (top-right, inside axes, away from title)
    Npix = None
    if daily_npz.exists():
        try:
            d = np.load(daily_npz)
            if "n_hr" in d:
                Npix = int(np.sum(d["n_hr"]))
        except Exception:
            Npix = None
    # if Npix is not None:
    #     ax.text(0.98, 0.98, f"N ≈ {Npix:,}", transform=ax.transAxes,
    #             ha="right", va="top", fontsize=9, color="0.35",
    #             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.8))

    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(fontsize=9)
    # Make main y-axis respect the new floor
    ax.set_ylim(bottom=max(ax.get_ylim()[0], y_floor))    

    # Place GEN vs HR metrics (top-left) and LR vs HR (bottom-right) to avoid overlap
    boxprops = dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.85)
    if gen_text:
        ax.text(
            0.02, 0.4, gen_text, transform=ax.transAxes,
            va="top", ha="left", fontsize=8,
            bbox=boxprops,
        )

    if lr_text:
        ax.text(
            0.02, 0.02, lr_text, transform=ax.transAxes,
            va="bottom", ha="left", fontsize=8,
            bbox=boxprops,
        )


    # === Seasonal distributions figure ===
    try:
        if daily_npz.exists():
            d = np.load(daily_npz)
            bins_s = d["bins"] if "bins" in d else bins
            mids_s = 0.5 * (bins_s[:-1] + bins_s[1:])
            dates_s = d["dates"].astype(str) if "dates" in d else np.array([], dtype=str)
            # Helper: map yyyymmdd -> season label
            def _season(yyyymmdd: str) -> str:
                try:
                    m = datetime.strptime(yyyymmdd, "%Y%m%d").month
                except Exception:
                    return "UNK"
                if m in (12,1,2): return "DJF"
                if m in (3,4,5):  return "MAM"
                if m in (6,7,8):  return "JJA"
                return "SON"
            idxs = {k: [] for k in ("DJF","MAM","JJA","SON")}
            for i, ds in enumerate(dates_s):
                s = _season(ds)
                if s in idxs:
                    idxs[s].append(i)
            # Seasonal facecolors
            season_face = {
                "DJF": (0.35, 0.55, 0.85, 0.20),  # stronger light blue
                "MAM": (0.60, 0.80, 0.60, 0.20),  # stronger light green
                "JJA": (0.95, 0.85, 0.40, 0.22),  # stronger light yellow
                "SON": (0.95, 0.70, 0.60, 0.20),  # stronger light red/orange
            }
            figS, axs = plt.subplots(2, 2, figsize=(8,7), sharey=True)
            for axS, (lab, ids) in zip(axs.flat, idxs.items()):
                if not ids:
                    axS.set_title(f"{lab} (no data)"); axS.axis('off'); continue
                # light seasonal background
                if lab in season_face:
                    axS.set_facecolor(season_face[lab])
                ids = np.asarray(ids, dtype=int)
                # HR
                C_hr = d["counts_hr"][ids].sum(axis=0) if "counts_hr" in d else None
                C_gen = d["counts_gen"][ids].sum(axis=0) if "counts_gen" in d else None
                C_lr = d["counts_lr"][ids].sum(axis=0) if ("counts_lr" in d) else None

                def _norm_counts(C):
                    if C is None:
                        return None
                    s = float(np.sum(C))
                    return (C / s) if s > 0 else None
                
                pdf_hr = _norm_counts(C_hr)
                pdf_gen = _norm_counts(C_gen)
                pdf_lr  = _norm_counts(C_lr)
                
                # Prefer ensemble curve over PMM if available
                pdf_gen_pref = ens_curve if ens_curve is not None else pdf_gen

                if pdf_hr is not None:
                    axS.plot(mids_s, np.maximum(pdf_hr, eps), color=col_hr, lw=1.2, label="HR")
                if pdf_gen_pref is not None:
                    axS.plot(
                        mids_s,
                        np.maximum(pdf_gen_pref, eps),
                        color=(col_ens if ens_curve is not None else col_pmm),
                        lw=1.1,
                        label=("Gen (ensemble)" if ens_curve is not None else "PMM"),
                    )
                if pdf_lr is not None:
                    axS.plot(mids_s, np.maximum(pdf_lr, eps), color=col_lr, lw=1.1, ls="--", label="LR")

                # Reference wet-day line
                axS.axvline(wet_thr, ls=":", lw=0.6, color="0.4", alpha=0.6)

                def _p_from_C(C, p):
                    if C is None:
                        return None
                    c = np.cumsum(C.astype(float))
                    c /= max(c[-1], 1.0)
                    idx = int(np.clip(np.searchsorted(c, p), 0, len(bins_s) - 2))
                    return float(0.5 * (bins_s[idx] + bins_s[idx + 1]))

                p95s = _p_from_C(C_hr, 0.95)
                p99s = _p_from_C(C_hr, 0.99)

                axS.set_title(lab)
                axS.set_yscale("log")
                axS.set_ylim(bottom=y_floor)
                axS.set_xlabel("Precipitation (mm/day)")
                axS.grid(True, ls=":", alpha=0.4)

                # Add P95/P99 verticals and labels after y-limits are fixed
                ylim_s = axS.get_ylim()
                if p95s is not None:
                    axS.axvline(p95s, color="0.2", lw=0.6, ls="--", alpha=0.5)
                    axS.text(
                        p95s,
                        ylim_s[1] * 0.6,
                        "P95",
                        rotation=90,
                        va="top",
                        ha="right",
                        fontsize=7,
                        color="0.25",
                    )
                if p99s is not None:
                    axS.axvline(p99s, color="0.2", lw=0.6, ls="--", alpha=0.5)
                    axS.text(
                        p99s,
                        ylim_s[1] * 0.6,
                        "P99",
                        rotation=90,
                        va="top",
                        ha="right",
                        fontsize=7,
                        color="0.25",
                    )
            # Combined legend across panels (unique labels)
            handles, labels = [], []
            for _ax in axs.flat:
                h, l = _ax.get_legend_handles_labels()
                for hh, ll in zip(h, l):
                    if ll not in labels:
                        handles.append(hh)
                        labels.append(ll)

            if handles:
                # Place legend slightly below the subplots
                figS.legend(
                    handles,
                    labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.02),
                    ncol=len(labels),
                    fontsize=10,
                )

            figS.tight_layout(rect=(0.03, 0.05, 0.97, 0.98))
            _savefig(figS, figs / "dist_pooled_seasons.png", dpi=SET_DPI)
    except Exception as e:
        logger.info(f"[plot_distributional] Seasonal figure skipped: {e}")

    fig.tight_layout(rect=(0, 0, 0.88, 1.0))  # leave space on the right for the inset
    _savefig(fig, figs / "dist_pooled.png", dpi=SET_DPI)