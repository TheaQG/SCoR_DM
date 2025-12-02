from __future__ import annotations
from pathlib import Path
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, Optional
from evaluate.evaluate_prcp.overlay_utils import resolve_baseline_dirs

logger = logging.getLogger(__name__)

from evaluate.evaluate_prcp.plot_utils import (_nice, _savefig, _ensure_dir)
from scor_dm.variable_utils import get_color_for_model

SERIES_ORDER  = ["HR", "GEN_ENS", "GEN", "LR"]
SERIES_LABELS = {"HR": "HR (DANRA)", "GEN": "PMM", "GEN_ENS": "Generated (ens)", "LR": "LR upsampled"}
col_hr = get_color_for_model("HR")
col_pmm = get_color_for_model("pmm")
col_ens = get_color_for_model("ensemble")
col_lr = get_color_for_model("LR")
SERIES_COLORS = {"HR": col_hr, "GEN": col_pmm, "GEN_ENS": col_ens, "LR": col_lr}

SET_DPI = 300

def _get_bo(eval_cfg: Any | None) -> Optional[Dict[str, Any]]:
    if eval_cfg is None:
        return None
    bo = getattr(eval_cfg, "baselines_overlay", None)
    if not bo or not bo.get("enabled", False):
        return None
    return bo

# Normalize any slightly different labels coming from CSVs
def _norm_series_name(s: str) -> str:
    t = s.strip().upper()
    if t.startswith("HR"):
        return "HR"
    if t.startswith("GEN_ENS") or "GEN/ENS" in t or "ENS" in t:
        return "GEN_ENS"    
    if t.startswith("GEN") or "GENERATED" in t:
        return "GEN"
    if t.startswith("LR"):
        return "LR"
    if "DANRA" in t:
        return "HR"
    if "UPSAMPLED" in t:
        return "LR"
    return s.strip()

def _load_meta(tables_dir: Path) -> dict:
    meta_path = tables_dir / "ext_meta.npz"
    if meta_path.exists():
        try:
            d = np.load(meta_path, allow_pickle=True)
            out = {}
            for k in d.files:
                v = d[k]
                # unwrap 0-d arrays to Python scalars/strings
                if isinstance(v, np.ndarray) and v.shape == ():
                    v = v.item()
                out[k] = v
            return out
        except Exception as e:
            logger.warning(f"[_load_meta] Failed to read {meta_path}: {e}")
    return {}
    
def _read_csv(path: Path):
    if not path.exists():
        return [], []
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    if not lines:
        return [], []
    header = lines[0].split(",")
    rows = [l.split(",") for l in lines[1:]]
    # normalize series label in col 0
    for r in rows:
        if r:
            r[0] = _norm_series_name(r[0])
    return header, rows


def plot_return_levels(gev_csv: Path, out_png: Path, bo: Optional[Dict[str, Any]] = None):
    hdr, rows = _read_csv(gev_csv)
    if not rows:
        return

    rps = [int(c[3:-1]) for c in hdr if c.startswith("rl_") and c.endswith("y")]
    ks  = sorted({int(r[1]) for r in rows})
    _nice()
    fig, axs = plt.subplots(1, len(ks), figsize=(4.6*len(ks), 3.6), sharey=True)
    if len(ks) == 1:
        axs = axs.flatten()

    # Annotate with aggregation mode and blocks/year if available
    meta = _load_meta(gev_csv.parent)
    agg_kind = str(meta.get("agg_kind", "mean"))
    bpy = meta.get("blocks_per_year", None)
    if bpy is not None:
        fig.text(0.99, 0.01, f"Aggregation: {agg_kind}   •   Blocks/year = {float(bpy):.2f}",
                 ha="right", va="bottom", fontsize=9, color="0.3")

    for ax, k in zip(axs, ks):
        present = {r[0] for r in rows if int(r[1]) == k}
        for which in SERIES_ORDER:
            if which not in present:
                continue
            col = SERIES_COLORS[which]
            ls = ":" if which == "GEN_ENS" else "-"            
            r = [rr for rr in rows if rr[0]==which and int(rr[1])==k][0]
            nb = int(r[2])
            rl  = [float(x) for x in r[3:3+len(rps)]]
            lo  = [float(x) for x in r[3+len(rps):3+2*len(rps)]]
            hi  = [float(x) for x in r[3+2*len(rps):3+3*len(rps)]]
            # thin line + faint fill
            ax.plot(rps, rl, marker="o", ms=3, lw=1.5, color=col,
                    label=f"{SERIES_LABELS[which]} (n={nb})", ls=ls)
            ax.fill_between(rps, lo, hi, alpha=0.18, color=col, linewidth=0)
        ax.set_xscale("log")
        ax.set_xlabel("Return period (years)")
        ax.set_ylabel("Return level (mm)")
        ax.set_title(f"Rx{k}day – GEV")
        ax.grid(True, ls=":")
        ax.legend(fontsize=8, loc="best")

    # ---- Baseline overlays ----
    if bo:
        try:
            dirs = resolve_baseline_dirs(
                sample_root=bo["sample_root"],
                types=tuple(bo.get("types", ())),
                split=str(bo.get("split", "test")),
                eval_type="extremes",
            )
        except Exception as e:
            logger.warning(f"[plot_return_levels] resolve_baseline_dirs failed: {e}")
            dirs = {}
        labels = bo.get("labels", {})
        styles = bo.get("styles", {})
        for t, d in dirs.items():
            b_hdr, b_rows = _read_csv(d / "ext_rxk_gev.csv")
            if not b_rows:
                continue
            # prefer GEN rows if present, else LR
            has_gen = any(_norm_series_name(r[0]) == "GEN" for r in b_rows)
            sel = "GEN" if has_gen else ("LR" if any(_norm_series_name(r[0]) == "LR" for r in b_rows) else None)
            if sel is None:
                continue
            for ax, k in zip(axs, ks):
                m = [rr for rr in b_rows if _norm_series_name(rr[0]) == sel and int(rr[1]) == k]
                if not m:
                    continue
                r = m[0]
                rl  = [float(x) for x in r[3:3+len(rps)]]
                label = labels.get(t, t)
                style = dict(styles.get(t, {}))
                _kw = {"marker": "o", "ms": 2.5}
                if "lw" not in style and "linewidth" not in style:
                    _kw["lw"] = 1.4
                ax.plot(rps, rl, label=label, **_kw, **style)

    fig.tight_layout()
    _savefig(fig, out_png, dpi=SET_DPI)
    plt.close(fig)


def plot_pot(para_csv: Path, out_png: Path, bo: Optional[Dict[str, Any]] = None):
    hdr, rows = _read_csv(para_csv)
    if not rows:
        return
    rps = [int(c[3:-1]) for c in hdr if c.startswith("rl_") and c.endswith("y")]
    _nice()
    fig, ax = plt.subplots(figsize=(5.8, 3.9))

    # Read meta and annotate POT threshold mode
    meta = _load_meta(para_csv.parent)
    pot_kind = str(meta.get("pot_thr_kind", ""))
    pot_val  = meta.get("pot_thr_val", None)
    pot_u_hr = meta.get("pot_u_hr", None)
    # Compose a short descriptor
    if pot_kind == "hr_quantile" and pot_u_hr is not None:
        pot_desc = f"POT threshold: HR-quantile (u≈{float(pot_u_hr):.2f} mm/day)"
    elif pot_kind == "quantile" and pot_val is not None:
        pot_desc = f"POT threshold: per-series quantile q={float(pot_val):.2f}"
    elif pot_kind == "value" and pot_val is not None:
        pot_desc = f"POT threshold: fixed u={float(pot_val):.2f} mm/day"
    else:
        pot_desc = f"POT threshold: {pot_kind}"

    present = {r[0] for r in rows}
    for which in SERIES_ORDER:
        if which not in present:
            continue
        col = SERIES_COLORS[which]
        ls = ":" if which == "GEN_ENS" else "-"        
        r = [rr for rr in rows if rr[0] == which][0]
        # Columns: which,u,xi,beta,k_exc,lambda_per_day, rl..., rl_lo..., rl_hi...
        u = float(r[1])
        rl  = [float(x) for x in r[6:6+len(rps)]]
        lo  = [float(x) for x in r[6+len(rps):6+2*len(rps)]]
        hi  = [float(x) for x in r[6+2*len(rps):6+3*len(rps)]]
        ax.plot(rps, rl, marker="o", ms=3, lw=1.5, color=col, label=SERIES_LABELS[which], ls=ls)
        ax.fill_between(rps, lo, hi, alpha=0.18, color=col, linewidth=0) # type: ignore
    ax.set_xscale("log")
    ax.set_xlabel("Return period (years)")
    ax.set_ylabel("Return level (mm)")
    ax.set_title("POT / GPD return levels")
    ax.grid(True, ls=":")
    ax.legend(fontsize=8, loc="best")
    # put a small caption below the axes area
    fig.text(0.99, 0.01, pot_desc, ha="right", va="bottom", fontsize=9, color="0.3")    
    # ---- Baseline overlays ----
    if bo:
        try:
            dirs = resolve_baseline_dirs(
                sample_root=bo["sample_root"],
                types=tuple(bo.get("types", ())),
                split=str(bo.get("split", "test")),
                eval_type="extremes",
            )
        except Exception as e:
            logger.warning(f"[plot_pot] resolve_baseline_dirs failed: {e}")
            dirs = {}
        labels = bo.get("labels", {})
        styles = bo.get("styles", {})
        for t, d in dirs.items():
            b_hdr, b_rows = _read_csv(d / "ext_pot_gpd.csv")
            if not b_rows:
                continue
            has_gen = any(_norm_series_name(r[0]) == "GEN" for r in b_rows)
            sel = "GEN" if has_gen else ("LR" if any(_norm_series_name(r[0]) == "LR" for r in b_rows) else None)
            if sel is None:
                continue
            r = [rr for rr in b_rows if _norm_series_name(rr[0]) == sel]
            if not r:
                continue
            r = r[0]
            rl  = [float(x) for x in r[6:6+len(rps)]]
            label = labels.get(t, t)
            style = dict(styles.get(t, {}))
            _kw = {"marker": "o", "ms": 2.5}
            if "lw" not in style and "linewidth" not in style:
                _kw["lw"] = 1.4
            ax.plot(rps, rl, label=label, **_kw, **style)
    fig.tight_layout()
    _savefig(fig, out_png, dpi=SET_DPI)
    plt.close(fig)


def plot_tails(tails_csv: Path, out_png: Path, bo: Optional[Dict[str, Any]] = None):
    hdr, rows = _read_csv(tails_csv)
    if not rows:
        return
    # which,P95,P99,wet_freq,wet_hit_rate,n_days
    tables = tails_csv.parent
    meta = _load_meta(tables)
    wet_thr = meta.get("wet_thr_mm", None)
    tails_basis = str(meta.get("tails_basis", "domain_series"))
    basis_desc = "Tails basis: pooled pixels" if tails_basis.lower().startswith("pooled") else "Tails basis: domain series"

    data = {
        r[0]: {
            "P95": float(r[1]),
            "P99": float(r[2]),
            "wet": float(r[3]),
            "hit": float(r[4]),
        } for r in rows
    }
    series = [s for s in SERIES_ORDER if s in data]

    # Try to load error bands for GEN_ENS if available
    bands = None
    npz_path = tails_csv.parent / "ext_tails_ens_bands.npz"
    if npz_path.exists():
        try:
            bands = np.load(npz_path)
        except Exception as e:
            logger.warning(f"[plot_tails] Could not load ensemble bands: {e}")

    _nice()
    fig, axs = plt.subplots(1, 3, figsize=(12.4, 3.8))
    # --- left: P95/P99 grouped bars ---
    ax = axs[0]
    cats = ["P95", "P99"]
    x = np.arange(len(cats))
    w = 0.22
    for i, which in enumerate(series):
        offs = (i - (len(series)-1)/2) * w
        vals = [data[which]["P95"], data[which]["P99"]]
        yerr = None
        if which == "GEN_ENS" and bands is not None:
            try:
                yerr = [bands.get("P95_std", None), bands.get("P99_std", None)]
                yerr = np.array([float(yerr[0]) if yerr[0] is not None else 0.0,
                                 float(yerr[1]) if yerr[1] is not None else 0.0])
            except Exception:
                yerr = None        
        ax.bar(x + offs, vals, width=w,
               label=SERIES_LABELS[which],
               color=SERIES_COLORS[which],
               edgecolor="black", linewidth=0.7, alpha=0.8,
               yerr=yerr if yerr is not None else None,
               capsize=2)
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_title(f"Distribution tails ({basis_desc.split(':')[-1].strip()})")
    ax.set_ylabel("mm/day")
    ax.grid(True, ls=":")
    ax.legend(fontsize=8, loc="best")

    # --- middle: wet-day frequency grouped bars ---
    ax = axs[1]
    x = np.arange(1)  # single category
    for i, which in enumerate(series):
        offs = (i - (len(series)-1)/2) * w
        yerr = None
        if which == "GEN_ENS" and bands is not None:
            try:
                y = float(bands.get("wet_freq_std", 0.0))
                yerr = np.array([y])
            except Exception:
                yerr = None        
        ax.bar(x + offs, [data[which]["wet"]], width=w,
               label=SERIES_LABELS[which],
               color=SERIES_COLORS[which],
               edgecolor="black", linewidth=0.7, alpha=0.8,
               yerr=yerr if yerr is not None else None,
               capsize=2)
    ax.set_xticks([0]); ax.set_xticklabels(["≥ threshold"])
    ax.set_ylim(0, 1)
    ax.set_title("Wet-day frequency")
    ax.set_ylabel("fraction")
    ax.grid(True, ls=":")

    # --- right: wet-hit rate grouped bars ---
    ax = axs[2]
    for i, which in enumerate(series):
        offs = (i - (len(series)-1)/2) * w
        yerr = None
        if which == "GEN_ENS" and bands is not None:
            try:
                y = float(bands.get("hit_std", 0.0))
                yerr = np.array([y])
            except Exception:
                yerr = None        
        ax.bar([0 + offs], [data[which]["hit"]], width=w,
               label=SERIES_LABELS[which],
               color=SERIES_COLORS[which],
               edgecolor="black", linewidth=0.7, alpha=0.8,
               yerr=yerr if yerr is not None else None,
               capsize=2)
    ax.set_xticks([0]); ax.set_xticklabels(["HR-wet days predicted wet"])
    ax.set_ylim(0, 1)
    ax.set_title("Wet-day hit rate")
    ax.set_ylabel("fraction")
    ax.grid(True, ls=":")

    # Annotate the basis and wet threshold as a caption below the plot
    cap = basis_desc
    if wet_thr is not None:
        cap += f"   -   Wet threshold = {float(wet_thr):.2f} mm/day"
    fig.text(0.99, 0.01, cap, ha="right", va="bottom", fontsize=9, color="0.3")

    # ---- Baseline overlays (mini-bars inside LR bar) ----
    if bo:
        try:
            dirs = resolve_baseline_dirs(
                sample_root=bo["sample_root"],
                types=tuple(bo.get("types", ())),
                split=str(bo.get("split", "test")),
                eval_type="extremes",
            )
        except Exception as e:
            logger.warning(f"[plot_tails] resolve_baseline_dirs failed: {e}")
            dirs = {}
        labels = bo.get("labels", {})
        styles = bo.get("styles", {})
        # Where is the LR bar located in each grouped cluster?
        i_lr = series.index("LR") if "LR" in series else None
        off_lr = (i_lr - (len(series)-1)/2) * w if i_lr is not None else 0.0
        btypes = list(dirs.items())
        nb = len(btypes)
        if nb > 0:
            w_b = w * 0.28      # width of each baseline mini-bar (inside LR bar)
            gap = w * 0.02      # small gap between adjacent mini-bars
            for j, (t, d) in enumerate(btypes):
                b_hdr, b_rows = _read_csv(d / "ext_tails.csv")
                if not b_rows:
                    continue
                has_gen = any(_norm_series_name(r[0]) == "GEN" for r in b_rows)
                sel = "GEN" if has_gen else ("LR" if any(_norm_series_name(r[0]) == "LR" for r in b_rows) else None)
                if sel is None:
                    continue
                r = [rr for rr in b_rows if _norm_series_name(rr[0]) == sel]
                if not r:
                    continue
                r = r[0]
                try:
                    p95 = float(r[1]); p99 = float(r[2])
                    wet = float(r[3]); hit = float(r[4])
                except Exception:
                    continue
                label = labels.get(t, t)
                style = dict(styles.get(t, {}))
                color = style.get("color", style.get("c", None))
                hatch = style.get("hatch", "///")
                edgecolor = color if color is not None else "k"
                rel = (j - (nb-1)/2)
                dx_j = rel * (w_b + gap)
                # positions centered around the LR bar location
                xl = [0 + off_lr + dx_j, 1 + off_lr + dx_j]
                xm = 0 + off_lr + dx_j
                xr = 0 + off_lr + dx_j
                axs[0].bar(xl, [p95, p99], width=w_b, fill=False, edgecolor=edgecolor, linewidth=1.3, hatch=hatch, label=label, zorder=6)
                axs[1].bar([xm], [wet],    width=w_b, fill=False, edgecolor=edgecolor, linewidth=1.3, hatch=hatch, zorder=6)
                axs[2].bar([xr], [hit],    width=w_b, fill=False, edgecolor=edgecolor, linewidth=1.3, hatch=hatch, zorder=6)
        axs[0].legend(fontsize=8, loc="best")

    fig.tight_layout()
    _savefig(fig, out_png, dpi=SET_DPI)
    plt.close(fig)

def plot_extremes(out_root: str | Path, eval_cfg: Any | None = None):
    out_root = Path(out_root)
    figs = _ensure_dir(out_root / "figures")
    tables = out_root / "tables"

    bo = _get_bo(eval_cfg)
    plot_return_levels(tables / "ext_rxk_gev.csv", figs / "ext_return_levels_rxk.png", bo=bo)
    plot_pot(tables / "ext_pot_gpd.csv", figs / "ext_pot_gpd.png", bo=bo)
    plot_tails(tables / "ext_tails.csv", figs / "ext_tails.png", bo=bo)