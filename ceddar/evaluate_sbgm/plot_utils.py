"""
    Publication plotting + table utilities for EDM evaluation.

    Reads CSV/JSON artifacts from <eval_root>/{tables,figures} and writes:
        - reliability_{thr}.png
        - spread_skill.png
        - fss_curves.png            (FSS vs scale_km per threshold)
        - psd_slope_bar.png
        - pit_hist.png
        - rank_hist.png
        - gev_rx{1,5}_return_levels.png
        - pot_diagnostics_{hr,pmm}.png

    Tables:
        - summary_metrics.csv           (CRPS mean, FSS@10km for {1, 5, 10 mm}, tails, wet-day freq)
        - summary_metrics.tex           (LaTeX version of above)
"""

from __future__ import annotations
import json
import csv
import logging
import torch
from pathlib import Path
from typing import Dict, Optional, Sequence, Any, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from sbgm.evaluate_sbgm.metrics_univariate import compute_isotropic_psd

logger = logging.getLogger(__name__)
# ---------- helpers ----------

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def _nice():
    plt.rcParams.update({
        "figure.figsize": (6,4),
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
    })

def _axes_as_list(axs_obj: Any) -> List[Axes]:
    """
    Normalize matplotlib's various axes return types into a List[Axes] for type checkers.
    Handles:
      - a single Axes
      - numpy.ndarray of Axes
      - list/tuple of Axes
    """
    if isinstance(axs_obj, Axes):
        return [axs_obj]
    if isinstance(axs_obj, np.ndarray):
        flat = axs_obj.ravel().tolist()
        return [a for a in flat if isinstance(a, Axes)]
    if isinstance(axs_obj, (list, tuple)):
        return [a for a in axs_obj if isinstance(a, Axes)]
    return []

# ---------- reliability ----------

def plot_reliability(eval_root: str,
                     thr_mm_list=(1,5,10),
                     baseline_eval_dirs: Optional[Dict[str, str]] = None,
                     *,
                     min_count_to_show: int = 300, # hide dots with few samples
                     rebin_edges: Optional[Sequence[float]] = None # e.g. [0, .02, .05, .1, .15, .2, .3, 1.0]
                     ):
    """
        Plot reliability diagrams from evaluation CSV, **count-weighted**.
        Reads reliability_bins.csv from eval_root/tables/. Contains one row per day per bin.
        Then:
            1) group by threshold and bin_center
            2) *weight* prob_pred and freq_obs by 'count'
            3) optionally rebin to provided edges (to avoid super-sparse tail bins)
            4) Limit the axes to [0, max_prob + 0.05] if max_prob<1.0 for better visibility.
    """
    tdir = Path(eval_root) / "tables"
    fdir = _ensure_dir(Path(eval_root) / "figures")
    # read CSV without pandas
    rows = []
    with open(tdir / "reliability_bins.csv", 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append({
                    "date": r.get("date", ""),
                    "thr": float(r.get("thr", 0.0)),
                    "bin_center": float(r.get("bin_center", r.get("prob_pred", 0.0))),
                    "prob_pred": float(r.get("prob_pred", 0.0)),
                    "freq_obs": float(r.get("freq_obs", 0.0)),
                    "count": int(float(r.get("count", 0))),
                })
            except Exception:
                continue
    
    if rebin_edges is not None:
        rebin_edges = list(rebin_edges)
    else:
        # Sensible default for precipitation: finer at low probs
        rebin_edges = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]

    for thr in thr_mm_list:
        # 1) Collect rows for this threshold
        rel = [r for r in rows if abs(r["thr"] - float(thr)) < 1e-9]

        # 2) Firdt group by original bin_center, count-weighted
        tmp = {}
        for r in rel:
            bc = r["bin_center"]
            cnt = max(r["count"], 0)
            if bc not in tmp:
                tmp[bc] = {
                    "prob_wsum": 0.0,
                    "freq_wsum": 0.0,
                    "count": 0,
                }
            tmp[bc]["prob_wsum"] += r["prob_pred"] * cnt
            tmp[bc]["freq_wsum"] += r["freq_obs"] * cnt
            tmp[bc]["count"] += cnt
        
        # 3) Turn into list of (prob_pred, freq_obs, count)
        agg = []
        for bc, d in tmp.items():
            c = d["count"]
            if c == 0:
                continue
            agg.append({
                "p": d["prob_wsum"] / c,
                "f": d["freq_wsum"] / c,
                "c": c,
            })
        
        # 4) Rebin to custom edges (also count-weighted)
        # bins: [e0, e1), [e1, e2), ..., [en-1, en]
        rbins = []
        for i in range(len(rebin_edges)-1):
            e0, e1 = rebin_edges[i], rebin_edges[i+1]
            num_p = num_f = tot_c = 0.0
            for item in agg:
                # include right edge in last bin
                if e0 <= item["p"] < e1 or (i == len(rebin_edges) - 2 and item["p"] == e1):
                    num_p += item["p"] * item["c"]
                    num_f += item["f"] * item["c"]
                    tot_c += item["c"]
            if tot_c > 0:
                mid = 0.5 * (e0 + e1)
                rbins.append({
                    "p": num_p / tot_c,
                    "f": num_f / tot_c,
                    "c": tot_c,
                })

        _nice()
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], "--", lw=1, label="Perfect")

        # 3-level visibility thresholds
        high_N = max(500, min_count_to_show)   # include more bins
        mid_N  = max(100, int(0.3 * high_N))   # allow smaller bins to show faintly

        # split bins
        hi  = [b for b in rbins if b["c"] >= high_N]
        mid = [b for b in rbins if mid_N <= b["c"] < high_N]
        low = [b for b in rbins if b["c"] < mid_N]

        # --- high-count (trustworthy) ---
        if hi:
            xs_hi = np.array([b["p"] for b in hi], dtype=float)
            ys_hi = np.array([b["f"] for b in hi], dtype=float)
            Ns_hi = np.array([b["c"] for b in hi], dtype=float)

            # sort by x so line doesn't jump
            order = np.argsort(xs_hi)
            xs_hi = xs_hi[order]
            ys_hi = ys_hi[order]
            Ns_hi = Ns_hi[order]

            # approximate binomial std for observed freq
            # se = sqrt( p * (1 - p) / N )
            se_hi = np.sqrt(np.clip(ys_hi * (1.0 - ys_hi) / np.maximum(Ns_hi, 1.0), 0.0, 1.0))

            # connect line + dots
            ax.plot(xs_hi, ys_hi, "--", color="black", lw=1.0, marker="o", ms=5, label=f"Thr ≥ {thr} mm")

            # error bars (make them faint)
            ax.errorbar(xs_hi, ys_hi, yerr=se_hi,
                        fmt="none", ecolor="0.4", elinewidth=0.8, alpha=0.6, capsize=2)
        else:
            xs_hi = ys_hi = Ns_hi = np.array([])

        # --- mid-count (ok, but not great) ---
        if mid:
            ax.scatter([b["p"] for b in mid],
                       [b["f"] for b in mid],
                       s=28, alpha=0.6, edgecolor="none")

        # --- low-count (just to show the trend) ---
        if low:
            ax.scatter([b["p"] for b in low],
                       [b["f"] for b in low],
                       s=18, alpha=0.25, edgecolor="none")

        # Right axis: put bars for bins we actually drew (hi+mid)
        ax2 = ax.twinx()
        counts_for_bar = hi + mid
        if counts_for_bar:
            ax2.bar([b["p"] for b in counts_for_bar],
                    [b["c"] for b in counts_for_bar],
                    width=0.03, alpha=0.25)
            ax2.set_ylim(0, max(b["c"] for b in counts_for_bar) * 1.15)
        else:
            ax2.set_ylim(0, 1.0)
        ax2.set_ylabel("Bin count")  # <-- label for 2nd y-axis
        
        # === Overlays from baselines ===
        if baseline_eval_dirs:
            for name, bdir in baseline_eval_dirs.items():
                btab = Path(bdir) / "tables" / "reliability_bins.csv"
                if not btab.exists():
                    continue
                try:
                    # Read baseline rows
                    browz = []
                    with open(btab, 'r') as fb:
                        br = csv.DictReader(fb)
                        for rr in br:
                            try:
                                browz.append({
                                    "thr": float(rr.get("thr", 0.0)),
                                    "bin_center": float(rr.get("bin_center", rr.get("prob_pred", 0.0))),
                                    "prob_pred": float(rr.get("prob_pred", 0.0)),
                                    "freq_obs": float(rr.get("freq_obs", 0.0)),
                                    "count": int(float(rr.get("count", 0))),
                                })
                            except Exception:
                                continue

                    # Repeat same aggregation for baseline
                    tmpb = {}
                    for r in browz:
                        if abs(r["thr"] - float(thr)) >= 1e-9:
                            continue
                        bc = r["bin_center"]
                        cnt = max(r["count"], 0)
                        if bc not in tmpb:
                            tmpb[bc] = {
                                "prob_wsum": 0.0,
                                "freq_wsum": 0.0,
                                "count": 0,
                            }
                        tmpb[bc]["prob_wsum"] += r["prob_pred"] * cnt
                        tmpb[bc]["freq_wsum"] += r["freq_obs"] * cnt
                        tmpb[bc]["count"] += cnt
                    agg_b = []
                    for bc, d in tmpb.items():
                        c = d["count"]
                        if c == 0:
                            continue
                        agg_b.append({
                            "p": d["prob_wsum"] / c,
                            "f": d["freq_wsum"] / c,
                            "c": c,
                        })
                    # Optional: no rebin for baseline, just plot directly
                    xb = [a["p"] for a in agg_b if a["c"] >= min_count_to_show]
                    yb = [a["f"] for a in agg_b if a["c"] >= min_count_to_show]
                    ax.plot(xb, yb, marker="x", linestyle="--", linewidth=1.0, label=f"{name} (≥ {thr} mm)")
                except Exception:
                    logger.warning(f"Could not plot baseline reliability for {name}")
                    continue
        x_max = max([b["p"] for b in rbins], default=1.0)
        y_max = max([b["f"] for b in rbins], default=1.0)
        ax.set_xlim(0, 1)#ax.set_xlim(0, x_max)
        ax.set_ylim(0, 1)#ax.set_ylim(0, y_max)
        ax.set_xlabel("Model probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(f"Reliability diagram (Thr ≥ {thr} mm/day)")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(str(fdir / f"reliability_{int(thr)}mm.png"), dpi=200)
        plt.close(fig)

# ---------- spread–skill ----------

def plot_spread_skill(eval_root: str,
                      *,
                      n_plot_bins: int = 20,
                      min_count_to_show: int = 1000,
                      make_skill_vs_spread: bool = True
                      ):
    """
        Read spread_skill.csv and make clean plot

        CSV has rows per day per bin. Here:
            1) Aggregate by bin_center, **count-weighted**
            2) Rebin to ~n_plot_bins evenly over x
            3) Plot spread and skill vs bin_center as two lines
            4) Optionally: separate skill-vs-spread scatter to see underdispersion.
    """
    tdir = Path(eval_root) / "tables"
    fdir = _ensure_dir(Path(eval_root) / "figures")
    
    # Read
    rows = []
    with open(tdir / "spread_skill.csv", 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append({
                    "bin_center": float(r.get("bin_center", 0.0)),
                    "spread": float(r.get("spread", 0.0)),
                    "skill": float(r.get("skill", 0.0)),
                    "count": int(float(r.get("count", 0))),
                })
            except Exception:
                continue
    
    if not rows:
        logger.warning("No valid rows found in spread_skill.csv. Skipping plot.")
        return

    # 1) Group by original bin_center, count-weighted
    tmp = {}
    for r in rows:
        x = r["bin_center"]
        c = max(r["count"], 0)
        if x not in tmp:
            tmp[x] = {
                "spread_wsum": 0.0,
                "skill_wsum": 0.0,
                "count": 0,
            }
        tmp[x]["spread_wsum"] += r["spread"] * c
        tmp[x]["skill_wsum"]  += r["skill"]  * c
        tmp[x]["count"]       += c

    pts = []
    for x, d in tmp.items():
        c = d["count"]
        if c == 0:
            continue
        pts.append({
            "x": x,
            "spread": d["spread_wsum"] / c,
            "skill":  d["skill_wsum"]  / c,
            "count":  c,
        })

    # Sort by x
    pts.sort(key=lambda z: z["x"])
    xs_all = np.array([p["x"] for p in pts], dtype=float)
    spread_all = np.array([p["spread"] for p in pts], dtype=float)
    skill_all = np.array([p["skill"] for p in pts], dtype=float)
    count_all = np.array([p["count"] for p in pts], dtype=float)

    # 2) Rebin to fixed number of bins (even in x)
    x_min, x_max = xs_all.min(), xs_all.max()
    edges = np.linspace(x_min, x_max + 1e-9, n_plot_bins + 1)
    Xb, Sb, Kb, Cb = [], [], [], []
    for i in range(n_plot_bins):
        e0, e1 = edges[i], edges[i+1]
        m = (xs_all >= e0) & (xs_all < e1)
        if not np.any(m):
            continue
        cnt = count_all[m].sum()
        if cnt == 0:
            continue
        Xb.append((e0 + e1) / 2)
        Sb.append((spread_all[m] * count_all[m]).sum() / cnt)
        Kb.append((skill_all[m] * count_all[m]).sum() / cnt)
        Cb.append(cnt)

    _nice()
    fig, ax = plt.subplots()
    Xb = np.array(Xb)
    Sb = np.array(Sb)
    Kb = np.array(Kb)
    Cb = np.array(Cb)

    # draw lines only through well-populated bins
    mask_good = Cb >= min_count_to_show
    mask_low  = ~mask_good

    # main lines
    ax.plot(Xb[mask_good], Sb[mask_good],
            marker="o", label="Spread")
    ax.plot(Xb[mask_good], Kb[mask_good],
            marker="s", label="Skill (MAE vs obs)")

    # faint background points (to show that there *is* data there)
    if np.any(mask_low):
        ax.scatter(Xb[mask_low], Sb[mask_low],
                   s=14, alpha=0.25, edgecolor="none")
        ax.scatter(Xb[mask_low], Kb[mask_low],
                   s=14, alpha=0.25, edgecolor="none")

    ax.set_xlabel("Ensemble mean (bin)")
    ax.set_ylabel("Value (mm/day)")
    ax.set_title("Spread-skill")
    ax.legend()
    
    ax.set_xlabel("Ensemble mean (bin)")
    ax.set_ylabel("Value (mm/day)")
    ax.set_title("Spread-skill")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(fdir / "spread_skill.png"), dpi=200)
    plt.close(fig)

    if make_skill_vs_spread:
        _nice()
        fig, ax = plt.subplots()

        # We have: Sb (mean spread), Kb (mean skill), Cb (counts) per rebinned bin.
        # We DON'T have per-bin raw rows here, so we estimate a simple SE from the
        # rebinned values as: se ~ (value * 0.35) / sqrt(N), which just gives you
        # a visual idea of stability (you can swap to a better formula if you later
        # keep per-row std in the CSV).
        eps = 1e-6
        rel_spread_se = 0.35   # heuristic
        rel_skill_se  = 0.35

        se_S = (np.abs(Sb) * rel_spread_se) / np.sqrt(np.maximum(Cb, 1.0))
        se_K = (np.abs(Kb) * rel_skill_se)  / np.sqrt(np.maximum(Cb, 1.0))

        # sort by spread so the line is clean
        order = np.argsort(Sb)
        Sb_s = Sb[order]
        Kb_s = Kb[order]
        se_S = se_S[order]
        se_K = se_K[order]
        Cb_s = Cb[order]

        # main line + markers
        ax.plot(Sb_s, Kb_s, "--", color="black", lw=1.0, marker="o", ms=4, label="Binned points")

        # error bars (vertical) to show variance in skill
        ax.errorbar(Sb_s, Kb_s, yerr=se_K,
                    fmt="none", ecolor="0.4", elinewidth=0.7, alpha=0.5, capsize=2)

        # faint background “size” info — optional
        if Cb_s.size and Cb_s.max() > 0:
            ax.scatter(Sb_s, Kb_s,
                       s=np.clip(Cb_s / Cb_s.max(), 0.1, 1.0) * 80,
                       alpha=0.25, edgecolor="none")

        maxv = float(max(Sb_s.max(), Kb_s.max()) * 1.05)
        ax.plot([0, maxv], [0, maxv], "k--", lw=1, label="spread = skill")
        ax.set_xlim(0, maxv)
        ax.set_ylim(0, maxv)
        ax.set_xlabel("Spread (mm/day)")
        ax.set_ylabel("Skill (MAE, mm/day)")
        ax.set_title("Spread vs skill (diagnostics)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(str(fdir / "spread_vs_skill.png"), dpi=200)
        plt.close(fig)
        

    # bins = {}
    # for r in rows:
    #     bc = r["bin_center"]
    #     if bc not in bins:
    #         bins[bc] = {"spread_sum":0.0, "skill_sum":0.0, "count_sum":0, "n":0}
    #     bins[bc]["spread_sum"] += r["spread"]
    #     bins[bc]["skill_sum"]  += r["skill"]
    #     bins[bc]["count_sum"]  += r["count"]
    #     bins[bc]["n"]          += 1

    # gb = []
    # for bc, d in bins.items():
    #     n = max(d["n"], 1)
    #     gb.append({
    #         "bin_center": bc,
    #         "spread": d["spread_sum"]/n,
    #         "skill":  d["skill_sum"]/n,
    #         "count":  d["count_sum"],
    #     })
    # gb.sort(key=lambda x: x["bin_center"])

    # _nice()
    # fig, ax = plt.subplots()
    # xs  = [r["bin_center"] for r in gb]
    # ys1 = [r["spread"] for r in gb]
    # ys2 = [r["skill"]  for r in gb]
    # ax.plot(xs, ys1, marker="o", label="Spread")
    # ax.plot(xs, ys2, marker="s", label="Skill (MAE vs obs)")
    # ax.set_xlabel("Ensemble mean (bin) • or another binning variable")
    # ax.set_ylabel("Value (units of mm/day)")
    # ax.set_title("Spread–skill")
    # ax.legend()
    # fig.tight_layout()
    # fig.savefig(str(fdir / "spread_skill.png"), dpi=200)
    # plt.close(fig)

# ---------- FSS curves ----------

def plot_fss_curves(eval_root: str,
                    thr_mm_list=(1,5,10),
                    baseline_eval_dirs: Optional[Dict[str, str]] = None
                    ):
    tdir = Path(eval_root) / "tables"
    fdir = _ensure_dir(Path(eval_root) / "figures")
    with open(tdir / "fss_summary.csv", 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        fss_cols = [c for c in fieldnames if c.lower().startswith("fss_")]
        items = []
        for row in reader:
            try:
                thr = float(row.get("thr", 0.0))
                for c in fss_cols:
                    try:
                        sc = float(c.split("_")[1].replace("km",""))
                    except Exception:
                        continue
                    val = float(row.get(c, 0.0))
                    items.append((thr, sc, val))
            except Exception:
                continue

    if not items:
        raise RuntimeError("No FSS_* columns found in fss_summary.csv")

    from collections import defaultdict
    by_thr = defaultdict(list)
    for thr, sc, val in items:
        by_thr[thr].append((sc, val))

    _nice()
    fig, ax = plt.subplots()
    for thr in thr_mm_list:
        pairs = sorted(by_thr.get(float(thr), []), key=lambda t: t[0])
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        ax.plot(xs, ys, marker="o", label=f"≥ {int(thr)} mm")
    
    # === Overlays from baselines ===
    if baseline_eval_dirs:
        for name, bdir in baseline_eval_dirs.items():
            bcsv = Path(bdir) / "tables" / "fss_summary.csv"
            if not bcsv.exists():
                continue
            try:
                with open(bcsv, 'r') as fb:
                    br = csv.DictReader(fb)
                    fss_cols_b = [c for c in (br.fieldnames or []) if c.lower().startswith("fss_")]
                    items_b = []
                    for row in br:
                        try:
                            thr_b = float(row.get("thr", 0.0))
                            for c in fss_cols_b:
                                try:
                                    sc = float(c.split("_")[1].replace("km",""))
                                except Exception:
                                    continue
                                val = float(row.get(c, 0.0))
                                items_b.append((thr_b, sc, val))
                        except Exception:
                            continue
                if not items_b:
                    logger.warning(f"Could not plot baseline FSS for {name}")
                    continue
                from collections import defaultdict as _dd
                by_thr_b = _dd(list)
                for thr_b, sc, val in items_b:
                    by_thr_b[thr_b].append((sc, val))
                for thr in thr_mm_list:
                    pairs_b = sorted(by_thr_b.get(float(thr), []), key=lambda t: t[0])
                    xs_b = [p[0] for p in pairs_b]
                    ys_b = [p[1] for p in pairs_b]
                    ax.plot(xs_b, ys_b, marker="x", linestyle="--", linewidth=1.2, label=f"{name} (≥ {int(thr)} mm)")
            except Exception:
                logger.warning(f"Could not plot baseline FSS for {name}")
                continue

    ax.set_xlabel("Neighborhood scale (km)")
    ax.set_ylabel("FSS")
    ax.set_ylim(0,1)
    ax.set_title("FSS vs scale")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(fdir / "fss_curves.png"), dpi=200)
    plt.close(fig)

# ---------- PSD slope ----------

def plot_psd_slope_bar(eval_root: str,
                       baseline_eval_dirs: Optional[Dict[str, str]] = None):
    tdir = Path(eval_root) / "tables"
    fdir = _ensure_dir(Path(eval_root) / "figures")
    with open(tdir / "psd_slope_summary.json", "r") as f:
        summ = json.load(f)

    labs, genv, obsv = [], [], []
    # Accept dict-of-dicts or list-of-dicts
    if isinstance(summ, dict):
        items = summ.items()
    elif isinstance(summ, list):
        items = [(str(i), itm) for i, itm in enumerate(summ)]
    else:
        items = []

    for k, v in items:
        if isinstance(v, dict) and ("gen_slope" in v) and ("hr_slope" in v):
            try:
                labs.append(k)
                genv.append(float(v["gen_slope"]))
                obsv.append(float(v["hr_slope"]))
            except Exception:
                continue

    # collect baseline slopes per season/key
    baseline_series = {}  # name -> list of gen slopes aligned to labs
    if baseline_eval_dirs and labs:
        for name, bdir in baseline_eval_dirs.items():
            bjson = Path(bdir) / "tables" / "psd_slope_summary.json"
            if not bjson.exists():
                logger.warning(f"Could not find baseline PSD slope file for {name}")
                continue
            try:
                bs = json.load(open(bjson, "r"))
                bvals = []
                if isinstance(bs, dict):
                    if "gen_slope" in bs and "hr_slope" in bs:
                        bvals = [float(bs["gen_slope"]) for _ in labs]
                    else:
                        for lab in labs:
                            v = bs.get(lab, {}).get("gen_slope", None)
                            if v is None and isinstance(bs.get(lab, None), (int, float)):
                                v = bs.get(lab)
                            bvals.append(float(v) if v is not None else np.nan)
                elif isinstance(bs, list):
                    for i, lab in enumerate(labs):
                        try:
                            v = bs[i].get("gen_slope", None)
                        except Exception:
                            v = None
                        bvals.append(float(v) if v is not None else np.nan)
                baseline_series[name] = bvals
            except Exception:
                logger.warning(f"Could not parse baseline PSD slope file for {name}")
                continue

    if not labs:
        return
    
    _nice()
    x = np.arange(len(labs))
    k = 2 + len(baseline_series) # Obs + PMM + baselines
    w = 0.8 / max(k, 1)
    fig, ax = plt.subplots()
    off = -0.4 + w/2
    ax.bar(x + off, obsv, width=w, label="HR (DANRA)") ; off += w
    ax.bar(x + off, genv, width=w, label="PMM (gen)")
    for name, bvals in baseline_series.items():
        off += w
        ax.bar(x + off, bvals, width=w, label=name)

    ax.set_xticks(x, labs)
    ax.set_ylabel("PSD slope (dimensionless)")
    ax.set_title("PSD slopes by season")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(fdir / "psd_slope_bar.png"), dpi=200)
    plt.close(fig)

# ---------- PIT & Rank ----------

def plot_pit_and_rank(eval_root: str, pit_bins=20):
    fdir = _ensure_dir(Path(eval_root) / "figures")
    # PIT
    pit_npz = fdir / "pit_values_all.npz"
    if pit_npz.exists():
        pits = np.load(pit_npz)["pits"]
        _nice()
        fig, ax = plt.subplots()
        ax.hist(pits, bins=pit_bins, range=(0,1), density=True)
        ax.hlines(1.0, 0, 1, linestyles="dashed")
        ax.set_xlim(0,1); ax.set_ylim(0, max(1.5, ax.get_ylim()[1]))
        ax.set_xlabel("PIT")
        ax.set_ylabel("Density")
        ax.set_title("PIT histogram")
        fig.tight_layout()
        fig.savefig(str(fdir / "pit_hist.png"), dpi=200)
        plt.close(fig)
    # Rank
    rank_npz = fdir / "rank_hist_counts.npz"
    if rank_npz.exists():
        counts = np.load(rank_npz)["counts"]
        _nice()
        fig, ax = plt.subplots()
        ax.bar(np.arange(len(counts)), counts)
        ax.set_xlabel("Rank (0..M)")
        ax.set_ylabel("Count")
        ax.set_title("Rank histogram")
        fig.tight_layout()
        fig.savefig(str(fdir / "rank_hist.png"), dpi=200)
        plt.close(fig)

    

# ---------- Extremes plots (optional; robust to missing keys) ----------

def plot_return_levels(eval_root: str):
    tdir = Path(eval_root) / "tables"
    fdir = _ensure_dir(Path(eval_root) / "figures")
    def _safe(path):
        return json.load(open(path,"r")) if path.exists() else None
    rx1_hr  = _safe(tdir / "gev_rx1_hr.json")
    rx1_pmm = _safe(tdir / "gev_rx1_pmm.json")
    rx5_hr  = _safe(tdir / "gev_rx5_hr.json")
    rx5_pmm = _safe(tdir / "gev_rx5_pmm.json")
    def _plot_one(name, hr, pm):
        if not hr or not pm: return
        # expect {"return_periods":[...], "return_levels":[...], "ci_low":[...], "ci_high":[...]}
        rp  = np.array(hr.get("return_periods", []))
        rlo = np.array(hr.get("ci_low", [])); rhi = np.array(hr.get("ci_high", []))
        gp  = np.array(pm.get("return_periods", []))
        plo = np.array(pm.get("ci_low", [])); phi = np.array(pm.get("ci_high", []))
        if rp.size==0 or gp.size==0: return
        _nice()
        fig, ax = plt.subplots()
        ax.plot(rp, (rlo+rhi)/2, "-o", label="HR (DANRA)")
        ax.fill_between(rp, rlo, rhi, alpha=0.2) # type: ignore
        ax.plot(gp, (plo+phi)/2, "-s", label="PMM (gen)")
        ax.fill_between(gp, plo, phi, alpha=0.2) # type: ignore
        ax.set_xscale("log")
        ax.set_xlabel("Return period (years)")
        ax.set_ylabel("Return level (mm/day)")
        ax.set_title(name)
        ax.legend()
        fig.tight_layout()
        fig.savefig(str(fdir / f"{name}_return_levels.png"), dpi=200)
        plt.close(fig)
    _plot_one("rx1", rx1_hr, rx1_pmm)
    _plot_one("rx5", rx5_hr, rx5_pmm)

# ---------- Summary table (CSV + LaTeX) ----------

def write_summary_table(eval_root: str):
    tdir = Path(eval_root) / "tables"

    # CRPS mean
    crps_vals = []
    p = tdir / "crps_daily.csv"
    if p.exists():
        with open(p, 'r') as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    crps_vals.append(float(r.get("crps", 0.0)))
                except Exception:
                    pass
    crps_mean = (sum(crps_vals) / len(crps_vals)) if crps_vals else float('nan')

    # FSS@10km
    fss10 = {}
    p = tdir / "fss_summary.csv"
    if p.exists():
        with open(p, 'r') as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            cand = None
            for key in ["FSS_10km","fss_10km","FSS_10"]:
                if key in fields:
                    cand = key; break
            for r in reader:
                try:
                    thr = int(float(r.get("thr", 0)))
                    if cand is not None:
                        fss10[thr] = float(r.get(cand, 0.0))
                except Exception:
                    pass

    # tails summary
    tails_path = tdir / "tails_summary.json"
    if tails_path.exists():
        tails = json.load(open(tails_path, 'r'))
        wet_freq = tails.get("wet_day_frequency", float('nan'))
        p95 = tails.get("p95", float('nan'))
        p99 = tails.get("p99", float('nan'))
    else:
        wet_freq = p95 = p99 = float('nan')

    # write CSV
    out_csv = tdir / "summary_metrics.csv"
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["CRPS_mean","FSS10km_1mm","FSS10km_5mm","FSS10km_10mm","WetDayFreq","P95","P99"])
        w.writerow([
            crps_mean,
            fss10.get(1, float('nan')),
            fss10.get(5, float('nan')),
            fss10.get(10, float('nan')),
            wet_freq, p95, p99,
        ])

    # minimal LaTeX table
    out_tex = tdir / "summary_metrics.tex"
    vals = [
        ("CRPS_mean", crps_mean),
        ("FSS10km_1mm", fss10.get(1, float('nan'))),
        ("FSS10km_5mm", fss10.get(5, float('nan'))),
        ("FSS10km_10mm", fss10.get(10, float('nan'))),
        ("WetDayFreq", wet_freq),
        ("P95", p95),
        ("P99", p99),
    ]
    with open(out_tex, 'w') as f:
        f.write("\\begin{tabular}{lr}\n\\toprule\n")
        for k, v in vals:
            try:
                f.write(f"{k} & {float(v):.3f}\\\\\n")
            except Exception:
                f.write(f"{k} & {v}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def plot_psd_curves(
    gen_bt: torch.Tensor,  # [B,1,H,W] PMM (or forecast)
    hr_bt:  torch.Tensor,  # [B,1,H,W] observations
    mask: torch.Tensor | None = None,
    dx_km: float | None = None,
    out_dir: str | None = None,
    seasons: tuple = ("ALL",),  # kept for API compatibility; not used to split here
    fname: str = "psd_curves.png",
    baseline_eval_dirs: Optional[Dict[str, str]] = None,
):
    """
    Plot isotropic PSD curves averaged over the batch for GEN vs HR.
    If dx_km is None, defaults to 1.0 (relative wavenumber).
    """
    if dx_km is None: dx_km = 1.0
    out_path = Path(out_dir) if out_dir is not None else Path(".")
    out_path.mkdir(parents=True, exist_ok=True)

    psd_gen = compute_isotropic_psd(gen_bt, dx_km=dx_km, mask=mask)
    psd_hr  = compute_isotropic_psd(hr_bt,  dx_km=dx_km, mask=mask)

    k  = psd_gen["k"].detach().cpu().numpy()
    Pg = psd_gen["psd"].detach().cpu().numpy()
    Ph = psd_hr["psd"].detach().cpu().numpy()

    # drop k=0 for log plots
    k_plot  = k[1:]
    Pg_plot = Pg[1:]
    Ph_plot = Ph[1:]

    plt.figure(figsize=(6,4))
    plt.loglog(k_plot, Pg_plot, label="PMM (gen)", linewidth=1.8)
    plt.loglog(k_plot, Ph_plot, label="HR (DANRA)", linewidth=1.8)
    plt.xlabel("Wavenumber k (1/km)")
    plt.ylabel("Power (arb.)")
    plt.title("Isotropic PSD (batch mean)")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / fname, dpi=200)
    plt.close()


def plot_psd_curves_eval(
    eval_root: str,
    baseline_eval_dirs: Optional[Dict[str, str]] = None,
    fname: str = "psd_curves.png",
    use_wavelength: bool = True,   # <--- new toggle
):
    tdir = Path(eval_root) / "tables"
    fdir = _ensure_dir(Path(eval_root) / "figures")
    npz_path = tdir / "psd_curves.npz"
    if not npz_path.exists():
        logger.warning(f"PSD curves NPZ not found: {npz_path}. Skipping plot.")
        return

    data = np.load(npz_path)

    # --- required (HR/PMM) ---
    k_hr = data["k"]              # (Nh,)
    psd_hr = data["psd_hr"]       # (Nh,)
    psd_pmm = data["psd_pmm"]     # (Nh,)
    ci_hr_lo = data.get("psd_hr_ci_lo", None)
    ci_hr_hi = data.get("psd_hr_ci_hi", None)
    ci_pmm_lo = data.get("psd_pmm_ci_lo", None)
    ci_pmm_hi = data.get("psd_pmm_ci_hi", None)

    # --- optional (LR) ---
    k_lr = data.get("k_lr", None)
    psd_lr = data.get("psd_lr", None)
    ci_lr_lo = data.get("psd_lr_ci_lo", None)
    ci_lr_hi = data.get("psd_lr_ci_hi", None)
    k_nyq_lr = data.get("k_nyquist_lr", None)
    if k_nyq_lr is not None:
        # could be array([val]) or scalar
        k_nyq_lr = float(np.array(k_nyq_lr).reshape(-1)[0])

    norm_mode = data.get("normalize", "none")
    if isinstance(norm_mode, np.ndarray):
        norm_mode = norm_mode.item()

    _nice()
    fig, ax = plt.subplots()

    # ========= HR / PMM =========
    order = None
    if use_wavelength:
        # k [1/km] -> lambda [km]
        lam_hr = np.where(k_hr > 0, 1.0 / k_hr, np.nan)
        # sort from large -> small wavelength (optional)
        order_hr = np.argsort(lam_hr)[::-1]
        lam_hr = lam_hr[order_hr]
        psd_hr_p = psd_hr[order_hr]
        psd_pmm_p = psd_pmm[order_hr]

        ax.plot(lam_hr, psd_hr_p, label="HR (DANRA)", color="black", lw=0.8)
        if ci_hr_lo is not None and ci_hr_hi is not None and ci_hr_lo.size == psd_hr.size:
            ax.fill_between(lam_hr,
                            ci_hr_lo[order_hr],
                            ci_hr_hi[order_hr],
                            color="black", alpha=0.2)

        ax.plot(lam_hr, psd_pmm_p, label="PMM (gen)", color="blue", lw=0.8)
        if ci_pmm_lo is not None and ci_pmm_hi is not None and ci_pmm_lo.size == psd_pmm.size:
            ax.fill_between(lam_hr,
                            ci_pmm_lo[order_hr],
                            ci_pmm_hi[order_hr],
                            color="blue", alpha=0.15)
    else:
        ax.plot(k_hr, psd_hr, label="HR (DANRA)", color="black", lw=0.9)
        if ci_hr_lo is not None and ci_hr_hi is not None and ci_hr_lo.size == psd_hr.size:
            ax.fill_between(k_hr, ci_hr_lo, ci_hr_hi, color="black", alpha=0.2)

        ax.plot(k_hr, psd_pmm, label="PMM (gen)", color="blue", lw=0.9)
        if ci_pmm_lo is not None and ci_pmm_hi is not None and ci_pmm_lo.size == psd_pmm.size:
            ax.fill_between(k_hr, ci_pmm_lo, ci_pmm_hi, color="blue", alpha=0.15)

    # ========= LR (optional) =========
    if (k_lr is not None) and (psd_lr is not None):
        # make sure shapes match first
        k_lr = np.array(k_lr)
        psd_lr = np.array(psd_lr)
        assert k_lr.shape == psd_lr.shape, f"LR k/PSD shape mismatch: {k_lr.shape} vs {psd_lr.shape}"

        # Nyquist from file
        if k_nyq_lr is not None and k_nyq_lr > 0:
            k_ny = float(k_nyq_lr)
        else:
            k_ny = None
        logger.info(f"[PSD DEBUG] LR Nyquist wavenumber: {k_ny}")
        if use_wavelength:
            lam_lr = np.where(k_lr > 0, 1.0 / k_lr, np.nan)
            # sort from large -> small wavelength (optional)
            order = np.argsort(lam_lr)[::-1]
            lam_lr = lam_lr[order]
            psd_lr = psd_lr[order]
            k_lr = k_lr[order]
        else:
            lam_lr = k_lr  # just for naming below

        # masks
        finite = np.isfinite(lam_lr) & np.isfinite(psd_lr)

        if k_ny is None:
            # No info -> draw single dashed line
            ax.plot(lam_lr[finite], psd_lr[finite],
                    label="LR (ERA5)", color="deeppink", lw=0.9, linestyle="-")
        else:
            # trusted: k <= k_ny (i.e. lambda >= lambda_ny)
            trusted = finite & (k_lr <= k_ny + 1e-12)
            ghost = finite & (k_lr > k_ny + 1e-12)
            logger.info(f"[PSD DEBUG] LR trusted wavenumbers: {k_lr[trusted]}")
            logger.info(f"[PSD DEBUG] LR ghost wavenumbers: {k_lr[ghost]}")

            # main, trusted part
            ax.plot(lam_lr[trusted], psd_lr[trusted],
                    label="LR (ERA5)", color="deeppink", lw=0.9, linestyle="-")
            # ghost/informative-only part
            if np.any(ghost):
                ax.plot(lam_lr[ghost], psd_lr[ghost],
                        color="deeppink", lw=0.6, linestyle="--", alpha=0.3)
                
            # Vertical line at Nyquist
            lam_ny = 1.0 / k_ny if use_wavelength else k_ny
            ax.axvline(x=lam_ny, color="black", linestyle="-", lw=0.7, label="LR Nyquist")

        # Only draw CIs at trusted scales
        if (ci_lr_lo is not None) and (ci_lr_hi is not None):
            ci_lr_lo = np.array(ci_lr_lo)
            ci_lr_hi = np.array(ci_lr_hi)
            if use_wavelength:
                ci_lr_lo = ci_lr_lo[order]
                ci_lr_hi = ci_lr_hi[order]
            if k_ny is not None:
                ci_mask = finite & (k_lr <= k_ny + 1e-12) & (ci_lr_lo.size == psd_lr.size) & (ci_lr_hi.size == psd_lr.size)
            else:
                ci_mask = finite & (ci_lr_lo.size == psd_lr.size) & (ci_lr_hi.size == psd_lr.size)
            if ci_lr_lo.size == psd_lr.size and ci_lr_hi.size == psd_lr.size:
                # Convert to Python sequences (lists) so the matplotlib type hints accept them
                ax.fill_between(np.asarray(lam_lr[ci_mask]).tolist(),
                                np.asarray(ci_lr_lo[ci_mask]).tolist(),
                                np.asarray(ci_lr_hi[ci_mask]).tolist(),
                                color="deeppink", alpha=0.15)
                       

    # --- baselines (they will likely be on HR k only) ---
    if baseline_eval_dirs:
        for name, bdir in baseline_eval_dirs.items():
            bnpz = Path(bdir) / "tables" / "psd_curves.npz"
            if not bnpz.exists():
                continue
            try:
                bd = np.load(bnpz)
                k_b = bd["k"]
                psd_b = bd["psd_pmm"]
                if use_wavelength:
                    lam_b = np.where(k_b > 0, 1.0 / k_b, np.nan)
                    order_b = np.argsort(lam_b)[::-1]
                    lam_b = lam_b[order_b]
                    psd_b = psd_b[order_b]
                    val_b = np.isfinite(lam_b) & np.isfinite(psd_b)
                    ax.plot(lam_b[val_b], psd_b[val_b],
                            lw=0.7, linestyle=":", label=f"{name} (gen)")
                else:
                    ax.plot(k_b, psd_b, lw=0.7, linestyle=":", label=f"{name} (gen)")
            except Exception:
                logger.warning(f"Could not plot baseline PSD curves for {name}")
                continue

    ax.set_xscale("log")
    ax.set_yscale("log")
    if use_wavelength:
        ax.invert_xaxis()
        ax.set_xlabel("Wavelength λ (km)")
    else:
        ax.set_xlabel("Radial wavenumber k (1/km)")

    if norm_mode == "none":
        ax.set_ylabel("Spectral power")
    elif norm_mode == "per_field":
        ax.set_ylabel("Spectral power (per-field normalized)")
    else:
        ax.set_ylabel(f"Spectral power (norm={norm_mode})")

    ax.set_title("Isotropic Power Spectral Density (PSD)")
    ax.grid(True, which="both", ls=":")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(fdir / fname), dpi=200)
    plt.close(fig)



def plot_date_montages(eval_root: str,
                       dates: list[str],
                       baselines: Optional[Dict[str, str]] = None,
                       n_members: int = 3,
                       fname_prefix: str = "montage_",
                       cmap: str = "Blues"):
    """
    For each date, create a panel comparing:
      HR | PMM | M1..Mk | (each available baseline)
    Reads from:
      <eval_root>/../generated_samples/{lr_hr_phys,pmm_phys,ensembles_phys}/DATE.npz
    Baselines are dict name->baseline_eval_dir; we derive their generated_samples/pmm path.

    Colormap + normalization:
      vmin=0, vmax=99.5th percentile across all shown panels for that date.
    Saves PNG to <eval_root>/figures/{fname_prefix}{date}.png
    """
    eval_root_path = Path(eval_root)
    figs_dir  = _ensure_dir(eval_root_path / "figures")
    gen_root  = eval_root_path.parent / "generated_samples"

    def _get_np(p, key):
        if not p.exists(): return None
        try:
            d = np.load(p, allow_pickle=True)
            a = d.get(key, None)
            if a is None: return None
            a = np.nan_to_num(a.squeeze(), nan=0.0, posinf=0.0, neginf=0.0)
            return a
        except Exception:
            return None

    for d in dates:
        hr_p  = gen_root / "lr_hr_phys"     / f"{d}.npz"
        pmm_p = gen_root / "pmm_phys"       / f"{d}.npz"
        ens_p = gen_root / "ensembles_phys" / f"{d}.npz"
        if not pmm_p.exists():
            # fallback to model space
            hr_p  = gen_root / "lr_hr"     / f"{d}.npz"
            pmm_p = gen_root / "pmm"       / f"{d}.npz"
            ens_p = gen_root / "ensembles" / f"{d}.npz"

        hr  = _get_np(hr_p,  "hr")
        pmm = _get_np(pmm_p, "pmm")
        ens = _get_np(ens_p, "ens")  # [M,H,W] or [M,1,H,W] squeezed above
        if hr is None or pmm is None:
            continue
        if ens is not None and ens.ndim == 4:
            ens = ens[:,0]  # -> [M,H,W]

        # collect panels
        panels = [("HR", hr), ("PMM", pmm)]
        if ens is not None and ens.ndim == 3 and ens.shape[0] > 0:
            m = min(n_members, ens.shape[0])
            for i in range(m):
                panels.append((f"M{i+1}", ens[i]))

        # baselines
        if baselines:
            for name, b_eval in baselines.items():
                # from <eval_dir>/.../baselines/<name>/<split>/tables ⇒ gen at ../../generated_samples/pmm
                b_gen = Path(b_eval).parent.parent / "generated_samples" / "pmm" / f"{d}.npz"
                b_img = _get_np(b_gen, "pmm")
                if b_img is not None:
                    panels.append((name, b_img))

        # common normalization
        vals_for_scale = np.concatenate([np.ravel(x) for _, x in panels])
        vmax = np.percentile(vals_for_scale, 99.5)
        vmax = max(1.0, float(vmax))
        vmin = 0.0

        _nice()
        C = len(panels)
        fig, axs_obj = plt.subplots(1, C, figsize=(3.5*C, 3.2), constrained_layout=True, squeeze=False)
        axes_list = _axes_as_list(axs_obj)
        if len(axes_list) != C and len(axes_list) > 0:
            # pad defensively if some backends return fewer Axes
            axes_list = axes_list + [axes_list[-1]] * (C - len(axes_list))

        ims = []
        for ax, (lab, img) in zip(axes_list, panels):
            im = ax.imshow(img, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
            ims.append(im)
            ax.set_title(lab)
            ax.set_xticks([]); ax.set_yticks([])

        # one shared colorbar on the right
        cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cb = fig.colorbar(ims[0], cax=cax)
        cb.set_label("mm/day")

        fig.suptitle(d)
        fig.savefig(str(figs_dir / f"{fname_prefix}{d}.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)



from typing import Literal

def plot_pooled_pixel_distributions(
    eval_root: str,
    baseline_eval_dirs: Optional[Dict[str, str]] = None,
    yscale: Literal["linear", "log", "symlog", "logit"] = "log",
    fname: str = "pooled_pixel_distributions.png",
):
    """
    Plots pooled pixel distributions for Obs (HR), PMM (model), optional LR, and optional baselines.
    Relies on CSVs written by metrics_univariate.compute_and_save_pooled_pixel_distributions:
      tables/pixel_dist_bins.csv
      tables/pixel_dist_hr.csv
      tables/pixel_dist_pmm.csv
      tables/pixel_dist_lr.csv (optional)
    Baselines (if provided): each should have its own tables/pixel_dist_pmm.csv (+ bins).
    """
    tdir = Path(eval_root) / "tables"
    fdir = _ensure_dir(Path(eval_root) / "figures")

    bins_csv = tdir / "pixel_dist_bins.csv"
    pmm_csv  = tdir / "pixel_dist_pmm.csv"
    hr_csv   = tdir / "pixel_dist_hr.csv"

    if not (bins_csv.exists() and pmm_csv.exists() and hr_csv.exists()):
        logger.warning("[plot] Missing pooled distribution CSVs under %s", tdir)
        return

    # Read bins
    bin_left, bin_right = [], []
    with open(bins_csv, "r") as f:
        rd = csv.DictReader(f)
        for row in rd:
            try:
                bin_left.append(float(row["bin_left"]))
                bin_right.append(float(row["bin_right"]))
            except Exception:
                continue
    if not bin_left:
        logger.warning("[plot] No bins in %s", bins_csv)
        return
    centers = (np.array(bin_left) + np.array(bin_right)) / 2.0

    def _read_counts(path: Path):
        idx, cnt = [], []
        with open(path, "r") as f:
            rd = csv.DictReader(f)
            for r in rd:
                try:
                    idx.append(int(float(r["bin_idx"])))
                    cnt.append(int(float(r["count"])))
                except Exception:
                    continue
        return np.array(idx, dtype=int), np.array(cnt, dtype=float)

    _, H_pmm = _read_counts(pmm_csv)
    _, H_hr  = _read_counts(hr_csv)
    H_lr = None
    lr_csv = tdir / "pixel_dist_lr.csv"
    if lr_csv.exists():
        _, H_lr = _read_counts(lr_csv)

    # Plot
    _nice()
    fig, ax = plt.subplots()
    # Normalize to probabilities
    def _norm(h): 
        s = max(h.sum(), 1.0)
        return h / s
    ax.plot(centers, _norm(H_hr),  label="HR (DANRA)", linewidth=1.8)
    ax.plot(centers, _norm(H_pmm), label="PMM (gen)",  linewidth=1.8)
    if H_lr is not None:
        ax.plot(centers, _norm(H_lr), label="ERA5 (LR→HR)", linewidth=1.5)

    # Baseline overlays
    if baseline_eval_dirs:
        for name, bdir in baseline_eval_dirs.items():
            b_bins = Path(bdir) / "tables" / "pixel_dist_bins.csv"
            b_csv  = Path(bdir) / "tables" / "pixel_dist_pmm.csv"
            if not (b_bins.exists() and b_csv.exists()):
                continue
            # For simplicity we assume same binning; if not, we still plot against our centers.
            _, H_b = _read_counts(b_csv)
            try:
                ax.plot(centers, _norm(H_b), linestyle="--", linewidth=1.2, label=name)
            except Exception:
                logger.warning("[plot] Could not overlay baseline %s", name)

    ax.set_xlabel("Precipitation (mm/day)")
    ax.set_ylabel("Probability")
    if yscale in {"log","symlog"}:
        ax.set_yscale(yscale)
    ax.set_title("Pooled pixel distributions")
    ax.grid(True, which="both", ls=":")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(fdir / fname), dpi=200)
    plt.close(fig)


def plot_yearly_maps(eval_root: str,
                     years: Optional[Sequence[int]] = None,
                     which: Sequence[str] = ("mean","sum","rx1","rx5"),
                     baselines: Optional[Dict[str, str]] = None,
                     cmap: str = "Blues",
                     *,
                     add_ratio: bool = False,
                     ratio_metrics: Sequence[str] = ("mean","sum"),
                     ratio_clip: float = 100.0,
                     ratio_cmap: str = "RdBu_r", # diverging
                     ):
    """
    Read maps from <eval_root>/maps/year_YYYY_*.npz and plot side-by-side panels:
      HR | PMM | (LR if present)
    Apply ONE land-sea mask consistently to all panels.

    Works for both:
      - main model eval:   .../generated_samples/evaluation/{split}
      - baseline eval:     .../generated_samples/evaluation/baselines/{type}/{split}
    """
    eval_root_path = Path(eval_root)
    maps_dir = eval_root_path / "maps"
    figs_dir = _ensure_dir(eval_root_path / "figures")
    if not maps_dir.exists():
        logger.warning("[plot] maps dir missing: %s", maps_dir)
        return

    # --- find a matching generated-samples root to look for LSM ---
    # 1) default (main model):  <eval_root>/../generated_samples
    cand_gen_roots = [eval_root_path.parent / "generated_samples"]

    # 2) baseline structure: parse ".../evaluation/baselines/<type>/<split>"
    parts = list(eval_root_path.parts)
    try:
        idx = parts.index("evaluation")
        if idx + 1 < len(parts) and parts[idx+1] == "baselines":
            # build: .../generation/baselines/<type>/<split>/lsm  (preferred)
            base = Path(*parts[:idx])  # .../generated_samples
            btype = parts[idx+2] if idx+2 < len(parts) else ""
            bsplit = parts[idx+3] if idx+3 < len(parts) else ""
            cand_gen_roots.append(base / "generation" / "baselines" / btype / bsplit)
            # also try .../generated_samples/baselines/<type>/<split>
            cand_gen_roots.append(base / "generated_samples" / "baselines" / btype / bsplit)
    except ValueError:
        pass

    lsm_mask = None
    for gr in cand_gen_roots:
        lsm_dir = gr / "lsm"
        if not lsm_dir.exists():
            continue
        cand = next(lsm_dir.glob("*.npz"), None)
        if cand is None:
            continue
        try:
            d = np.load(cand, allow_pickle=True)
            mm = d.get("lsm", None)
            if mm is None:
                mm = d.get("lsm_hr", None)
            if mm is None:
                continue
            m = np.asarray(mm)
            while m.ndim > 2 and 1 in m.shape:
                m = np.squeeze(m)
            if m.ndim == 3 and m.shape[0] <= 8:
                m = m[0]
            lsm_mask = (m > 0.5)
            break
        except Exception:
            continue

    def _squeeze2d(a: np.ndarray | None) -> np.ndarray | None:
        if a is None: return None
        x = np.asarray(a)
        while x.ndim > 2 and 1 in x.shape:
            x = np.squeeze(x)
        # tolerate channel-first arrays: pick the first channel
        if x.ndim == 3 and x.shape[0] <= 8:
            x = x[0]
        return x if x.ndim == 2 else None

    def _mask(x: np.ndarray | None) -> np.ndarray | None:
        if x is None: return None
        if lsm_mask is None: return x
        try:
            out = x.copy()
            m = lsm_mask
            if m.shape != out.shape:
                m = np.broadcast_to(m, out.shape)
            out[~m] = np.nan
            return out
        except Exception:
            return x

    # infer years if not provided
    if years is None:
        yrs = []
        for p in maps_dir.glob("year_*_means.npz"):
            try:
                yrs.append(int(p.stem.split("_")[1]))
            except Exception:
                continue
        years = sorted(set(yrs))

    for y in years:
        for metric in which:
            p = maps_dir / f"year_{int(y)}_{metric}.npz"
            if not p.exists():
                continue
            try:
                d = np.load(p, allow_pickle=True)
            except Exception:
                continue

            hr  = _mask(_squeeze2d(d.get("hr", None)))
            pmm = _mask(_squeeze2d(d.get("pmm", None)))
            lr  = _mask(_squeeze2d(d.get("lr", None)))

            panels = [("HR (DANRA)", hr), ("PMM (gen)", pmm)]
            if lr is not None:
                panels.append(("ERA5 (LR→HR)", lr))

            vals = [v for _, v in panels if v is not None]
            if not vals:
                continue
            pool = np.concatenate([v[np.isfinite(v)].ravel() for v in vals if np.isfinite(v).any()])
            if pool.size == 0:
                continue
            vmin, vmax = 0.0, float(np.percentile(pool, 99.5))

            _nice()
            fig, axs = plt.subplots(1, len(panels), figsize=(4*len(panels), 4), constrained_layout=True)
            axs_list = _axes_as_list(axs)
            ims = []
            for ax, (lab, img) in zip(axs_list, panels):
                if img is None:
                    ax.axis("off")
                    continue
                im = ax.imshow(img, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
                ims.append(im)
                ax.set_title(lab)
                ax.set_xticks([]); ax.set_yticks([])
            if ims:
                cbar = fig.colorbar(ims[-1], ax=axs_list, shrink=0.8)
                if metric in {"mean","means"}:
                    cbar.set_label("mm/day")
                else:
                    cbar.set_label("mm (annual)")
            fig.suptitle(f"{int(y)} • Yearly {metric}")
            fig.savefig(str(figs_dir / f"year_{int(y)}_{metric}.png"), dpi=200)
            plt.close(fig)

            # Optional: percent-difference panel (PMM vs HR) 
            if add_ratio and (metric in set(ratio_metrics)):
                # require both HR and PMM to compute percent difference
                if hr is None or pmm is None:
                    logger.warning("Skipping ratio plot for year %s metric %s: missing HR or PMM", y, metric)
                else:
                    hr_safe = np.copy(hr)
                    eps = 1e-8
                    mask_valid = np.isfinite(hr_safe) & (np.abs(hr_safe) > eps) & np.isfinite(pmm)
                    ratio = np.full_like(pmm, np.nan, dtype=np.float32)
                    ratio[mask_valid] = 100.0 * (pmm[mask_valid]/hr_safe[mask_valid] - 1.0)

                    # Symmetric color scaling around 0
                    if np.isfinite(ratio).any():
                        vmax_auto = np.nanpercentile(np.abs(ratio[np.isfinite(ratio)]), 99.0)
                        vmax = float(min(max(vmax_auto, 1.0), float(ratio_clip)))
                    else:
                        vmax = float(ratio_clip)
                    vmin = -vmax

                    _nice()
                    fig_r, ax_r = plt.subplots(1, 1, figsize=(5,4), constrained_layout=True)
                    imr = ax_r.imshow(ratio, origin="lower", vmin=vmin, vmax=vmax, cmap=ratio_cmap)
                    ax_r.set_title(f"PMM vs HR (% difference)")
                    ax_r.set_xticks([]); ax_r.set_yticks([])
                    cbar = fig_r.colorbar(imr, ax=ax_r, shrink=0.8)
                    cbar.set_label("% difference (PMM/HR - 1) * 100")
                    fig_r.suptitle(f"{int(y)} • Yearly {metric} Percent Difference")
                    out_name = figs_dir / f"year_{int(y)}_{metric}_ratio.png"
                    fig_r.savefig(str(out_name), dpi=200)
                    plt.close(fig_r)







# ---------- Driver ----------

def make_publication_outputs(cfg):
    cfg_full_gen_eval = cfg.get("full_gen_eval", {})
    eval_root = cfg_full_gen_eval.get("eval_dir", None)  # If None, use default
    if eval_root is None:
        from sbgm.evaluate_sbgm.evaluation_main import _default_eval_dir
        eval_root = _default_eval_dir(cfg)

    thresholds = cfg_full_gen_eval.get("thresholds_mm", [1,5,10])
    pit_bins = cfg_full_gen_eval.get("pit_bins", 20)


    # --- baseline overlay preparation ---
    compare_with_baselines = bool(cfg_full_gen_eval.get("compare_with_baselines", False))
    baseline_names = cfg_full_gen_eval.get("baseline_names", ["bilinear","qm","unet_sr"])
    baseline_split = str(cfg_full_gen_eval.get("baseline_split", "test"))

    baseline_eval_dirs = None
    if compare_with_baselines:
        base_root = Path(cfg["paths"]["sample_dir"]) / "evaluation" / "baselines"
        cand = {}
        for name in baseline_names:
            d = base_root / name / baseline_split
            if d.exists():
                cand[name] = str(d)
        if cand:
            baseline_eval_dirs = cand

    eval_root_str = str(eval_root)
    plot_reliability(eval_root_str, thr_mm_list=thresholds, baseline_eval_dirs=baseline_eval_dirs)
    plot_spread_skill(eval_root_str)
    plot_fss_curves(eval_root_str, thr_mm_list=thresholds, baseline_eval_dirs=baseline_eval_dirs)
    plot_psd_slope_bar(eval_root_str, baseline_eval_dirs=baseline_eval_dirs)
    plot_psd_curves_eval(eval_root_str, baseline_eval_dirs=baseline_eval_dirs)
    plot_pit_and_rank(eval_root_str, pit_bins=pit_bins)
    plot_return_levels(eval_root_str)
    plot_pooled_pixel_distributions(eval_root_str, baseline_eval_dirs=baseline_eval_dirs)
    plot_yearly_maps(eval_root_str, which=["mean","sum","rx1","rx5"], baselines=baseline_eval_dirs)

    # date montages
    dates_for_panels = cfg.get("full_gen_eval", {}).get("montage_member_dates", [])
    n_members = int(cfg.get("full_gen_eval", {}).get("n_montage_members", 3))
    if dates_for_panels:
        plot_date_montages(eval_root_str, dates_for_panels, baselines=baseline_eval_dirs, n_members=n_members)

    write_summary_table(eval_root_str)

