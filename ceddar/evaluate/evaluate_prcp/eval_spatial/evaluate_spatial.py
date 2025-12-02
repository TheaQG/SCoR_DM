from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence, Dict, Any, List
import logging
import numpy as np
import torch

from evaluate.evaluate_prcp.eval_spatial.metrics_spatial import (
    accumulate_daily_fields,
    compute_spatial_climatologies,
    save_maps_npz,
    summarize_group_spatial_maps,
    write_spatial_summary_csv
)


logger = logging.getLogger(__name__)

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

# ---- small helpers for grouping ----
def _year_of(d: str) -> int:
    s = d.strip()
    if len(s) == 8 and s.isdigit():
        return int(s[:4])
    # assume YYYY-MM-DD
    return int(s[:4])

def _season_of(d: str) -> str:
    s = d.strip()
    if len(s) == 8 and s.isdigit():
        m = int(s[4:6])
    else:
        m = int(s[5:7])
    if m in (12, 1, 2): return "DJF"
    if m in (3, 4, 5):  return "MAM"
    if m in (6, 7, 8):  return "JJA"
    return "SON"

# ---- ensemble loading helper ----
def _load_members_for_date(resolver, date: str) -> List[torch.Tensor]:
    """
    Try several resolver APIs to load all generated ensemble members for a given date.
    Falls back to PMM as a single-member list if no ensemble API is available.
    Expected tensor per member: [H, W].
    """
    # Preferred: a single array/tensor [M,H,W] via load_ens
    if hasattr(resolver, "load_ens"):
        m = resolver.load_ens(date)
        if isinstance(m, np.ndarray) and m.ndim == 3:
            return [torch.from_numpy(m[i]) for i in range(m.shape[0])]
        if isinstance(m, torch.Tensor) and m.dim() == 3:
            return [m[i] for i in range(m.shape[0])]

    if hasattr(resolver, "load_members"):
        m = resolver.load_members(date)
        return [torch.as_tensor(x) for x in m] if isinstance(m, (list, tuple)) else [torch.as_tensor(m)]

    if hasattr(resolver, "load_gen_members"):
        m = resolver.load_gen_members(date)
        return [torch.as_tensor(x) for x in m]

    if hasattr(resolver, "load_ensemble"):
        m = resolver.load_ensemble(date)  # np.ndarray [M,H,W] or Tensor [M,H,W]
        if isinstance(m, np.ndarray) and m.ndim == 3:
            return [torch.from_numpy(m[i]) for i in range(m.shape[0])]
        if isinstance(m, torch.Tensor) and m.dim() == 3:
            return [m[i] for i in range(m.shape[0])]

    # Fallback: PMM as a single "member"
    if hasattr(resolver, "load_pmm"):
        x = resolver.load_pmm(date)
        return [torch.as_tensor(x)]

    return []

def run_spatial(
    resolver,                 # EvalDataResolver-like (load_obs/load_pmm/load_lr/load_mask/list_dates[/ensemble])
    eval_cfg,                 # namespace-like; we read attributes with getattr
    out_root: str | Path,
    *,
    which_source: str = "pmm",   # kept for API compatibility; not used for routing
    group_by: str = "year",      # "year" | "season" | "all"
    seasons: Sequence[str] = ("ALL","DJF","MAM","JJA","SON"),
    make_plots: bool = True,
) -> None:
    """
    Spatial maps evaluation:
      - Compute pixelwise climatologies (mean, sum, wetfreq, Rx1, Rx5, P95/P99)
      - For HR, LR, and **Ensemble mean/std** (if available)
      - Grouped by year, season, or whole period
    Writes per-group NPZ bundles to <out_root>/tables and then calls plotter.
    """
    out_root = Path(out_root)
    tables_dir = _ensure_dir(out_root / "tables")
    _ensure_dir(out_root / "figures")

    # What to include (defaults: HR, LR, and ENS; PMM off)
    hr_first  = bool(getattr(eval_cfg, "spatial_include_hr",  True))
    lr_first  = bool(getattr(eval_cfg, "spatial_include_lr",  True))
    ens_first = bool(getattr(eval_cfg, "spatial_include_ens", True))
    gen_first = bool(getattr(eval_cfg, "spatial_include_gen", False))  # PMM OFF by default

    # Core knobs
    wet_thr  = float(getattr(eval_cfg, "spatial_wet_thr_mm", 1.0))
    rxk_days = tuple(getattr(eval_cfg, "spatial_rxk_days", (1, 5)))
    pct_list = tuple(getattr(eval_cfg, "spatial_percentiles", (95.0, 99.0)))

    # Loaders
    def loader(label: str):
        if label == "hr":
            return resolver.load_obs
        if label in {"pmm", "gen", "generated"}:
            return resolver.load_pmm
        if label == "lr":
            return resolver.load_lr
        raise ValueError(f"Unknown source: {label}")

    # dates
    all_dates: List[str] = list(resolver.list_dates())
    if not all_dates:
        logger.warning("[spatial] No dates found. Nothing to do.")
        return

    # split dates
    def split_dates(dates: List[str]) -> Dict[str, List[str]]:
        if group_by == "all":
            return {"ALL": dates}
        if group_by == "season":
            out: Dict[str, List[str]] = {s: [] for s in seasons}
            for d in dates:
                s = _season_of(d)
                if s in out:
                    out[s].append(d)
            return {s: out.get(s, []) for s in seasons if s in out}
        # default: by year
        buckets: Dict[str, List[str]] = {}
        for d in dates:
            y = _year_of(d)
            buckets.setdefault(str(y), []).append(d)
        return dict(sorted(buckets.items()))

    groups = split_dates(all_dates)
    logger.info("[spatial] Groups: %s", ", ".join(f"{k}({len(v)})" for k, v in groups.items()))

    # build source list (no PMM by default)
    sources: List[str] = []
    if hr_first:  sources.append("hr")
    if ens_first: sources.append("ens")
    if lr_first:  sources.append("lr")
    if gen_first: sources.append("pmm")
    if not sources:
        sources = ["ens"]

    summary_rows: List[Dict[str, Any]] = []
    # main loop
    for gname, gdates in groups.items():
        if not gdates:
            logger.warning("[spatial] Group %s is empty; skipping.", gname)
            continue

        def mask_fn(d: str):
            return resolver.load_mask(d)
        
        group_maps: Dict[str, Dict[str, torch.Tensor]] = {}
        for src in sources:
            if src == "ens":
                # ---- Ensemble path: compute per-member maps, then aggregate mean/std ----
                all_members: List[List[torch.Tensor]] = []
                kept_dates: List[str] = []
                for d in gdates:
                    mems = _load_members_for_date(resolver, d)
                    if not mems:
                        continue
                    m = mask_fn(d) if bool(getattr(eval_cfg, "eval_land_only", True)) else None
                    if m is not None:
                        m = (m > 0.5)
                    day_members: List[torch.Tensor] = []
                    for x in mems:
                        x = x.float()
                        if x.dim() == 4 and x.shape[:2] == (1, 1): x = x.squeeze(0).squeeze(0)
                        elif x.dim() == 3 and x.shape[0] == 1:    x = x.squeeze(0)
                        if x.dim() != 2:
                            x = x.reshape(x.shape[-2], x.shape[-1])
                        if m is not None:
                            x = x.clone()
                            x[~m] = float("nan")
                        day_members.append(x)
                    all_members.append(day_members)
                    kept_dates.append(d)

                if not all_members:
                    logger.warning("[spatial] No ensemble fields in group %s; skipping 'ens'.", gname)
                    continue

                # transpose: list of members, each with list over time
                M = len(all_members[0])
                member_stacks: List[List[torch.Tensor]] = [[] for _ in range(M)]
                for day in all_members:
                    while len(day) < M:
                        day.append(day[-1])  # pad rare shorter days
                    for mi in range(M):
                        member_stacks[mi].append(day[mi])

                # compute per-member maps
                member_maps: List[Dict[str, torch.Tensor]] = []
                for mi in range(M):
                    cmaps = compute_spatial_climatologies(
                        member_stacks[mi],
                        wet_thr_mm=wet_thr,
                        percentiles=pct_list,
                        rxk_days=rxk_days,
                    )
                    member_maps.append(cmaps)

                # aggregate across members
                keys = member_maps[0].keys()
                agg_mean: Dict[str, torch.Tensor] = {}
                agg_std:  Dict[str, torch.Tensor] = {}
                for k in keys:
                    stk = torch.stack([m[k] for m in member_maps], dim=0)  # [M,H,W]
                    agg_mean[k] = torch.nanmean(stk, dim=0)
                    agg_std[k]  = torch.sqrt(torch.nanmean((stk - agg_mean[k])**2, dim=0))

                save_maps_npz(tables_dir / f"spatial_ensmean_{gname}.npz", **agg_mean)
                save_maps_npz(tables_dir / f"spatial_ensstd_{gname}.npz",  **agg_std)
                (tables_dir / f"spatial_ensmean_{gname}.meta.txt").write_text(
                    f"source=ensmean\ngroup={gname}\nn_days={len(kept_dates)}\nM={M}\n"
                    f"wet_thr_mm={wet_thr}\nrxk_days={list(rxk_days)}\npercentiles={list(pct_list)}\n"
                )
                (tables_dir / f"spatial_ensstd_{gname}.meta.txt").write_text(
                    f"source=ensstd\ngroup={gname}\nn_days={len(kept_dates)}\nM={M}\n"
                    f"wet_thr_mm={wet_thr}\nrxk_days={list(rxk_days)}\npercentiles={list(pct_list)}\n"
                )
                group_maps["ensmean"] = agg_mean
                group_maps["ensstd"]  = agg_std
                continue  # next source

            # ---- HR / LR path ----
            load_fn = loader(src)
            kept, daily_list, union_mask = accumulate_daily_fields(
                gdates, load_fn=load_fn, mask_fn=mask_fn,
                use_mask=bool(getattr(eval_cfg, "eval_land_only", True))
            )
            if not daily_list:
                logger.warning("[spatial] No %s fields in group %s", src, gname)
                continue

            maps = compute_spatial_climatologies(
                daily_list,
                wet_thr_mm=wet_thr,
                percentiles=pct_list,
                rxk_days=rxk_days,
            )
            npz_path = tables_dir / f"spatial_{src}_{gname}.npz"
            save_maps_npz(npz_path, **maps)
            (tables_dir / f"spatial_{src}_{gname}.meta.txt").write_text(
                f"source={src}\n"
                f"group={gname}\n"
                f"n_days={len(kept)}\n"
                f"wet_thr_mm={wet_thr}\n"
                f"rxk_days={list(rxk_days)}\n"
                f"percentiles={list(pct_list)}\n"
            )
            group_maps[src] = maps

        # Summarize per-group maps into rows (metrics live in metrics_spatial)
        summary_rows.extend(summarize_group_spatial_maps(gname, group_maps))

    write_spatial_summary_csv(tables_dir, summary_rows)
    # Plotting
    if make_plots:
        try:
            from .plot_spatial import plot_spatial_maps
            plot_spatial_maps(out_root)
            logger.info("[spatial] Plots saved to %s", str(out_root / "figures"))
        except Exception as e:
            logger.warning(f"[spatial] Could not produce plots: {e}")