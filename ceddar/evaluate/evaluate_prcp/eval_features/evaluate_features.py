# sbgm/evaluate/evaluate_prcp/eval_features/evaluate_features.py
from __future__ import annotations
from pathlib import Path
from typing import Sequence, Dict, List, Optional
import numpy as np
import logging
import torch

from evaluate.evaluate_prcp.eval_features.metrics_features import compute_sal, compute_sal_object
from evaluate.evaluate_prcp.eval_features.plot_features import plot_features_all

logger = logging.getLogger(__name__)


def _year_of(d: str) -> int:
    s = d.strip()
    if len(s) == 8 and s.isdigit():
        return int(s[:4])
    # Fallback for ISO-like strings "YYYY-MM-DD"
    return int(s[:4])


def _season_of(d: str) -> str:
    s = d.strip()
    if len(s) == 8 and s.isdigit():
        m = int(s[4:6])
    else:
        m = int(s[5:7])
    if m in (12, 1, 2):
        return "DJF"
    if m in (3, 4, 5):
        return "MAM"
    if m in (6, 7, 8):
        return "JJA"
    return "SON"


def _group_dates(dates: List[str], group_by: str, seasons: Sequence[str]) -> Dict[str, List[str]]:
    if group_by == "all":
        return {"ALL": dates}
    if group_by == "season":
        out: Dict[str, List[str]] = {s: [] for s in seasons}
        for d in dates:
            s = _season_of(d)
            if s in out:
                out[s].append(d)
        return {s: out.get(s, []) for s in seasons if s in out}
    # default: year
    buckets: Dict[str, List[str]] = {}
    for d in dates:
        y = _year_of(d)
        buckets.setdefault(str(y), []).append(d)
    return dict(sorted(buckets.items()))


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _to_hw_np(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        arr = x.detach().cpu().float().squeeze().numpy()
    else:
        arr = np.asarray(x).squeeze()
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[-2], arr.shape[-1])
    return arr


def _apply_mask(arr: np.ndarray | None, mask) -> np.ndarray | None:
    if arr is None:
        return None
    if mask is None:
        return arr
    m = _to_hw_np(mask)
    if m is None:
        return arr
    m = m > 0.5
    out = arr.copy()
    out[~m] = np.nan
    return out


def _mean_map(resolver, dates, src: str, use_mask: bool = True):
    if src == "HR":
        loader = resolver.load_obs
    elif src in ("GEN", "PMM"):
        loader = resolver.load_pmm
    elif src == "LR":
        loader = resolver.load_lr
    else:
        raise ValueError(f"Unknown source: {src}")
    acc = []
    for d in dates:
        x = loader(d)
        if x is None:
            continue
        x = _to_hw_np(x)
        m = resolver.load_mask(d) if use_mask else None
        x = _apply_mask(x, m)
        acc.append(x)
    if not acc:
        return None
    with np.errstate(all="ignore"):
        return np.nanmean(np.stack(acc, axis=0), axis=0)


def _mean_maps_ensemble(
    resolver,
    dates,
    *,
    use_mask: bool = True,
    n_members: Optional[int] = None,
    seed: int = 1234,
):
    """
    Build per-member mean maps over the given dates. Returns [M, H, W] or None.
    """
    # Probe effective ensemble size
    M_eff = None
    first_mask = None
    for d in dates:
        try:
            s = resolver.fetch(d, want_ensemble=True, n_members=n_members, seed=seed)  # type: ignore[attr-defined]
            ens = getattr(s, "ens", None)
            msk = getattr(s, "mask", None)
        except Exception:
            ens = resolver.load_ens(d)
            msk = resolver.load_mask(d)
        if ens is None:
            continue
        M_eff = int(ens.shape[0]) if torch.is_tensor(ens) else int(np.asarray(ens).shape[0])
        first_mask = msk
        break
    if M_eff is None:
        return None

    per_member: list[list[np.ndarray]] = [[] for _ in range(M_eff)]
    for d in dates:
        try:
            s = resolver.fetch(d, want_ensemble=True, n_members=n_members, seed=seed)  # type: ignore[attr-defined]
            ens = getattr(s, "ens", None)
            msk = getattr(s, "mask", None)
        except Exception:
            ens = resolver.load_ens(d)
            msk = resolver.load_mask(d)
        if ens is None:
            continue
        arr = ens.detach().cpu().numpy() if torch.is_tensor(ens) else np.asarray(ens)
        m_np = None
        if use_mask:
            m_np = _to_hw_np(msk)
            if m_np is not None:
                m_np = (m_np > 0.5).astype(bool)
        m_here = min(M_eff, arr.shape[0])
        for mi in range(m_here):
            fld = _to_hw_np(arr[mi])
            if fld is None:
                continue
            if m_np is not None:
                try:
                    fld = fld.copy()
                    fld[~m_np] = np.nan
                except Exception:
                    f = fld.reshape(-1)
                    mf = m_np.reshape(-1)
                    f[~mf] = np.nan
                    fld = f.reshape(fld.shape)
            per_member[mi].append(fld)

    out = []
    for mi in range(M_eff):
        if not per_member[mi]:
            mm = _to_hw_np(first_mask)
            if mm is not None:
                out.append(np.full(mm.shape, np.nan, dtype=float))
            continue
        with np.errstate(all="ignore"):
            out.append(np.nanmean(np.stack(per_member[mi], axis=0), axis=0))
    if not out:
        return None
    return np.stack(out, axis=0)


def run_features(
    resolver,
    eval_cfg,
    out_root: str | Path,
    *,
    group_by: str = "year",
    seasons: Sequence[str] = ("ALL", "DJF", "MAM", "JJA", "SON"),
    make_plots: bool = True,
    plot_only: bool = False,
) -> None:
    """
    Main entry point for feature/object-based evaluation (SAL).
    """
    out_root = Path(out_root)
    tables_dir = _ensure_dir(out_root / "tables")
    figs_dir = _ensure_dir(out_root / "figures")

    include_lr = bool(getattr(eval_cfg, "feat_include_lr", True))

    # All available dates and grouping
    all_dates: List[str] = list(resolver.list_dates())
    if not all_dates:
        logger.warning("[evaluate_features] No dates found — aborting.")
        return
    groups = _group_dates(all_dates, group_by, seasons)

    # Plot-only mode
    if plot_only:
        sal_files = list(tables_dir.glob("sal_*.npz"))
        if not sal_files:
            logger.warning("[evaluate_features] No existing SAL tables found for plot_only=True.")
            return
        for f in sal_files:
            gname = f.stem.replace("sal_", "")
            data = dict(np.load(f, allow_pickle=True))
            plot_features_all(figs_dir, gname, data)
        return

    land_only = bool(getattr(eval_cfg, "eval_land_only", True))
    use_ens = bool(getattr(eval_cfg, "use_ensemble", False))
    n_members = getattr(eval_cfg, "ensemble_n_members", None)
    ens_seed = int(getattr(eval_cfg, "ensemble_member_seed", 1234))

    for gname, gdates in groups.items():
        if not gdates:
            continue
        hr_map = _mean_map(resolver, gdates, "HR", use_mask=land_only)
        gen_map = _mean_map(resolver, gdates, "GEN", use_mask=land_only)
        lr_map = _mean_map(resolver, gdates, "LR", use_mask=land_only) if include_lr else None

        if hr_map is None or gen_map is None:
            logger.warning("[evaluate_features] Group %s: missing HR/GEN data; skipping.", gname)
            continue

        mode = getattr(eval_cfg, "sal_structure_mode", "std_proxy")  # "std_proxy" or "object"
        logger.info("[evaluate_features] Group %s: computing SAL (%s mode)...", gname, mode)

        # helper so PMM and ensemble use the SAME kernel + parameters
        def _sal_call(hr_map, tst_map, lr_map=None):
            if mode == "object":
                return compute_sal_object(
                    hr_map, tst_map, lr_map,
                    threshold_kind=getattr(eval_cfg, "sal_threshold_kind", "quantile"),
                    threshold_value=float(getattr(eval_cfg, "sal_threshold_value", 0.90)),
                    connectivity=int(getattr(eval_cfg, "sal_connectivity", 8)),
                    min_area_px=int(getattr(eval_cfg, "sal_min_area_px", 9)),
                    smooth_sigma=getattr(eval_cfg, "sal_smooth_sigma", None),
                    peakedness=getattr(eval_cfg, "sal_peakedness_mode", "largest"),
                )
            else:
                return compute_sal(hr_map, tst_map, lr_map)

        # --- PMM / LR metrics ---
        sal_metrics = _sal_call(hr_map, gen_map, lr_map)

        # --- Ensemble metrics (per-member mean maps vs HR) ---
        if use_ens:
            try:
                ens_maps = _mean_maps_ensemble(
                    resolver,
                    gdates,
                    use_mask=land_only,
                    n_members=n_members,
                    seed=ens_seed,
                )
            except Exception:
                ens_maps = None

            if ens_maps is not None and ens_maps.size:
                A_arr, S_arr, L_arr, SAL_arr = [], [], [], []
                for mi in range(ens_maps.shape[0]):
                    pair = _sal_call(hr_map, ens_maps[mi], None).get("GEN_vs_HR", {})
                    a = float(pair.get("A", np.nan))
                    s = float(pair.get("S", np.nan))
                    l = float(pair.get("L", np.nan))
                    A_arr.append(a)
                    S_arr.append(s)
                    L_arr.append(l)
                    SAL_arr.append(float(np.sqrt(a*a + s*s + (l*l if np.isfinite(l) else 0.0))))
                A_arr = np.array(A_arr, dtype=float)
                S_arr = np.array(S_arr, dtype=float)
                L_arr = np.array(L_arr, dtype=float)
                SAL_arr = np.array(SAL_arr, dtype=float)
                sal_metrics.update(
                    {
                        "GEN_ENS_vs_HR": {
                            "A": float(np.nanmean(A_arr)),
                            "S": float(np.nanmean(S_arr)),
                            "L": float(np.nanmean(L_arr)),
                            "SAL": float(np.nanmean(SAL_arr)),
                        },
                        "GEN_ENS_std": {
                            "A": float(np.nanstd(A_arr)),
                            "S": float(np.nanstd(S_arr)),
                            "L": float(np.nanstd(L_arr)),
                            "SAL": float(np.nanstd(SAL_arr)),
                        },
                        "GEN_ENS_q10": {
                            "A": float(np.nanpercentile(A_arr, 10)),
                            "S": float(np.nanpercentile(S_arr, 10)),
                            "L": float(np.nanpercentile(L_arr, 10)),
                            "SAL": float(np.nanpercentile(SAL_arr, 10)),
                        },
                        "GEN_ENS_q90": {
                            "A": float(np.nanpercentile(A_arr, 90)),
                            "S": float(np.nanpercentile(S_arr, 90)),
                            "L": float(np.nanpercentile(L_arr, 90)),
                            "SAL": float(np.nanpercentile(SAL_arr, 90)),
                        },
                    }
                )

        # Ensure values are numpy arrays to satisfy numpy.savez_compressed requirements
        save_dict = {k: np.asarray(v) for k, v in sal_metrics.items()}
        np.savez_compressed(tables_dir / f"sal_{gname}.npz", **save_dict)

        if make_plots:
            plot_features_all(figs_dir, gname, sal_metrics)