from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Dict, Any, List, Tuple

import numpy as np
import torch
import logging
from scipy.stats import genextreme as scipy_gev
from scipy.stats import genpareto as scipy_gpd

logger = logging.getLogger(__name__)

# ------------------ small utilities ------------------
def _ensure_float(t: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(t, np.ndarray):
        t = torch.from_numpy(t)
    return t if torch.is_floating_point(t) else t.float()

def _mask_to_hw(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    m = mask
    if m.dtype != torch.bool:
        m = m > 0.5
    if m.dim() == 4 and m.shape[:2] == (1, 1):
        m = m.squeeze(0).squeeze(0)
    elif m.dim() == 3 and m.shape[0] == 1:
        m = m.squeeze(0)
    return m

def to_daily_series_hwstack(fields: List[torch.Tensor],
                            mask_hw: Optional[torch.Tensor] = None,
                            agg: str = "mean") -> np.ndarray:
    xs = []
    for x in fields:
        x = _ensure_float(x).detach()
        if mask_hw is not None:
            vals = x[mask_hw]
            xs.append(vals.mean() if agg == "mean" else vals.sum() if vals.numel() else torch.tensor(float("nan")))
        else:
            xs.append(x.mean() if agg == "mean" else x.sum())
    return torch.stack(xs).cpu().numpy().astype(np.float64)

def parse_dates_to_np(dates: Sequence[str]) -> np.ndarray:
    out = []
    for s in dates:
        s = s.strip()
        if len(s) == 8 and s.isdigit():
            out.append(np.datetime64(f"{s[:4]}-{s[4:6]}-{s[6:8]}"))
        else:
            out.append(np.datetime64(s))
    return np.array(out, dtype="datetime64[D]")

def seasonal_block_index(dates_np: np.ndarray) -> np.ndarray:
    yy = dates_np.astype('datetime64[Y]').astype(int) + 1970
    mm = (dates_np.astype('datetime64[M]') - dates_np.astype('datetime64[Y]')).astype(int) + 1
    season = np.select(
        [np.isin(mm, [12,1,2]), np.isin(mm, [3,4,5]), np.isin(mm, [6,7,8]), np.isin(mm, [9,10,11])],
        [1, 2, 3, 4], default=0
    )
    year_adj = yy + (mm == 12).astype(int)
    return (year_adj * 10 + season).astype(int)

def rxk_from_series(series: np.ndarray, k: int, block_id: Optional[np.ndarray] = None) -> np.ndarray:
    s = np.asarray(series, dtype=float)
    if block_id is None:
        block_id = np.zeros_like(s, dtype=int)
    vals = []
    for b in np.unique(block_id):
        x = s[block_id == b]
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        if k <= 1 or x.size < k:
            vals.append(np.nanmax(x))
        else:
            roll = np.convolve(x, np.ones(int(k), dtype=float), mode="valid")
            vals.append(np.nanmax(roll) if roll.size else np.nanmax(x))
    return np.asarray(vals, dtype=float)

# ------------------ 1) GEV (block maxima) ------------------
def _gev_return_level(c, loc, scale, rp_years: float, blocks_per_year: float = 1.0) -> float:
    p = 1.0 - 1.0/(rp_years * blocks_per_year)
    return float(scipy_gev.ppf(p, c, loc=loc, scale=scale))

def fit_gev_block_maxima_with_ci(block_maxima: np.ndarray,
                                 rps_years: Sequence[float],
                                 blocks_per_year: float = 1.0,
                                 n_boot: int = 1000,
                                 seed: int = 42) -> Dict[str, Any]:
    x = np.asarray(block_maxima, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 8:
        raise ValueError("Too few block maxima to fit GEV reliably.")
    c, loc, scale = scipy_gev.fit(x)
    rls = np.array([_gev_return_level(c, loc, scale, rp, blocks_per_year) for rp in rps_years])

    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, len(rps_years)))
    for b in range(n_boot):
        xb = rng.choice(x, size=x.size, replace=True)
        try:
            cb, lb, sb = scipy_gev.fit(xb)
            boot[b, :] = [_gev_return_level(cb, lb, sb, rp, blocks_per_year) for rp in rps_years]
        except Exception:
            boot[b, :] = np.nan
    lo = np.nanpercentile(boot, 2.5, axis=0)
    hi = np.nanpercentile(boot, 97.5, axis=0)
    return {
        "n_blocks": int(x.size),
        "shape": float(c), "loc": float(loc), "scale": float(scale),
        "rps": np.array(rps_years, dtype=float),
        "rl": rls, "rl_lo": lo, "rl_hi": hi,
    }

# ------------------ 2) POT (GPD) ------------------
def fit_pot_gpd_with_ci(y_daily: np.ndarray,
                        threshold: float,
                        rps_years: Sequence[float],
                        days_per_year: float = 365.25,
                        n_boot: int = 1000,
                        seed: int = 42) -> Dict[str, Any]:
    y = np.asarray(y_daily, dtype=float)
    y = y[np.isfinite(y)]
    N = y.size
    exc = y[y > threshold] - threshold
    k = exc.size
    if k < 10:
        raise ValueError(f"Too few exceedances above u={threshold:.2f} (k={k}).")
    c, loc, scale = scipy_gpd.fit(exc, floc=0.0)
    xi, beta = c, scale
    lam = k / N

    def _rl(T):
        a = lam * T * days_per_year
        if np.isclose(xi, 0.0):
            return threshold + beta * np.log(a)
        return threshold + (beta/xi) * (np.power(a, xi) - 1.0)

    rls = np.array([_rl(rp) for rp in rps_years])

    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, len(rps_years)))
    for b in range(n_boot):
        yb = rng.choice(y, size=N, replace=True)
        excb = yb[yb > threshold] - threshold
        kb = excb.size
        if kb < 5:
            boot[b, :] = np.nan; continue
        try:
            cb, lb, sb = scipy_gpd.fit(excb, floc=0.0)
            xib, betab = cb, sb
            lam_b = kb / N
            vals = []
            for rp in rps_years:
                a = lam_b * rp * days_per_year
                if np.isclose(xib, 0.0):
                    rl = threshold + betab * np.log(a)
                else:
                    rl = threshold + (betab/xib) * (np.power(a, xib) - 1.0)
                vals.append(rl)
            boot[b, :] = vals
        except Exception:
            boot[b, :] = np.nan
    lo = np.nanpercentile(boot, 2.5, axis=0)
    hi = np.nanpercentile(boot, 97.5, axis=0)
    return {
        "u": float(threshold), "xi": float(xi), "beta": float(beta),
        "k_exc": int(k), "lambda_per_day": float(lam),
        "rps": np.array(rps_years, dtype=float), "rl": rls,
        "rl_lo": lo, "rl_hi": hi, "N_days": int(N),
    }

# ------------------ 3) Percentiles & wet-day frequency ------------------
def percentiles_and_wetfreq(y_daily: np.ndarray,
                            wet_thr: float = 1.0,
                            p_list: Sequence[float] = (95.0, 99.0)) -> Dict[str, Any]:
    y = np.asarray(y_daily, dtype=float)
    y = y[np.isfinite(y)]
    out = {f"P{int(p)}": float(np.percentile(y, p)) for p in p_list}
    out["wet_freq"] = float(np.mean(y > wet_thr))
    out["n_days"] = int(y.size)
    return out

# ------------------ High-level helpers ------------------
@dataclass
class SeriesBundle:
    dates: np.ndarray         # [T] datetime64[D]
    hr: np.ndarray            # [T]
    gen: np.ndarray           # [T]
    lr: Optional[np.ndarray]  # [T] or None

@dataclass
class EnsembleSeriesBundle:
    dates: np.ndarray     # [T]
    gen_members: np.ndarray  # [M,T]

def build_daily_series(resolver,
                       dates: Sequence[str],
                       *,
                       mask_hw: Optional[torch.Tensor],
                       agg: str = "mean",
                       include_lr: bool = True) -> SeriesBundle:
    mask_hw = _mask_to_hw(mask_hw)
    hr_fields, gen_fields, lr_fields, dates_collected = [], [], [], []
    for d in dates:
        hr = resolver.load_obs(d)
        gen = resolver.load_pmm(d)
        if hr is None or gen is None:
            continue
        if not torch.is_tensor(hr): hr = torch.from_numpy(np.asarray(hr))
        if not torch.is_tensor(gen): gen = torch.from_numpy(np.asarray(gen))
        hr_fields.append(hr.squeeze())
        gen_fields.append(gen.squeeze())
        dates_collected.append(d)
        if include_lr:
            try:
                lr = resolver.load_lr(d)
                if lr is not None:
                    if not torch.is_tensor(lr): lr = torch.from_numpy(np.asarray(lr))
                    while lr.dim() > 2 and lr.shape[0] == 1:
                        lr = lr.squeeze(0)
                    lr_fields.append(lr.squeeze())
                else:
                    lr_fields.append(None)
            except Exception as e:
                logger.error(f"[build_daily_series] Error loading LR for date {d}: {e}")
                lr_fields.append(None)

    # At this point, hr_fields, gen_fields, dates_collected are aligned, lr_fields is same length if include_lr else empty
    # Find keep_idx for which LR exists (not None) across all collected days
    keep_idx = []
    if include_lr and len(lr_fields) == len(hr_fields):
        for i, lf in enumerate(lr_fields):
            if lf is not None:
                keep_idx.append(i)

    orig_days = len(hr_fields)
    lr_days = len(keep_idx)

    if include_lr:
        if lr_days > 0:
            # Filter all fields and dates by keep_idx
            hr_fields_filt = [hr_fields[i] for i in keep_idx]
            gen_fields_filt = [gen_fields[i] for i in keep_idx]
            lr_fields_filt = [lr_fields[i] for i in keep_idx]
            dates_filt = [dates_collected[i] for i in keep_idx]
            hr_fields, gen_fields, lr_fields, dates_collected = hr_fields_filt, gen_fields_filt, lr_fields_filt, dates_filt
        else:
            lr_fields = []

    dt = parse_dates_to_np(dates_collected)
    hr_series  = to_daily_series_hwstack(hr_fields,  mask_hw, agg=agg)
    gen_series = to_daily_series_hwstack(gen_fields, mask_hw, agg=agg)
    lr_series = None
    if include_lr and len(lr_fields) == len(hr_fields) and len(lr_fields) > 0:
        lr_series = to_daily_series_hwstack(lr_fields, mask_hw, agg=agg)
        # Ensure alignment
        assert len(hr_series) == len(gen_series) == len(lr_series) == len(dt)
    else:
        # No LR: return None
        lr_series = None
        assert len(hr_series) == len(gen_series) == len(dt)

    logger.info(f"[build_daily_series] Collected {orig_days} days for HR/GEN; retained {lr_days} days for LR after alignment.")
    logger.debug(f"[build_daily_series] Final series lengths: HR={len(hr_series)}, GEN={len(gen_series)}, LR={(len(lr_series) if lr_series is not None else 0)}")

    return SeriesBundle(dates=dt, hr=hr_series, gen=gen_series, lr=lr_series)

def build_daily_series_ensemble(resolver,
                                dates: Sequence[str],
                                *,
                                mask_hw: Optional[torch.Tensor],
                                agg: str = "mean",
                                n_members: Optional[int] = None,
                                seed: int = 1234) -> Optional[EnsembleSeriesBundle]:
    mask_hw = _mask_to_hw(mask_hw)
    M_eff = None
    # probe first date to get M
    for d in dates:
        try:
            sample = resolver.fetch(d, want_ensemble=True, n_members=n_members, seed=seed)  # type: ignore[attr-defined]
            ens = getattr(sample, "ens", None)
        except Exception:
            ens = resolver.load_ens(d)
        if ens is not None:
            M_eff = int(ens.shape[0]) if torch.is_tensor(ens) else int(np.asarray(ens).shape[0])
            break
    if M_eff is None:
        return None
    series_by_member: List[List[float]] = [list() for _ in range(M_eff)]
    valid_dates: List[str] = []
    for d in dates:
        try:
            sample = resolver.fetch(d, want_ensemble=True, n_members=n_members, seed=seed)  # type: ignore[attr-defined]
            ens = getattr(sample, "ens", None)
        except Exception:
            ens = resolver.load_ens(d)
        if ens is None:
            continue
        if torch.is_tensor(ens):
            ens_np = ens.detach().cpu().numpy()
        else:
            ens_np = np.asarray(ens)
        # shape [M,H,W]
        if mask_hw is not None:
            m = mask_hw.detach().cpu().numpy().astype(bool)
            vals = [ens_np[i][m] for i in range(min(M_eff, ens_np.shape[0]))]
        else:
            vals = [ens_np[i].reshape(-1) for i in range(min(M_eff, ens_np.shape[0]))]
        for i, v in enumerate(vals):
            if agg == "mean":
                series_by_member[i].append(float(np.nanmean(v)))
            else:
                series_by_member[i].append(float(np.nansum(v)))
        valid_dates.append(d)
    if not valid_dates:
        return None
    dt = parse_dates_to_np(valid_dates)
    gen_members = np.array([np.array(s, dtype=np.float64) for s in series_by_member], dtype=np.float64)
    return EnsembleSeriesBundle(dates=dt, gen_members=gen_members)

def pooled_pixel_percentiles_and_wetfreq(
        resolver, dates, mask_hw=None, wet_thr=1.0, p_list=(95.0, 99.0), include_lr=True):
    import torch
    mask_hw = _mask_to_hw(mask_hw)
    acc = {"HR": [], "GEN": [], "LR": []}
    for d in dates:
        hr = resolver.load_obs(d)
        gen = resolver.load_pmm(d)
        if hr is None or gen is None:
            continue
        if not torch.is_tensor(hr): hr = torch.from_numpy(np.asarray(hr))
        if not torch.is_tensor(gen): gen = torch.from_numpy(np.asarray(gen))
        if mask_hw is not None:
            hr = hr[mask_hw]; gen = gen[mask_hw]
        acc["HR"].append(hr.reshape(-1).cpu().numpy())
        acc["GEN"].append(gen.reshape(-1).cpu().numpy())
        if include_lr:
            lr = resolver.load_lr(d)
            if lr is not None:
                if not torch.is_tensor(lr): lr = torch.from_numpy(np.asarray(lr))
                while lr.dim() > 2 and lr.shape[0] == 1: lr = lr.squeeze(0)
                if mask_hw is not None: lr = lr[mask_hw]
                acc["LR"].append(lr.reshape(-1).cpu().numpy())

    out = {}
    for key, chunks in acc.items():
        if key == "LR" and not chunks:  # no LR available
            continue
        x = np.concatenate(chunks).astype(float)
        x = x[np.isfinite(x)]
        out[key] = {
            **{f"P{int(p)}": float(np.percentile(x, p)) for p in p_list},
            "wet_freq": float(np.mean(x > wet_thr)),
            "n_points": int(x.size),
        }
    return out

def pooled_pixel_percentiles_and_wetfreq_ens(resolver, dates, mask_hw=None, wet_thr=1.0, p_list=(95.0,99.0), n_members=None, seed=1234):
    import torch
    mask_hw = _mask_to_hw(mask_hw)
    chunks = []
    for d in dates:
        try:
            sample = resolver.fetch(d, want_ensemble=True, n_members=n_members, seed=seed)  # type: ignore[attr-defined]
            ens = getattr(sample, "ens", None)
        except Exception:
            ens = resolver.load_ens(d)
        if ens is None:
            continue
        if torch.is_tensor(ens):
            arr = ens.detach().cpu().numpy()
        else:
            arr = np.asarray(ens)
        # [M,H,W] -> pool all member pixels
        if mask_hw is not None:
            m = mask_hw.detach().cpu().numpy().astype(bool)
            for i in range(arr.shape[0]):
                chunks.append(arr[i][m].reshape(-1))
        else:
            for i in range(arr.shape[0]):
                chunks.append(arr[i].reshape(-1))
    if not chunks:
        return None
    x = np.concatenate(chunks).astype(float)
    x = x[np.isfinite(x)]
    out = {**{f"P{int(p)}": float(np.percentile(x, p)) for p in p_list},
           "wet_freq": float(np.mean(x > wet_thr)),
           "n_points": int(x.size)}
    return out

def pooled_wet_hit_rate(resolver, dates, mask_hw=None, wet_thr=1.0, include_lr=True):
    """
    Compute a *proper* pooled wet-day hit-rate using pixel-level HR-wet masks per day.

    Returns
    -------
    dict : {
       "GEN": float or np.nan,
       "LR":  float or np.nan,
       "denom_total": int   # total count of HR-wet pixels across all days
    }
    """
    import torch
    mask_hw = _mask_to_hw(mask_hw)
    num_gen = 0
    num_lr = 0
    denom_total = 0
    have_lr_any = False

    for d in dates:
        hr = resolver.load_obs(d)
        gen = resolver.load_pmm(d)
        if hr is None or gen is None:
            continue
        if not torch.is_tensor(hr): hr = torch.from_numpy(np.asarray(hr))
        if not torch.is_tensor(gen): gen = torch.from_numpy(np.asarray(gen))

        # Squeeze possible singleton dims
        hr = hr.squeeze()
        gen = gen.squeeze()

        if include_lr:
            lr = resolver.load_lr(d)
            if lr is not None:
                if not torch.is_tensor(lr): lr = torch.from_numpy(np.asarray(lr))
                while lr.dim() > 2 and lr.shape[0] == 1:
                    lr = lr.squeeze(0)
                lr = lr.squeeze()
                have_lr_any = True
            else:
                lr = None
        else:
            lr = None

        if mask_hw is not None:
            hr_m = hr[mask_hw]
            gen_m = gen[mask_hw]
            lr_m  = (lr[mask_hw] if (lr is not None) else None)
        else:
            hr_m, gen_m, lr_m = hr.reshape(-1), gen.reshape(-1), (lr.reshape(-1) if lr is not None else None)

        # Build per-day HR-wet mask
        hr_wet = (hr_m > wet_thr)
        denom = int(hr_wet.sum().item()) if torch.is_tensor(hr_wet) else int(hr_wet.sum())
        if denom == 0:
            continue

        denom_total += denom

        # Count hits for GEN
        gen_hits = int(((gen_m > wet_thr) & hr_wet).sum().item()) if torch.is_tensor(gen_m) else int(((gen_m > wet_thr) & hr_wet).sum())
        num_gen += gen_hits

        # Count hits for LR if available
        if lr_m is not None:
            lr_hits = int(((lr_m > wet_thr) & hr_wet).sum().item())
            num_lr += lr_hits

    out = {
        "GEN": (float(num_gen) / denom_total) if denom_total > 0 else float("nan"),
        "denom_total": int(denom_total),
    }
    if have_lr_any:
        out["LR"] = (float(num_lr) / denom_total) if denom_total > 0 else float("nan")
    else:
        out["LR"] = np.nan
    return out


def pooled_wet_hit_rate_ens(resolver, dates, mask_hw=None, wet_thr=1.0, n_members=None, seed=1234):
    import torch
    mask_hw = _mask_to_hw(mask_hw)
    num = 0.0
    denom_total = 0
    for d in dates:
        hr = resolver.load_obs(d)
        try:
            sample = resolver.fetch(d, want_ensemble=True, n_members=n_members, seed=seed)  # type: ignore[attr-defined]
            ens = getattr(sample, "ens", None)
        except Exception:
            ens = resolver.load_ens(d)
        if hr is None or ens is None:
            continue
        if not torch.is_tensor(hr): hr = torch.from_numpy(np.asarray(hr))
        hr = hr.squeeze()
        if torch.is_tensor(ens): arr = ens.detach().cpu().numpy()
        else: arr = np.asarray(ens)
        # mask
        if mask_hw is not None:
            m = mask_hw.detach().cpu().numpy().astype(bool)
            hr_m = hr[m].detach().cpu().numpy()
            arr = np.stack([arr[i][m] for i in range(arr.shape[0])], axis=0)  # [M,N]
        else:
            hr_m = hr.reshape(-1).detach().cpu().numpy()
            arr = arr.reshape(arr.shape[0], -1)
        hr_wet = (hr_m > wet_thr)
        denom = int(hr_wet.sum())
        if denom == 0:
            continue
        # expected hits across members: average indicator over members
        prob_wet = (arr > wet_thr).mean(axis=0)  # [N]
        # expected hits are prob_wet when HR is wet
        num += float(prob_wet[hr_wet].sum())
        denom_total += denom
    if denom_total == 0:
        return float("nan")
    return float(num / denom_total)


# ------------------ Helper: per-member wet-hit rate stats (mean, std) ------------------
def pooled_wet_hit_rate_ens_member_stats(resolver, dates, mask_hw=None, wet_thr=1.0, n_members=None, seed=1234):
    """Return (mean, std) of per-member wet-hit rates vs HR across all dates."""
    import torch
    mask_hw = _mask_to_hw(mask_hw)
    # determine M
    M_eff = None
    for d in dates:
        hr = resolver.load_obs(d)
        try:
            sample = resolver.fetch(d, want_ensemble=True, n_members=n_members, seed=seed)  # type: ignore[attr-defined]
            ens = getattr(sample, "ens", None)
        except Exception:
            ens = resolver.load_ens(d)
        if hr is None or ens is None:
            continue
        if torch.is_tensor(ens):
            M_eff = int(ens.shape[0])
        else:
            M_eff = int(np.asarray(ens).shape[0])
        break
    if M_eff is None:
        return float("nan"), float("nan")
    per_member_hits = np.zeros((M_eff,), dtype=float)
    per_member_den  = np.zeros((M_eff,), dtype=float)
    for d in dates:
        hr = resolver.load_obs(d)
        try:
            sample = resolver.fetch(d, want_ensemble=True, n_members=n_members, seed=seed)  # type: ignore[attr-defined]
            ens = getattr(sample, "ens", None)
        except Exception:
            ens = resolver.load_ens(d)
        if hr is None or ens is None:
            continue
        if not torch.is_tensor(hr): hr = torch.from_numpy(np.asarray(hr))
        hr = hr.squeeze()
        if torch.is_tensor(ens): arr = ens.detach().cpu().numpy()
        else: arr = np.asarray(ens)
        if mask_hw is not None:
            m = mask_hw.detach().cpu().numpy().astype(bool)
            hr_m = hr[m].detach().cpu().numpy()
            arr = np.stack([arr[i][m] for i in range(arr.shape[0])], axis=0)
        else:
            hr_m = hr.reshape(-1).detach().cpu().numpy()
            arr = arr.reshape(arr.shape[0], -1)
        hr_wet = (hr_m > wet_thr)
        denom = int(hr_wet.sum())
        if denom == 0:
            continue
        per_member_den += denom
        for mi in range(min(M_eff, arr.shape[0])):
            hits = int(((arr[mi] > wet_thr) & hr_wet).sum())
            per_member_hits[mi] += float(hits)
    rates = np.divide(per_member_hits, per_member_den, out=np.full_like(per_member_hits, np.nan), where=per_member_den>0)
    rates = rates[np.isfinite(rates)]
    if rates.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(rates)), float(np.std(rates))

# ------------------ Member-mean pooled tails (percentiles/wet freq and wet-hit rate) ------------------
def percentiles_and_wetfreq_ens_member_mean(resolver, dates, mask_hw=None, wet_thr=1.0, p_list=(95.0,99.0), n_members=None, seed=1234):
    """Compute per-member pooled-pixel tails, then average across members.
    Returns dict with mean P95/P99, mean wet_freq, and an approximate mean n_points."""
    import torch
    mask_hw = _mask_to_hw(mask_hw)
    # accumulate per member lists of pixel values across all dates
    member_vals: list[list[np.ndarray]] = []
    M_eff = None
    # probe M
    for d in dates:
        try:
            sample = resolver.fetch(d, want_ensemble=True, n_members=n_members, seed=seed)  # type: ignore[attr-defined]
            ens = getattr(sample, "ens", None)
        except Exception:
            ens = resolver.load_ens(d)
        if ens is not None:
            M_eff = int(ens.shape[0]) if torch.is_tensor(ens) else int(np.asarray(ens).shape[0])
            break
    if M_eff is None:
        return None
    member_vals = [[] for _ in range(M_eff)]
    for d in dates:
        try:
            sample = resolver.fetch(d, want_ensemble=True, n_members=n_members, seed=seed)  # type: ignore[attr-defined]
            ens = getattr(sample, "ens", None)
        except Exception:
            ens = resolver.load_ens(d)
        if ens is None:
            continue
        if torch.is_tensor(ens):
            arr = ens.detach().cpu().numpy()
        else:
            arr = np.asarray(ens)
        if mask_hw is not None:
            m = mask_hw.detach().cpu().numpy().astype(bool)
            for i in range(min(M_eff, arr.shape[0])):
                member_vals[i].append(arr[i][m].reshape(-1))
        else:
            for i in range(min(M_eff, arr.shape[0])):
                member_vals[i].append(arr[i].reshape(-1))
    # compute per-member statistics
    p95s, p99s, wets, ns = [], [], [], []
    for i in range(M_eff):
        if not member_vals[i]:
            continue
        x = np.concatenate(member_vals[i]).astype(float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        p95s.append(np.percentile(x, 95))
        p99s.append(np.percentile(x, 99))
        wets.append(np.mean(x > wet_thr))
        ns.append(int(x.size))
    if not p95s:
        return None
    out = {
        "P95": float(np.mean(p95s)),
        "P99": float(np.mean(p99s)),
        "wet_freq": float(np.mean(wets)),
        "n_points": int(np.mean(ns) if ns else 0),
        # optional spread (not consumed by plots but useful for debugging)
        "P95_std": float(np.std(p95s)),
        "P99_std": float(np.std(p99s)),
        "wet_freq_std": float(np.std(wets)),
    }
    return out

def pooled_wet_hit_rate_ens_member_mean(resolver, dates, mask_hw=None, wet_thr=1.0, n_members=None, seed=1234):
    """Compute wet-hit rate per member vs HR, then average across members."""
    import torch
    mask_hw = _mask_to_hw(mask_hw)
    per_member_hits = []
    per_member_denoms = []
    # first, collect per-member numerator and denominator over all dates
    # We'll do it in two passes per date to avoid big memory spikes
    # Layout: for each member, accumulate sum of hits and sum of HR-wet counts
    # Determine M
    M_eff = None
    for d in dates:
        hr = resolver.load_obs(d)
        try:
            sample = resolver.fetch(d, want_ensemble=True, n_members=n_members, seed=seed)  # type: ignore[attr-defined]
            ens = getattr(sample, "ens", None)
        except Exception:
            ens = resolver.load_ens(d)
        if hr is None or ens is None:
            continue
        if torch.is_tensor(ens):
            M_eff = int(ens.shape[0])
        else:
            M_eff = int(np.asarray(ens).shape[0])
        break
    if M_eff is None:
        return float("nan")
    per_member_hits = [0.0 for _ in range(M_eff)]
    per_member_denoms = [0 for _ in range(M_eff)]
    for d in dates:
        hr = resolver.load_obs(d)
        try:
            sample = resolver.fetch(d, want_ensemble=True, n_members=n_members, seed=seed)  # type: ignore[attr-defined]
            ens = getattr(sample, "ens", None)
        except Exception:
            ens = resolver.load_ens(d)
        if hr is None or ens is None:
            continue
        if not torch.is_tensor(hr): hr = torch.from_numpy(np.asarray(hr))
        hr = hr.squeeze()
        if torch.is_tensor(ens): arr = ens.detach().cpu().numpy()
        else: arr = np.asarray(ens)
        # mask
        if mask_hw is not None:
            m = mask_hw.detach().cpu().numpy().astype(bool)
            hr_m = hr[m].detach().cpu().numpy()
            arr = np.stack([arr[i][m] for i in range(arr.shape[0])], axis=0)  # [M,N]
        else:
            hr_m = hr.reshape(-1).detach().cpu().numpy()
            arr = arr.reshape(arr.shape[0], -1)
        hr_wet = (hr_m > wet_thr)
        denom = int(hr_wet.sum())
        if denom == 0:
            continue
        per_member_denoms = [d0 + denom for d0 in per_member_denoms]
        for mi in range(min(M_eff, arr.shape[0])):
            hits = int(((arr[mi] > wet_thr) & hr_wet).sum())
            per_member_hits[mi] += float(hits)
    # per-member rates then mean across members
    rates = [ (per_member_hits[i] / per_member_denoms[i]) if per_member_denoms[i] > 0 else np.nan for i in range(M_eff) ]
    rates = np.array(rates, dtype=float)
    rates = rates[np.isfinite(rates)]
    if rates.size == 0:
        return float("nan")
    return float(np.mean(rates))