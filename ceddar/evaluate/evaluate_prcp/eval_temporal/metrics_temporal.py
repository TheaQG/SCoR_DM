from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
from dataclasses import dataclass
import numpy as np
import torch

# ---------- helpers ----------

def _to_hw(t: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(t):
        t = torch.as_tensor(t)
    t = t.float()
    if t.dim() == 4 and t.shape[:2] == (1, 1):
        t = t.squeeze(0).squeeze(0)
    elif t.dim() == 3 and t.shape[0] == 1:
        t = t.squeeze(0)
    if t.dim() != 2:
        t = t.reshape(t.shape[-2], t.shape[-1])
    return t

def _apply_mask(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return x
    m = mask
    if m.dtype != torch.bool:
        m = m > 0.5
    if m.dim() == 4 and m.shape[:2] == (1, 1):
        m = m.squeeze(0).squeeze(0)
    elif m.dim() == 3 and m.shape[0] == 1:
        m = m.squeeze(0)
    x = x.clone()
    x[~m] = float("nan")
    return x

def _nanmean_hw(x: torch.Tensor) -> float:
    x_np = x.detach().cpu().numpy()
    return float(np.nanmean(x_np))

# ---------- build daily domain-mean series (HR/PMM/LR) ----------

def build_domain_mean_series(
    dates: Sequence[str],
    resolver,
    *,
    use_mask: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Returns dict with keys present among {"HR","PMM","LR"} mapping to
    daily domain-mean series (np.ndarray, shape [T]) aligned to 'dates'.
    Missing days are NaN.
    """
    out: Dict[str, List[float]] = {"HR": [], "PMM": [], "LR": []}
    have = {"HR": False, "PMM": False, "LR": False}

    for d in dates:
        m = resolver.load_mask(d) if use_mask else None

        hr = resolver.load_obs(d)
        gen = resolver.load_pmm(d)
        lr  = resolver.load_lr(d)

        if hr is not None:
            hr = _apply_mask(_to_hw(hr), m)
            out["HR"].append(_nanmean_hw(hr)); have["HR"] = True
        else:
            out["HR"].append(np.nan)

        if gen is not None:
            gen = _apply_mask(_to_hw(gen), m)
            out["PMM"].append(_nanmean_hw(gen)); have["PMM"] = True
        else:
            out["PMM"].append(np.nan)

        if lr is not None:
            lr = _apply_mask(_to_hw(lr), m)
            out["LR"].append(_nanmean_hw(lr)); have["LR"] = True
        else:
            out["LR"].append(np.nan)

    out_np: Dict[str, np.ndarray] = {}
    for k in ["HR", "PMM", "LR"]:
        if have[k]:
            out_np[k] = np.asarray(out[k], dtype=np.float32)
    return out_np

# ---------- ensemble domain-mean series ----------

@dataclass
class EnsembleSeries:
    member_series: np.ndarray  # shape [M, T]
    mean: np.ndarray           # shape [T]
    std: np.ndarray            # shape [T]

def _ensure_member_stack(arr) -> Optional[np.ndarray]:
    if arr is None:
        return None
    if torch.is_tensor(arr):
        a = arr.detach().cpu()
        if a.dim() == 4 and a.shape[1] == 1:
            a = a.squeeze(1)
        if a.dim() == 3:
            return a.numpy()
        return None
    a = np.asarray(arr)
    if a.ndim == 4 and a.shape[1] == 1:
        a = np.squeeze(a, axis=1)
    if a.ndim == 3:
        return a
    return None

def build_domain_mean_series_ensemble(
    dates: Sequence[str],
    resolver,
    *,
    use_mask: bool = True,
    n_members: Optional[int] = None,
    seed: int = 1234,
) -> Optional[EnsembleSeries]:
    """Return per-member domain-mean time series across dates.
    Output shapes: member_series [M, T], mean [T], std [T]. Returns None if no ensemble available.
    """
    # Detect M on first available date
    M_detect = None
    for d in dates:
        ens = None
        for name in ("load_ens", "load_gen_members", "load_members", "load_ensemble"):
            fn = getattr(resolver, name, None)
            if callable(fn):
                try:
                    ens = fn(d)
                    if ens is not None:
                        break
                except Exception:
                    continue
        ens = _ensure_member_stack(ens)
        if ens is not None:
            M_detect = int(ens.shape[0]); break
    if not M_detect:
        return None

    rng = np.random.RandomState(seed)
    sel = np.arange(M_detect)
    if n_members is not None and int(n_members) < M_detect:
        sel = rng.choice(M_detect, size=int(n_members), replace=False)
    M = int(sel.size)

    series_list: List[List[float]] = [[] for _ in range(M)]

    for d in dates:
        m = resolver.load_mask(d) if use_mask else None
        ens = None
        for name in ("load_ens", "load_gen_members", "load_members", "load_ensemble"):
            fn = getattr(resolver, name, None)
            if callable(fn):
                try:
                    ens = fn(d)
                    if ens is not None:
                        break
                except Exception:
                    continue
        ens = _ensure_member_stack(ens)
        if ens is None:
            for i in range(M):
                series_list[i].append(np.nan)
            continue
        ens = ens[sel]
        if m is not None:
            m_t = torch.as_tensor(m).float()
            if m_t.dim() == 4 and m_t.shape[:2] == (1, 1):
                m_t = m_t.squeeze(0).squeeze(0)
            elif m_t.dim() == 3 and m_t.shape[0] == 1:
                m_t = m_t.squeeze(0)
            m_bool = (m_t > 0.5).numpy()
        else:
            m_bool = None
        for i in range(M):
            arr = ens[i]
            if m_bool is not None:
                arr = np.where(m_bool, arr, np.nan)
            series_list[i].append(float(np.nanmean(arr)))

    member_series = np.asarray(series_list, dtype=np.float64)  # [M, T]
    mean = np.nanmean(member_series, axis=0)
    std  = np.nanstd(member_series, axis=0)
    return EnsembleSeries(member_series=member_series, mean=mean, std=std)

# ---------- autocorrelation ----------

def autocorr(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Lag-1..max_lag autocorrelation with NaN-handling (pairwise complete)."""
    s = np.asarray(series, dtype=np.float64)
    mask = np.isfinite(s)
    if mask.sum() < 2:
        return np.full(max_lag, np.nan)
    mu = np.nanmean(s)
    s = s - mu
    var = np.nanvar(s)
    if not np.isfinite(var) or var == 0.0:
        return np.full(max_lag, np.nan)
    ac = np.full(max_lag, np.nan)
    for k in range(1, max_lag + 1):
        x = s[:-k]; y = s[k:]
        m = np.isfinite(x) & np.isfinite(y)
        ac[k - 1] = float(np.dot(x[m], y[m]) / (m.sum() * var)) if m.sum() >= 2 else np.nan
    return ac

def aggregate_autocorr_over_members(member_series: np.ndarray, max_lag: int) -> np.ndarray:
    """Return mean over members of each member's autocorr."""
    M = member_series.shape[0]
    ac_all = []
    for i in range(M):
        ac_all.append(autocorr(member_series[i], max_lag))
    ac_all = np.asarray(ac_all)
    return np.nanmean(ac_all, axis=0)

# ---------- wet/dry spell (Markov) ----------

def binarize_wet(series: np.ndarray, wet_thr_mm: float) -> np.ndarray:
    b = np.asarray(series, dtype=np.float64)
    return (b >= wet_thr_mm).astype(np.int8)  # 1=wet, 0=dry

def transition_matrix(b: np.ndarray) -> np.ndarray:
    """Return 2x2 matrix P where P[i,j] = P(state_t=j | state_{t-1}=i)."""
    P = np.zeros((2, 2), dtype=np.float64)
    valid = np.isfinite(b)
    x = b[valid].astype(np.int32)
    if x.size < 2:
        return np.full((2, 2), np.nan)
    for i in (0, 1):
        idx = np.where(x[:-1] == i)[0]
        if idx.size == 0:
            P[i, :] = np.nan
            continue
        nxt = x[idx + 1]
        P[i, 0] = np.mean(nxt == 0)
        P[i, 1] = np.mean(nxt == 1)
    return P

def spell_lengths(b: np.ndarray, state: int) -> np.ndarray:
    """Run-lengths for 'state' (0=dry, 1=wet)."""
    x = b.astype(np.int8)
    if x.size == 0:
        return np.zeros(0, dtype=np.int32)
    runs: List[int] = []
    cur = 0
    for v in x:
        if v == state:
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
            cur = 0
    if cur > 0:
        runs.append(cur)
    return np.asarray(runs, dtype=np.int32)

def spell_histogram(lengths: np.ndarray, max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (bins, pmf) where bins = 1..max_len and pmf sums to 1 over observed lengths."""
    if lengths.size == 0:
        bins = np.arange(1, max_len + 1, dtype=np.int32)
        return bins, np.full_like(bins, np.nan, dtype=np.float64)
    bins = np.arange(1, max_len + 1, dtype=np.int32)
    counts = np.zeros_like(bins, dtype=np.float64)
    for L in lengths:
        if 1 <= L <= max_len:
            counts[L - 1] += 1.0
    pmf = counts / counts.sum() if counts.sum() > 0 else np.full_like(counts, np.nan)
    return bins, pmf

def geometric_fit_from_P(P: np.ndarray, state: int) -> float:
    """
    Geometric parameter p for spell of 'state' given transition matrix.
    For wet spells (state=1): p = 1 - P[1,1]; for dry (state=0): p = 1 - P[0,0].
    """
    if not np.all(np.isfinite(P)):
        return np.nan
    stay = P[state, state]
    if not np.isfinite(stay):
        return np.nan
    p = 1.0 - stay
    return float(p) if 0.0 < p <= 1.0 else np.nan

def aggregate_spell_pmf_over_members(
    member_series: np.ndarray,
    wet_thr_mm: float,
    max_spell: int,
    mode: str = "member_mean",  # or "pool"
) -> dict:
    """Compute wet/dry spell PMFs across members."""
    M = member_series.shape[0]
    wet_bins = np.arange(1, max_spell + 1, dtype=np.int32)
    dry_bins = np.arange(1, max_spell + 1, dtype=np.int32)

    if mode == "pool":
        wet_all = []
        dry_all = []
        P_sum = np.zeros((2, 2), dtype=float)
        P_cnt = 0
        for i in range(M):
            b = binarize_wet(member_series[i], wet_thr_mm)
            wet_all.append(spell_lengths(b, 1))
            dry_all.append(spell_lengths(b, 0))
            P = transition_matrix(b)
            if np.all(np.isfinite(P)):
                P_sum += P; P_cnt += 1
        wet_L = np.concatenate([w for w in wet_all if w.size]) if any(w.size for w in wet_all) else np.zeros(0, dtype=int)
        dry_L = np.concatenate([w for w in dry_all if w.size]) if any(w.size for w in dry_all) else np.zeros(0, dtype=int)
        wb, wet_pmf = spell_histogram(wet_L, max_spell)
        db, dry_pmf = spell_histogram(dry_L, max_spell)
        P_bar = (P_sum / max(P_cnt, 1))
        return {
            "wet_bins": wb, "wet_pmf": wet_pmf, "wet_geom_p": geometric_fit_from_P(P_bar, 1),
            "dry_bins": db, "dry_pmf": dry_pmf, "dry_geom_p": geometric_fit_from_P(P_bar, 0),
        }

    # member_mean
    wet_pmfs = []; dry_pmfs = []; Ps = []
    for i in range(M):
        b = binarize_wet(member_series[i], wet_thr_mm)
        wb, wpmf = spell_histogram(spell_lengths(b, 1), max_spell)
        db, dpmf = spell_histogram(spell_lengths(b, 0), max_spell)
        wet_pmfs.append(wpmf); dry_pmfs.append(dpmf)
        P = transition_matrix(b)
        if np.all(np.isfinite(P)):
            Ps.append(P)
    wet_pmf = np.nanmean(np.asarray(wet_pmfs), axis=0)
    dry_pmf = np.nanmean(np.asarray(dry_pmfs), axis=0)
    P_bar = np.nanmean(np.asarray(Ps), axis=0) if Ps else np.full((2, 2), np.nan)
    return {
        "wet_bins": wet_bins, "wet_pmf": wet_pmf, "wet_geom_p": geometric_fit_from_P(P_bar, 1),
        "dry_bins": dry_bins, "dry_pmf": dry_pmf, "dry_geom_p": geometric_fit_from_P(P_bar, 0),
    }

# ---------- package for single streams ----------

def compute_temporal_metrics(
    series_dict: Dict[str, np.ndarray],
    *,
    wet_thr_mm: float = 1.0,
    max_lag: int = 30,
    max_spell: int = 30,
) -> Dict[str, dict]:
    """
    Returns nested dict keyed by {"HR","PMM","LR"} with:
      - "autocorr": [max_lag] array
      - "P": 2x2 transition matrix
      - "wet_bins","wet_pmf","wet_geom_p"
      - "dry_bins","dry_pmf","dry_geom_p"
    """
    out: Dict[str, dict] = {}
    for k, s in series_dict.items():
        d: dict = {}
        d["autocorr"] = autocorr(s, max_lag)

        b = binarize_wet(s, wet_thr_mm)
        P = transition_matrix(b)
        d["P"] = P

        wet_L = spell_lengths(b, 1)
        dry_L = spell_lengths(b, 0)
        wb, wpmf = spell_histogram(wet_L, max_spell)
        db, dpmf = spell_histogram(dry_L, max_spell)
        d["wet_bins"], d["wet_pmf"] = wb, wpmf
        d["dry_bins"], d["dry_pmf"] = db, dpmf

        d["wet_geom_p"] = geometric_fit_from_P(P, 1)
        d["dry_geom_p"] = geometric_fit_from_P(P, 0)
        out[k] = d
    return out