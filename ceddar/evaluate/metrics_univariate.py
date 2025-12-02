"""
Univariate metrics for ensemble forecasts.
Contains:
    - PIT values and rank histograms from ensembles
    - Extremes: Rx1day/Rx5day, GEV fit to block maxima, POT with GPD fit, with bootstrap CIs
    - Reliability diagrams for threshold exceedance
    - Spread-skill diagnostic
    - Probability-Matched Mean (PMM) for univariate fields
    - CRPS for ensembles
    - Isotropic PSD computation (2D FFT and radial averaging)
    
To implement:
    - Full pixel distributions with distribution based metrics (Wasserstein, KL divergence, KS, etc.)
    - Yearly metrics map (e.g., annual average, sum, Rx1day, Rx5day, R95p, R99p, etc.)


"""

import torch
import logging
import csv
import numpy as np
from typing import Optional, Tuple, Dict, Sequence, Deque, DefaultDict, Any, cast
from dataclasses import dataclass, field
from collections import defaultdict, deque
from numpy.typing import NDArray
from pathlib import Path

# Extremes needs SciPy
from scipy.stats import genextreme as scipy_gev
from scipy.stats import genpareto as scipy_gpd
from scipy.stats import wasserstein_distance as _wasserstein
from scipy.stats import ks_2samp as _ks2
from scipy.stats import entropy as _kl_entropy

logger = logging.getLogger(__name__)

# =========================
# Helpers (Torch)
# =========================

def _ensure_float(t: torch.Tensor) -> torch.Tensor:
    return t if torch.is_floating_point(t) else t.float()

def _broadcast_mask(mask: Optional[torch.Tensor], target: torch.Tensor) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if mask.dtype != torch.bool:
        mask = mask > 0.5
    while mask.dim() < target.dim():
        mask = mask.unsqueeze(1)
    return mask.expand_as(target)

# =========================
# PIT & Rank histograms
# =========================
@torch.no_grad()
def pit_values_from_ensemble(
    obs: torch.Tensor,     # [B,H,W]
    ens: torch.Tensor,     # [B,M,H,W]
    mask: Optional[torch.Tensor] = None,
    randomized: bool = True,
) -> torch.Tensor:
    """
    Randomized PIT for empirical ensemble CDF (good for mixed discrete-continuous precip).
    For each (b,i,j): PIT = F^-(y-) + U * (F(y) - F^-(y-)), where F is empirical CDF from ensemble.
    Returns: 1-D tensor of PIT values over all valid pixels.
    """
    obs = _ensure_float(obs)
    ens = _ensure_float(ens)
    B, M, H, W = ens.shape

    if mask is not None:
        m = _broadcast_mask(mask, obs.unsqueeze(1))
        if m is not None:
            m = m.squeeze(1)  # [B,H,W]
        else:
            m = torch.ones_like(obs, dtype=torch.bool)
    else:
        m = torch.ones_like(obs, dtype=torch.bool)

    obs_flat = obs[m]                                         # [N]
    ens_flat = ens.permute(0,2,3,1)[m]                        # [N, M]

    # Sort ensemble per location
    ens_sorted, _ = torch.sort(ens_flat, dim=1)               # [N, M]

    # counts strictly less and equal
    less = (ens_sorted < obs_flat.unsqueeze(1)).sum(dim=1)    # [N]
    equal = (ens_sorted == obs_flat.unsqueeze(1)).sum(dim=1)  # [N]

    if randomized:
        # U in [0,1)
        U = torch.rand_like(obs_flat)
        pit = (less.float() + U * equal.float()) / float(M)
    else:
        pit = (less.float() + 0.5 * equal.float()) / float(M)   

    return pit  # [N]

@torch.no_grad()
def rank_histogram(
    obs: torch.Tensor,     # [B,H,W]
    ens: torch.Tensor,     # [B,M,H,W]
    mask: Optional[torch.Tensor] = None,
    randomize_ties: bool = True,
) -> torch.Tensor:
    """
    Rank histogram counts over M+1 bins (ranks 0..M).
    For ties, we either randomize the rank uniformly over the tied interval (recommended),
    or place at mid-rank.
    Returns: counts [M+1] as float tensor (not normalized).
    """
    obs = _ensure_float(obs)
    ens = _ensure_float(ens)
    B, M, H, W = ens.shape

    if mask is not None:
        m = _broadcast_mask(mask, obs.unsqueeze(1))
        if m is not None:
            m = m.squeeze(1)  # [B,H,W]
        else:
            m = torch.ones_like(obs, dtype=torch.bool)
    else:
        m = torch.ones_like(obs, dtype=torch.bool)

    obs_flat = obs[m]                           # [N]
    ens_flat = ens.permute(0,2,3,1)[m]          # [N, M]
    Mfloat = float(M)

    # Sort ensemble & compute how many are < and <= obs
    ens_sorted, _ = torch.sort(ens_flat, dim=1)
    less = (ens_sorted < obs_flat.unsqueeze(1)).sum(dim=1).float()   # [N]
    leq  = (ens_sorted <= obs_flat.unsqueeze(1)).sum(dim=1).float()  # [N]
    ties = leq - less                                                # [N]

    if randomize_ties:
        U = torch.rand_like(less)
        rank = less + U * ties
    else:
        rank = less + 0.5 * ties

    # rank is in [0, M]; bin to integer bins 0..M
    # numerical guard
    rank = torch.clamp(rank, 0.0, Mfloat)
    bins = torch.round(rank).to(torch.int64)  # nearest bin; alt: floor with epsilon handling
    # Better: place into nearest integer with a tiny jitter to avoid boundary pile-ups
    counts = torch.bincount(bins, minlength=M+1).to(torch.float32)
    return counts  # [M+1]

# =========================
# Extremes: Rx1day / Rx5day, GEV & POT with bootstrap CIs
# =========================

def _rolling_sum_np(x: np.ndarray, k: int) -> np.ndarray:
    # x shape [T], returns rolling sum length T-k+1
    if k == 1:
        return x.copy()
    c = np.cumsum(np.insert(x, 0, 0.0))
    return c[k:] - c[:-k]

def rxk_series(series: np.ndarray, k: int, block_index: np.ndarray | None = None):
    """
    Compute RxK (maximum K-day running *sum*) within each block.
    - `series`: 1-D array (e.g., daily basin-mean precipitation in mm/day)
    - `k`: window length (days)
    - `block_index`: same length as `series`; equal values define a block
      (e.g., seasonal block id). If None, everything is one block.

    Returns
    -------
    values : np.ndarray
        Array of RxK values (one per non-empty block). If a block has
        fewer than `k` valid samples, it falls back to the max daily value
        in that block. Blocks with no finite values are skipped.
    meta : dict
        Metadata with {"k": k}.
    """
    s = np.asarray(series, dtype=float)
    if s.ndim != 1:
        s = s.reshape(-1)

    if block_index is None:
        block_index = np.zeros_like(s, dtype=int)
    else:
        block_index = np.asarray(block_index)
        if block_index.shape != s.shape:
            raise ValueError("block_index must have the same shape as series")

    values = []
    for b in np.unique(block_index):
        x = s[block_index == b]
        x = x[np.isfinite(x)]  # drop NaNs/inf
        if x.size == 0:
            # nothing to contribute from this block
            continue
        if k <= 1:
            values.append(np.max(x))
            continue
        if x.size < k:
            # Not enough samples for a k-day sum; fall back to daily max
            values.append(np.max(x))
            continue
        # Valid rolling k-day sums
        roll = np.convolve(x, np.ones(int(k), dtype=float), mode='valid')
        # Guard (shouldn’t be empty after size check, but be safe)
        values.append(np.max(roll) if roll.size > 0 else np.max(x))

    return np.asarray(values, dtype=float), {"k": int(k)}

def _return_level_gev(c, loc, scale, rp_years: float, block_per_year: float = 1.0) -> float:
    """
    Return level for GEV with SciPy's parameterization:
    genextreme(c). Here c is shape (xi), loc, scale.
    rp_years refers to return period in years; block_per_year is number of blocks per year.
    """
    # Non-exceedance probability for block maxima over return period:
    # For block maxima, return period (years) -> probability p = 1 - 1/(rp_years * block_per_year)
    p = 1.0 - 1.0/(rp_years * block_per_year)
    return float(scipy_gev.ppf(p, c, loc=loc, scale=scale))

def fit_gev_block_maxima_with_ci(
    block_maxima: np.ndarray,                                 # e.g., Rx1day annual maxima per year
    rps_years: Sequence[float] = (2, 5, 10, 20, 50),
    block_per_year: float = 1.0,                              # 4 for seasonal blocks, 1 for annual
    n_boot: int = 1000,
    random_state: Optional[int] = 42,
) -> Dict[str, np.ndarray | float]:
    """
    MLE fit of GEV to block maxima with nonparametric bootstrap CIs (resample maxima).
    Returns dict with params, return levels, and 95% CIs.
    """
    rng = np.random.default_rng(random_state)
    # SciPy's MLE (allows c any real)
    c, loc, scale = scipy_gev.fit(block_maxima)  # returns (c, loc, scale)
    rls = np.array([_return_level_gev(c, loc, scale, rp, block_per_year) for rp in rps_years])

    # Bootstrap resampling of maxima (same sample size)
    boot_rl = np.empty((n_boot, len(rps_years)))
    for b in range(n_boot):
        sample = rng.choice(block_maxima, size=block_maxima.size, replace=True)
        try:
            cb, lb, sb = scipy_gev.fit(sample)
            boot_rl[b, :] = [_return_level_gev(cb, lb, sb, rp, block_per_year) for rp in rps_years]
        except Exception:
            boot_rl[b, :] = np.nan

    lo = np.nanpercentile(boot_rl, 2.5, axis=0)
    hi = np.nanpercentile(boot_rl, 97.5, axis=0)

    return {
        "gev_shape": c,
        "gev_loc": loc,
        "gev_scale": scale,
        "return_periods_years": np.array(rps_years, dtype=float),
        "return_levels": rls,
        "return_levels_lo": lo,
        "return_levels_hi": hi,
        "n_blocks": block_maxima.size,
    }

def fit_pot_gpd_with_ci(
    y_daily: np.ndarray,                    # daily precip (mm)
    threshold: float,                       # u (e.g., seasonal P95 over wet days)
    days_per_year: float = 365.25,
    rps_years: Sequence[float] = (2, 5, 10, 20, 50),
    n_boot: int = 1000,
    random_state: Optional[int] = 42,
) -> Dict[str, np.ndarray | float]:
    """
    POT with GPD above threshold u, MLE using SciPy, RLs with bootstrap CIs.
    Return level formula:
      RL_T = u + (beta/xi) * ( (lambda_u * T_years)**xi - 1 )   if xi != 0
           = u + beta * log(lambda_u * T_years)                  if xi == 0
    where lambda_u = k / N is rate of exceedances per day.
    """
    y = np.asarray(y_daily, dtype=float)
    N = y.size
    exc = y[y > threshold] - threshold
    k = exc.size
    if k < 10:
        raise ValueError(f"Too few exceedances above u={threshold:.2f} (k={k}).")

    # SciPy GPD MLE (loc fixed at 0)
    c, loc, scale = scipy_gpd.fit(exc, floc=0.0)   # c=xi, scale=beta
    xi, beta = c, scale
    lam_u = k / N                                  # per day
    # Return levels
    rls = []
    for rp in rps_years:
        a = lam_u * rp * days_per_year
        if xi == 0.0:
            rl = threshold + beta * np.log(a)
        else:
            rl = threshold + (beta/xi) * (np.power(a, xi) - 1.0)
        rls.append(rl)
    rls = np.array(rls)

    # Bootstrap daily series with replacement of days (iid assumption);
    # for stronger dependence handling, you could block-bootstrap by weeks.
    rng = np.random.default_rng(random_state)
    boot_rl = np.empty((n_boot, len(rps_years)))
    for b in range(n_boot):
        yb = rng.choice(y, size=N, replace=True)
        excb = yb[yb > threshold] - threshold
        kb = excb.size
        if kb < 5:
            boot_rl[b, :] = np.nan
            continue
        try:
            cb, lb, sb = scipy_gpd.fit(excb, floc=0.0)
            xib, betab = cb, sb
            lam_b = kb / N
            vals = []
            for rp in rps_years:
                a = lam_b * rp * days_per_year
                if xib == 0.0:
                    rl = threshold + betab * np.log(a)
                else:
                    rl = threshold + (betab/xib) * (np.power(a, xib) - 1.0)
                vals.append(rl)
            boot_rl[b, :] = vals
        except Exception:
            boot_rl[b, :] = np.nan

    lo = np.nanpercentile(boot_rl, 2.5, axis=0)
    hi = np.nanpercentile(boot_rl, 97.5, axis=0)

    return {
        "gpd_xi": xi,
        "gpd_beta": beta,
        "threshold": threshold,
        "exceedances": int(k),
        "lambda_per_day": lam_u,
        "return_periods_years": np.array(rps_years, dtype=float),
        "return_levels": rls,
        "return_levels_lo": lo,
        "return_levels_hi": hi,
        "N_days": int(N),
    }

# =========================
# Glue: building series & blocks from Torch tensors
# =========================

def to_numpy_1d_series(x: torch.Tensor, mask: Optional[torch.Tensor] = None, agg: str = "mean") -> np.ndarray:
    """
    Convert daily HR fields to a 1D daily series via spatial aggregation.
    Args:
      x: [T,H,W] tensor (obs precip in mm/day)
      mask: [H,W] or [1,H,W] boolean (e.g., basin mask). If None, aggregate over all pixels.
      agg: 'mean' | 'sum' (use 'sum' for basin-total precip; 'mean' for areal average)
    """
    x = _ensure_float(x).detach().cpu()    
    if mask is not None:
        # Normalize mask to [H,W]
        if mask.dim() == 4 and mask.shape[:2] == (1, 1):
            mask = mask.squeeze(0).squeeze(0)
        elif mask.dim() == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)        
        if mask.dtype != torch.bool:
            mask = mask > 0.5
        mask = mask.to(x.device)
        x = x[:, mask.squeeze()]  # [T, Npix_valid]
    else:
        x = x.view(x.shape[0], -1)
    if agg == "sum":
        series = x.sum(dim=1)
    else:
        series = x.mean(dim=1)
    return series.numpy()

def seasonal_block_index(dates_np: np.ndarray) -> np.ndarray:
    """
    Build block indices per day for seasonal maxima.
    dates_np: array of 'YYYYMMDD' ints or np.datetime64
    Returns: integer block id per day (e.g., 2018*10 + season_code)
    """
    # Accept both integer yyyymmdd and datetime64
    if np.issubdtype(dates_np.dtype, np.datetime64):
        dt = dates_np.astype('datetime64[D]')
        yy = dt.astype('datetime64[Y]').astype(int) + 1970
        mm = (dt.astype('datetime64[M]') - dt.astype('datetime64[Y]')).astype(int) + 1
    else:
        # yyyymmdd -> year, month
        yy = dates_np // 10000
        mm = (dates_np % 10000) // 100

    # DJF(1), MAM(2), JJA(3), SON(4) — with DJF belonging to the *winter year* of Jan/Feb,
    # and Dec assigned to year+1 so that DJF is contiguous.
    season = np.select(
        [np.isin(mm, [12,1,2]), np.isin(mm, [3,4,5]), np.isin(mm, [6,7,8]), np.isin(mm, [9,10,11])],
        [1, 2, 3, 4], default=0
    )
    year_adj = yy + ((mm == 12).astype(int))  # Dec goes to next year
    block_id = year_adj * 10 + season
    return block_id.astype(int)

@torch.no_grad()
def pmm_from_ensemble(ens: torch.Tensor, # [B,M,H,W] ensemble tensor
                      mask: Optional[torch.Tensor] = None, # [B, H, W] or [H, W] boolean mask 
                      exclude_zeros: bool = False, # if True, exclude strictly 0.0 values from the pooled distribution
                      ) -> torch.Tensor:
    """
        Probability-Matched Mean (PMM) for univariate fields.

        For each sample b:
        1) Compute the ensemble-mean field μ(x) over members M.
        2) Take the ranks of μ(x) over VALID pixels.
        3) Pool *all* ensemble values across members at VALID pixels and sort that pooled 1-D list.
        4) Assign to each pixel the pooled value at the corresponding rank (quantile) of μ(x).

        This preserves the pooled marginal distribution of the ensemble while using μ(x) to supply the spatial
        pattern (classic PMM used in QPF).

        Args:
            ens: [B,M,H,W] univariate precipitation ensemble in model/physical space
            mask: Optional boolean mask broadcastable to [B,H,W]. Pixels where mask=False are left as the ensemble mean (i.e. PMM falls back to mean).
            exclude_zeros: If True, exclude strictly 0.0 values from the pooled distribution (useful for precipitation).
        Returns:
            pmm: [B,1,H,W] Probability-Matched Mean fields

    """
    if ens.dim() != 4:
        raise ValueError(f"ens must be [B, M, H, W] for univariate PMM, got {ens.shape}")

    ens = ens if torch.is_floating_point(ens) else ens.float()
    B, M, H, W = ens.shape

    device = ens.device

    # Prepare mask broadcast -> [B, H, W]
    if mask is None:
        mask_bhw = torch.ones((B, H, W), dtype=torch.bool, device=device)
    else:
        m = mask
        if m.dtype != torch.bool:
            m = m > 0.5
        while m.dim() < 3:
            m = m.unsqueeze(0)
        if m.shape[0] == 1 and B > 1:
            m = m.expand(B, -1, -1)
        mask_bhw = m.to(device)

    mean_field = ens.mean(dim=1)                 # [B, H, W]
    pmm_out = torch.empty((B, 1, H, W), dtype=ens.dtype, device=device)


    for b in range(B):
        valid = mask_bhw[b].view(-1)             # [H*W]
        if valid.sum() == 0:
            # No valid pixels → fall back to mean
            pmm_out[b, 0] = mean_field[b]
            continue

        # Spatial pattern ranks from ensemble mean
        mf_vals = mean_field[b].view(-1)[valid]  # [Nv]
        # ranks: 0..Nv-1
        ranks = torch.argsort(torch.argsort(mf_vals))

        # Pooled distribution across all M members at valid pixels
        pooled = ens[b].view(M, -1)[:, valid]    # [M, Nv]
        if exclude_zeros:
            pooled_flat = pooled.reshape(-1)
            pooled_flat = pooled_flat[pooled_flat != 0.0]
        else:
            pooled_flat = pooled.reshape(-1)

        if pooled_flat.numel() == 0:
            # Degenerate: no values after excluding zeros → fall back to mean
            out_vals = mf_vals
        else:
            pooled_sorted, _ = torch.sort(pooled_flat)  # [K]
            Nv = mf_vals.numel()
            # Quantile mapping: place rank r at q=(r+0.5)/Nv (midpoint rule)
            q = (ranks.to(torch.float32) + 0.5) / float(Nv)
            # Convert quantiles to indices in pooled_sorted
            # Use (K-1) so that q=1 maps to last index
            K = pooled_sorted.numel()
            idx = torch.clamp((q * (K - 1)).round().to(torch.long), 0, K - 1)
            out_vals = pooled_sorted[idx]

        # Write back into full image (fallback to mean for invalid)
        flat = mean_field[b].view(-1).clone()
        flat[valid] = out_vals
        pmm_out[b, 0] = flat.view(H, W)

    return pmm_out


def crps_ensemble(
    obs: torch.Tensor,              # [H,W]
    ens: torch.Tensor,              # [M,H,W]
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",        # 'mean' | 'sum' | 'none'
) -> torch.Tensor:
    """
    Continuous Ranked Probability Score (CRPS) for ensemble forecasts.
    0 = perfect; higher = worse.
    Fair CRPS estimator (Hersbach, 2000):
      CRPS = (1/M) * sum_i |X_i - y| - (1/(2 M^2)) * sum_{i,j} |X_i - X_j|

    Vectorized per-pixel. If reduction='none', returns [H,W]; otherwise scalar.
    """
    if obs.dim() != 2 or ens.dim() != 3:
        raise ValueError("obs must be [H,W] and ens must be [M,H,W]")
    M = ens.shape[0]
    obs = obs.to(ens.device, ens.dtype)

    # term1: (1/M) sum |Xi - y|
    term1 = (ens - obs.unsqueeze(0)).abs().mean(dim=0)  # [H,W]

    # term2: (1/(2M^2)) sum_{i,j} |Xi - Xj|  = (1/M^2) * sum_{i<j} (v_j - v_i)
    v = ens.view(M, -1)                  # [M, H*W]
    v_sorted, _ = torch.sort(v, dim=0)   # ascending
    i = torch.arange(M, device=ens.device, dtype=ens.dtype).unsqueeze(1)  # [M,1]
    # sum_{i<j} (v_j - v_i) = sum_k (2k - M + 1) * v_(k)
    pair_sum = ( (2*i - (M - 1)) * v_sorted ).sum(dim=0)  # [H*W]
    term2 = (pair_sum / (M * M)).view_as(term1)           # [H,W]

    crps = term1 - term2  # [H,W]

    if mask is not None:
        mask = mask.to(ens.device)
        if mask.dtype != torch.bool:
            mask = mask > 0.5
        vals = crps[mask]
        if vals.numel() == 0:
            return torch.tensor(0.0, device=ens.device)
        if reduction == "mean": return vals.mean()
        if reduction == "sum":  return vals.sum()
        return vals
    else:
        if reduction == "mean": return crps.mean()
        if reduction == "sum":  return crps.sum()
        return crps


def reliability_exceedance_lr_binned(
    obs: torch.Tensor,                  # [H,W] (mm/day)
    ens: torch.Tensor,                  # [M,H,W] (mm/day)
    threshold: float,
    lr_covariate: Optional[torch.Tensor] = None,  # [H,W]; if None, bin by forecast prob
    n_bins: int = 10,
    mask: Optional[torch.Tensor] = None,
    return_brier: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Reliability diagram data for threshold exceedance.
      p_hat = fraction of members >= threshold
      o     = 1(obs >= threshold)
    If lr_covariate is provided, bins are quantiles of that field; else equal-width bins over p_hat in [0,1].
    Returns: bin_center, prob_pred, freq_obs, count, and (optionally) Brier decomposition terms.

    NOTE: IMPLEMENT BOOTSTRAP CIs FOR DIAGRAM POINTS?
    """
    device = ens.device
    p_hat = (ens >= float(threshold)).float().mean(dim=0)  # [H,W]
    o = (obs.to(device) >= float(threshold)).float()       # [H,W]

    if mask is not None:
        mask = mask.to(device)
        if mask.dtype != torch.bool:
            mask = mask > 0.5
        p_hat = p_hat[mask]
        o = o[mask]
        cov = lr_covariate[mask] if (lr_covariate is not None) else None
    else:
        cov = lr_covariate

    if cov is not None:
        cov = cov.to(device).float().view(-1)
        q = torch.linspace(0, 1, n_bins + 1, device=device)
        edges = torch.quantile(cov, q)
        edges[0]  = cov.min() - 1e-6
        edges[-1] = cov.max() + 1e-6
        which = torch.bucketize(cov, edges) - 1  # [N]
        bin_center = 0.5 * (edges[:-1] + edges[1:])
        p_src = p_hat.view(-1)
        o_src = o.view(-1)
    else:
        p_src = p_hat.view(-1)
        o_src = o.view(-1)
        edges = torch.linspace(0, 1, n_bins + 1, device=device)
        which = torch.bucketize(p_src, edges) - 1
        bin_center = 0.5 * (edges[:-1] + edges[1:])

    prob_pred, freq_obs, count = [], [], []
    for b in range(n_bins):
        sel = (which == b)
        n = int(sel.sum().item())
        count.append(n)
        if n == 0:
            prob_pred.append(0.0); freq_obs.append(0.0); continue
        prob_pred.append(p_src[sel].mean().item())
        freq_obs.append(o_src[sel].mean().item())

    out = {
        "bin_center": bin_center.detach().cpu(),
        "prob_pred": torch.tensor(prob_pred, dtype=torch.float32),
        "freq_obs": torch.tensor(freq_obs, dtype=torch.float32),
        "count": torch.tensor(count, dtype=torch.int64),
    }

    if return_brier:
        o_bar = float(o_src.mean().item()) if o_src.numel() else 0.0
        N = max(int(o_src.numel()), 1)
        rel = 0.0
        res = 0.0
        for k in range(n_bins):
            Nk = int(out["count"][k].item())
            if Nk == 0: continue
            pk = float(out["prob_pred"][k].item())
            ok = float(out["freq_obs"][k].item())
            w = Nk / N
            rel += w * (pk - ok) ** 2
            res += w * (ok - o_bar) ** 2
        unc = o_bar * (1.0 - o_bar)
        out.update({
            "brier": torch.tensor(rel - res + unc, dtype=torch.float32),
            "reliability": torch.tensor(rel, dtype=torch.float32),
            "resolution": torch.tensor(res, dtype=torch.float32),
            "uncertainty": torch.tensor(unc, dtype=torch.float32),
        })

    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.cpu()
    return out


def spread_skill(
    obs: torch.Tensor,             # [H,W]
    ens: torch.Tensor,             # [M,H,W]
    point: str = "pmm",            # 'pmm' | 'mean' | 'median'
    mask: Optional[torch.Tensor] = None,
    n_bins: int = 10,
) -> Dict[str, torch.Tensor]:
    """
    Spread–skill diagnostic:
      spread = ensemble std per pixel
      skill  = |point_estimate - obs| per pixel
    We bin by spread (quantile bins) to assess calibration of spread.

    NOTE: IMPLEMENT VARIANCE PER BIN AS WELL?
    """

    device = ens.device
    obs = obs.to(device).to(ens.dtype)
    if mask is not None:
        mask = (mask.to(device) > 0.5)

    spread = ens.std(dim=0)  # [H,W]
    if point == "mean":
        pt = ens.mean(dim=0)
    elif point == "median":
        pt = ens.median(dim=0).values
    else:
        pt = pmm_from_ensemble(ens.unsqueeze(0)).squeeze(0).squeeze(0)  # [H,W]
    ae = (pt - obs).abs()

    if mask is not None:
        spread = spread[mask]
        ae = ae[mask]

    q = torch.linspace(0, 1, n_bins + 1, device=device)
    edges = torch.quantile(spread, q)
    edges[0]  = spread.min() - 1e-9
    edges[-1] = spread.max() + 1e-9
    which = torch.bucketize(spread, edges) - 1
    centers = 0.5 * (edges[:-1] + edges[1:])

    sp_mean, sk_mean, count = [], [], []
    for b in range(n_bins):
        sel = (which == b)
        n = int(sel.sum().item())
        count.append(n)
        if n == 0:
            sp_mean.append(0.0); sk_mean.append(0.0); continue
        sp_mean.append(spread[sel].mean().item())
        sk_mean.append(ae[sel].mean().item())

    return {
        "bin_center": centers.detach().cpu(),
        "spread": torch.tensor(sp_mean, dtype=torch.float32),
        "skill": torch.tensor(sk_mean, dtype=torch.float32),
        "count": torch.tensor(count, dtype=torch.int64),
    }



def compute_isotropic_psd(
    batch: torch.Tensor,        # [B,1,H,W]
    dx_km: float,
    mask: Optional[torch.Tensor] = None,  # [B,1,H,W] or broadcastable
    *,
    normalize: str = "none", # "none" | "per_field" | "match_ref"
    ref_power: Optional[torch.Tensor] = None,  # [H,W] reference PSD to match total power if normalize="match_ref"
    max_k: float | None = None,
    window: str | None = "hann",  # None | "hann"
) -> Dict[str, torch.Tensor]:
    """
    Isotropic (radially averaged) 2D PSD for a batch.
    
    Inputs:
        btach:  torch tensor 
            Input fields, shape [B,1,H,W]. Should already be in physical space (e.g., precip in mm/day).
        dx_km: float
            Grid spacing in kilometers.
        mask:  torch tensor, optional
            Broadcastable boolean mask where True indicates valid pixels. Masked points are set to zero before FFT.
        normalize: {"none", "per_field", "match_ref"}, default "none"
            - "none": keep raw spectral power 
            - "per_field": for each sample, divide its radially averaged spectrum by its own total variance/power
                            then average over the batch. This removes amplitude differences and keeps shape only.
            - "match_ref": after averaging over the batch, scale the spectrum so that its total power matches that of ref_power.
                            Requires ref_power (1D) input (typically HR/obs spectrum).
        ref_power: torch tensor, optional
            1D reference spectrum to match when normalize="match_ref". Shape [N_bins].

    Returns:
        dict with keys:
            "k": torch tensor [N_bins], radial wavenumber bin centers (1/km)
            "psd": batch-mean (and possibly normalized) isotropic PSD [N_bins]
            "psd_std": [N_bins] std across samples (after chosen normalization)
            "psd_n": [N_bins] number of samples contributing to each bin (usually B)
            "psd_ci_lo": [N_bins] lower 95% CI across samples
            "psd_ci_hi": [N_bins] upper 95% CI across samples

    """
    if batch.dim() != 4 or batch.shape[1] != 1:
        raise ValueError(f"batch must be [B,1,H,W], got {batch.shape}")
    B, _, H, W = batch.shape
    device = batch.device
    x = batch.to(torch.float32)

    # apply mask
    if mask is not None:
        m = mask
        while m.dim() < 4:
            m = m.unsqueeze(0)
        if m.shape[0] == 1 and B > 1:
            m = m.expand(B, -1, -1, -1)
        x = x.masked_fill(~m.bool(), 0.0)

    # Optional windowing to reduce edge effects leakage (can result in high-frequency noise)
    if window is not None:
        if window == "hann":
            wy = torch.hann_window(H, device=x.device).view(1, 1, H, 1)
            wx = torch.hann_window(W, device=x.device).view(1, 1, 1, W)
            w2d = wy * wx  # [1,1,H,W]
            x = x * w2d
        else:
            raise ValueError(f"Unsupported window type: {window}")

    # FFT -> power per sample
    X = torch.fft.rfft2(x, norm='ortho')           # [B,1,H,W//2+1]
    P2 = (X.real**2 + X.imag**2).squeeze(1)      # [B,H,W//2+1]

    # Build wavenumber grid
    ky = torch.fft.fftfreq(H, d=dx_km, device=device)   # 1/km
    kx = torch.fft.rfftfreq(W, d=dx_km, device=device)
    Ky, Kx = torch.meshgrid(ky, kx, indexing='ij')
    Kr = torch.sqrt(Kx**2 + Ky**2)                      # [H, W//2+1]

    kr = Kr.flatten()
    kmax_all = kr.max().item()
    n_bins = int(min(H, W) // 2)
    edges = torch.linspace(0, kmax_all, n_bins + 1, device=device)
    k_centers = 0.5 * (edges[:-1] + edges[1:]) # [N_bins]

    # Precompute which-bin for each FFT cell
    which = torch.bucketize(kr, edges) - 1  # [H*(W//2+1)]

    # Radial average per sample, keep individual spectra for optional normalization
    psd_per_sample: list[torch.Tensor] = []
    for b in range(B):
        p = P2[b].flatten()
        psd_b = torch.zeros(n_bins, device=device) # [N_bins]
        counts_b = torch.zeros(n_bins, device=device)
        for i in range(n_bins):
            sel = (which == i)
            if sel.any():
                psd_b[i] = p[sel].mean()
                counts_b[i] = sel.sum()
        # Optional per-field normalization
        if normalize == "per_field":
            # total power = sum(psd_b * shell_width); here we just use simple L1 over bins
            tot = psd_b.sum()
            if tot > 0.0:
                psd_b = psd_b / tot
        psd_per_sample.append(psd_b)

    # Stack all samples -> [B, N_bins]
    psd_all = torch.stack(psd_per_sample, dim=0)  # [B, N_bins]
    psd_mean = psd_all.mean(dim=0)         # [N_bins]
    psd_std = psd_all.std(dim=0, unbiased=True)  # [N_bins]
    psd_n = torch.full_like(psd_mean, float(B)) # Using all B samples for every bin

    if normalize == "match_ref":
        if ref_power is None:
            raise ValueError("ref_power must be provided when normalize='match_ref'")
        ref_power = ref_power.to(device).to(psd_mean.dtype)
        num = (psd_mean * ref_power).sum()
        den = (psd_mean * psd_mean).sum() + 1e-12
        scale = num / den
        psd_mean = psd_mean * scale
        psd_std = psd_std * scale

    # 95% CI over the batch
    eps = 1e-12
    se = psd_std / torch.sqrt(psd_n.clamp(min=1.0))
    ci_lo = psd_mean - 1.96 * se
    ci_hi = psd_mean + 1.96 * se

    # Cut to Nyquist or user-specified max_k
    nyq = 1.0 / (2.0 * dx_km)

    if max_k is not None:
        keep = (k_centers <= float(max_k))
    else:
        keep = (k_centers <= nyq + 1e-9)

    k_out = k_centers[keep]
    psd_out = psd_mean[keep]
    psd_std_out = psd_std[keep]
    psd_n_out = psd_n[keep]
    ci_lo_out = ci_lo[keep]
    ci_hi_out = ci_hi[keep]


    return {
        "k": k_out.detach().cpu(),
        "psd": psd_out.detach().cpu(),
        "psd_std": psd_std_out.detach().cpu(),
        "psd_n": psd_n_out.detach().cpu(),
        "psd_ci_lo": ci_lo_out.detach().cpu(),
        "psd_ci_hi": ci_hi_out.detach().cpu(),
    }




@torch.no_grad()
def _flatten_valid_np(arr: np.ndarray, mask: Optional[torch.Tensor], non_neg: bool = True) -> np.ndarray:
    """ Return 1D numpy array of valid, finite, optionally non-negative pixels"""
    x = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if non_neg:
        x = np.maximum(x, 0.0)
    if mask is not None:
        m = mask
        if isinstance(m, torch.Tensor):
            if m.dtype != torch.bool:
                m = m > 0.5
            # normalize to [H,W]
            if m.dim() == 4 and m.shape[:2] == (1, 1):
                m = m.squeeze(0).squeeze(0)
            elif m.dim() == 3 and m.shape[0] == 1:
                m = m.squeeze(0)
            m = m.cpu().numpy().astype(bool)
        if m.shape != x.shape:
            m = np.broadcast_to(m, x.shape)
        x = x[m]
    else:
        x = x.reshape(-1)
    return x

def _write_bins_and_counts_csv(tables_dir: "Path", bins: np.ndarray, counts: Dict[str, np.ndarray]) -> None:
    import csv
    # bins
    with open(tables_dir / "pixel_dist_bins.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin_left", "bin_right"])
        for i in range(len(bins) - 1):
            w.writerow([float(bins[i]), float(bins[i+1])])
    # each series
    for name, H in counts.items():
        with open(tables_dir / f"pixel_dist_{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["bin_idx", "count"])
            for i, c in enumerate(H.astype(int)):
                w.writerow([i, int(c)])

def _distribution_metrics(ref: np.ndarray, comp: np.ndarray, eps: float = 1e-12) -> Dict[str, float]:
    """
    Compute Wasserstein-1 (Earth Mover), KS statistic, and KL(ref||comp).
    KL stabilised with small epsilon on discrete histograms after normalizing to probabilities.
    NOTE: KL is asymmetric; we use KL(HR||X) where HR is the reference.
    """
    if ref.size == 0 or comp.size == 0:
        return {"wasserstein": float("nan"), "ks_stat": float("nan"), "ks_p": float("nan"), "kl_hr_to_x": float("nan")}
    try:
        w = float(_wasserstein(ref, comp))
    except Exception:
        w = float("nan")
    try:
        # Some SciPy versions return a namedtuple-like object, others a plain tuple.
        ks_any: Any = _ks2(ref, comp, alternative="two-sided")  # type: ignore[call-arg]
        if hasattr(ks_any, "statistic") and hasattr(ks_any, "pvalue"):
            ks_stat = float(getattr(ks_any, "statistic")); ks_p = float(getattr(ks_any, "pvalue"))
        else:
            # Fall back to tuple unpacking
            ks_tup = cast(Tuple[float, float], ks_any)
            ks_stat = float(ks_tup[0]); ks_p = float(ks_tup[1])
    except Exception:
        ks_stat = float("nan"); ks_p = float("nan")

    # KL on *coarsened* support for robustness: use common fixed bins from pooled 99.5th percentile
    vmax = np.percentile(np.concatenate([ref, comp]), 99.5) if ref.size and comp.size else 1.0
    vmax = max(1.0, float(vmax))
    edges = np.linspace(0.0, vmax, 128+1, dtype=np.float32)
    Hr, _ = np.histogram(ref, bins=edges)
    Hx, _ = np.histogram(comp, bins=edges)
    pr = Hr.astype(np.float64); px = Hx.astype(np.float64)
    pr = pr / max(pr.sum(), 1.0); px = px / max(px.sum(), 1.0)
    pr = np.clip(pr, eps, 1.0); px = np.clip(px, eps, 1.0)
    try:
        kl = float(_kl_entropy(pr, px))
    except Exception:
        kl = float("nan")
    return {"wasserstein": w, "ks_stat": ks_stat, "ks_p": ks_p, "kl_hr_to_x": kl}


# --- Insert/replace: compute_and_save_pooled_pixel_distributions and compute_and_save_yearly_maps ---

def compute_and_save_pooled_pixel_distributions(
    gen_root: str | Path,
    out_root: str | Path,
    mask_global: Optional[torch.Tensor] = None,
    include_lr: bool = True,
    n_bins: int = 80,
    vmax_percentile: float = 99.5,
    save_samples_cap: int = 200_000,
) -> bool:
    """
    Pools valid pixels across all dates from:
      <gen_root>/{pmm_phys|pmm}/DATE.npz  (key='pmm')
      <gen_root>/{lr_hr_phys|lr_hr}/DATE.npz (keys: 'hr' and optional 'lr')
      <gen_root>/lsm/DATE.npz (optional; key 'lsm' or 'lsm_hr')
    Writes CSVs under <out_root>/tables:
      - pixel_dist_bins.csv
      - pixel_dist_hr.csv
      - pixel_dist_pmm.csv
      - pixel_dist_lr.csv (optional)
      - pixel_dist_metrics.csv (Wasserstein, KS, KL vs HR)
    Returns True iff outputs were written.
    """
    gen_root = Path(gen_root)
    out_root = Path(out_root)
    tables_dir = out_root / "tables"
    figs_dir   = out_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Prefer physical-space folders if present
    pmm_dir  = gen_root / "pmm_phys"
    lrhr_dir = gen_root / "lr_hr_phys"
    if not pmm_dir.exists():  pmm_dir  = gen_root / "pmm"
    if not lrhr_dir.exists(): lrhr_dir = gen_root / "lr_hr"
    lsm_dir = gen_root / "lsm"
    if not lsm_dir.exists(): lsm_dir = None

    dates = sorted([p.stem for p in pmm_dir.glob("*.npz")])
    logger.info("[metrics] pooled-dists: pmm_dir=%s lrhr_dir=%s lsm_dir=%s n_dates=%d",
                pmm_dir, lrhr_dir, lsm_dir, len(dates))
    if len(dates) == 0:
        logger.warning("[metrics] pooled-dists: no dates found → nothing written")
        return False

    def _get_np(folder: Path, date: str, key: str):
        p = folder / f"{date}.npz"
        if not p.exists(): return None
        try:
            d = np.load(p, allow_pickle=True)
            a = d.get(key, None)
            if a is None: return None
            return np.nan_to_num(a.squeeze(), nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            return None

    X_hr, X_pmm, X_lr = [], [], []
    skipped = 0
    for d in dates:
        pmm = _get_np(pmm_dir,  d, "pmm")
        hr  = _get_np(lrhr_dir, d, "hr")
        if pmm is None or hr is None:
            skipped += 1
            continue
        lr  = _get_np(lrhr_dir, d, "lr") if include_lr else None

        # optional per-date mask
        m = None
        if lsm_dir is not None:
            m = _get_np(lsm_dir, d, "lsm")
            if m is None: m = _get_np(lsm_dir, d, "lsm_hr")
            if m is not None:
                m = torch.from_numpy((m > 0.5) if isinstance(m, np.ndarray) else m)

        X_hr.append(_flatten_valid_np(hr,  m, non_neg=True))
        X_pmm.append(_flatten_valid_np(pmm, m, non_neg=True))
        if lr is not None:
            X_lr.append(_flatten_valid_np(lr, m, non_neg=True))

    if not X_hr or not X_pmm:
        logger.warning("[metrics] pooled-dists: no valid HR/PMM vectors after loading; skipped=%d", skipped)
        return False

    X_hr  = np.concatenate(X_hr,  axis=0)
    X_pmm = np.concatenate(X_pmm, axis=0)
    X_lr  = np.concatenate(X_lr, axis=0) if X_lr else None

    # cap samples to limit file size / plotting speed
    def _cap(x: np.ndarray | None) -> np.ndarray | None:
        if x is None: return None
        if save_samples_cap and x.size > save_samples_cap:
            idx = np.linspace(0, x.size - 1, save_samples_cap).astype(int)
            return x[idx]
        return x

    X_hr_c  = _cap(X_hr)
    X_pmm_c = _cap(X_pmm)
    X_lr_c  = _cap(X_lr)

    vmax = np.percentile(
        np.concatenate([a for a in [X_hr_c, X_pmm_c, X_lr_c] if a is not None]),
        vmax_percentile
    )
    vmax = max(1.0, float(vmax))
    edges = np.linspace(0.0, vmax, int(n_bins) + 1, dtype=np.float32)

    H_hr = H_pmm = H_lr = None
    counts = {}
    if X_hr_c is not None:
        H_hr, _ = np.histogram(X_hr_c, bins=edges)
        counts["hr"] = H_hr
    if X_pmm_c is not None:
        H_pmm, _ = np.histogram(X_pmm_c, bins=edges)
        counts["pmm"] = H_pmm
    if X_lr_c is not None:
        H_lr, _ = np.histogram(X_lr_c, bins=edges)
        counts["lr"] = H_lr

    _write_bins_and_counts_csv(tables_dir, edges, counts)
    logger.info("[metrics] pooled-dists: wrote %s", tables_dir / "pixel_dist_bins.csv")

    # distance metrics vs HR
    rows = []
    if X_hr_c is not None and X_pmm_c is not None:
        rows.append({"ref": "hr", "comp": "pmm", **_distribution_metrics(X_hr_c, X_pmm_c)})
    if X_hr_c is not None and X_lr_c is not None:
        rows.append({"ref": "hr", "comp": "lr", **_distribution_metrics(X_hr_c, X_lr_c)})

    out_csv = tables_dir / "pixel_dist_metrics.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)
    logger.info("[metrics] pooled-dists: wrote %s", out_csv)

    # optional sample NPZ for quick checks
    np.savez_compressed(
        figs_dir / "pixel_dist_samples.npz",
        hr=(X_hr_c if X_hr_c is not None else np.array([], dtype=np.float32)),
        pmm=(X_pmm_c if X_pmm_c is not None else np.array([], dtype=np.float32)),
        lr=(X_lr_c if X_lr_c is not None else np.array([], dtype=np.float32))
    )
    return True


@torch.no_grad()
def compute_and_save_yearly_maps(
    gen_root: str | Path,
    out_root: str | Path,
    which: Sequence[str] = ("mean","sum","rx1","rx5"),
    include_lr: bool = True,
    mask: Optional[torch.Tensor] = None,
) -> bool:
    """
    Aggregate daily HR/PMM (and optional LR) into yearly maps and save under:
      <out_root>/maps/year_YYYY_{mean|sum|rx1|rx5}.npz
    Each NPZ contains keys: 'hr', 'pmm', and optionally 'lr'.
    Returns True iff at least one year's maps were written.
    """

    gen_root = Path(gen_root)
    out_root = Path(out_root)

    # Prefer physical-space folders if present
    pmm_dir  = (gen_root / "pmm_phys") if (gen_root / "pmm_phys").exists() else (gen_root / "pmm")
    lrhr_dir = (gen_root / "lr_hr_phys") if (gen_root / "lr_hr_phys").exists() else (gen_root / "lr_hr")

    # Create out dirs
    maps_dir = out_root / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Optional static mask normalization
    mask_np: np.ndarray | None = None
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            m = mask
            if m.dtype != torch.bool:
                m = m > 0.5
            # normalize to [H,W]
            if m.dim() == 4 and m.shape[:2] == (1, 1):
                m = m.squeeze(0).squeeze(0)
            elif m.dim() == 3 and m.shape[0] == 1:
                m = m.squeeze(0)
            mask_np = m.detach().cpu().numpy().astype(bool)
        else:
            mask_np = np.asarray(mask, dtype=bool)
    logger.info("[metrics] yearly-maps: pmm_dir=%s lrhr_dir=%s mask=%s", pmm_dir, lrhr_dir, "yes" if mask_np is not None else "no")

    dates = sorted([p.stem for p in pmm_dir.glob("*.npz")])
    logger.info("[metrics] yearly-maps: n_dates=%d", len(dates))
    if len(dates) == 0:
        logger.warning("[metrics] yearly-maps: no dates found → nothing written")
        return False

    @dataclass
    class YearAgg:
        # running sums for mean & sum (NaN-masked)
        sum_hr: np.ndarray | None = None
        sum_pmm: np.ndarray | None = None
        sum_lr: np.ndarray | None = None
        # valid-pixel counters for mean (per-pixel), NaN-safe
        cnt_hr: np.ndarray | None = None
        cnt_pmm: np.ndarray | None = None
        cnt_lr: np.ndarray | None = None
        # day counter (for provenance)
        n_days: int = 0
        # Rx1 (daily max) accumulators (NaN-aware)
        rx1_hr: np.ndarray | None = None
        rx1_pmm: np.ndarray | None = None
        rx1_lr: np.ndarray | None = None
        # Rx5 (rolling 5-day sum) accumulators and buffers
        rx5_hr: np.ndarray | None = None
        rx5_pmm: np.ndarray | None = None
        rx5_lr: np.ndarray | None = None
        last5_hr: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=5))
        last5_pmm: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=5))
        last5_lr: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=5))

    def _year_from(date_str: str) -> int:
        try:
            return int(str(date_str)[:4])
        except Exception:
            return -1

    acc: DefaultDict[int, YearAgg] = defaultdict(YearAgg)

    def _get_np(date: str, src: str, key: str):
        """Safe npz loader with dir/key logic."""
        p = (pmm_dir if src == "pmm" else lrhr_dir) / f"{date}.npz"
        if not p.exists():
            return None
        try:
            d = np.load(p, allow_pickle=True)
            a = d.get(key, None)
            if a is None:
                return None
            x = np.nan_to_num(a.squeeze().astype(np.float32), nan=np.nan, posinf=np.nan, neginf=np.nan)
            if mask_np is not None:
                # Apply spatial mask → invalid pixels set to NaN
                if mask_np.shape != x.shape:
                    try:
                        m = np.broadcast_to(mask_np, x.shape)
                    except Exception:
                        logger.warning("[metrics] yearly-maps: mask shape %s not compatible with %s; ignoring mask.", mask_np.shape, x.shape)
                        m = None
                else:
                    m = mask_np
                if m is not None:
                    x = np.where(m, x, np.nan)
            # We also clamp negatives to zero for precip fields to avoid NaN propagation in sums
            x = np.where(np.isfinite(x), np.maximum(x, 0.0), np.nan)
            return x
        except Exception:
            return None

    # Accumulate per year
    n_loaded = 0

    for d in dates:
        y = _year_from(d)
        if y < 0:
            continue
        pmm = _get_np(d, "pmm", "pmm")
        hr  = _get_np(d, "lrhr", "hr")
        if pmm is None or hr is None:
            continue
        lr = _get_np(d, "lrhr", "lr") if include_lr else None

        A = acc[y]
        H, W = hr.shape
        # initialize counters on first valid day
        if A.sum_hr is None:
            A.sum_hr  = np.zeros((H, W), dtype=np.float32)
            A.sum_pmm = np.zeros((H, W), dtype=np.float32)
            A.cnt_hr  = np.zeros((H, W), dtype=np.float32)
            A.cnt_pmm = np.zeros((H, W), dtype=np.float32)
            A.rx1_hr  = np.full((H, W), np.nan, dtype=np.float32)
            A.rx1_pmm = np.full((H, W), np.nan, dtype=np.float32)
            if include_lr and (lr is not None):
                A.sum_lr = np.zeros((H, W), dtype=np.float32)
                A.cnt_lr = np.zeros((H, W), dtype=np.float32)
                A.rx1_lr = np.full((H, W), np.nan, dtype=np.float32)

        # --- sums + per-pixel counts (NaN-aware) ---
        A.sum_hr  += np.nan_to_num(hr,  nan=0.0)
        A.sum_pmm += np.nan_to_num(pmm, nan=0.0)
        if A.cnt_hr is not None:
            A.cnt_hr  += np.isfinite(hr).astype(np.float32)
        else:
            # initialize if unexpectedly None (defensive)
            A.cnt_hr = np.isfinite(hr).astype(np.float32)

        if A.cnt_pmm is not None:
            A.cnt_pmm += np.isfinite(pmm).astype(np.float32)
        else:
            # initialize if unexpectedly None (defensive)
            A.cnt_pmm = np.isfinite(pmm).astype(np.float32)

        if include_lr and (lr is not None):
            A.sum_lr += np.nan_to_num(lr, nan=0.0)
            if A.cnt_lr is not None:
                A.cnt_lr += np.isfinite(lr).astype(np.float32)
            else:
                # initialize if unexpectedly None (defensive)
                A.cnt_lr = np.isfinite(lr).astype(np.float32)

        # --- Rx1 (NaN-aware max) ---
        if A.rx1_hr is None:
            A.rx1_hr = hr.copy()
        else:
            A.rx1_hr = np.fmax(cast(np.ndarray, A.rx1_hr), hr)

        if A.rx1_pmm is None:
            A.rx1_pmm = pmm.copy()
        else:
            A.rx1_pmm = np.fmax(cast(np.ndarray, A.rx1_pmm), pmm)

        if include_lr and (lr is not None):
            if A.rx1_lr is None:
                A.rx1_lr = lr.copy()
            else:
                A.rx1_lr = np.fmax(cast(np.ndarray, A.rx1_lr), lr)

        # --- Rx5 (rolling 5-day sums, then NaN-aware max over window sums) ---
        A.last5_hr.append(hr);  A.last5_pmm.append(pmm)
        if include_lr and (lr is not None): A.last5_lr.append(lr)

        if len(A.last5_hr) == 5:
            # nansum across the 5-day window
            s5_hr  = np.nansum(np.stack(list(A.last5_hr),  axis=0), axis=0)
            s5_pmm = np.nansum(np.stack(list(A.last5_pmm), axis=0), axis=0)
            if A.rx5_hr is None:
                A.rx5_hr  = s5_hr.copy()
                A.rx5_pmm = s5_pmm.copy()
            else:
                A.rx5_hr  = np.fmax(cast(np.ndarray, A.rx5_hr),  s5_hr)
                A.rx5_pmm = np.fmax(cast(np.ndarray, A.rx5_pmm), s5_pmm)
            if include_lr and (lr is not None) and (len(A.last5_lr) == 5):
                s5_lr = np.nansum(np.stack(list(A.last5_lr), axis=0), axis=0)
                if A.rx5_lr is None:
                    A.rx5_lr = s5_lr.copy()
                else:
                    A.rx5_lr = np.fmax(cast(np.ndarray, A.rx5_lr), s5_lr)

        A.n_days += 1
        n_loaded += 1

    if n_loaded == 0:
        logger.warning("[metrics] yearly-maps: nothing to aggregate (no valid HR/PMM pairs).")
        return False

    def _nan_div(num: np.ndarray | None, den: np.ndarray | None) -> np.ndarray | None:
        if num is None or den is None:
            return None
        out = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0.0)
        return out

    wrote_any = False
    for y, A in acc.items():
        if A.n_days <= 0:
            logger.warning("[metrics] yearly-maps: year %d has no valid days → skipping", y)
            continue
        wrote_any = True

        # --- MEAN (per-pixel using valid counts ---
        if "mean" in which:
            out = {}
            hr_mean  = _nan_div(A.sum_hr,  A.cnt_hr)
            pmm_mean = _nan_div(A.sum_pmm, A.cnt_pmm)
            if hr_mean  is not None: out["hr"]  = hr_mean.astype(np.float32)
            if pmm_mean is not None: out["pmm"] = pmm_mean.astype(np.float32)
            if include_lr and (A.sum_lr is not None) and (A.cnt_lr is not None):
                lr_mean = _nan_div(A.sum_lr, A.cnt_lr)
                if lr_mean is not None:
                    out["lr"] = lr_mean.astype(np.float32)
            if out:
                p = maps_dir / f"year_{y}_mean.npz"
                np.savez_compressed(p, **out)
                logger.info("[metrics] yearly-maps: wrote %s", p.name)

        # --- SUM (NaN treated as zero contribution) ---
        if "sum" in which:
            out = {}
            if A.sum_hr  is not None: out["hr"]  = A.sum_hr.astype(np.float32)
            if A.sum_pmm is not None: out["pmm"] = A.sum_pmm.astype(np.float32)
            if include_lr and (A.sum_lr is not None): out["lr"] = A.sum_lr.astype(np.float32)
            if out:
                p = maps_dir / f"year_{y}_sum.npz"
                np.savez_compressed(p, **out)
                logger.info("[metrics] yearly-maps: wrote %s", p.name)

        # --- RX1 (daily max, NaN-aware) ---
        if "rx1" in which and (A.rx1_hr is not None):
            out = {"hr": A.rx1_hr.astype(np.float32)}
            if A.rx1_pmm is not None: out["pmm"] = A.rx1_pmm.astype(np.float32)
            if include_lr and (A.rx1_lr is not None): out["lr"] = A.rx1_lr.astype(np.float32)
            p = maps_dir / f"year_{y}_rx1.npz"
            np.savez_compressed(p, **out)
            logger.info("[metrics] yearly-maps: wrote %s", p.name)

        # --- RX5 (max 5-day sum, NaN-aware) ---
        if "rx5" in which and (A.rx5_hr is not None):
            out = {"hr": A.rx5_hr.astype(np.float32)}
            if A.rx5_pmm is not None: out["pmm"] = A.rx5_pmm.astype(np.float32)
            if include_lr and (A.rx5_lr is not None): out["lr"] = A.rx5_lr.astype(np.float32)
            p = maps_dir / f"year_{y}_rx5.npz"
            np.savez_compressed(p, **out)
            logger.info("[metrics] yearly-maps: wrote %s", p.name)

    if wrote_any:
        logger.info("[metrics] yearly-maps: all done → %s", maps_dir)
    else:
        logger.warning("[metrics] yearly-maps: nothing written (no accumulations).")
    return wrote_any
