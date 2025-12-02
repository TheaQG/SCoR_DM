"""
    Probabilistic metrics for precipitation downscaling ensemble evaluation.

    This module is self-contained. It probides only the number-crunching
    parts of the precipitation probabilistic metrics evaluation block.

    It covers:
        - CRPS
        - Randomized PIT histograms from ensembles
        - Rank histogram
        - Reliability for threshold exceedance (with optional LR reference)
        - Spread-skill relationship (binned for spread)
        - Small aggregation helpers for reliability/rank histograms

    All plotting and I/O happens in:
        sbgm.evaluate.evaluate_prcp.eval_probabilistic.plot_probabilistic (plotting)
        sbgm.evaluate.evaluate_prcp.eval_probabilistic.evaluate_probabilistic (I/O)
"""

from __future__ import annotations
from typing import Optional, Sequence, Dict, List, Any

import torch
import logging 

logger = logging.getLogger(__name__)
# ================================================================================
# Small internal helpers
# ================================================================================


def _ensure_float(x: torch.Tensor) -> torch.Tensor:
    return x if torch.is_floating_point(x) else x.float()

def _normalize_mask(
        mask: Optional[torch.Tensor],
        target_shape: torch.Size,
        device: torch.device
) -> Optional[torch.Tensor]:
    """
        Normalize mask to the final target shape (broadcasting style)

        If normalization fails (e.g. full-domain mask [589,789] vs cutout [128,128]),
        we log a warning and fall back to an all-True mask on the target shape
        (i.e. effectively no spatial restriction for this metric call).
    """
    if mask is None:
        return None
    m = mask.to(device)
    if m.dtype != torch.bool:
        m = m > 0.5

    # Squeeze leading singleton dimensions
    while m.dim() > 2 and m.shape[0] == 1:
        m = m.squeeze(0)

    # Normalize target_shape to a tuple
    ts = tuple(target_shape)

    # Direct matches
    if m.shape == ts:
        return m

    # [H,W] -> [B,H,W], only if H,W match target
    if m.dim() == 2 and len(ts) == 3:
        B, H, W = ts
        if m.shape == (H, W):
            return m.unsqueeze(0).expand(B, -1, -1)

    # [1,H,W] -> [H,W], only if H,W match target
    if m.dim() == 3 and len(ts) == 2 and m.shape[0] == 1 and m.shape[1:] == ts:
        return m.squeeze(0)
    
    # Last resort: try broadcast via expand; on failure, fall back to all-True mask
    try:
        return m.expand(target_shape)
    except Exception as e:
        logger.warning(
            "[prob_metrics] Mask normalization failed: mask.shape=%s, "
            "target_shape=%s. Falling back to full-domain mask "
            "(no spatial restriction). Error: %s",
            tuple(mask.shape),
            tuple(target_shape),
            e,
        )
        # Fallback: all-True mask on the target shape
        return torch.ones(target_shape, dtype=torch.bool, device=device)
    



# ================================================================================
# 1. CRPS
# ================================================================================

def crps_ensemble(
        obs: torch.Tensor, # [H,W]
        ens: torch.Tensor, # [M,H,W]
        mask: Optional[torch.Tensor] = None, # [H,W] or [M,H,W]
        reduction: str = "mean",
) -> torch.Tensor:
    """
        Continuous Ranked Probability Score (CRPS) for downscaling ensemble.
        Fair CRPS estimator (Hersbach 2000):
            CRPS = (1/M) * sum_i |X_i - Y| - (1/(2 * M^2)) * sum_{i,j} |X_i - X_j|
        
        If reduction = "none", the function returns [H,W] tensor of CRPS values (map of CRPS).
        Otherwise returns scalar on same device as inputs.
    """
    if obs.dim() != 2 or ens.dim() != 3:
        raise ValueError("crps_ensemble expects obs=[H,W] and ens=[M,H,W]")
    M = ens.shape[0]
    obs = obs.to(ens.device, ens.dtype)

    # term 1
    term1 = (ens - obs.unsqueeze(0)).abs().mean(dim=0)  # [H,W]

    # term 2 (vectorized pairwise diff via sorted trick)
    v = ens.view(M, -1)                  # [M, H*W]
    v_sorted, _ = torch.sort(v, dim=0)   # [M, H*W]
    i = torch.arange(M, device=ens.device, dtype=ens.dtype).unsqueeze(1)  # [M,1]
    pair_sum = ((2 * i - (M - 1)) * v_sorted).sum(dim=0)                  # [H*W]
    term2 = (pair_sum / (M * M)).view_as(term1)                           # [H,W]

    crps = term1 - term2  # [H,W]

    if mask is not None:
        m = _normalize_mask(mask, crps.shape, ens.device)
        vals = crps[m]
        if vals.numel() == 0:
            return torch.tensor(0.0, device=ens.device, dtype=crps.dtype)
        if reduction == "mean":
            return vals.mean()
        if reduction == "sum":
            return vals.sum()
        return vals  # [N_valid]
    else:
        if reduction == "mean":
            return crps.mean()
        if reduction == "sum":
            return crps.sum()
        return crps

# ================================================================================
# 1b. Energy score (multivariate CRPS)
# ================================================================================

@torch.no_grad()
def energy_score(
    obs: torch.Tensor,           # [H,W]
    ens: torch.Tensor,           # [M,H,W]
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Energy score for spatial fields, following Gneiting et al. (2008).

    ES = 1/M * sum_i ||X_i - y|| - 1/(2 M^2) * sum_{i,j} ||X_i - X_j||

    We treat the whole field as *one* multivariate vector.
    If `mask` is given, we only use the valid pixels.
    """
    if obs.dim() != 2 or ens.dim() != 3:
        raise ValueError("energy_score expects obs=[H,W] and ens=[M,H,W]")

    M, H, W = ens.shape
    device = ens.device
    obs = obs.to(device, ens.dtype)
    ens = ens.to(device, ens.dtype)

    # flatten + optional mask
    if mask is not None:
        m = _normalize_mask(mask, obs.shape, device=device)  # [H,W] (bool)
        m_flat = m.view(-1) # type: ignore
        y = obs.view(-1)[m_flat]               # [N]
        X = ens.view(M, -1)[:, m_flat]         # [M,N]
    else:
        y = obs.view(-1)                       # [H*W]
        X = ens.view(M, -1)                    # [M,H*W]

    # term 1: mean distance from each member to obs
    # diff_to_obs: [M]
    diff_to_obs = torch.linalg.vector_norm(X - y.unsqueeze(0), dim=1)
    term1 = diff_to_obs.mean()

    # term 2: mean pairwise distance inside ensemble
    # torch.cdist → [M,M], safe because M is small
    pairwise = torch.cdist(X, X, p=2.0)
    term2 = 0.5 * pairwise.mean()

    return term1 - term2


# ================================================================================
# 1c. Variogram score
# ================================================================================

@torch.no_grad()
def variogram_score(
    obs: torch.Tensor,           # [H,W]
    ens: torch.Tensor,           # [M,H,W]
    *,
    mask: Optional[torch.Tensor] = None,
    p: float = 0.5,
    max_pairs: int = 4000,
    seed: int = 0,
) -> torch.Tensor:
    """
    Variogram score (Gneiting et al., 2008) for spatial ensemble fields.

    VS = (1 / K) * Σ_k ( |y_i - y_j|^p - 1/M Σ_m |x_{m,i} - x_{m,j}|^p )^2

    We approximate the full pair sum by drawing up to `max_pairs` random pixel pairs.
    This keeps it O(M * max_pairs) instead of O(M * (HW)^2).
    """
    if obs.dim() != 2 or ens.dim() != 3:
        raise ValueError("variogram_score expects obs=[H,W] and ens=[M,H,W]")

    M, H, W = ens.shape
    device = ens.device
    obs = obs.to(device, ens.dtype)
    ens = ens.to(device, ens.dtype)

    # flatten + mask
    if mask is not None:
        m = _normalize_mask(mask, obs.shape, device=device)
        valid = m.view(-1) # type: ignore
        y = obs.view(-1)[valid]           # [N]
        X = ens.view(M, -1)[:, valid]     # [M,N]
    else:
        y = obs.view(-1)                  # [H*W]
        X = ens.view(M, -1)               # [M,H*W]

    N = y.numel()
    if N <= 1:
        return torch.tensor(0.0, device=device)

    # how many pairs we can realistically sample
    K = min(max_pairs, N * (N - 1) // 2)

    g = torch.Generator(device=device)
    g.manual_seed(seed)

    idx_i = torch.randint(0, N, (K,), generator=g, device=device)
    idx_j = torch.randint(0, N, (K,), generator=g, device=device)
    same = idx_i == idx_j
    if same.any():
        idx_j[same] = (idx_j[same] + 1) % N

    # obs variogram part
    y_diff = (y[idx_i] - y[idx_j]).abs().pow(p)     # [K]

    # ensemble variogram part
    x_i = X[:, idx_i]   # [M,K]
    x_j = X[:, idx_j]   # [M,K]
    ens_diff = (x_i - x_j).abs().pow(p).mean(dim=0)  # [K]

    vs = (y_diff - ens_diff).pow(2).mean()
    return vs

# ================================================================================
# 2. PIT histograms
# ================================================================================

@torch.no_grad()
def pit_values_from_ensemble(
    obs: torch.Tensor, # [B,H,W]
    ens: torch.Tensor, # [B,M,H,W]
    mask: Optional[torch.Tensor] = None, # [B,H,W] or [H,W]
    randomized: bool = True,

) -> torch.Tensor:
    """
        (Randomized) PIT values for empirical ensemble CDF.

        Returns:
            pit: torch.Tensor [N_valid] of PIT values in [0,1]
    """
    obs = _ensure_float(obs)
    ens = _ensure_float(ens)
    B, M, H, W = ens.shape
    device = ens.device

    m = _normalize_mask(mask, torch.Size((B, H, W)), device=device)
    if m is None:
        m = torch.ones((B, H, W), dtype=torch.bool, device=device)

    # flatten valid
    obs_flat = obs[m]                                # [N]
    ens_flat = ens.permute(0, 2, 3, 1)[m]            # [N, M]

    # sort per-location
    ens_sorted, _ = torch.sort(ens_flat, dim=1)      # [N, M]

    less = (ens_sorted < obs_flat.unsqueeze(1)).sum(dim=1)    # [N]
    equal = (ens_sorted == obs_flat.unsqueeze(1)).sum(dim=1)  # [N]

    if randomized:
        U = torch.rand_like(less, dtype=torch.float32)
        pit = (less.float() + U * equal.float()) / float(M)
    else:
        pit = (less.float() + 0.5 * equal.float()) / float(M)

    return pit.cpu()    



# ================================================================================
# 3. Rank histograms
# ================================================================================


@torch.no_grad()
def rank_histogram(
    obs: torch.Tensor, # [B,H,W]
    ens: torch.Tensor, # [B,M,H,W]
    mask: Optional[torch.Tensor] = None, # [B,H,W] or [H,W]
    randomize_ties: bool = True,
) -> torch.Tensor:
    """
        Count-based rank histogram over M+1 bins.
        Returns a CPU tensor of shape [M+1] with counts.
    """
    obs = _ensure_float(obs)
    ens = _ensure_float(ens)
    B, M, H, W = ens.shape
    device = ens.device

    m = _normalize_mask(mask, torch.Size((B, H, W)), device=device)
    if m is None:
        m = torch.ones((B, H, W), dtype=torch.bool, device=device)

    obs_flat = obs[m]                          # [N]
    ens_flat = ens.permute(0, 2, 3, 1)[m]      # [N, M]

    ens_sorted, _ = torch.sort(ens_flat, dim=1)
    less = (ens_sorted < obs_flat.unsqueeze(1)).sum(dim=1).float()
    leq  = (ens_sorted <= obs_flat.unsqueeze(1)).sum(dim=1).float()
    ties = leq - less

    if randomize_ties:
        U = torch.rand_like(less)
        rank = less + U * ties
    else:
        rank = less + 0.5 * ties

    rank = torch.clamp(rank, 0.0, float(M))
    bins = torch.round(rank).to(torch.int64)
    counts = torch.bincount(bins, minlength=M + 1).to(torch.float32)

    return counts.cpu()

def accumulate_rank_histograms(
    hist_list: Sequence[torch.Tensor],
) -> torch.Tensor:
    """
    Sum a list of rank-hist counts (all must have same length).
    """
    acc: Optional[torch.Tensor] = None
    for h in hist_list:
        if acc is None:
            acc = h.clone()
        else:
            acc += h
    return acc if acc is not None else torch.zeros(1, dtype=torch.float32)





# ================================================================================
# 4. Reliability for threshold exceedance
# ================================================================================

def reliability_exceedance_binned(
        obs: torch.Tensor, # [H,W]
        ens: torch.Tensor, # [M,H,W]
        threshold: float,
        *,
        lr_covariate: Optional[torch.Tensor] = None, # e.g LR precipitation, [H,W]
        n_bins: int = 10,
        mask: Optional[torch.Tensor] = None, # [H,W] or [M,H,W]
        return_brier: bool = True,
) -> Dict[str, torch.Tensor]:
    """
        Reliability diagram ingredients for P(X >= threshold)
       
        If LR covariate is provided, compute conditional reliability, P(X >= threshold | [lr_covariate])
        and bins are defined as quantiles over the LR covariate.
        If not LR covariate, bins are equal-width over forecast probability.

        If return_brier=True, also compute Brier score and Brier skill score (climatology is observed frequency).
            Brier score: Brier score is a measure of the accuracy of probabilistic predictions, defined as the 
                         mean squared difference between predicted probabilities and the observed outcomes.
    """
    device = ens.device
    p_hat = (ens >= float(threshold)).float().mean(dim=0)  # [H,W]
    o = (obs.to(device) >= float(threshold)).float()       # [H,W]

    if mask is not None:
        mask2 = _normalize_mask(mask, p_hat.shape, device=device)
        p_hat = p_hat[mask2]
        o = o[mask2]
        cov = lr_covariate[mask2] if lr_covariate is not None else None
    else:
        cov = lr_covariate

    if cov is not None:
        cov = cov.to(device).float().view(-1)
        q = torch.linspace(0, 1, n_bins + 1, device=device)
        edges = torch.quantile(cov, q)
        edges[0]  = cov.min() - 1e-6
        edges[-1] = cov.max() + 1e-6
        which = torch.bucketize(cov, edges) - 1
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
            prob_pred.append(0.0)
            freq_obs.append(0.0)
            continue
        prob_pred.append(p_src[sel].mean().item())
        freq_obs.append(o_src[sel].mean().item())

    out: Dict[str, torch.Tensor] = {
        "bin_center": bin_center.detach().cpu(),
        "prob_pred": torch.tensor(prob_pred, dtype=torch.float32),
        "freq_obs": torch.tensor(freq_obs, dtype=torch.float32),
        "count": torch.tensor(count, dtype=torch.int64),
    }

    if return_brier:
        # Brier decomposition
        o_bar = float(o_src.mean().item()) if o_src.numel() else 0.0
        N = max(int(o_src.numel()), 1)
        rel = 0.0
        res = 0.0
        for k in range(n_bins):
            Nk = int(out["count"][k].item())
            if Nk == 0:
                continue
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

    return out

def aggregate_reliability_bins(
        rel_list: Sequence[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
        Aggregate multiple reliability bin dicts one (all for the same threshold,
        same number of bins) into a single reliability dict by count-weighted average.

        Evaluator will loop over all dates, and then call THIS function to aggregate
        into a single reliability diagram over all dates.
    """
    if len(rel_list) == 0:
        return {}

    # start from first
    base = rel_list[0]
    bin_center = base["bin_center"].clone()
    counts = base["count"].clone().to(torch.float64)  # to avoid rounding while summing
    prob_pred = base["prob_pred"].clone().to(torch.float64) * counts
    freq_obs = base["freq_obs"].clone().to(torch.float64) * counts

    for r in rel_list[1:]:
        c = r["count"].to(torch.float64)
        counts += c
        prob_pred += r["prob_pred"].to(torch.float64) * c
        freq_obs += r["freq_obs"].to(torch.float64) * c

    nz = counts > 0
    prob_pred[nz] /= counts[nz]
    freq_obs[nz] /= counts[nz]

    out = {
        "bin_center": bin_center,
        "prob_pred": prob_pred.to(torch.float32),
        "freq_obs": freq_obs.to(torch.float32),
        "count": counts.to(torch.int64),
    }
    return out




# ================================================================================
# 5. Spread-skill relationship
# ================================================================================

def spread_skill_binned(
    obs: torch.Tensor,             # [H,W]
    ens: torch.Tensor,             # [M,H,W]
    *,
    point_field: Optional[torch.Tensor] = None,  # precomputed deterministic field, e.g. PMM, shape [H,W]
    point: str = "mean",                         # fallback if point_field is None: 'mean' | 'median'
    mask: Optional[torch.Tensor] = None,
    n_bins: int = 10,
) -> Dict[str, torch.Tensor]:
    """
        Spread-skill diagnostic (binned by forecast spread), using a supplied deterministic field (PMM).

        If point_field is provided, uses this directly as the deterministic forecast to compare against observations.
        PMM is already saved by generation module, so no need to recompute here.

        If point_field is None, fall back to either ensemble mean or median, depending on `point` argument.

        Returns a dict with:
            "bin_center": [n_bins] float32 tensor of bin centers (spread)
            "spread": [n_bins] float32 tensor of mean spread per bin
            "skill": [n_bins] float32 tensor of mean skill per bin
            "count": [n_bins] int64 tensor of counts per bin
    """
    device = ens.device
    obs = obs.to(device).to(ens.dtype)
    
    # spread is always the ensemble std
    spread = ens.std(dim=0)  # [H,W]

    # choose point field
    if point_field is not None:
        pt = point_field.to(device).to(ens.dtype)  # [H,W]
    else:
        if point == "median":
            pt = ens.median(dim=0).values  # [H,W]
        else:
            pt = ens.mean(dim=0)  # [H,W]

    # absolute error vs obs
    ae = (pt - obs).abs()
    
    # apply mask if given
    if mask is not None:
        m = _normalize_mask(mask, spread.shape, device=device)
        spread = spread[m]
        ae = ae[m]

    # guard against empty
    if spread.numel() == 0:
        return {
            "bin_center": torch.zeros(n_bins),
            "spread": torch.zeros(n_bins),
            "skill": torch.zeros(n_bins),
            "count": torch.zeros(n_bins, dtype=torch.int64),
        }
    
    # bin by spread quantiles (robust across datasets)
    q = torch.linspace(0, 1, n_bins + 1, device=device)
    edges = torch.quantile(spread, q)
    edges[0] = spread.min() - 1e-9
    edges[-1] = spread.max() + 1e-9
    which = torch.bucketize(spread, edges) - 1  # [N]
    centers = 0.5 * (edges[:-1] + edges[1:])

    # aggregate per bin
    sp_mean, sk_mean, count = [], [], []
    for b in range(n_bins):
        sel = (which == b)
        n = int(sel.sum().item())
        count.append(n)
        if n == 0:
            sp_mean.append(0.0)
            sk_mean.append(0.0)
            continue
        sp_mean.append(spread[sel].mean().item())
        sk_mean.append(ae[sel].mean().item())

    return {
        "bin_center": centers.detach().cpu(),
        "spread": torch.tensor(sp_mean, dtype=torch.float32),
        "skill": torch.tensor(sk_mean, dtype=torch.float32),
        "count": torch.tensor(count, dtype=torch.int64),
    }