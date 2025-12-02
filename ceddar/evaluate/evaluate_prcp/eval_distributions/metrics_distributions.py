# sbgm/evaluate/evaluate_prcp/eval_distributional/metrics_distributional.py
from __future__ import annotations
from pathlib import Path
from typing import Sequence, Optional, Dict, Any, List
import numpy as np
import torch

import logging
logger = logging.getLogger(__name__)

# Module-level helper for safe capping of arrays
def _cap_array(x: np.ndarray | None, cap: int, *, rng: np.random.Generator | None = None) -> np.ndarray | None:
    """Return at most `cap` elements from 1D array `x`.
    If `x` is None or shorter than `cap`, return it unchanged.
    Sampling is done from the actual length of `x` to avoid out-of-bounds.
    """
    if x is None:
        return None
    x = np.asarray(x)
    n = x.shape[0]
    if cap is None or cap <= 0 or n <= cap:
        return x
    if rng is None:
        rng = np.random.default_rng()
    idx = rng.choice(n, size=cap, replace=False)
    return x[idx]

def _flatten_valid_torch(x: torch.Tensor, mask: Optional[torch.Tensor]) -> np.ndarray:
    # x: [H,W], mask: [H,W] bool
    x = x.float()
    if mask is not None:
        mask = (mask > 0.5)
        x = x[mask]
    return x.detach().cpu().numpy().astype(np.float32)


def collect_pooled_distributions(
    *,
    resolver,
    dates: Sequence[str],
    include_lr: bool,
    n_bins: int,
    vmax_percentile: float,
    save_samples_cap: int,
) -> Dict[str, Any]:
    hr_all: List[np.ndarray] = []
    gen_all: List[np.ndarray] = []
    lr_all: List[np.ndarray] = []

    for d in dates:
        hr = resolver.load_obs(d)
        gen = resolver.load_pmm(d)
        m = resolver.load_mask(d)
        if hr is None or gen is None:
            continue
        if not torch.is_tensor(hr):  hr = torch.from_numpy(np.asarray(hr))
        if not torch.is_tensor(gen): gen = torch.from_numpy(np.asarray(gen))
        if m is not None and not torch.is_tensor(m): m = torch.from_numpy(np.asarray(m))

        hr = hr.squeeze()
        gen = gen.squeeze()
        if m is not None: m = m.squeeze()

        hr_all.append(_flatten_valid_torch(hr, m))
        gen_all.append(_flatten_valid_torch(gen, m))

        if include_lr:
            try:
                lr = resolver.load_lr(d)
            except Exception:
                lr = None
            if lr is not None:
                if not torch.is_tensor(lr):
                    lr = torch.from_numpy(np.asarray(lr))
                lr = lr.squeeze()
                # try to broadcast mask
                lr_all.append(_flatten_valid_torch(lr, m))

    if not hr_all:
        raise RuntimeError("No valid HR pixels collected for distributional evaluation.")

    hr_vec = np.concatenate(hr_all, axis=0)
    gen_vec = np.concatenate(gen_all, axis=0) if gen_all else np.empty((0,), dtype=np.float32)
    lr_vec  = np.concatenate(lr_all, axis=0)  if lr_all  else None

    # cap to at most `save_samples_cap`, but sample from the actual length of each vector
    rng = np.random.default_rng(504)
    hr_vec = _cap_array(hr_vec, save_samples_cap, rng=rng)
    gen_vec = _cap_array(gen_vec, save_samples_cap, rng=rng)
    if include_lr and lr_vec is not None and lr_vec.size > 0:
        lr_vec = _cap_array(lr_vec, save_samples_cap, rng=rng)

    if hr_vec is None or hr_vec.size == 0:
        logger.warning("[distributions] HR pooled vector ended up empty after capping/filtering – downstream plots may skip HR.")
    if gen_vec is None or gen_vec.size == 0:
        logger.warning("[distributions] GEN pooled vector ended up empty after capping/filtering – downstream plots may skip GEN.")

    # Define bin range from HR only to avoid LR/GEN capping the tail; include full HR max
    if hr_vec is not None and hr_vec.size > 0:
        # If the user asked for a percentile < 100, use that as a soft suggestion,
        # but never below the true max so we keep the HR tail visible.
        hr_pctl = np.percentile(hr_vec, vmax_percentile) if (0 < vmax_percentile <= 100) else hr_vec.max()
        vmax = max(float(hr_pctl), float(np.max(hr_vec)))
    else:
        vmax = 1.0
    # tiny epsilon to make sure the rightmost edge contains the max
    vmax = max(vmax, 1.0)
    bins = np.linspace(0.0, vmax + 1e-6, n_bins + 1, dtype=np.float32)

    hr_hist, _ = np.histogram(hr_vec if hr_vec is not None else np.empty((0,), dtype=np.float32), bins=bins)
    gen_hist, _ = np.histogram(gen_vec if gen_vec is not None else np.empty((0,), dtype=np.float32), bins=bins)
    lr_hist = None
    if lr_vec is not None:
        lr_hist, _ = np.histogram(lr_vec if lr_vec is not None else np.empty((0,), dtype=np.float32), bins=bins)

    return {
        "bins": bins,
        "hr_hist": hr_hist,
        "gen_hist": gen_hist,
        "lr_hist": lr_hist,
        "hr_vec": hr_vec,
        "gen_vec": gen_vec,
        "lr_vec": lr_vec,
    }


# NEW: Per-day histograms for uncertainty bands
def collect_daily_histograms(
    *,
    resolver,
    dates: Sequence[str],
    include_lr: bool,
    bins: np.ndarray,
) -> Dict[str, Any]:
    """
    Build per-day histograms on a fixed set of global `bins`.
    Returns dict with:
      - counts_hr [D,B], counts_gen [D,B], counts_lr [D,B] (optional)
      - n_hr [D], n_gen [D], n_lr [D] (optional)
      - dates [D]
    """
    B = int(len(bins) - 1)
    counts_hr: List[np.ndarray] = []
    counts_gen: List[np.ndarray] = []
    counts_lr: List[np.ndarray] = []
    n_hr: List[int] = []
    n_gen: List[int] = []
    n_lr: List[int] = []
    out_dates: List[str] = []

    for d in dates:
        hr = resolver.load_obs(d)
        gen = resolver.load_pmm(d)
        m = resolver.load_mask(d)
        if hr is None or gen is None:
            continue
        if not torch.is_tensor(hr):  hr = torch.from_numpy(np.asarray(hr))
        if not torch.is_tensor(gen): gen = torch.from_numpy(np.asarray(gen))
        if m is not None and not torch.is_tensor(m): m = torch.from_numpy(np.asarray(m))
        hr = hr.squeeze(); gen = gen.squeeze()
        if m is not None: m = m.squeeze()

        hr_vec = _flatten_valid_torch(hr, m)
        gen_vec = _flatten_valid_torch(gen, m)

        H_hr, _ = np.histogram(hr_vec, bins=bins)
        H_gen, _ = np.histogram(gen_vec, bins=bins)

        counts_hr.append(H_hr.astype(np.int64))
        counts_gen.append(H_gen.astype(np.int64))
        n_hr.append(int(hr_vec.size))
        n_gen.append(int(gen_vec.size))

        if include_lr:
            try:
                lr = resolver.load_lr(d)
            except Exception:
                lr = None
            if lr is not None:
                if not torch.is_tensor(lr):
                    lr = torch.from_numpy(np.asarray(lr))
                lr = lr.squeeze()
                lr_vec = _flatten_valid_torch(lr, m)
                H_lr, _ = np.histogram(lr_vec, bins=bins)
                counts_lr.append(H_lr.astype(np.int64))
                n_lr.append(int(lr_vec.size))
            else:
                # keep alignment with dates by appending zeros
                counts_lr.append(np.zeros(B, dtype=np.int64))
                n_lr.append(0)

        out_dates.append(str(d))

    out: Dict[str, Any] = {
        "bins": np.asarray(bins),
        "counts_hr": np.stack(counts_hr, axis=0) if counts_hr else np.zeros((0, B), dtype=np.int64),
        "counts_gen": np.stack(counts_gen, axis=0) if counts_gen else np.zeros((0, B), dtype=np.int64),
        "n_hr": np.asarray(n_hr, dtype=np.int64),
        "n_gen": np.asarray(n_gen, dtype=np.int64),
        "dates": np.asarray(out_dates, dtype="U"),
    }
    if include_lr:
        # If LR was never present, counts_lr may be empty lists; normalize shape
        if any(c is not None for c in counts_lr):
            out["counts_lr"] = np.stack(counts_lr, axis=0)
            out["n_lr"] = np.asarray(n_lr, dtype=np.int64)
    return out


def collect_ensemble_histograms(
    *,
    resolver,
    dates: Sequence[str],
    bins: np.ndarray,
    mode: str = "pool",  # "pool" | "member_mean"
    n_members: Optional[int] = None,
    seed: int = 1234,
) -> Dict[str, Any]:
    """
    Build ensemble-aware histograms on fixed `bins`.
    Returns a dict with keys depending on `mode`:
      - common: counts_members [M,B], n_members [M]
      - if mode=="pool": counts_pool [B]
      - if mode=="member_mean": pdf_mean [B], pdf_q10 [B], pdf_q50 [B], pdf_q90 [B]
    Memory-safe: streams per date and accumulates per-member counts.
    """
    # Ensure bins as np.ndarray (monotone, length B+1)
    bins = np.asarray(bins)
    B = int(len(bins) - 1)

    # First, determine effective number of members by peeking at first available date
    M_eff = None
    for d in dates:
        try:
            # Prefer unified fetch if available
            sample = resolver.fetch(d, want_ensemble=True, n_members=n_members, seed=seed)  # type: ignore[attr-defined]
            ens = getattr(sample, "ens", None)
        except Exception:
            # Fallback to legacy API
            ens = resolver.load_ens(d)
        if ens is not None:
            if torch.is_tensor(ens):
                M_eff = int(ens.shape[0])
            else:
                M_eff = int(np.asarray(ens).shape[0])
            break
    if M_eff is None:
        logger.warning("[distributions] No ensemble members found across dates – skipping ensemble histograms.")
        return {}

    counts_members = np.zeros((M_eff, B), dtype=np.int64)
    npix_members = np.zeros((M_eff,), dtype=np.int64)

    for d in dates:
        # Load ensemble + mask
        ens = None
        msk = None
        try:
            sample = resolver.fetch(d, want_ensemble=True, n_members=n_members, seed=seed)  # type: ignore[attr-defined]
            ens = getattr(sample, "ens", None)
            msk = getattr(sample, "mask", None)
        except Exception:
            try:
                ens = resolver.load_ens(d)
                msk = resolver.load_mask(d)
            except Exception:
                ens = None
        if ens is None:
            continue

        # Normalize types
        if torch.is_tensor(msk):
            msk_np = (msk > 0.5).detach().cpu().numpy() if msk is not None else None
        elif msk is None:
            msk_np = None
        else:
            msk_np = np.asarray(msk) > 0

        ens_np = ens.detach().cpu().numpy() if torch.is_tensor(ens) else np.asarray(ens)

        m_here = min(M_eff, ens_np.shape[0])
        for mi in range(m_here):
            fld = ens_np[mi]
            if msk_np is not None:
                try:
                    fld = fld[msk_np]
                except Exception:
                    fld = fld.reshape(-1)[msk_np.reshape(-1)]
            else:
                fld = fld.reshape(-1)
            H, _ = np.histogram(fld, bins=bins)
            counts_members[mi] += H.astype(np.int64)
            npix_members[mi] += int(fld.size)

    out: Dict[str, Any] = {
        "bins": bins,
        "counts_members": counts_members,
        "n_members": npix_members,
        "mode": str(mode),
    }

    if mode == "pool":
        out["counts_pool"] = counts_members.sum(axis=0)
    elif mode == "member_mean":
        eps = 1.0
        denom = np.maximum(npix_members.astype(np.float64), eps)[:, None]
        pdf_members = counts_members.astype(np.float64) / denom  # [M,B]
        out["pdf_mean"] = pdf_members.mean(axis=0)
        out["pdf_q10"] = np.percentile(pdf_members, 10, axis=0)
        out["pdf_q50"] = np.percentile(pdf_members, 50, axis=0)
        out["pdf_q90"] = np.percentile(pdf_members, 90, axis=0)
    else:
        logger.warning(f"[distributions] Unknown ensemble pooling mode '{mode}', valid are 'pool'|'member_mean'.")

    return out


def compute_distributional_metrics(pooled: Dict[str, Any], ensembles: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """    
    Return rows: {ref, comp, wasserstein, ks_stat, ks_p, kl_hr_to_x}
    """
    from scipy.stats import ks_2samp
    from scipy.stats import entropy as kl  # KL(p||q)
    rows: List[Dict[str, Any]] = []

    hr = pooled["hr_vec"]
    gen = pooled["gen_vec"]
    lr  = pooled.get("lr_vec", None)

    # helper to make discrete probs
    def _disc(x: np.ndarray, bins: np.ndarray) -> np.ndarray:
        H, _ = np.histogram(x, bins=bins)
        p = H.astype(np.float64)
        s = max(p.sum(), 1.0)
        p = p / s
        p = np.clip(p, 1e-12, 1.0)
        return p

    # Helper to safely extract Python float from scalar or 0-d array
    def _to_float(x):
        try:
            arr = np.asarray(x)
            if arr.size == 1:
                return float(arr.item())
            # fall back to first element if ks_2samp ever returns vectorized output
            return float(arr.ravel()[0])
        except Exception:
            # last resort – try plain float, and if that fails, return nan
            try:
                return float(x)
            except Exception:
                return float("nan")

    # Helper to safely compute W1 (Wasserstein-1) approximation for 1D arrays
    def _w1_approx(x: np.ndarray, y: np.ndarray) -> float:
        x1 = np.asarray(x).ravel()
        y1 = np.asarray(y).ravel()
        n = int(min(x1.size, y1.size))
        if n == 0:
            return float('nan')
        xs = np.sort(x1)[:n]
        ys = np.sort(y1)[:n]
        return float(np.abs(xs - ys).mean())

    bins = pooled["bins"]

    # --- HR vs GEN (PMM, legacy) ---
    if gen is not None and gen.size:
        ks_res = ks_2samp(hr.ravel(), gen.ravel())
        ks_stat = _to_float(getattr(ks_res, "statistic", ks_res[0]))
        ks_p    = _to_float(getattr(ks_res, "pvalue", ks_res[1]))
        # wasserstein
        w1 = _w1_approx(hr, gen)
        p_hr = _disc(hr, bins)
        p_gen = _disc(gen, bins)
        rows.append({
            "ref": "hr",
            "comp": "gen_pmm",
            "wasserstein": float(w1),
            "ks_stat": ks_stat,
            "ks_p": ks_p,
            "kl_hr_to_x": float(kl(p_hr, p_gen)),
        })

    # --- HR vs LR ---
    if lr is not None and lr.size:
        ks_res = ks_2samp(hr.ravel(), lr.ravel())
        ks_stat = _to_float(getattr(ks_res, "statistic", ks_res[0]))
        ks_p    = _to_float(getattr(ks_res, "pvalue", ks_res[1]))
        w1 = _w1_approx(hr, lr)
        p_hr = _disc(hr, bins)
        p_lr = _disc(lr, bins)
        rows.append({
            "ref": "hr",
            "comp": "lr",
            "wasserstein": float(w1),
            "ks_stat": ks_stat,
            "ks_p": ks_p,
            "kl_hr_to_x": float(kl(p_hr, p_lr)),
        })

    # --- HR vs GEN (ensemble) ---
    if ensembles is not None and isinstance(ensembles, dict) and len(ensembles) > 0:
        mode = ensembles.get("mode", "pool")
        rng = np.random.default_rng(123)
        p_hr = _disc(hr, bins)

        if mode == "pool" and "counts_pool" in ensembles:
            counts = np.asarray(ensembles["counts_pool"]).astype(np.float64)
            p_gen = counts / max(counts.sum(), 1.0)
            mids = 0.5 * (bins[:-1] + bins[1:])
            n_samp = min(hr.size, 200_000)
            samp = rng.choice(mids, size=n_samp, replace=True, p=p_gen)
            ks_res = ks_2samp(hr.ravel(), samp.ravel())
            ks_stat = _to_float(getattr(ks_res, "statistic", ks_res[0]))
            ks_p    = _to_float(getattr(ks_res, "pvalue", ks_res[1]))
            w1 = _w1_approx(hr, samp)
            rows.append({
                "ref": "hr",
                "comp": "gen_ens_pool",
                "wasserstein": float(w1),
                "ks_stat": ks_stat,
                "ks_p": ks_p,
                "kl_hr_to_x": float(kl(p_hr, p_gen)),
            })

        if mode == "member_mean" and "pdf_mean" in ensembles:
            p_gen = np.asarray(ensembles["pdf_mean"]).astype(np.float64)
            p_gen = p_gen / max(p_gen.sum(), 1.0)
            mids = 0.5 * (bins[:-1] + bins[1:])
            n_samp = min(hr.size, 200_000)
            samp = rng.choice(mids, size=n_samp, replace=True, p=p_gen)
            ks_res = ks_2samp(hr.ravel(), samp.ravel())
            ks_stat = _to_float(getattr(ks_res, "statistic", ks_res[0]))
            ks_p    = _to_float(getattr(ks_res, "pvalue", ks_res[1]))
            w1 = _w1_approx(hr, samp)
            rows.append({
                "ref": "hr",
                "comp": "gen_ens_mean",
                "wasserstein": float(w1),
                "ks_stat": ks_stat,
                "ks_p": ks_p,
                "kl_hr_to_x": float(kl(p_hr, p_gen)),
            })

    return rows