"""
    Scale-dependent metrics for evaluating precipitation fields.

    This module is metrics only.
    No plotting or data loading/saving.

    It provides:
        - Computation of Isotropic PSD (Power Spectral Density) for HR, LR, and generated fields.
        - Comparison of PSDs between generated and reference fields.
        - FSS at multiple scales across thresholds.

    To be called from sbgm/evaluate/evaluate_prcp/eval_scale/evaluate_scale.py
    which will:
        - iterate over available dates via EvalDataResolver
        - load HR / LR / generated fields
        - call these functions
        - aggregate and write to <eval_root>/prcp/scale/tables/... and
          <eval_root>/prcp/scale/figures/...
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence, Dict, Tuple

import numpy as np
import torch
import logging

import torch.nn.functional as F

logger = logging.getLogger(__name__)






# ================================================================================
# 1. FSS at multiple scales
# ================================================================================

def _to_tensor(x):
    return x if isinstance(x, torch.Tensor) else torch.tensor(x)

@torch.no_grad()
def compute_fss_at_scales(
    gen_bt: torch.Tensor, hr_bt: torch.Tensor, *, mask: torch.Tensor | None,
    grid_km_per_px: float, fss_km: list[float], thr_mm: float, eps: float = 1e-8
) -> dict[str, float]:
    """
    Fractions Skill Score (FSS) for exceedance over thr_mm at different scales.
    Uses MASKED pooling so ocean/invalid pixels do not dilute coastal boxes.
    gen_bt, hr_bt: [B,1,H,W] (mm/day), mask: [B,1,H,W] (1=valid, 0=invalid) or None.
    """
    gen_bt = _to_tensor(gen_bt).float()
    hr_bt  = _to_tensor(hr_bt).float()

    X = (gen_bt > thr_mm).float()   # binary exceedance
    Y = (hr_bt  > thr_mm).float()

    if mask is not None:
        M = _to_tensor(mask).float().clamp(0, 1)
    else:
        # treat whole domain as valid
        M = torch.ones_like(X)

    out: dict[str, float] = {}

    for km in fss_km:
        rad_px = int(max(1, round(float(km) / float(grid_km_per_px))))
        ksize  = 2 * rad_px + 1
        area   = float(ksize * ksize)

        # masked sums in the box
        Xm = F.avg_pool2d(X * M, kernel_size=ksize, stride=1, padding=rad_px) * area
        Ym = F.avg_pool2d(Y * M, kernel_size=ksize, stride=1, padding=rad_px) * area
        Mm = F.avg_pool2d(M,      kernel_size=ksize, stride=1, padding=rad_px) * area

        # masked box-averages
        Xs = Xm / (Mm + eps)
        Ys = Ym / (Mm + eps)

        # global (masked) means for MSE and normalization
        msum = M.sum().clamp_min(1.0)
        num  = (((Xs - Ys) ** 2) * M).sum() / msum                      # ⟨(Xs−Ys)^2⟩_mask
        den  = (((Xs ** 2 + Ys ** 2) * M).sum() / msum) + eps           # ⟨Xs^2 + Ys^2⟩_mask + ε

        fss = 1.0 - num / den
        out[f"{int(km)}km"] = float(torch.clamp(fss, 0.0, 1.0))
    return out

@torch.no_grad()
def compute_iss_at_scales(
    gen_bt: torch.Tensor,          # [B,1,H,W] generated / PMM
    hr_bt: torch.Tensor,           # [B,1,H,W] HR / DANRA
    *,
    mask: torch.Tensor | None,
    grid_km_per_px: float,
    iss_scales_km: list[float],
    thr_mm: float,
    eps: float = 1e-8,
) -> dict[str, float]:
    """
    Intensity-Scale Skill (ISS) following the Casati-style idea, but using the same
    neighbourhood/box-filter machinery as FSS.

    Steps per scale:
      1) binarize at thr_mm  -> X, Y in {0,1}
      2) smooth with box of size corresponding to scale
      3) compute MSE_mod = <(Xs - Ys)^2>
      4) compute base rates p_f = <Xs>, p_o = <Ys>
      5) compute MSE_rand = p_f + p_o - 2*p_f*p_o
      6) ISS = 1 - MSE_mod / MSE_rand

    If MSE_rand is tiny (e.g. almost no rain), we return ISS = 1.0 for that scale.
    """
    gen_bt = gen_bt.float()
    hr_bt = hr_bt.float()

    if mask is not None:
        mask = mask.bool()

    # 1) binary exceedance
    X = (gen_bt > thr_mm).float()
    Y = (hr_bt > thr_mm).float()

    out: dict[str, float] = {}

    for km in iss_scales_km:
        rad_px = int(max(1, round(float(km) / float(grid_km_per_px))))
        k = 2 * rad_px + 1
        area = float(k * k)

        # --- masked box sums, then masked averages (identical idea as FSS) ---
        if mask is not None:
            m = mask.float()
        else:
            m = torch.ones_like(X)

        Xm = F.avg_pool2d(X * m, kernel_size=k, stride=1, padding=rad_px) * area
        Ym = F.avg_pool2d(Y * m, kernel_size=k, stride=1, padding=rad_px) * area
        Mm = F.avg_pool2d(m,     kernel_size=k, stride=1, padding=rad_px) * area

        Xs = Xm / (Mm + eps)   # masked box-average exceedance “fractions”
        Ys = Ym / (Mm + eps)

        # Global masked means for MSE_mod and base rates
        msum   = m.sum().clamp_min(1.0)
        mse_mod = (((Xs - Ys) ** 2) * m).sum() / msum
        p_f     = (Xs * m).sum() / msum
        p_o     = (Ys * m).sum() / msum

        mse_rand = p_f + p_o - 2.0 * p_f * p_o          # ≥ 0
        mse_rand = torch.clamp(mse_rand, min=0.02)      # guard very dry days

        iss = 1.0 - mse_mod / (mse_rand + eps)
        out[f"{int(km)}km"] = float(torch.clamp(iss, 0.0, 1.0))

    return out




# ================================================================================
# 2a. Isotropic PSD computation and comparison
# ================================================================================

@torch.no_grad()
def compute_isotropic_psd_batched(
    batch: torch.Tensor,                # [B,1,H,W] in physical units (e.g., mm/day)
    dx_km: float,                       # grid spacing in km
    mask: Optional[torch.Tensor] = None,# [B,1,H,W] or broadcastable
    *,
    window: str = "hann",               # window type for FFT: None | "hann"
    detrend: str = "mean",              # detrending method: "mean" | "none"
    normalize: str = "none",            # normalization method: "none" | "per_field" | "match_ref"
    ref_power: Optional[torch.Tensor] = None,
    max_k: float | None = None,
) -> Dict[str, torch.Tensor]:
    """
        Batched isotropic PSD computation with:
            - optional masking
            - optional mean-detrending (per field)
            - optional windowing
            - CI from sample std
            - Nyquist returned
        
        Returns dict with:
            {
                "k": [N] wavenumbers in cycles per km,
                "psd": [N] mean isotropic PSD,
                "psd_std": [N] stddev of isotropic PSD,
                "psd_n": [N] normalized isotropic PSD (if normalize != "none"),
                "psd_ci_lo": [N] lower 95% CI,
                "psd_ci_hi": [N] upper 95% CI,
                "nyquist": scalar tensor
            }
    """
    if batch.dim() != 4 or batch.shape[1] != 1:
        raise ValueError(f"batch must be [B,1,H,W], got {tuple(batch.shape)}")

    B, _, H, W = batch.shape
    device = batch.device
    x = batch.to(torch.float32).clone()

    # 1) apply mask -> zero out invalid
    if mask is not None:
        m = mask
        while m.dim() < 4:
            m = m.unsqueeze(0)
        if m.shape[0] == 1 and B > 1:
            m = m.expand(B, -1, -1, -1)
        m = m.to(device).to(torch.bool)
        x = x.masked_fill(~m, 0.0)

    # 2) drop NaNs/inf -> zero 
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # 3) detrend: remove mean per sample (AFTER masking & NaN cleanup)
    if detrend == "mean":
        # mean over spatial dims, keeping batch/channel
        mean_per_sample = x.mean(dim=(-1, -2), keepdim=True)
        x = x - mean_per_sample
    elif detrend == "none":
        pass
    else:
        raise ValueError(f"Unsupported detrend='{detrend}'")

    # 4) optional window
    if window is not None:
        if window == "hann":
            wy = torch.hann_window(H, device=device).view(1, 1, H, 1)
            wx = torch.hann_window(W, device=device).view(1, 1, 1, W)
            w2d = wy * wx
            x = x * w2d
        else:
            raise ValueError(f"Unsupported window='{window}'")

    # 5) FFT -> power
    # rfft2 because fields are real; we keep 'ortho' 
    X = torch.fft.rfft2(x, norm="ortho")              # [B,1,H,W//2+1]
    P2 = (X.real**2 + X.imag**2).squeeze(1)           # [B,H,W//2+1]

    # 6) build radial k-grid (cycles/km)
    ky = torch.fft.fftfreq(H, d=dx_km, device=device)      # [H]
    kx = torch.fft.rfftfreq(W, d=dx_km, device=device)     # [W//2+1]
    Ky, Kx = torch.meshgrid(ky, kx, indexing="ij")         # [H, W//2+1]
    Kr = torch.sqrt(Kx**2 + Ky**2)                         # [H, W//2+1]

    kr = Kr.flatten()                                      # [H*(W//2+1)]
    kmax_all = kr.max().item()
    n_bins = int(min(H, W) // 2)
    edges = torch.linspace(0.0, kmax_all, n_bins + 1, device=device)
    k_centers = 0.5 * (edges[:-1] + edges[1:])             # [N]

    which = torch.bucketize(kr, edges) - 1                 # [H*(W//2+1)]
    
    # 7) radial averaging per sample
    psd_per_sample = []
    for b in range(B):
        p = P2[b].flatten()                                 # [H*(W//2+1)]
        psd_b = torch.zeros(n_bins, device=device)
        for i in range(n_bins):
            sel = (which == i)
            if sel.any():
                psd_b[i] = p[sel].mean()
            else:
                psd_b[i] = 0.0

        # optional per-field normalization (shape only)
        if normalize == "per_field":
            tot = psd_b.sum()
            if tot > 0:
                psd_b = psd_b / tot

        psd_per_sample.append(psd_b)

    psd_all = torch.stack(psd_per_sample, dim=0)  # [B, N]
    psd_mean = psd_all.mean(dim=0)                # [N]

    if B > 1:
        psd_std = psd_all.std(dim=0, unbiased=True)
    else:
        # only one field in the batch: no spread → std=0, CI=mean
        psd_std = torch.zeros_like(psd_mean)

    psd_n = torch.full_like(psd_mean, float(B))

    # 8) match_ref normalization (after averaging)
    if normalize == "match_ref":
        if ref_power is None:
            raise ValueError("normalize='match_ref' but ref_power is None")
        ref_power = ref_power.to(device).to(psd_mean.dtype)
        # simple least-squares scaling
        num = (psd_mean * ref_power).sum()
        den = (psd_mean * psd_mean).sum() + 1e-12
        scale = num / den
        psd_mean = psd_mean * scale
        psd_std = psd_std * scale

    # 9) 95% CI
    if B > 1:
        se = psd_std / torch.sqrt(psd_n.clamp(min=1.0))
    else:
        se = torch.zeros_like(psd_std)

    ci_lo = psd_mean - 1.96 * se
    ci_hi = psd_mean + 1.96 * se

    # 10) cut to Nyquist / user max
    nyq = 1.0 / (2.0 * dx_km)  # cycles per km
    if max_k is not None:
        keep = (k_centers <= float(max_k))
    else:
        keep = (k_centers <= nyq + 1e-9)

    k_out = k_centers[keep].detach().cpu()
    psd_out = psd_mean[keep].detach().cpu()
    psd_std_out = psd_std[keep].detach().cpu()
    psd_n_out = psd_n[keep].detach().cpu()
    ci_lo_out = ci_lo[keep].detach().cpu()
    ci_hi_out = ci_hi[keep].detach().cpu()

    return {
        "k": k_out,
        "psd": psd_out,
        "psd_std": psd_std_out,
        "psd_n": psd_n_out,
        "psd_ci_lo": ci_lo_out,
        "psd_ci_hi": ci_hi_out,
        "nyquist": torch.tensor(nyq),
    }


# Helper: run PSD on a single 2D field (so we don't always have to build [B,1,H,W])
@torch.no_grad()
def psd_from_2d(
    field_2d: torch.Tensor | np.ndarray,
    *,
    dx_km: float,
    mask_2d: Optional[torch.Tensor | np.ndarray] = None,
    window: str = "hann",
    detrend: str = "mean",
    normalize: str = "none",
    max_k: float | None = None,
) -> Dict[str, torch.Tensor]:
    # --- normalize field to [H, W] ---
    if isinstance(field_2d, np.ndarray):
        field_2d = torch.from_numpy(field_2d)
    field_2d = field_2d.to(torch.float32)

    # squeeze away leading singleton dims: [1,1,H,W] -> [H,W], [1,H,W] -> [H,W]
    while field_2d.dim() > 2 and field_2d.shape[0] == 1:
        field_2d = field_2d.squeeze(0)
    # after first squeeze we can have [1,H,W] again
    if field_2d.dim() > 2 and field_2d.shape[0] == 1:
        field_2d = field_2d.squeeze(0)

    if field_2d.dim() != 2:
        raise ValueError(
            f"psd_from_2d expects [H,W] or singleton-broadcastable dims, got {tuple(field_2d.shape)}"
        )

    H, W = field_2d.shape
    batch = field_2d.view(1, 1, H, W)

    # --- normalize mask too ---
    if mask_2d is not None:
        if isinstance(mask_2d, np.ndarray):
            mask_2d = torch.from_numpy(mask_2d)
        mask_2d = mask_2d.to(torch.bool)
        # same squeezing logic
        while mask_2d.dim() > 2 and mask_2d.shape[0] == 1:
            mask_2d = mask_2d.squeeze(0)
        if mask_2d.dim() == 2:
            mask_2d = mask_2d.view(1, 1, H, W)
    else:
        mask_2d = None

    return compute_isotropic_psd_batched(
        batch,
        dx_km=dx_km,
        mask=mask_2d,
        window=window,
        detrend=detrend,
        normalize=normalize,
        ref_power=None,
        max_k=max_k,
    )


# Helper: interpolate one PSD onto another PSD's k-grid
def align_psd_on_k(
    src_psd: Dict[str, torch.Tensor],
    tgt_k: torch.Tensor | np.ndarray,
) -> np.ndarray:
    """
    src_psd: dict from compute_isotropic_psd_batched
    tgt_k:   target k-grid (typically HR k) to interpolate onto

    Returns: np.ndarray of PSD values on tgt_k
    """
    k_src = src_psd["k"].detach().cpu().numpy()
    P_src = src_psd["psd"].detach().cpu().numpy()
    k_tgt = np.asarray(tgt_k, dtype=np.float64)
    # simple 1D linear interpolation, clamp outside with 0
    P_tgt = np.interp(k_tgt, k_src, P_src, left=0.0, right=0.0)
    return P_tgt


# ================================================================================
# 2b. PSD comparison on common k-grid (HR as reference)
# ================================================================================


def compare_psd_triplet(
    *,
    hr_psd: Dict[str, torch.Tensor],
    gen_psd: Optional[Dict[str, torch.Tensor]] = None,
    lr_psd: Optional[Dict[str, torch.Tensor]] = None,
    low_k_max: float = 1.0 / 200.0,   # "preservation" band (>= 200 km)
    high_k_min: float = 1.0 / 20.0,   # "fine-scale" band (<= 20 km)
) -> Dict[str, float | np.ndarray]:
    """
    Compare PSDs of HR, generated, and (optionally) LR on a *common* k-grid.

    - take HR's k-grid as the reference.
    - interpolate GEN and LR onto that grid.
    - integrate power in a low-k and high-k band.
    - return simple ratios that you can tabulate and plot.

    ALSO return lr_nyquist so the plotter can 'ghost' LR above that.
    """
    # --- reference grid
    k_hr = hr_psd["k"].detach().cpu().numpy()
    P_hr = hr_psd["psd"].detach().cpu().numpy()

    # initialize outputs
    out: Dict[str, float | np.ndarray] = {
        "k": k_hr,
        "psd_hr": P_hr,
        "hr_nyquist": float(hr_psd.get("nyquist", torch.tensor(np.max(k_hr))).item()),
    }

    # --- generated
    if gen_psd is not None:
        P_gen = align_psd_on_k(gen_psd, k_hr)
        out["psd_gen"] = P_gen
    else:
        P_gen = None

    # --- low-res
    if lr_psd is not None:
        P_lr = align_psd_on_k(lr_psd, k_hr)
        lr_nyq = float(lr_psd.get("nyquist", torch.tensor(0.0)).item())
        out["psd_lr"] = P_lr
        out["lr_nyquist"] = lr_nyq
    else:
        P_lr = None
        lr_nyq = None

    # ---------- integrate bands ----------
    def _band_int(k, P, kmin, kmax):
        m = (k >= kmin) & (k <= kmax)
        if not np.any(m):
            return np.nan
        return float(np.trapz(P[m], k[m]))

    # HR low/high
    hr_low = _band_int(k_hr, P_hr, 0.0, low_k_max)
    hr_high = _band_int(k_hr, P_hr, high_k_min, k_hr.max())
    out["hr_lowk_power"] = hr_low
    out["hr_highk_power"] = hr_high

    # GEN low/high
    if P_gen is not None:
        gen_low = _band_int(k_hr, P_gen, 0.0, low_k_max)
        gen_high = _band_int(k_hr, P_gen, high_k_min, k_hr.max())
        out["gen_lowk_power"] = gen_low
        out["gen_highk_power"] = gen_high

        # how well does GEN preserve HR low-k?
        if hr_low > 0:
            out["gen_lowk_vs_hr"] = gen_low / hr_low
        # how well does GEN match HR fine-scale?
        if hr_high > 0:
            out["gen_highk_vs_hr"] = gen_high / hr_high

    # LR low/high
    if P_lr is not None:
        lr_low = _band_int(k_hr, P_lr, 0.0, low_k_max)
        lr_high = _band_int(k_hr, P_lr, high_k_min, k_hr.max())
        out["lr_lowk_power_on_hrgrid"] = lr_low
        out["lr_highk_power_on_hrgrid"] = lr_high
        # how well does GEN preserve LR large scales? (your “low-k LR preservation”)
        if P_gen is not None and lr_low > 0:
            out["gen_lowk_vs_lr"] = out["gen_lowk_power"] / lr_low  # type: ignore

    # keep the bands so plotter can re-use
    out["low_k_max"] = float(low_k_max)
    out["high_k_min"] = float(high_k_min)

    return out