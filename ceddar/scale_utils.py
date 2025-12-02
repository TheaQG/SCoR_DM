# sbgm/scale_utils.py
"""
    Utilities for scale analysis, e.g., power spectral density (PSD) computations,
    finding intersection scales between high-res and low-res data, and mapping
    spatial scales to Gaussian blur sigma or diffusion time t* for VE SDEs.
        1. Compute isotrocpic PSDs of LR vs HR on DK grid (optionally land only)
        2. Find the intersection wavenumber k* (or wavelength λ*) where HR overtakes LR
        3. Map either a user-requested preserve_scale_km or the found λ* to a Gaussian 
           sigma* that can be plugged into EDM training and sampling as noise level
        (3. For VE SDEs, map sigma* to t* via inversion of sigma(t) = marginal_prob_std_fn(t))
    
    What sigmaØ will control: 
        - For EDM sampling: Set the minimum noise in the sampler to sigma* (sigma_min=max(sigma_min,sigma*))
          Effect: Will preserve large-scale content (features/details) above scale ~ λ* and let the stochastic
          refinement add only sub-scales below λ*.
"""
import math
import numpy as np
import torch
import torch.fft as fft

def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x)

def detrend2d(x: np.ndarray) -> np.ndarray:
    # Remove mean (cheap & robust). If you want linear detrend, add it later.
    return x - np.nanmean(x)

def hann2d(h, w):
    hy = np.hanning(h)
    hx = np.hanning(w)
    return np.outer(hy, hx)

def isotropic_psd(field_2d: np.ndarray,
                  pixel_km: float,
                  window: str = "hann",
                  eps: float = 1e-12):
    """
    Compute isotropic 2D power spectral density and collapse to radial bins.
    Returns (k_cpkm, Pk) where k in cycles per km.
    """
    x = np.array(field_2d, dtype=np.float64)
    mask = np.isfinite(x)
    if not mask.all():
        x = np.where(mask, x, 0.0)

    x = detrend2d(x)
    H, W = x.shape

    if window == "hann":
        w2 = hann2d(H, W)
        xw = x * w2
    else:
        xw = x

    # 2D FFT, shift zero-freq to center
    F = fft.fftshift(fft.fft2(torch.from_numpy(xw))).numpy()
    P2d = (F * np.conj(F)).real / (H * W + eps)

    # Build frequency grids in cycles per km
    dkx = 1.0 / (W * pixel_km)
    dky = 1.0 / (H * pixel_km)
    kx = (np.arange(W) - W//2) * dkx
    ky = (np.arange(H) - H//2) * dky
    KX, KY = np.meshgrid(kx, ky)
    Kr = np.sqrt(KX**2 + KY**2)  # cycles per km

    # Radial binning
    kmax = Kr.max()
    nbins = int(np.ceil(np.sqrt(H*H + W*W)/2))
    kbins = np.linspace(0.0, kmax, nbins+1)
    kvals = 0.5 * (kbins[:-1] + kbins[1:])
    Pvals = np.zeros_like(kvals)
    for i in range(nbins):
        sel = (Kr >= kbins[i]) & (Kr < kbins[i+1])
        if np.any(sel):
            Pvals[i] = np.nanmean(P2d[sel])
        else:
            Pvals[i] = np.nan

    # Clean NaNs at tails
    good = np.isfinite(Pvals) & (Pvals > 0)
    return kvals[good], Pvals[good]

def find_intersection_k(k_hr, P_hr, k_lr, P_lr, smooth_gamma: float = 0.0):
    """
    Find k* where HR and LR PSDs intersect (HR overtakes LR).
    We look for the first sign change of log-ratio or the minimal |ratio-1|.
    """
    # Interpolate both onto a common k-grid
    kmin = max(min(k_hr), min(k_lr))
    kmax = min(max(k_hr), max(k_lr))
    if kmax <= kmin:
        return None
    k_common = np.linspace(kmin, kmax, 512)
    P_hr_i = np.interp(k_common, k_hr, P_hr)
    P_lr_i = np.interp(k_common, k_lr, P_lr)

    if smooth_gamma > 0:
        # Optional gentle smoothing in log space to reduce noise
        P_hr_i = np.exp(np.convolve(np.log(P_hr_i+1e-20), np.ones(5)/5, mode='same'))
        P_lr_i = np.exp(np.convolve(np.log(P_lr_i+1e-20), np.ones(5)/5, mode='same'))

    ratio = (P_hr_i + 1e-20) / (P_lr_i + 1e-20)

    # Prefer actual crossing near 1
    sign = np.sign(np.log(ratio))
    idx = np.where(np.diff(sign) != 0)[0]
    if len(idx) > 0:
        # choose the first crossing (from low to high k)
        i = int(idx[0])
        # linear interpolate around i
        r0, r1 = ratio[i], ratio[i+1]
        t = (1 - r0) / (r1 - r0 + 1e-12)
        t = np.clip(t, 0.0, 1.0)
        k_star = (1 - t) * k_common[i] + t * k_common[i+1]
        return float(k_star)

    # Fallback: closest to 1
    i = int(np.argmin(np.abs(ratio - 1.0)))
    return float(k_common[i])

def wavelength_km_from_k_cpkm(k_cpkm: float) -> float:
    """λ = 1/k if k in cycles per km."""
    if k_cpkm <= 0:
        return float('inf')
    return 1.0 / k_cpkm

def sigma_star_from_preserve_scale(preserve_scale_km: float,
                                   pixel_km: float,
                                   c: float = 1.0) -> float:
    """
    Map a spatial scale (km) to a Gaussian blur-equivalent σ* in pixels, then reuse σ* as a
    convenient EDM min-noise. For a Gaussian, the 50% MTF cutoff is ~ k_c = 1/(2πσ_px).
    So σ_px ≈ 1 / (2π k_px) = λ_px / (2π). We allow a tunable factor c∈[0.7,1.3].
    """
    if preserve_scale_km <= 0:
        return 0.0
    # wavelength in pixels
    lambda_px = preserve_scale_km / pixel_km
    sigma_px = c * (lambda_px / (2.0 * math.pi))
    # Return σ* in "pixel units" – for EDM you can treat it as sigma_min proxy.
    return float(max(0.0, sigma_px))

def t_star_from_sigma_ve(sigma_target: float,
                         marginal_prob_std_fn,
                         t_lo: float = 1e-5,
                         t_hi: float = 1.0) -> float:
    """
    Invert sigma(t)≈marginal_prob_std_fn(t) to get t* for a VE process.
    Simple bisection; assumes sigma(t) is monotone increasing.
    """
    target = float(sigma_target)
    lo, hi = t_lo, t_hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        s_mid = float(marginal_prob_std_fn(torch.tensor([mid])).item())
        if s_mid < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)