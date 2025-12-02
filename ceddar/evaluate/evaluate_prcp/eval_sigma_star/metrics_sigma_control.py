"""
Compute sigma*-dependent evaluation metrics (self-contained):
- LR-GEN correlation in a low-pass band (≤ LR Nyquist), with smooth taper
- PSD slope in a mesoscale band (e.g., 5-20 km), for GEN and HR
- High-k power gain > LR Nyquist: G_high = sum_k>k_Nyq P_GEN(k) / sum_k>k_Nyq P_HR(k)
- CRPS of the ensemble vs HR (Hersbach fair estimator)

All numeric helpers (mask handling, low-pass, PSD) are implemented locally so
this block can evolve independently of other evaluation modules.

Saves:
  - tables/metrics_by_sigma.csv : per-(sigma*, date) rows
  - tables/agg_summary.csv      : per-sigma* aggregates (mean/std for each metric)
  - tables/sigma_psd_curves.npz : k-grid and mean/std PSD curves for HR, LR (upsampled), and GEN per σ*, plus LR Nyquist
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from typing import DefaultDict

import logging
import numpy as np

import torch
import torch.nn.functional as F
import csv
from collections import defaultdict

from evaluate.data_resolver import EvalDataResolver

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# 0) Small helpers
# ------------------------------------------------------------------------------

def _ensure_float(x: torch.Tensor) -> torch.Tensor:
    return x if torch.is_floating_point(x) else x.float()

def _normalize_mask(mask: Optional[torch.Tensor],
                    target_shape: torch.Size,
                    device: torch.device) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    m = mask.to(device)
    if m.dtype != torch.bool:
        m = m > 0.5
    # squeeze leading singleton dims if present
    while m.dim() > 2 and m.shape[0] == 1:
        m = m.squeeze(0)
    # broadcast to target if needed
    if m.shape == target_shape:
        return m
    if m.dim() == 2 and len(target_shape) == 3:
        return m.unsqueeze(0).expand(target_shape[0], -1, -1)
    try:
        return m.expand(target_shape)  # best-effort
    except Exception:
        logger.warning(f"[sigma_control] Could not broadcast mask {tuple(m.shape)} to {tuple(target_shape)}")
        return None

# ------------------------------------------------------------------------------
# 1) CRPS (Hersbach fair estimator) — local implementation
# ------------------------------------------------------------------------------

@torch.no_grad()
def crps_ensemble_local(
    obs: torch.Tensor,              # [H,W]
    ens: torch.Tensor,              # [M,H,W]
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Continuous Ranked Probability Score for ensembles (Hersbach, 2000).
    If reduction="mean", returns scalar; else returns [H,W] map on ens.device.
    """
    if obs.dim() != 2 or ens.dim() != 3:
        raise ValueError("crps_ensemble_local expects obs=[H,W], ens=[M,H,W]")

    obs = _ensure_float(obs)
    ens = _ensure_float(ens)
    device = ens.device
    dtype = ens.dtype
    obs = obs.to(device=device, dtype=dtype)

    M = ens.shape[0]

    # term 1: mean absolute diff to obs
    term1 = (ens - obs.unsqueeze(0)).abs().mean(dim=0)  # [H,W]

    # term 2: pairwise term via sorted trick
    V = ens.view(M, -1)                    # [M, H*W]
    V_sorted, _ = torch.sort(V, dim=0)     # [M, H*W]
    i = torch.arange(M, device=device, dtype=dtype).unsqueeze(1)  # [M,1]
    pair_sum = ((2 * i - (M - 1)) * V_sorted).sum(dim=0)          # [H*W]
    term2 = (pair_sum / (M * M)).view_as(term1)                   # [H,W]

    crps = term1 - term2  # [H,W]

    if mask is not None:
        m = _normalize_mask(mask, crps.shape, device)
        if m is not None:
            vals = crps[m]
            if reduction == "mean":
                return vals.mean()
            return vals

    return crps.mean() if reduction == "mean" else crps

# ------------------------------------------------------------------------------
# 2) Low-pass filtering with smooth taper in Fourier domain
# ------------------------------------------------------------------------------

def _tapered_lowpass_mask(
    H: int, W: int,
    k_cut: float,            # cutoff in cycles/km
    k_taper: float,          # start of taper (cycles/km), < k_cut
    dx_km: float,            # grid spacing in km
    device: torch.device,
) -> torch.Tensor:
    """
    Build a 2D isotropic low-pass mask with a cosine taper between k_taper and k_cut.
    Pass-band: k <= k_taper => 1.0
    Transition: k_taper < k < k_cut => 0.5 * (1 + cos(pi * (k - k_taper)/(k_cut - k_taper)))
    Stop-band: k >= k_cut => 0.0
    """
    ky = torch.fft.fftfreq(H, d=dx_km, device=device)
    kx = torch.fft.rfftfreq(W, d=dx_km, device=device)
    Ky, Kx = torch.meshgrid(ky, kx, indexing="ij")
    Kr = torch.sqrt(Kx**2 + Ky**2)    

    m = torch.ones_like(Kr)
    # stop-band
    m = torch.where(Kr >= k_cut, torch.zeros_like(m), m)
    # tapered band
    t = (Kr > k_taper) & (Kr < k_cut)
    # avoid division by zero if taper==cut (shouldn't happen)
    denom = max(k_cut - k_taper, 1e-9)
    m = torch.where(t, 0.5 * (1.0 + torch.cos(np.pi * (Kr - k_taper) / denom)), m)
    return m

@torch.no_grad()
def lowpass_filter_field(
    field_2d: torch.Tensor,      # [H,W] on CPU/GPU
    *,
    dx_km: float,
    k_cut: float,                 # cycles/km
    taper_frac: float = 0.2,      # fraction of k_cut where taper starts
) -> torch.Tensor:
    """
    Smooth isotropic low-pass using rFFT with cosine taper; returns filtered field on same device/dtype.
    """
    if field_2d.dim() != 2:
        raise ValueError("lowpass_filter_field expects [H,W]")
    device = field_2d.device
    dtype = field_2d.dtype
    H, W = field_2d.shape

    X = torch.fft.rfft2(field_2d.to(torch.float32), norm="ortho")  # [H, W//2+1]
    k_taper = (1.0 - float(taper_frac)) * float(k_cut)
    M = _tapered_lowpass_mask(H, W, float(k_cut), float(k_taper), float(dx_km), device=device)
    Xf = X * M
    out = torch.fft.irfft2(Xf, s=(H, W), norm="ortho")
    return out.to(dtype)

@torch.no_grad()
def lp_correlation_lr_gen(
    gen_2d: torch.Tensor,     # [H,W], generated PMM (or ens mean) on HR grid
    lr_2d: torch.Tensor,      # [1,h,w] or [H,W] upsampled LR stored by generator
    mask_2d: Optional[torch.Tensor],
    *,
    hr_dx_km: float,
    lr_dx_km: float,
    taper_frac: float = 0.2,
) -> float:
    """
    Correlation between LP(gen) and LP(lr) where LP passes up to LR Nyquist.
    We first ensure lr_2d aligns to [H,W] if it comes as [1,h,w] and matches PMM grid.
    """
    # normalize shapes to [H,W]
    g = gen_2d
    if g.dim() == 3 and g.shape[0] == 1:
        g = g.squeeze(0)
    H, W = g.shape

    lr = lr_2d
    # If LR is not on [H,W], upsample with bilinear to match HR.
    # Normalize LR to 2D [H,W] or 4D [N=1,C=1,H,W] as needed.
    if lr.dim() == 2:
        lr_2d = lr.to(g.device, g.dtype)
    elif lr.dim() == 3:
        # Accept [1,h,w] or [C,h,w]; collapse channel if singleton
        if lr.shape[0] == 1:
            lr_2d = lr.squeeze(0).to(g.device, g.dtype)
        else:
            # If shape is [C,h,w] with C>1, take first channel for correlation
            lr_2d = lr[0].to(g.device, g.dtype)
    elif lr.dim() == 4:
        # Expect [N=1,C=1,h,w] or similar
        if lr.shape[0] == 1 and lr.shape[1] == 1:
            lr_2d = lr.squeeze(0).squeeze(0).to(g.device, g.dtype)
        else:
            lr_2d = lr[0,0].to(g.device, g.dtype)
    else:
        raise ValueError(f"Unexpected LR tensor shape {tuple(lr.shape)}")

    if lr_2d.shape != (H, W):
        # Use bilinear upsampling from 2D by promoting to 4D
        lr_4d = lr_2d.unsqueeze(0).unsqueeze(0)  # [1,1,h,w]
        lr_up = F.interpolate(lr_4d, size=(H, W), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
    else:
        lr_up = lr_2d

    # Low-pass cutoff at LR Nyquist
    k_lr_nyq = 1.0 / (2.0 * float(lr_dx_km))
    g_lp = lowpass_filter_field(g, dx_km=float(hr_dx_km), k_cut=k_lr_nyq, taper_frac=taper_frac)
    l_lp = lowpass_filter_field(lr_up, dx_km=float(hr_dx_km), k_cut=k_lr_nyq, taper_frac=taper_frac)

    if mask_2d is not None:
        m = mask_2d.to(g.device)
        while m.dim() > 2 and m.shape[0] == 1:
            m = m.squeeze(0)
        m = m.bool()
        gv = g_lp[m].detach().cpu().numpy()
        lv = l_lp[m].detach().cpu().numpy()
    else:
        gv = g_lp.detach().cpu().numpy().ravel()
        lv = l_lp.detach().cpu().numpy().ravel()

    # If masked LR variance is ~0, try again without mask (domain too small / dry day)
    gv0 = gv; lv0 = lv
    sgv0 = float(np.std(gv0))
    slv0 = float(np.std(lv0))
    if slv0 < 1e-12:
        gv_full = g_lp.detach().cpu().numpy().ravel()
        lv_full = l_lp.detach().cpu().numpy().ravel()
        s_full = float(np.std(lv_full))
        if s_full >= 1e-12:
            gv, lv = gv_full, lv_full
            logger.info("[sigma_control] LP corr fallback: using unmasked domain (masked LR std≈0, full std=%.3e)", s_full)

    if gv.size == 0 or lv.size == 0:
        return float("nan")

    # Numerically safe Pearson correlation (avoid np.corrcoef NaNs when std=0)
    gv = gv.astype(np.float64)
    lv = lv.astype(np.float64)
    gv -= gv.mean()
    lv -= lv.mean()
    sgv = gv.std()
    slv = lv.std()
    if sgv < 1e-12 or slv < 1e-12:
        logger.warning("[sigma_control] LP corr std zero after fallback (sgv=%.3e, slv=%.3e) -> NaN", sgv, slv)
        return float("nan")
    r = float(np.dot(gv, lv) / (gv.size * sgv * slv))
    return r

@torch.no_grad()
def isotropic_psd_curve(
    field_2d: torch.Tensor,      # [H,W]
    *,
    dx_km: float,
    window: bool = True,
    detrend_mean: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute isotropic PSD curve (k, P(k)) for a single 2D field.
    Returns:
        k_centers: [N] cycles/km
        psd_binned: [N] power
    """
    x = field_2d
    while x.dim() > 2 and x.shape[0] == 1:
        x = x.squeeze(0)
    if x.dim() != 2:
        raise ValueError("isotropic_psd_curve expects [H,W]")

    x = x.to(torch.float32)
    H, W = x.shape
    device = x.device

    if detrend_mean:
        x = x - x.mean()

    if window:
        wy = torch.hann_window(H, device=device).view(H, 1)
        wx = torch.hann_window(W, device=device).view(1, W)
        x = x * (wy * wx)

    X = torch.fft.rfft2(x, norm="ortho")
    P2 = (X.real**2 + X.imag**2)  # [H, W//2+1]

    ky = torch.fft.fftfreq(H, d=dx_km, device=device)
    kx = torch.fft.rfftfreq(W, d=dx_km, device=device)
    Ky, Kx = torch.meshgrid(ky, kx, indexing="ij")
    Kr = torch.sqrt(Kx**2 + Ky**2)  # [H, W//2+1]

    kr = Kr.flatten()
    p = P2.flatten()
    n_bins = int(min(H, W) // 2)
    edges = torch.linspace(0.0, kr.max().item(), n_bins + 1, device=device)
    which = torch.bucketize(kr, edges) - 1
    k_centers = 0.5 * (edges[:-1] + edges[1:])

    psd_binned = torch.zeros(n_bins, device=device)
    for i in range(n_bins):
        sel = (which == i)
        if sel.any():
            psd_binned[i] = p[sel].mean()

    return k_centers, psd_binned

# ------------------------------------------------------------------------------
# 3) PSD slope in a mesoscale band (isotropic, log–log fit)
# ------------------------------------------------------------------------------

@torch.no_grad()
def psd_slope_band(
    field_2d: torch.Tensor,      # [H,W]
    *,
    dx_km: float,
    band_km: Tuple[float, float] = (5.0, 20.0),
    window: bool = True,
    detrend_mean: bool = True,
) -> float:
    """
    Fit slope of isotropic PSD in log–log space over band_km = (lo, hi) km.
    We construct isotropic PSD via rFFT2 and radial averaging.
    Returns: slope (float), negative for realistic precipitation scaling.
    """
    x = field_2d
    while x.dim() > 2 and x.shape[0] == 1:
        x = x.squeeze(0)
    if x.dim() != 2:
        raise ValueError("psd_slope_band expects [H,W] or [1,H,W]")

    x = x.to(torch.float32)
    H, W = x.shape
    device = x.device

    if detrend_mean:
        x = x - x.mean()

    if window:
        wy = torch.hann_window(H, device=device).view(H, 1)
        wx = torch.hann_window(W, device=device).view(1, W)
        x = x * (wy * wx)

    X = torch.fft.rfft2(x, norm="ortho")
    P2 = (X.real**2 + X.imag**2)  # [H, W//2+1]

    ky = torch.fft.fftfreq(H, d=dx_km, device=device)
    kx = torch.fft.rfftfreq(W, d=dx_km, device=device)
    Ky, Kx = torch.meshgrid(ky, kx, indexing="ij")
    Kr = torch.sqrt(Kx**2 + Ky**2)  # [H, W//2+1]

    # radial bins
    kr = Kr.flatten()
    p = P2.flatten()
    n_bins = int(min(H, W) // 2)
    edges = torch.linspace(0.0, kr.max().item(), n_bins + 1, device=device)
    which = torch.bucketize(kr, edges) - 1
    k_centers = 0.5 * (edges[:-1] + edges[1:])

    psd_binned = torch.zeros(n_bins, device=device)
    for i in range(n_bins):
        sel = which == i
        if sel.any():
            psd_binned[i] = p[sel].mean()

    # select mesoscale band in k-space: k in (1/hi, 1/lo)
    k_lo = 1.0 / float(band_km[1])
    k_hi = 1.0 / float(band_km[0])
    m = (k_centers > k_lo) & (k_centers < k_hi) & (psd_binned > 0)

    if not bool(m.any()):
        return float("nan")

    k_sel = k_centers[m].detach().cpu().numpy()
    P_sel = psd_binned[m].detach().cpu().numpy()

    # linear fit in log10–log10
    coeff = np.polyfit(np.log10(k_sel), np.log10(P_sel + 1e-20), 1)
    slope = float(coeff[0])
    return slope

# ------------------------------------------------------------------------------
# 4) Main orchestration
# ------------------------------------------------------------------------------

def _cfg_get(cfg, dotted: str, default=None):
    """Nested get for OmegaConf-like objects."""
    cur = cfg
    for part in dotted.split("."):
        if not hasattr(cur, part):
            return default
        cur = getattr(cur, part)
    return cur

def evaluate_sigma_control(
    cfg,
    sigma_star_grid: List[float],
    gen_base_dir: str | Path,
    out_dir: str | Path,
) -> Dict[str, str]:
    """
    Evaluate generated samples across σ* values and write per-sigma and aggregated metrics.
    Returns dict with file paths for downstream plotting.
    """
    gen_base_dir = Path(gen_base_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[sigma_control] Evaluating %d sigma* values; output -> %s",
            len(sigma_star_grid), str(tables_dir))

    # Spacings (km) for filters/PSD — fall back to sane defaults if not in cfg
    hr_dx_km = float(_cfg_get(cfg, "full_gen_eval.hr_dx_km", 2.5))
    lr_dx_km = float(_cfg_get(cfg, "full_gen_eval.lr_dx_km", 31.0))
    taper_frac = float(_cfg_get(cfg, "full_gen_eval.sigma_control.corr_taper_frac", 0.2))
    eval_land_only = bool(_cfg_get(cfg, "full_gen_eval.eval_land_only", True))
    max_dates = int(_cfg_get(cfg, "full_gen_eval.max_dates", -1))
    band = _cfg_get(cfg, "full_gen_eval.sigma_control.psd_band_km", (5.0, 20.0))
    if isinstance(band, (list, tuple)) and len(band) == 2:
        psd_band = (float(band[0]), float(band[1]))
    else:
        psd_band = (5.0, 20.0)
    # Read config for CRPS rainy threshold and high‑k robustness
    crps_rain_thresh = _cfg_get(cfg, "full_gen_eval.sigma_control.crps_rain_thresh", None)
    hk_min_hr_frac = float(_cfg_get(cfg, "full_gen_eval.sigma_control.hk_min_hr_frac", 1e-4))
    hk_min_hr_abs = float(_cfg_get(cfg, "full_gen_eval.sigma_control.hk_min_hr_abs", 1e-12))

    results: List[Dict] = []

    # PSD curve accumulators per sigma*: store per-date GEN-ensemble-mean PSDs and HR PSDs
    psd_acc_gen: DefaultDict[float, list[np.ndarray]] = defaultdict(list)
    psd_acc_hr: DefaultDict[float, list[np.ndarray]] = defaultdict(list)
    k_ref_np: np.ndarray | None = None
    psd_acc_lr: DefaultDict[float, list[np.ndarray]] = defaultdict(list)

    for sigma_star in sigma_star_grid:
        subdir = gen_base_dir / f"sigma_star={float(sigma_star):.2f}"
        if not subdir.exists():
            logger.warning(f"[sigma_control] Missing generation folder for sigma*={sigma_star:.2f}: {subdir}")
            continue
        logger.info("[sigma_control] σ*=%.2f | reading from %s",
            float(sigma_star), str(subdir))

        resolver = EvalDataResolver(
            gen_root=subdir,
            eval_land_only=eval_land_only,
            lr_phys_key="lr",  # prefer canonical LR saved by generator
        )

        dates = resolver.list_dates()
        if max_dates > 0:
            dates = dates[:max_dates]

        logger.info("[sigma_control] σ*=%.2f | %d dates to evaluate (max_dates=%d)",
            float(sigma_star), len(dates), max_dates)

        for date in dates:
            hr = resolver.load_obs(date)          # [H,W] or None
            pmm = resolver.load_pmm(date)         # [H,W] or None
            ens = resolver.load_ens(date)         # [M,H,W] or None
            lr = resolver.load_lr(date)           # [1,h,w] or [H,W] or None
            mask = resolver.load_mask(date)       # [H,W] bool or None

            # sanity checks
            if hr is None or pmm is None or ens is None or lr is None:
                logger.info(f"[sigma_control] Skipping {date} (missing arrays)")
                continue

            # --- PSD curves for GEN (ensemble mean per date) and HR ---
            hk_gain = float("nan")
            # compute GEN ensemble mean field for PSD (average across members)
            if isinstance(ens, torch.Tensor):
                gen_mean_2d = ens.mean(dim=0)  # [H,W]
            else:
                gen_mean_2d = torch.from_numpy(np.asarray(ens)).mean(dim=0)  # type: ignore
            try:
                # GEN ensemble mean already computed above
                k_gen, P_gen = isotropic_psd_curve(gen_mean_2d, dx_km=hr_dx_km)
                k_hr,  P_hr  = isotropic_psd_curve(hr, dx_km=hr_dx_km)

                # --- High-k power gain above LR Nyquist (phase-insensitive) ---
                k_arr = k_hr.detach().cpu().numpy()
                P_gen_np = P_gen.detach().cpu().numpy()
                P_hr_np = P_hr.detach().cpu().numpy()
                k_nyq = 1.0 / (2.0 * float(lr_dx_km))
                hi = (k_arr > k_nyq)
                if np.any(hi):
                    num = float(np.sum(P_gen_np[hi]))
                    den = float(np.sum(P_hr_np[hi]))
                    total_hr = float(np.sum(P_hr_np))
                    # Skip days where HR high‑k power is numerically negligible
                    if den <= 0.0 or den < hk_min_hr_abs or den < hk_min_hr_frac * max(total_hr, hk_min_hr_abs):
                        hk_gain = float("nan")
                    else:
                        hk_gain = num / den
                else:
                    hk_gain = float("nan")

                # Keep a single common k-grid (from HR) as reference
                if k_ref_np is None:
                    k_ref_np = k_hr.detach().cpu().numpy()

                psd_acc_gen[float(sigma_star)].append(P_gen.detach().cpu().numpy())
                psd_acc_hr[float(sigma_star)].append(P_hr.detach().cpu().numpy())

                # LR upsampled to HR grid for PSD reference
                try:
                    lr_arr = lr
                    # Normalize LR shapes to [H,W]
                    if lr_arr.dim() == 2:
                        lr2d = lr_arr
                    elif lr_arr.dim() == 3 and lr_arr.shape[0] == 1:
                        lr2d = lr_arr.squeeze(0)
                    elif lr_arr.dim() == 4 and lr_arr.shape[0] == 1 and lr_arr.shape[1] == 1:
                        lr2d = lr_arr.squeeze(0).squeeze(0)
                    else:
                        lr2d = lr_arr[0] if lr_arr.dim() == 3 else lr_arr[0,0]
                    if tuple(lr2d.shape) != (hr.shape[-2], hr.shape[-1]):
                        lr4d = lr2d.unsqueeze(0).unsqueeze(0)
                        lr2d = F.interpolate(lr4d, size=(hr.shape[-2], hr.shape[-1]),
                                             mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
                    k_lrpsd, P_lr = isotropic_psd_curve(lr2d, dx_km=hr_dx_km)
                    psd_acc_lr[float(sigma_star)].append(P_lr.detach().cpu().numpy())
                except Exception as e:
                    logger.warning(f"[sigma_control] LR PSD curve failed on {date} @ sigma*={sigma_star:.2f}: {e}")
            except Exception as e:
                logger.warning(f"[sigma_control] PSD curve failed on {date} @ sigma*={sigma_star:.2f}: {e}")

            # Diagnostics: std before LP (masked) to understand zero-variance cases
            try:
                if mask is not None:
                    m = mask.bool()
                    std_lr_raw = float(torch.nan_to_num(lr.squeeze()[m].float(), nan=0.0).std().cpu().item())
                else:
                    std_lr_raw = float(torch.nan_to_num(lr.squeeze().float(), nan=0.0).std().cpu().item())
                logger.debug("[sigma_control] %s σ*=%.2f | raw LR std=%.3e", date, float(sigma_star), std_lr_raw)
            except Exception:
                pass

            # 1) Low-pass correlation (≤ LR Nyquist)
            r_lp = lp_correlation_lr_gen(
                pmm, lr, mask,
                hr_dx_km=hr_dx_km, lr_dx_km=lr_dx_km, taper_frac=taper_frac
            )

            # 2) PSD slope for GEN (ensemble mean) and HR in mesoscale band
            slope_gen = psd_slope_band(gen_mean_2d, dx_km=hr_dx_km, band_km=psd_band)
            slope_hr  = psd_slope_band(hr,         dx_km=hr_dx_km, band_km=psd_band)
            slope_err = slope_gen - slope_hr if (np.isfinite(slope_gen)  and np.isfinite(slope_hr)) else np.nan
            
            # 3) CRPS over land (if available)
            crps_mask = mask
            if crps_rain_thresh is not None:
                try:
                    thr = float(crps_rain_thresh)
                    rain_mask = hr > thr
                    crps_mask = rain_mask if crps_mask is None else (crps_mask & rain_mask)
                except Exception:
                    pass
            crps = crps_ensemble_local(hr, ens, mask=crps_mask, reduction="mean").item()

            results.append({
                "date": date,
                "sigma_star": float(sigma_star),
                "r_lp": float(r_lp),
                "slope_gen": float(slope_gen) if np.isfinite(slope_gen) else np.nan,
                "slope_hr": float(slope_hr) if np.isfinite(slope_hr) else np.nan,
                "slope_err": float(slope_err) if np.isfinite(slope_err) else np.nan,
                "crps": float(crps),
                "hk_gain": float(hk_gain),
            })

    metrics_path = tables_dir / "metrics_by_sigma.csv"
    summary_path = tables_dir / "agg_summary.csv"
    
    header_metrics = ["date","sigma_star","r_lp","slope_gen","slope_hr","slope_err","crps","hk_gain"]
    header_summary = ["sigma_star",
                    "r_lp_mean","r_lp_std",
                    "slope_gen_mean","slope_gen_std",
                    "slope_hr_mean","slope_hr_std",
                    "slope_err_mean","slope_err_std",
                    "crps_mean","crps_std",
                    "hk_gain_mean","hk_gain_std"]

    if len(results) == 0:
        logger.warning("[sigma_control] No results computed — check inputs.")
        # write empty CSVs with headers for pipeline stability
        with open(metrics_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header_metrics)
            writer.writeheader()
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header_summary)
        return {"metrics": str(metrics_path), "summary": str(summary_path)}

    # Write metrics CSV
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header_metrics)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    # Aggregate per sigma_star
    agg_dict = defaultdict(lambda: defaultdict(list))
    for row in results:
        sigma = row["sigma_star"]
        for key in ["r_lp", "slope_gen", "slope_hr", "slope_err", "crps", "hk_gain"]:
            agg_dict[sigma][key].append(row.get(key, float("nan")))
    
    agg_rows = []
    for sigma in sorted(agg_dict.keys()):
        vals = agg_dict[sigma]
        r_lp_arr = np.array(vals["r_lp"], dtype=np.float64)
        slope_gen_arr = np.array(vals["slope_gen"], dtype=np.float64)
        slope_hr_arr = np.array(vals["slope_hr"], dtype=np.float64)
        slope_err_arr = np.array(vals["slope_err"], dtype=np.float64)
        crps_arr = np.array(vals["crps"], dtype=np.float64)
        hk_gain_arr = np.array(vals.get("hk_gain", []), dtype=np.float64)

        row = [
            sigma,
            float(np.nanmean(r_lp_arr)), float(np.nanstd(r_lp_arr)),
            float(np.nanmean(slope_gen_arr)), float(np.nanstd(slope_gen_arr)),
            float(np.nanmean(slope_hr_arr)), float(np.nanstd(slope_hr_arr)),
            float(np.nanmean(slope_err_arr)), float(np.nanstd(slope_err_arr)),
            float(np.nanmean(crps_arr)), float(np.nanstd(crps_arr)),
            float(np.nanmean(hk_gain_arr)), float(np.nanstd(hk_gain_arr)),
        ]
        agg_rows.append(row)

    # Write summary CSV
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header_summary)
        writer.writerows(agg_rows)

    logger.info("[sigma_control] Wrote summary: %s (σ* values=%d)", str(summary_path), len(agg_rows))        

    # --- Save PSD curves (averaged across dates) for each sigma* ---
    try:
        if k_ref_np is not None and len(psd_acc_gen) > 0:
            sigmas_sorted = np.array(sorted(psd_acc_gen.keys()), dtype=float)
            # Ensure consistent HR reference by averaging HR PSDs aggregated under any sigma
            # (HR PSD does not depend on sigma; pick the first available sigma group)
            first_sigma = float(sigmas_sorted[0])
            hr_stack = np.stack(psd_acc_hr[first_sigma], axis=0) if psd_acc_hr.get(first_sigma) else None
            if hr_stack is None or hr_stack.size == 0:
                # try any other sigma
                for s in sigmas_sorted:
                    if psd_acc_hr.get(float(s)):
                        hr_stack = np.stack(psd_acc_hr[float(s)], axis=0)
                        break

            # --- LOG-DOMAIN EPSILON ---
            eps = 1e-14

            # HR mean/std and log-domain mean/std
            if hr_stack is not None and hr_stack.size > 0:
                psd_hr_mean = hr_stack.mean(axis=0)
                psd_hr_std  = hr_stack.std(axis=0)
                hr_log = np.log10(np.clip(hr_stack, eps, None))
                psd_hr_logmu  = hr_log.mean(axis=0)
                psd_hr_logstd = hr_log.std(axis=0)
            else:
                psd_hr_mean = np.zeros_like(k_ref_np)
                psd_hr_std  = np.zeros_like(k_ref_np)
                psd_hr_logmu  = np.zeros_like(k_ref_np)
                psd_hr_logstd = np.zeros_like(k_ref_np)                

            # Stack GEN means/stds per sigma* and log-domain stats
            S = len(sigmas_sorted)
            K = k_ref_np.shape[0]
            psd_gen_mean = np.zeros((S, K), dtype=np.float64)
            psd_gen_std  = np.zeros((S, K), dtype=np.float64)
            psd_gen_logmu = np.zeros((S, K), dtype=np.float64)
            psd_gen_logstd = np.zeros((S, K), dtype=np.float64)            
            for i, s in enumerate(sigmas_sorted):
                arrs = psd_acc_gen[float(s)]
                if len(arrs) == 0:
                    continue
                stack = np.stack(arrs, axis=0)  # [N_dates, K]
                psd_gen_mean[i] = stack.mean(axis=0)
                psd_gen_std[i]  = stack.std(axis=0)
                stack_log = np.log10(np.clip(stack, eps, None))
                psd_gen_logmu[i]  = stack_log.mean(axis=0)
                psd_gen_logstd[i] = stack_log.std(axis=0)

            # Aggregate LR curves (σ*-invariant, but follow same grouping for simplicity)
            lr_stack = None
            for s in sigmas_sorted:
                if psd_acc_lr.get(float(s)):
                    lr_stack = np.stack(psd_acc_lr[float(s)], axis=0)
                    break
            if lr_stack is not None and lr_stack.size > 0:
                psd_lr_mean = lr_stack.mean(axis=0)
                psd_lr_std  = lr_stack.std(axis=0)
                lr_log = np.log10(np.clip(lr_stack, eps, None))
                psd_lr_logmu  = lr_log.mean(axis=0)
                psd_lr_logstd = lr_log.std(axis=0)                
            else:
                psd_lr_mean = np.zeros_like(k_ref_np)
                psd_lr_std  = np.zeros_like(k_ref_np)
                psd_lr_logmu  = np.zeros_like(k_ref_np)
                psd_lr_logstd = np.zeros_like(k_ref_np)

            np.savez(
                tables_dir / "sigma_psd_curves.npz",
                k=k_ref_np,
                sigma_vals=sigmas_sorted,
                psd_hr_mean=psd_hr_mean,
                psd_hr_std=psd_hr_std,
                psd_lr_mean=psd_lr_mean,
                psd_lr_std=psd_lr_std,                
                psd_gen_mean=psd_gen_mean,
                psd_gen_std=psd_gen_std,
                psd_hr_logmu=psd_hr_logmu,
                psd_hr_logstd=psd_hr_logstd,
                psd_lr_logmu=psd_lr_logmu,
                psd_lr_logstd=psd_lr_logstd,
                psd_gen_logmu=psd_gen_logmu,
                psd_gen_logstd=psd_gen_logstd,                
                lr_nyquist=(1.0 / (2.0 * lr_dx_km)),
                psd_band_km=np.array(psd_band, dtype=np.float64),                
            )
        logger.info("[sigma_control] Saved PSD curves: %s",
            str(tables_dir / "sigma_psd_curves.npz"))
    except Exception as e:
        logger.warning(f"[sigma_control] Failed to save sigma_psd_curves.npz: {e}")

    logger.info(f"[sigma_control] Wrote metrics to {out_dir}")
    return {"metrics": str(metrics_path), "summary": str(summary_path)}
