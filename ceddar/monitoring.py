"""
    Monitoring functions for EDM.
    Currently includes:
        - edm_cosine_metric: Cosine similarity metric for EDM models as per Karras et al. (2022).
        - _masked_corrcoef_per_sample: Mean Pearson correlation across the batch, computed per-sample over masked pixels.
        - report_precip_extremes: Reports extremes in a back-transformed precipitation tensor.

    TODO:
        - FSS (Fractions Skill Score) implementation at 5, 10, 20 km scales
        - PSD slope metric
        - P95/P99 metrics
        - Wet-day frequency 
        - Other metrics from "Evaluating Generative Models via Precision and Recall" (Sajjadi et al. 2018)
"""

import torch
import logging
import os
import math
import json 

import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime

from losses import EDMLoss

from scale_utils import (isotropic_psd,
                         find_intersection_k,
                         wavelength_km_from_k_cpkm,
                         sigma_star_from_preserve_scale,
                         t_star_from_sigma_ve)

logger = logging.getLogger(__name__)

def _to_tensor(x):
    return x if isinstance(x, torch.Tensor) else torch.tensor(x)

@torch.no_grad()
def compute_fss_at_scales(gen_bt: torch.Tensor, hr_bt: torch.Tensor, *, mask:torch.Tensor|None,
                  grid_km_per_px: float, fss_km: list[float], thr_mm: float, eps: float=1e-8) -> dict[str, float]:
    """
        Fractions Skill Score (FSS) for exceedance over threshold thr_mm at different spatial scales (km).
        gen_bt, hr_bt: [B,1,H,W] back-transformed precipitation tensors in mm/day.
        mask: [B,1,H,W] land mask (1=land, 0=sea) or None
    """
    gen_bt = _to_tensor(gen_bt).float()
    hr_bt = _to_tensor(hr_bt).float()
    if mask is not None:
        mask = mask.bool().expand_as(gen_bt)

    X = (gen_bt > thr_mm).float() # Binary exceedance for generated
    Y = (hr_bt > thr_mm).float() # Binary exceedance for HR

    out = {}
    for km in fss_km:
        rad_px = int(max(1, round(float(km) / float(grid_km_per_px)))) # Radius in pixels (int)
        k = 2 * rad_px + 1 # Odd kernel size (int)
        # Box filter via average pooling
        Xs = F.avg_pool2d(X, kernel_size=k, stride=1, padding=rad_px)
        Ys = F.avg_pool2d(Y, kernel_size=k, stride=1, padding=rad_px)
        if mask is not None:
            m = mask.float()
            num = ((Xs - Ys) ** 2 * m).sum() / (m.sum() + eps) # Mean squared error over land
            den = (((Xs ** 2 + Ys ** 2) * m).sum() / (m.sum() + eps)) + eps
        else:
            num = ((Xs - Ys) ** 2).mean() # Mean squared error over all pixels
            den = (Xs ** 2 + Ys ** 2).mean() + eps
        fss = 1.0 - num / den # Fractions Skill Score
        out[f'{int(km)}km'] = float(fss)#float(fss.clamp(0.0, 1.0)) # Clamp to [0,1]
    
    return out

# === PSD Slope metric
@torch.no_grad()
def _radial_psd_slope_single(img: torch.Tensor, *, mask:torch.Tensor|None=None, ignore_low_k_bins: int=1) -> float:
    """
        Compute the slope of the radially-averaged 2D power spectrum for a single 2D field.
        Returns the log-log slope (beta) from a linear fit of log10(P(k)) vs log10(k).
        The lowest *ignore_low_k_bins* raidal frequency bins are dropped to avoid DC/very-low-k dominance.
    """
    # Ensure 2-D (H,W)
    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]
    elif img.ndim == 4 and img.shape[0] == 1 and img.shape[1] == 1:
        img = img[0,0]
    elif img.ndim != 2:
        img = img.squeeze()
        if img.ndim != 2:
            raise ValueError(f"Input image must be 2D, got shape {img.shape}")
    
    # Optional mask: zero out masked pixels (keeping shape)
    if mask is not None:
        m = mask
        if m.ndim == 3 and m.shape[0] == 1:
            m = m[0]
        elif m.ndim == 4 and m.shape[0] == 1 and m.shape[1] == 1:
            m = m[0,0]
        if m.dtype != torch.bool:
            m = (m > 0.5)
        img = img * m.float()

    # 2D FFT and power spectrum
    F = torch.fft.fft2(img.float())
    P = (F.real ** 2 + F.imag ** 2)  # Power spectrum
    P = torch.fft.fftshift(P)  # Shift zero freq to center

    H, W = img.shape[-2:] # Height, Width
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0  # Center coordinates
    y = torch.arange(H, device=img.device)
    x = torch.arange(W, device=img.device)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    R = torch.sqrt((X - cx) ** 2 + (Y - cy) ** 2)  # Radial distances

    # Radial bins: 1 px per bin
    r = R.flatten()
    p = P.flatten()
    rmax = int(torch.max(R).item())
    if rmax < (ignore_low_k_bins + 2):
        return float('nan')  # Not enough radial bins to compute slope
    
    # Bin by integer radius
    nbins = rmax + 1
    sums = torch.zeros(nbins, device=img.device)
    counts = torch.zeros(nbins, device=img.device)
    idx = r.long().clamp(0, rmax)
    sums.scatter_add_(0, idx, p) # Sum power in each radial bin
    ones = torch.ones_like(p, device=img.device)
    counts.scatter_add_(0, idx, ones) # Count pixels in each radial bin

    with torch.no_grad():
        valid = counts > 0
        radii = torch.arange(nbins, device=img.device)[valid]
        prof = (sums[valid] / counts[valid]).clamp(min=1e-12)  # Average power per bin, avoid log(0)

    # Drop DC and a few low-k bins
    if radii.numel() <= (ignore_low_k_bins + 1):
        return float('nan')  # Not enough bins to fit
    radii = radii[ignore_low_k_bins:]
    prof = prof[ignore_low_k_bins:]

    # Linear fit in log-log space
    k = radii.detach().cpu().numpy()
    Pk = prof.detach().cpu().numpy()
    if np.any(k <= 0) or np.any(np.isnan(Pk)):
        return float('nan')
    
    xlog = np.log10(k)
    ylog = np.log10(Pk)
    # Guard against degenerate cases (e.g. constant image)
    if not np.all(np.isfinite(xlog)) or not np.all(np.isfinite(ylog)):
        return float('nan')
    if xlog.size < 3:
        return float('nan')
    beta, _ = np.polyfit(xlog, ylog, 1)  # Slope is beta
    
    return float(beta)


@torch.no_grad()
def compute_psd_slope(
    gen_bt: torch.Tensor,
    hr_bt: torch.Tensor|None= None,
    *,
    mask:torch.Tensor|None=None,
    ignore_low_k_bins: int=1
) -> dict[str, float]:
    """
        Compute radial PSD slope (log-log slope of P(k) vs k) for generated fields and (optionally) HR reference fields.
        Inputs are expected to be back-transformed precipitation (mm/day), shaped [B, 1, H, W] or [B, H, W].
        If *mask* is provided ([B, 1, H, W] or [B, H, W]) only masked pixels are used in the computation.
        Returns a dict with keys
        - 'psd_slope_gen': mean PSD slope for generated fields across the batch
        - 'psd_slope_hr': mean PSD slope for HR fields across the batch (if hr_bt is provided)
        - 'psd_slope_delta': difference in mean PSD slope (gen - hr) if hr_bt is provided
    """
    def _prep(t):
        t = _to_tensor(t).float()
        if t.ndim == 3: # [B,H,W] -> [B,1,H,W]
            t = t[:, None, :, :]
        return t
    
    G = _prep(gen_bt)
    M = None
    if mask is not None:
        M = _prep(mask).bool()
    Href = _prep(hr_bt) if hr_bt is not None else None

    betas_g = []
    betas_h = [] if Href is not None else None

    B = G.shape[0]
    for i in range(B):
        mi = M[i] if M is not None else None
        betas_g.append(_radial_psd_slope_single(G[i], mask=mi, ignore_low_k_bins=ignore_low_k_bins))
        if Href is not None and betas_h is not None:
            betas_h.append(_radial_psd_slope_single(Href[i], mask=mi, ignore_low_k_bins=ignore_low_k_bins))

    def _clean_mean(arr):
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(arr.mean()) if arr.size > 0 else float('nan')
    
    out = {'psd_slope_gen': _clean_mean(betas_g)}
    if betas_h is not None:
        mhr = _clean_mean(betas_h)
        out['psd_slope_hr'] = mhr
        if np.isfinite(out['psd_slope_gen']) and np.isfinite(mhr):
            out['psd_slope_delta'] = float(mhr - out['psd_slope_gen'])

    return out


def compute_sigma_star_from_loader(hr_batch: torch.Tensor,
                                   lr_batch: torch.Tensor,
                                   pixel_km: float,
                                   lsm_hr: torch.Tensor | None = None,
                                   land_only: bool = True,
                                   window: str = "hann"):
    """
        Compute sigma* from intersection of isotropic PSDs of HR and LR fields in a batch.
    """
    # hr_batch, lr_batch: [B,1,H,W] or [B,H,W]
    def _prep(b):
        if isinstance(b, torch.Tensor): b = b.detach().cpu()
        if b.ndim == 4: b = b[:, 0] # [B,1,H,W] -> [B,H,W]
        return b
    
    Hs = _prep(hr_batch).numpy()
    Ls = _prep(lr_batch).numpy()
    if land_only and (lsm_hr is not None):
        M = _prep(lsm_hr).numpy() > 0.5
    else:
        M = None

    # Average PSDs over batch
    P_hr_all, P_lr_all = [], []
    k_ref = None
    for i in range(Hs.shape[0]):
        h = Hs[i]
        l = Ls[i]
        if M is not None:
            h = np.where(M[i], h, np.nan)
            l = np.where(M[i], l, np.nan)
        k_hr, P_hr = isotropic_psd(h, pixel_km, window=window)
        k_lr, P_lr = isotropic_psd(l, pixel_km, window=window)
        if k_ref is None:
            k_ref = (k_hr, k_lr)
        P_hr_all.append(P_hr)
        P_lr_all.append(P_lr)

    # Stack common k grid by interpolation
    kmin = max(min(k_ref[0]), min(k_ref[1])) if k_ref is not None else 0.0
    kmax = min(max(k_ref[0]), max(k_ref[1])) if k_ref is not None else 1.0
    k_common = np.linspace(kmin, kmax, 512)
    Ph = []; Pl = []
    if P_hr_all is None or P_lr_all is None:
        logger.warning("No PSDs computed from batches. Returning None for lambda*.")
        return None
    if k_ref is None:
        logger.warning("No reference k grids found. Returning None for lambda*.")
        return None

    for (k_hr, P_hr), (k_lr, P_lr) in zip([k_ref]*len(P_hr_all), [k_ref]*len(P_lr_all)):
        # Reuse first's k; could refine but ok.
        # Actually better: just recompute interp for each entry; keeping simple for brevity
        Ph.append(np.interp(k_common, k_hr, P_hr_all[0]))
        Pl.append(np.interp(k_common, k_lr, P_lr_all[0]))
    P_hr_mean = np.interp(k_common, k_ref[0], np.mean(np.stack(P_hr_all,0),0))
    P_lr_mean = np.interp(k_common, k_ref[1], np.mean(np.stack(P_lr_all,0),0))
    # Find intersection k*
    k_star = find_intersection_k(k_ref[0], np.mean(np.stack(P_hr_all,0),0),
                                 k_ref[1], np.mean(np.stack(P_lr_all,0),0))
    if k_star is None:
        logger.warning("Could not find intersection k* between HR and LR PSDs. Returning None for lambda*.")
        return None
    
    lam_star = wavelength_km_from_k_cpkm(k_star) # in km
    return lam_star


# INSERT THIS IN TRAINING LOOP WHERE APPROPRIATE:
# lam_star = compute_sigma_star_from_loader(hr_batch=x0_hr, lr_batch=lr_up, pixel_km=self.pixel_km, lsm_hr=lsm_hr, land_only=self.eval_land_only)
# if lam_star is not None:
#     preserve_km = cfg.get('scale_control', {}).get('preserve_scale_km') or lam_star
#     c_sigma = float(cfg.get('scale_control', {}).get('c_sigma', 1.0))
#     sigma_star_px = sigma_star_from_preserve_scale(preserve_km, self.pixel_km, c=c_sigma)
#     self.sigma_star_px = sigma_star_px  # cache for sampler
#     if not self.edm_enabled:
#         self.t_star = t_star_from_sigma_ve(sigma_star_px, self.marginal_prob_std_fn)
# INSERT THIS IN SAMPLING LOOP WHERE APPROPRIATE: (Enforces that sampler never tries to "invent" scales larger than target - consistent with "preserve large scales")
# sigma_min_user = cfg['edm'].get('sigma_min', 0.002)
# sigma_min_eff = max(sigma_min_user, getattr(self, 'sigma_star_px', 0.0))
# samples = edm_sampler(self.model, ..., sigma_min=sigma_min_eff, ...)

def _plot_reliability_curve( probs: torch.Tensor, targets: torch.Tensor,
                            bins: int = 15, save_path: str | None = None,
                            title: str = "RainGate reliability curve"):
    """
        Plot reliability (calibration) curve for wet probabilities vs truth
        probs, targets are 1D tensors in [0,1] and {0,1} respectively
    """
    probs = probs.detach().cpu().float().clamp(0, 1)
    targets = targets.detach().cpu().float().clamp(0, 1)
    if probs.numel() == 0:
        logger.warning("No valid probabilities to plot reliability curve.")
        return
    # Bin edges
    edges = torch.linspace(0, 1, bins + 1)
    bin_ids = torch.bucketize(probs, edges, right=True) - 1  # Bin indices [0, bins-1]
    # Aggregate
    acc = torch.zeros(bins)
    conf = torch.zeros(bins)
    cnt = torch.zeros(bins)
    for b in range(bins):
        m = bin_ids == b
        n = int(m.sum())
        if n == 0:
            continue
        cnt[b] = n
        conf[b] = probs[m].mean()
        acc[b] = targets[m].mean()
    # Remove empty bins
    keep = cnt > 0
    conf = conf[keep].numpy()
    acc = acc[keep].numpy()
    # ECE (Expected Calibration Error)
    w = (cnt[keep] / cnt[keep].sum()).numpy()
    ece = float((w * np.abs(acc - conf)).sum())
    # Plot
    plt.figure(figsize=(4.2, 4.2))
    plt.plot([0, 1], [0, 1], '--', lw=1, label='Perfectly calibrated', color='gray')
    plt.plot(conf, acc, marker='o', lw=1.5, label=f'RainGate (ECE={ece:.3f})', color='blue')
    plt.xlabel('Predicted probability')
    plt.ylabel('Observed frequency')
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(frameon=False)
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def _save_weight_map_viz(weight_map: torch.Tensor,
                            wet_probs: torch.Tensor | None,
                            wet_target: torch.Tensor | None,
                            epoch: int, step: int, prefix: str = 'train',
                            save_path: str | None = None):
    """Save a small panel showing the pixel weight map (and optional prob/target) for the first sample.
    Expects tensors with shapes: weight_map [B,1,H,W]; wet_probs [B,1,H,W] or None; wet_target [B,1,H,W] or None."""
    import matplotlib.pyplot as plt
    import numpy as np
    try:
        wm = weight_map.detach().cpu()
        if wm.ndim == 4:
            wm = wm[0,0]
        elif wm.ndim == 3:
            wm = wm[0]
        else:
            return
        fig, axs = plt.subplots(1, 3, figsize=(10,3))
        im0 = axs[0].imshow(wm.numpy(), origin='lower', interpolation='nearest')
        axs[0].set_title('weight_map')
        axs[0].set_xticks([]); axs[0].set_yticks([])
        fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
        if wet_probs is not None:
            p = wet_probs.detach().cpu()
            p = p[0,0] if p.ndim == 4 else (p[0] if p.ndim == 3 else p)
            im1 = axs[1].imshow(p.numpy(), origin='lower', interpolation='nearest')
            axs[1].set_title('wet_prob')
            axs[1].set_xticks([]); axs[1].set_yticks([])
            fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
        else:
            axs[1].axis('off')
        if wet_target is not None:
            t = wet_target.detach().cpu()
            t = t[0,0] if t.ndim == 4 else (t[0] if t.ndim == 3 else t)
            im2 = axs[2].imshow(t.numpy(), origin='lower', interpolation='nearest')
            axs[2].set_title('wet_target')
            axs[2].set_xticks([]); axs[2].set_yticks([])
            fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
        else:
            axs[2].axis('off')
        for ax in axs:
            for spine in ax.spines.values():
                spine.set_visible(False)
        if save_path is not None:
            out = os.path.join(save_path, f'{prefix}_wm_e{epoch:03d}_s{step:06d}.png')
            fig.savefig(out, dpi=200, bbox_inches='tight')
        else:
            logger.warning("save_path is None, skipping saving weight_map visualization.")
        plt.close(fig)
    except Exception as e:
        logger.warning(f"[debug] Failed to save weight_map viz: {e}")

@torch.no_grad()
def compute_p95_p99_and_wet_day(
    gen_bt: torch.Tensor,
    hr_bt: torch.Tensor|None = None,
    *,
    mask:torch.Tensor|None=None,
    wet_threshold_mm: float=1.0) -> dict[str, float]:
    """
        Compute P95, P99 and wet-day frequency (> wet_threshold_mm) for generated fields and (optionally) HR reference fields.
        Inputs can be [B,1,H,W] or [B,H,W] or numpy arrays.
        *mask* is optional and should be broadcastable to [B,1,H,W] or [B,H,W].
        Returns keys:
        - 'gen_p95', 'gen_p99', 'gen_wet_freq'
        - 'hr_p95', 'hr_p99', 'hr_wet_freq' (if hr_bt is provided)
    """
    def _prep(t):
        if t is None:
            return None
        t = _to_tensor(t).float()
        if t.ndim == 3: # [B,H,W] -> [B,1,H,W]
            t = t[:, None, :, :]
        return t
    
    G = _prep(gen_bt)
    H = _prep(hr_bt) 
    M = _prep(mask)
    if M is not None:
        M = M.bool()

    def _metrics(t: torch.Tensor | None, m: torch.Tensor | None):
        if t is None:
            return float('nan'), float('nan'), float('nan')
        if m is not None:
            t = t.masked_fill(~m.expand_as(t), float('nan'))
        # Flatten over spatial dims and channels
        flat = t.reshape(t.shape[0], -1)
        # Remove NaNs
        flat_np = flat.detach().cpu().numpy()
        flat_np = flat_np[~np.isnan(flat_np)]
        if flat_np.size == 0:
            return float('nan'), float('nan'), float('nan')
        p95 = float(np.percentile(flat_np, 95))
        p99 = float(np.percentile(flat_np, 99))
        wet_freq = float(np.mean(flat_np > wet_threshold_mm))
        return p95, p99, wet_freq
    
    p95g, p99g, wfg = _metrics(G, M)
    out = {'gen_p95': p95g, 'gen_p99': p99g, 'gen_wet_freq': wfg}
    if H is not None:
        p95h, p99h, wfh = _metrics(H, M)
        out.update({'hr_p95': p95h, 'hr_p99': p99h, 'hr_wet_freq': wfh})

    return out



@torch.no_grad() # Disable gradient computation for monitoring
def in_loop_metrics(loss_obj, model, x0, *, cond_img=None, lsm_cond=None, topo_cond=None, y=None, lr_ups=None, sdf_cond=None, eval_land_only: bool = False, land_mask=None):
    """
        Compute a set of monitoring metrics for EDM models during training.
        Returns a dict with keys:
        - 'edm_cosine': Cosine similarity metric between predicted x0_hat and x0
        - 'hr_lr_corr': Pearson correlation coefficient between predicted x0_hat and lr_ups (if lr_ups is provided)

    """
    if not isinstance(loss_obj, EDMLoss):
        logger.warning("edm_cosine_metric is only defined for EDMLoss. Returning None.")
        return None  # Metric only defined for EDMLoss
    
    B = x0.shape[0]
    device = x0.device
    dtype = x0.dtype

    # Sample sigma from log-normal distribution
    sigma = loss_obj.sample_sigma(B, device, dtype=dtype)
    n = torch.randn_like(x0)
    x_t = x0 + sigma.view(B, 1, 1, 1) * n
    
    # Model is EDMPrecondUNet, predict x0_hat
    x0_hat = model(x_t, sigma, cond_img=cond_img, lsm_cond=lsm_cond, topo_cond=topo_cond, y=y, lr_ups=lr_ups)

    # --- Fallback for HR–LR correlation ---
    # If lr_ups is not provided, try to derive it from cond_img (first channel),
    # which is already upsampled to HR size in the dataloader.
    lr_for_corr = lr_ups
    if (lr_for_corr is None) and (cond_img is not None):
        try:
            if cond_img.ndim == 4 and cond_img.shape[1] >= 1:
                lr_for_corr = cond_img[:, :1, :, :]
        except Exception:
            lr_for_corr = None

    # Choose mask (prefer explicit land_mask, then lsm_cond); fall back to unmasked if none
    mask_use = land_mask if land_mask is not None else lsm_cond
    have_mask = mask_use is not None

    if eval_land_only and have_mask:
        # Land-only metrics using the available mask
        cos = masked_cosine_similarity(x0_hat, x0, mask=mask_use, weighted=True)
        r = masked_corrcoef_per_sample(
            x0_hat,
            lr_for_corr.expand_as(x0_hat) if lr_for_corr is not None else x0_hat,
            mask=mask_use
        ) if lr_for_corr is not None else float('nan')
    else:
        # Either eval_land_only=False or we lack a mask → compute *unmasked* metrics
        if eval_land_only and not have_mask:
            logger.info("eval_land_only=True but no land mask available; falling back to unmasked cosine/correlation.")        
        cos = masked_cosine_similarity(x0_hat, x0, mask=None, weighted=False)
        r = masked_corrcoef_per_sample(
            x0_hat,
            lr_for_corr.expand_as(x0_hat) if lr_for_corr is not None else x0_hat,
            mask=None
        ) if lr_for_corr is not None else float('nan')

    return {'edm_cosine': float(cos), 'hr_lr_corr': float(r)}


# @torch.no_grad() # Disable gradient computation for monitoring
# def edm_cosine_metric(loss_obj, model, x0, *, cond_img=None, lsm_cond=None, topo_cond=None, y=None, lr_ups=None, sdf_cond=None):
#     """
#     Compute the cosine similarity metric for EDM models.
#     Similarity metric between predicted x0_hat and x0 for EDM.
#     """
#     if not isinstance(loss_obj, EDMLoss):
#         logger.warning("edm_cosine_metric is only defined for EDMLoss. Returning None.")
#         return None  # Metric only defined for EDMLoss
    
#     B = x0.shape[0]
#     device = x0.device
#     dtype = x0.dtype

#     # Sample sigma from log-normal distribution
#     sigma = loss_obj.sample_sigma(B, device, dtype=dtype)
#     n = torch.randn_like(x0)
#     x_t = x0 + sigma.view(B, 1, 1, 1) * n
    
#     # Model is EDMPrecondUNet, predict x0_hat
#     x0_hat = model(x_t, sigma, cond_img=cond_img, lsm_cond=lsm_cond, topo_cond=topo_cond, y=y, lr_ups=lr_ups)

#     # Flatten per-sample and compute cosine
#     cos = F.cosine_similarity(x0_hat.flatten(1), x0.flatten(1), dim=1, eps=1e-8).mean()
#     return float(cos)

# @torch.no_grad()
# def hr_lr_corrcoef(loss_obj, model, x0, *, cond_img=None, lsm_cond=None, topo_cond=None, y=None, lr_ups=None):
#     """
#         Compute the Pearson correlation coefficient between predicted x0_hat and lr_ups (upsampled low-res input).
#         Returns mean correlation over the batch.
#     """
#     if not isinstance(loss_obj, EDMLoss):
#         logger.warning("hr_lr_corrcoef is only defined for EDMLoss. Returning None.")
#         return None  # Metric only defined for EDMLoss
#     if lr_ups is None:
#         logger.warning("lr_ups is required for hr_lr_corrcoef. Returning None.")
#         return None

#     B = x0.shape[0]
#     device = x0.device
#     dtype = x0.dtype

#     # Sample sigma from log-normal distribution
#     sigma = loss_obj.sample_sigma(B, device, dtype=dtype)
#     n = torch.randn_like(x0)
#     x_t = x0 + sigma.view(B, 1, 1, 1) * n
    
#     # Model is EDMPrecondUNet, predict x0_hat
#     with torch.no_grad():
#         x0_hat = model(x_t, sigma, cond_img=cond_img, lsm_cond=lsm_cond, topo_cond=topo_cond, y=y, lr_ups=lr_ups)

#     # Compute masked correlation coefficient per sample and average
#     r = masked_corrcoef_per_sample(x0_hat, lr_ups.expand_as(x0_hat), mask=None)
#     return float(r)



@torch.no_grad()
def masked_cosine_similarity(x_pred: torch.Tensor, x_true: torch.Tensor,
                             mask: torch.Tensor | None = None,
                             eps: float = 1e-8,
                             weighted: bool = True) -> torch.Tensor:
    """
    x_pred, x_true: [B, C, H, W]
    mask: [B, 1, H, W] or [B, H, W] or [1, H, W] or [H, W]; 1/True = land
    returns: scalar mean across batch (weighted by valid pixels if weighted=True)
    """
    if mask is None:
        return F.cosine_similarity(x_pred.flatten(1), x_true.flatten(1), dim=1, eps=eps).mean()

    # normalize mask shape & type
    if mask.ndim == 2:
        mask = mask[None, None]                 # [1,1,H,W]
    elif mask.ndim == 3:
        mask = mask[:, None]                    # [B,1,H,W]
    mask = (mask > 0.5) if mask.dtype != torch.bool else mask
    m = mask.to(x_pred.dtype)                   # float 0/1
    if m.shape[0] != x_pred.shape[0]:
        m = m.expand(x_pred.shape[0], 1, *x_pred.shape[-2:])
    m = m.expand_as(x_pred)                     # [B,C,H,W]

    # cosine = (x·y)/(||x||·||y||) but only over land pixels
    num   = (x_pred * x_true * m).flatten(1).sum(dim=1)
    normx = torch.sqrt((x_pred.pow(2) * m).flatten(1).sum(dim=1) + eps)
    normy = torch.sqrt((x_true.pow(2) * m).flatten(1).sum(dim=1) + eps)
    cos_i = num / (normx * normy + eps)         # [B]

    # samples with no land pixels → NaN, drop them
    valid_pix = m.flatten(1).sum(dim=1)         # [B]
    finite = torch.isfinite(cos_i) & (valid_pix > 0)
    if not finite.any():
        return torch.tensor(float('nan'), device=x_pred.device)

    if weighted:
        w = valid_pix[finite]
        return (cos_i[finite] * w).sum() / (w.sum() + eps)
    else:
        return cos_i[finite].mean()

def masked_corrcoef_per_sample(
        a: torch.Tensor, # [B, C, H, W]
        b: torch.Tensor, # [B, C, H, W]
        mask: torch.Tensor | None = None, # [B, 1, H, W] or [B, H, W] (bool or {0,1} float)
        eps: float = 1e-8,
        weighted: bool = True
) -> torch.Tensor:
    """
        Peasron r computed per-sample over masked pixels, then averaged over the batch.
        - Ignores samples with <2 finite pixels or ~0 variance.
        - Works for C>1 (flattens all masked pixels across channels)
        - If weighted = True, averages with weights = number of valid pixels per sample
    """
    a = a.detach()
    b = b.detach()

    if b.shape[1] == 1 and a.shape[1] > 1:
        # Broadcast single-channel b to match a's channels
        b = b.expand_as(a)

    B = a.shape[0]
    rs, ws = [], []
    
    for i in range(B):
        mi = None
        if mask is not None:
            mi = mask[i]
            if mi.dtype != torch.bool:
                mi = (mi > 0.5)
            # Broadcast to all channels
            if mi.ndim == 3:
                mi = mi.expand_as(a[i])

        ai = a[i][mi] if mi is not None else a[i].reshape(-1)
        bi = b[i][mi] if mi is not None else b[i].reshape(-1)

        # Keep only finite values
        finite = torch.isfinite(ai) & torch.isfinite(bi)
        ai = ai[finite]
        bi = bi[finite]
        n = ai.numel()
        if n < 2:
            continue  # Not enough valid pixels

        # Standardize
        ai = ai - ai.mean()
        bi = bi - bi.mean()
        std_prob = ai.std(unbiased=False) * bi.std(unbiased=False)
        if std_prob.abs() < eps:
            continue  # Near-zero variance

        r = (ai * bi).mean() / (std_prob + eps)
        if torch.isfinite(r):
            rs.append(r)
            ws.append(torch.tensor(float(n), device=r.device))
    if not rs:
        return torch.tensor(float('nan'), device=a.device)
    
    rs = torch.stack(rs)
    if weighted:
        ws = torch.stack(ws)
        r_mean = (rs * ws).sum() / (ws.sum() + eps)
        return r_mean
    else:
        return rs.mean()


def report_precip_extremes(x_bt: torch.Tensor, name: str, cap_mm_day: float = 500.0):
    """
        Reports extremes in a back-transformed precipitation tensor.
        Values below 0 are counted as negative, values above cap_mm_day are counted as extreme.
    """
    flat = x_bt.flatten(1)
    p999 = torch.quantile(flat, 0.999, dim=1)
    mx = torch.max(flat, dim=1).values
    n_ex = 0
    vals_ex = []
    n_b0 = 0
    vals_b0 = []
    for i, (p, m) in enumerate(zip(p999.tolist(), mx.tolist())):
        if m > max(5.0 * p, cap_mm_day):
            logger.info(f"{name} sample {i} has extreme precipitation: max={m:.1f} mm/day > max(5xp99.9={p:.1f} mm/day)")
            n_ex += 1
            vals_ex.append(m)
        if m < 0:
            logger.info(f"{name} sample {i} has negative precipitation: max={m:.1f} mm/day < 0")
            n_b0 += 1
            vals_b0.append(m)
    if n_b0 > 0 and n_ex > 0:
        return {'has_extreme': True, 'n_extreme': n_ex, 'extreme_values': vals_ex,
                'has_below_zero': True, 'n_below_zero': n_b0, 'below_zero_values': vals_b0}
    if n_ex > 0:
        return {'has_extreme': True, 'n_extreme': n_ex, 'extreme_values': vals_ex}
    if n_b0 > 0:
        return {'has_below_zero': True, 'has_below_zero': True, 'n_below_zero': n_b0, 'below_zero_values': vals_b0}

    return {'has_extreme': False}






# === Diagnostics helpers for EDM training ===
@torch.no_grad()
def _finite_mask(x: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(x)
@torch.no_grad()
def tensor_stats(x: torch.Tensor, name: str, pctiles=(0.1, 1, 5, 50, 95, 99, 99.9), log_fn=logger.info):
    """
        Quick, safe stats with NaN/Inf awareness. Logs: shape, dtype, device, finite ratio, mean, std, min, max, a few percentiles. 
    """
    if x is None:
        log_fn(f"{name}: None")
        return
    
    # cpu snapshot for percentiles (subsample to keep cheap)
    x_detached = x.detach()
    mask = _finite_mask(x_detached)
    n_total = x_detached.numel()
    n_finite = int(mask.sum().item())

    log_fn(f"[{name}] shape={tuple(x_detached.shape)}, dtype={x_detached.dtype}, device={x_detached.device}, finite_ratio={n_finite} / {n_total} ({100.0 * n_finite / n_total:.2f}%)")

    if n_finite == 0:
        log_fn(f"[{name}] !!! No finite values (NaN/Inf everywhere) !!!")
        return
    
    xf = x_detached[mask]
    # downsample if huge
    if xf.numel() > 1_000_000:
        idx = torch.randint(0, xf.numel(), (200_000,), device=xf.device)
        xf = xf.view(-1)[idx]

    # core stats on device
    x_min = float(xf.min().item())
    x_max = float(xf.max().item())
    x_mean = float(xf.mean().item())
    x_std = float(xf.std(unbiased=False).item())

    # move small vector for percentiles
    xcpu = xf.float().flatten().cpu()
    # percentiles
    pcts = {}
    for p in pctiles:
        q = torch.quantile(xcpu, torch.tensor(float(p) / 100.0))
        pcts[p] = float(q.item())

    pts_str = " ".join([f"P{int(p)}={pcts[p]:.4g}" for p in pctiles])
    log_fn(f"[{name}] min={x_min:.4g} max={x_max:.4g} mean={x_mean:.4g} std={x_std:.4g} | {pts_str}")

@torch.no_grad()
def save_histogram(x:torch.Tensor, save_path: str, bins: int = 200, range_: tuple[float,float] | None = None):
    """
        Save a histogram of tensor x to the specified path.
        x: input tensor
    """
    x = x.detach().flatten()
    x = x[torch.isfinite(x)]  # Keep only finite values
    if x.numel() == 0:
        logger.warning(f"[save_histogram]: No finite values in tensor, skipping histogram save to {save_path}.")
        return
    xcpu = x.float().cpu()
    lo = float(xcpu.min().item()) if range_ is None else range_[0]
    hi = float(xcpu.max().item()) if range_ is None else range_[1]
    
    hist, edges = np.histogram(xcpu.numpy(), bins=bins, range=(lo, hi))
    name = f"hist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path = os.path.join(save_path, name)
    with open(path, 'w') as f:
        json.dump({"edges": edges.tolist(), "hist": hist.tolist()}, f)

@torch.no_grad()
def plot_saved_histograms(save_path: str, fig_save_path: str | None = None):
    """
        Plot all saved histograms in the current directory (files named hist_*.json).
    """
    import glob
    files = glob.glob(os.path.join(save_path, "hist_*.json"))
    if len(files) == 0:
        logger.warning(f"No histogram files found in {save_path}.")
        return
    
    plt.figure(figsize=(10,6))
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
        edges = np.array(data['edges'])
        hist = np.array(data['hist'])
        centers = 0.5 * (edges[:-1] + edges[1:])
        plt.plot(centers, hist, label=os.path.basename(file))
    
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Saved Histograms')
    plt.legend()
    plt.grid()

    if fig_save_path is not None:
        plt.savefig(fig_save_path)
        logger.info(f"Saved histogram plot to {fig_save_path}.")
    else:
        plt.show()
    plt.close()
    

