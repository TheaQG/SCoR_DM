import torch
import torch.nn as nn
from typing import Optional
from score_unet import EDMPrecondUNet


class EDMLoss(nn.Module):
    """
        Karras EDM loss (simple form): sample sigma from ~ logN(P_mean, P_std),
        perturb x0 -> x_t, predict x0_hat with preconditioning, MSE on x0
    """
    def __init__(self, P_mean: float = -1.2,
                 P_std: float = 1.2,
                 sigma_data: float = 1.0,
                 use_sdf_weight: bool = False,
                 max_land_weight: float = 1.0,
                 min_sea_weight: float = 0.5,
                 normalize_sdf: bool = True):
        super().__init__()
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.use_sdf_weight = use_sdf_weight
        self.max_land_weight = max_land_weight
        self.min_sea_weight = min_sea_weight
        self.normalize_sdf = normalize_sdf

    def sample_sigma(self, B: int, device: torch.device, dtype=None):
        return (torch.randn(B, device=device, dtype=dtype) * self.P_std + self.P_mean).exp()

    def forward(self,
                edm_model: EDMPrecondUNet,
                x0: torch.Tensor,
                *,
                cond_img: torch.Tensor | None = None,
                lsm_cond: torch.Tensor | None = None,
                topo_cond: torch.Tensor | None = None,
                y: torch.Tensor | None = None,
                lr_ups: torch.Tensor | None = None,
                sdf_cond: torch.Tensor | None = None,
                pixel_weight_map: torch.Tensor | None = None
                ):
        B = x0.shape[0]
        device = x0.device
        dtype = x0.dtype

        sigma = self.sample_sigma(B, device, dtype=dtype)  # [B]
        
        n = torch.randn_like(x0)  # [B, C, H, W]
        x_t = x0 + sigma.view(B, 1, 1, 1) * n  # [B, C, H, W]
        x0_hat = edm_model(x_t, sigma, cond_img=cond_img, lsm_cond=lsm_cond, topo_cond=topo_cond, y=y, lr_ups=lr_ups)

        # EDM weight (per-sample scalra) - added for guiding the model to focus on low-noise samples
        s2 = sigma**2
        sd2 = self.sigma_data**2
        w = (s2 + sd2) / ((sigma * self.sigma_data)**2)
        w = w.view(B, 1, 1, 1)

        err2 = (x0_hat - x0)**2  # [B, C, H, W]

        # Optional rain-gate pixel weighting
        if pixel_weight_map is not None:
            wmap = pixel_weight_map
            # Expand channel dim if needed
            if err2.dim() == 4 and wmap.dim() == 4 and err2.shape[1] != wmap.shape[1]:
                wmap = wmap.expand(err2.shape[0], err2.shape[1], *err2.shape[2:])
            err2 = err2 * wmap  # [B, C, H, W

        if self.use_sdf_weight and (sdf_cond is not None):
            # Weighting based on SDF: higher weights near land-sea boundary
            sdf_w = torch.sigmoid(sdf_cond) * (self.max_land_weight - self.min_sea_weight) + self.min_sea_weight
            if self.normalize_sdf:
                # Keep mean ~1 to avoid changing overall loss scale
                sdf_w = sdf_w / (sdf_w.mean(dim=(1,2,3), keepdim=True) + 1e-8)

            err2 = err2 * sdf_w  # [B, C, H, W]
        
        loss = (w * err2).mean()  # weighted MSE
        
        return loss

class DSMLoss(nn.Module):
    """
        VE-DSM (Denoising Score Matching) loss: 
        
    """
    def __init__(self, marginal_prob_std_fn, t_eps: float = 1e-3,
                 use_sdf_weight: bool = True,
                 max_land_weight: float = 1.0,
                 min_sea_weight: float = 0.5):
        super().__init__()
        self.marginal_prob_std_fn = marginal_prob_std_fn
        self.t_eps = t_eps
        self.use_sdf_weight = use_sdf_weight
        self.max_land_weight = max_land_weight
        self.min_sea_weight = min_sea_weight

    def forward(self, model, x,
                *,
                y: Optional[torch.Tensor] = None,
                cond_img: Optional[torch.Tensor] = None,
                lsm_cond: Optional[torch.Tensor] = None,
                topo_cond: Optional[torch.Tensor] = None,
                sdf_cond: Optional[torch.Tensor] = None,
                pixel_weight_map: Optional[torch.Tensor] = None,
                ):
        B = x.shape[0]
        device = x.device

        # sample t away from 0 to avoid dead gradients
        t = torch.rand(B, device=device) * (1.0 - self.t_eps) + self.t_eps  # [B], uniform(T_eps, 1)
        z = torch.randn_like(x)  # [B, C, H, W], standard normal noise
        std = self.marginal_prob_std_fn(t)  # [B], std of perturbation kernel
        x_t = x + std.view(B, 1, 1, 1) * z  # [B, C, H, W], perturbed x at time t

        # forward score model
        score = model(x_t, t, y=y, cond_img=cond_img, lsm_cond=lsm_cond, topo_cond=topo_cond)  # score estimate

        per_pix = (score * std.view(B, 1, 1, 1) + z)**2  # [B, C, H, W]
        if self.use_sdf_weight and (sdf_cond is not None):
            sdf_w = torch.sigmoid(sdf_cond) * (self.max_land_weight - self.min_sea_weight) + self.min_sea_weight
            per_pix = per_pix * sdf_w
        if pixel_weight_map is not None:
            wmap = pixel_weight_map
            if per_pix.dim() == 4 and wmap.dim() == 4 and per_pix.shape[1] != wmap.shape[1]:
                wmap = wmap.expand(per_pix.shape[0], per_pix.shape[1], *per_pix.shape[2:])
            per_pix = per_pix * wmap
        loss = torch.mean(torch.sum(per_pix, dim=(1, 2, 3)))

        return loss
