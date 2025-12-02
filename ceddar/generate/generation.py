"""
    EDM-downscaling - generation runner (mirrors TrainingPipeline_general.generate_and_plot_samples)

    Outputs per-day npzs:
        - ensembles/{date}.npz   -> {'ens': [M,1,H,W]}  (model space)
        - pmm/{date}.npz         -> {'pmm': [1,1,H,W]}  (model space)
        - lr_hr/{date}.npz       -> {'hr': [1,1,H,W] or None, 'lr_hr': [1,1,H,W] or None} (model space)
    and meta/manifest.json for reproducibility
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from scor_dm.special_transforms import build_back_transforms_from_stats, lr_baseline_to_hr_zspace
from scor_dm.utils import extract_samples, get_model_string
from scor_dm.score_sampling import edm_sampler
from scor_dm.monitoring import (
    report_precip_extremes,
)
from evaluate.metrics_univariate import (
    pmm_from_ensemble,
)

logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    output_root: str
    ensemble_size: int = 32
    max_dates: Optional[int] = -1  # -1 means all dates
    sampler_steps: int = 40
    seed: int = 504

    # EDM controls
    use_edm: bool = True
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    S_churn: float = 0.0
    S_min: float = 0.0
    S_max: float = float("inf")
    S_noise: float = 1.0
    predict_residual: bool = False  # if True, pass lr_ups to sampler

    # Saving: "physical" (default), "model", or "both"
    save_space: str = "physical"
    physical_dtype: str = "float32"  # when saving physical space, cast to this dtype

    # Logging
    log_every: int = 25


def _save_npz(path: Path, **arrays):
    """Save arrays to npz at path, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {}
    for k, v in arrays.items():
        if v is None:
            out[k] = None
        elif isinstance(v, np.ndarray):
            out[k] = v
        elif isinstance(v, torch.Tensor):
            out[k] = v.detach().float().cpu().numpy()
        else:
            out[k] = np.array(v)
    np.savez_compressed(path, **out)

def _repeat_to_M(x, M: int):
    """ Prepare tensor to ensemble size M by repeating along batch dim."""
    if x is None:
        return None
    # Assume batch-first [B, ...]; take first item in batch and repeat to M
    x1 = x[:1]
    reps = [M] + [1] * (x1.dim() - 1)
    return x1.repeat(*reps)

def _build_back_transforms(cfg: dict):
    full_domain_dims_hr = cfg['highres'].get('full_domain_dims', None)
    full_domain_dims_str_hr = f"{full_domain_dims_hr[0]}x{full_domain_dims_hr[1]}" if full_domain_dims_hr is not None else "full_domain"
    crop_region_hr = cfg['highres'].get('cutout_domains', None)
    crop_region_hr_str = '_'.join(map(str, crop_region_hr)) if crop_region_hr is not None else 'no_crop'

    full_domain_dims_lr = cfg['lowres'].get('full_domain_dims', None)
    full_domain_dims_str_lr = f"{full_domain_dims_lr[0]}x{full_domain_dims_lr[1]}" if full_domain_dims_lr is not None else "full_domain"
    crop_region_lr = cfg['lowres'].get('cutout_domains', None)
    crop_region_lr_str = '_'.join(map(str, crop_region_lr)) if crop_region_lr is not None else 'no_crop'

    return build_back_transforms_from_stats(
        hr_var=cfg['highres']['variable'],
        hr_model=cfg['highres']['model'],
        domain_str_hr=full_domain_dims_str_hr,
        crop_region_str_hr=crop_region_hr_str,
        hr_scaling_method=cfg['highres']['scaling_method'],
        hr_buffer_frac=cfg['highres'].get('buffer_frac', 0.0),
        lr_vars=cfg['lowres']['condition_variables'],
        lr_model=cfg['lowres']['model'],
        domain_str_lr=full_domain_dims_str_lr,
        crop_region_str_lr=crop_region_lr_str,
        lr_scaling_methods=cfg['lowres']['scaling_methods'],
        lr_buffer_frac=cfg['lowres'].get('buffer_frac', 0.0),
        split='train',  # match training.generate_and_plot_samples
        stats_dir_root=cfg['paths']['stats_load_dir'],
        eps=cfg['transforms'].get('prcp_eps', 0.01),
    )


class GenerationRunner:
    def __init__(self, model: torch.nn.Module, cfg: dict, device: str, out_root: Path, gen_config: GenerationConfig, quicklook: bool = False):

        self.model = model
        self.cfg = cfg
        self.device = device
        self.out_root = out_root
        # Ensure base output root exists even for quicklook (no subdirs yet=> 
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.gen_config = gen_config

        self.global_prcp_eps = cfg['transforms'].get('prcp_eps', 0.01)


        self.hr_var = cfg['highres']['variable']
        self.hr_scaling_method = cfg['highres']['scaling_method']
        self.full_domain_dims_hr = cfg['highres']['full_domain_dims']
        self.crop_region_hr = cfg['highres']['cutout_domains']

        self.lr_vars = cfg['lowres']['condition_variables']
        self.lr_scaling_methods = cfg['lowres']['scaling_methods']
        self.full_domain_dims_lr = cfg['lowres']['full_domain_dims']
        self.crop_region_lr = cfg['lowres']['cutout_domains']

        # Indices of LR channels corresponding to the HR target variable (can be multiple)
        self._lr_target_indices = []
        if self.hr_var in self.lr_vars:
            self._lr_target_indices = [i for i, v in enumerate(self.lr_vars) if v == self.hr_var]
            
        # Cache strings for stats lookups
        self._dom_hr_str = f"{self.full_domain_dims_hr[0]}x{self.full_domain_dims_hr[1]}" if self.full_domain_dims_hr is not None else "full_domain"
        self._dom_lr_str = f"{self.full_domain_dims_lr[0]}x{self.full_domain_dims_lr[1]}" if self.full_domain_dims_lr is not None else "full_domain"
        self._crop_hr_str = '_'.join(map(str, self.crop_region_hr)) if self.crop_region_hr is not None else "no_crop"
        self._crop_lr_str = '_'.join(map(str, self.crop_region_lr)) if self.crop_region_lr is not None else "no_crop"
        self._stats_root = self.cfg['paths']['stats_load_dir']
        self._hr_method_for_target = self.hr_scaling_method
        # Assume LR scaling methods is a list aligned with lr_vars; get method for the target variable
        if self.hr_var in self.lr_vars:
            idx_t = self.lr_vars.index(self.hr_var)
            self._lr_method_for_target = self.lr_scaling_methods[idx_t]
        else:
            self._lr_method_for_target = None  # Target variable not in LR vars
            logger.warning(f"HR target variable '{self.hr_var}' not found in LR condition variables {self.lr_vars}. Cannot determine LR scaling method for target - residuals may not be aligned.")
        

        # cutout stationarity control + optional static LSM for evaluation
        # Support both plain dict configs and OmegaConf-style DictConfig.
        if isinstance(cfg, dict):
            eval_cfg = cfg.get("evaluation", None)
        else:
            eval_cfg = getattr(cfg, "evaluation", None)

        if isinstance(eval_cfg, dict):
            eval_stationary_cfg = eval_cfg.get("stationary_cutout", None)
        else:
            eval_stationary_cfg = getattr(eval_cfg, "stationary_cutout", None) if eval_cfg is not None else None

        if isinstance(eval_stationary_cfg, dict):
            # mirror training_utils logic: 'hr_enabled' and 'hr_bounds'
            self.stationary_cutout = bool(eval_stationary_cfg.get("hr_enabled", True))
            self._hr_bounds_eval = eval_stationary_cfg.get("hr_bounds", None)
        elif eval_stationary_cfg is not None:
            # allow legacy boolean / object configs and try to read hr_bounds attribute if present
            self.stationary_cutout = bool(eval_stationary_cfg)
            self._hr_bounds_eval = getattr(eval_stationary_cfg, "hr_bounds", None)
        else:
            # no explicit evaluation cutout config
            self.stationary_cutout = True            
            self._hr_bounds_eval = None

        # Fallbacks: allow geometry configured under full_gen_eval or highres if not set above
        if self._hr_bounds_eval is None:
            if isinstance(cfg, dict):
                fg = cfg.get("full_gen_eval", None)
            else:
                fg = getattr(cfg, "full_gen_eval", None)

            if isinstance(fg, dict):
                sc = fg.get("stationary_cutout", None)
            else:
                sc = getattr(fg, "stationary_cutout", None) if fg is not None else None

            hb = None
            if isinstance(sc, dict):
                hb = sc.get("hr_bounds", None)
            elif sc is not None:
                hb = getattr(sc, "hr_bounds", None)
            if hb is not None:
                self._hr_bounds_eval = hb

        if self._hr_bounds_eval is None:
            if isinstance(cfg, dict):
                highres_cfg = cfg.get("highres", None)
            else:
                highres_cfg = getattr(cfg, "highres", None)

            if isinstance(highres_cfg, dict):
                sc = highres_cfg.get("stationary_cutout", None)
            else:
                sc = getattr(highres_cfg, "stationary_cutout", None) if highres_cfg is not None else None

            hb = None
            if isinstance(sc, dict):
                hb = sc.get("bounds", None)
            elif sc is not None:
                hb = getattr(sc, "bounds", None)
            if hb is not None:
                self._hr_bounds_eval = hb


        logger.info("[generation] Evaluation HR bounds for static LSM: %s", self._hr_bounds_eval)

        self._first_lsm: Optional[torch.Tensor] = None
        self._lsm_stationary_ok: bool = True
        self._lsm_static: Optional[torch.Tensor] = None

        # dirs (if quicklook, don't create)
        if not quicklook:
            (self.out_root / 'lsm').mkdir(parents=True, exist_ok=True)
            (self.out_root / 'ensembles').mkdir(parents=True, exist_ok=True)
            (self.out_root / 'pmm').mkdir(parents=True, exist_ok=True)
            (self.out_root / 'lr_hr').mkdir(parents=True, exist_ok=True)
            (self.out_root / 'meta').mkdir(parents=True, exist_ok=True)

            # Additional dirs when saving physical/model explicitly
            if self.gen_config.save_space in ('physical', 'both'):
                (self.out_root / 'ensembles_phys').mkdir(parents=True, exist_ok=True)
                (self.out_root / 'pmm_phys').mkdir(parents=True, exist_ok=True)
                (self.out_root / 'lr_hr_phys').mkdir(parents=True, exist_ok=True)
            if self.gen_config.save_space in ('model', 'both'):
                (self.out_root / 'ensembles_model').mkdir(parents=True, exist_ok=True)
                (self.out_root / 'pmm_model').mkdir(parents=True, exist_ok=True)

        # sentinel 
        mon_cfg = cfg.get('monitoring', {}).get('extreme_prcp', {})
        self.sentinel_thr = float(mon_cfg.get('threshold_mm', 500.0)) # mm/day
        self.clamp_in_gen = bool(mon_cfg.get('clamp_in_generation', True))

        # back-transforms for quick sanity checks
        self.back_transforms = _build_back_transforms(cfg)
        self.bt_gen = self.back_transforms.get('generated', None)
        self.bt_hr = self.back_transforms.get(f"{cfg['highres']['variable']}_hr", None)
        # Get the matching LR back-transform for the target variable, if available
        self.bt_lr_target = None
        if self.hr_var in self.lr_vars:
            idx_t = self.lr_vars.index(self.hr_var)
            self.bt_lr_target = self.back_transforms.get(f"{self.hr_var}_lr", None)
        else:
            self.bt_lr_target = None
            logger.warning(f"HR target variable '{self.hr_var}' not found in LR condition variables {self.lr_vars}. Cannot build LR back-transform for target variable.")
        
        # sampler selection
        if bool(cfg.get('edm', {}).get('enabled', False)) or self.gen_config.use_edm:
            self._sampler_kind = 'edm'
            self._sampler_fn = edm_sampler
        else:
            logger.error("Currently only EDM sampler is supported in generation.")
            # st = cfg.get('sampler', {}).get('sampler_type', 'pc_sampler')
            # if st == 'pc_sampler':
            #     self._sampler_kind = 'sde'
            #     self._sampler_fn = pc_sampler
            # elif st == 'Euler_Maruyama_sampler':
            #     self._sampler_kind = 'sde'
            #     self._sampler_fn = Euler_Maruyama_sampler
            # elif st == 'ode_sampler':
            #     self._sampler_kind = 'ode'
            #     self._sampler_fn = ode_sampler
            # else:
            #     raise ValueError(f"Unknown sampler_type {st} in config")

    def _get_static_lsm(self) -> Optional[torch.Tensor]:
        """
        Fallback land-sea mask used when the dataset does not yield an LSM tensor.

        This loads the full-domain LSM from cfg['paths']['lsm_path'], flips it to match
        the DANRA orientation (np.flipud), and then crops to the evaluation HR bounds
        if they are set. The HR bounds are resolved in this order:
          1) cfg['evaluation']['stationary_cutout']['hr_bounds']
          2) cfg['full_gen_eval']['stationary_cutout']['hr_bounds']
          3) cfg['highres']['stationary_cutout']['bounds']

        The result is cached as a torch.bool tensor on CPU.
        """
        # Return cached static mask if already built
        if self._lsm_static is not None:
            return self._lsm_static

        # Resolve LSM path
        try:
            lsm_path = self.cfg["paths"]["lsm_path"]
        except Exception as e:
            logger.warning("[generation] _get_static_lsm: could not resolve paths.lsm_path: %s", e)
            return None

        # Load LSM from npz or raw array
        try:
            arr = np.load(lsm_path, allow_pickle=True)
            if isinstance(arr, np.lib.npyio.NpzFile):  # type: ignore
                data = None
                # Try typical keys in order of preference
                for k in ("lsm_hr", "lsm", "data", "mask"):
                    if k in arr.files:
                        data = arr[k]
                        break
                if data is None:
                    logger.warning(
                        "[generation] _get_static_lsm: no suitable key in %s (tried lsm_hr, lsm, data, mask)",
                        lsm_path,
                    )
                    return None
            else:
                data = arr
        except Exception as e:
            logger.warning("[generation] _get_static_lsm: failed to load LSM from %s: %s", lsm_path, e)
            return None

        data = np.asarray(data)
        # Ensure 2D; squeeze singleton dimensions if necessary
        if data.ndim != 2:
            data = np.squeeze(data)
            if data.ndim != 2:
                logger.warning(
                    "[generation] _get_static_lsm: expected 2D LSM array, got shape %s after squeeze", data.shape
                )
                return None

        # Flip to match DANRA orientation (as in previous implementation)
        data = np.flipud(data)

        # Optional crop to stationary HR evaluation bounds [y0, y1, x0, x1]
        hr_bounds = self._hr_bounds_eval
        if hr_bounds is not None:
            if len(hr_bounds) != 4:
                logger.warning(
                    "[generation] _get_static_lsm: hr_bounds must have length 4, got %s", hr_bounds
                )
            else:
                y0, y1, x0, x1 = [int(v) for v in hr_bounds]
                data = data[y0:y1, x0:x1]

        # Convert to bool mask and cache
        lsm_bool = (data > 0.5)
        self._lsm_static = torch.from_numpy(lsm_bool.astype(np.bool_))
        logger.info(
            "[generation] Built static LSM from %s with shape %s (hr_bounds=%s)",
            lsm_path,
            tuple(self._lsm_static.shape),
            hr_bounds,
        )

        return self._lsm_static
    
    @torch.no_grad()
    def _build_lr_ups_baseline(self, cond_images: torch.Tensor | None):
        """
            Extract LR baseline channel (same variable as HR target) from cond_images and upsample to HR resolution.
            Ensure it is expressed in HR z-space (or HR min-max space) before using for residual EDM.
            Returns [B, 1, H, W] or raises if unavailable when predict_residual is True.
        """
        if cond_images is None:
            raise ValueError("cond_images is None, cannot extract LR baseline for residual prediction.")
        
        cond_vars = self.cfg['lowres']['condition_variables']
        target_var = self.hr_var
        if target_var not in cond_vars:
            raise ValueError(f"Target variable '{target_var}' not found in condition variables {cond_vars}, cannot extract LR baseline for residual prediction.")
        
        idx = cond_vars.index(target_var)
        if cond_images.shape[1] <= idx:
            raise ValueError(f"cond_images has shape {cond_images.shape}, cannot extract channel index {idx} for variable '{target_var}'.")
        lr_in_lr_space = cond_images[:, idx:idx+1, :, :]  # [B, 1, h, w] - cond images already upsampled to HR size

        if self.cfg.get('edm', {}).get('baseline_space', 'hr') == 'lr':
            logger.info(f"baseline_space requested is 'lr'; using LR baseline channel as-is in LR space for residual prediction.")
            return lr_in_lr_space  # Already in LR space, just upsampled to HR size
        
        # Find the LR scaling method corresponding to baseline channel
        lr_method_for_baseline = self._lr_method_for_target 

        # Ensure lr_method_for_baseline is a string
        if lr_method_for_baseline is None:
            raise ValueError("LR scaling method for baseline is None. Cannot proceed with lr_baseline_to_hr_zspace. Please check your configuration.")

        # logger.info(f"Converting LR baseline channel from LR space to HR space using lr_baseline_to_hr_zspace with LR method '{lr_method_for_baseline}' and HR method '{self.hr_scaling_method}'.")
        # Remap using transform/back-transform stack
        lr_in_hr_space = lr_baseline_to_hr_zspace(
            lr_chan_norm=lr_in_lr_space,
            # LR meta
            lr_variable=self.hr_var,
            lr_model=self.cfg['lowres']['model'],
            lr_domain_str=self._dom_lr_str,
            lr_crop_region_str=self._crop_lr_str,
            lr_split=self.cfg['transforms'].get('scaling_split', 'train'),
            lr_scaling_method=lr_method_for_baseline,
            lr_buffer_frac=self.cfg['lowres'].get('buffer_frac', 0.0),
            lr_stats_dir_root=self.cfg['paths']['stats_load_dir'],
            # HR meta
            hr_variable=self.hr_var,
            hr_model=self.cfg['highres']['model'],
            hr_domain_str=self._dom_hr_str,
            hr_crop_region_str=self._crop_hr_str,
            hr_split=self.cfg['transforms'].get('scaling_split', 'train'),
            hr_scaling_method=self.hr_scaling_method,
            hr_buffer_frac=self.cfg['highres'].get('buffer_frac', 0.0),
            hr_stats_dir_root=self.cfg['paths']['stats_load_dir'],

            eps=self.global_prcp_eps
        )

        return lr_in_hr_space
    
    @torch.no_grad()
    def run(self, gen_dataloader, save=True, output_results: bool = False):
        model_name = get_model_string(self.cfg)
        logger.info(f"[generation] Using model: {model_name}")

        n_days = 0
        edm_cfg = self.cfg.get('edm', {})
        guidance_cfg = self.cfg.get('classifier_free_guidance', {})
        M = int(self.gen_config.ensemble_size)
        steps = int(edm_cfg.get('sampling_steps', self.gen_config.sampler_steps)) # default to config value if not in model cfg
        img_size = int(self.cfg['highres']['data_size'][0])  # assume square 
        results = [] if output_results else None

        logger.info(f"[generation] cfg: ensemble_size={M}, steps={steps}, save_space={self.gen_config.save_space}, predict_residual={self.gen_config.predict_residual}, out_root={self.out_root}")

        for idx, samples in enumerate(tqdm(gen_dataloader, desc="Generating samples", unit='batch')):
            hr_phys = None
            lr_phys = None
            dates = samples['date'] if ('date' in samples and isinstance(samples['date'], (list, tuple)) and len(samples['date']) > 0) else [f"idx{idx:04d}"]
            date0 = dates[0]  # use first date for naming
            logger.info(f"[generation] Generating for date {date0} ({idx+1}/{len(gen_dataloader)}) with ensemble size {M}.")
            # Extract model-space tensors
            x_gen, y_gen, cond_images_gen, lsm_hr_gen, lsm_gen, sdf_gen, topo_gen, hr_points_gen, lr_points_gen = extract_samples(samples, self.device)

            # --- Save/check land-sea mask(s) ---
            try:
                # Prefer LSM from dataset; if missing, fall back to static LSM from paths.lsm_path
                lsm0 = None

                # lsm_hr_gen is expected as [B,1,H,W] bool/0-1
                lsm = lsm_hr_gen
                if lsm is not None and torch.is_tensor(lsm):
                    lsm_cpu = (lsm.detach().cpu() > 0.5).to(torch.bool)
                    # assume B==1 in generation; take [0]
                    lsm0 = lsm_cpu[0, 0] if lsm_cpu.dim() == 4 else lsm_cpu.squeeze()
                else:
                    static_lsm = self._get_static_lsm()
                    if static_lsm is not None:
                        lsm0 = static_lsm
                        logger.info("[generation] Using static LSM fallback for date %s", dates[0])

                if lsm0 is not None:                    
                    # Always save per-date mask if saving is enabled
                    if save:
                        _save_npz(self.out_root / 'lsm' / f'{dates[0]}.npz', lsm_hr=lsm0.numpy())
                        logger.info("[generation] Saved per-date land mask → %s", self.out_root / 'lsm' / f'{dates[0]}.npz')
                    # Set/compare canonical mask, and save canonical on first encounter if saving
                    if self._first_lsm is None:
                        self._first_lsm = lsm0.clone()
                        if self.stationary_cutout and save:
                            _save_npz(self.out_root / 'meta' / 'land_mask.npz', lsm_hr=lsm0.numpy())
                            logger.info("[generation] Saved canonical land mask → %s", self.out_root / 'meta' / 'land_mask.npz')
                    else:
                        if not torch.equal(self._first_lsm, lsm0):
                            self._lsm_stationary_ok = False
            except Exception as e:
                logger.warning(f"[generation] Could not record LSM for {dates[0]}: {e}")

            # Optional baseline for residual EDM
            lr_ups_baseline = None
            if (bool(self.cfg.get('edm', {}).get('enabled', False)) and bool(self.cfg['edm'].get('predict_residual', False))) or self.gen_config.predict_residual:
                lr_ups_baseline = self._build_lr_ups_baseline(cond_images_gen)  # [B,1,H,W] or None

            # Freeze conditioning to a single date and tile to M samples
            if x_gen is not None and x_gen.shape[0] != 1:
                logger.warning(f"[generation] x_gen batch size {x_gen.shape[0]} != 1; freezing to first item and tiling to ensemble size {M}.")


            y_1 = y_gen[:1] if y_gen is not None else None
            cond_img_1 = cond_images_gen[:1] if cond_images_gen is not None else None
            lsm_1 = lsm_gen[:1] if lsm_gen is not None else None
            topo_1 = topo_gen[:1] if topo_gen is not None else None
            lr_ups_1 = lr_ups_baseline[:1] if lr_ups_baseline is not None else None
            x_hr_1 = x_gen[:1] if x_gen is not None else None

            y_M = _repeat_to_M(y_1, M)
            cond_images_M = _repeat_to_M(cond_img_1, M)
            lsm_M = _repeat_to_M(lsm_1, M)
            topo_M = _repeat_to_M(topo_1, M)
            lr_ups_M = _repeat_to_M(lr_ups_1, M)

            # Sample ensemble (model space)
            if self._sampler_kind == 'edm':
                generated = self._sampler_fn(
                    score_model=self.model,
                    batch_size=M,
                    num_steps=steps,
                    device=self.device,
                    img_size=img_size,
                    y=y_M,
                    cond_img=cond_images_M,
                    lsm_cond=lsm_M,
                    topo_cond=topo_M,
                    sigma_min=float(edm_cfg.get('sigma_min', self.gen_config.sigma_min)),
                    sigma_max=float(edm_cfg.get('sigma_max', self.gen_config.sigma_max)),
                    rho=float(edm_cfg.get('rho', self.gen_config.rho)),
                    S_churn=float(edm_cfg.get('S_churn', self.gen_config.S_churn)),
                    S_min=float(edm_cfg.get('S_min', self.gen_config.S_min)),
                    S_max=float(edm_cfg.get('S_max', self.gen_config.S_max)),
                    S_noise=float(edm_cfg.get('S_noise', self.gen_config.S_noise)),
                    lr_ups=lr_ups_M,
                    cfg_guidance=guidance_cfg if guidance_cfg.get('enabled', False) else None,
                    sigma_star=float(edm_cfg.get('sigma_star', 1.0)),
                )
            else:
                raise NotImplementedError("Currently only EDM sampler is supported in generation.")
            
            ens_model = generated.detach().cpu().float() # [M,1,H,W]

            # PMM (model space) - pmm_full is [1, 1, H, W]
            ens_for_pmm = ens_model.squeeze(1).unsqueeze(0)  # [1, M, H, W] (1 batch for pmm fn)
            pmm_full = pmm_from_ensemble(ens_for_pmm) # [1,1,H,W]

            # Back-transform to physical sapce (always compute for saving/evaluation)
            gen_phys = None
            pmm_phys = None
            hr_phys = None
            try:
                if callable(self.bt_gen):
                    gen_phys = self.bt_gen(ens_model) # [M,1,H,W] -> physical space
                    pmm_phys = self.bt_gen(pmm_full) # [1,1,H,W] -> physical space
                if callable(self.bt_hr) and (x_gen is not None):
                    hr_phys = self.bt_hr(x_gen)
            except Exception as e:
                logger.warning(f"[generation] Failed to back-transform generated samples or HR: {e}")

            # Extreme sentinel/clamp on physical ensemble (does not modify saved model-space arrays)
            if gen_phys is not None:
                try:
                    gen_phys_t = gen_phys if isinstance(gen_phys, torch.Tensor) else torch.tensor(gen_phys)
                    chk = report_precip_extremes(x_bt=gen_phys_t, name='generate_hr', cap_mm_day=self.sentinel_thr)
                    if chk.get('has_extreme', False) and self.clamp_in_gen:
                        gen_phys = torch.clamp(gen_phys_t, min=0.0, max=self.sentinel_thr)
                        logger.warning(f"[generation] Clamped extreme values >{self.sentinel_thr} mm/day in generated physical samples for date {dates[0]}")
                except Exception as e:
                    logger.warning(f"[generation] Extreme sentinel/clamp check failed: {e}")
            def _cast_phys(x):
                if x is None or not isinstance(x, torch.Tensor):
                    return x
                if self.gen_config.physical_dtype == "float16":
                    return x.half()
                return x.float()
            gen_phys = _cast_phys(gen_phys)
            pmm_phys = _cast_phys(pmm_phys)
            hr_phys = _cast_phys(hr_phys)

            # Save (model space) if requested or for backward compatibility in legacy dirs
            if save:
                if self.gen_config.save_space in ('model', 'both'):
                    _save_npz(self.out_root / 'ensembles_model' / f'{date0}.npz', ens=ens_model)  # model space 
                    _save_npz(self.out_root / 'pmm_model' / f'{date0}.npz', pmm=pmm_full)  # model space
                    logger.info("[generation] Saved ensembles_model → %s", self.out_root / 'ensembles_model' / f'{date0}.npz')
                    logger.info("[generation] Saved pmm_model → %s", self.out_root / 'pmm_model' / f'{date0}.npz')
                    # Keep legacy dirs if saving model space
                    _save_npz(self.out_root / 'ensembles' / f'{date0}.npz', ens=ens_model)  # model space 
                    _save_npz(self.out_root / 'pmm' / f'{date0}.npz', pmm=pmm_full)  # model space
                    logger.info("[generation] Saved ensembles (legacy) → %s", self.out_root / 'ensembles' / f'{date0}.npz')
                    logger.info("[generation] Saved pmm (legacy) → %s", self.out_root / 'pmm' / f'{date0}.npz')

                # Save (physical space) for evaluation
                if self.gen_config.save_space in ('physical', 'both'):
                    if gen_phys is None or pmm_phys is None:
                        logger.warning(f"[generation] Physical arrays missing; skipping saving physical space npz for date {date0}")
                    else:
                        _save_npz(self.out_root / 'ensembles_phys' / f'{date0}.npz', ens=gen_phys)  # physical space ens | pmm_model | 
                        _save_npz(self.out_root / 'pmm_phys' / f'{date0}.npz', pmm=pmm_phys)  # physical space
                        logger.info("[generation] Saved ensembles_phys → %s", self.out_root / 'ensembles_phys' / f'{date0}.npz')
                        logger.info("[generation] Saved pmm_phys → %s", self.out_root / 'pmm_phys' / f'{date0}.npz')


            # Save HR
            hr = x_hr_1.detach().cpu().float() if x_hr_1 is not None else None  
            
            # Find corresponding LR channel(s) upsampled to HR for saving.
            # The target variable may appear multiple times in cond_images (e.g., LR-scaled and HR-scaled).
            lrs_model_by_kind = {}   # {"lrspace": tensor, "hrspace": tensor}
            if cond_images_gen is not None and self._lr_target_indices:
                for idx in self._lr_target_indices:
                    if cond_images_gen.shape[1] <= idx:
                        continue
                    chan = cond_images_gen[:1, idx:idx+1, :, :].detach().cpu().float()  # [1,1,H,W]
                    # Decide which inverse to use based on configured scaling method for this channel
                    try:
                        method = self.lr_scaling_methods[idx] if idx < len(self.lr_scaling_methods) else None
                    except Exception:
                        method = None
                    # Heuristic: if method matches HR scaling for target, treat as "hrspace", else "lrspace"
                    kind = None
                    if isinstance(method, str) and self.hr_scaling_method and (self.hr_scaling_method in method):
                        kind = "hrspace"
                    elif isinstance(method, str) and ("lr" in method or "LR" in method):
                        kind = "lrspace"
                    else:
                        # fallback: first occurrence goes to lrspace if not set, second to hrspace
                        kind = "lrspace" if "lrspace" not in lrs_model_by_kind else "hrspace"
                    lrs_model_by_kind[kind] = chan
            else:
                lrs_model_by_kind = {}

            if self.gen_config.save_space in ('physical', 'both'):
                # Back-transform HR reference
                hr_phys = hr.detach().cpu().float() if hr is not None else None
                if callable(self.bt_hr) and hr_phys is not None:
                    try:
                        hr_phys = self.bt_hr(hr_phys)
                        hr_phys = _cast_phys(hr_phys)
                    except Exception as e:
                        logger.warning(f"[generation] Failed to back-transform HR reference to physical space: {e}")
                # Back-transform LR reference(s) depending on their scaling method
                lr_lrspace_phys = None
                lr_hrspace_phys = None
                if "lrspace" in lrs_model_by_kind and callable(self.bt_lr_target):
                    try:
                        lr_lrspace_phys = self.bt_lr_target(lrs_model_by_kind["lrspace"])
                        lr_lrspace_phys = _cast_phys(lr_lrspace_phys)
                    except Exception as e:
                        logger.warning(f"[generation] Failed to back-transform LR(lrspace) to physical: {e}")
                if "hrspace" in lrs_model_by_kind and callable(self.bt_hr):
                    try:
                        lr_hrspace_phys = self.bt_hr(lrs_model_by_kind["hrspace"])
                        lr_hrspace_phys = _cast_phys(lr_hrspace_phys)
                    except Exception as e:
                        logger.warning(f"[generation] Failed to back-transform LR(hrspace) to physical via HR inverse: {e}")
                # Select canonical 'lr' to keep compatibility (prefer lrspace if present)
                lr_phys = lr_lrspace_phys if lr_lrspace_phys is not None else lr_hrspace_phys
                if save:
                    _save_npz(
                        self.out_root / 'lr_hr_phys' / f'{date0}.npz',
                        hr=hr_phys,
                        lr=lr_phys,
                        lr_lrspace=lr_lrspace_phys,
                        lr_hrspace=lr_hrspace_phys
                    )
                    logger.info("[generation] Saved lr_hr_phys → %s", self.out_root / 'lr_hr_phys' / f'{date0}.npz')
            # === Optional in-memory return for quicklook ===
            if results is not None:
                # Ensure everything is CPU tensors to avoid holding GPU regs
                def _cpu(x):
                    return x.detach().cpu() if (x is not None and torch.is_tensor(x)) else x
                results.append({
                    "date": date0,
                    # model-space (compact and deterministic; physical below)
                    "ensemble_model": _cpu(ens_model),  # [M,1,H,W]
                    "pmm_model": _cpu(pmm_full),        # [1,1,H,W]
                    "hr_model": _cpu(hr),               # [1,1,H,W] or None
                    "lr_model": _cpu(lrs_model_by_kind.get("lrspace", lrs_model_by_kind.get("hrspace"))),  # [1,1,H,W] or None
                    "lr_model_lrspace": _cpu(lrs_model_by_kind.get("lrspace")),   # [1,1,H,W] or None
                    "lr_model_hrspace": _cpu(lrs_model_by_kind.get("hrspace")),   # [1,1,H,W] or None
                    # physical-space (may be large and non-deterministic due to back-transform)
                    "ensemble_phys": _cpu(gen_phys),    # [M,1,H,W]
                    "pmm_phys": _cpu(pmm_phys),         # [1,1,H,W]
                    "hr_phys": _cpu(hr_phys),           # [1,1,H,W] or None
                    "lr_phys": _cpu(lr_phys),           # [1, 1,H,W] or None
                })
                logger.info("[generation] Appended quicklook payload for %s", date0)

            n_days += 1
            # Stop early if a cap is configured
            if self.gen_config.max_dates is not None and self.gen_config.max_dates > 0 and n_days >= self.gen_config.max_dates:
                logger.info(f"[generation] Reached max_dates={self.gen_config.max_dates}, stopping generation early.")
                break

        # Manifest for reproducibility
        manifest = {
            'model': model_name,
            'ensemble_size': M,
            'sampler_kind': self._sampler_kind,
            'sampler_steps': steps,
            'seed': int(self.cfg.get('evaluation', {}).get('seed', self.gen_config.seed)),
            'n_days': n_days,
            'save_space': self.gen_config.save_space,
            'physical_dtype': self.gen_config.physical_dtype,
            'stationary_cutout_cfg': bool(self.stationary_cutout),
            'lsm_stationary_observed': bool(self._lsm_stationary_ok and (self._first_lsm is not None)),
        }
        if save:
            (self.out_root / 'meta').mkdir(parents=True, exist_ok=True)
            (self.out_root / 'meta' / 'manifest.json').write_text(json.dumps(manifest, indent=2))
            logger.info("[generation] Manifest path → %s", self.out_root / 'meta' / 'manifest.json')
        else:
            logger.info("[generation] save=False (quicklook) → skipping manifest write")
        logger.info(f"[generation] Done. Wrote {n_days} days to {self.out_root}. Manifest: {manifest}")


        if output_results:
            logger.info("[generation] Returning in-memory results for %d day(s)", len(results) if results is not None else 0)
            return {
                "manifest": manifest,
                "out_root": str(self.out_root),
                "results": results
            }
