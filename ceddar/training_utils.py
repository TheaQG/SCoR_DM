import os
import torch
import torch.nn as nn 
import zarr
import logging

import numpy as np


from torch.utils.data import DataLoader, Subset, SequentialSampler
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from functools import partial

from data_modules import DANRA_Dataset_cutouts_ERA5_Zarr
from score_unet import ScoreNet, Encoder, Decoder, EDMPrecondUNet, marginal_prob_std
from losses import EDMLoss, DSMLoss
from utils import build_data_path, get_model_string
from variable_utils import get_units
from special_transforms import build_back_transforms_from_stats
# from evaluation.evaluation import evaluate_model




# # Set up logging
logger = logging.getLogger(__name__)

# Deterministic seeding for DataLoader workers (generation
_def_base_seed = 1234

def _worker_init_fn(worker_id):
    import random as _random
    import numpy as _np
    seed = _def_base_seed + worker_id
    _random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    

def _get(cfg, path, default=None):
    """
        Safe nested get: path like 'a.b.c'
    """
    node = cfg
    for k in path.split('.'):
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node

def get_loss_fn(cfg, marginal_prob_std_fn_in=None):
    edm_cfg = cfg.get('edm', {})

    if bool(edm_cfg.get('enabled', False)):
        # === EDM branch ===
        P_mean          = float(edm_cfg.get('P_mean', -1.2)) # NVLabs defaults
        P_std           = float(edm_cfg.get('P_std', 1.2))
        sigma_data      = float(edm_cfg.get('sigma_data', 1.0)) # MUST match model preconditioning

        use_sdf         = bool(_get(cfg, 'stationary_conditions.geographic_conditions.sample_w_sdf', False))
        max_land_w      = float(_get(cfg, 'stationary_conditions.geographic_conditions.max_land_weight', 1.0))
        min_sea_w       = float(_get(cfg, 'stationary_conditions.geographic_conditions.min_sea_weight', 0.5))


        return EDMLoss(
                P_mean=P_mean,
                P_std=P_std,
                sigma_data=sigma_data,
                use_sdf_weight=use_sdf,
                max_land_weight=max_land_w,
                min_sea_weight=min_sea_w)

    # === DSM default branch ===
    ve_cfg = cfg.get('ve_dsm', {})
    t_eps = float(ve_cfg.get('t_eps', 1e-3))
    mprob = marginal_prob_std_fn_in

    if mprob is None:
        raise ValueError("marginal_prob_std_fn must be provided for VE-DSM loss.")
    
    use_sdf = bool(_get(cfg, 'stationary_conditions.geographic_conditions.sample_w_sdf', True))
    max_land_w = float(_get(cfg, 'stationary_conditions.geographic_conditions.max_land_weight', 1.0))
    min_sea_w = float(_get(cfg, 'stationary_conditions.geographic_conditions.min_sea_weight', 0.5))
    return DSMLoss(
                marginal_prob_std_fn=mprob,
                t_eps=t_eps,
                use_sdf_weight=use_sdf,
                max_land_weight=max_land_w,
                min_sea_weight=min_sea_w)

def get_dataloader(cfg, verbose=True):
    '''
        Get the dataloader for training and validation datasets based on the configuration.
        Args:
            cfg (dict): Configuration dictionary containing data settings.
            verbose (bool): If True, print detailed information about the data types and sizes.
        Returns:
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
            gen_loader (DataLoader): DataLoader for the generation dataset.
    '''
    # Print information about data types
    hr_unit, lr_units = get_units(cfg)
    logger.info(f"\nUsing HR data type: {cfg['highres']['model']} {cfg['highres']['variable']} [{hr_unit}]")

    for i, cond in enumerate(cfg['lowres']['condition_variables']):
        logger.info(f"Using LR data type {i+1}: {cfg['lowres']['model']} {cond} [{lr_units[i]}]")

    # Set image dimensions based on config (if None, use default values)
    hr_data_size = tuple(cfg['highres']['data_size']) if cfg['highres']['data_size'] is not None else None
    if hr_data_size is None:
        hr_data_size = (128, 128)

    lr_data_size = tuple(cfg['lowres']['data_size']) if cfg['lowres']['data_size'] is not None else None    
    if lr_data_size is None:
        lr_data_size_use = hr_data_size
    else:
        lr_data_size_use = lr_data_size

    # Check if resize factor is set and print sizes (if verbose)
    if cfg['lowres']['resize_factor'] > 1:
        hr_data_size_use = (hr_data_size[0] // cfg['lowres']['resize_factor'], hr_data_size[1] // cfg['lowres']['resize_factor'])
        lr_data_size_use = (lr_data_size_use[0] // cfg['lowres']['resize_factor'], lr_data_size_use[1] // cfg['lowres']['resize_factor'])
    else:
        hr_data_size_use = hr_data_size
    if verbose:
        logger.info(f"\n\nHigh-resolution data size: {hr_data_size_use}")
        if cfg['lowres']['resize_factor'] > 1:
            logger.info(f"\tHigh-resolution data size after resize: {hr_data_size_use}")
        logger.info(f"Low-resolution data size: {lr_data_size_use}")
        if cfg['lowres']['resize_factor'] > 1:
            logger.info(f"\tLow-resolution data size after resize: {lr_data_size_use}")

    # Set full domain size 
    full_domain_dims = tuple(cfg['highres']['full_domain_dims']) if cfg['highres']['full_domain_dims'] is not None else None


    # Use helper functions to create the path for the zarr files
    hr_data_dir_train = build_data_path(cfg['paths']['data_dir'], cfg['highres']['model'], cfg['highres']['variable'], full_domain_dims, 'train')
    hr_data_dir_valid = build_data_path(cfg['paths']['data_dir'], cfg['highres']['model'], cfg['highres']['variable'], full_domain_dims, 'valid')
    hr_data_dir_gen = build_data_path(cfg['paths']['data_dir'], cfg['highres']['model'], cfg['highres']['variable'], full_domain_dims, 'test')
    
    # Loop over lr_vars and create paths for low-resolution data
    lr_cond_dirs_train = {}
    lr_cond_dirs_valid = {}
    lr_cond_dirs_gen = {}

    for i, cond in enumerate(cfg['lowres']['condition_variables']):
        lr_cond_dirs_train[cond] = build_data_path(cfg['paths']['data_dir'], cfg['lowres']['model'], cond, full_domain_dims, 'train')
        lr_cond_dirs_valid[cond] = build_data_path(cfg['paths']['data_dir'], cfg['lowres']['model'], cond, full_domain_dims, 'valid')
        lr_cond_dirs_gen[cond] = build_data_path(cfg['paths']['data_dir'], cfg['lowres']['model'], cond, full_domain_dims, 'test')
    
    # Set scaling and matching
    full_domain_dims_str_hr = f"{full_domain_dims[0]}x{full_domain_dims[1]}" if full_domain_dims is not None else "full_domain"
    full_domain_dims_str_lr = f"{full_domain_dims[0]}x{full_domain_dims[1]}" if full_domain_dims is not None else "full_domain"
    crop_region_hr = cfg['highres']['cutout_domains'] if cfg['highres']['cutout_domains'] is not None else "full_region"
    crop_region_hr_str = '_'.join(map(str, crop_region_hr)) #if isinstance(crop_region_hr, (list, tuple)) else crop_region_hr
    crop_region_lr = cfg['lowres']['cutout_domains'] if cfg['lowres']['cutout_domains'] is not None else "full_region"
    crop_region_lr_str = '_'.join(map(str, crop_region_lr)) #if isinstance(crop_region_lr, (list, tuple)) else crop_region_lr

    # NOTE: Maybe remove? Should be handled in dataset class
    back_transforms = build_back_transforms_from_stats(
                        hr_var              = cfg['highres']['variable'],
                        hr_model            = cfg['highres']['model'],
                        domain_str_hr       = full_domain_dims_str_hr,
                        crop_region_str_hr  = crop_region_hr_str,
                        hr_scaling_method   = cfg['highres']['scaling_method'],
                        hr_buffer_frac      = cfg['highres']['buffer_frac'] if 'buffer_frac' in cfg['highres'] else 0.0,
                        lr_vars             = cfg['lowres']['condition_variables'],
                        lr_model            = cfg['lowres']['model'],
                        domain_str_lr       = full_domain_dims_str_lr,
                        crop_region_str_lr  = crop_region_lr_str,
                        lr_scaling_methods  = cfg['lowres']['scaling_methods'],
                        lr_buffer_frac      = cfg['lowres']['buffer_frac'] if 'buffer_frac' in cfg['lowres'] else 0.0,
                        split               = cfg['transforms']['scaling_split'] if 'scaling_split' in cfg['transforms'] else 'train',
                        stats_dir_root      = cfg['paths']['stats_load_dir'],
                        eps                 = cfg['transforms'].get('prcp_eps', 0.01)
                        )

    if cfg['stationary_conditions']['geographic_conditions']['sample_w_sdf']:
        logger.info('SDF weighted loss enabled. Setting lsm and topo to true.\n')
        sample_w_geo = True
    else:
        sample_w_geo = cfg['stationary_conditions']['geographic_conditions']['sample_w_geo']

    if sample_w_geo:
        logger.info('Using geographical features for sampling.\n')
        
        geo_variables = cfg['stationary_conditions']['geographic_conditions']['geo_variables']
        data_dir_lsm = cfg['paths']['lsm_path']
        data_dir_topo = cfg['paths']['topo_path']

        data_lsm = np.flipud(np.load(data_dir_lsm)['data'])
        data_topo = np.flipud(np.load(data_dir_topo)['data'])

        if cfg['transforms']['scaling']:
            if cfg['stationary_conditions']['geographic_conditions']['topo_min'] is None or cfg['stationary_conditions']['geographic_conditions']['topo_max'] is None:
                topo_min, topo_max = np.min(data_topo), np.max(data_topo)
            else:
                topo_min = cfg['stationary_conditions']['geographic_conditions']['topo_min']
                topo_max = cfg['stationary_conditions']['geographic_conditions']['topo_max']
            if cfg['stationary_conditions']['geographic_conditions']['norm_min'] is None or cfg['stationary_conditions']['geographic_conditions']['norm_max'] is None:
                norm_min, norm_max = np.min(data_lsm), np.max(data_lsm)
            else:
                norm_min = cfg['stationary_conditions']['geographic_conditions']['norm_min']
                norm_max = cfg['stationary_conditions']['geographic_conditions']['norm_max']
            OldRange = (topo_max - topo_min)
            NewRange = (norm_max - norm_min)
            data_topo = ((data_topo - topo_min) * NewRange / OldRange) + norm_min
    else: 
        geo_variables = None
        data_lsm = None
        data_topo = None

    # Setup cutouts. If cutout domains None, use default (170, 350, 340, 520) (DK area with room for shuffle)
    cutout_domains = tuple(cfg['highres']['cutout_domains']) if cfg['highres']['cutout_domains'] is not None else (170, 350, 340, 520)
    lr_cutout_domains = tuple(cfg['lowres']['cutout_domains']) if cfg['lowres']['cutout_domains'] is not None else (170, 350, 340, 520)

    # --- Stationary cutout geometry for TRAIN/VAL ---
    # Match YAML: highres.stationary_cutout / lowres.stationary_cutout use "enabled" + "bounds"
    highres_stationary_cfg = cfg['highres'].get('stationary_cutout', {}) or {}
    stationary_cutout_hr = bool(highres_stationary_cfg.get('enabled', False))
    hr_bounds = highres_stationary_cfg.get('bounds', None)

    lowres_stationary_cfg = cfg['lowres'].get('stationary_cutout', {}) or {}
    stationary_cutout_lr = bool(lowres_stationary_cfg.get('enabled', False))
    lr_bounds = lowres_stationary_cfg.get('bounds', None)

    # --- Stationary cutout geometry for GENERATION/EVALUATION ---
    # 1) Prefer evaluation.stationary_cutout
    eval_stationary_cfg = cfg.get('evaluation', {}).get('stationary_cutout', {}) or {}
    stationary_cutout_gen_hr = bool(eval_stationary_cfg.get('hr_enabled', stationary_cutout_hr))
    stationary_cutout_gen_lr = bool(eval_stationary_cfg.get('lr_enabled', stationary_cutout_lr))
    hr_bounds_gen = eval_stationary_cfg.get('hr_bounds', None)
    lr_bounds_gen = eval_stationary_cfg.get('lr_bounds', None)

    # 2) If not set there, fall back to full_gen_eval.stationary_cutout (for new full evaluation driver)
    fg_cfg = cfg.get('full_gen_eval', {}) or {}
    fg_stationary = fg_cfg.get('stationary_cutout', {}) or {}
    if hr_bounds_gen is None:
        hr_bounds_gen = fg_stationary.get('hr_bounds', None)
    if lr_bounds_gen is None:
        lr_bounds_gen = fg_stationary.get('lr_bounds', None)

    # 3) Finally, fall back to training geometry if still None
    if hr_bounds_gen is None:
        hr_bounds_gen = hr_bounds
    if lr_bounds_gen is None:
        lr_bounds_gen = lr_bounds

    # Setup conditional seasons (classification)
    if cfg['stationary_conditions']['seasonal_conditions']['sample_w_cond_season']:
        n_seasons = cfg['stationary_conditions']['seasonal_conditions']['n_seasons']
    else:
        n_seasons = None


    # Make zarr groups
    data_train_zarr = zarr.open_group(hr_data_dir_train, mode='r')
    data_valid_zarr = zarr.open_group(hr_data_dir_valid, mode='r')
    data_gen_zarr = zarr.open_group(hr_data_dir_gen, mode='r')

    n_samples_train = len(list(data_train_zarr.keys()))
    n_samples_valid = len(list(data_valid_zarr.keys()))
    n_samples_gen = len(list(data_gen_zarr.keys()))

    # Setup cache

    if cfg['data_handling']['cache_size'] == 0:
        cache_size_train = n_samples_train//2
        cache_size_valid = n_samples_valid//2
    else:
        cache_size_train = cfg['data_handling']['cache_size']
        cache_size_valid = cfg['data_handling']['cache_size']

    if verbose:
        logger.info(f"\n\n\nNumber of training samples: {n_samples_train}")
        logger.info(f"Number of validation samples: {n_samples_valid}")
        logger.info(f"Cache size for training: {cache_size_train}")
        logger.info(f"Cache size for validation: {cache_size_valid}\n\n\n")


    # Setup datasets

    train_dataset = DANRA_Dataset_cutouts_ERA5_Zarr(
                            hr_variable_dir_zarr=hr_data_dir_train,
                            hr_data_size=hr_data_size_use,
                            n_samples=n_samples_train,
                            cache_size=cache_size_train,
                            hr_variable=cfg['highres']['variable'],
                            hr_model=cfg['highres']['model'],
                            hr_scaling_method=cfg['highres']['scaling_method'],
                            # hr_scaling_params=cfg['highres']['scaling_params'],
                            lr_conditions=cfg['lowres']['condition_variables'],
                            lr_model=cfg['lowres']['model'],
                            lr_scaling_methods=cfg['lowres']['scaling_methods'],
                            # lr_scaling_params=cfg['lowres']['scaling_params'],
                            lr_cond_dirs_zarr=lr_cond_dirs_train,
                            geo_variables=geo_variables,
                            lsm_full_domain=data_lsm,
                            topo_full_domain=data_topo,
                            conditional_seasons=cfg['stationary_conditions']['seasonal_conditions']['sample_w_cond_season'],
                            use_sin_cos_embedding=cfg['stationary_conditions']['seasonal_conditions'].get('use_sin_cos_embedding', False),
                            use_leap_years=cfg['stationary_conditions']['seasonal_conditions'].get('use_leap_years', True),
                            cfg = cfg,
                            split = "train",
                            shuffle=True,
                            cutouts=cfg['transforms']['sample_w_cutouts'],
                            cutout_domains=list(cutout_domains) if cfg['transforms']['sample_w_cutouts'] else None,
                            n_samples_w_cutouts=n_samples_train,
                            sdf_weighted_loss=cfg['stationary_conditions']['geographic_conditions']['sample_w_sdf'],
                            scale=cfg['transforms']['scaling'],
                            save_original=cfg['visualization']['show_both_orig_scaled'],
                            n_classes=n_seasons,
                            lr_data_size=tuple(lr_data_size_use) if lr_data_size_use is not None else None,
                            lr_cutout_domains=list(lr_cutout_domains) if lr_cutout_domains is not None else None,
                            resize_factor=cfg['lowres']['resize_factor'],
                            fixed_cutout_hr=stationary_cutout_hr,
                            fixed_hr_bounds=hr_bounds,
                            fixed_cutout_lr=stationary_cutout_lr,
                            fixed_lr_bounds=lr_bounds,
    )

    val_dataset = DANRA_Dataset_cutouts_ERA5_Zarr(
                            hr_variable_dir_zarr=hr_data_dir_valid,
                            hr_data_size=hr_data_size_use,
                            n_samples=n_samples_valid,
                            cache_size=cache_size_valid,
                            hr_variable=cfg['highres']['variable'],
                            hr_model=cfg['highres']['model'],
                            hr_scaling_method=cfg['highres']['scaling_method'],
                            # hr_scaling_params=cfg['highres']['scaling_params'],
                            lr_conditions=cfg['lowres']['condition_variables'],
                            lr_model=cfg['lowres']['model'],
                            lr_scaling_methods=cfg['lowres']['scaling_methods'],
                            # lr_scaling_params=cfg['lowres']['scaling_params'],
                            lr_cond_dirs_zarr=lr_cond_dirs_valid,
                            geo_variables=geo_variables,
                            lsm_full_domain=data_lsm,
                            topo_full_domain=data_topo,
                            conditional_seasons=cfg['stationary_conditions']['seasonal_conditions']['sample_w_cond_season'],
                            use_sin_cos_embedding=cfg['stationary_conditions']['seasonal_conditions'].get('use_sin_cos_embedding', False),
                            use_leap_years=cfg['stationary_conditions']['seasonal_conditions'].get('use_leap_years', True),
                            cfg = cfg,
                            split = "valid",
                            shuffle=True,
                            cutouts=cfg['transforms']['sample_w_cutouts'],
                            cutout_domains=list(cutout_domains) if cfg['transforms']['sample_w_cutouts'] else None,
                            n_samples_w_cutouts=n_samples_valid,
                            sdf_weighted_loss=cfg['stationary_conditions']['geographic_conditions']['sample_w_sdf'],
                            scale=cfg['transforms']['scaling'],
                            save_original=cfg['visualization']['show_both_orig_scaled'],
                            n_classes=n_seasons,
                            lr_data_size=tuple(lr_data_size_use) if lr_data_size_use is not None else None,
                            lr_cutout_domains=list(lr_cutout_domains) if lr_cutout_domains is not None else None,
                            resize_factor=cfg['lowres']['resize_factor'],
                            fixed_cutout_hr=stationary_cutout_hr,
                            fixed_hr_bounds=hr_bounds,
                            fixed_cutout_lr=stationary_cutout_lr,
                            fixed_lr_bounds=lr_bounds,
    )

    gen_dataset = DANRA_Dataset_cutouts_ERA5_Zarr(
                            hr_variable_dir_zarr=hr_data_dir_gen,
                            hr_data_size=hr_data_size_use,
                            n_samples=n_samples_gen,
                            cache_size=cfg['data_handling']['cache_size'],
                            hr_variable=cfg['highres']['variable'],
                            hr_model=cfg['highres']['model'],
                            hr_scaling_method=cfg['highres']['scaling_method'],
                            # hr_scaling_params=cfg['highres']['scaling_params'],
                            lr_conditions=cfg['lowres']['condition_variables'],
                            lr_model=cfg['lowres']['model'],
                            lr_scaling_methods=cfg['lowres']['scaling_methods'],
                            # lr_scaling_params=cfg['lowres']['scaling_params'],
                            lr_cond_dirs_zarr=lr_cond_dirs_gen,
                            geo_variables=geo_variables,
                            lsm_full_domain=data_lsm,
                            topo_full_domain=data_topo,
                            conditional_seasons=cfg['stationary_conditions']['seasonal_conditions']['sample_w_cond_season'],
                            use_sin_cos_embedding=cfg['stationary_conditions']['seasonal_conditions'].get('use_sin_cos_embedding', False),
                            use_leap_years=cfg['stationary_conditions']['seasonal_conditions'].get('use_leap_years', True),                            
                            cfg = cfg,
                            split = "gen",
                            shuffle=False,
                            cutouts=cfg['transforms']['sample_w_cutouts'],
                            cutout_domains=list(cutout_domains) if cfg['transforms']['sample_w_cutouts'] else None,
                            n_samples_w_cutouts=n_samples_gen,
                            sdf_weighted_loss=cfg['stationary_conditions']['geographic_conditions']['sample_w_sdf'],
                            scale=cfg['transforms']['scaling'],
                            save_original=cfg['visualization']['show_both_orig_scaled'],
                            n_classes=n_seasons,
                            lr_data_size=tuple(lr_data_size_use) if lr_data_size_use is not None else None,
                            lr_cutout_domains=list(lr_cutout_domains) if lr_cutout_domains is not None else None,
                            resize_factor=cfg['lowres']['resize_factor'],
                            fixed_cutout_hr=stationary_cutout_gen_hr,
                            fixed_hr_bounds=hr_bounds_gen,
                            fixed_cutout_lr=stationary_cutout_gen_lr,
                            fixed_lr_bounds=lr_bounds_gen,
    )

    logger.info(
        "[seasonal] conditional=%s, sin/cos=%s, leap_years=%s\n",
        cfg['stationary_conditions']['seasonal_conditions']['sample_w_cond_season'],
        cfg['stationary_conditions']['seasonal_conditions'].get('use_sin_cos_embedding', False),
        cfg['stationary_conditions']['seasonal_conditions'].get('use_leap_years', True),
    )

    # Setup dataloaders
    raw_workers = int(cfg['data_handling'].get('num_workers', 0) or 0)
    pin = bool(cfg['data_handling'].get('pin_memory', torch.cuda.is_available())) and torch.cuda.is_available()
    persist = raw_workers > 0

    train_kwargs = dict(
        batch_size=int(cfg['training']['batch_size']),
        shuffle=True,
        num_workers=int(raw_workers),
        pin_memory=bool(pin),
        persistent_workers=bool(persist),
        drop_last=True)
    
    if persist:
        train_kwargs['prefetch_factor'] = 4  # Each worker preloads 4 batches

    train_loader = DataLoader(train_dataset, **train_kwargs) # type: ignore


    val_kwargs = dict(
        batch_size=int(cfg['training']['batch_size']),
        shuffle=False,
        num_workers=int(raw_workers),
        pin_memory=bool(pin),
        persistent_workers=bool(persist),
        drop_last=(len(val_dataset) % cfg['training']['batch_size']) != 0)
    
    if persist:
        val_kwargs['prefetch_factor'] = 2  # Each worker preloads 2 batches
    val_loader = DataLoader(val_dataset, **val_kwargs) # type: ignore


    gen_bs = int(cfg['data_handling']['n_gen_samples'])
    # Take the first gen_bs samples deterministically
    fixed_ids = list(range(min(gen_bs, len(gen_dataset))))
    gen_subset = Subset(gen_dataset, fixed_ids)

    base_seed = int(cfg['evaluation'].get('seed', _def_base_seed))
    g = torch.Generator()
    g.manual_seed(base_seed)
    gen_loader = DataLoader(
        gen_subset,
        batch_size              = gen_bs,
        shuffle                 = False,
        sampler                 = SequentialSampler(gen_subset),
        num_workers             = 0, #max(2, num_workers // 4),
        worker_init_fn          = _worker_init_fn,
        generator               = g,
        drop_last               = False,
        )


    # Print dataset information
    # if verbose:
    logger.info(f"\nTraining dataset: {len(train_dataset)} samples")
    logger.info(f"Validation dataset: {len(val_dataset)} samples")
    logger.info(f"Generation dataset: {len(gen_dataset)} samples\n")
    logger.info(f"Batch size: {cfg['training']['batch_size']}")
    logger.info(f"Number of workers: {int(cfg['data_handling']['num_workers'])}\n")
    
    # Return the dataloaders
    return train_loader, val_loader, gen_loader


def get_gen_dataloader(cfg, verbose=True):
    '''
        Get the dataloader for training and validation datasets based on the configuration.
        Args:
            cfg (dict): Configuration dictionary containing data settings.
            verbose (bool): If True, print detailed information about the data types and sizes.
        Returns:
            gen_loader (DataLoader): DataLoader for the generation dataset.
    '''
    # Print information about data types
    hr_unit, lr_units = get_units(cfg)
    logger.info(f"\nUsing HR data type: {cfg['highres']['model']} {cfg['highres']['variable']} [{hr_unit}]")

    for i, cond in enumerate(cfg['lowres']['condition_variables']):
        logger.info(f"Using LR data type {i+1}: {cfg['lowres']['model']} {cond} [{lr_units[i]}]")

    # Set image dimensions based on config (if None, use default values)
    hr_data_size = tuple(cfg['highres']['data_size']) if cfg['highres']['data_size'] is not None else None
    if hr_data_size is None:
        hr_data_size = (128, 128)

    lr_data_size = tuple(cfg['lowres']['data_size']) if cfg['lowres']['data_size'] is not None else None    
    if lr_data_size is None:
        lr_data_size_use = hr_data_size
    else:
        lr_data_size_use = lr_data_size

    # Check if resize factor is set and print sizes (if verbose)
    if cfg['lowres']['resize_factor'] > 1:
        hr_data_size_use = (hr_data_size[0] // cfg['lowres']['resize_factor'], hr_data_size[1] // cfg['lowres']['resize_factor'])
        lr_data_size_use = (lr_data_size_use[0] // cfg['lowres']['resize_factor'], lr_data_size_use[1] // cfg['lowres']['resize_factor'])
    else:
        hr_data_size_use = hr_data_size
        lr_data_size_use = lr_data_size_use
    if verbose:
        logger.info(f"\n\nHigh-resolution data size: {hr_data_size_use}")
        if cfg['lowres']['resize_factor'] > 1:
            logger.info(f"\tHigh-resolution data size after resize: {hr_data_size_use}")
        logger.info(f"Low-resolution data size: {lr_data_size_use}")
        if cfg['lowres']['resize_factor'] > 1:
            logger.info(f"\tLow-resolution data size after resize: {lr_data_size_use}")

    # Set full domain size 
    full_domain_dims = tuple(cfg['highres']['full_domain_dims']) if cfg['highres']['full_domain_dims'] is not None else None


    # Use helper functions to create the path for the zarr files
    hr_data_dir_train = build_data_path(cfg['paths']['data_dir'], cfg['highres']['model'], cfg['highres']['variable'], full_domain_dims, 'train')
    hr_data_dir_valid = build_data_path(cfg['paths']['data_dir'], cfg['highres']['model'], cfg['highres']['variable'], full_domain_dims, 'valid')
    hr_data_dir_gen = build_data_path(cfg['paths']['data_dir'], cfg['highres']['model'], cfg['highres']['variable'], full_domain_dims, 'test')
    
    # Loop over lr_vars and create paths for low-resolution data
    lr_cond_dirs_train = {}
    lr_cond_dirs_valid = {}
    lr_cond_dirs_gen = {}

    for i, cond in enumerate(cfg['lowres']['condition_variables']):
        lr_cond_dirs_train[cond] = build_data_path(cfg['paths']['data_dir'], cfg['lowres']['model'], cond, full_domain_dims, 'train')
        lr_cond_dirs_valid[cond] = build_data_path(cfg['paths']['data_dir'], cfg['lowres']['model'], cond, full_domain_dims, 'valid')
        lr_cond_dirs_gen[cond] = build_data_path(cfg['paths']['data_dir'], cfg['lowres']['model'], cond, full_domain_dims, 'test')

    # Set scaling and matching
    full_domain_dims_str_hr = f"{full_domain_dims[0]}x{full_domain_dims[1]}" if full_domain_dims is not None else "full_domain"
    full_domain_dims_str_lr = f"{full_domain_dims[0]}x{full_domain_dims[1]}" if full_domain_dims is not None else "full_domain"
    crop_region_hr = cfg['highres']['cutout_domains'] if cfg['highres']['cutout_domains'] is not None else "full_region"
    crop_region_hr_str = '_'.join(map(str, crop_region_hr)) #if isinstance(crop_region_hr, (list, tuple)) else crop_region_hr
    crop_region_lr = cfg['lowres']['cutout_domains'] if cfg['lowres']['cutout_domains'] is not None else "full_region"
    crop_region_lr_str = '_'.join(map(str, crop_region_lr)) #if isinstance(crop_region_lr, (list, tuple)) else crop_region_lr

    # NOTE: Maybe remove? Should be handled in dataset class
    back_transforms = build_back_transforms_from_stats(
                        hr_var              = cfg['highres']['variable'],
                        hr_model            = cfg['highres']['model'],
                        domain_str_hr       = full_domain_dims_str_hr,
                        crop_region_str_hr  = crop_region_hr_str,
                        hr_scaling_method   = cfg['highres']['scaling_method'],
                        hr_buffer_frac      = cfg['highres']['buffer_frac'] if 'buffer_frac' in cfg['highres'] else 0.0,
                        lr_vars             = cfg['lowres']['condition_variables'],
                        lr_model            = cfg['lowres']['model'],
                        domain_str_lr       = full_domain_dims_str_lr,
                        crop_region_str_lr  = crop_region_lr_str,
                        lr_scaling_methods  = cfg['lowres']['scaling_methods'],
                        lr_buffer_frac      = cfg['lowres']['buffer_frac'] if 'buffer_frac' in cfg['lowres'] else 0.0,
                        split               = cfg['transforms']['scaling_split'] if 'scaling_split' in cfg['transforms'] else 'train',
                        stats_dir_root      = cfg['paths']['stats_load_dir'],
                        eps                 = cfg['transforms'].get('prcp_eps', 0.01)
                        )

    if cfg['stationary_conditions']['geographic_conditions']['sample_w_sdf']:
        logger.info('SDF weighted loss enabled. Setting lsm and topo to true.\n')
        sample_w_geo = True
    else:
        sample_w_geo = cfg['stationary_conditions']['geographic_conditions']['sample_w_geo']

    if sample_w_geo:
        logger.info('Using geographical features for sampling.\n')
        
        geo_variables = cfg['stationary_conditions']['geographic_conditions']['geo_variables']
        data_dir_lsm = cfg['paths']['lsm_path']
        data_dir_topo = cfg['paths']['topo_path']

        data_lsm = np.flipud(np.load(data_dir_lsm)['data'])
        data_topo = np.flipud(np.load(data_dir_topo)['data'])

        if cfg['transforms']['scaling']:
            if cfg['stationary_conditions']['geographic_conditions']['topo_min'] is None or cfg['stationary_conditions']['geographic_conditions']['topo_max'] is None:
                topo_min, topo_max = np.min(data_topo), np.max(data_topo)
            else:
                topo_min = cfg['stationary_conditions']['geographic_conditions']['topo_min']
                topo_max = cfg['stationary_conditions']['geographic_conditions']['topo_max']
            if cfg['stationary_conditions']['geographic_conditions']['norm_min'] is None or cfg['stationary_conditions']['geographic_conditions']['norm_max'] is None:
                norm_min, norm_max = np.min(data_lsm), np.max(data_lsm)
            else:
                norm_min = cfg['stationary_conditions']['geographic_conditions']['norm_min']
                norm_max = cfg['stationary_conditions']['geographic_conditions']['norm_max']
            OldRange = (topo_max - topo_min)
            NewRange = (norm_max - norm_min)
            data_topo = ((data_topo - topo_min) * NewRange / OldRange) + norm_min
    else: 
        geo_variables = None
        data_lsm = None
        data_topo = None

    # Setup cutouts. If cutout domains None, use default (170, 350, 340, 520) (DK area with room for shuffle)
    cutout_domains = tuple(cfg['highres']['cutout_domains']) if cfg['highres']['cutout_domains'] is not None else (170, 350, 340, 520)
    lr_cutout_domains = tuple(cfg['lowres']['cutout_domains']) if cfg['lowres']['cutout_domains'] is not None else (170, 350, 340, 520)

    stationary_cutout_gen_hr = bool(cfg['evaluation'].get('stationary_cutout', {}).get('hr_enabled', False))
    hr_bounds_gen = cfg['evaluation'].get('stationary_cutout', {}).get('hr_bounds', None)
    stationary_cutout_gen_lr = bool(cfg['evaluation'].get('stationary_cutout', {}).get('lr_enabled', False))
    lr_bounds_gen = cfg['evaluation'].get('stationary_cutout', {}).get('lr_bounds', None)

    # Setup conditional seasons (classification)
    if cfg['stationary_conditions']['seasonal_conditions']['sample_w_cond_season']:
        n_seasons = cfg['stationary_conditions']['seasonal_conditions']['n_seasons']
    else:
        n_seasons = None


    # Make zarr groups
    data_gen_zarr = zarr.open_group(hr_data_dir_gen, mode='r')

    n_samples_gen = len(list(data_gen_zarr.keys()))

    # Setup dataset
    gen_dataset = DANRA_Dataset_cutouts_ERA5_Zarr(
                            hr_variable_dir_zarr=hr_data_dir_gen,
                            hr_data_size=hr_data_size_use,
                            n_samples=n_samples_gen,
                            cache_size=cfg['data_handling']['cache_size'],
                            hr_variable=cfg['highres']['variable'],
                            hr_model=cfg['highres']['model'],
                            hr_scaling_method=cfg['highres']['scaling_method'],
                            # hr_scaling_params=cfg['highres']['scaling_params'],
                            lr_conditions=cfg['lowres']['condition_variables'],
                            lr_model=cfg['lowres']['model'],
                            lr_scaling_methods=cfg['lowres']['scaling_methods'],
                            # lr_scaling_params=cfg['lowres']['scaling_params'],
                            lr_cond_dirs_zarr=lr_cond_dirs_gen,
                            geo_variables=geo_variables,
                            lsm_full_domain=data_lsm,
                            topo_full_domain=data_topo,
                            conditional_seasons=cfg['stationary_conditions']['seasonal_conditions']['sample_w_cond_season'],
                            use_sin_cos_embedding=cfg['stationary_conditions']['seasonal_conditions'].get('use_sin_cos_embedding', False),
                            use_leap_years=cfg['stationary_conditions']['seasonal_conditions'].get('use_leap_years', True),                            
                            cfg = cfg,
                            split = "gen",
                            shuffle=False,
                            cutouts=cfg['transforms']['sample_w_cutouts'],
                            cutout_domains=list(cutout_domains) if cfg['transforms']['sample_w_cutouts'] else None,
                            n_samples_w_cutouts=n_samples_gen,
                            sdf_weighted_loss=cfg['stationary_conditions']['geographic_conditions']['sample_w_sdf'],
                            scale=cfg['transforms']['scaling'],
                            save_original=cfg['visualization']['show_both_orig_scaled'],
                            n_classes=n_seasons,
                            lr_data_size=tuple(lr_data_size_use) if lr_data_size_use is not None else None,
                            lr_cutout_domains=list(lr_cutout_domains) if lr_cutout_domains is not None else None,
                            resize_factor=cfg['lowres']['resize_factor'],
                            fixed_cutout_hr=stationary_cutout_gen_hr,
                            fixed_hr_bounds=hr_bounds_gen,
                            fixed_cutout_lr=stationary_cutout_gen_lr,
                            fixed_lr_bounds=lr_bounds_gen,
                            )
    # Setup dataloaders
    gen_bs = int(cfg['data_handling']['n_gen_samples'])
    fixed_ids = list(range(min(gen_bs, len(gen_dataset))))
    gen_subset = Subset(gen_dataset, fixed_ids)

    base_seed = int(cfg['data_handling'].get('seed', _def_base_seed))
    g = torch.Generator()
    g.manual_seed(base_seed)

    gen_loader = DataLoader(
        gen_subset,
        batch_size              = gen_bs,
        shuffle                 = False,
        sampler                 = SequentialSampler(gen_subset),
        num_workers             = 0, 
        worker_init_fn          = _worker_init_fn,
        generator               = g,
        drop_last               = False,
    )

    # Print dataset information
    # if verbose:
    logger.info(f"Generation dataset: {len(gen_dataset)} samples\n")
    
    # Return the dataloaders
    return gen_loader



def get_final_gen_dataloader(cfg, split: str = "test", verbose: bool = True):
    """
    Deterministic dataloader over the full temporal split (train/valid/test) for
    *final* generation/evaluation.

    - Uses the same dataset class as training/validation.
    - Respects stationary cutout geometry for generation/evaluation.
    - Respects the dataset's internal common-date intersection (gen_dataset.n_samples)
      instead of the raw HR count.
    - Optionally truncates via full_gen_eval.max_dates.
    """
    # --- Basic geometry (mirror get_dataloader / get_gen_dataloader) ---
    hr_unit, lr_units = get_units(cfg)
    logger.info(
        f"\n[get_final_gen_dataloader] Using HR data type: "
        f"{cfg['highres']['model']} {cfg['highres']['variable']} [{hr_unit}]"
    )
    for i, cond in enumerate(cfg['lowres']['condition_variables']):
        logger.info(
            f"[get_final_gen_dataloader] Using LR data type {i+1}: "
            f"{cfg['lowres']['model']} {cond} [{lr_units[i]}]"
        )

    # HR / LR sizes
    hr_data_size = tuple(cfg['highres']['data_size']) if cfg['highres']['data_size'] is not None else (128, 128)
    lr_data_size = tuple(cfg['lowres']['data_size']) if cfg['lowres']['data_size'] is not None else None
    lr_data_size_use = lr_data_size if lr_data_size is not None else hr_data_size

    if cfg['lowres']['resize_factor'] > 1:
        rf = cfg['lowres']['resize_factor']
        hr_data_size_use = (hr_data_size[0] // rf, hr_data_size[1] // rf)
        lr_data_size_use = (lr_data_size_use[0] // rf, lr_data_size_use[1] // rf)
    else:
        hr_data_size_use = hr_data_size

    if verbose:
        logger.info(f"[get_final_gen_dataloader] High-resolution data size: {hr_data_size_use}")
        logger.info(f"[get_final_gen_dataloader] Low-resolution data size: {lr_data_size_use}")

    # Full domain dims
    full_domain_dims = tuple(cfg['highres']['full_domain_dims']) if cfg['highres']['full_domain_dims'] is not None else None

    # Paths for each split
    hr_data_dir_train = build_data_path(cfg['paths']['data_dir'], cfg['highres']['model'],
                                        cfg['highres']['variable'], full_domain_dims, 'train')
    hr_data_dir_valid = build_data_path(cfg['paths']['data_dir'], cfg['highres']['model'],
                                        cfg['highres']['variable'], full_domain_dims, 'valid')
    hr_data_dir_gen   = build_data_path(cfg['paths']['data_dir'], cfg['highres']['model'],
                                        cfg['highres']['variable'], full_domain_dims, 'test')

    lr_cond_dirs_train = {}
    lr_cond_dirs_valid = {}
    lr_cond_dirs_gen   = {}
    for cond in cfg['lowres']['condition_variables']:
        lr_cond_dirs_train[cond] = build_data_path(cfg['paths']['data_dir'], cfg['lowres']['model'],
                                                   cond, full_domain_dims, 'train')
        lr_cond_dirs_valid[cond] = build_data_path(cfg['paths']['data_dir'], cfg['lowres']['model'],
                                                   cond, full_domain_dims, 'valid')
        lr_cond_dirs_gen[cond]   = build_data_path(cfg['paths']['data_dir'], cfg['lowres']['model'],
                                                   cond, full_domain_dims, 'test')

    # Strings for stats / back-transforms (mainly needed by dataset)
    full_domain_dims_str_hr = f"{full_domain_dims[0]}x{full_domain_dims[1]}" if full_domain_dims is not None else "full_domain"
    full_domain_dims_str_lr = f"{full_domain_dims[0]}x{full_domain_dims[1]}" if full_domain_dims is not None else "full_domain"
    crop_region_hr = cfg['highres']['cutout_domains'] if cfg['highres']['cutout_domains'] is not None else "full_region"
    crop_region_lr = cfg['lowres']['cutout_domains'] if cfg['lowres']['cutout_domains'] is not None else "full_region"
    crop_region_hr_str = '_'.join(map(str, crop_region_hr))
    crop_region_lr_str = '_'.join(map(str, crop_region_lr))

    # Back transforms (kept for completeness; dataset may use them)
    _ = build_back_transforms_from_stats(
        hr_var              = cfg['highres']['variable'],
        hr_model            = cfg['highres']['model'],
        domain_str_hr       = full_domain_dims_str_hr,
        crop_region_str_hr  = crop_region_hr_str,
        hr_scaling_method   = cfg['highres']['scaling_method'],
        hr_buffer_frac      = cfg['highres'].get('buffer_frac', 0.0),
        lr_vars             = cfg['lowres']['condition_variables'],
        lr_model            = cfg['lowres']['model'],
        domain_str_lr       = full_domain_dims_str_lr,
        crop_region_str_lr  = crop_region_lr_str,
        lr_scaling_methods  = cfg['lowres']['scaling_methods'],
        lr_buffer_frac      = cfg['lowres'].get('buffer_frac', 0.0),
        split               = cfg['transforms'].get('scaling_split', 'train'),
        stats_dir_root      = cfg['paths']['stats_load_dir'],
        eps                 = cfg['transforms'].get('prcp_eps', 0.01),
    )

    # Geo/static fields
    if cfg['stationary_conditions']['geographic_conditions']['sample_w_sdf']:
        logger.info('[get_final_gen_dataloader] SDF weighted loss enabled → forcing geo sampling.')
        sample_w_geo = True
    else:
        sample_w_geo = cfg['stationary_conditions']['geographic_conditions']['sample_w_geo']

    if sample_w_geo:
        logger.info('[get_final_gen_dataloader] Using geographical features for sampling.')
        geo_variables = cfg['stationary_conditions']['geographic_conditions']['geo_variables']
        data_dir_lsm = cfg['paths']['lsm_path']
        data_dir_topo = cfg['paths']['topo_path']

        data_lsm = np.flipud(np.load(data_dir_lsm)['data'])
        data_topo = np.flipud(np.load(data_dir_topo)['data'])

        if cfg['transforms']['scaling']:
            if (cfg['stationary_conditions']['geographic_conditions']['topo_min'] is None or
                cfg['stationary_conditions']['geographic_conditions']['topo_max'] is None):
                topo_min, topo_max = np.min(data_topo), np.max(data_topo)
            else:
                topo_min = cfg['stationary_conditions']['geographic_conditions']['topo_min']
                topo_max = cfg['stationary_conditions']['geographic_conditions']['topo_max']

            if (cfg['stationary_conditions']['geographic_conditions']['norm_min'] is None or
                cfg['stationary_conditions']['geographic_conditions']['norm_max'] is None):
                norm_min, norm_max = np.min(data_lsm), np.max(data_lsm)
            else:
                norm_min = cfg['stationary_conditions']['geographic_conditions']['norm_min']
                norm_max = cfg['stationary_conditions']['geographic_conditions']['norm_max']

            OldRange = (topo_max - topo_min)
            NewRange = (norm_max - norm_min)
            data_topo = ((data_topo - topo_min) * NewRange / OldRange) + norm_min
    else:
        geo_variables = None
        data_lsm = None
        data_topo = None

    # Cutouts
    cutout_domains    = tuple(cfg['highres']['cutout_domains']) if cfg['highres']['cutout_domains'] is not None else (170, 350, 340, 520)
    lr_cutout_domains = tuple(cfg['lowres']['cutout_domains']) if cfg['lowres']['cutout_domains'] is not None else (170, 350, 340, 520)

    # --- Stationary cutout geometry (same logic as in get_dataloader) ---
    highres_stationary_cfg = cfg['highres'].get('stationary_cutout', {}) or {}
    stationary_cutout_hr = bool(highres_stationary_cfg.get('enabled', False))
    hr_bounds = highres_stationary_cfg.get('bounds', None)

    lowres_stationary_cfg = cfg['lowres'].get('stationary_cutout', {}) or {}
    stationary_cutout_lr = bool(lowres_stationary_cfg.get('enabled', False))
    lr_bounds = lowres_stationary_cfg.get('bounds', None)

    eval_stationary_cfg = cfg.get('evaluation', {}).get('stationary_cutout', {}) or {}
    stationary_cutout_gen_hr = bool(eval_stationary_cfg.get('hr_enabled', stationary_cutout_hr))
    stationary_cutout_gen_lr = bool(eval_stationary_cfg.get('lr_enabled', stationary_cutout_lr))
    hr_bounds_gen = eval_stationary_cfg.get('hr_bounds', None)
    lr_bounds_gen = eval_stationary_cfg.get('lr_bounds', None)

    fg_cfg = cfg.get('full_gen_eval', {}) or {}
    fg_stationary = fg_cfg.get('stationary_cutout', {}) or {}
    if hr_bounds_gen is None:
        hr_bounds_gen = fg_stationary.get('hr_bounds', None)
    if lr_bounds_gen is None:
        lr_bounds_gen = fg_stationary.get('lr_bounds', None)

    if hr_bounds_gen is None:
        hr_bounds_gen = hr_bounds
    if lr_bounds_gen is None:
        lr_bounds_gen = lr_bounds

    # Seasonal conditioning
    if cfg['stationary_conditions']['seasonal_conditions']['sample_w_cond_season']:
        n_seasons = cfg['stationary_conditions']['seasonal_conditions']['n_seasons']
    else:
        n_seasons = None

    # --- Choose split-specific dirs ---
    split_norm = str(split).lower()
    if split_norm in ("train", "training"):
        hr_dir = hr_data_dir_train
        lr_cond_dirs = lr_cond_dirs_train
        ds_split = "train"
    elif split_norm in ("val", "valid", "validation"):
        hr_dir = hr_data_dir_valid
        lr_cond_dirs = lr_cond_dirs_valid
        ds_split = "valid"
    else:
        # default: test → dataset split name "gen"
        hr_dir = hr_data_dir_gen
        lr_cond_dirs = lr_cond_dirs_gen
        ds_split = "gen"

    data_zarr = zarr.open_group(hr_dir, mode='r')
    n_samples_full = len(list(data_zarr.keys()))

    # Cache size for final generation: prefer cache_size_gen, else cache_size, else 0
    cache_size_gen = cfg['data_handling'].get('cache_size_gen', None)
    if cache_size_gen is None:
        cache_size_gen = cfg['data_handling'].get('cache_size', 0)
    cache_size_gen = int(cache_size_gen)

    if verbose:
        logger.info(
            f"[get_final_gen_dataloader] Split='{split_norm}', raw HR samples={n_samples_full}, "
            f"cache_size_gen={cache_size_gen}"
        )

    # --- Build dataset for this split ---
    gen_dataset = DANRA_Dataset_cutouts_ERA5_Zarr(
        hr_variable_dir_zarr=hr_dir,
        hr_data_size=hr_data_size_use,
        n_samples=n_samples_full,
        cache_size=cache_size_gen,
        hr_variable=cfg['highres']['variable'],
        hr_model=cfg['highres']['model'],
        hr_scaling_method=cfg['highres']['scaling_method'],
        lr_conditions=cfg['lowres']['condition_variables'],
        lr_model=cfg['lowres']['model'],
        lr_scaling_methods=cfg['lowres']['scaling_methods'],
        lr_cond_dirs_zarr=lr_cond_dirs,
        geo_variables=geo_variables,
        lsm_full_domain=data_lsm,
        topo_full_domain=data_topo,
        conditional_seasons=cfg['stationary_conditions']['seasonal_conditions']['sample_w_cond_season'],
        use_sin_cos_embedding=cfg['stationary_conditions']['seasonal_conditions'].get('use_sin_cos_embedding', False),
        use_leap_years=cfg['stationary_conditions']['seasonal_conditions'].get('use_leap_years', True),
        cfg=cfg,
        split=ds_split,
        shuffle=False,
        cutouts=cfg['transforms']['sample_w_cutouts'],
        cutout_domains=list(cutout_domains) if cfg['transforms']['sample_w_cutouts'] else None,
        n_samples_w_cutouts=n_samples_full,
        sdf_weighted_loss=cfg['stationary_conditions']['geographic_conditions']['sample_w_sdf'],
        scale=cfg['transforms']['scaling'],
        save_original=cfg['visualization']['show_both_orig_scaled'],
        n_classes=n_seasons,
        lr_data_size=tuple(lr_data_size_use) if lr_data_size_use is not None else None,
        lr_cutout_domains=list(lr_cutout_domains) if lr_cutout_domains is not None else None,
        resize_factor=cfg['lowres']['resize_factor'],
        fixed_cutout_hr=stationary_cutout_gen_hr,
        fixed_hr_bounds=hr_bounds_gen,
        fixed_cutout_lr=stationary_cutout_gen_lr,
        fixed_lr_bounds=lr_bounds_gen,
    )

    # --- Respect common-date intersection + full_gen_eval.max_dates ---
    n_samples_total = len(gen_dataset)  # raw length (e.g. 1062)
    n_samples_internal = getattr(gen_dataset, "n_samples", None)  # dataset's intersection (e.g. 644)

    if isinstance(n_samples_internal, int) and 0 < n_samples_internal <= n_samples_total:
        n_base = n_samples_internal
    else:
        n_base = n_samples_total

    max_dates = int(_get(cfg, "full_gen_eval.max_dates", -1) or -1)
    if max_dates > 0:
        n_use = min(n_base, max_dates)
    else:
        n_use = n_base

    logger.info(
        "[get_final_gen_dataloader] Split='%s', n_samples_total=%d, n_base=%d, using n_use=%d",
        split_norm, n_samples_total, n_base, n_use,
    )

    # Deterministic sequential sampler over indices 0..n_use-1
    sampler = SequentialSampler(range(n_use))

    gen_loader = DataLoader(
        gen_dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=0,          # keep generation single-threaded & deterministic
        worker_init_fn=_worker_init_fn,
        drop_last=False,
    )

    return gen_loader



def infer_in_channels(cfg: dict) -> int:
    # TODO: Should be more general - e.g. if multiple LR conds with different channels (HR/LR scaling), multiple geo channels (mask+value)
    # low-res conditions
    n_lr = len(cfg['lowres']['condition_variables']) if cfg['lowres']['condition_variables'] is not None else 0

    if cfg['lowres']['dual_lr']:
        n_lr += 1 # Add one extra LR channel (dual LR)

    n_geo = 0
    if cfg['stationary_conditions']['geographic_conditions']['sample_w_geo']:
        geo_variables = cfg["stationary_conditions"]["geographic_conditions"]["geo_variables"]
        # If using mask in classifier, double the number of geo channels (mask + value)
        if cfg['stationary_conditions']['geographic_conditions']['with_mask']:
            n_geo = 2 * len(geo_variables)
        else:
            n_geo = len(geo_variables)
    return n_lr + n_geo

def get_model(cfg):
    '''
        Get the model based on the configuration.
        Args:
            cfg (dict): Configuration dictionary containing model settings.
        Returns:
            score_model (ScoreNet): The score model instance.
            checkpoint_path (str): Path to the model checkpoint.
            checkpoint_name (str): Name of the model checkpoint file.
    '''
    
    # Define model parameters
    input_channels = infer_in_channels(cfg)
    output_channels = 1#len(cfg['highres']['variable'])  # Assuming a single output channel for the high-resolution variable
    
    # Log the number of channels
    logger.info(f"Input channels: {input_channels}")
    logger.info(f"Output channels: {output_channels}")

    device = get_device()

    # === Model architecture knobs (decoder upsampling/norm/activation) ===
    model_cfg = cfg.get('model', {})
    use_resize_conv = bool(model_cfg.get('use_resize_conv', True))
    decoder_norm = model_cfg.get('decoder_norm', 'group')  # Options: 'group', 'instance', None
    decoder_gn_groups = int(model_cfg.get('decoder_gn_groups', 8))  # Number of groups for GroupNorm
    decoder_activation_name = model_cfg.get('decoder_activation', 'SiLU')  # Options: 'relu', 'sily', 'gelu', etc.
    decoder_activation_name_lower = decoder_activation_name.lower()

    _act_map = {'relu': nn.ReLU,
                'silu': nn.SiLU,
                'gelu': nn.GELU,}
    decoder_activation = _act_map.get(decoder_activation_name_lower, nn.ReLU)  # Default to SiLU if not found
    logger.info(f"[MODEL] use_resize_conv: {use_resize_conv}, decoder_norm: {decoder_norm}, decoder_gn_groups: {decoder_gn_groups}, decoder_activation: {decoder_activation_name}")

    if cfg['lowres']['condition_variables'] is not None:
        sample_w_cond_img = True
    else:
        sample_w_cond_img = False

    # Setup model checkpoint name and path
    save_str = get_model_string(cfg)
    checkpoint_name = save_str + '.pth.tar'

    checkpoint_dir = os.path.join(cfg['paths']['path_save'], cfg['paths']['checkpoint_dir'])

    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    # Create the model

    encoder = Encoder(input_channels=input_channels,
                      time_embedding=cfg['sampler']['time_embedding'],
                      cond_on_img=sample_w_cond_img,
                      block_layers=cfg['sampler']['block_layers'],
                      num_classes=cfg['stationary_conditions']['seasonal_conditions']['n_seasons'] if cfg['stationary_conditions']['seasonal_conditions']['sample_w_cond_season'] else None,
                      n_heads=cfg['sampler']['num_heads'],
                      )
    decoder = Decoder(last_fmap_channels=cfg['sampler']['last_fmap_channels'],
                      output_channels=output_channels,
                      time_embedding=cfg['sampler']['time_embedding'],
                      n_heads=cfg['sampler']['num_heads'],
                      use_resize_conv=use_resize_conv,
                      norm=decoder_norm,
                      gn_groups=decoder_gn_groups,
                      activation=decoder_activation,
                      )
    
    edm_cfg = cfg.get('edm', {})
    edm_enabled = bool(edm_cfg.get('enabled', False))

    if edm_enabled:
        sigma_data = float(edm_cfg.get('sigma_data', 1.0))
        predict_residual = bool(edm_cfg.get('predict_residual', False)) # NOTE: Start with False, when EDM is stable, try True
        score_model = EDMPrecondUNet(encoder=encoder,
                                     decoder=decoder,
                                     sigma_data=sigma_data,
                                     predict_residual=predict_residual).to(device)
        
    else:
        sigma = float(cfg.get('ve_dsm', {}).get('sigma', 25.0))
        mprob = partial(marginal_prob_std, sigma=sigma)
        score_model = ScoreNet(marginal_prob_std=mprob,
                            encoder=encoder,
                            decoder=decoder,
                            debug_pre_sigma_div=False
                            )

    if hasattr(score_model, "debug_pre_sigma_div"):
        object.__setattr__(score_model, "debug_pre_sigma_div", False)

    return score_model, checkpoint_dir, checkpoint_name


def get_optimizer(cfg, model):
    '''
        Get the optimizer based on the configuration.
        Args:
            cfg (dict): Configuration dictionary containing optimizer settings.
            model (torch.nn.Module): The model to optimize.
        Returns:
            optimizer (torch.optim.Optimizer): The optimizer instance.
    '''

    if cfg['training']['optimizer'] == 'adam':
        optimizer = Adam(model.parameters(),
                         lr=cfg['training']['learning_rate'],
                         weight_decay=cfg['training']['weight_decay'])
    elif cfg['training']['optimizer'] == 'sgd':
        optimizer = SGD(model.parameters(),
                        lr=cfg['training']['learning_rate'],
                        momentum=cfg['training']['momentum'],
                        weight_decay=cfg['training']['weight_decay'])
    elif cfg['training']['optimizer'] == 'adamw':
        optimizer = AdamW(model.parameters(),
                          lr=cfg['training']['learning_rate'],
                          weight_decay=cfg['training']['weight_decay'])
    else:
        raise ValueError(f"Optimizer {cfg['training']['optimizer']} not recognized. Use 'adam', 'sgd', or 'adamw'.")
    
    return optimizer


def get_scheduler(cfg, optimizer):
    '''
        Get the learning rate scheduler based on the configuration.
        Args:
            cfg (dict): Configuration dictionary containing scheduler settings.
            optimizer (torch.optim.Optimizer): The optimizer to schedule.
        Returns:
            scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler instance.
    '''
    lr_scheduler_type = cfg['training'].get('lr_scheduler', None)
    if lr_scheduler_type == 'Step':
        scheduler = StepLR(optimizer,
                           step_size=cfg['training']['lr_scheduler_params']['step_size'],
                           gamma=cfg['training']['lr_scheduler_params']['gamma'])
                           
    elif lr_scheduler_type == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer,
                                      mode='min',
                                      factor=cfg['training']['lr_scheduler_params']['factor'],
                                      patience=cfg['training']['lr_scheduler_params']['patience'],
                                      verbose=True)
    elif lr_scheduler_type == 'CosineAnnealing':
        scheduler = CosineAnnealingLR(optimizer,
                                      T_max=cfg['training']['lr_scheduler_params']['T_max'],
                                      eta_min=cfg['training']['lr_scheduler_params']['eta_min'])
    elif lr_scheduler_type == None:
        scheduler = None
        logger.warning("No learning rate scheduler specified. Using the optimizer's default learning rate.")
    else:
        raise ValueError(f"Scheduler {lr_scheduler_type} not recognized. Use 'Step', 'ReduceLROnPlateau', or 'CosineAnnealing'.")

    return scheduler



def get_device(verbose=True):
    """
    Get the device to be used for training.
    
    Returns:
        torch.device: The device (CPU or GPU) to be used.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        logger.info(f"Using device: {device}")
    return device


def apply_cfg_dropout(
        *,
        cond_images: torch.Tensor | None,
        lsm_cond: torch.Tensor | None,
        topo_cond: torch.Tensor | None,
        y: torch.Tensor | None,
        lr_ups: torch.Tensor | None,
        cfg_guidance: dict | None) -> tuple[
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
            dict]:
    """
        Apply classifier-free guidance drops independently to LR (cond_imgs, lr_ups), GEO (lsm/topo)
        and CLASS (y). Returns possibly modified tensors and an info dict.
        cfg_guidance keys used:
            'enabled': bool,
            'drop_prob_lr': float,              drop probability for LR dynamic conditions (+ seasons)
            'drop_prob_geo': float,             drop probability for geo/static conditions
            'drop_prob_class': float,           drop probability for class conditions
            'null_label_id': int,               category id for "null" label (for long seasons)
            'null_scalar_value': float          value to use when dropping scalar seasons
            'null_geo_value': float             value to use when dropping static geo (topo)
            'null_lr_strategy': str             'zero' | 'noise' | 'scalar' (how to drop LR conds)
            'null_lr_scalar': float             value to use when null_lr_strategy is 'scalar'
    """
    enabled = bool(cfg_guidance.get('enabled', False)) if cfg_guidance is not None else False
    if not enabled:
        return cond_images, lsm_cond, topo_cond, y, lr_ups, {'dropped_lr': False, 'dropped_geo': False, 'dropped_class': False} # Always return 5 + info dict
    
    # Probabilities
    p_lr        = float(cfg_guidance.get('drop_prob_lr', 0.1)) if cfg_guidance is not None else 0.1
    p_geo       = float(cfg_guidance.get('drop_prob_geo', p_lr)) if cfg_guidance is not None else p_lr
    p_class     = float(cfg_guidance.get('drop_prob_class', 0.0)) if cfg_guidance is not None else 0.0

    # Null strategies/constants
    null_label_id       = int(cfg_guidance.get('null_label_id', 0)) if cfg_guidance is not None else 0
    null_scalar         = float(cfg_guidance.get('null_scalar_value', 0.0)) if cfg_guidance is not None else 0.0
    null_geo_value      = float(cfg_guidance.get('null_geo_value', -5.0)) if cfg_guidance is not None else -5.0
    lr_null_strategy    = str(cfg_guidance.get('null_lr_strategy', 'zero')) if cfg_guidance is not None else 'zero'
    lr_null_scalar      = float(cfg_guidance.get('null_lr_scalar', 0.0)) if cfg_guidance is not None else 0.0

    # Draw Bernoulli once per batch (keeps branches balanced and cheaper)
    dev_available = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev_lr  = cond_images.device if cond_images is not None else (lr_ups.device if lr_ups is not None else dev_available)
    dev_geo = lsm_cond.device if lsm_cond is not None else (topo_cond.device if topo_cond is not None else dev_available)
    dev_cls = y.device if isinstance(y, torch.Tensor) else dev_available

    drop_lr_batch    = (torch.rand((), device=dev_lr) < p_lr).item()
    drop_geo_batch   = (torch.rand((), device=dev_geo) < p_geo).item()
    drop_class_batch = (torch.rand((), device=dev_cls) < p_class).item() if isinstance(y, torch.Tensor) else False

    # === LR group (cond_ims + LR_ups) ===
    def _null_lr_like(t):
        if t is None:
            return None
        if lr_null_strategy == 'noise':
            return torch.randn_like(t)
        elif lr_null_strategy == 'scalar':
            return t.new_full(t.shape, lr_null_scalar)
        return torch.zeros_like(t) # 'zero' or default

    if drop_lr_batch:
        cond_images = _null_lr_like(cond_images)
        lr_ups      = _null_lr_like(lr_ups)



    # === GEO group (lsm + topo, value || mask convention) ===
    def _null_geo_like(t):
        if t is None:
            return None
        if t.ndim >= 2 and t.shape[1] >= 2:
            # Assume value + mask convention, set value to null_geo_value, keep mask as is
            out = t.clone()
            out[:, 0, ...] = null_geo_value # Set value channel to null_geo_value
            out[:, 1, ...] = 0.0 # Set mask channel to zero (no land)
            return out
        return t.new_full(t.shape, null_geo_value)

    if drop_geo_batch:
        lsm_cond  = _null_geo_like(lsm_cond)
        topo_cond = _null_geo_like(topo_cond)

    # === CLASS group (y, either categorical or cos/sin) ===
    if drop_class_batch and isinstance(y, torch.Tensor):
        if y.dtype in (torch.float16, torch.float32, torch.float64):
            # Scalar seasons (e.g. cos/sin day-of-year)
            y = torch.zeros_like(y).fill_(null_scalar) # sin/cos DOY (both == null_scalar)
        else:
            # Categorical seasons (long tensor of class indices)
            y = torch.full_like(y, null_label_id) # categorical DOY/season/month
    # Info dict
    info = {
        'dropped_lr': bool(drop_lr_batch),
        'dropped_geo': bool(drop_geo_batch),
        'dropped_class': bool(drop_class_batch)
    }

    return cond_images, lsm_cond, topo_cond, y, lr_ups, info


# def apply_cfg_dropout(
#         cond_images: torch.Tensor | None,
#         lsm: torch.Tensor | None,
#         topo: torch.Tensor | None,
#         seasons: torch.Tensor | None,
#         lr_ups: torch.Tensor | None,
#         cfg_guidance: dict | None
# ):
#     """
#     Classifier-free guidance style dropout for conditioning signals.
#     - Supports separate drop probabilities for LR dynamic conditions and geo/static conditions 
#     - Drops all LR channels per sample together, using Bernoulli masks (good for CFG)
#     - Drops lsm and topo together per sample (so "geo off" really means no geography)
#     - Handles seasons whther it is categorical (LongTensor labels) or continuous scalars (e.g. cos/sin day-of-year),
#       using null_label_id or null_scalar_value from cfg_guidance respectively.
#     - Works with any tensor shapes by broadcasting the per-sample mask to [B, 1, ...] as needed.

#     Args:
#         cond_images: Low-res dynamic conditions [B, C_lr, H, W] (already upsampled/aligned to model grid).
#         lsm:         Land-sea mask or static mask(s)         [B, C_geo1, H, W] or [B,1,H,W] (may be None).
#         topo:        Topography/static feature(s)            [B, C_geo2, H, W] or [B,1,H,W] (may be None).
#         seasons:     Seasonal condition. Can be:
#                      - Long tensor of class indices [B] or [B, 1]
#                      - Float tensor of scalar(s)   [B] or [B, 1] (e.g., cos(day), sin(day))
#         lr_ups:      Low-res conditions upsampled to high-res grid [B, C_lr, H_hr, W_hr] (may be None).
#         cfg_guidance: Dict with keys:
#             {
#               'enabled': bool,
#               'drop_prob_lr': float,     # drop probability for LR dynamic conditions (+ seasons)
#               'drop_prob_geo': float,    # drop probability for geo/static conditions
#               'null_label_id': int,      # category id for "null" label (for long seasons)
#               'null_scalar_value': float # value to use when dropping scalar seasons
#             }

#     Returns:
#         Tuple[cond_images, lsm, topo, seasons, lr_ups] with per-sample drops applied.
#     """
#     if not cfg_guidance or not cfg_guidance.get('enabled', False):
#         return cond_images, lsm, topo, seasons, lr_ups # Always return 5 
    
#     # Resolve per-group drop probabilities
#     p_cond = float(cfg_guidance.get('drop_prob_lr', 0.1))
#     p_geo = float(cfg_guidance.get('drop_prob_geo', 0.1))
#     null_label_id = int(cfg_guidance.get('null_label_id', 0))
#     null_scalar = float(cfg_guidance.get('null_scalar_value', 0.0))

    
#     # Choose a reference tensor to get B/device
#     ref = None
#     for t in (cond_images, lsm, topo, seasons):
#         if t is not None:
#             ref = t
#             break

#     if ref is None:
#         # No drop possible
#         return cond_images, lsm, topo, seasons
#     B = ref.shape[0]
#     device = ref.device
    
#     # === Build per-sample Bernoulli masks ===
#     # LR dynamic (and seasons share p_cond)
#     mask_cond = (torch.rand(B, device=device) < p_cond)  # True -> drop this sample's LR (+season)
#     # Geo/static (drop lsm/topo together per sample)
#     mask_geo  = (torch.rand(B, device=device) < p_geo)   # True -> drop this sample's geo

#     # Helper to expand [B] -> broadcast shape of a target tensor
#     def _expand_mask(m: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         # Target can be [B, ...]. We want mask shaped [B,1,1,1] or [B,1] etc. to broadcast.
#         view_shape = [B] + [1] * (target.dim() - 1)
#         return m.view(*view_shape)

#     # === Apply to LR dynamic conditions ===
#     if cond_images is not None:
#         m = _expand_mask(mask_cond, cond_images)
#         # Zero is a sensible "null" for continuous LR channels
#         cond_images = torch.where(m, torch.zeros_like(cond_images), cond_images)
    
#     predict_residual = bool(cfg_guidance.get('predict_residual', False))
#     if lr_ups is not None and not predict_residual:
#         m = _expand_mask(mask_cond, lr_ups)
#         lr_ups = torch.where(m, torch.zeros_like(lr_ups), lr_ups)

#     # === Apply to static geo (drop together) ===
#     if lsm is not None:
#         m_geo = _expand_mask(mask_geo, lsm)
#         lsm = torch.where(m_geo, torch.zeros_like(lsm), lsm)
#     if topo is not None:
#         m_geo = _expand_mask(mask_geo, topo)
#         topo = torch.where(m_geo, torch.zeros_like(topo), topo)

#     # === Apply to seasonal condition (shares LR mask) ===
#     if seasons is not None:
#         # Accept [B], [B,1], or more
#         if seasons.dtype in (torch.long, torch.int64, torch.int32):
#             # categorical labels
#             if seasons.dim() == 1:
#                 seasons = seasons.clone()
#                 seasons[mask_cond] = null_label_id
#             else:
#                 # e.g., [B,1]
#                 m = _expand_mask(mask_cond, seasons)
#                 seasons = torch.where(m, torch.full_like(seasons, null_label_id), seasons)
#         else:
#             # float scalar(s)
#             fill_val = null_scalar
#             if seasons.dim() == 1:
#                 m = mask_cond
#             else:
#                 m = _expand_mask(mask_cond, seasons)
#             seasons = torch.where(m, torch.full_like(seasons, fill_val), seasons)

#     return cond_images, lsm, topo, seasons, lr_ups
