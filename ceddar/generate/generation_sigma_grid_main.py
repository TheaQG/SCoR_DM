import os
import logging
from pathlib import Path
import numpy as np
import torch
from typing import Any

from scor_dm.training_utils import get_model, get_final_gen_dataloader
from generate.generation import GenerationRunner, GenerationConfig
from scor_dm.utils import get_model_string

logger = logging.getLogger(__name__)

def _resolve_base_out_dir(cfg) -> Path:
    """Base dir: <paths.sample_dir>/generation/<model_name>/ (robust to dict/attr cfg)."""
    model_name_str = get_model_string(cfg)
    # Try attribute access first, fall back to dict-style
    sample_dir = None
    if hasattr(cfg, "paths") and hasattr(cfg.paths, "sample_dir"):
        sample_dir = cfg.paths.sample_dir
    elif isinstance(cfg, dict) and "paths" in cfg and "sample_dir" in cfg["paths"]:
        sample_dir = cfg["paths"]["sample_dir"]
    else:
        raise KeyError("Could not resolve cfg.paths.sample_dir")
    base = Path(sample_dir) / "generation" / model_name_str
    base.mkdir(parents=True, exist_ok=True)
    return base

def _build_generation_config(cfg, out_root: Path) -> GenerationConfig:
    cfg_full_gen_eval = cfg.get('full_gen_eval', cfg)
    M = int(cfg_full_gen_eval.get('ensemble_size', cfg.data_handling.get('n_gen_samples', 32)))
    edm = cfg.get('edm', {})

    return GenerationConfig(
        output_root=str(out_root),
        ensemble_size=M,
        sampler_steps=int(edm.get('sampling_steps', 40)),
        seed=int(cfg_full_gen_eval.get('seed', 1234)),
        use_edm=bool(edm.get('enabled', True)),
        sigma_min=float(edm.get('sigma_min', 0.002)),
        sigma_max=float(edm.get('sigma_max', 80.0)),
        rho=float(edm.get('rho', 7.0)),
        S_churn=float(edm.get('S_churn', 0.0)),
        S_min=float(edm.get('S_min', 0.0)),
        S_max=float(edm.get('S_max', float('inf'))),
        S_noise=float(edm.get('S_noise', 1.0)),
        predict_residual=bool(edm.get('predict_residual', False)),
        save_space="physical",
        max_dates=int(cfg_full_gen_eval.get('max_dates', -1)),
    )


# Helper: float or None
def _float_or_none(x: Any):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None

def generation_sigma_grid_main(cfg):
    """
    Generate ensembles across a grid of sigma_star values.
    For each sigma_star, outputs go to:
      <sample_dir>/generation/<model_name>/sigma_star=<val>/
    """
    # ----------------------- Seed -----------------------
    seed = int(getattr(getattr(cfg, "full_gen_eval", {}), "seed", 1234) if not isinstance(cfg, dict) else cfg.get("full_gen_eval", {}).get("seed", 1234))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # ----------------------- Device -----------------------
    device = getattr(getattr(cfg, "training", {}), "device", None)
    if device is None and isinstance(cfg, dict):
        device = cfg.get("training", {}).get("device", "cpu")
    if device is None:
        device = "cpu"

    # ----------------------- Model & checkpoint -----------------------
    model, ckpt_dir, ckpt_name = get_model(cfg)
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    ckpt = torch.load(ckpt_path, map_location=device)
    if "network_params" not in ckpt:
        raise KeyError(f"Checkpoint missing 'network_params': {ckpt_path}")
    model.load_state_dict(ckpt["network_params"])
    model.eval()
    logger.info(f"[generation_sigma_grid_main] Loaded checkpoint: {ckpt_path}")

    # ----------------------- Data (deterministic, split-aware) -----------------------
    # Decide which temporal split to generate for: train / val / test
    if isinstance(cfg, dict):
        full_gen_eval = cfg.get("full_gen_eval", {})
        split_cfg = str(full_gen_eval.get("split", "test")).lower()
    else:
        full_gen_eval = getattr(cfg, "full_gen_eval", {})
        split_cfg = str(getattr(full_gen_eval, "split", "test")).lower()

    if split_cfg in ("val", "valid", "validation"):
        split_for_dataset = "valid"
    elif split_cfg == "train":
        split_for_dataset = "train"
    else:
        split_for_dataset = "test"

    # Make sure data_handling exists
    if isinstance(cfg, dict):
        cfg.setdefault("data_handling", {})
        dh = cfg["data_handling"]
        dh["split"] = split_for_dataset
        dh["shuffle"] = False
        dh["drop_last"] = False
    else:
        if not hasattr(cfg, "data_handling") or cfg.data_handling is None:
            cfg.data_handling = {}
        cfg.data_handling["split"] = split_for_dataset
        cfg.data_handling["shuffle"] = False
        cfg.data_handling["drop_last"] = False

    logger.info(f"[generation_sigma_grid_main] Using data split='{split_for_dataset}' for sigma* grid generation")

    gen_dataloader = get_final_gen_dataloader(cfg, split=split_for_dataset)

    # ----------------------- Sigma* values -----------------------
    full_gen_eval = cfg.get('full_gen_eval', {}) if isinstance(cfg, dict) else getattr(cfg, "full_gen_eval", {})
    grid = full_gen_eval.get('sigma_star_grid', [1.0]) if isinstance(full_gen_eval, dict) else getattr(full_gen_eval, "sigma_star_grid", [1.0])
    if isinstance(grid, (float, int)):
        grid = [float(grid)]
    grid = [float(x) for x in grid]
    logger.info(f"[generation_sigma_grid_main] sigma_star_grid = {grid}")

    # ----------------------- Sigma* ramp settings (optional late-step control) -----------------------
    scfg = full_gen_eval.get('sigma_control', {}) if isinstance(full_gen_eval, dict) else getattr(full_gen_eval, "sigma_control", {})
    edm_cfg = cfg.get('edm', {}) if isinstance(cfg, dict) else getattr(cfg, "edm", {})
    ramp_mode = str(scfg.get('sigma_star_mode', edm_cfg.get('sigma_star_mode', 'global')) if isinstance(scfg, dict) else getattr(scfg, "sigma_star_mode", getattr(edm_cfg, "sigma_star_mode", "global")))
    ramp_start_frac = float(scfg.get('ramp_start_frac', edm_cfg.get('ramp_start_frac', 0.60)) if isinstance(scfg, dict) else getattr(scfg, "ramp_start_frac", getattr(edm_cfg, "ramp_start_frac", 0.60)))
    ramp_end_frac   = float(scfg.get('ramp_end_frac', edm_cfg.get('ramp_end_frac', 0.85)) if isinstance(scfg, dict) else getattr(scfg, "ramp_end_frac", getattr(edm_cfg, "ramp_end_frac", 0.85)))
    ramp_start_sigma = _float_or_none(scfg.get('ramp_start_sigma', getattr(getattr(cfg, 'edm', {}), 'ramp_start_sigma', None) if not isinstance(cfg, dict) else cfg.get('edm', {}).get('ramp_start_sigma', None)))
    ramp_end_sigma   = _float_or_none(scfg.get('ramp_end_sigma',   getattr(getattr(cfg, 'edm', {}), 'ramp_end_sigma',   None) if not isinstance(cfg, dict) else cfg.get('edm', {}).get('ramp_end_sigma',   None)))

    # ----------------------- Base output -----------------------
    base_out = _resolve_base_out_dir(cfg)

    # ----------------------- Loop over sigma* -----------------------
    for sstar in grid:
        # 1) Set effective sigma_star in config (read by GenerationRunner via edm_cfg)
        if isinstance(cfg, dict):
            cfg.setdefault('edm', {})
            cfg['edm']['sigma_star'] = float(sstar)
            # --- Push ramp settings into cfg.edm for sampler ---
            cfg['edm']['sigma_star_mode'] = ramp_mode
            cfg['edm']['ramp_start_frac'] = ramp_start_frac
            cfg['edm']['ramp_end_frac']   = ramp_end_frac
            cfg['edm']['ramp_start_sigma'] = ramp_start_sigma
            cfg['edm']['ramp_end_sigma']   = ramp_end_sigma
        else:
            if not hasattr(cfg, 'edm'):
                cfg.edm = type('EDM', (), {})()
            cfg.edm.sigma_star = float(sstar) # type: ignore
            cfg.edm.sigma_star_mode = ramp_mode # type: ignore
            cfg.edm.ramp_start_frac = ramp_start_frac # type: ignore
            cfg.edm.ramp_end_frac   = ramp_end_frac # type: ignore
            cfg.edm.ramp_start_sigma = ramp_start_sigma # type: ignore
            cfg.edm.ramp_end_sigma   = ramp_end_sigma # type: ignore

        # Mirror into full_gen_eval.sigma_control for components that read from there
        if not hasattr(cfg, 'full_gen_eval') and isinstance(cfg, dict):
            cfg.setdefault('full_gen_eval', {})
        if hasattr(cfg, 'full_gen_eval') and not hasattr(cfg.full_gen_eval, 'sigma_control') and not isinstance(cfg, dict): # type: ignore
            setattr(cfg.full_gen_eval, 'sigma_control', {})
        sc = cfg['full_gen_eval'].get('sigma_control', {}) if isinstance(cfg, dict) else cfg.full_gen_eval.sigma_control
        try:
            sc['sigma_star_mode'] = ramp_mode
            sc['ramp_start_frac'] = ramp_start_frac
            sc['ramp_end_frac'] = ramp_end_frac
            sc['ramp_start_sigma'] = ramp_start_sigma
            sc['ramp_end_sigma'] = ramp_end_sigma
        except TypeError:
            # If sc is an object-like config
            setattr(sc, 'sigma_star_mode', ramp_mode)
            setattr(sc, 'ramp_start_frac', ramp_start_frac)
            setattr(sc, 'ramp_end_frac', ramp_end_frac)
            setattr(sc, 'ramp_start_sigma', ramp_start_sigma)
            setattr(sc, 'ramp_end_sigma', ramp_end_sigma)

        logger.info(f"[generation_sigma_grid_main] σ*: {sstar:.2f} | mode={ramp_mode} | ramp_frac=({ramp_start_frac:.2f},{ramp_end_frac:.2f}) | ramp_sigma=({ramp_start_sigma},{ramp_end_sigma})")

        # 2) Subdir for this sigma*
        subdir = base_out / f"sigma_star={sstar:.2f}"
        subdir.mkdir(parents=True, exist_ok=True)

        # 3) Build runner config, pointing to subdir
        gen_cfg = _build_generation_config(cfg, subdir)

        # 4) Run generator
        logger.info(f"[generation_sigma_grid_main] Generating for sigma_star={sstar:.2f} -> {subdir}")
        try:
            runner = GenerationRunner(model=model, cfg=cfg, device=device, out_root=subdir, gen_config=gen_cfg)
        except TypeError as e:
            logger.warning(f"[generation_sigma_grid_main] GenerationRunner signature issue, retrying without out_root: {e}")
            runner = GenerationRunner(model=model, cfg=cfg, device=device, gen_config=gen_cfg) # type: ignore
        runner.run(gen_dataloader)

    logger.info(f"[generation_sigma_grid_main] Done. Outputs at: {base_out}")
    return base_out