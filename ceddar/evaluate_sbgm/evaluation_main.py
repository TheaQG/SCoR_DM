"""
    Entry point like generation_main, but for evaluation after generation has been done.
    Reads generation outputs from cfg or CLI, builds mask, and runs EvaluationRunner.
"""

import os
import logging
import numpy as np
import torch
from pathlib import Path
from omegaconf import OmegaConf
import yaml

from sbgm.utils import get_model_string
from sbgm.training_utils import get_model  # only for model string & dims if needed
from sbgm.evaluate_sbgm.evaluation import EvaluationRunner, EvaluationConfig, load_all_baselines

logger = logging.getLogger(__name__)


def _default_gen_dir(cfg) -> Path:
    model_str = get_model_string(cfg)
    sample_root = cfg["paths"]["sample_dir"]
    return Path(sample_root) / "generation" / model_str

def _default_eval_dir(cfg) -> Path:
    model_str = get_model_string(cfg)
    sample_root = cfg["paths"]["sample_dir"]
    return Path(sample_root) / "evaluation" / model_str



def evaluation_main(cfg):
    cfg_full_gen_eval = cfg.get("full_gen_eval", {})
    do_prob = bool(cfg_full_gen_eval.get("do_prob", True))
    do_cap = bool(cfg_full_gen_eval.get("do_cap", True))
    do_ext = bool(cfg_full_gen_eval.get("do_ext", True))
    gen_dir = cfg_full_gen_eval.get("gen_dir", None)  # If None, use default
    eval_dir = cfg_full_gen_eval.get("eval_dir", None)

    # Logging & seed
    seed = int(cfg_full_gen_eval.get("seed", 1234))
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); np.random.seed(seed)
    # logger.info(f"[evaluation_main] Configuration:\n{OmegaConf.to_yaml(cfg)}")

    device = torch.device(cfg["training"]["device"])

    # Directories
    gen_root = Path(gen_dir) if gen_dir is not None else _default_gen_dir(cfg)
    eval_root = Path(eval_dir) if eval_dir is not None else _default_eval_dir(cfg)
    eval_root.mkdir(parents=True, exist_ok=True)

    # Land mask (optional)
    try:
        mask = None  # replace with your own loader if you have one
        # mask = load_land_mask_if_any(cfg)  # returns torch.bool [H,W]
    except Exception as e:
        logger.warning(f"[evaluation_main] No land mask loaded: {e}")
        mask = None

    # Build EvaluationConfig
    ev_cfg = EvaluationConfig(
        gen_dir=str(gen_root),
        out_dir=str(eval_root),
        grid_km_per_px=float(cfg_full_gen_eval.get("grid_km_per_px", 2.5)),
        lr_grid_km_per_px=float(cfg_full_gen_eval.get("lr_grid_km_per_px", 31.0)),
        fss_scales_km=tuple(cfg_full_gen_eval.get("fss_scales_km", (5,10,20))),
        thresholds_mm=tuple(cfg_full_gen_eval.get("thresholds_mm", (1.0,5.0,10.0))),
        wet_threshold_mm=float(cfg_full_gen_eval.get("wet_threshold_mm", 1.0)),
        reliability_bins=int(cfg_full_gen_eval.get("reliability_bins", 10)),
        spread_skill_bins=int(cfg_full_gen_eval.get("spread_skill_bins", 10)),
        pit_bins=int(cfg_full_gen_eval.get("pit_bins", 20)),
        psd_ignore_low_k_bins=int(cfg_full_gen_eval.get("psd_ignore_low_k_bins", 1)),
        psd_normalize=str(cfg_full_gen_eval.get("psd_normalize", "none")),
        random_ref_kind=str(cfg_full_gen_eval.get("random_ref_kind", "phase_randomized")),
        seasonal_summaries=bool(cfg_full_gen_eval.get("seasonal_summaries", True)),
        region_mask_path=cfg_full_gen_eval.get("region_mask_path", None),
        pixel_dist_n_bins=int(cfg_full_gen_eval.get("pixel_dist_n_bins", 100)),
        pixel_dist_vmax_percentile=float(cfg_full_gen_eval.get("pixel_dist_vmax_percentile", 99.5)),
        pixel_dist_save_cap=int(cfg_full_gen_eval.get("pixel_dist_save_cap", 2_000_000)),
        add_yearly_ratio_diff=bool(cfg_full_gen_eval.get("add_yearly_ratio_diff", True)),
        yearly_maps=tuple(cfg_full_gen_eval.get("yearly_maps", ("mean", "sum", "rx1", "rx5"))),
        seasons=tuple(cfg_full_gen_eval.get("seasons", ("ALL","DJF","MAM","JJA","SON"))),
        seed=seed,
        eval_land_only=bool(cfg_full_gen_eval.get("eval_land_only", False)),
    )

    # Set up baselines
    use_baselines = bool(cfg_full_gen_eval.get("compare_with_baselines", True))
    baseline_names = cfg_full_gen_eval.get("baseline_names", None)  # If None, use all available
    split = str(cfg_full_gen_eval.get("baseline_split", "test"))

    baseline_data = None
    if use_baselines:
        baseline_data = load_all_baselines(cfg, split=split, names=baseline_names)
        logger.info(f"[evaluation_main] Loaded baseline data for {len(baseline_data)} baselines: {list(baseline_data.keys())}")

    # NEW: build baseline eval directories map (name -> eval dir)
    baseline_eval_dirs = None
    if use_baselines:
        base_root = Path(cfg["paths"]["sample_dir"]) / "evaluation" / "baselines"
        names = baseline_names or (list(baseline_data.keys()) if baseline_data else [])
        baseline_eval_dirs = {name: str(base_root / name / split) for name in names}

    runner = EvaluationRunner(
        cfg_yaml=cfg,
        eval_cfg=ev_cfg,
        device=device,
        mask=mask,
        baseline_data=baseline_data,
        baseline_eval_dirs=baseline_eval_dirs,  # <-- now actually populated
    )
    runner.run_all(do_prob=do_prob, do_cap=do_cap, do_ext=do_ext)

    logger.info(f"[evaluation_main] Done. Outputs at: {eval_root}")
    return eval_root