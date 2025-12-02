""" 
    UNIFIED CLI INTERFACE FOR SBGM_SD

    main_app.py
    This script serves as the main control point for the full SBGM_SD application.
    Tasks implemented:
        - Running the training process  
        - Running the generation process on a trained model
        - Running the evaluation process from generated samples
        - Full model pipeline: training --> generation --> evaluation
        - 

    Tasks to be implemented:
        - Data structuring (train/test/eval splits)
        - Running full Dataset statistics based on config
"""
import argparse
import os
import logging



from scor_dm.utils import get_model_string, load_config
from scor_dm.logging_utils import (
    cfg_hash, make_run_name, ensure_run_dir,
    setup_logging, write_run_manifest, log_banner
)
from baselines.baseline_main import run as run_baselines
# from baselines.baseline_eval import run_all as run_baseline_eval
from baselines.evaluate_baselines.evaluation_baselines import run_all_baselines

def check_model_exists(cfg):
    model_name = get_model_string(cfg)
    ckpt_dir = os.path.join(cfg.paths.checkpoint_dir, model_name + '.pth.tar')
    exists = os.path.exists(ckpt_dir) # and any(f.endswith(".pth.tar") for f in os.listdir(ckpt_dir))
    return exists, ckpt_dir

def check_generated_samples_exist(cfg):
    model_name = get_model_string(cfg)
    gen_dir = os.path.join(cfg.paths.sample_dir, "generation", model_name, "generated_samples")
    exists = os.path.exists(gen_dir) and any(f.startswith("gen_samples") for f in os.listdir(gen_dir))
    return exists, gen_dir


# Use setup_logger and write_run_manifest in main
def main():
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description="SBGM full pipeline launcher")
    parser.add_argument("--config_path", required=True, help="Path to the yaml config")
    
    parser.add_argument(
        "--mode",
        choices=[
            "train", "generate", "evaluate", "full_pipeline",
            "data_splits", "quicklook", "baseline", "baseline_eval",
            "sigma_star_generation", "sigma_star_evaluation",
            "sampler_grid_generation", "sampler_grid_evaluation",  # <-- add these
        ],
        default="full_pipeline"
    )
    
    parser.add_argument("--baseline_type", choices=["bilinear", "qm", "unet_sr"], default="bilinear", help="If mode is 'baseline', which baseline to run.")
    parser.add_argument("--baseline_split", choices=["train", "valid", "test"], default="test", help="If mode is 'baseline', which split to use.")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--skip_evaluation", action="store_true")
    parser.add_argument("--make_plots", action="store_true", help="If set, make publication-ready plots after evaluation.")
    parser.add_argument("--dry_run", action="store_true", help="If set, no actual training/generation/evaluation will be performed, only config parsing and logging setup.")
    args = parser.parse_args()


    from omegaconf import DictConfig, ListConfig, OmegaConf # type: ignore
    from typing import cast

    cfg = load_config(args.config_path)
    # Ensure cfg is a DictConfig (some loaders may return a ListConfig); accept a single-element ListConfig wrapping a DictConfig
    if isinstance(cfg, ListConfig):
        if len(cfg) == 1 and isinstance(cfg[0], DictConfig):
            cfg = cfg[0]
        else:
            raise RuntimeError("Expected a DictConfig or a single-element ListConfig containing a DictConfig for cfg.")
    cfg = cast(DictConfig, cfg)

    # # Apply baseline CLI overrides if provided
    # if args.baseline_type is not None:
    #     if not hasattr(cfg, 'baseline') or cfg.baseline is None:
    #         cfg.baseline = {}
    #     cfg.baseline['type'] = args.baseline_type
    # if args.baseline_split is not None:
    #     if not hasattr(cfg, 'baseline') or cfg.baseline is None:
    #         cfg.baseline = {}
    #     cfg.baseline['split'] = args.baseline_split

    # === Build run context ===
    model_name = get_model_string(cfg)
    h = cfg_hash(cfg)
    run_name = make_run_name(cfg.experiment.name, h)
    run_dir = ensure_run_dir(cfg.paths.log_dir, model_name)

    make_plots = args.make_plots or (args.mode in ['evaluate', 'full_pipeline'] and not args.skip_evaluation)

    # === Logging + manifest ===
    cfg_py = OmegaConf.to_container(cfg, resolve=True) # Convert to plain dict for logging
    file_level = getattr(cfg, "logging", {}).get("file_level", "INFO")
    console_level = getattr(cfg, "logging", {}).get("console_level", "WARNING")
    log_path = setup_logging(run_dir, run_name,
                             file_level=getattr(cfg, "logging", {}).get("file_level", "INFO"),
                             console_level=getattr(cfg, "logging", {}).get("console_level", "WARNING")
                             )
    logger.info("Unified log file: %s", log_path) # This line should appear in both log file and SLURM .out

    write_run_manifest(run_dir, run_name, cfg, model_name)
    
    logger.info("=== ENTERED SBGM_SD MAIN APP ===")
    logger.info("Experiment      : %s", cfg.experiment.name)
    logger.info("Mode            : %s", args.mode)
    logger.info("Config          : %s", args.config_path)
    logger.info("Run dir         : %s", run_dir)
    logger.info("Log file        : %s", log_path)
    logger.info("Model key       : %s", model_name)
    logger.info("Cfg hash        : %s", h)

    # Imports kept here to avoid circular imports
    from scor_dm.cli import (
        launch_sbgm,
        launch_generation,
        launch_evaluation,
        launch_quicklook,
        launch_generation_sigma_star,
        launch_evaluation_sigma_star,
        launch_generation_sampler_grid,       # <-- add
        launch_evaluation_sampler_grid,       # <-- add
    )
    from data_analysis_pipeline.cli import launch_split_creation

    # === Dispatch with banners ===
    if args.mode == "data_splits":
        log_banner("DATA SPLIT CREATION START")
        launch_split_creation.run(cfg)
        log_banner("DATA SPLIT CREATION DONE")

    if args.mode == "train":
        log_banner("TRAINING START")
        launch_sbgm.run_training(cfg)
        log_banner("TRAINING DONE")

    elif args.mode == "generate":
        log_banner("GENERATION START")
        launch_generation.run_generation(cfg)
        log_banner("GENERATION DONE")

    elif args.mode == "evaluate":
        log_banner("EVALUATION START")
        # exists, gen_dir = check_generated_samples_exist(cfg)
        # if not exists:
        #     raise RuntimeError(f"Cannot evaluate: generated samples not found in {gen_dir}")
        launch_evaluation.run_evaluation(cfg, make_plots=make_plots)
        log_banner("EVALUATION DONE")

    elif args.mode == "quicklook":
        log_banner("QUICKLOOK START")
        exists, ckpt_dir = check_model_exists(cfg)
        if not exists:
            raise RuntimeError(f"Cannot run quicklook: model checkpoint not found in {ckpt_dir}")
        launch_quicklook.run_quicklook(cfg)
        log_banner("QUICKLOOK DONE")

    elif args.mode == "full_pipeline":
        log_banner("TRAINING START")
        exists, ckpt_dir = check_model_exists(cfg)
        if args.skip_train and not exists:
            raise RuntimeError(f"Cannot skip training: no trained model found in {ckpt_dir}")
        if not args.skip_train:
            launch_sbgm.run_training(cfg)
        log_banner("TRAINING DONE")

        log_banner("QUICKLOOK START")
        exists, ckpt_dir = check_model_exists(cfg)
        if not exists:
            raise RuntimeError(f"Cannot run quicklook: model checkpoint not found in {ckpt_dir}")
        launch_quicklook.run_quicklook(cfg)
        log_banner("QUICKLOOK DONE")

        log_banner("GENERATION START")
        exists, gen_dir = check_generated_samples_exist(cfg)
        # if args.skip_generation and not exists:
        #     raise RuntimeError(f"Cannot skip generation: no samples found in {gen_dir}")
        if not args.skip_generation:
            launch_generation.run_generation(cfg)
        log_banner("GENERATION DONE")

        log_banner("EVALUATION START")
        if not args.skip_evaluation:
            launch_evaluation.run_evaluation(cfg, make_plots=make_plots)
        log_banner("EVALUATION DONE")

    elif args.mode == "baseline":
        log_banner(f"BASELINE START")
        run_baselines(cfg)
        log_banner(f"BASELINE DONE")

    elif args.mode == "baseline_eval":
        log_banner(f"BASELINE EVALUATION START")
        # run_baseline_eval(cfg)
        run_all_baselines(cfg)
        log_banner(f"BASELINE EVALUATION DONE")

    elif args.mode == "sigma_star_generation":
        log_banner("SIGMA_STAR GENERATION START")
        launch_generation_sigma_star.run(cfg)
        log_banner("SIGMA_STAR GENERATION DONE")

    elif args.mode == "sigma_star_evaluation":
        log_banner("SIGMA_STAR EVALUATION START")
        # use args.make_plots to also toggle making qualitative example montages
        launch_evaluation_sigma_star.run(cfg, make_plots=make_plots, make_examples=args.make_plots)
        log_banner("SIGMA_STAR EVALUATION DONE")

    elif args.mode == "sampler_grid_generation":
        log_banner("SAMPLER GRID GENERATION START")
        launch_generation_sampler_grid.run(cfg)
        log_banner("SAMPLER GRID GENERATION DONE")

    elif args.mode == "sampler_grid_evaluation":
        log_banner("SAMPLER GRID EVALUATION START")
        # make_plots can control whether the evaluation makes plots or only tables
        launch_evaluation_sampler_grid.run(cfg, make_plots=make_plots)
        log_banner("SAMPLER GRID EVALUATION DONE")

    logger.info("=== SBGM_SD MAIN APP DONE ===")


if __name__ == "__main__":
    main()