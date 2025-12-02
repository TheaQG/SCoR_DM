"""
Main entrypoint for σ*-dependent evaluation.
"""
import json
import logging
from pathlib import Path
from evaluate.evaluate_prcp.eval_sigma_star.metrics_sigma_control import evaluate_sigma_control
from evaluate.evaluate_prcp.eval_sigma_star.plot_sigma_control import plot_sigma_control, plot_sigma_control_examples_grid, plot_sigma_control_psd_curves
from scor_dm.utils import get_model_string

logger = logging.getLogger(__name__)

def run(cfg, make_plots=True):
    model_name = get_model_string(cfg)
    sigma_grid = cfg.full_gen_eval.sigma_star_grid
    base_gen = Path(cfg.paths.sample_dir) / "generation" / model_name
    out_dir = Path(cfg.paths.sample_dir) / "evaluation" / model_name / "prcp" / "sigma_control"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read sigma_control config for possible subset of sigma* for examples/PSD
    scfg = getattr(getattr(cfg, "full_gen_eval", {}), "sigma_control", {}) if hasattr(cfg, "full_gen_eval") else {}
    example_sigma_subset = getattr(scfg, "example_sigma_subset", None)

    logger.info(f"[SigmaControl] Evaluating σ* grid {sigma_grid} for {model_name}")

    metrics_paths = evaluate_sigma_control(cfg, sigma_grid, base_gen, out_dir)
    logger.info("[SigmaControl] Metrics written: %s", metrics_paths)

    # Write sigma-control metadata for plotting/context
    try:
        meta = {
            "sigma_star_grid": [float(s) for s in sigma_grid],
            "ramp": {
                "mode": str(getattr(scfg, "sigma_star_mode", getattr(getattr(cfg, "edm", {}), "sigma_star_mode", "global"))),
                "start_frac": float(getattr(scfg, "ramp_start_frac", getattr(getattr(cfg, "edm", {}), "ramp_start_frac", 0.60))),
                "end_frac": float(getattr(scfg, "ramp_end_frac", getattr(getattr(cfg, "edm", {}), "ramp_end_frac", 0.85))),
                "start_sigma": getattr(scfg, "ramp_start_sigma", getattr(getattr(cfg, "edm", {}), "ramp_start_sigma", None)),
                "end_sigma": getattr(scfg, "ramp_end_sigma", getattr(getattr(cfg, "edm", {}), "ramp_end_sigma", None)),
            }
        }
        with open(Path(out_dir) / "sigma_control_meta.json", "w") as f:
            json.dump(meta, f)
    except Exception as e:
        logger.warning(f"[SigmaControl] Failed to write sigma_control_meta.json: {e}")

    figures_dir = Path(out_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if make_plots:
        plot_sigma_control(
            metrics_paths["summary"],
            figures_dir,
            combined=bool(getattr(getattr(cfg, "full_gen_eval", {}), "sigma_control_plot_combined", True)),
        )
        plot_sigma_control_examples_grid(
            cfg,
            sigma_star_grid=sigma_grid,
            gen_base_dir=base_gen,
            out_dir=out_dir,
            sigma_star_subset=example_sigma_subset,            
            n_members=int(getattr(getattr(cfg, "full_gen_eval", {}), "example_n_members", 3)),
            date=getattr(getattr(cfg, "full_gen_eval", {}), "example_date", None),
            land_only=bool(getattr(getattr(cfg, "full_gen_eval", {}), "eval_land_only", True)),
            fname="examples_sigma_grid.png",
        )
        # PSD curves per sigma_star (ensemble-average across dates)
        plot_sigma_control_psd_curves(out_dir, sigma_subset=example_sigma_subset)

    logger.info(f"[SigmaControl] Done. Results in {out_dir}")
    logger.info("[SigmaControl] Figures in: %s", str(Path(out_dir) / "figures"))
    return out_dir