# sbgm/cli/launch_evaluation_sigma_star.py
"""
CLI: Launch σ*-dependent evaluation (correlation/PSD/CRPS) for precipitation.

Usage:
    python -m sbgm.cli.launch_evaluation_sigma_star --config /path/to/config.yaml --make-examples

Notes:
- Loads your YAML config (OmegaConf).
- Calls the precipitation sigma*-evaluation orchestrator, which expects generation outputs under:
    <paths.sample_dir>/generation/<model_name>/sigma_star=<val>/
- Results are written under:
    <paths.evaluation_dir>/<model_name>/prcp/sigma_control/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List, cast

from omegaconf import OmegaConf, DictConfig # type: ignore

from scor_dm.evaluate.evaluate_prcp.eval_sigma_star.evaluate_sigma_control import run as run_sigma_star_eval
from scor_dm.evaluate.evaluate_prcp.eval_sigma_star.plot_sigma_control import plot_sigma_control_examples_grid

logger = logging.getLogger("launch_evaluation_sigma_star")


def run(cfg: DictConfig, make_plots: bool = True, make_examples: bool = False):
    """
    Programmatic entrypoint so main_app can call this launcher without argparse.
    """
    logger.info("[launch_evaluation_sigma_star] Programmatic run() called.")
    out_dir = run_sigma_star_eval(cfg, make_plots=make_plots)

    if make_examples:
        model_name = cfg.experiment.name
        gen_base = Path(cfg.paths.sample_dir) / "generation" / model_name
        sigma_grid = list(cfg.full_gen_eval.sigma_star_grid)
        plot_sigma_control_examples_grid(            cfg,
            sigma_star_grid=sigma_grid,
            gen_base_dir=gen_base,
            out_dir=out_dir,
            n_members=int(getattr(getattr(cfg, "full_gen_eval", {}), "example_n_members", 3)),
            date=getattr(getattr(cfg, "full_gen_eval", {}), "example_date", None),
            land_only=bool(getattr(getattr(cfg, "full_gen_eval", {}), "eval_land_only", True)),
            fname="examples_sigma_grid.png",
        )

    logger.info(f"[launch_evaluation_sigma_star] Done. Results at: {out_dir}")
    return out_dir


def _setup_logging(verbosity: int):
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Launch σ*-dependent evaluation for precipitation.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--no-plots", action="store_true", help="Disable summary figure creation.")
    parser.add_argument("--make-examples", action="store_true", help="Also write qualitative example montages.")
    parser.add_argument("-v", "--verbose", action="count", default=1, help="Increase verbosity (-v, -vv).")
    args = parser.parse_args(argv)

    _setup_logging(args.verbose)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error(f"Config not found: {cfg_path}")
        return 2

    cfg = cast(DictConfig, OmegaConf.load(str(cfg_path)))

    make_plots = not args.no_plots
    make_examples = bool(args.make_examples)

    run(cfg, make_plots=make_plots, make_examples=make_examples)
    return 0


if __name__ == "__main__":
    sys.exit(main())