"""
CLI: Launch generation across a grid of sigma_star values.

Usage:
    python -m sbgm.cli.launch_generation_sigma_star --config /path/to/config.yaml \
        --sigma-star-grid 0.70,0.85,1.00,1.15,1.30 \
        --ensemble-size 32 --max-dates -1 --seed 1234

Notes:
- The script loads your YAML config (OmegaConf).
- It injects/overrides `full_gen_eval.sigma_star_grid`, `full_gen_eval.ensemble_size`,
  `full_gen_eval.max_dates`, and `full_gen_eval.seed` if specified.
- It then calls `generation_sigma_grid_main(cfg)` which writes outputs under:
    <paths.sample_dir>/generation/<model_name>/sigma_star=<val>/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, cast

from omegaconf import OmegaConf, DictConfig # type: ignore

from scor_dm.generate.generation_sigma_grid_main import generation_sigma_grid_main

logger = logging.getLogger("launch_generation_sigma_star")

def run(cfg):
    """
    Programmatic entrypoint so main_app can call this launcher without argparse.
    Returns the base output path from generation_sigma_grid_main(cfg).
    """
    logger.info("[launch_generation_sigma_star] Programmatic run() called.")
    out_base = generation_sigma_grid_main(cfg)
    logger.info(f"[launch_generation_sigma_star] Done. Outputs at: {out_base}")
    return out_base

def _parse_sigma_grid(arg: Optional[str]) -> Optional[List[float]]:
    if arg is None or arg == "":
        return None
    # Allow either comma-separated or single float
    try:
        if "," in arg:
            return [float(x.strip()) for x in arg.split(",") if x.strip() != ""]
        return [float(arg)]
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Could not parse --sigma-star-grid '{arg}': {e}") from e


def _ensure_nested(cfg: DictConfig, path: str):
    """
    Ensure nested keys exist in an OmegaConf DictConfig.
    Example: _ensure_nested(cfg, "full_gen_eval")
    """
    parts = path.split(".")
    cur = cfg
    for p in parts:
        if p not in cur or cur.get(p) is None:
            cur[p] = {}
        cur = cur[p]


def _apply_overrides(cfg: DictConfig,
                     sigma_star_grid: Optional[List[float]],
                     ensemble_size: Optional[int],
                     max_dates: Optional[int],
                     seed: Optional[int]):
    # Ensure the section exists
    _ensure_nested(cfg, "full_gen_eval")

    if sigma_star_grid is not None:
        cfg.full_gen_eval.sigma_star_grid = [float(x) for x in sigma_star_grid]
        logger.info(f"Override: full_gen_eval.sigma_star_grid = {cfg.full_gen_eval.sigma_star_grid}")
    if ensemble_size is not None:
        cfg.full_gen_eval.ensemble_size = int(ensemble_size)
        logger.info(f"Override: full_gen_eval.ensemble_size = {cfg.full_gen_eval.ensemble_size}")
    if max_dates is not None:
        cfg.full_gen_eval.max_dates = int(max_dates)
        logger.info(f"Override: full_gen_eval.max_dates = {cfg.full_gen_eval.max_dates}")
    if seed is not None:
        cfg.full_gen_eval.seed = int(seed)
        logger.info(f"Override: full_gen_eval.seed = {cfg.full_gen_eval.seed}")


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
    parser = argparse.ArgumentParser(description="Launch generation across sigma_star grid.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--sigma-star-grid", type=str, default=None,
                        help="Comma-separated list or single value, e.g. '0.7,0.85,1.0,1.15,1.3'.")
    parser.add_argument("--ensemble-size", type=int, default=None, help="Override ensemble size (M).")
    parser.add_argument("--max-dates", type=int, default=None, help="Cap number of dates to process (-1 = all).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for generation.")
    parser.add_argument("-v", "--verbose", action="count", default=1, help="Increase verbosity (-v, -vv).")

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error(f"Config not found: {cfg_path}")
        return 2

    # Load OmegaConf config
    cfg = cast(DictConfig, OmegaConf.load(str(cfg_path)))

    # Basic sanity check for paths.sample_dir
    if "paths" not in cfg or "sample_dir" not in cfg.paths:
        logger.error("Your config must define paths.sample_dir (output root for generated samples).")
        return 2
    sample_dir = Path(cfg.paths.sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Apply optional overrides
    sigma_grid = _parse_sigma_grid(args.sigma_star_grid)
    _apply_overrides(cfg, sigma_grid, args.ensemble_size, args.max_dates, args.seed)

    # Run generation across the grid
    out_base = generation_sigma_grid_main(cfg)
    logger.info(f"Done. Outputs at: {out_base}")

    return 0


if __name__ == "__main__":
    sys.exit(main())