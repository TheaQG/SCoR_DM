# sbgm/cli/launch_evaluation_sampler_grid.py

from __future__ import annotations
import logging
from typing import Any

from scor_dm.evaluate.evaluate_sampler_grid_main import evaluate_sampler_grid_main

logger = logging.getLogger("launch_evaluation_sampler_grid")


def run(cfg: Any, make_plots: bool = False):
    """
    Programmatic entrypoint so main_app can run sampler grid evaluation.
    `make_plots` can be used inside evaluate_sampler_grid_main via cfg.full_gen_eval.make_plots if you like.
    """
    logger.info("[launch_evaluation_sampler_grid] Starting sampler grid evaluation.")
    # If you want make_plots to be visible downstream, you can set it on cfg:
    if isinstance(getattr(cfg, "full_gen_eval", None), dict):
        cfg.full_gen_eval["make_plots"] = bool(make_plots)  # type: ignore
    elif hasattr(cfg, "full_gen_eval"):
        setattr(cfg.full_gen_eval, "make_plots", bool(make_plots))  # type: ignore

    out_base = evaluate_sampler_grid_main(cfg)
    logger.info(f"[launch_evaluation_sampler_grid] Done. Outputs at: {out_base}")
    return out_base