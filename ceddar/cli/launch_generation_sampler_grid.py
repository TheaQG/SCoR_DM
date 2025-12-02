# sbgm/cli/launch_generation_sampler_grid.py

from __future__ import annotations
import logging
from typing import Any

from scor_dm.generate.generation_sampler_grid_main import generation_sampler_grid_main

logger = logging.getLogger("launch_generation_sampler_grid")


def run(cfg: Any):
    """
    Programmatic entrypoint so main_app can run the sampler grid generation.
    """
    logger.info("[launch_generation_sampler_grid] Starting sampler grid generation.")
    out_base = generation_sampler_grid_main(cfg)
    logger.info(f"[launch_generation_sampler_grid] Done. Outputs at: {out_base}")
    return out_base