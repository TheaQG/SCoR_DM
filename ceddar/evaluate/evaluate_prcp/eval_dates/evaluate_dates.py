# sbgm/evaluate/evaluate_prcp/eval_dates/evaluate_dates.py
from __future__ import annotations
from pathlib import Path
from typing import Sequence
import logging

from evaluate.evaluate_prcp.eval_dates.plot_dates import plot_dates_montages

logger = logging.getLogger(__name__)


def run_dates(
    resolver,
    eval_cfg,
    out_root: str | Path,
    *,
    dates: Sequence[str] | None = None,
    include_lr: bool = True,
    include_members: bool = True,
    n_members: int = 3,
    cmap: str = "Blues",
    percentile: float = 99.5,
    land_only: bool = True,
) -> None:
    """
    Entry point for the 'eval_dates' block (pure plotting).
    """
    # fallback to a few dates if none provided
    if not dates:
        try:
            all_dates = list(resolver.list_dates())
            dates = all_dates[:8]
            logger.info("[eval_dates] Using first %d dates: %s", len(dates), dates)
        except Exception:
            logger.warning("[eval_dates] No dates provided and resolver.list_dates() failed.")
            return

    plot_dates_montages(
        resolver=resolver,
        out_root=out_root,
        dates=list(dates),
        include_lr=bool(include_lr),
        include_members=bool(include_members),
        n_members=int(n_members),
        cmap=str(cmap),
        percentile=float(percentile),
        land_only=bool(land_only if hasattr(eval_cfg, "eval_land_only") else True),
    )