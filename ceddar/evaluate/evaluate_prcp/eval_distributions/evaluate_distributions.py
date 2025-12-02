# sbgm/evaluate/evaluate_prcp/eval_distributional/evaluate_distributional.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
import numpy as np
import torch

from evaluate.evaluate_prcp.eval_distributions.metrics_distributions import (
    collect_pooled_distributions,
    collect_daily_histograms,
    compute_distributional_metrics,
    collect_ensemble_histograms
)
from evaluate.evaluate_prcp.eval_distributions.plot_distributions import (
    plot_distributional,
)

logger = logging.getLogger(__name__)


def run_distributional(
    resolver,
    eval_cfg,
    out_root: str | Path,
    *,
    plot_only: bool = False,
) -> None:
    """
    Distributional (1D, pooled-pixel) evaluation.

    Inputs via resolver (same style as for scale):
        - list_dates()
        - load_obs(date)   -> HR [H,W]
        - load_pmm(date)   -> GEN/PMM [H,W]
        - load_lr(date)    -> LR [H,W]     (optional)
        - load_mask(date)  -> mask [H,W]   (optional)

    Outputs:
        <out_root>/tables/dist_bins.csv
        <out_root>/tables/dist_hr.csv
        <out_root>/tables/dist_gen.csv
        <out_root>/tables/dist_lr.csv  (if LR available)
        <out_root>/tables/dist_metrics.csv  (KL/KS/W1 vs HR)
    """
    out_root = Path(out_root)
    tables_dir = out_root / "tables"
    figs_dir = out_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    if plot_only:
        plot_distributional(out_root, eval_cfg=eval_cfg)
        logger.info("[eval_distributional] plot_only=True – done plotting.")
        return

    dates = list(resolver.list_dates())
    if not dates:
        logger.warning("[eval_distributional] No dates from resolver – nothing to do.")
        return
    logger.info(f"[eval_distributional] Running on {len(dates)} dates.")

    # knobs
    n_bins: int = int(getattr(eval_cfg, "dist_n_bins", 80))
    vmax_pct: float = float(getattr(eval_cfg, "dist_vmax_percentile", 99.5))
    include_lr: bool = bool(getattr(eval_cfg, "dist_include_lr", True))
    save_cap: int = int(getattr(eval_cfg, "dist_save_cap", 200_000))

    pooled = collect_pooled_distributions(
        resolver=resolver,
        dates=dates,
        include_lr=include_lr,
        n_bins=n_bins,
        vmax_percentile=vmax_pct,
        save_samples_cap=save_cap,
    )

    # write CSVs
    bins = pooled["bins"]  # [B+1]
    np.savetxt(tables_dir / "dist_bins.csv", bins, delimiter=",", header="bin_edge", comments="")

    def _write_series(name: str, arr: np.ndarray):
        p = tables_dir / f"dist_{name}.csv"
        with open(p, "w") as f:
            f.write("bin_idx,count\n")
            for i, c in enumerate(arr.astype(int)):
                f.write(f"{i},{int(c)}\n")

    _write_series("hr", pooled["hr_hist"])
    _write_series("gen", pooled["gen_hist"])
    if pooled.get("lr_hist") is not None:
        _write_series("lr", pooled["lr_hist"])

    # daily histograms for CI shading and alternative (day-weighted) views
    try:
        daily = collect_daily_histograms(
            resolver=resolver,
            dates=dates,
            include_lr=include_lr,
            bins=bins,
        )
        np.savez_compressed(
            tables_dir / "dist_daily.npz",
            **daily
        )
    except Exception as e:
        logger.warning(f"[eval_distributional] Could not build daily histograms: {e}")

    # Ensemble-native histograms (optional)
    ens_out: Dict[str, Any] | None = None
    try:
        if bool(getattr(eval_cfg, "use_ensemble", False)):
            mode = str(getattr(eval_cfg, "dist_ensemble_pool_mode", "pool"))
            ens_out = collect_ensemble_histograms(
                resolver=resolver,
                dates=dates,
                bins=bins,
                mode=mode,
                n_members=getattr(eval_cfg, "ensemble_n_members", None),
                seed=int(getattr(eval_cfg, "ensemble_member_seed", 1234)),
            )
            if ens_out:
                if mode == "pool" and "counts_pool" in ens_out:
                    p = tables_dir / "dist_gen_ens_pool.csv"
                    with open(p, "w") as f:
                        f.write("bin_idx,count\n")
                        for i, c in enumerate(ens_out["counts_pool" ].astype(int)):
                            f.write(f"{i},{int(c)}\n")
                if mode == "member_mean" and "pdf_mean" in ens_out:
                    p = tables_dir / "dist_gen_ens_mean.csv"
                    with open(p, "w") as f:
                        f.write("bin_idx,pdf\n")
                        for i, v in enumerate(ens_out["pdf_mean" ].astype(float)):
                            f.write(f"{i},{float(v)}\n")
                # Save extra arrays for optional plotting of spread
                np.savez_compressed(
                    tables_dir / "dist_member_histograms.npz",
                    **{k: v for k, v in ens_out.items() if k in ("bins","counts_members","n_members","pdf_mean","pdf_q10","pdf_q50","pdf_q90","mode")}
                )
    except Exception as e:
        logger.warning(f"[eval_distributional] Ensemble histogram build failed: {e}")

    # metrics (vs HR)
    metrics_rows = compute_distributional_metrics(pooled, ensembles=ens_out)
    with open(tables_dir / "dist_metrics.csv", "w") as f:
        f.write("ref,comp,wasserstein,ks_stat,ks_p,kl_hr_to_x\n")
        for r in metrics_rows:
            f.write("{ref},{comp},{wasserstein:.6f},{ks_stat:.6f},{ks_p:.6f},{kl_hr_to_x:.6f}\n"
                    .format(**{k: (v if v is not None else float("nan")) for k, v in r.items()}))

    # plots
    if bool(getattr(eval_cfg, "make_plots", True)):
        plot_distributional(out_root, eval_cfg=eval_cfg)
        logger.info(f"[eval_distributional] Plots saved to {figs_dir}")