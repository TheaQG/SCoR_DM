from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import shutil
import logging
import numpy as np
import torch

from scor_dm.utils import get_model_string
from evaluate.evaluation import EvaluationConfig, EvaluationRunner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions to enumerate sampler grid, matching generation logic
def _as_float_list(x, fallback: float | None = None):
    """Normalize config entry to a list[float]."""
    if x is None:
        return [] if fallback is None else [float(fallback)]
    if isinstance(x, (float, int)):
        return [float(x)]
    return [float(v) for v in x]

def _enumerate_sampler_combos(rho_grid, S_churn_grid, sigma_scale_grid):
    """
    Enumerate all (rho, S_churn, sigma_scale) combinations with a flat index.
    Returns a list of dicts: [{"idx": i, "rho": ..., "S_churn": ..., "sigma_scale": ...}, ...]
    """
    combos = []
    idx = 0
    for rho in rho_grid:
        for S_churn in S_churn_grid:
            for sigma_scale in sigma_scale_grid:
                combos.append(
                    {
                        "idx": idx,
                        "rho": float(rho),
                        "S_churn": float(S_churn),
                        "sigma_scale": float(sigma_scale),
                    }
                )
                idx += 1
    return combos


def _sampler_gen_root(cfg) -> Path:
    """
    Root directory for sampler-grid *generated* samples:
        <paths.sample_dir>/generation/<model_name>/sampler_grid/
    """
    model_str = get_model_string(cfg)
    sample_root = cfg["paths"]["sample_dir"]
    return Path(sample_root) / "generation" / model_str / "sampler_grid"


def _sampler_eval_root(cfg) -> Path:
    """
    Root directory for sampler-grid *evaluation* outputs:
        <paths.sample_dir>/evaluation/<model_name>/sampler_grid/
    """
    model_str = get_model_string(cfg)
    sample_root = cfg["paths"]["sample_dir"]
    return Path(sample_root) / "evaluation" / model_str / "sampler_grid"


def _build_eval_cfg_for_combo(cfg, gen_dir: Path, eval_dir: Path) -> EvaluationConfig:
    """
    Build an EvaluationConfig for a single sampler-grid combo directory.
    This mirrors evaluation_main.py, but with explicit gen_dir / eval_dir.
    """
    fe: Dict[str, Any] = cfg.get("full_gen_eval", {})

    # Parse baselines overlay block if present
    bo = fe.get("baselines_overlay", {})
    baselines_overlay = {
        "enabled": bool(bo.get("enabled", False)),
        "types": tuple(bo.get("types", ())),
        "split": str(bo.get("split", "test")),
        "labels": dict(bo.get("labels", {})),
        "styles": dict(bo.get("styles", {})),
        "sample_root": str(cfg["paths"]["sample_dir"]),
    }

    return EvaluationConfig(
        gen_dir=str(gen_dir),
        out_dir=str(eval_dir),

        eval_land_only=bool(fe.get("eval_land_only", False)),
        prefer_phys=bool(fe.get("prefer_phys", True)),
        lr_key=str(fe.get("lr_key", "lr")),
        region_mask_path=fe.get("region_mask_path", None),
        grid_km_per_px=float(fe.get("grid_km_per_px", 2.5)),
        lr_grid_km_per_px=float(fe.get("lr_grid_km_per_px", 31.0)),
        seasons=tuple(fe.get("seasons", ("ALL", "DJF", "MAM", "JJA", "SON"))),

        hr_dx_km=float(fe.get("hr_dx_km", fe.get("grid_km_per_px", 2.5))),
        lr_dx_km=float(fe.get("lr_dx_km", fe.get("lr_grid_km_per_px", 31.0))),

        baselines_overlay=baselines_overlay,

        crps_examples_n_members=int(fe.get("crps_examples_n_members", 4)),

        # FSS / ISS / thresholds
        thresholds_mm=tuple(fe.get("thresholds_mm", (1.0, 5.0, 10.0))),
        fss_scales_km=tuple(fe.get("fss_scales_km", (5, 10, 20))),
        fss_thresholds_mm=tuple(
            fe.get("fss_thresholds_mm", fe.get("thresholds_mm", (1.0, 5.0, 10.0)))
        ),
        compute_lr_fss=bool(fe.get("compute_lr_fss", True)),

        iss_thresholds_mm=tuple(fe.get("iss_thresholds_mm", fe.get("thresholds_mm", (1.0, 5.0, 10.0)))),
        iss_scales_km=tuple(fe.get("iss_scales_km", (5, 10, 20))),
        compute_lr_iss=bool(fe.get("compute_lr_iss", True)),

        # Reliability / PIT / spread–skill
        reliability_bins=int(fe.get("reliability_bins", 10)),
        spread_skill_bins=int(fe.get("spread_skill_bins", 10)),
        pit_bins=int(fe.get("pit_bins", 20)),

        # Variogram for scale metrics
        variogram_p=float(fe.get("variogram_p", 0.5)),
        variogram_max_pairs=int(fe.get("variogram_max_pairs", 5000)),

        low_k_max=float(fe.get("low_k_max", 1.0 / 200.0)),
        high_k_min=float(fe.get("high_k_min", 1.0 / 20.0)),
        make_plots=bool(fe.get("make_plots", True)),

        # Distributional config
        dist_n_bins=int(fe.get("pixel_dist_n_bins", 80)),
        dist_vmax_percentile=float(fe.get("pixel_dist_vmax_percentile", 99.5)),
        dist_include_lr=bool(fe.get("pixel_dist_include_lr", True)),
        dist_save_cap=int(fe.get("pixel_dist_save_cap", 200_000)),

        # Extremes (you may or may not use these in sampler grid tasks)
        ext_agg_kind=str(fe.get("ext_agg_kind", "mean")),
        ext_rxk_days=tuple(fe.get("ext_rxk_days", (1, 5))),
        ext_gev_rps_years=tuple(fe.get("ext_gev_rps_years", (2, 5, 10, 20, 50))),
        ext_blocks_per_year=float(fe.get("ext_blocks_per_year", 1.0)),
        ext_pot_thr_kind=str(fe.get("ext_pot_thr_kind", "hr_quantile")),
        ext_pot_thr_val=float(fe.get("ext_pot_thr_val", 0.95)),
        ext_pot_rps_years=tuple(fe.get("ext_pot_rps_years", (2, 5, 10, 20, 50))),
        ext_days_per_year=float(fe.get("ext_days_per_year", 365.25)),
        ext_wet_threshold_mm=float(fe.get("ext_wet_threshold_mm", 1.0)),
        include_lr=bool(fe.get("ext_include_lr", False)),
        ext_tails_basis=str(fe.get("ext_tails_basis", "pooled_pixels")),

        # Spatial
        spatial_corr_kinds=tuple(fe.get("spatial_corr_kinds", ("pearson", "spearman"))),
        spatial_deseasonalize=bool(fe.get("spatial_deseasonalize", True)),
        spatial_vmin=float(fe.get("spatial_vmin", None)) if fe.get("spatial_vmin", None) is not None else None, # type: ignore
        spatial_vmax=float(fe.get("spatial_vmax", None)) if fe.get("spatial_vmax", None) is not None else None, # type: ignore
        spatial_show_diff=bool(fe.get("spatial_show_diff", True)),
        spatial_include_gen=bool(fe.get("spatial_include_gen", False)),
        spatial_include_ens=bool(fe.get("spatial_include_ens", True)),
        spatial_include_hr=bool(fe.get("spatial_include_hr", True)),
        spatial_include_lr=bool(fe.get("spatial_include_lr", True)),

        # Temporal
        temporal_include_lr=bool(fe.get("temporal_include_lr", True)),
        temporal_wet_thr_mm=float(fe.get("temporal_wet_thr_mm", 1.0)),
        temporal_max_lag=int(fe.get("temporal_max_lag", 30)),
        temporal_max_spell=int(fe.get("temporal_max_spell", 25)),
        temporal_group_by=str(fe.get("temporal_group_by", "year")),
        temporal_ensemble_pool_mode=str(fe.get("temporal_ensemble_pool_mode", "member_mean")),

        # Dates / per-date plotting
        dates_list=fe.get("dates_list", []),
        dates_include_lr=bool(fe.get("dates_include_lr", True)),
        dates_include_members=bool(fe.get("dates_include_members", True)),
        dates_n_members=int(fe.get("dates_n_members", 3)),
        dates_cmap=str(fe.get("dates_cmap", "Blues")),
        dates_percentile=float(fe.get("dates_percentile", 99.5)),

        # Ensemble usage
        use_ensemble=bool(fe.get("use_ensemble", True)),
        ensemble_n_members=fe.get("ensemble_n_members", None),
        ensemble_member_seed=int(fe.get("ensemble_member_seed", 1234)),
        ensemble_reduction_fallback=str(fe.get("ensemble_reduction_fallback", "pmm")),
        ensemble_cache_members=bool(fe.get("ensemble_cache_members", False)),
        dist_ensemble_pool_mode=str(fe.get("dist_ensemble_pool_mode", "pool")),

        # SAL / features
        sal_structure_mode=str(fe.get("sal_structure_mode", "object")),
        sal_threshold_kind=str(fe.get("sal_threshold_kind", "quantile")),
        sal_threshold_value=float(fe.get("sal_threshold_value", 0.90)),
        sal_connectivity=int(fe.get("sal_connectivity", 8)),
        sal_min_area_px=int(fe.get("sal_min_area_px", 9)),
        sal_smooth_sigma=(float(_tmp) if (_tmp := fe.get("sal_smooth_sigma", 0.75)) is not None else None),
        sal_peakedness_mode=str(fe.get("sal_peakedness_mode", "largest")),
    )


def evaluate_sampler_grid_main(cfg):
    """
    Loop over all sampler-grid generation subdirectories and run a (possibly simplified)
    evaluation suite for each.

    Expected directory layout:
      generation/<model_name>/sampler_grid/rho=.._Schurn=.._sigscale=../
          ensembles_phys/
          pmm_phys/
          lr_hr_phys/
          lsm/
          meta/land_mask.npz
    """
    fe: Dict[str, Any] = cfg.get("full_gen_eval", {})

    # Whether to delete generated sampler-grid files after evaluation
    sampler_cleanup = bool(fe.get("sampler_cleanup", False))

    # ------ RNG seeding, like evaluation_main ------
    seed = int(fe.get("seed", 1234))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(cfg["training"]["device"])

    # ------ Root dirs ------
    gen_root = _sampler_gen_root(cfg)
    eval_root = _sampler_eval_root(cfg)
    eval_root.mkdir(parents=True, exist_ok=True)

    logger.info(f"[evaluate_sampler_grid_main] sampler gen_root: {gen_root}")
    logger.info(f"[evaluate_sampler_grid_main] sampler eval_root: {eval_root}")

    if not gen_root.exists():
        raise FileNotFoundError(f"Sampler grid generation root does not exist: {gen_root}")

    # Which tasks to run per combo?
    default_tasks = ["prcp_probabilistic", "prcp_scale", "prcp_distributional"]
    sampler_tasks = fe.get("sampler_eval_tasks", default_tasks)

    logger.info(f"[evaluate_sampler_grid_main] Using tasks for sampler grid: {sampler_tasks}")

    # ------ Discover sampler-grid combo dirs currently on disk ------
    combo_dirs = sorted([d for d in gen_root.iterdir() if d.is_dir()])
    n_found = len(combo_dirs)
    logger.info(f"[evaluate_sampler_grid_main] Found {n_found} sampler combos on disk.")

    # ------ Reconstruct full sampler grid to align indices with generation ------
    sampler_grid = fe.get("sampler_grid", {})
    edm_cfg = cfg.get("edm", {})

    rho_grid = _as_float_list(
        sampler_grid.get("rho", None),
        fallback=edm_cfg.get("rho", 7.0),
    )
    S_churn_grid = _as_float_list(
        sampler_grid.get("S_churn", None),
        fallback=edm_cfg.get("S_churn", 0.0),
    )
    sigma_scale_grid = _as_float_list(
        sampler_grid.get("sigma_scale", None),
        fallback=1.0,
    )

    all_combos = _enumerate_sampler_combos(rho_grid, S_churn_grid, sigma_scale_grid)
    n_all_combos = len(all_combos)
    logger.info(
        "[evaluate_sampler_grid_main] Sampler grid size (rho × S_churn × sigma_scale) = %d",
        n_all_combos,
    )

    # Optional slicing of combos for parallel evaluation via environment variables
    start_idx_env = os.environ.get("SAMPLER_COMBO_INDEX_START", None)
    end_idx_env = os.environ.get("SAMPLER_COMBO_INDEX_END", None)

    if start_idx_env is not None or end_idx_env is not None:
        start_idx = int(start_idx_env) if start_idx_env is not None else 0
        end_idx = int(end_idx_env) if end_idx_env is not None else (n_all_combos - 1)

        selected_combo_meta = [
            c for c in all_combos if start_idx <= c["idx"] <= end_idx
        ]

        selected_combo_dirs = []
        for c in selected_combo_meta:
            rho = c["rho"]
            S_churn = c["S_churn"]
            sigma_scale = c["sigma_scale"]
            combo_name = f"rho={rho:.2f}_Schurn={S_churn:.2f}_sigscale={sigma_scale:.2f}"
            combo_dir = gen_root / combo_name
            if combo_dir.exists():
                selected_combo_dirs.append(combo_dir)
            else:
                logger.warning(
                    "[evaluate_sampler_grid_main] Requested combo idx=%d (%s) not found on disk at %s",
                    c["idx"],
                    combo_name,
                    combo_dir,
                )

        logger.info(
            "[evaluate_sampler_grid_main] Restricting combos via env: "
            "SAMPLER_COMBO_INDEX_START=%s, SAMPLER_COMBO_INDEX_END=%s -> %d / %d combos on disk (grid size %d)",
            start_idx_env,
            end_idx_env,
            len(selected_combo_dirs),
            n_found,
            n_all_combos,
        )
    else:
        selected_combo_dirs = combo_dirs
        logger.info(
            "[evaluate_sampler_grid_main] Evaluating all %d sampler combos on disk (grid size %d).",
            len(selected_combo_dirs),
            n_all_combos,
        )

    for combo_dir in selected_combo_dirs:
        combo_name = combo_dir.name
        combo_eval_dir = eval_root / combo_name
        combo_eval_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[evaluate_sampler_grid_main] Evaluating combo '{combo_name}'")
        logger.info(f"  -> gen_dir:  {combo_dir}")
        logger.info(f"  -> eval_dir: {combo_eval_dir}")

        # Build eval config for this combo
        ev_cfg = _build_eval_cfg_for_combo(cfg, combo_dir, combo_eval_dir)

        runner = EvaluationRunner(
            cfg_yaml=cfg,
            eval_cfg=ev_cfg,
            device=device,
            baseline_eval_dirs=None,
            plot_only=bool(fe.get("plot_only", False)),
        )

        runner.run(tasks=list(sampler_tasks))

        # Optional cleanup: delete generated files for this combo after evaluation
        if sampler_cleanup:
            try:
                logger.info(
                    "[evaluate_sampler_grid_main] sampler_cleanup=True, removing generated sampler dir: %s",
                    combo_dir,
                )
                shutil.rmtree(combo_dir)
            except Exception as e:
                logger.warning(
                    "[evaluate_sampler_grid_main] Failed to remove generated sampler dir %s: %s",
                    combo_dir,
                    e,
                )

    logger.info(f"[evaluate_sampler_grid_main] Done. Outputs at: {eval_root}")

    return eval_root