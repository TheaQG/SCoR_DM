from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import json
import logging

from evaluate.data_resolver import EvalDataResolver
from evaluate.evaluate_prcp.eval_probabilistic.evaluate_probabilistic import run_probabilistic
from evaluate.evaluate_prcp.eval_scale.evaluate_scale import run_scale
from evaluate.evaluate_prcp.eval_distributions.evaluate_distributions import run_distributional
from evaluate.evaluate_prcp.eval_extremes.evaluate_extremes import run_extremes
from evaluate.evaluate_prcp.eval_spatial.evaluate_spatial import run_spatial
from evaluate.evaluate_prcp.eval_temporal.evaluate_temporal import run_temporal
from evaluate.evaluate_prcp.eval_features.evaluate_features import run_features
from evaluate.evaluate_prcp.eval_dates.evaluate_dates import run_dates

logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    gen_dir: str
    out_dir: str
    eval_land_only: bool = True
    prefer_phys: bool = True
    region_mask_path: Optional[str] = None
    lr_key: Optional[str] = "lr" # which LR key to use from lr_hr_phys: "lr" | "lr_lrspace" | "lr_hrspace"
    grid_km_per_px: float = 2.5
    lr_grid_km_per_px: float = 31.0
    
    # New field for baselines overlay
    baselines_overlay: Optional[Dict[str, Any]] = None
    
    crps_examples_n_members: int = 4
    thresholds_mm: tuple = (1.0, 5.0, 10.0)
    fss_thresholds_mm: tuple = (1.0, 5.0, 10.0)
    fss_scales_km: tuple = (5, 10, 20)
    iss_thresholds_mm: tuple = (1.0, 5.0, 10.0)
    iss_scales_km: tuple = (5, 10, 20)
    compute_lr_iss: bool = True
    seasons: tuple = ("ALL", "DJF", "MAM", "JJA", "SON")
    reliability_bins: int = 10
    spread_skill_bins: int = 10
    variogram_p: float = 0.5
    variogram_max_pairs: int = 5000
    pit_bins: int = 20
    hr_dx_km: float = 2.5
    lr_dx_km: float = 31.0
    low_k_max: float = 1.0 / 200.0
    high_k_min: float = 1.0 / 20.0
    compute_lr_fss: bool = True
    make_plots: bool = True
    # Distributional evaluation config
    dist_n_bins: int = 80
    dist_vmax_percentile: float = 99.5
    dist_include_lr: bool = True
    dist_save_cap: int = 200_000
    # Extremes evaluation config
    ext_agg_kind: str = "mean"                 # "mean" or "sum"
    ext_rxk_days: tuple = (1, 5)
    ext_gev_rps_years: tuple = (2, 5, 10, 20, 50)
    ext_blocks_per_year: float = 1.0            # 4.0 for seasonal blocks
    ext_pot_thr_kind: str = "hr_quantile"          # "quantile" | "value" | "hr_quantile"
    ext_pot_thr_val: float = 0.95               # if kind==quantile -> quantile; else absolute mm/day
    ext_pot_rps_years: tuple = (2, 5, 10, 20, 50)
    ext_days_per_year: float = 365.25
    ext_wet_threshold_mm: float = 1.0
    include_lr: bool = False
    ext_tails_basis: str = "pooled_pixels" # "pooled_pixels" | "domain_series"
    # Spatial evaluation config
    spatial_corr_kinds: tuple = ("pearson", "spearman")
    spatial_deseasonalize: bool = True
    spatial_vmin: Optional[float] = None
    spatial_vmax: Optional[float] = None
    spatial_show_diff: bool = True
    spatial_include_gen: bool = False
    spatial_include_ens: bool = True
    spatial_include_hr: bool = True
    spatial_include_lr: bool = True
    # temporal evaluation config
    temporal_include_lr: bool = True
    temporal_wet_thr_mm: float = 1.0
    temporal_max_lag: int = 30
    temporal_max_spell: int = 25
    temporal_group_by: str = "year"
    temporal_ensemble_pool_mode: str = "member_mean"   # or "pool"
    # per-date evaluation config
    dates_list: Optional[List[str]] = None
    dates_include_lr: bool = True
    dates_include_members: bool = True
    dates_n_members: int = 3
    dates_cmap: str = "Blues"
    dates_percentile: float = 99.5
    # Ensemble evaluation config
    use_ensemble: bool = True
    ensemble_n_members: Optional[int] = None
    ensemble_member_seed: int = 1234
    ensemble_reduction_fallback: str = "pmm"  # "pmm" | "ens_mean" | "pmm_then_metric"
    ensemble_cache_members: bool = False
    # Distributional ensemble pooling mode
    dist_ensemble_pool_mode: str = "pool"   # "pool" | "member_mean" | "pmm"
    # Features evaluation config
    sal_structure_mode: str = "object"     # "object" | "std_proxy"
    sal_threshold_kind: str = "quantile"   # "quantile" | "absolute"
    sal_threshold_value: float = 0.90
    sal_connectivity: int = 8
    sal_min_area_px: int = 9
    sal_smooth_sigma: Optional[float] = 0.75
    sal_peakedness_mode: str = "largest"  # "largest" | "herfindahl"
class EvaluationRunner:
    """
        Clean runner that uses EvalDataResolver to perform evaluations.
    """

    def __init__(
            self, 
            cfg_yaml: dict,
            eval_cfg: EvaluationConfig,
            device: torch.device,
            baseline_eval_dirs: Optional[Dict[str, str]] = None,
            plot_only: bool = False
    ):
        self.cfg_yaml = cfg_yaml
        self.eval_cfg = eval_cfg
        self.device = device
        self.baseline_eval_dirs = baseline_eval_dirs
        self.plot_only = plot_only

        # Data access
        self.data = EvalDataResolver(
            gen_root=eval_cfg.gen_dir,
            eval_land_only=eval_cfg.eval_land_only,
            roi_mask_path=eval_cfg.region_mask_path,
            prefer_phys=eval_cfg.prefer_phys,
            lr_phys_key=eval_cfg.lr_key
        )

        # Output setup
        self.out_root = Path(eval_cfg.out_dir)
        (self.out_root / "tables").mkdir(parents=True, exist_ok=True)
        (self.out_root / "figures").mkdir(parents=True, exist_ok=True)

    def should_compute(self, rel_table_name: str) -> bool:
        """
            Return False when running in plot_only mode and the table already exists.
        """
        table_path = self.out_root / "tables" / rel_table_name
        if self.plot_only and table_path.exists():
            logger.info(f"[EvaluationRunner] plot_only=True and table {table_path} exists; skipping computation.")
            return False
        return True


    def run(self, tasks: Optional[List[str]] = None):
        """
            Run the evaluation tasks specified.

            Args:
                tasks: List of task names to run. If None, run all available tasks.
        """
        dates = self.data.list_dates()
        logger.info(f"[EvaluationRunner] Found {len(dates)} dates for evaluation.")

        manifest = {
            "gen_dir": self.eval_cfg.gen_dir,
            "out_dir": self.eval_cfg.out_dir,
            "n_dates": len(dates),
            "thresholds_mm": list(self.eval_cfg.thresholds_mm),
            "fss_scales_km": list(self.eval_cfg.fss_scales_km),
            "seasons": list(self.eval_cfg.seasons),
            "use_ensemble": bool(self.eval_cfg.use_ensemble),
            "ensemble_n_members": self.eval_cfg.ensemble_n_members,            
        }
        # Before writing manifest.json, check should_compute
        if self.should_compute("manifest.json"):
            (self.out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
        else:
            logger.info(f"[EvaluationRunner] Skipping manifest.json write (plot_only and file exists).")

        logger.info(f"[EvaluationRunner] Starting evaluation with tasks: {tasks if tasks is not None else 'all available tasks'}")
        # ===== 
        # Dispatch to evaluation tasks
        # =====
        if tasks is None:
            # Sensible default: run prcipitation probabilistic evaluation
            tasks = ["prcp_probabilistic", "prcp_scale", "prcp_distributional", "prcp_extremes"]
        for task in tasks:
            # Normalize and log each task
            task_norm = str(task).strip().lower()
            logger.info(f"[EvaluationRunner] Dispatching task: '{task}' (normalized: '{task_norm}')")
            # 1) Precipitation probabilistic evaluation
            if task_norm in ("prcp_probabilistic", "prcp_prob", "prob", "probabilistic"):
                out_dir = self.out_root / "prcp" / "probabilistic"
                out_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"[EvaluationRunner] Running task '{task}' -> {out_dir}")
                
                run_probabilistic(
                    resolver=self.data,
                    eval_cfg=self.eval_cfg,
                    out_root=out_dir,
                    plot_only=self.plot_only,
                )
                continue
            
            # 2) Precipitation scale evaluation
            if task_norm in ("prcp_scale", "scale", "prcp_psd", "scale_dependent"):
                out_dir = self.out_root / "prcp" / "scale"
                out_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"[EvaluationRunner] Running task '{task}' -> {out_dir}")
                
                # build a light eval cfg-like object for scale code
                class _ScaleCfg:
                    hr_dx_km: float
                    lr_dx_km: float
                    fss_thresholds_mm: tuple
                    fss_scales_km: tuple
                    compute_lr_fss: bool
                    low_k_max: float
                    high_k_min: float
                    make_plots: bool
                    # ensemble flags for scale
                    use_ensemble: bool
                    ensemble_n_members: int | None
                    ensemble_member_seed: int
                    baselines_overlay: Any | None

                sc = _ScaleCfg()
                sc.hr_dx_km = float(self.eval_cfg.hr_dx_km or self.eval_cfg.grid_km_per_px)
                sc.lr_dx_km = float(self.eval_cfg.lr_dx_km or self.eval_cfg.lr_grid_km_per_px)
                sc.fss_thresholds_mm = tuple(self.eval_cfg.fss_thresholds_mm)
                sc.fss_scales_km = tuple(self.eval_cfg.fss_scales_km)
                sc.compute_lr_fss = bool(self.eval_cfg.compute_lr_fss)
                sc.low_k_max = float(self.eval_cfg.low_k_max)
                sc.high_k_min = float(self.eval_cfg.high_k_min)
                sc.make_plots = bool(self.eval_cfg.make_plots)
                sc.use_ensemble = bool(self.eval_cfg.use_ensemble)
                sc.ensemble_n_members = self.eval_cfg.ensemble_n_members
                sc.ensemble_member_seed = int(self.eval_cfg.ensemble_member_seed)
                sc.baselines_overlay = getattr(self.eval_cfg, "baselines_overlay", None)

                run_scale(
                    resolver=self.data,
                    eval_cfg=sc,
                    out_root=out_dir,
                    plot_only=self.plot_only,
                )
                continue

            # 3) Precipitation distributional evaluation
            if task_norm in ("prcp_distributional", "prcp_dist", "distributional", "dist"):
                out_dir = self.out_root / "prcp" / "distributional"
                out_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"[EvaluationRunner] Running task '{task}' -> {out_dir}")

                run_distributional(
                    resolver=self.data,
                    eval_cfg=self.eval_cfg,
                    out_root=out_dir,
                    plot_only=self.plot_only,
                )
                continue

            # 4) Precipitation extremes evaluation
            if task_norm in ("prcp_extremes", "prcp_ext", "extremes", "ext"):
                out_dir = self.out_root / "prcp" / "extremes"
                out_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"[EvaluationRunner] Running task '{task}' -> {out_dir}")

                run_extremes(
                    resolver=self.data,
                    eval_cfg=self.eval_cfg,
                    out_root=out_dir,
                    plot_only=self.plot_only,
                )
                continue
            # 5) Precipitation spatial structure evaluation
            if task_norm in ("prcp_spatial", "spatial", "spatial_maps"):
                out_dir = self.out_root / "prcp" / "spatial"
                out_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"[EvaluationRunner] Running task '{task}' -> {out_dir}")

                run_spatial(
                    resolver=self.data,
                    eval_cfg=self.eval_cfg,
                    out_root=out_dir,
                )
                continue


            # 6) Precipitation temporal evaluation
            if task_norm in ("prcp_temporal", "temporal", "time", "timeseries"):
                out_dir = self.out_root / "prcp" / "temporal"
                out_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"[EvaluationRunner] Running task '{task}' -> {out_dir}")

                run_temporal(
                    resolver=self.data,
                    eval_cfg=self.eval_cfg,
                    out_root=out_dir,
                    group_by=getattr(self.eval_cfg, "temporal_group_by", "year"),
                    seasons=getattr(self.eval_cfg, "seasons", ("ALL","DJF","MAM","JJA","SON")),
                    make_plots=bool(getattr(self.eval_cfg, "make_plots", True)),
                )
                continue

            # 7) Precipitation feature/object-based evaluation (SAL)
            if task_norm in ("prcp_features", "features", "objects", "sal"):
                out_dir = self.out_root / "prcp" / "features"
                out_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"[EvaluationRunner] Running task '{task}' -> {out_dir}")

                run_features(
                    resolver=self.data,
                    eval_cfg=self.eval_cfg,
                    out_root=out_dir,
                )
                continue

            # 8) Precipitation per-date evaluation (pure plotting)
            if task_norm in ("prcp_dates", "dates", "per_date", "eval_dates"):
                out_dir = self.out_root / "prcp" / "dates"
                out_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"[EvaluationRunner] Running task '{task}' -> {out_dir}")

                run_dates(
                    resolver=self.data,
                    eval_cfg=self.eval_cfg,
                    out_root=out_dir,
                    dates=getattr(self.eval_cfg, "dates_list", None),
                    include_lr=bool(getattr(self.eval_cfg, "dates_include_lr", True)),
                    include_members=bool(getattr(self.eval_cfg, "dates_include_members", True)),
                    n_members=int(getattr(self.eval_cfg, "dates_n_members", 3)),
                    cmap=str(getattr(self.eval_cfg, "dates_cmap", "Blues")),
                    percentile=float(getattr(self.eval_cfg, "dates_percentile", 99.5)),
                    land_only=bool(getattr(self.eval_cfg, "eval_land_only", True)),
                )
                continue

            logger.warning(f"[EvaluationRunner] Unknown task '{task}' (normalize '{task_norm}); skipping.")                
        # To be implemented: calls to...
        # if "scale" in tasks: evaluate_scale.run(...)
        # ... etc.
