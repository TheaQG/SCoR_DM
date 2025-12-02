from __future__ import annotations
from pathlib import Path
from typing import Sequence, Optional, Dict, Any, List

import numpy as np
import logging

from evaluate.evaluate_prcp.eval_extremes.metrics_extremes import (
    build_daily_series, seasonal_block_index, rxk_from_series,
    fit_gev_block_maxima_with_ci, fit_pot_gpd_with_ci,
    percentiles_and_wetfreq, SeriesBundle, pooled_pixel_percentiles_and_wetfreq,
    pooled_wet_hit_rate, build_daily_series_ensemble, pooled_pixel_percentiles_and_wetfreq_ens,
    pooled_wet_hit_rate_ens_member_stats,
    pooled_wet_hit_rate_ens, pooled_wet_hit_rate_ens_member_mean, percentiles_and_wetfreq_ens_member_mean
)
from evaluate.evaluate_prcp.eval_extremes.plot_extremes import plot_extremes

logger = logging.getLogger(__name__)

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_extremes(
    resolver,
    eval_cfg,
    out_root: str | Path,
    *,
    plot_only: bool = False,
) -> None:
    """
    Extremes block:
      - build daily basin-mean series (HR, GEN, optional LR)
      - Rx1day, Rx5day block maxima -> GEV fit with CIs
      - POT/GPD above threshold (fixed value or quantile from HR/GEN/LR)
      - P95/P99 & wet-day freq
    """
    out_root = Path(out_root)
    tables = _ensure_dir(out_root / "tables")
    _ensure_dir(out_root / "figures")

    if plot_only:
        plot_extremes(out_root, eval_cfg=eval_cfg)
        logger.info("[extremes] plot_only=True – plotted.")
        return

    # ------------------ config knobs ------------------
    agg_kind: str = getattr(eval_cfg, "ext_agg_kind", "mean")                 # "mean" or "sum"
    rxks: Sequence[int] = getattr(eval_cfg, "ext_rxk_days", (1, 5))
    gev_rps: Sequence[float] = getattr(eval_cfg, "ext_gev_rps_years", (2, 5, 10, 20, 50))
    blocks_per_year: float = float(getattr(eval_cfg, "ext_blocks_per_year", 1.0))  # 4.0 if seasonal
    pot_thr_kind: str = getattr(eval_cfg, "ext_pot_thr_kind", "quantile")     # "quantile"|"value"
    pot_thr_val: float = float(getattr(eval_cfg, "ext_pot_thr_val", 0.95))    # q or absolute mm/day
    pot_rps: Sequence[float] = getattr(eval_cfg, "ext_pot_rps_years", (2, 5, 10, 20, 50))
    days_per_year: float = float(getattr(eval_cfg, "ext_days_per_year", 365.25))
    wet_thr: float = float(getattr(eval_cfg, "ext_wet_threshold_mm", 1.0))
    include_lr: bool = bool(getattr(eval_cfg, "include_lr", True))

    dates = list(resolver.list_dates())
    if not dates:
        logger.warning("[extremes] No dates from resolver.")
        return

    # ------------------ daily series ------------------
    sb: SeriesBundle = build_daily_series(
        resolver, dates, mask_hw=resolver.load_mask(dates[0]),
        agg=agg_kind, include_lr=include_lr
    )
    ens_sb = None
    if bool(getattr(eval_cfg, "use_ensemble", False)):
        ens_sb = build_daily_series_ensemble(
            resolver, dates, mask_hw=resolver.load_mask(dates[0]),
            agg=agg_kind,
            n_members=getattr(eval_cfg, "ensemble_n_members", None),
            seed=int(getattr(eval_cfg, "ensemble_member_seed", 1234)),
        )    
    logger.info(f"[extremes] Built daily series for {len(sb.dates)} days; "
                f"HR nans: {np.isnan(sb.hr).sum()}, GEN nans: {np.isnan(sb.gen).sum()}" +
                (f", LR nans: {np.isnan(sb.lr).sum()}" if sb.lr is not None else "")
               )

    np.savez_compressed(
        tables / "ext_daily_series.npz",
        dates=sb.dates,
        hr=sb.hr, gen=sb.gen,
        lr=(sb.lr if sb.lr is not None else np.array([])),
        agg_kind=agg_kind,
    )

    # assemble available series (order: HR -> GEN -> LR)
    series_list: List[tuple[str, np.ndarray]] = [("HR", sb.hr), ("GEN", sb.gen)]
    if sb.lr is not None:
        series_list.append(("LR", sb.lr))
    else:
        logger.info("[extremes] No LR series available; skipping LR extremes.")

    # # ------------------ RxK -> GEV ------------------
    # block_id = None
    # if blocks_per_year >= 1.5:  # treat as seasonal if ≥ ~2
    #     block_id = seasonal_block_index(sb.dates)

    # gev_rows = ["which,k_days,n_blocks," +
    #             ",".join([f"rl_{int(r)}y" for r in gev_rps]) + "," +
    #             ",".join([f"rl_{int(r)}y_lo" for r in gev_rps]) + "," +
    #             ",".join([f"rl_{int(r)}y_hi" for r in gev_rps])]
    # for which, series in series_list:
    #     for k in rxks:
    #         rx = rxk_from_series(series, k=k, block_id=block_id)
    #         try:
    #             fit = fit_gev_block_maxima_with_ci(
    #                 rx, rps_years=gev_rps,
    #                 blocks_per_year=(4.0 if block_id is not None else 1.0)
    #             )
    #             row = [which, str(int(k)), str(int(fit["n_blocks"]))] + \
    #                   [f"{v:.6f}" for v in fit["rl"]] + \
    #                   [f"{v:.6f}" for v in fit["rl_lo"]] + \
    #                   [f"{v:.6f}" for v in fit["rl_hi"]]
    #             gev_rows.append(",".join(row))
    #         except Exception as e:
    #             logger.warning(f"[extremes] GEV failed for {which} Rx{k}: {e}")
    # if ens_sb is not None:
    #     for k in rxks:
    #         # per-member block maxima -> fit -> aggregate return levels
    #         rls = []
    #         try:
    #             for mi in range(ens_sb.gen_members.shape[0]):
    #                 rx = rxk_from_series(ens_sb.gen_members[mi], k=k, block_id=block_id)
    #                 fit = fit_gev_block_maxima_with_ci(
    #                     rx, rps_years=gev_rps,
    #                     blocks_per_year=(4.0 if block_id is not None else 1.0),
    #                     n_boot=200,
    #                 )
    #                 rls.append(fit["rl"])  # [R]
    #             rls = np.stack(rls, axis=0)  # [M,R]
    #             rl_mean = np.nanmean(rls, axis=0)
    #             rl_lo  = np.nanpercentile(rls, 10, axis=0)
    #             rl_hi  = np.nanpercentile(rls, 90, axis=0)
    #             nb = int(np.median([len(rxk_from_series(ens_sb.gen_members[mi], k=k, block_id=block_id)) for mi in range(ens_sb.gen_members.shape[0])]))
    #             row = ["GEN_ENS", str(int(k)), str(nb)] + \
    #                   [f"{v:.6f}" for v in rl_mean] + \
    #                   [f"{v:.6f}" for v in rl_lo] + \
    #                   [f"{v:.6f}" for v in rl_hi]
    #             gev_rows.append(",".join(row))
    #         except Exception as e:
    #             logger.warning(f"[extremes] Ensemble GEV failed for Rx{k}: {e}")
    # (tables / "ext_rxk_gev.csv").write_text("\n".join(gev_rows))

    # # ------------------ POT/GPD ------------------
    # # --- choose POT threshold behavior ---
    # if pot_thr_kind == "hr_quantile":
    #     u_hr = float(np.nanpercentile(series_list[0][1], pot_thr_val * 100.0))  # HR is first
    # else:
    #     u_hr = None
    # pot_rows = ["which,u,xi,beta,k_exc,lambda_per_day," +
    #             ",".join([f"rl_{int(r)}y" for r in pot_rps]) + "," +
    #             ",".join([f"rl_{int(r)}y_lo" for r in pot_rps]) + "," +
    #             ",".join([f"rl_{int(r)}y_hi" for r in pot_rps])]
    # for which, series in series_list:
    #     if pot_thr_kind == "quantile":
    #         u = float(np.nanpercentile(series, pot_thr_val * 100.0))
    #     elif pot_thr_kind == "hr_quantile":
    #         if u_hr is None:
    #             raise ValueError("pot_thr_kind == 'hr_quantile' but u_hr was not set")
    #         u = float(u_hr)
    #     else:  # "value"
    #         u = float(pot_thr_val)

    #     try:
    #         fit = fit_pot_gpd_with_ci(
    #             series, threshold=u, rps_years=pot_rps, days_per_year=days_per_year
    #         )
    #         row = [which, f"{fit['u']:.6f}", f"{fit['xi']:.6f}", f"{fit['beta']:.6f}",
    #                str(int(fit["k_exc"])), f"{fit['lambda_per_day']:.8f}"] + \
    #               [f"{v:.6f}" for v in fit["rl"]] + \
    #               [f"{v:.6f}" for v in fit["rl_lo"]] + \
    #               [f"{v:.6f}" for v in fit["rl_hi"]]
    #         pot_rows.append(",".join(row))
    #     except Exception as e:
    #         logger.warning(f"[extremes] POT failed for {which}: {e}")
    # if ens_sb is not None:
    #     try:
    #         # choose u per-member if quantile, else fixed as configured or HR-quantile
    #         for_medians_u = []
    #         rls = []
    #         for mi in range(ens_sb.gen_members.shape[0]):
    #             series = ens_sb.gen_members[mi]
    #             if pot_thr_kind == "quantile":
    #                 u = float(np.nanpercentile(series, pot_thr_val * 100.0))
    #             elif pot_thr_kind == "hr_quantile":
    #                 u = float(u_hr) if 'u_hr' in locals() and u_hr is not None else float(np.nanpercentile(sb.hr, pot_thr_val * 100.0))
    #             else:
    #                 u = float(pot_thr_val)
    #             for_medians_u.append(u)
    #             fit = fit_pot_gpd_with_ci(series, threshold=u, rps_years=pot_rps, days_per_year=days_per_year, n_boot=200)
    #             rls.append(fit["rl"])  # [R]
    #         rls = np.stack(rls, axis=0)
    #         rl_mean = np.nanmean(rls, axis=0)
    #         rl_lo  = np.nanpercentile(rls, 10, axis=0)
    #         rl_hi  = np.nanpercentile(rls, 90, axis=0)
    #         u_med = float(np.nanmedian(np.array(for_medians_u)))
    #         row = ["GEN_ENS", f"{u_med:.6f}", "nan", "nan",  # xi,beta not defined for aggregated
    #                "nan", "nan"] + \
    #               [f"{v:.6f}" for v in rl_mean] + \
    #               [f"{v:.6f}" for v in rl_lo] + \
    #               [f"{v:.6f}" for v in rl_hi]
    #         pot_rows.append(",".join(row))
    #     except Exception as e:
    #         logger.warning(f"[extremes] Ensemble POT failed: {e}")
    # (tables / "ext_pot_gpd.csv").write_text("\n".join(pot_rows))

    tails_rows = ["which,P95,P99,wet_freq,wet_hit_rate,n_days"]
    basis = getattr(eval_cfg, "ext_tails_basis", "domain_series").lower()

    if basis == "pooled_pixels":
        pooled = pooled_pixel_percentiles_and_wetfreq(
            resolver, dates, mask_hw=resolver.load_mask(dates[0]),
            wet_thr=wet_thr, p_list=(95.0, 99.0), include_lr=include_lr)

        # Proper pooled wet-day hit-rate: average over all HR-wet pixels across all days
        hits = pooled_wet_hit_rate(
            resolver, dates, mask_hw=resolver.load_mask(dates[0]),
            wet_thr=wet_thr, include_lr=include_lr
        )

        for which in ["HR", "GEN", "LR"]:
            if which not in pooled:
                continue
            if which == "HR":
                wet_hit = 1.0
            elif which == "GEN":
                wet_hit = float(hits.get("GEN", float("nan")))
            else:  # LR
                wet_hit = float(hits.get("LR", float("nan")))
            tails_rows.append(",".join([
                which,
                f"{pooled[which]['P95']:.6f}",
                f"{pooled[which]['P99']:.6f}",
                f"{pooled[which]['wet_freq']:.6f}",
                f"{wet_hit:.6f}",
                str(int(pooled[which]['n_points']))
            ]))
        if ens_sb is not None:
            ens_pool = percentiles_and_wetfreq_ens_member_mean(
                resolver, dates, mask_hw=resolver.load_mask(dates[0]),
                wet_thr=wet_thr, p_list=(95.0, 99.0),
                n_members=getattr(eval_cfg, "ensemble_n_members", None),
                seed=int(getattr(eval_cfg, "ensemble_member_seed", 1234)),
            )
            ens_hit = pooled_wet_hit_rate_ens_member_mean(
                resolver, dates, mask_hw=resolver.load_mask(dates[0]),
                wet_thr=wet_thr,
                n_members=getattr(eval_cfg, "ensemble_n_members", None),
                seed=int(getattr(eval_cfg, "ensemble_member_seed", 1234)),
            )
            if ens_pool is not None and np.isfinite(ens_hit):
                tails_rows.append(",".join([
                    "GEN_ENS",
                    f"{ens_pool['P95']:.6f}", f"{ens_pool['P99']:.6f}",
                    f"{ens_pool['wet_freq']:.6f}", f"{ens_hit:.6f}", str(int(ens_pool['n_points']))
                ]))
                # Also compute std for hit-rate and persist stds for plotting error bars

                hit_mean, hit_std = pooled_wet_hit_rate_ens_member_stats(
                    resolver, dates, mask_hw=resolver.load_mask(dates[0]),
                    wet_thr=wet_thr,
                    n_members=getattr(eval_cfg, "ensemble_n_members", None),
                    seed=int(getattr(eval_cfg, "ensemble_member_seed", 1234)),
                )
                try:
                    np.savez_compressed(
                        tables / "ext_tails_ens_bands.npz",
                        P95_std=np.float64(ens_pool.get("P95_std", np.nan)),
                        P99_std=np.float64(ens_pool.get("P99_std", np.nan)),
                        wet_freq_std=np.float64(ens_pool.get("wet_freq_std", np.nan)),
                        hit_std=np.float64(hit_std),
                    )
                except Exception:
                    pass
    else:
        # current domain-series logic
        hr_is_wet = (series_list[0][1] > wet_thr).astype(bool)
        for which, series in series_list:
            t = percentiles_and_wetfreq(series, wet_thr=wet_thr, p_list=(95.0, 99.0))
            is_wet = (series > wet_thr).astype(bool)
            denom = hr_is_wet.sum()
            wet_hit = float(((is_wet & hr_is_wet).sum()) / denom) if denom > 0 else float("nan")
            tails_rows.append(",".join([
                which,
                f"{t['P95']:.6f}", f"{t['P99']:.6f}",
                f"{t['wet_freq']:.6f}", f"{wet_hit:.6f}", str(int(t['n_days']))
            ]))
    (tables / "ext_tails.csv").write_text("\n".join(tails_rows))

    # Save meta so plots can annotate the actual threshold and configuration
    meta_kwargs = {
        "wet_thr_mm": np.float64(wet_thr),
        "tails_basis": str(basis),
        "pot_thr_kind": str(pot_thr_kind),
        "pot_thr_val": np.float64(pot_thr_val),
        "agg_kind": str(agg_kind),
        "blocks_per_year": np.float64(blocks_per_year),
    }
    # If we computed an HR-quantile threshold, store it explicitly
    # try:
    #     if pot_thr_kind == "hr_quantile" and "u_hr" in locals() and u_hr is not None:
    #         meta_kwargs["pot_u_hr"] = np.float64(u_hr)
    # except Exception:
    #     pass
    np.savez_compressed(tables / "ext_meta.npz", **meta_kwargs)

    # ------------------ plots ------------------
    try:
        plot_extremes(out_root, eval_cfg=eval_cfg)
    except Exception as e:
        logger.warning(f"[extremes] Plotting failed: {e}")

    logger.info(f"[extremes] Done. Outputs at: {out_root}")
