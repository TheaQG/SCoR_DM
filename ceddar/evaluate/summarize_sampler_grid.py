from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional
import csv
import json
import math
import logging

logger = logging.getLogger(__name__)

@dataclass
class SamplerSummary:
    sampler_id: str
    rho: float
    Schurn: float
    sigma_scale: float

    # Distributional metrics (HR vs GEN / LR)
    dist_W1_GEN: Optional[float] = None
    dist_KS_GEN: Optional[float] = None
    dist_KL_GEN: Optional[float] = None
    dist_W1_LR: Optional[float] = None
    dist_KS_LR: Optional[float] = None
    dist_KL_LR: Optional[float] = None

    # Probabilistic (means from prob_summary.csv)
    prob_CRPS_mean: Optional[float] = None
    prob_PMM_MAE_mean: Optional[float] = None
    prob_EnergyScore_mean: Optional[float] = None
    prob_VariogramScore_mean: Optional[float] = None
    prob_PIT_KS_D: Optional[float] = None
    prob_RankHist_max_abs_z: Optional[float] = None
    prob_SpreadSkill_slope: Optional[float] = None
    prob_SpreadSkill_r: Optional[float] = None

    # Extremes (P95/P99, wet freq, hit rate; plus HR ratios)
    ext_P95_HR: Optional[float] = None
    ext_P99_HR: Optional[float] = None
    ext_wetfreq_HR: Optional[float] = None

    ext_P95_GEN: Optional[float] = None
    ext_P99_GEN: Optional[float] = None
    ext_wetfreq_GEN: Optional[float] = None
    ext_wethit_GEN: Optional[float] = None

    ext_P95_ratio_GEN_HR: Optional[float] = None
    ext_P99_ratio_GEN_HR: Optional[float] = None
    ext_wetfreq_ratio_GEN_HR: Optional[float] = None

    # Scale-dependent summary (from scale_overview.csv)
    scale_FSS_gen_mean: Optional[float] = None
    scale_ISS_gen_mean: Optional[float] = None
    scale_FSS_ens_mean: Optional[float] = None
    scale_ISS_ens_mean: Optional[float] = None
    scale_FSS_lr_mean: Optional[float] = None
    scale_ISS_lr_mean: Optional[float] = None

# ---------------------------------------------------------------------------
# Helpers to parse individual blocks
# ---------------------------------------------------------------------------

def _parse_sampler_name(name: str):
    """Extract rho, Schurn, sigscale from folder name like 'rho=5.00_Schurn=2.00_sigscale=0.90'."""
    rho = math.nan
    schurn = math.nan
    sigscale = math.nan
    parts = name.split("_")
    for p in parts:
        if p.startswith("rho="):
            try:
                rho = float(p.split("=", 1)[1])
            except ValueError:
                pass
        elif p.startswith("Schurn="):
            try:
                schurn = float(p.split("=", 1)[1])
            except ValueError:
                pass
        elif p.startswith("sigscale="):
            try:
                sigscale = float(p.split("=", 1)[1])
            except ValueError:
                pass
    return rho, schurn, sigscale


def _load_csv_as_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        logger.warning(f"[sampler_grid_summary] Missing CSV: {path}")
        return []
    with path.open() as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def _parse_distributional(prcp_root: Path, out: SamplerSummary) -> None:
    """
    dist_metrics.csv header:
        ref,comp,wasserstein,ks_stat,ks_p,kl_hr_to_x
    We want HR vs GEN and HR vs LR rows.
    """
    path = prcp_root / "distributional" / "tables" / "dist_metrics.csv"
    rows = _load_csv_as_rows(path)
    for r in rows:
        ref = r.get("ref", "").upper()
        comp = r.get("comp", "").upper()
        if ref != "HR":
            continue
        try:
            w = float(r.get("wasserstein", "nan"))
            ks = float(r.get("ks_stat", "nan"))
            kl = float(r.get("kl_hr_to_x", "nan"))
        except ValueError:
            continue
        if comp.startswith("GEN"):  # "GEN" or ensemble variants
            out.dist_W1_GEN = w
            out.dist_KS_GEN = ks
            out.dist_KL_GEN = kl
        elif comp.startswith("LR"):
            out.dist_W1_LR = w
            out.dist_KS_LR = ks
            out.dist_KL_LR = kl

def _parse_probabilistic(prcp_root: Path, out: SamplerSummary) -> None:
    """
    prob_summary.csv header:
        metric,mean,std,N
    metric names from evaluate_probabilistic.py:
      - CRPS_ensemble
      - PMM_MAE
      - EnergyScore
      - VariogramScore
      - PIT_KS_D
      - RankHist_max_abs_z
      - SpreadSkill_slope
      - SpreadSkill_pearson_r
      (+ some seasonal rows we ignore here)
    """
    path = prcp_root / "probabilistic" / "tables" / "prob_summary.csv"
    rows = _load_csv_as_rows(path)
    for r in rows:
        metric = r.get("metric", "")
        mlow = metric.lower()
        try:
            mean_val = float(r.get("mean", "nan"))
        except ValueError:
            continue

        if mlow == "crps_ensemble":
            out.prob_CRPS_mean = mean_val
        elif mlow == "pmm_mae":
            out.prob_PMM_MAE_mean = mean_val
        elif mlow == "energyscore":
            out.prob_EnergyScore_mean = mean_val
        elif mlow == "variogramscore":
            out.prob_VariogramScore_mean = mean_val
        elif mlow == "pit_ks_d":
            out.prob_PIT_KS_D = mean_val
        elif mlow == "rankhist_max_abs_z":
            out.prob_RankHist_max_abs_z = mean_val
        elif mlow == "spreadskill_slope":
            out.prob_SpreadSkill_slope = mean_val
        elif mlow == "spreadskill_pearson_r":
            out.prob_SpreadSkill_r = mean_val


def _parse_extremes(prcp_root: Path, out: SamplerSummary) -> None:
    """
    ext_tails.csv header:
        which,P95,P99,wet_freq,wet_hit_rate,n_days
    where which in {"HR","GEN","LR", "GEN_ENS"} depending on config.
    """
    path = prcp_root / "extremes" / "tables" / "ext_tails.csv"
    rows = _load_csv_as_rows(path)

    by_which: Dict[str, Dict[str, str]] = {}
    for r in rows:
        which = r.get("which", "").upper()
        if which:
            by_which[which] = r

    def _get_float(row: Optional[Dict[str, str]], key: str) -> Optional[float]:
        if row is None:
            return None
        try:
            return float(row.get(key, "nan"))
        except ValueError:
            return None

    hr = by_which.get("HR")
    gen = by_which.get("GEN") or by_which.get("GEN_ENS")

    out.ext_P95_HR = _get_float(hr, "P95")
    out.ext_P99_HR = _get_float(hr, "P99")
    out.ext_wetfreq_HR = _get_float(hr, "wet_freq")

    out.ext_P95_GEN = _get_float(gen, "P95")
    out.ext_P99_GEN = _get_float(gen, "P99")
    out.ext_wetfreq_GEN = _get_float(gen, "wet_freq")
    out.ext_wethit_GEN = _get_float(gen, "wet_hit_rate")

    # ratios vs HR (only if both are finite)
    if out.ext_P95_HR and out.ext_P95_GEN and out.ext_P95_HR != 0.0:
        out.ext_P95_ratio_GEN_HR = out.ext_P95_GEN / out.ext_P95_HR
    if out.ext_P99_HR and out.ext_P99_GEN and out.ext_P99_HR != 0.0:
        out.ext_P99_ratio_GEN_HR = out.ext_P99_GEN / out.ext_P99_HR
    if out.ext_wetfreq_HR and out.ext_wetfreq_GEN and out.ext_wetfreq_HR != 0.0:
        out.ext_wetfreq_ratio_GEN_HR = out.ext_wetfreq_GEN / out.ext_wetfreq_HR


def _parse_scale(prcp_root: Path, out: SamplerSummary) -> None:
    """
    scale_overview.csv header:
        metric,detail,value
    The code in evaluate_scale.py populates:
      FSS,gen_mean
      FSS,ens_mean
      FSS,lr_mean
      ISS,gen_mean
      ISS,ens_mean
      ISS,lr_mean
      PSD,band_ratio_csv,scale_psd_band_ratios_avg.csv
      PSD,slopes_csv,scale_psd_slopes_avg.csv
    We only pull the scalar FSS/ISS means here.
    """
    path = prcp_root / "scale" / "tables" / "scale_overview.csv"
    rows = _load_csv_as_rows(path)
    for r in rows:
        metric = r.get("metric", "").upper()
        detail = r.get("detail", "").lower()
        try:
            val = float(r.get("value", "nan"))
        except ValueError:
            continue
        if metric == "FSS":
            if detail == "gen_mean":
                out.scale_FSS_gen_mean = val
            elif detail == "ens_mean":
                out.scale_FSS_ens_mean = val
            elif detail == "lr_mean":
                out.scale_FSS_lr_mean = val
        elif metric == "ISS":
            if detail == "gen_mean":
                out.scale_ISS_gen_mean = val
            elif detail == "ens_mean":
                out.scale_ISS_ens_mean = val
            elif detail == "lr_mean":
                out.scale_ISS_lr_mean = val


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def summarize_sampler_grid(
    sampler_grid_root: str | Path,
    out_csv: Optional[str | Path] = None,
    out_json: Optional[str | Path] = None,
) -> tuple[Path, Path]:
    """
    Walk a sampler_grid root and collect metrics into a single CSV/JSON.

    Parameters
    ----------
    sampler_grid_root:
        Path to .../evaluation/<MODEL_KEY>/sampler_grid
    out_csv / out_json:
        Optional explicit output paths. If None, written inside sampler_grid_root.

    Returns
    -------
    (csv_path, json_path)
    """
    sampler_grid_root = Path(sampler_grid_root)
    if out_csv is None:
        out_csv = sampler_grid_root / "sampler_grid_summary.csv"
    else:
        out_csv = Path(out_csv)
    if out_json is None:
        out_json = sampler_grid_root / "sampler_grid_summary.json"
    else:
        out_json = Path(out_json)

    logger.info(f"[sampler_grid_summary] Collecting metrics from {sampler_grid_root}")

    summaries: List[SamplerSummary] = []

    for combo_dir in sorted(sampler_grid_root.iterdir()):
        if not combo_dir.is_dir():
            continue
        if combo_dir.name.startswith(".") or combo_dir.name.lower() == "old":
            continue

        sampler_id = combo_dir.name
        rho, schurn, sigscale = _parse_sampler_name(sampler_id)
        logger.info(f"[sampler_grid_summary] Processing {sampler_id} (rho={rho}, Schurn={schurn}, sigscale={sigscale})")

        prcp_root = combo_dir / "prcp"
        if not prcp_root.exists():
            logger.warning(f"[sampler_grid_summary] No prcp/ subdir found in {combo_dir}, skipping.")
            continue

        summary = SamplerSummary(
            sampler_id=sampler_id,
            rho=rho,
            Schurn=schurn,
            sigma_scale=sigscale,
        )

        _parse_distributional(prcp_root, summary)
        _parse_probabilistic(prcp_root, summary)
        _parse_extremes(prcp_root, summary)
        _parse_scale(prcp_root, summary)

        summaries.append(summary)

    if not summaries:
        raise RuntimeError(f"[sampler_grid_summary] No sampler combos found under {sampler_grid_root}")

    # ---- write CSV ----
    records: List[Dict[str, Any]] = [asdict(s) for s in summaries]
    fieldnames: List[str] = sorted({k for r in records for k in r.keys()})

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    logger.info(f"[sampler_grid_summary] Wrote CSV summary to {out_csv}")

    # ---- write JSON ----
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(records, f, indent=2)
    logger.info(f"[sampler_grid_summary] Wrote JSON summary to {out_json}")

    return out_csv, out_json


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Summarize sampler-grid evaluation metrics into a single table.")
    parser.add_argument(
        "--sampler_grid_root",
        required=True,
        help="Path to .../generated_samples/evaluation/<MODEL_KEY>/sampler_grid",
    )
    parser.add_argument(
        "--out_csv",
        default=None,
        help="Optional path for CSV output (default: sampler_grid_summary.csv in sampler_grid_root).",
    )
    parser.add_argument(
        "--out_json",
        default=None,
        help="Optional path for JSON output (default: sampler_grid_summary.json in sampler_grid_root).",
    )
    args = parser.parse_args()

    summarize_sampler_grid(
        sampler_grid_root=args.sampler_grid_root,
        out_csv=args.out_csv,
        out_json=args.out_json,
    )