"""
    Evaluation runner for EDM downscaling (univariate precipitation)

    Reads generation artifacts (prefer physical-space) and computes:
        A) Probabilistic performance (per-day, ensemble):
            - CRPS (Continuous Ranked Probability Score),  mean over pixels/masks
            - Reliability for exceedance (>= 1/5/10 mm/day), LR-binned optional
            - Spread-skill (using PMM as point estimator)
            - PIT (Probability Integral Transform) histograms and rank histograms
        B) Capability (across all days, using daily PMM fields):
            - FSS at 1/5/10 mm and 5/10/20 km scales
            - PSD slope + full PSD curves (with LR_ups reference)
            - P95/P99 + wet-day frequency
        C) Extremes (basin-mean daily series):
            - GEV first for Rx1day/Rx5day with bootstrap CIs
            - POT/GPD over a threshold with bootstrap CIs
    Outputs tables (JSON/CSV) and figures to <eval_out_root>/tables and <eval_out_root>/figures
"""

from __future__ import annotations
import json
import logging
import csv
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Iterable, Dict, Any, List

import numpy as np
import torch
import matplotlib.pyplot as plt

from sbgm.utils import get_model_string
# from sbgm.training_utils import load_land_mask_if_any  # if you have something similar; else set mask=None

from sbgm.monitoring import (
    compute_fss_at_scales,                 # on batches [B,1,H,W]
    compute_psd_slope,                     # returns dict/slopes
    compute_p95_p99_and_wet_day,           # returns dict
)
from sbgm.evaluate_sbgm.metrics_univariate import (
    crps_ensemble,                         # obs [H,W], ens [M,H,W]
    reliability_exceedance_lr_binned,      # returns per-bin stats
    spread_skill,                          # returns bin stats
    pit_values_from_ensemble,
    rank_histogram,
    to_numpy_1d_series,
    seasonal_block_index,
    rxk_series,
    fit_gev_block_maxima_with_ci,
    fit_pot_gpd_with_ci,
    compute_isotropic_psd,
    compute_and_save_pooled_pixel_distributions,
    compute_and_save_yearly_maps,
)

from sbgm.evaluate_sbgm.plot_utils import (
    plot_pooled_pixel_distributions,
    plot_yearly_maps,
    plot_reliability,
    plot_spread_skill,
    plot_fss_curves,
    plot_psd_slope_bar,
    plot_psd_curves_eval,
    plot_pit_and_rank,
)

logger = logging.getLogger(__name__)

# --- helpers for robust date parsing and season filtering ---
def _to_datetime64(d: str) -> np.datetime64:
    """Accept 'YYYY-MM-DD' or 'YYYYMMDD' stems and return np.datetime64('YYYY-MM-DD')."""
    try:
        if isinstance(d, str):
            s = d.strip()
            if len(s) == 8 and s.isdigit():
                s = f"{s[:4]}-{s[4:6]}-{s[6:8]}"
            return np.datetime64(s)
        return np.datetime64(d)
    except Exception:
        # Fallback: treat as NaT so caller can skip
        return np.datetime64('NaT')
def _season_of_month(m: int) -> str:
    # DJF, MAM, JJA, SON
    if m in (12, 1, 2):
        return "DJF"
    if m in (3, 4, 5):
        return "MAM"
    if m in (6, 7, 8):
        return "JJA"
    return "SON"
def _filter_dates_by_season(dates: Iterable[str], season: str) -> list[str]:
    """Return only those date stems belonging to the requested climatological season."""
    if season.upper() == "ALL":
        return list(dates)
    out = []
    for d in dates:
        dt = _to_datetime64(d)
        if str(dt) == "NaT":
            continue
        # works because np.datetime64 -> 'YYYY-MM-DD'
        s = str(dt)
        m = int(s[5:7])
        if _season_of_month(m) == season.upper():
            out.append(d)
    return out


# === Helpers for baseline evaluation metrics implementation ===
def _baseline_base_dir(cfg, baseline_name: str, split: str) -> Path:
    base_root = Path(cfg['paths']['sample_dir']) 
    return base_root / 'evaluation' / 'baselines' / baseline_name / split

def load_baseline_fss(cfg, baseline_name: str, split: str):
    """Return dict: { 'thr_list': [...], 'scales_km': [...], 'values': 2D np.array[T,S] } or None."""
    d = _baseline_base_dir(cfg, baseline_name, split) / 'tables' / 'fss_summary.csv'
    if not d.exists(): return None
    import csv
    thrs, scales = [], None
    rows = []
    with open(d, 'r') as f:
        r = csv.DictReader(f)
        for row in r:
            if scales is None:
                scales = [int(c.split('_')[1].replace('km','')) for c in row if c.startswith('FSS_')]
            thrs.append(float(row['thr']))
            rows.append([float(row[f'FSS_{s}km']) for s in scales])
    return {'thr_list': thrs, 'scales_km': scales, 'values': np.array(rows, dtype=float)}

def load_baseline_psd(cfg, baseline_name: str, split: str):
    """Return dict from psd_slope_summary.json plus (optionally) full curves if you want to read them later."""
    d = _baseline_base_dir(cfg, baseline_name, split) / 'tables' / 'psd_slope_summary.json'
    if not d.exists(): return None
    return json.loads(d.read_text())

def load_baseline_tails(cfg, baseline_name: str, split: str):
    """Return dict with p95/p99 and wet-day freq for HR and baseline."""
    d = _baseline_base_dir(cfg, baseline_name, split) / 'tables' / 'tails_summary.json'
    if not d.exists(): return None
    return json.loads(d.read_text())

def load_baseline_reliability(cfg, baseline_name: str, split: str):
    """
    Returns:
      dict[float, list[dict]] mapping threshold -> rows with keys:
        'bin_center', 'prob_pred', 'freq_obs', 'count'
      or None if the CSV does not exist.
    """
    p = _baseline_base_dir(cfg, baseline_name, split) / 'tables' / 'reliability_bins.csv'
    if not p.exists():
        return None

    out = {}
    try:
        import csv
        with open(p, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # robust numeric parsing
                try:
                    thr = float(row.get('thr', 'nan'))
                    bin_center = float(row.get('bin_center', row.get('prob_pred', 'nan')))
                    prob_pred  = float(row.get('prob_pred', 'nan'))
                    freq_obs   = float(row.get('freq_obs', 'nan'))
                    count_raw  = row.get('count', '0')
                    # 'count' can sometimes be float-like in CSV; coerce to int safely
                    try:
                        count = int(float(count_raw))
                    except Exception:
                        count = 0
                except Exception:
                    # skip malformed lines
                    continue

                if thr not in out:
                    out[thr] = []
                out[thr].append({
                    'bin_center': bin_center,
                    'prob_pred':  prob_pred,
                    'freq_obs':   freq_obs,
                    'count':      count,
                })
    except Exception:
        return None

    return out

def load_all_baselines(cfg, split: str, names=None):
    """
    names = list like ['bilinear','qm','unet_sr'] or None -> use cfg.baseline.compare_list or all three.
    Returns a dict:
      {
        name: {
           'fss': {...} or None,
           'psd': {...} or None,
           'tails': {...} or None,
           'reliability': {thr->df} or None
        }, ...
      }
    """
    if names is None:
        names = cfg.get('baseline', {}).get('compare_list', ['bilinear','qm','unet_sr'])
    out = {}
    for name in names:
        out[name] = dict(
            fss        = load_baseline_fss(cfg, name, split),
            psd        = load_baseline_psd(cfg, name, split),
            tails      = load_baseline_tails(cfg, name, split),
            reliability= load_baseline_reliability(cfg, name, split),
        )
    return out

@dataclass
class EvaluationConfig:
    gen_dir: str                     # Root directory where generated samples are stored 
    out_dir: str                     # Root directory to save evaluation outputs
    grid_km_per_px: float = 2.5      # Spatial resolution of the data in km/px
    lr_grid_km_per_px: float = 31.0  # Spatial resolution of the LR data in km/px
    fss_scales_km: tuple = (5, 10, 20)  # Scales (in km) at which to compute FSS
    thresholds_mm: tuple = (1.0, 5.0, 10.0)  # Thresholds (in mm/day) for exceedance reliability and FSS
    wet_threshold_mm: float = 1.0  # Threshold (in mm/day) to define wet days for P95/P99
    reliability_bins: int = 10  # Number of bins for reliability diagrams
    spread_skill_bins: int = 10  # Number of bins for spread-skill analysis
    pit_bins: int = 10  # Number of bins for PIT histograms
    psd_ignore_low_k_bins: int = 1  # Number of lowest k bins to ignore in PSD slope fitting
    psd_normalize: str = "none"  # "none" | "per_field" | "match_ref"
    random_ref_kind: str = "phase_randomized"  # Kind of random reference for FSS, "iid_marginal" | "spatial_shuffle" | "phase_randomized"
    seasonal_summaries: bool = True
    region_mask_path: Optional[str] = None
    pixel_dist_n_bins: int = 100        # Number of bins for pooled pixel distributions
    pixel_dist_vmax_percentile: float = 99.5  # Max value percentile for pooled pixel distributions
    pixel_dist_save_cap: int = 2_000_000  # Max number of pixel samples to save for pooled pixel distributions
    yearly_maps: tuple = ("mean", "sum", "rx1", "rx5")  # Types of yearly maps to compute
    seasons: tuple = ("ALL", "DJF", "MAM", "JJA", "SON")  # Seasons to consider for seasonal analysis
    add_yearly_ratio_diff: bool = True
    seed: int = 504                  # Random seed for reproducibility
    eval_land_only: bool = True    # Whether to evaluate only over land pixels if a land mask is provided


class EvaluationRunner:
    def __init__(self,
                 cfg_yaml: dict,
                 eval_cfg: EvaluationConfig,
                 device: torch.device,
                 mask: Optional[torch.Tensor] = None,
                 baseline_data: Optional[Dict[str, Dict[str, Any]]] = None,
                 baseline_eval_dirs: Optional[Dict[str, str]] = None):
        self.cfg_yaml = cfg_yaml
        self.eval_cfg = eval_cfg
        self.device = device
        self.mask = mask  # [H,W] bool or None
        self.baseline_data = baseline_data
        self.baseline_eval_dirs = baseline_eval_dirs

        self.gen_root = Path(eval_cfg.gen_dir)
        # Prefer physical-space outputs; fall back to model-space if needed
        self.dir_ens_phys = self.gen_root / "ensembles_phys"
        self.dir_pmm_phys = self.gen_root / "pmm_phys"
        self.dir_lrhr_phys = self.gen_root / "lr_hr_phys"

        self.dir_ens_model = self.gen_root / "ensembles"
        self.dir_pmm_model = self.gen_root / "pmm"

        # Land/sea masks
        self.dir_lsm = self.gen_root / 'lsm'
        self.mask_global: Optional[torch.Tensor] = None  # [H,W] bool
        try:
            p = self.gen_root / 'meta' / 'land_mask.npz'
            if p.exists():
                arr = np.load(p, allow_pickle=True).get('lsm_hr', None)
                if arr is not None:
                    self.mask_global = torch.from_numpy(arr).to(torch.bool)
        except Exception as e:
            logger.warning(f"[eval] Could not load global land mask: {e}")

        self.out_root = Path(eval_cfg.out_dir)
        (self.out_root / "tables").mkdir(parents=True, exist_ok=True)
        (self.out_root / "figures").mkdir(parents=True, exist_ok=True)

        self.eval_land_only = bool(getattr(self.eval_cfg, 'eval_land_only', True))
        logger.info("[eval] Masking mode: eval_land_only=%s", self.eval_land_only)

        # Optional ROI mask (e.g., Denmark-only)
        self.roi_mask: Optional[torch.Tensor] = None
        try:
            roi_path = eval_cfg.region_mask_path
            if isinstance(roi_path, str):
                p = Path(roi_path)
                if p.suffix.lower() in {".npz", ".npy"} and p.exists():
                    arr = np.load(p, allow_pickle=True)
                    if isinstance(arr, np.lib.npyio.NpzFile): # type: ignore
                        # try common keys
                        a = arr.get("mask", None)
                        if a is None: a = arr.get("lsm_hr", None)
                        if a is None: a = arr.get("roi", None)
                        if a is None: a = arr.get("data", None)
                    else:
                        a = arr
                    if a is not None:
                        m = torch.from_numpy(np.asarray(a)).to(torch.bool)
                        # normalize to [H,W]
                        if m.dim() == 4 and m.shape[:2] == (1, 1): m = m.squeeze(0).squeeze(0)
                        elif m.dim() == 3 and m.shape[0] == 1:     m = m.squeeze(0)
                        self.roi_mask = m
                        logger.info("[eval] Loaded ROI mask from %s with shape %s", p, tuple(self.roi_mask.shape))
        except Exception as e:
            logger.warning(f"[eval] Could not load region mask: {e}")

    # Helper to point to seasonal subfolders
    def _get_baseline_eval_dirs(self) -> Optional[Dict[str, str]]:
        """Return baseline eval dirs aligned to the current output folder (append season if applicable)."""
        if not self.baseline_eval_dirs:
            return None
        season = self.out_root.name.lower()
        if season in {"all", "djf", "mam", "jja", "son"}:
            return {k: str(Path(v) / season) for k, v in self.baseline_eval_dirs.items()}
        return self.baseline_eval_dirs

    # ---------- I/O helpers --------
    def _list_dates(self) -> Iterable[str]:
        base = self.dir_pmm_phys if self.dir_pmm_phys.exists() else self.dir_pmm_model
        dates = sorted([f.stem for f in base.glob("*.npz")])
        logger.info("[eval] Found %d date files under %s", len(dates), base)
        return dates

    def _load_npz(self, folder: Path, date: str, key: str):
        p = folder / f"{date}.npz"
        if not p.exists(): return None
        d = np.load(p, allow_pickle=True)
        return d.get(key, None)

    def _load_obs(self, date: str) -> Optional[torch.Tensor]:
        # prefer physical HR if available
        x = self._load_npz(self.dir_lrhr_phys, date, "hr")
        if x is None:
            # fallback to model-space HR is not ideal; if needed, you can apply back-transform here.
            x = self._load_npz(self.gen_root / "lr_hr", date, "hr")
        if x is None: return None
        t = torch.from_numpy(x).squeeze(0)  # [1,1,H,W] -> [1,H,W] -> squeeze to [H,W] below
        return t.squeeze(0)

    def _load_ens(self, date: str) -> Optional[torch.Tensor]:
        x = self._load_npz(self.dir_ens_phys, date, "ens")
        if x is None:
            x = self._load_npz(self.dir_ens_model, date, "ens")
        if x is None: return None
        t = torch.from_numpy(x).squeeze(1)  # [M,1,H,W] -> [M,H,W]
        return t

    def _load_pmm(self, date: str) -> Optional[torch.Tensor]:
        x = self._load_npz(self.dir_pmm_phys, date, "pmm")
        if x is None:
            x = self._load_npz(self.dir_pmm_model, date, "pmm")
        if x is None: return None
        t = torch.from_numpy(x).squeeze(0)  # [1,1,H,W] -> [1,H,W] -> squeeze to [H,W] below
        return t.squeeze(0)

    def _load_mask(self, date: str) -> Optional[torch.Tensor]:
        # Prefer a global canonical mask; else try per-date; else return None
        if self.mask_global is not None:
            m = self.mask_global
        else:
            p = self.dir_lsm / f"{date}.npz"
            if not p.exists():
                return None
            try:
                arr = np.load(p, allow_pickle=True).get('lsm_hr', None)
                if arr is None:
                    return None
                m = torch.from_numpy(arr).to(torch.bool)
            except Exception as e:
                logger.warning(f"[eval] Failed loading per-date mask for {date}: {e}")
                return None

        # Normalize mask to [H,W]
        if m.dim() == 4 and m.shape[:2] == (1, 1):   # [1,1,H,W]
            m = m.squeeze(0).squeeze(0)
        elif m.dim() == 3 and m.shape[0] == 1:       # [1,H,W]
            m = m.squeeze(0)
        elif m.dim() == 2:
            pass
        else:
            logger.warning("[eval] Unexpected mask shape %s; coercing last two dims as HxW.", tuple(m.shape))
            m = m.reshape(m.shape[-2], m.shape[-1])
        
        # Intersect with ROI if provided
        if hasattr(self, "roi_mask") and (self.roi_mask is not None):
            try:
                rm = self.roi_mask
                if rm.shape != m.shape:
                    # attempt to coerce to same HxW if leading singleton dims exist
                    if rm.dim() == 4 and rm.shape[:2] == (1, 1): rm = rm.squeeze(0).squeeze(0)
                    elif rm.dim() == 3 and rm.shape[0] == 1:     rm = rm.squeeze(0)
                if rm.shape == m.shape:
                    m = (m & rm)
                else:
                    logger.warning("[eval] ROI mask shape %s did not match mask %s; skipping ROI intersection.", tuple(rm.shape), tuple(m.shape))
            except Exception:
                pass
        
        return m
    
    def _eval_distributions(self):
        """
            Compute pooled pixel distributions and yearly maps using already-generated artifacts.
            Reuses same implementations as baseline eval for consistency.
        """
        tables_dir = self.out_root / "tables"
        figs_dir = self.out_root / "figures"

        try:
            logger.info("[eval] Computing pooled pixel distributions...")
            # knobs (inherit defaults from EvaluationConfig or cfg_yaml.evaluation if present)
            n_bins = self.eval_cfg.pixel_dist_n_bins
            vmax_pct = self.eval_cfg.pixel_dist_vmax_percentile
            save_cap = self.eval_cfg.pixel_dist_save_cap

            logger.info("[eval] Computing pooled pixel distributions (land-only=%s)...", self.eval_land_only)
            ok = compute_and_save_pooled_pixel_distributions(
                gen_root=self.gen_root,
                out_root=self.out_root,
                mask_global=(self.mask_global if self.eval_land_only else None),
                include_lr=True,
                n_bins=n_bins,
                vmax_percentile=vmax_pct,
                save_samples_cap=save_cap,
            )
            if ok:
                logger.info("[eval] Wrote pooled pixel distributions under %s", self.out_root / "tables")
            else:
                logger.warning("[eval] Failed to compute pooled pixel distributions (empty or missing inputs).")
            
            # Optional plots
            try:
                plot_pooled_pixel_distributions(
                    eval_root=str(self.out_root),
                    baseline_eval_dirs=None,)
            except Exception as e:
                logger.warning(f"[eval] Could not plot pooled pixel distributions: {e}")

        except Exception as e:
            logger.warning(f"[eval] Exception during pooled pixel distributions computation: {e}")

    def _eval_yearly_maps(self):
        """
            Compute yearly maps using already-generated artifacts.
            Reuses same implementations as baseline eval for consistency.
        """
        try:
            logger.info("[eval] Computing yearly maps...")
            which_maps = tuple(self.eval_cfg.yearly_maps)
            
            # Build a static mask (land-only or ROI) if needed
            mask_for_maps = None
            if self.eval_land_only and (self.mask_global is not None):
                mask_for_maps = self.mask_global
            
            if getattr(self, "roi_mask", None) is not None:
                rm = self.roi_mask
                # normalize rm to [H,W]
                if rm is None or not hasattr(rm, "dim"):
                    logger.warning("[eval] ROI mask is not a tensor-like object; skipping ROI intersection for yearly maps.")
                else:
                    if rm.dim() == 4 and rm.shape[:2] == (1, 1):   # [1,1,H,W]
                        rm = rm.squeeze(0).squeeze(0)
                    elif rm.dim() == 3 and rm.shape[0] == 1:       # [1,H,W]
                        rm = rm.squeeze(0)
                    if mask_for_maps is None:
                        mask_for_maps = rm
                    elif mask_for_maps.shape == rm.shape:
                        mask_for_maps = (mask_for_maps & rm)
                        logger.info("[eval] Combined ROI mask for yearly maps.")
                    else:
                        logger.warning("[eval] ROI mask shape %s did not match existing mask %s; skipping ROI intersection for yearly maps.", tuple(rm.shape), tuple(mask_for_maps.shape))
                    

            ok = compute_and_save_yearly_maps(
                gen_root=self.gen_root,
                out_root=self.out_root,
                which=which_maps,
                include_lr=True,
                mask=mask_for_maps, # To help function mask HR/PMM/LR consistently
            )
            if ok:
                logger.info("[eval] Wrote yearly maps under %s", self.out_root / "tables")
                try:
                    plot_yearly_maps(
                        eval_root=str(self.out_root),
                        years=None,
                        which=which_maps,
                        baselines=self._get_baseline_eval_dirs(),
                    )
                except Exception as e:
                    logger.warning(f"[eval] Could not plot yearly maps: {e}")
            else:
                logger.warning("[eval] Failed to compute yearly maps (insufficient accumulation?).")
        
        except Exception as e:
            logger.warning(f"[eval] Exception during yearly maps computation: {e}")

    # ---------- Probabilistic metrics (per day) ----------
    def eval_probabilistic(self, dates_subset: Optional[List[str]] = None):
        """
        Compute per-day probabilistic metrics and save tables/figures
        Evaluates: 
            - CRPS (mean over pixels/masks)
            - Reliability for exceedance by thresholds
            - Spread-skill (using PMM as point)
            - PIT histograms and rank histograms
        Outputs CSV tables and NPZ files for PIT/rank histograms
        """
        tables_dir = self.out_root / "tables"
        figs_dir   = self.out_root / "figures"
        tables_dir.mkdir(parents=True, exist_ok=True)
        figs_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[eval] Probabilistic -> tables: crps_daily.csv, reliability_bins.csv, spread_skill.csv; figs: pit_hist.png, rank_hist.png")

        rows_crps = []
        rows_rel = []
        rows_ss = []

        pit_values_all = []
        rank_counts = None

        for date in (dates_subset if dates_subset is not None else self._list_dates()):
            obs = self._load_obs(date)      # [H,W] or None
            ens = self._load_ens(date)      # [M,H,W] or None
            if obs is None or ens is None: 
                logger.warning(f"[eval] Missing obs/ens for {date}, skipping.")
                continue

            # Prefer mask saved during generation (stationary canonical or per-date),
            # fall back to user-provided mask.
            mask = self._load_mask(date) if self.eval_land_only else None

            # CRPS (mean over masked pixels)
            crps_val = crps_ensemble(obs, ens, mask=mask, reduction="mean")
            rows_crps.append({"date": date, "crps": float(crps_val)})

            # Reliability for exceedance by thresholds — explode per-bin vectors into scalar rows
            for thr in self.eval_cfg.thresholds_mm:
                rel = reliability_exceedance_lr_binned(
                    obs=obs, ens=ens, threshold=float(thr),
                    lr_covariate=None,  # or pass basin-mean LR if you like
                    n_bins=int(self.eval_cfg.reliability_bins),
                    mask=mask, return_brier=True,
                )
                bc   = rel.get("bin_center", [])
                pp   = rel.get("prob_pred", [])
                fobs = rel.get("freq_obs", [])
                cnt  = rel.get("count", [])
                L = min(len(bc), len(pp), len(fobs), len(cnt))

                added = 0
                for i in range(L):
                    def _f(x):
                        if torch.is_tensor(x): return float(x.detach().cpu().item())
                        try: return float(x)
                        except: return float('nan')
                    def _i(x):
                        if torch.is_tensor(x): return int(x.detach().cpu().item())
                        try: return int(x)
                        except: return 0

                    rows_rel.append({
                        "date": date,
                        "thr": float(thr),
                        "bin_center": _f(bc[i]),
                        "prob_pred":  _f(pp[i]),
                        "freq_obs":   _f(fobs[i]),
                        "count":      _i(cnt[i]),
                    })
                    added += 1
                logger.info("[eval] reliability: date=%s thr=%.1f → %d rows", date, float(thr), added)

            # Spread–skill (using PMM as point) — explode per-bin vectors into scalar rows
            ss = spread_skill(obs=obs, ens=ens, point="pmm", mask=mask,
                            n_bins=int(self.eval_cfg.spread_skill_bins))
            bc  = ss.get("bin_center", [])
            spr = ss.get("spread", [])
            skl = ss.get("skill", [])
            cnt = ss.get("count", [])
            L = min(len(bc), len(spr), len(skl), len(cnt))

            added = 0
            for i in range(L):
                def _f(x):
                    if torch.is_tensor(x): return float(x.detach().cpu().item())
                    try: return float(x)
                    except: return float('nan')
                def _i(x):
                    if torch.is_tensor(x): return int(x.detach().cpu().item())
                    try: return int(x)
                    except: return 0

                rows_ss.append({
                    "date": date,
                    "bin_center": _f(bc[i]),
                    "spread":     _f(spr[i]),
                    "skill":      _f(skl[i]),
                    "count":      _i(cnt[i]),
                })
                added += 1
            logger.info("[eval] spread–skill: date=%s → %d rows", date, added)


            # PIT and rank hist (accumulate) — expected by metrics: obs [B,H,W], ens [B,M,H,W]
            obs_bhw  = obs.unsqueeze(0)                   # [1,H,W]
            ens_bmhw = ens.unsqueeze(0)                   # [1,M,H,W]

            # Make mask [B,H,W] (B=1), squeeze any channel dim if present
            mask_bhw = None
            if mask is not None:
                m = mask
                if m.dim() == 4 and m.shape[1] == 1:      # [B,1,H,W] -> [B,H,W]
                    m = m.squeeze(1)
                elif m.dim() == 3 and m.shape[0] != 1:    # [H,W,?] unexpected; fall back to [1,H,W]
                    m = m[:1]
                elif m.dim() == 2:                        # [H,W] -> [1,H,W]
                    m = m.unsqueeze(0)
                mask_bhw = m

            pits = pit_values_from_ensemble(obs_bhw, ens_bmhw, mask=mask_bhw)  # 1-D tensor
            pit_values_all.append(pits.cpu())
            rh = rank_histogram(obs_bhw, ens_bmhw, mask=mask_bhw)              # [M+1]
            rank_counts = rh if rank_counts is None else (rank_counts + rh)

        # --- write CSVs without pandas ---
        def _write_csv(path, rows):
            if not rows:
                open(path, 'w').close(); return
            norm_rows = []
            for r in rows:
                out = {}
                for k, v in r.items():
                    if isinstance(v, torch.Tensor):
                        out[k] = float(v.item()) if v.numel() == 1 else str(v.detach().cpu().numpy().tolist())
                    else:
                        out[k] = v
                norm_rows.append(out)
            header = list(norm_rows[0].keys())
            with open(path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=header)
                w.writeheader()
                for r in norm_rows:
                    w.writerow(r)

        _write_csv(str(tables_dir / "crps_daily.csv"), rows_crps)
        logger.info(f"[eval] Wrote CRPS table to {tables_dir / 'crps_daily.csv'}")
        _write_csv(str(tables_dir / "reliability_bins.csv"), rows_rel)
        logger.info(f"[eval] Wrote Reliability table to {tables_dir / 'reliability_bins.csv'}")
        _write_csv(str(tables_dir / "spread_skill.csv"), rows_ss)
        logger.info(f"[eval] Wrote Spread Skill table to {tables_dir / 'spread_skill.csv'}")

        # Save PIT histogram
        if len(pit_values_all) > 0:
            pits = torch.cat(pit_values_all, dim=0).numpy()
            np.savez_compressed(figs_dir / "pit_values_all.npz", pits=pits)
            logger.info("[eval] Saved PIT values → %s", figs_dir / "pit_values_all.npz")
        if rank_counts is not None:
            np.savez_compressed(figs_dir / "rank_hist_counts.npz", counts=rank_counts.cpu().numpy())
            logger.info("[eval] Saved rank histogram counts → %s", figs_dir / "rank_hist_counts.npz")

        # === Plots (with baseline overlays if available) ===
        try: 
            plot_reliability(eval_root=str(self.out_root),
                             thr_mm_list=self.eval_cfg.thresholds_mm,
                             baseline_eval_dirs=self._get_baseline_eval_dirs())
        except Exception as e:
            logger.warning(f"[eval] Could not plot reliability diagrams: {e}")
        try:
            plot_spread_skill(eval_root=str(self.out_root))
        except Exception as e:
            logger.warning(f"[eval] Could not plot spread–skill: {e}")
        try:
            plot_pit_and_rank(eval_root=str(self.out_root),
                              pit_bins=int(self.eval_cfg.pit_bins))
        except Exception as e:
            logger.warning(f"[eval] Could not plot PIT/rank histograms: {e}")



    # ---------- Capability metrics on PMM (across all days) ----------
    def eval_capability(self, dates_subset: Optional[List[str]] = None):
        """
        Compute capability metrics using daily PMM fields and save tables/figures
        Evaluates: 
            - FSS at 1/5/10 mm and 5/10/20 km scales
            - PSD slope + full PSD curves (with LR_ups reference)
            - P95/P99 + wet-day frequency
        Outputs tables (CSV/JSON) and figures
        """
        tables_dir = self.out_root / "tables"
        figs_dir   = self.out_root / "figures"
        tables_dir.mkdir(parents=True, exist_ok=True)
        figs_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[eval] Capability -> FSS, PSD slope, p95/p99+wet; yearly_maps=%s", ",".join(self.eval_cfg.yearly_maps))

        PMM, HR, dates_used = [], [], []
        for date in (dates_subset if dates_subset is not None else self._list_dates()):
            pmm = self._load_pmm(date)   # [H,W]
            obs = self._load_obs(date)   # [H,W]
            if pmm is None or obs is None:
                continue
            PMM.append(pmm.unsqueeze(0).unsqueeze(0))  # -> [1,1,H,W]
            HR.append(obs.unsqueeze(0).unsqueeze(0))
            dates_used.append(date)

        if len(PMM) == 0:
            logger.warning("[eval] No PMM/HR pairs found.")
            return

        pmm_bt = torch.cat(PMM, dim=0)  # [B,1,H,W]
        hr_bt  = torch.cat(HR, dim=0)   # [B,1,H,W]

        # Sanitize: replace NaNs/±inf and clamp negatives to 0 for precip
        pmm_bt = torch.nan_to_num(pmm_bt, nan=0.0, posinf=None, neginf=0.0).clamp_min(0.0)
        hr_bt  = torch.nan_to_num(hr_bt,  nan=0.0, posinf=None, neginf=0.0).clamp_min(0.0)
        logger.info("[eval] Capability stack: pmm_bt=%s hr_bt=%s (after nan→num and clamp≥0)", tuple(pmm_bt.shape), tuple(hr_bt.shape))

        # Build per-sample mask batch aligned to dates_used
        mask_bt = None
        if dates_used:
            mask_list = []
            all_have_mask = True
            for d in dates_used:
                m = self._load_mask(d) if self.eval_land_only else None
                if m is None:
                    m = self.mask
                if m is None:
                    all_have_mask = False
                    break
                # Normalize to [H,W]
                if m.dim() == 4 and m.shape[:2] == (1, 1):
                    m = m.squeeze(0).squeeze(0)
                elif m.dim() == 3 and m.shape[0] == 1:
                    m = m.squeeze(0)
                mask_list.append(m.unsqueeze(0).unsqueeze(0))  # [1,1,H,W]
            if all_have_mask and len(mask_list) == len(dates_used):
                mask_bt = torch.cat(mask_list, dim=0)
                logger.info("[eval] Using per-date mask batch → %s", tuple(mask_bt.shape))
            else:
                mask_bt = None
                logger.info("[eval] Proceeding without mask for capability (not all dates had masks).")
        # FSS for thresholds x scales
        fss_rows = []
        for thr in self.eval_cfg.thresholds_mm:
            # Basic checks
            assert pmm_bt.shape == hr_bt.shape and pmm_bt.dim() == 4 and pmm_bt.shape[1] == 1
            if mask_bt is not None:
                assert mask_bt.shape[:1] == pmm_bt.shape[:1] and mask_bt.shape[2:] == pmm_bt.shape[2:]
            
            scores = compute_fss_at_scales(gen_bt=pmm_bt, hr_bt=hr_bt, mask=mask_bt,
                                        grid_km_per_px=self.eval_cfg.grid_km_per_px,
                                        fss_km=list(self.eval_cfg.fss_scales_km),
                                        thr_mm=float(thr))
            # Normalize to columns like FSS_5km, FSS_10km, FSS_20km (what plot_utils expects)
            row = {"thr": float(thr)}
            for km in self.eval_cfg.fss_scales_km:
                # try several likely key variants that compute_fss_at_scales() might return
                key_candidates = [
                    f"FSS_{int(km)}km", f"FSS_{float(km)}km",
                    f"{int(km)}km", f"{float(km)}km",
                    f"FSS_{km}", str(km), f"k{int(km)}", f"k{float(km)}",
                ]
                val = None
                for kc in key_candidates:
                    if isinstance(scores, dict) and kc in scores:
                        val = scores[kc]
                        break
                if val is None:
                    # fallback: if scores is an ordered dict/list aligned with fss_scales_km
                    try:
                        idx = list(self.eval_cfg.fss_scales_km).index(km)
                        if isinstance(scores, dict):
                            val = list(scores.values())[idx]
                        else:
                            val = scores[idx]
                    except Exception:
                        val = float('nan')
                if isinstance(val, torch.Tensor):
                    val = float(val.item()) if val.numel() == 1 else float(val.mean().item())
                row[f"FSS_{int(km)}km"] = float(val)
            fss_rows.append(row)

        # write fss_summary.csv without pandas
        if fss_rows:
            header = list(fss_rows[0].keys())
            with open(tables_dir / "fss_summary.csv", 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=header)
                w.writeheader()
                for r in fss_rows:
                    r2 = {k: (float(v.item()) if isinstance(v, torch.Tensor) and v.numel()==1 else v) for k,v in r.items()}
                    w.writerow(r2)
        else:
            open(tables_dir / "fss_summary.csv", 'w').close()
        logger.info(f"[eval] Wrote FSS summary to {tables_dir / 'fss_summary.csv'}")


        # # PSD slope and series (save CSV, handle plotting in plot_utils)
        # psd_summ = compute_psd_slope(gen_bt=pmm_bt, hr_bt=hr_bt, mask=mask_bt,
        #                              ignore_low_k_bins=self.eval_cfg.psd_ignore_low_k_bins)
        # (tables_dir / "psd_slope_summary.json").write_text(json.dumps(psd_summ, indent=2))
        # logger.info(f"[eval] Wrote PSD slope summary to {tables_dir / 'psd_slope_summary.json'}")
        # # Save full isotropic PSD series as CSV, handl plotting in plot_utils
        # try:
        #     psd_gen = compute_isotropic_psd(pmm_bt, dx_km=self.eval_cfg.grid_km_per_px, mask=mask_bt)
        #     psd_hr  = compute_isotropic_psd(hr_bt,  dx_km=self.eval_cfg.grid_km_per_px, mask=mask_bt)
        #     k = psd_gen["k"].detach().cpu().numpy() 
        #     Pg = psd_gen["psd"].detach().cpu().numpy()
        #     Ph = psd_hr["psd"].detach().cpu().numpy()

        #     # Write CSV with columns: k, PSD_gen, PSD_hr
        #     out_csv = tables_dir / "psd_curves.csv"
        #     with open(out_csv, 'w', newline='') as f:
        #         w = csv.writer(f)
        #         w.writerow(["k","psd_pmm","psd_hr"])
        #         for i in range(len(k)):
        #             try: 
        #                 w.writerow([float(k[i]), float(Pg[i]), float(Ph[i])])
        #             except Exception:
        #                 # best-effort: skip malformed rows
        #                 continue
        #     logger.info(f"[eval] Wrote PSD curves to {out_csv}")
        # except Exception as e:
        #     logger.warning(f"[eval] Saving PSD curves CSV failed: {e}")

        # P95/P99 and wet-day frequency
        tails = compute_p95_p99_and_wet_day(
            pmm_bt, hr_bt=hr_bt, mask=mask_bt,
            wet_threshold_mm=self.eval_cfg.wet_threshold_mm
        )
        def _py(v):
            import numpy as _np
            if isinstance(v, torch.Tensor):
                return float(v.detach().cpu().item()) if v.numel() == 1 else [float(x) for x in v.detach().cpu().flatten()]
            if isinstance(v, _np.generic):
                return float(v.item())
            return v
        tails_py = {k: _py(v) for k, v in (tails or {}).items()}
        (tables_dir / "tails_summary.json").write_text(json.dumps(tails_py, indent=2))
        logger.info("[eval] Wrote tails summary to %s : %s", tables_dir / "tails_summary.json", tails_py)

        # Additionally write a flat CSV so downstream summary scripts don’t produce NaNs
        # Columns: choose PMM (gen_) as the model’s point estimate and include HR for reference
        def _first_scalar(val):
            # If val is a list, return its first element; else return as is
            if isinstance(val, list):
                return float(val[0]) if val else float('nan')
            return float(val)
        tails_row = {
            "P95":        _first_scalar(tails_py.get("gen_p95", np.nan)),
            "P99":        _first_scalar(tails_py.get("gen_p99", np.nan)),
            "WetDayFreq": _first_scalar(tails_py.get("gen_wet_freq", np.nan)),
            "HR_P95":     _first_scalar(tails_py.get("hr_p95", np.nan)),
            "HR_P99":     _first_scalar(tails_py.get("hr_p99", np.nan)),
            "HR_WetDayFreq": _first_scalar(tails_py.get("hr_wet_freq", np.nan)),
        }
        tails_csv = tables_dir / "tails_summary_flat.csv"
        with open(tails_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(tails_row.keys()))
            w.writeheader(); w.writerow(tails_row)
        logger.info("[eval] Wrote flat tails CSV to %s : %s", tails_csv, tails_row)

        # === Plots (with baseline overlays if available) ===
        try:
            plot_fss_curves(eval_root=str(self.out_root),
                            thr_mm_list=self.eval_cfg.thresholds_mm,
                            baseline_eval_dirs=self._get_baseline_eval_dirs())
        except Exception as e:
            logger.warning(f"[eval] Could not plot FSS curves: {e}")

    def _eval_psd(self, dates_subset: Optional[list[str]] = None):
        """
        Compute isotropic PSD for HR, PMM (generated) and LR (physical, in LR space),
        using the SAME logic as metrics_univariate.compute_isotropic_psd, and save
        everything so plotting does NOT have to recompute.

        Outputs:
          - tables/psd_slope_summary.json        (what you already had)
          - tables/psd_curves.npz                (all curves + CIs + meta)
        """
        tables_dir = self.out_root / "tables"
        figures_dir = self.out_root / "figures"
        tables_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)

        # ---------- 1) collect batches ----------
        # we reuse the same dates we used for capability (pmm_phys)
        dates = list(dates_subset) if dates_subset is not None else list(self._list_dates())
        if len(dates) == 0:
            logger.warning("[eval] PSD: no dates found, skipping.")
            return

        hr_list = []
        pmm_list = []
        lr_list = []

        for d in dates:
            hr = self._load_obs(d)           # [H,W]
            pmm = self._load_pmm(d)          # [H,W]
            # physical LR in LR space (living under lr_hr_phys)
            lr_pair = self._load_npz(self.dir_lrhr_phys, d, "lr")  # ← this is how your other code does it
            if hr is None or pmm is None:
                # skip dates with missing data
                continue

            # turn masks to [H,W]
            mask = self._load_mask(d) if self.eval_land_only else None

            hr_list.append((hr, mask))
            pmm_list.append((pmm, mask))

            if lr_pair is not None:
                # lr_pair is np.ndarray, shape could be [C,H',W'] or [1,H',W']
                # we want LR as torch [1,H',W']
                lr_t = torch.from_numpy(lr_pair)
                if lr_t.ndim == 3:
                    # if multi-channel LR → take first (your precip LR)
                    lr_t = lr_t[0:1, ...]
                elif lr_t.ndim == 2:
                    lr_t = lr_t.unsqueeze(0)
                lr_list.append((lr_t, None))   # LR mask usually in LR domain → we skip mask here
            else:
                lr_list.append(None)

        if len(hr_list) == 0:
            logger.warning("[eval] PSD: collected 0 valid HR/PMM pairs, skipping.")
            return

        # ---------- 2) stack to batches ----------
        # HR/PMM → all on HR grid
        hr_batch = torch.stack([x[0] for x in hr_list], dim=0).unsqueeze(1)   # [B,1,H,W]
        pmm_batch = torch.stack([x[0] for x in pmm_list], dim=0).unsqueeze(1) # [B,1,H,W]
        # masks: either all None or per-sample
        hr_mask = None
        if any(x[1] is not None for x in hr_list):
            ms = []
            for _, m in hr_list:
                if m is None:
                    m = torch.ones_like(hr_list[0][0], dtype=torch.bool)  # [H,W]
                # normalize masks to [H,W]
                if m.dim() == 4 and m.shape[:2] == (1, 1):
                    m = m.squeeze(0).squeeze(0)
                elif m.dim() == 3 and m.shape[0] == 1:
                    m = m.squeeze(0)
                ms.append(m)
            hr_mask = torch.stack(ms, dim=0).unsqueeze(1)   # [B,1,H,W]

        # LR batch: must allow that some dates didn’t have LR
        lr_valid = [x for x in lr_list if x is not None]
        lr_batch = None
        if len(lr_valid) > 0:
            lr_batch = torch.stack([x[0] for x in lr_valid], dim=0).squeeze(1)  # [B,1,HL,WL]
        logger.info(
            "[eval] PSD stacks: HR=%s PMM=%s LR=%s (valid LR dates=%d / %d)",
            tuple(hr_batch.shape),
            tuple(pmm_batch.shape),
            (tuple(lr_batch.shape) if lr_batch is not None else "None"),
            len(lr_valid),
            len(lr_list),
        )
        # ---------- 3) compute PSDs using your centralized function ----------
        # Different grid spacing for HR/PMM vs LR
        dx_km = float(self.eval_cfg.grid_km_per_px)
        dx_lr_km = float(self.eval_cfg.lr_grid_km_per_px)

        # choose normalization here – this is the single source of truth
        psd_norm_mode = getattr(self.eval_cfg, "psd_normalize", "none")

        # Set the Nyquist cutoff frequency based on HR grid spacing
        k_nyquist = 0.5 / dx_km  # cycles per km
        k_nyquist_lr = 0.5 / dx_lr_km  # cycles per km

        hr_psd = compute_isotropic_psd(
            batch=hr_batch, dx_km=dx_km,
            mask=hr_mask, normalize=psd_norm_mode,
            ref_power=None, max_k=k_nyquist
        )
        pmm_psd = compute_isotropic_psd(
            batch=pmm_batch, dx_km=dx_km,
            mask=hr_mask, normalize=psd_norm_mode,
            # we could pass ref_power=hr_psd["psd"] if normalize=="match_ref"
            ref_power=(hr_psd["psd"] if psd_norm_mode == "match_ref" else None),
            max_k=k_nyquist
        )
        lr_psd = None
        if lr_batch is not None:
            lr_psd = compute_isotropic_psd(
                batch=lr_batch, dx_km=dx_km,
                mask=None,
                normalize=psd_norm_mode,
                ref_power=(hr_psd["psd"] if psd_norm_mode == "match_ref" else None),
                max_k=k_nyquist
            )
            # # Remove k and PSD below Nyquist of LR grid
            # valid_idx = lr_psd["k"] <= k_nyquist_lr
            # lr_psd = {k: (v[valid_idx] if isinstance(v, torch.Tensor) else v)
            #           for k, v in lr_psd.items()}


        # ---------- 4) slope summary (what you already had) ----------
        # just reuse your existing helper
        slope_res = compute_psd_slope(gen_bt=pmm_batch, hr_bt=hr_batch,
                                      mask=hr_mask, ignore_low_k_bins=self.eval_cfg.psd_ignore_low_k_bins)
        (tables_dir / "psd_slope_summary.json").write_text(json.dumps(slope_res, indent=2))

        # ---------- 5) save full curves for plotting ----------
        out_npz = {
            "k": hr_psd["k"].numpy(),
            "psd_hr": hr_psd["psd"].numpy(),
            "psd_hr_ci_lo": hr_psd.get("psd_ci_lo", torch.tensor([])).numpy(),
            "psd_hr_ci_hi": hr_psd.get("psd_ci_hi", torch.tensor([])).numpy(),
            "psd_pmm": pmm_psd["psd"].numpy(),
            "psd_pmm_ci_lo": pmm_psd.get("psd_ci_lo", torch.tensor([])).numpy(),
            "psd_pmm_ci_hi": pmm_psd.get("psd_ci_hi", torch.tensor([])).numpy(),
            "normalize": psd_norm_mode,
        }
        if lr_psd is not None:
            out_npz.update({
                "k_lr": lr_psd["k"].numpy(),
                "psd_lr": lr_psd["psd"].numpy(),
                "psd_lr_ci_lo": lr_psd.get("psd_ci_lo", torch.tensor([])).numpy(),
                "psd_lr_ci_hi": lr_psd.get("psd_ci_hi", torch.tensor([])).numpy(),
                "k_nyquist_lr": k_nyquist_lr,
            })

        np.savez_compressed(tables_dir / "psd_curves.npz", **out_npz)
        logger.info("[eval] Wrote PSD curves to %s", tables_dir / "psd_curves.npz")
        logger.info("[eval] Wrote PSD slope to %s", tables_dir / "psd_slope_summary.json")

        try:
            plot_psd_curves_eval(eval_root=str(self.out_root),
                            baseline_eval_dirs=self._get_baseline_eval_dirs())
        except Exception as e:
            logger.warning(f"[eval] Could not plot PSD curves: {e}")

    # ---------- Extremes (basin-mean series) ----------
    def eval_extremes(self, dates_subset: Optional[List[str]] = None):
        """
        Compute extreme value metrics using basin-mean daily series and save tables/figures
        Evaluates: 
            - GEV first for Rx1day/Rx5day with bootstrap CIs
            - POT/GPD over a threshold with bootstrap CIs
        Outputs tables (JSON)
        """
        tables_dir = self.out_root / "tables"
        figs_dir = self.out_root / "figures"
        tables_dir.mkdir(parents=True, exist_ok=True)
        figs_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[eval] Extremes -> GEV (Rx1day/Rx5day), POT/GPD; tables: gev_*.json, pot_*.json")

        # Build daily basin-mean series from HR + PMM
        dates = list(dates_subset if dates_subset is not None else self._list_dates())
        HR = []
        PMM = []
        for date in dates:
            obs = self._load_obs(date)     # [H,W]
            pmm = self._load_pmm(date)     # [H,W]
            if obs is None or pmm is None: continue
            HR.append(obs.unsqueeze(0))
            PMM.append(pmm.unsqueeze(0))
        if len(HR) == 0:
            logger.warning("[eval] No data for extremes.")
            return

        hr_stack  = torch.stack(HR, 0).squeeze(1)   # [T,H,W]
        pmm_stack = torch.stack(PMM, 0).squeeze(1)  # [T,H,W]

        # Prefer a canonical mask; if only per-date masks exist, intersect them
        if self.mask_global is not None:
            basin = self.mask_global
        else:
            # Intersect all per-date masks (logical AND) so extremes compare the same area
            masks = []
            for d in dates:
                md = self._load_mask(d) if self.eval_land_only else None
                if md is not None:
                    masks.append(md)
            basin = None
            if len(masks) > 0:
                mb = masks[0].clone()
                for md in masks[1:]:
                    mb &= md
                basin = mb

        series_hr  = to_numpy_1d_series(hr_stack,  mask=basin, agg="mean")  # mm/day
        series_pmm = to_numpy_1d_series(pmm_stack, mask=basin, agg="mean")

        # Dates to np.datetime64 (assumes YYYY-MM-DD filenames)
        dates_np = np.array([np.datetime64(d) for d in dates])
        blk = seasonal_block_index(dates_np)

        # GEV on Rx1day + Rx5day (seasonal blocks, 4 per year)
        rx1_hr, _  = rxk_series(series_hr,  1, block_index=blk)
        rx1_pmm,_  = rxk_series(series_pmm, 1, block_index=blk)
        rx5_hr,_   = rxk_series(series_hr,  5, block_index=blk)
        rx5_pmm,_  = rxk_series(series_pmm, 5, block_index=blk)

        min_gev_n = 8  # pragmatic minimum; adjust if desired
        if (len(rx1_hr) < min_gev_n) or (len(rx1_pmm) < min_gev_n) or \
        (len(rx5_hr) < min_gev_n) or (len(rx5_pmm) < min_gev_n):
            logger.warning(
                "[eval] Too few block maxima for GEV fitting "
                "(n_rx1_hr=%d, n_rx1_pmm=%d, n_rx5_hr=%d, n_rx5_pmm=%d). Skipping GEV/POT.",
                len(rx1_hr), len(rx1_pmm), len(rx5_hr), len(rx5_pmm)
            )
            return

        def _to_py(obj):
            import numpy as _np
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
            if isinstance(obj, _np.generic):
                return obj.item()
            if isinstance(obj, dict):
                return {k: _to_py(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_py(v) for v in obj]
            return obj

        def _dump(name, res):
            (tables_dir / name).write_text(json.dumps(_to_py(res), indent=2))

        _dump("gev_rx1_hr.json",  fit_gev_block_maxima_with_ci(rx1_hr,  block_per_year=4.0))
        _dump("gev_rx1_pmm.json", fit_gev_block_maxima_with_ci(rx1_pmm, block_per_year=4.0))
        _dump("gev_rx5_hr.json",  fit_gev_block_maxima_with_ci(rx5_hr,  block_per_year=4.0))
        _dump("gev_rx5_pmm.json", fit_gev_block_maxima_with_ci(rx5_pmm, block_per_year=4.0))
        logger.info("[eval] Wrote GEV JSONs (rx1/rx5 for HR and PMM) to %s", tables_dir)

        # POT/GPD threshold (wet-day P95 on HR series)
        wet_hr = series_hr[np.isfinite(series_hr) & (series_hr >= self.eval_cfg.wet_threshold_mm)]
        if wet_hr.size >= 10:
            u = float(np.percentile(wet_hr, 95.0))
            wrote_any = False

            # HR POT
            try:
                res_hr = fit_pot_gpd_with_ci(series_hr, threshold=u)
                _dump("pot_hr.json", res_hr)
                wrote_any = True
            except ValueError as e:
                logger.warning("[eval] POT/HR skipped: %s", e)

            # PMM POT (may have fewer exceedances than HR at same u)
            try:
                res_pmm = fit_pot_gpd_with_ci(series_pmm, threshold=u)
                _dump("pot_pmm.json", res_pmm)
                wrote_any = True
            except ValueError as e:
                logger.warning("[eval] POT/PMM skipped: %s", e)

            if wrote_any:
                logger.info("[eval] Wrote POT JSONs (threshold=HR P95=%.2f mm/day) to %s", u, tables_dir)
            else:
                logger.warning("[eval] POT skipped for both HR and PMM (insufficient exceedances at u=%.2f).", u)
        else:
            logger.warning("[eval] Too few wet HR days (N=%d) to set a robust POT threshold; skipping POT.", wet_hr.size)

    @contextmanager
    def _season_out(self, season_name: str):
        old_root = self.out_root
        try:
            sroot = old_root / 'seasonal' / season_name.lower()
            (sroot / "tables").mkdir(parents=True, exist_ok=True)
            (sroot / "figures").mkdir(parents=True, exist_ok=True)
            self.out_root = sroot
            yield sroot
        finally:
            self.out_root = old_root

    # ---------- Orchestrator ----------
    def run_all(self, do_prob=True, do_cap=True, do_ext=True):
        logger.info("[Eval] === Evaluation plan ===")
        logger.info("       - Probabilistic metrics: %s (CRPS, reliability@%s mm, spread-skill, PIT/rank histograms)",
                    do_prob, ",".join(str(x) for x in self.eval_cfg.thresholds_mm))
        logger.info("       - Capability metrics:    %s (FSS@%s km, PSD slope and curves, P95/P99/wet freq, pooled dists, yearly maps=%s)",
                    do_cap, ",".join(str(x) for x in self.eval_cfg.fss_scales_km), ",".join(self.eval_cfg.yearly_maps))
        logger.info("       - Extremes metrics:      %s (GEV for Rx1day/Rx5day, POT/GPD over threshold %.1f mm)",
                    do_ext, self.eval_cfg.wet_threshold_mm)
        logger.info("       - Seasonal sub-evaluations: %s over %s",
                    self.eval_cfg.seasonal_summaries, ",".join(self.eval_cfg.seasons or []))

        if do_prob:
            self.eval_probabilistic()
        if do_cap:
            self.eval_capability()
            self._eval_psd()
        if do_ext:
            self.eval_extremes()
        
        logger.info("[eval] run_all: do_prob=%s, do_cap=%s, do_ext=%s", do_prob, do_cap, do_ext)
        
        # manifest
        manifest = {
            "gen_dir": str(self.gen_root),
            "out_dir": str(self.out_root),
            "thresholds_mm": list(map(float, self.eval_cfg.thresholds_mm)),
            "fss_scales_km": list(map(float, self.eval_cfg.fss_scales_km)),
            "grid_km_per_px": float(self.eval_cfg.grid_km_per_px),
        }
        (self.out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
        logger.info("[eval] Wrote manifest to %s", self.out_root / "manifest.json")


        # Optional pooled pixel distributions and yearly maps (ALL + seasonal)
        try:
            self._eval_distributions()
        except Exception as e:
            logger.warning(f"[eval] Exception during pooled pixel distributions eval: {e}")
        try:
            self._eval_yearly_maps()
        except Exception as e:
            logger.warning(f"[eval] Exception during yearly maps eval: {e}")

        # Optional seasonal subfolders
        do_seasonal = self.eval_cfg.seasons is not None and len(self.eval_cfg.seasons) > 0
        if do_seasonal:
            self.run_seasonal(do_prob=do_prob, do_cap=do_cap, do_ext=do_ext)


    def run_seasonal(self, do_prob=True, do_cap=True, do_ext=True):
        for season in self.eval_cfg.seasons:
            sel = _filter_dates_by_season(self._list_dates(), season)
            logger.info("[eval][%s] %d dates selected", season, len(sel))
            with self._season_out(season):
                logger.info("[eval][%s] Outputs under: %s", season, self.out_root)
                if do_prob:
                    logger.info("[eval][%s] Running probabilistic metrics...", season)
                    self.eval_probabilistic(dates_subset=sel)
                if do_cap:
                    logger.info("[eval][%s] Running capability metrics...", season)
                    self.eval_capability(dates_subset=sel)
                    self._eval_psd(dates_subset=sel)
                if do_ext:
                    logger.info("[eval][%s] Running extremes metrics...", season)
                    self.eval_extremes(dates_subset=sel)