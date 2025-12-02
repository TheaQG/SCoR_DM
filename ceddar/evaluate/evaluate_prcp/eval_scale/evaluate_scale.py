from __future__ import annotations
from pathlib import Path
from typing import Sequence, Optional, Any, Dict, List

import numpy as np
import torch
import logging

from evaluate.evaluate_prcp.eval_scale.metrics_scale import (
    psd_from_2d,
    compare_psd_triplet,
    compute_fss_at_scales,
    compute_iss_at_scales,
    align_psd_on_k,
)

from evaluate.evaluate_prcp.eval_scale.plot_scale import (
    plot_scale
)

logger = logging.getLogger(__name__)



def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_scale(
        resolver,
        eval_cfg,
        out_root: str | Path,
        *,
        plot_only: bool = False,
) -> None:
    """
        Scale-dependent evaluation (PSD + FSS) for precipitation fields.

        Assumptions:
            Resolver provides:
                - list_dates()
                - load_obs(date) -> HR (DANRA) on HR grid
                - load_gen(date) -> GEN on HR grid
                - load_pmm(date) (optional, may return None) -> PMM on HR grid
                - load_lr(date) (optional, may return None) -> LR already regridded to HR grid 
                                                                but semantically low-res (31 km).
                                                                PSD computed with lr_dx_km.
                - load_mask(date) (optional, may return None) -> land mask on HR grid
        
    """
    out_root = Path(out_root)
    tables_dir = _ensure_dir(out_root / "tables")
    figs_dir = _ensure_dir(out_root / "figures")

    if plot_only:
        plot_scale(out_root, eval_cfg=eval_cfg)
        logger.info("[eval_scale] plot_only=True – done plotting.")
        return

    # ================================================================================
    # 1. read config
    # ================================================================================
    hr_dx_km: float = float(getattr(eval_cfg, "hr_dx_km", 2.5))
    lr_dx_km: float = float(getattr(eval_cfg, "lr_dx_km", 31.0))

    fss_thresholds: Sequence[float] = getattr(eval_cfg, "fss_thresholds_mm", (1.0, 5.0, 10.0))
    fss_scales_km: Sequence[float] = getattr(eval_cfg, "fss_scales_km", (5.0, 10.0, 20.0, 40.0))
    compute_lr_fss: bool = bool(getattr(eval_cfg, "compute_lr_fss", True))

    iss_thresholds: Sequence[float] = getattr(eval_cfg, "iss_thresholds_mm", tuple(fss_thresholds))
    iss_scales_km: Sequence[float] = getattr(eval_cfg, "iss_scales_km", tuple(fss_scales_km))
    compute_lr_iss: bool = bool(getattr(eval_cfg, "compute_lr_iss", compute_lr_fss))

    low_k_max: float = float(getattr(eval_cfg, "low_k_max", 1.0 / 200.0))
    high_k_min: float = float(getattr(eval_cfg, "high_k_min", 1.0 / 20.0))

    # --- ensemble switches (optional) ---
    use_ensemble: bool = bool(getattr(eval_cfg, "use_ensemble", True))
    ensemble_n_members = getattr(eval_cfg, "ensemble_n_members", None)
    ensemble_member_seed = int(getattr(eval_cfg, "ensemble_member_seed", 1234))

    logger.info(f"[eval_scale] use_ensemble={use_ensemble}, n_members={ensemble_n_members}, seed={ensemble_member_seed}")

    dates: List[str] = list(resolver.list_dates())
    if not dates:
        logger.warning("[eval_scale] No dates found from resolver – nothing to do.")
        return

    logger.info(f"[eval_scale] Running on {len(dates)} dates.")

    # Ensemble availability counters
    ens_dates_ok = 0
    ens_dates_badshape = 0
    ens_dates_missing = 0

    # ================================================================================
    # 2. Accumulators 
    # ================================================================================
    psd_hr_list: List[np.ndarray] = []
    psd_gen_list: List[np.ndarray] = []
    psd_lr_list: List[np.ndarray] = []
    psd_lr_hr_list: List[np.ndarray] = [] # LR evaluated on HR spacing (for checking regrid quality)
    psd_have_lr = False
    k_ref: Optional[np.ndarray] = None
    lr_nyquist: Optional[float] = None

    psd_summary_lines: List[str] = [
        "date,hr_lowk,gen_lowk,gen_lowk_vs_hr,gen_lowk_vs_lr,hr_highk,gen_highk,gen_highk_vs_hr"
    ]

    fss_header = ["date", "thr_mm"] + [f"fss_{int(s)}km" for s in fss_scales_km]
    fss_lines: List[str] = [",".join(fss_header)]
    fss_store: Dict[tuple[float, float], List[float]] = {}

    iss_header = ["date", "thr_mm"] + [f"iss_{int(s)}km" for s in iss_scales_km]
    iss_lines: List[str] = [",".join(iss_header)]
    iss_store: Dict[tuple[float, float], List[float]] = {}

    # Ensemble PSD accumulators (store per-day mean over members, then aggregate over days)
    psd_gen_ens_mean_list: List[np.ndarray] = []
    psd_gen_ens_std_list:  List[np.ndarray] = []
    psd_gen_ens_have = False

    # Ensemble FSS/ISS daily rows (like existing *_lines but with _ENS)
    fss_ens_lines: List[str] = [",".join(fss_header)]  # same header: date,thr_mm,fss_5km,...
    iss_ens_lines: List[str] = [",".join(iss_header)]

    # Ensemble summaries (per (thr, scale) collect member means per day → then grand mean/std)
    fss_ens_store: Dict[tuple[float, float], List[float]] = {}
    iss_ens_store: Dict[tuple[float, float], List[float]] = {}

    # ================================================================================
    # 3. Loop over dates
    # ================================================================================

    for d in dates:
        logger.info(f"[eval_scale] Processing {d} ...")
        hr = resolver.load_obs(d)
        gen = resolver.load_pmm(d)
        mask = resolver.load_mask(d)

        if hr is None or gen is None:
            logger.warning(f"[eval_scale] Missing HR or GEN for {d} – skipping.")
            continue

        if not torch.is_tensor(hr):
            hr = torch.from_numpy(np.asarray(hr))
        if not torch.is_tensor(gen):
            gen = torch.from_numpy(np.asarray(gen))
        if mask is not None and not torch.is_tensor(mask):
            mask = torch.from_numpy(np.asarray(mask))
        # determine HR spatial shape early so ensemble validation can use it
        if hr.dim() < 2:
            logger.warning(f"[eval_scale] HR for {d} has shape {tuple(hr.shape)} – expected at least 2D; skipping.")
            continue
        H, W = int(hr.shape[-2]), int(hr.shape[-1])

        # Optional: load ensemble members on HR grid
        gen_members = None
        if use_ensemble:
            # unified helper to try multiple resolver APIs
            def _try_load(name: str):
                fn = getattr(resolver, name, None)
                if not callable(fn):
                    return None
                # try with (date, n, seed)
                try:
                    return fn(d, n=ensemble_n_members, seed=ensemble_member_seed)
                except TypeError:
                    pass
                # try with (date, n_members, seed)
                try:
                    return fn(d, n_members=ensemble_n_members, seed=ensemble_member_seed)
                except TypeError:
                    pass
                # try with (date) only
                try:
                    return fn(d)
                except Exception:
                    return None

            for name in ("load_ens", "load_gen_members", "load_members", "load_ensemble"):
                gen_members = _try_load(name)
                if gen_members is not None:
                    break

        if gen_members is not None:
            # Ensure gen_members is a torch tensor of float type (handles np arrays, lists, or tensors)
            gen_members = torch.as_tensor(gen_members, dtype=torch.float32)
            # common shapes: [M,H,W], [M,1,H,W]; squeeze channel if present
            if gen_members.dim() == 4 and gen_members.shape[1] == 1:
                gen_members = gen_members.squeeze(1)
            # optional subsample if a larger set returned but user asked for n
            if ensemble_n_members is not None and gen_members.shape[0] > int(ensemble_n_members):
                torch.manual_seed(ensemble_member_seed)
                idx = torch.randperm(gen_members.shape[0])[: int(ensemble_n_members)]
                gen_members = gen_members[idx]
                
            if gen_members.dim() != 3 or gen_members.shape[1:] != (H, W):
                ens_dates_badshape += 1
                logger.warning(
                    f"[eval_scale] Ensemble for {d} has shape {tuple(gen_members.shape)}, expected [M,H,W]. Skipping ensemble for this date.")
                gen_members = None
            else:
                ens_dates_ok += 1
        else:
            if use_ensemble:
                ens_dates_missing += 1

        hr = hr.float()
        gen = gen.float()
        H, W = hr.shape
        
        # try LR – ALREADY on HR grid (H,W), but sometimes saved as (1,1,H,W)
        try:
            lr = resolver.load_lr(d)
        except Exception:
            lr = None

        lr_for_fss = None
        if lr is not None:
            if not torch.is_tensor(lr):
                lr = torch.from_numpy(np.asarray(lr))
            lr = lr.float()

            # --- make LR shape-robust ---
            # possible shapes we have seen:
            # (H, W)
            # (1, H, W)
            # (1, 1, H, W)
            orig_shape = tuple(lr.shape)
            # 1) drop leading singleton dims
            while lr.dim() > 2 and lr.shape[0] == 1:
                lr = lr.squeeze(0)
            # 2) if we now have e.g. (1, H, W) again, drop again
            if lr.dim() > 2 and lr.shape[0] == 1:
                lr = lr.squeeze(0)
            # 3) if there are still >2 dims, assume last two are spatial
            if lr.dim() > 2:
                lr = lr.reshape(*lr.shape[-2:])                

            if lr.shape == (H, W):
                lr_for_fss = lr
            else:
                logger.warning(
                    f"[eval_scale] LR for {d} has shape {orig_shape} -> coerced to {tuple(lr.shape)} "
                    f"but still != HR shape {(H, W)}. Will use it for PSD (with lr_dx_km) but SKIP LR FSS/ISS baseline.")

        # ================================================================================
        # 3a. PSD computation and comparison
        # ================================================================================ 
        hr_psd = psd_from_2d(
            hr,
            dx_km=hr_dx_km,
            mask_2d=mask,
            window="hann",
            detrend="none",
            normalize="none",
        )
        gen_psd = psd_from_2d(
            gen,
            dx_km=hr_dx_km,
            mask_2d=mask,
            window="hann",
            detrend="none",
            normalize="none",
        )

        if lr is not None:
            # even if LR is on HR grid, we **evaluate** it with its native spacing
            lr_psd = psd_from_2d(
                lr,
                dx_km=lr_dx_km,
                mask_2d=None,          # you *can* pass a coarsened mask here later
                window="hann",
                detrend="none",
                normalize="none",
            )
        else:
            lr_psd = None
        
        # also compute LR PSD on HR spacing so we can visualize LR beyond its Nyquist
        if lr is not None:
            lr_psd_hr = psd_from_2d(
                lr,
                dx_km=hr_dx_km,
                mask_2d=None,
                window="hann",
                detrend="none",
                normalize="none",
            )
        else:
            lr_psd_hr = None

        comp = compare_psd_triplet(
            hr_psd=hr_psd,
            gen_psd=gen_psd,
            lr_psd=lr_psd,
            low_k_max=low_k_max,
            high_k_min=high_k_min,
        )

        k_this = np.asarray(comp["k"])
        if k_ref is None:
            k_ref = k_this
        # ensure PSD entries are numpy arrays (convert scalars to ndarrays)
        psd_hr_arr = np.asarray(comp["psd_hr"])
        psd_hr_list.append(psd_hr_arr)
        psd_gen_arr = np.asarray(comp.get("psd_gen", np.zeros_like(psd_hr_arr)))
        psd_gen_list.append(psd_gen_arr)
        if "psd_lr" in comp:
            psd_have_lr = True
            psd_lr_arr = np.asarray(comp["psd_lr"])
            psd_lr_list.append(psd_lr_arr)
            lr_nyquist = float(comp.get("lr_nyquist", 0.0))
        else:
            psd_lr_arr = np.zeros_like(psd_hr_arr)
            psd_lr_list.append(psd_lr_arr)

        # store LR-on-HR-grid PSD (or zeros if we didn't have LR)
        if lr_psd_hr is not None:
            psd_lr_hr_list.append(np.asarray(lr_psd_hr["psd"]))
        else:
            psd_lr_hr_list.append(np.zeros_like(psd_hr_arr))

        psd_summary_lines.append(
            ",".join([
                d,
                f"{comp.get('hr_lowk_power', np.nan):.6e}",
                f"{comp.get('gen_lowk_power', np.nan):.6e}",
                f"{comp.get('gen_lowk_vs_hr', np.nan):.6f}",
                f"{comp.get('gen_lowk_vs_lr', np.nan):.6f}" if "gen_lowk_vs_lr" in comp else "",
                f"{comp.get('hr_highk_power', np.nan):.6e}",
                f"{comp.get('gen_highk_power', np.nan):.6e}",
                f"{comp.get('gen_highk_vs_hr', np.nan):.6f}",
            ])
        )

        # ---------------- Ensemble PSD on HR spacing ----------------
        if gen_members is not None:
            M = int(gen_members.shape[0])
            psd_m_list = []
            for mi in range(M):
                psd_m = psd_from_2d(
                    gen_members[mi],
                    dx_km=hr_dx_km,
                    mask_2d=mask,
                    window="hann",
                    detrend="none",
                    normalize="none",
                )
                # align member PSD to HR k-grid used this date
                Pm_on_hr = align_psd_on_k(psd_m, k_this)
                psd_m_list.append(Pm_on_hr)
            psd_m_arr = np.stack(psd_m_list, axis=0)  # [M,K]
            psd_gen_ens_have = True
            psd_gen_ens_mean_list.append(psd_m_arr.mean(axis=0))
            psd_gen_ens_std_list.append(psd_m_arr.std(axis=0, ddof=1) if M > 1 else np.zeros_like(psd_m_arr[0]))

        # ================================================================================
        # 3b. FSS computation: GEN vs HR 
        # ================================================================================
        gen_b = gen.view(1, 1, H, W)
        hr_b = hr.view(1, 1, H, W)
        mask_b = mask.view(1, 1, H, W) if mask is not None else None

        for thr in fss_thresholds:
            fss_dict = compute_fss_at_scales(
                gen_bt=gen_b,
                hr_bt=hr_b,
                mask=mask_b,
                grid_km_per_px=hr_dx_km,
                fss_km=list(fss_scales_km),
                thr_mm=float(thr),
            )
            line_vals = [d, f"{float(thr):.2f}"] + [
                f"{float(fss_dict[f'{int(s)}km']):.6f}" for s in fss_scales_km
            ]
            fss_lines.append(",".join(line_vals))

            for s in fss_scales_km:
                key = (float(thr), float(s))
                fss_store.setdefault(key, []).append(float(fss_dict[f"{int(s)}km"]))

        for thr in iss_thresholds:
            iss_dict = compute_iss_at_scales(
                gen_bt=gen_b,
                hr_bt=hr_b,
                mask=mask_b,
                grid_km_per_px=hr_dx_km,
                iss_scales_km=list(iss_scales_km),
                thr_mm=float(thr),
            )
            line_vals = [d, f"{float(thr):.2f}"] + [
                f"{float(iss_dict[f'{int(s)}km']):.6f}" for s in iss_scales_km
            ]
            iss_lines.append(",".join(line_vals))

            for s in iss_scales_km:
                key = (float(thr), float(s))
                iss_store.setdefault(key, []).append(float(iss_dict[f"{int(s)}km"]))

        # ---------------- Ensemble FSS/ISS ----------------
        if gen_members is not None:
            M = int(gen_members.shape[0])
            # Compute per-member FSS/ISS, then record the *member-mean* per scale per date
            for thr in fss_thresholds:
                vals_per_scale: Dict[float, List[float]] = {float(s): [] for s in fss_scales_km}
                for mi in range(M):
                    gb = gen_members[mi].view(1,1,H,W)
                    fss_m = compute_fss_at_scales(
                        gen_bt=gb, hr_bt=hr_b, mask=mask_b,
                        grid_km_per_px=hr_dx_km,
                        fss_km=list(fss_scales_km),
                        thr_mm=float(thr),
                    )
                    for s in fss_scales_km:
                        vals_per_scale[float(s)].append(float(fss_m[f"{int(s)}km"]))
                # member-mean row (same shape as GEN row)
                line_vals = [d, f"{float(thr):.2f}"] + [f"{float(np.mean(vals_per_scale[float(s)])):.6f}" for s in fss_scales_km]
                fss_ens_lines.append(",".join(line_vals))
                # store for summary
                for s in fss_scales_km:
                    key = (float(thr), float(s))
                    fss_ens_store.setdefault(key, []).append(float(np.mean(vals_per_scale[float(s)])))

            for thr in iss_thresholds:
                vals_per_scale: Dict[float, List[float]] = {float(s): [] for s in iss_scales_km}
                for mi in range(M):
                    gb = gen_members[mi].view(1,1,H,W)
                    iss_m = compute_iss_at_scales(
                        gen_bt=gb, hr_bt=hr_b, mask=mask_b,
                        grid_km_per_px=hr_dx_km,
                        iss_scales_km=list(iss_scales_km),
                        thr_mm=float(thr),
                    )
                    for s in iss_scales_km:
                        vals_per_scale[float(s)].append(float(iss_m[f"{int(s)}km"]))
                line_vals = [d, f"{float(thr):.2f}"] + [f"{float(np.mean(vals_per_scale[float(s)])):.6f}" for s in iss_scales_km]
                iss_ens_lines.append(",".join(line_vals))
                for s in iss_scales_km:
                    key = (float(thr), float(s))
                    iss_ens_store.setdefault(key, []).append(float(np.mean(vals_per_scale[float(s)])))

        # ===============================================================================
        # 3c. Optional: LR baseline FSS 
        if compute_lr_fss and lr_for_fss is not None:
            lr_b = lr_for_fss.view(1, 1, H, W)
            for thr in fss_thresholds:
                fss_lr = compute_fss_at_scales(
                    gen_bt=lr_b,       # yes, treat LR-on-HR-grid as "gen"
                    hr_bt=hr_b,
                    mask=mask_b,
                    grid_km_per_px=hr_dx_km,
                    fss_km=list(fss_scales_km),
                    thr_mm=float(thr),
                )
                line_vals = [d, f"{float(thr):.2f}_LR"] + [
                    f"{float(fss_lr[f'{int(s)}km']):.6f}" for s in fss_scales_km
                ]
                fss_lines.append(",".join(line_vals))
        
        if compute_lr_iss and lr_for_fss is not None:
            lr_b = lr_for_fss.view(1, 1, H, W)
            for thr in iss_thresholds:
                iss_lr = compute_iss_at_scales(
                    gen_bt=lr_b,      # treat LR as “forecast”
                    hr_bt=hr_b,
                    mask=mask_b,
                    grid_km_per_px=hr_dx_km,
                    iss_scales_km=list(iss_scales_km),
                    thr_mm=float(thr),
                )
                line_vals = [d, f"{float(thr):.2f}_LR"] + [
                    f"{float(iss_lr[f'{int(s)}km']):.6f}" for s in iss_scales_km
                ]
                iss_lines.append(",".join(line_vals))
    # ================================================================================
    # 4. After-loop: write outputs
    # ================================================================================
    (tables_dir / "scale_psd_summary.csv").write_text("\n".join(psd_summary_lines))

    if k_ref is not None:
        # stack to [N_days, K]
        psd_hr_arr    = np.stack(psd_hr_list,    axis=0)
        psd_gen_arr   = np.stack(psd_gen_list,   axis=0)
        psd_lr_arr    = np.stack(psd_lr_list,    axis=0)
        psd_lr_hr_arr = np.stack(psd_lr_hr_list, axis=0)
        N_days = psd_hr_arr.shape[0]

        # day-mean + 95% CI for HR / GEN (to be used directly by the plotter)
        def _mean_std_ci(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            mean = arr.mean(axis=0)
            if N_days > 1:
                std = arr.std(axis=0, ddof=1)
                se = std / np.sqrt(N_days)
                ci_lo = mean - 1.96 * se
                ci_hi = mean + 1.96 * se
            else:
                std = np.zeros_like(mean)
                ci_lo = mean.copy()
                ci_hi = mean.copy()
            return ci_lo, ci_hi, std

        hr_ci_lo, hr_ci_hi, hr_std = _mean_std_ci(psd_hr_arr)
        gen_ci_lo, gen_ci_hi, gen_std = _mean_std_ci(psd_gen_arr)
        
        np.savez_compressed(
            tables_dir / "scale_psd_curves.npz",
            k=k_ref,
            psd_hr=psd_hr_arr,          # [N_days, K]
            psd_gen=psd_gen_arr,        # [N_days, K]
            psd_lr=psd_lr_arr,          # [N_days, K]
            psd_lr_hr=psd_lr_hr_arr,    # [N_days, K]
            psd_hr_ci_lo=hr_ci_lo,      # [K]
            psd_hr_ci_hi=hr_ci_hi,      # [K]
            psd_gen_ci_lo=gen_ci_lo,    # [K]
            psd_gen_ci_hi=gen_ci_hi,    # [K]
            dates=np.array(dates, dtype="U"),
            lr_nyquist=np.array(0.0 if lr_nyquist is None else lr_nyquist),
        )
    if use_ensemble:
        logger.info(
            f"[eval_scale] Ensemble availability: ok={ens_dates_ok}, badshape={ens_dates_badshape}, missing={ens_dates_missing} (out of {len(dates)} dates)")
        if ens_dates_ok == 0:
            logger.warning("[eval_scale] No ensemble data were processed -> ensemble curves/tables will be absent.")
    # ---------------- Ensemble PSD curves (mean over members per day, then over days) ----------------
    if k_ref is not None and psd_gen_ens_have and psd_gen_ens_mean_list:
        psd_gen_ens_mean = np.stack(psd_gen_ens_mean_list, axis=0)  # [N_days,K]
        psd_gen_ens_std  = np.stack(psd_gen_ens_std_list,  axis=0)  # [N_days,K]
        N_days_ens = psd_gen_ens_mean.shape[0]

        # mean over days; SE for CI across days (member spread already inside std term but we just show day-to-day SE)
        gen_ens_mean = psd_gen_ens_mean.mean(axis=0)
        if N_days_ens > 1:
            gen_ens_std_days = psd_gen_ens_mean.std(axis=0, ddof=1)
            se = gen_ens_std_days / np.sqrt(N_days_ens)
            gen_ens_ci_lo = gen_ens_mean - 1.96 * se
            gen_ens_ci_hi = gen_ens_mean + 1.96 * se
        else:
            gen_ens_std_days = np.zeros_like(gen_ens_mean)
            gen_ens_ci_lo = gen_ens_mean.copy()
            gen_ens_ci_hi = gen_ens_mean.copy()

        # append to the same NPZ (new keys); re-save is fine here
        with np.load(tables_dir / "scale_psd_curves.npz") as _npz_existing:
            existing = {k: _npz_existing[k] for k in _npz_existing.files}
        np.savez_compressed(
            tables_dir / "scale_psd_curves.npz",
            **existing,
            psd_gen_ens_mean=gen_ens_mean,   # [K]
            psd_gen_ens_ci_lo=gen_ens_ci_lo, # [K]
            psd_gen_ens_ci_hi=gen_ens_ci_hi, # [K]
        )

    (tables_dir / "scale_fss_daily.csv").write_text("\n".join(fss_lines))
    (tables_dir / "scale_iss_daily.csv").write_text("\n".join(iss_lines))
    # Ensemble daily tables
    (tables_dir / "scale_fss_ens_daily.csv").write_text("\n".join(fss_ens_lines))
    (tables_dir / "scale_iss_ens_daily.csv").write_text("\n".join(iss_ens_lines))

    # Ensemble summaries (mean over days of per-day member means)
    fss_ens_summ = [",".join(["thr_mm"] + [f"fss_{int(s)}km" for s in fss_scales_km])]
    for thr in fss_thresholds:
        row = [f"{float(thr):.2f}"]
        for s in fss_scales_km:
            vals = fss_ens_store.get((float(thr), float(s)), [])
            row.append(f"{float(np.mean(vals)):.6f}" if vals else "")
        fss_ens_summ.append(",".join(row))
    (tables_dir / "scale_fss_ens_summary.csv").write_text("\n".join(fss_ens_summ))

    iss_ens_summ = [",".join(["thr_mm"] + [f"iss_{int(s)}km" for s in iss_scales_km])]
    for thr in iss_thresholds:
        row = [f"{float(thr):.2f}"]
        for s in iss_scales_km:
            vals = iss_ens_store.get((float(thr), float(s)), [])
            row.append(f"{float(np.mean(vals)):.6f}" if vals else "")
        iss_ens_summ.append(",".join(row))
    (tables_dir / "scale_iss_ens_summary.csv").write_text("\n".join(iss_ens_summ))
    
    # FSS summary
    summ_lines = [",".join(["thr_mm"] + [f"fss_{int(s)}km" for s in fss_scales_km])]
    for thr in fss_thresholds:
        row = [f"{float(thr):.2f}"]
        for s in fss_scales_km:
            vals = fss_store.get((float(thr), float(s)), [])
            if vals:
                row.append(f"{float(np.mean(vals)):.6f}")
            else:
                row.append("")
        summ_lines.append(",".join(row))
    (tables_dir / "scale_fss_summary.csv").write_text("\n".join(summ_lines))

    iss_summ_lines = [",".join(["thr_mm"] + [f"iss_{int(s)}km" for s in iss_scales_km])]
    for thr in iss_thresholds:
        row = [f"{float(thr):.2f}"]
        for s in iss_scales_km:
            vals = iss_store.get((float(thr), float(s)), [])
            if vals:
                row.append(f"{float(np.mean(vals)):.6f}")
            else:
                row.append("")
        iss_summ_lines.append(",".join(row))
    (tables_dir / "scale_iss_summary.csv").write_text("\n".join(iss_summ_lines))

    # --- Compact overview table across metrics ---
    try:
        import statistics as _st
        overview = ["metric,detail,value"]
        # FSS (PMM) mean across all (thr,scale)
        fss_vals = []
        for _, vals in fss_store.items():
            fss_vals.extend(vals)
        if fss_vals:
            overview.append(f"FSS,gen_mean,{_st.mean(fss_vals):.6f}")
        # ISS (PMM) mean across all (thr,scale)
        iss_vals = []
        for _, vals in iss_store.items():
            iss_vals.extend(vals)
        if iss_vals:
            overview.append(f"ISS,gen_mean,{_st.mean(iss_vals):.6f}")
        # Ensemble means if available
        if fss_ens_store:
            all_ens = []
            for _, vals in fss_ens_store.items():
                all_ens.extend(vals)
            if all_ens:
                overview.append(f"FSS_ens,mean,{_st.mean(all_ens):.6f}")
        if iss_ens_store:
            all_ens = []
            for _, vals in iss_ens_store.items():
                all_ens.extend(vals)
            if all_ens:
                overview.append(f"ISS_ens,mean,{_st.mean(all_ens):.6f}")
        # pointers to PSD summaries (written by plotter)
        overview.append("PSD,band_ratio_csv,scale_psd_band_ratios_avg.csv")
        overview.append("PSD,slopes_csv,scale_psd_slopes_avg.csv")
        (tables_dir / "scale_overview.csv").write_text("\n".join(overview))
    except Exception as e:
        logger.warning(f"[eval_scale] Failed to write overview CSV: {e}")

    if use_ensemble:
        logger.info(
            f"[eval_scale] Ensemble availability: ok={ens_dates_ok}, badshape={ens_dates_badshape}, missing={ens_dates_missing} (out of {len(dates)} dates)")
        if ens_dates_ok == 0:
            logger.warning("[eval_scale] No ensemble data were processed -> PSD/FSS/ISS ensemble plots will not appear.")

    logger.info(f"[eval_scale] Done. Outputs at: {out_root}")

    # ===============================================================================
    # 5. Plots
    # ===============================================================================
    make_plots = bool(getattr(eval_cfg, "make_plots", True))
    if make_plots:
        plot_scale(out_root, eval_cfg=eval_cfg)
        logger.info(f"[eval_scale] Plots saved to {figs_dir}")