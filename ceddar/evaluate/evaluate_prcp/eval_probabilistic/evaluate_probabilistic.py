"""
    Probabilistic metrics for precipitation evaluation.
    Gathers computation and plotting functions.

"""

from __future__ import annotations
from pathlib import Path
from typing import Sequence, Optional, Dict, Any, List

import numpy as np
import torch
import logging

# --- Local mask normalization utility for shape-safe accumulation ---
def _normalize_mask_local(mask, target_shape, device):
    """
    Normalize `mask` to shape == target_shape (broadcast-style).
    Accepts mask of shapes [H,W], [1,H,W], or [B,H,W] with B==1.
    Returns a boolean tensor on `device` with shape == target_shape.
    """
    if mask is None:
        return None
    m = mask
    if not torch.is_tensor(m):
        m = torch.as_tensor(m)
    m = m.to(device)
    if m.dtype != torch.bool:
        m = m > 0.5

    # Squeeze leading singleton batch/channel dims
    while m.dim() > 2 and m.shape[0] == 1:
        m = m.squeeze(0)

    # Normalize target_shape to a tuple (it may be torch.Size)
    ts = tuple(target_shape)

    # 1) Exact shape match
    if m.shape == ts:
        return m

    # 2) [H,W] -> [B,H,W], but only if H,W actually match
    if m.dim() == 2 and len(ts) == 3:
        B, H, W = ts
        if m.shape == (H, W):
            return m.unsqueeze(0).expand(B, -1, -1)

    # 3) [1,H,W] -> [H,W], only if H,W match
    if m.dim() == 3 and len(ts) == 2 and m.shape[0] == 1 and m.shape[1:] == ts:
        return m.squeeze(0)

    # 4) Last resort: attempt broadcast; on failure propagate the error
    try:
        return m.expand(ts)
    except Exception as e:
        raise ValueError(
            f"[prob_eval] Cannot normalize mask of shape {tuple(mask.shape)} "
            f"to target {ts}: {e}"
        )

logger = logging.getLogger(__name__)

from evaluate.evaluate_prcp.eval_probabilistic.metrics_probabilistic import (
    crps_ensemble,
    pit_values_from_ensemble,
    rank_histogram,
    reliability_exceedance_binned,
    aggregate_reliability_bins,
    spread_skill_binned,
    energy_score,
    variogram_score,
)
from evaluate.evaluate_prcp.eval_probabilistic.plot_probabilistic import (
    plot_probabilistic,
)

# Helper: select CRPS example dates (best/worst, deprioritize zeros)
def _select_crps_example_dates(crps_csv_path: Path, n_examples: int = 6) -> list[tuple[str, float]]:
    lines = crps_csv_path.read_text().strip().splitlines()
    if len(lines) <= 1:
        return []
    rows: list[tuple[str, float]] = []
    for ln in lines[1:]:
        s = ln.split(",")
        if len(s) < 2:
            continue
        date_s = s[0].strip()
        try:
            crps_v = float(s[1])
        except Exception:
            continue
        rows.append((date_s, crps_v))
    if not rows:
        return []
    rows_sorted = sorted(rows, key=lambda x: x[1])
    eps = 0.01
    nonzero_rows = [r for r in rows_sorted if r[1] > eps]
    zero_rows = [r for r in rows_sorted if r[1] <= eps]
    n_half = max(1, n_examples // 2)
    if len(nonzero_rows) >= n_half:
        best = nonzero_rows[:n_half]
    else:
        best = nonzero_rows + zero_rows[:(n_half - len(nonzero_rows))]
    worst = rows_sorted[-n_half:]
    return best + worst

# Helper: build and save member-based CRPS example payload for plotting
def _build_and_save_crps_examples_members(
    resolver,
    tables_dir: Path,
    *,
    n_members_to_show: int = 4,
    member_seed: int = 1234,
) -> None:
    crps_csv = tables_dir / "prob_crps_daily.csv"
    if not crps_csv.exists():
        logger.info(f"CRPS timeseries file not found: {crps_csv}")
        return
    selected = _select_crps_example_dates(crps_csv, n_examples=6)
    if not selected:
        return

    rng = np.random.RandomState(member_seed)
    payload: Dict[str, Any] = {"dates": np.array([d for d, _ in selected])}

    for d, _ in selected:
        obs = resolver.load_obs(d)            # [H,W]
        ens = resolver.load_ens(d)            # [M,H,W]
        try:
            pmm = resolver.load_pmm(d)        # [H,W] (optional but desired)
        except Exception:
            pmm = None
        try:
            mask = resolver.load_mask(d)
        except Exception:
            mask = None

        if obs is None or ens is None:
            continue

        obs_t = torch.from_numpy(np.asarray(obs)) if not torch.is_tensor(obs) else obs
        ens_t = torch.from_numpy(np.asarray(ens)) if not torch.is_tensor(ens) else ens

        # Ensemble CRPS (scalar) for the column title
        crps_val = crps_ensemble(obs_t, ens_t, mask=mask, reduction="mean")
        payload[f"CRPS_ENS_{d}"] = float(crps_val)

        # Save HR (always)
        payload[f"HR_{d}"] = np.asarray(obs_t.cpu()).astype(np.float32)

        # Save PMM if available (for last row)
        if pmm is not None:
            payload[f"PMM_{d}"] = np.asarray(pmm).astype(np.float32)

        # Choose members to display
        M = ens_t.shape[0]
        take = min(n_members_to_show, M)
        idx = rng.choice(M, size=take, replace=False)

        # Optional mask for MAE
        m = None
        if mask is not None:
            m = np.asarray(mask).astype(bool)

        # Save members and their MAE (CRPS for a single member reduces to MAE)
        hr_np = payload[f"HR_{d}"]
        for j, k in enumerate(idx):
            mem = ens_t[k].cpu().numpy().astype(np.float32)
            payload[f"MEM_{j}_{d}"] = mem
            if m is not None:
                mae = np.nanmean(np.abs(np.where(m, mem, np.nan) - np.where(m, hr_np, np.nan)))
            else:
                mae = float(np.mean(np.abs(mem - hr_np)))
            payload[f"MAE_MEM_{j}_{d}"] = float(mae)

    # Ensure all payload values are array-like before saving to satisfy type checkers
    np.savez_compressed(
        tables_dir / "prob_crps_examples_members.npz",
        **{k: np.asarray(v) for k, v in payload.items()}
    )

def run_probabilistic(
        resolver,
        eval_cfg,
        out_root: str | Path,
        *,
        plot_only: bool = False,
) -> None:
    """
        Main entry for precipitation probabilistic evaluation.

        Parameters:
            resolver
                Object that knows how to access evaluation data. Must provide:
                    - list_dates()
                    - load_obs(date)
                    - load_ens(date)
                    - load_pmm(date) (optional, may return None)
                    - load_mask(date) (optional, may return None)
            eval_cfg
                Config-like object with evaluation settings.
                    - thresholds_mm
                    - reliability_bins
                    - spread_skill_bins
                    - pit_bins
            out_root
                Directory to save outputs to.
            plot_only
                If True, only generate plots from (existing) data.
    """
    out_root = Path(out_root)
    tables_dir = out_root / "tables"
    figs_dir = out_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # if user only wants plots, just read + plot
    if plot_only:
        thresholds = getattr(eval_cfg, "thresholds_mm", (1.0, 5.0, 10.0))
        pit_bins = int(getattr(eval_cfg, "pit_bins", 20))
        plot_probabilistic(out_root, thresholds=thresholds, pit_bins=pit_bins)
        return
    
    # ================================================================================
    # 1) setup / config
    # ================================================================================
    thresholds: Sequence[float] = getattr(eval_cfg, "thresholds_mm", (1.0, 5.0, 10.0))
    n_rel_bins: int = int(getattr(eval_cfg, "reliability_bins", 10))
    n_ss_bins: int = int(getattr(eval_cfg, "spread_skill_bins", 10))
    pit_bins: int = int(getattr(eval_cfg, "pit_bins", 20))

    vs_p: float = float(getattr(eval_cfg, "variogram_p", 0.5))
    vs_max_pairs: int = int(getattr(eval_cfg, "variogram_max_pairs", 4000))

    dates: List[str] = list(resolver.list_dates())

    # data accumulators
    crps_lines: List[str] = ["date,crps"]
    es_lines: List[str] = ["date,energy_score"]
    vs_lines: List[str] = ["date,variogram_score"]
    all_pit: List[np.ndarray] = []
    rank_acc: Optional[torch.Tensor] = None
    rel_acc: Dict[float, List[Dict[str, torch.Tensor]]] = {float(t): [] for t in thresholds}
    ss_lines: List[str] = ["date,spread_mean,skill_mean"]
    # PMM (deterministic) CRPS equivalent (MAE)
    pmm_mae_lines: List[str] = ["date,pmm_mae"]

    # --- Accumulator shape holders ---
    H_acc: Optional[int] = None
    W_acc: Optional[int] = None

    # ================================================================================
    # 2) per-date loop
    # ================================================================================
    _crps_sum = None
    _crps_cnt = None
    for d in dates:
        logger.info(f"[eval_probabilistic] Processing date {d} ...")
        # load

        obs = resolver.load_obs(d)     # [H,W]
        ens = resolver.load_ens(d)     # [M,H,W]
        if obs is None or ens is None:
            # skip incomplete samples
            continue

        mask = resolver.load_mask(d)   # [H,W] or None
        # Debug: log raw mask shape if available
        if mask is not None and hasattr(mask, 'shape'):
            logger.debug(f"[eval_probabilistic] date={d} raw mask shape={getattr(mask, 'shape', None)}")
        pmm  = None
        # optional PMM (prefer phys)
        try:
            pmm = resolver.load_pmm(d)
        except Exception:
            pmm = None

        # make sure tensors are on a device (CPU is fine here)
        if not torch.is_tensor(obs):
            obs = torch.from_numpy(obs)
        if not torch.is_tensor(ens):
            ens = torch.from_numpy(ens)

        # Debug: log tensor shapes after conversion
        logger.debug(f"[eval_probabilistic] date={d} shapes: obs={tuple(obs.shape)}, ens={tuple(ens.shape)}, mask={getattr(mask, 'shape', None)}")

        # 2.0a PMM "CRPS" (equals MAE for deterministic forecast)
        if pmm is not None:
            if not torch.is_tensor(pmm):
                pmm_t = torch.from_numpy(np.asarray(pmm))
            else:
                pmm_t = pmm
            pmm_t = pmm_t.to(obs.device, obs.dtype)
            diff = (pmm_t - obs).abs()
            if mask is not None:
                try:
                    m_mae = _normalize_mask_local(mask, obs.shape, device=obs.device)
                    val = float(diff[m_mae].mean().item()) if m_mae is not None and m_mae.any() else float(diff.mean().item())
                except Exception:
                    val = float(diff.mean().item())
            else:
                val = float(diff.mean().item())
            pmm_mae_lines.append(f"{d},{val:.6f}")

        # 2.1 CRPS (domain average, masked)
        crps_val = crps_ensemble(obs, ens, mask=mask, reduction="mean")
        crps_lines.append(f"{d},{float(crps_val):.6f}")

        # # Energy score (field-wise, multivariate generalization of CRPS)
        # es_val = energy_score(obs, ens, mask=mask)
        # es_lines.append(f"{d},{float(es_val):.6f}")
        
        # # Variogram score (spatial dependence)
        # vs_val = variogram_score(
        #     obs,
        #     ens,
        #     mask=mask,
        #     p=vs_p,
        #     max_pairs=vs_max_pairs,
        #     seed=0,
        # )
        # vs_lines.append(f"{d},{float(vs_val):.6f}")

        # 2.1b CRPS map (for spatial mean later): get FULL field; apply mask only during accumulation
        crps_map = crps_ensemble(obs, ens, mask=None, reduction="none")  # [H,W], never flattened by mask
        # Ensure 2D [H,W]
        if crps_map.dim() == 1:
            N = int(crps_map.numel())
            if obs.dim() == 2 and obs.numel() == N:
                crps_map = crps_map.view(obs.shape[0], obs.shape[1])
            else:
                s = int(round(N ** 0.5))
                if s * s == N:
                    crps_map = crps_map.view(s, s)
                else:
                    raise RuntimeError(f"[eval_probabilistic] crps_map is 1D (len={N}) and cannot infer [H,W].")
        # Normalize mask to crps_map shape for accumulation
        m_acc = None
        try:
            m_acc = _normalize_mask_local(mask, crps_map.shape, device=crps_map.device) if mask is not None else None
        except Exception as e:
            logger.warning(f"[eval_probabilistic] Mask normalization failed on date {d}: {e}. Falling back to no mask for accumulation.")
            m_acc = None

        # === Masked accumulation (land-only) ===
        # Prepare weights for accumulation: 1 on valid (land), 0 on invalid (ocean)
        if m_acc is not None:
            w = m_acc.to(dtype=torch.float32)
        else:
            w = torch.ones_like(crps_map, dtype=torch.float32)

        if _crps_sum is None:
            # Initialize masked sum and counts
            _crps_sum = (crps_map * w).clone().float()
            _crps_cnt = w.clone().float()
            # Record accumulator spatial shape
            H_acc, W_acc = int(crps_map.shape[0]), int(crps_map.shape[1])
        else:
            # Ensure shapes match (broadcast if needed)
            if w.shape != _crps_sum.shape:
                logger.warning(f"[eval_probabilistic] Accumulator shape mismatch: sum={_crps_sum.shape}, w={w.shape}. Attempting to broadcast.")
                try:
                    w = w.expand_as(_crps_sum)
                except Exception:
                    logger.error(f"[eval_probabilistic] Failed to broadcast weights to {_crps_sum.shape}; falling back to all-ones for this date {d}.")
                    w = torch.ones_like(_crps_sum, dtype=torch.float32)

            # Masked accumulation (land-only)
            _crps_sum = _crps_sum + crps_map * w
            _crps_cnt = _crps_cnt + w

        # 2.2 PIT
        # metrics expect B>1, so wrap a batch dimension
        pits = pit_values_from_ensemble(
            obs.unsqueeze(0),            # [1,H,W]
            ens.unsqueeze(0),            # [1,M,H,W]
            mask=mask,
            randomized=True,
        )
        all_pit.append(pits.numpy())

        # 2.3 Rank histogram
        rh = rank_histogram(
            obs.unsqueeze(0),
            ens.unsqueeze(0),
            mask=mask,
            randomize_ties=True,
        )
        if rank_acc is None:
            rank_acc = rh
        else:
            rank_acc = rank_acc + rh

        # 2.4 Reliability per threshold
        for thr in thresholds:
            rel = reliability_exceedance_binned(
                obs=obs,
                ens=ens,
                threshold=float(thr),
                lr_covariate=None,               # hook for LR-stratified plots later
                n_bins=n_rel_bins,
                mask=mask,
                return_brier=True,
            )
            rel_acc[float(thr)].append(rel)

        # 2.5 Spread–skill
        ss = spread_skill_binned(
            obs=obs,
            ens=ens,
            point_field=pmm,              # use PRECOMPUTED PMM if available
            point="mean",                 # fallback if pmm is None
            mask=mask,
            n_bins=n_ss_bins,
        )
        # We want a per-date SINGLE number (for the time series plot),
        # so take a simple count-weighted mean over bins:
        cnt = ss["count"].numpy().astype(np.int64)
        spr = ss["spread"].numpy()
        skl = ss["skill"].numpy()
        w = cnt.clip(min=0)
        if w.sum() > 0:
            spread_mean = float((spr * w).sum() / w.sum())
            skill_mean = float((skl * w).sum() / w.sum())
        else:
            spread_mean = 0.0
            skill_mean = 0.0
        ss_lines.append(f"{d},{spread_mean:.6f},{skill_mean:.6f}")
    
    
    # ================================================================================
    # 3) write output tables
    # ================================================================================

    # 3.1 CRPS (daily domain-mean values)
    (tables_dir / "prob_crps_daily.csv").write_text("\n".join(crps_lines))

    # 3.1b temporally averaged CRPS map
    if _crps_sum is not None and _crps_cnt is not None:
        logger.debug(f"[eval_probabilistic] Final accumulators: sum={getattr(_crps_sum, 'shape', None)}, cnt={getattr(_crps_cnt, 'shape', None)}")
        if _crps_sum.shape != _crps_cnt.shape:
            logger.warning(f"[eval_probabilistic] Final shapes differ before mean: sum={_crps_sum.shape}, cnt={_crps_cnt.shape}. Trying to broadcast cnt.")
            try:
                _crps_cnt = _crps_cnt.expand_as(_crps_sum)
            except Exception as e:
                raise RuntimeError(f"[eval_probabilistic] Cannot align CRPS accumulators for mean: sum={_crps_sum.shape}, cnt={_crps_cnt.shape}: {e}")
        # === Land-only mean (avoid dividing over ocean) ===
        cnt = _crps_cnt
        if _crps_sum.shape != cnt.shape:
            logger.warning(f"[eval_probabilistic] Final shapes differ before mean: sum={_crps_sum.shape}, cnt={cnt.shape}. Trying to broadcast cnt.")
            try:
                cnt = cnt.expand_as(_crps_sum)
            except Exception as e:
                raise RuntimeError(f"[eval_probabilistic] Cannot align CRPS accumulators for mean: sum={_crps_sum.shape}, cnt={cnt.shape}: {e}")

        # Compute land-only mean; set ocean (cnt==0) to NaN so plots blank it out
        ocean = cnt <= 0.0
        safe_cnt = torch.where(ocean, torch.ones_like(cnt), cnt)  # dummy 1 to avoid divide-by-zero
        mean_map_t = (_crps_sum / safe_cnt).cpu()
        mean_map_t[ocean.cpu()] = float('nan')

        # Ensure [H,W] for plotting; prefer recorded H_acc,W_acc
        if mean_map_t.dim() == 1:
            N = int(mean_map_t.numel())
            if H_acc is not None and W_acc is not None and H_acc * W_acc == N:
                mean_map_t = mean_map_t.view(H_acc, W_acc)
            else:
                s = int(round(N ** 0.5))
                if s * s == N:
                    mean_map_t = mean_map_t.view(s, s)
                else:
                    logger.warning(f"[eval_probabilistic] mean_map is 1D of length {N}; saving as 1D (plot may fail).")

        mean_map = mean_map_t.numpy()
        np.savez_compressed(
            tables_dir / "prob_crps_mean_map.npz",
            crps_mean_map=mean_map,     # <-- correct key name
        )
    # Energy score (per day)
    (tables_dir / "prob_energy_daily.csv").write_text("\n".join(es_lines))
    # Variogram score (per day)
    (tables_dir / "prob_variogram_daily.csv").write_text("\n".join(vs_lines))

    # 3.2 PIT
    if all_pit:
        pit_all = np.concatenate(all_pit, axis=0)
    else:
        pit_all = np.array([], dtype=np.float32)
    np.savez_compressed(tables_dir / "prob_pit_values.npz", pit=pit_all)

    # 3.3 Rank
    if rank_acc is not None:
        np.savez_compressed(tables_dir / "prob_rank_histogram.npz", rank_hist=rank_acc.numpy())

    # 3.4 Reliability (aggregate across dates → write one file per threshold)
    for thr, lst in rel_acc.items():
        if not lst:
            continue
        agg = aggregate_reliability_bins(lst)
        bc = agg["bin_center"].numpy()
        pp = agg["prob_pred"].numpy()
        fo = agg["freq_obs"].numpy()
        cnt = agg["count"].numpy()

        lines = ["bin_center,prob_pred,freq_obs,count"]
        for b, p, f, c in zip(bc, pp, fo, cnt):
            lines.append(f"{b:.6f},{p:.6f},{f:.6f},{int(c)}")
        (tables_dir / f"prob_reliability_{thr:.1f}mm.csv").write_text("\n".join(lines))

    # 3.5 Spread–skill (per-date summary)
    (tables_dir / "prob_spread_skill.csv").write_text("\n".join(ss_lines))

    # PMM MAE (per day) — only written if we collected any
    if len(pmm_mae_lines) > 1:
        (tables_dir / "prob_pmm_mae_daily.csv").write_text("\n".join(pmm_mae_lines))

    # ------------------------------------------------------------------------------
    # 3.x Summary table (means / std over days) + spatial CRPS mean over land
    # ------------------------------------------------------------------------------
    def _vals_from_lines(lines: List[str]) -> np.ndarray:
        if len(lines) <= 1:
            return np.array([], dtype=float)
        out = []
        for ln in lines[1:]:
            parts = ln.split(",")
            if len(parts) >= 2:
                try:
                    out.append(float(parts[1]))
                except Exception:
                    pass
        return np.asarray(out, dtype=float)

    crps_daily = _vals_from_lines(crps_lines)
    es_daily   = _vals_from_lines(es_lines)
    vs_daily   = _vals_from_lines(vs_lines)
    pmm_daily  = _vals_from_lines(pmm_mae_lines)

    # overall spatial CRPS (nanmean over land-only map); may not exist if no dates processed
    overall_spatial_crps_mean = np.nan
    try:
        # if mean_map is in scope (saved above), reuse; otherwise try to load what we just wrote
        mean_map_npz = tables_dir / "prob_crps_mean_map.npz"
        if mean_map_npz.exists():
            mm = np.load(mean_map_npz)["crps_mean_map"]
            overall_spatial_crps_mean = float(np.nanmean(mm))
    except Exception:
        pass

    def _mm(arr: np.ndarray) -> float:
        return float(np.nanmean(arr)) if arr.size else float("nan")

    def _ss(arr: np.ndarray) -> float:
        return float(np.nanstd(arr, ddof=0)) if arr.size else float("nan")

    N_dates = int(crps_daily.size)

    # Build per-season CRPS stats (from the daily, domain-mean CRPS file)
    def _dates_vals_from_lines(lines: List[str]) -> tuple[List[str], np.ndarray]:
        if len(lines) <= 1:
            return [], np.array([], dtype=float)
        dates_, vals_ = [], []
        for ln in lines[1:]:
            parts = ln.split(",")
            if len(parts) >= 2:
                d = parts[0].strip()
                try:
                    v = float(parts[1])
                except Exception:
                    continue
                dates_.append(d)
                vals_.append(v)
        return dates_, np.asarray(vals_, dtype=float)

    def _season_of(yyyymmdd: str) -> str:
        try:
            m = int(yyyymmdd[4:6])
        except Exception:
            return "ALL"
        if m in (12, 1, 2):  return "DJF"
        if m in (3, 4, 5):   return "MAM"
        if m in (6, 7, 8):   return "JJA"
        return "SON"

    # Seasonal CRPS stats
    dates_list, vals_list = _dates_vals_from_lines(crps_lines)
    season_bins = {"DJF": [], "MAM": [], "JJA": [], "SON": []}
    for d, v in zip(dates_list, vals_list):
        season_bins[_season_of(d)].append(v)
    def _mmss(x):
        x = np.asarray(x, dtype=float)
        return (float(np.nanmean(x)) if x.size else float("nan"),
                float(np.nanstd(x, ddof=0)) if x.size else float("nan"))
    djf_m, djf_s = _mmss(season_bins["DJF"])
    mam_m, mam_s = _mmss(season_bins["MAM"])
    jja_m, jja_s = _mmss(season_bins["JJA"])
    son_m, son_s = _mmss(season_bins["SON"])

    # PIT KS statistic D vs Uniform(0,1)
    pit_KS_D = float("nan")
    try:
        if all_pit:
            x = np.sort(np.concatenate(all_pit).ravel())
            n = x.size
            if n > 0:
                y = np.arange(1, n + 1) / n
                y_prev = np.arange(0, n) / n
                D_plus = np.max(y - x)
                D_minus = np.max(x - y_prev)
                pit_KS_D = float(max(D_plus, D_minus))
    except Exception:
        pass

    # Rank histogram max |z|
    rank_max_abs_z = float("nan")
    try:
        if rank_acc is not None:
            counts = rank_acc.numpy().astype(float)
            N_rank = counts.sum()
            if N_rank > 0:
                K = counts.size
                p = 1.0 / K
                exp = N_rank * p
                sd = np.sqrt(N_rank * p * (1.0 - p))
                z = (counts - exp) / (sd if sd > 0 else 1.0)
                rank_max_abs_z = float(np.max(np.abs(z)))
    except Exception:
        pass

    # Spread–skill slope and Pearson r (from per-date means)
    ss_arr = np.genfromtxt(str(tables_dir / "prob_spread_skill.csv"), delimiter=",", names=True, dtype=None, encoding="utf-8")
    ss_slope, ss_r = float("nan"), float("nan")
    try:
        if ss_arr.size:
            x_sp = np.asarray(ss_arr["spread_mean"], dtype=float)
            y_sk = np.asarray(ss_arr["skill_mean"], dtype=float)
            m = np.isfinite(x_sp) & np.isfinite(y_sk)
            x_fit = x_sp[m]; y_fit = y_sk[m]
            if x_fit.size >= 2:
                denom = np.sum(x_fit * x_fit)
                ss_slope = float(np.sum(x_fit * y_fit) / denom) if denom != 0 else 0.0
                ss_r = float(np.corrcoef(x_fit, y_fit)[0, 1])
    except Exception:
        pass

    summary_rows = [
        ["metric","mean","std","N"],
        ["CRPS_ensemble", f"{_mm(crps_daily):.6f}", f"{_ss(crps_daily):.6f}", f"{N_dates}"],
        ["EnergyScore",    f"{_mm(es_daily):.6f}",   f"{_ss(es_daily):.6f}",   f"{es_daily.size}"],
        ["VariogramScore", f"{_mm(vs_daily):.6f}",   f"{_ss(vs_daily):.6f}",   f"{vs_daily.size}"],
    ]
    if pmm_daily.size:
        summary_rows.append(["PMM_MAE", f"{_mm(pmm_daily):.6f}", f"{_ss(pmm_daily):.6f}", f"{pmm_daily.size}"])
    summary_rows.append(["SpatialCRPS_land_mean", f"{overall_spatial_crps_mean:.6f}", "", ""])
    summary_rows.extend([
        ["CRPS_DJF", f"{djf_m:.6f}", f"{djf_s:.6f}", f"{len(season_bins['DJF'])}"],
        ["CRPS_MAM", f"{mam_m:.6f}", f"{mam_s:.6f}", f"{len(season_bins['MAM'])}"],
        ["CRPS_JJA", f"{jja_m:.6f}", f"{jja_s:.6f}", f"{len(season_bins['JJA'])}"],
        ["CRPS_SON", f"{son_m:.6f}", f"{son_s:.6f}", f"{len(season_bins['SON'])}"],
        ["PIT_KS_D", f"{pit_KS_D:.6f}", "", ""],
        ["RankHist_max_abs_z", f"{rank_max_abs_z:.6f}", "", ""],
        ["SpreadSkill_slope", f"{ss_slope:.6f}", "", ""],
        ["SpreadSkill_pearson_r", f"{ss_r:.6f}", "", ""],
    ])
    # Write CSV and a human-readable TXT
    (tables_dir / "prob_summary.csv").write_text("\n".join([",".join(r) for r in summary_rows]))
    (tables_dir / "prob_summary.txt").write_text(
        "Probabilistic evaluation summary\n"
        f"N_dates: {N_dates}\n"
        f"CRPS (ensemble): mean={_mm(crps_daily):.6f}, std={_ss(crps_daily):.6f}\n"
        + (f"PMM MAE: mean={_mm(pmm_daily):.6f}, std={_ss(pmm_daily):.6f}\n" if pmm_daily.size else "")
        + f"Energy score: mean={_mm(es_daily):.6f}, std={_ss(es_daily):.6f}\n"
        + f"Variogram score: mean={_mm(vs_daily):.6f}, std={_ss(vs_daily):.6f}\n"
        + f"Spatial CRPS (land mean): {overall_spatial_crps_mean:.6f}\n"
        + f"Seasonal CRPS (mean±std): "
        + f"DJF {djf_m:.3f}±{djf_s:.3f}, MAM {mam_m:.3f}±{mam_s:.3f}, "
        + f"JJA {jja_m:.3f}±{jja_s:.3f}, SON {son_m:.3f}±{son_s:.3f}\n"
        + f"PIT KS D: {pit_KS_D:.3f}\n"
        + f"Rank histogram max |z|: {rank_max_abs_z:.2f}\n"
        + f"Spread–skill slope: {ss_slope:.3f}, Pearson r: {ss_r:.3f}\n"        
    )

    # ================================================================================
    # 4) plots
    # ================================================================================
    # Build member-based CRPS example payload for plotting (HR + members only)
    try:
        n_show = int(getattr(eval_cfg, "crps_examples_n_members", 4))
        seed = int(getattr(eval_cfg, "ensemble_member_seed", 1234))
        _build_and_save_crps_examples_members(
            resolver,
            tables_dir,
            n_members_to_show=n_show,
            member_seed=seed,
        )
    except Exception as e:
        logger.warning(f"[eval_probabilistic] Could not build CRPS member examples: {e}")

    plot_probabilistic(
        out_root,
        gen_root=Path(eval_cfg.gen_dir),
        thresholds=thresholds,
        pit_bins=pit_bins,
    )
