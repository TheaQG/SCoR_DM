from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, Sequence, Dict, Tuple, List, Any
import numpy as np
import torch
import csv

# ---------- helpers ----------

def _to_hw(t: torch.Tensor) -> torch.Tensor:
    """Coerce tensor to float32 [H,W]. Accepts [H,W], [1,H,W], [1,1,H,W]."""
    if not torch.is_tensor(t):
        t = torch.as_tensor(t)
    t = t.float()
    if t.dim() == 4 and t.shape[:2] == (1,1):
        t = t.squeeze(0).squeeze(0)
    elif t.dim() == 3 and t.shape[0] == 1:
        t = t.squeeze(0)
    if t.dim() != 2:
        # last two dims are H, W
        t = t.reshape(t.shape[-2], t.shape[-1])
    return t

def _apply_mask(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Set ocean/invalid pixels to NaN (so sums/means can use nan-ops)."""
    if mask is None:
        return x
    m = mask
    if m.dtype != torch.bool:
        m = m > 0.5
    if m.dim() == 4 and m.shape[:2] == (1,1):
        m = m.squeeze(0).squeeze(0)
    elif m.dim() == 3 and m.shape[0] == 1:
        m = m.squeeze(0)
    # broadcast-safe
    x = x.clone()
    x[~m] = float('nan')
    return x

def _nanmean_stack(stack: List[torch.Tensor]) -> torch.Tensor:
    if not stack:
        raise ValueError("Empty stack in _nanmean_stack.")
    x = torch.stack(stack, dim=0)  # [T,H,W]
    # torch.nanmean on older builds may not exist; emulate
    num = torch.nansum(x, dim=0)
    den = torch.sum(torch.isfinite(x), dim=0).clamp_min(1)
    out = num / den
    out[den == 0] = float('nan')
    return out

def _nansum_stack(stack: List[torch.Tensor]) -> torch.Tensor:
    if not stack:
        raise ValueError("Empty stack in _nansum_stack.")
    return torch.nansum(torch.stack(stack, dim=0), dim=0)

def _wet_day_freq_stack(stack: List[torch.Tensor], wet_thr_mm: float) -> torch.Tensor:
    if not stack:
        raise ValueError("Empty stack in _wet_day_freq_stack.")
    x = torch.stack(stack, dim=0)  # [T,H,W]
    wet = (x >= float(wet_thr_mm)).to(torch.float32)
    # mask out NaNs in x: where all-NaN across time, leave NaN
    valid = torch.isfinite(x).any(dim=0)
    freq = wet.mean(dim=0)
    freq[~valid] = float('nan')
    return freq

def _rxk_map_from_stack(stack: List[torch.Tensor], k: int) -> torch.Tensor:
    """
    Pixelwise RxK on daily stack: compute rolling k-day sum per pixel and take max across time.
    If T < k for a pixel (or globally), fall back to max daily.
    """
    x = torch.stack(stack, dim=0)  # [T,H,W]
    T, H, W = x.shape
    # replace NaNs by 0 for rolling sums, but track validity
    valid = torch.isfinite(x)
    x2 = torch.nan_to_num(x, nan=0.0)

    # If the time series is too short for a k-day window, just use max daily
    if k <= 1 or T < k:
        # emulate nanmax: compute max with NaNs treated as -inf, then restore NaN where no valid data
        daily_max = torch.max(torch.nan_to_num(x, nan=float("-inf")), dim=0)[0]
        daily_valid = valid.any(dim=0)
        daily_max[~daily_valid] = float("nan")
        return daily_max

    # rolling sum via cumulative sum (along time)
    c = torch.cumsum(x2, dim=0)
    pad = torch.zeros((1, H, W), dtype=x2.dtype, device=x2.device)
    cpad = torch.cat([pad, c], dim=0)       # [T+1,H,W]
    rs = cpad[k:] - cpad[:-k]               # [T-k+1,H,W]

    # window validity
    vfloat = valid.to(torch.float32)
    cv = torch.cumsum(vfloat, dim=0)
    cvpad = torch.cat([torch.zeros((1, H, W), dtype=vfloat.dtype, device=vfloat.device), cv], dim=0)
    vc = cvpad[k:] - cvpad[:-k]             # [T-k+1,H,W]
    rs[vc == 0] = float("nan")

    # Max over windows; where all windows invalid -> fall back to max daily
    any_valid_window = torch.isfinite(rs).any(dim=0)
    mx_vals = torch.max(torch.nan_to_num(rs, nan=float("-inf")), dim=0)[0]
    mx_vals[~any_valid_window] = float("nan")

    any_valid_daily = valid.any(dim=0)
    fallback = torch.max(torch.nan_to_num(x, nan=float("-inf")), dim=0)[0]
    fallback[~any_valid_daily] = float("nan")

    mx = torch.where(torch.isfinite(mx_vals), mx_vals, fallback)
    return mx

def _nanpercentile_stack(stack: List[torch.Tensor], q: float) -> torch.Tensor:
    """Pixelwise percentile over time with NaN handling."""
    x = torch.stack(stack, dim=0)  # [T,H,W]
    # move to numpy for percentile if torch has no nanpercentile
    qv = float(q)
    arr = x.detach().cpu().numpy()
    with np.errstate(all='ignore'):
        p = np.nanpercentile(arr, qv, axis=0)
    return torch.from_numpy(p).to(x.device, x.dtype)

# ---------- public API ----------

def accumulate_daily_fields(
    dates: Iterable[str],
    load_fn,                 # (date)-> torch [H,W] or None
    mask_fn,                 # (date)-> torch [H,W] bool or None
    *,
    use_mask: bool = True,
) -> Tuple[List[str], List[torch.Tensor], Optional[torch.Tensor]]:
    """
    Load daily fields into memory (list of tensors), apply per-day mask→NaN,
    return (kept_dates, fields_list, union_mask) where union_mask is the AND
    over days (if masking), else None.
    """
    kept: List[str] = []
    fields: List[torch.Tensor] = []
    union_mask: Optional[torch.Tensor] = None

    for d in dates:
        x = load_fn(d)
        if x is None:
            continue
        x = _to_hw(x)
        m = mask_fn(d) if use_mask else None
        if m is not None:
            m = _to_hw(m) > 0.5
            x = _apply_mask(x, m)
            union_mask = m if union_mask is None else (union_mask & m)
        fields.append(x)
        kept.append(d)

    return kept, fields, union_mask

def compute_spatial_climatologies(
    fields: List[torch.Tensor],
    *,
    wet_thr_mm: float = 1.0,
    percentiles: Sequence[float] = (95.0, 99.0),
    rxk_days: Sequence[int] = (1,5),
) -> Dict[str, torch.Tensor]:
    """
    Compute pixelwise:
      - mean (mm/day)
      - sum (mm)  [sum of daily precip across period]
      - wet_day_freq (0..1) for given wet_thr_mm
      - rx1, rx5 (max of k-day running sums)
      - p95, p99 (over days)
    """
    if not fields:
        raise ValueError("No fields given to compute_spatial_climatologies.")

    # Ensure same device/dtype
    device = fields[0].device
    fields = [f.to(device=device, dtype=torch.float32) for f in fields]

    out: Dict[str, torch.Tensor] = {}
    out["mean"] = _nanmean_stack(fields)
    out["sum"]  = _nansum_stack(fields)
    out["wetfreq"] = _wet_day_freq_stack(fields, wet_thr_mm)

    for k in rxk_days:
        out[f"rx{k}"] = _rxk_map_from_stack(fields, int(k))

    for q in percentiles:
        out[f"p{int(q)}"] = _nanpercentile_stack(fields, float(q)).to(device=device, dtype=torch.float32)

    return out

def save_maps_npz(path: Path, **maps: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v.detach().cpu().numpy() for k, v in maps.items()}
    np.savez_compressed(path, **payload)

# ---------- summary helpers (domain-aggregated CSV) ----------
def _np(a):
    if a is None:
        return None
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)

def _nanmean(a):
    a = _np(a)
    return float(np.nanmean(a)) if a is not None and np.isfinite(a).any() else float("nan")

def _nanstd(a):
    a = _np(a)
    return float(np.nanstd(a)) if a is not None and np.isfinite(a).any() else float("nan")

def _nansum(a):
    a = _np(a)
    return float(np.nansum(a)) if a is not None and np.isfinite(a).any() else float("nan")

def _valid_frac(a):
    a = _np(a)
    if a is None:
        return float("nan")
    m = np.isfinite(a)
    return float(m.mean())

def _nanrmse(a, b):
    a = _np(a); b = _np(b)
    if a is None or b is None:
        return float("nan")
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return float("nan")
    return float(np.sqrt(np.nanmean((a[m] - b[m])**2)))

def _nanbias(a, b):
    a = _np(a); b = _np(b)
    if a is None or b is None:
        return float("nan")
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return float("nan")
    return float(np.nanmean(a[m] - b[m]))

def _nanratio(a, b, eps=1e-12):
    a = _np(a); b = _np(b)
    if a is None or b is None:
        return float("nan")
    m = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > eps)
    if not m.any():
        return float("nan")
    r = a[m] / b[m]
    return float(np.nanmean(r))

def _nancorr(a, b):
    a = _np(a); b = _np(b)
    if a is None or b is None:
        return float("nan")
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return float("nan")
    if m.sum() < 2:
        return float("nan")
    aa = a[m] - np.nanmean(a[m])
    bb = b[m] - np.nanmean(b[m])
    denom = np.sqrt(np.nanmean(aa**2)) * np.sqrt(np.nanmean(bb**2))
    if denom == 0 or not np.isfinite(denom):
        return float("nan")
    return float(np.nanmean(aa*bb) / denom)

def summarize_group_spatial_maps(group_name: str, group_maps: Dict[str, Dict[str, torch.Tensor]]) -> List[Dict[str, Any]]:
    """
    Create rows of domain-aggregated stats for a group.
    Expects `group_maps[label][var]` tensors for labels in {'hr','ensmean','ensstd','lr','pmm'} (optional).
    Produces:
      - Per-source rows with <var>_mean, <var>_std, sum_total, wetfreq_valid_frac
      - Comparison rows {ensmean, lr, pmm}_vs_hr with <var>_bias, <var>_rmse, <var>_ratio, <var>_corr
    """
    rows: List[Dict[str, Any]] = []
    var_order = ["mean","sum","wetfreq","rx1","rx5","p95","p99"]

    def _emit_source_rows(label: str, maps: Dict[str, torch.Tensor]):
        base: Dict[str, Any] = {"group": group_name, "source": label}
        for v in var_order:
            if v not in maps:
                continue
            arr = maps[v]
            base[f"{v}_mean"] = _nanmean(arr)
            base[f"{v}_std"]  = _nanstd(arr)
            if v == "sum":
                base["sum_total"] = _nansum(arr)
            if v == "wetfreq":
                base["wetfreq_valid_frac"] = _valid_frac(arr)
        rows.append(base)

    for label in ("hr","ensmean","ensstd","lr","pmm"):
        if label in group_maps:
            _emit_source_rows(label, group_maps[label])

    if "hr" in group_maps:
        hr_maps = group_maps["hr"]
        for comp_label in ("ensmean","lr","pmm"):
            if comp_label not in group_maps:
                continue
            cmaps = group_maps[comp_label]
            row: Dict[str, Any] = {"group": group_name, "source": f"{comp_label}_vs_hr"}
            for v in var_order:
                a = cmaps.get(v, None)
                b = hr_maps.get(v, None)
                if a is None or b is None:
                    continue
                row[f"{v}_bias"]  = _nanbias(a, b)
                row[f"{v}_rmse"]  = _nanrmse(a, b)
                row[f"{v}_ratio"] = _nanratio(a, b)
                row[f"{v}_corr"]  = _nancorr(a, b)
            rows.append(row)

    return rows

def write_spatial_summary_csv(tables_dir: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    tables_dir.mkdir(parents=True, exist_ok=True)
    header_keys = set()
    for r in rows:
        header_keys.update(r.keys())
    header = ["group","source"] + sorted(k for k in header_keys if k not in ("group","source"))
    out_csv = tables_dir / "spatial_summary.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})