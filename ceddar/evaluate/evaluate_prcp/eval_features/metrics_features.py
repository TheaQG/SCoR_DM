# sbgm/evaluate/evaluate_prcp/eval_features/metrics_features.py
from __future__ import annotations
import math
import numpy as np
from typing import Dict, Tuple
from scipy.ndimage import label as cc_label
from scipy.ndimage import gaussian_filter


def compute_sal(hr: np.ndarray, gen: np.ndarray, lr: np.ndarray | None = None) -> Dict[str, Dict[str, float]]:
    """
    Compute the Structure–Amplitude–Location (SAL) metrics per Wernli et al. (2008).

    A and S are computed from pooled values; L (location) is computed only when
    both inputs are 2D and have the same H×W shape. If L is undefined, SAL is
    formed from A and S only.

    Returns a dict with keys:
      - "GEN_vs_HR": {A, S, L, SAL}
      - "LR_vs_HR":  {A, S, L, SAL}  (only if lr is provided)
    """

    def _as_2d(x: np.ndarray) -> np.ndarray | None:
        x = np.asarray(x)
        return x if x.ndim == 2 else None

    def _sal_pair(ref: np.ndarray, test: np.ndarray) -> Dict[str, float]:
        ref = np.asarray(ref)
        test = np.asarray(test)

        # Amplitude (A) and Structure (S)
        A_ref = float(np.nanmean(ref))
        A_tst = float(np.nanmean(test))
        A = 2.0 * (A_tst - A_ref) / (A_tst + A_ref + 1e-8)

        S_ref = float(np.nanstd(ref))
        S_tst = float(np.nanstd(test))
        S = 2.0 * (S_tst - S_ref) / (S_tst + S_ref + 1e-8)
        # Location (L) – centroid distance normalized by domain diagonal
        ref2 = _as_2d(ref)
        tst2 = _as_2d(test)
        if ref2 is not None and tst2 is not None and ref2.shape == tst2.shape:
            tot_ref = float(np.nansum(ref2))
            tot_tst = float(np.nansum(tst2))
            if tot_ref > 0.0 and tot_tst > 0.0:
                yy, xx = np.indices(ref2.shape)
                cy_ref = float(np.nansum(yy * ref2) / tot_ref)
                cx_ref = float(np.nansum(xx * ref2) / tot_ref)
                cy_tst = float(np.nansum(yy * tst2) / tot_tst)
                cx_tst = float(np.nansum(xx * tst2) / tot_tst)
                diag = float(math.hypot(*ref2.shape)) + 1e-8
                L = float(math.hypot(cy_tst - cy_ref, cx_tst - cx_ref) / diag)
            else:
                L = float("nan")
        else:
            L = float("nan")

        SAL = float(np.sqrt(A*A + S*S + (L*L if np.isfinite(L) else 0.0)))
        return {"A": A, "S": S, "L": L, "SAL": SAL}

    out: Dict[str, Dict[str, float]] = {}
    if hr is not None and gen is not None:
        out["GEN_vs_HR"] = _sal_pair(hr, gen)
    if hr is not None and lr is not None:
        out["LR_vs_HR"] = _sal_pair(hr, lr)
    return out

def compute_sal_object(
    hr: np.ndarray,
    gen: np.ndarray,
    lr: np.ndarray | None = None,
    *,
    threshold_kind: str = "quantile",     # "quantile" or "absolute"
    threshold_value: float = 0.90,        # e.g. 0.90 for 90th pct OR absolute mm/day
    connectivity: int = 8,                # 4 or 8
    min_area_px: int = 9,                 # drop very small objects
    smooth_sigma: float | None = None,    # e.g. 0.75 to reduce salt-and-pepper
    peakedness: str = "largest",        
    eps: float = 1e-8,
) -> Dict[str, Dict[str, float]]:
    """
    Object-based SAL following Wernli et al. (2008) spirit:
      A = normalized mean difference (unchanged)
      L = L1 + L2; L1: CoM distance / diag; L2: |scatter_test - scatter_ref| / diag * 2
      S = normalized difference of 'peakedness' using objects above a threshold (mode: 'largest' mass fraction or 'herfindahl' index)
    """

    def _prep(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            x = x.reshape(x.shape[-2], x.shape[-1])
        if smooth_sigma and smooth_sigma > 0:
            x = gaussian_filter(x, sigma=float(smooth_sigma))
        return x

    def _threshold(x: np.ndarray) -> float:
        if threshold_kind == "quantile":
            return float(np.nanpercentile(x, threshold_value * 100.0))
        elif threshold_kind == "absolute":
            return float(threshold_value)
        else:
            raise ValueError(f"unknown threshold_kind={threshold_kind}")

    def _objects_and_masses(x: np.ndarray, thr: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mask = np.isfinite(x) & (x > thr)
        if connectivity == 4:
            structure = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=int)
        else:
            structure = np.ones((3,3), dtype=int)
        lab, nobj = cc_label(mask, structure=structure) # type: ignore
        masses = []
        cents  = []
        for k in range(1, nobj+1):
            idx = (lab == k)
            if idx.sum() < min_area_px:
                continue
            m = float(np.nansum(x[idx]))
            if m <= 0:
                continue
            yy, xx = np.nonzero(idx)
            # intensity-weighted centroid
            w = x[idx]
            wy = float(np.sum(yy * w) / (np.sum(w) + eps))
            wx = float(np.sum(xx * w) / (np.sum(w) + eps))
            masses.append(m)
            cents.append((wy, wx))
        return lab, np.array(masses, dtype=float), np.array(cents, dtype=float)

    def _peakedness(masses: np.ndarray, mode: str) -> float:
        if masses.size == 0:
            return 0.0
        tot = float(np.sum(masses)) + eps
        w = masses / tot
        if mode.lower() == "herfindahl":
            return float(np.sum(w * w))
        # default: largest object mass fraction
        return float(np.max(w))

    def _amplitude(a_ref: float, a_tst: float) -> float:
        return 2.0 * (a_tst - a_ref) / (a_tst + a_ref + eps)

    def _com(x: np.ndarray) -> np.ndarray:
        tot = float(np.nansum(x))
        if not np.isfinite(tot) or tot <= 0:
            return np.array([np.nan, np.nan], dtype=float)
        yy, xx = np.indices(x.shape)
        cy = float(np.nansum(yy * x) / (tot + eps))
        cx = float(np.nansum(xx * x) / (tot + eps))
        return np.array([cy, cx], dtype=float)

    def _location_terms(ref: np.ndarray, tst: np.ndarray,
                        cents_ref: np.ndarray, cents_tst: np.ndarray) -> Tuple[float, float, float]:
        diag = float(math.hypot(*ref.shape)) + eps
        c_ref = _com(ref); c_tst = _com(tst)
        if not np.all(np.isfinite(c_ref)) or not np.all(np.isfinite(c_tst)):
            return float("nan"), float("nan"), float("nan")
        L1 = float(math.hypot(*(c_tst - c_ref)) / diag)
        # scatter of object centroids around field CoM (mass-weighted)
        def _scatter(cfield: np.ndarray, cents: np.ndarray, masses: np.ndarray) -> float:
            if cents.size == 0 or masses.size == 0:
                return np.nan
            d = np.hypot(cents[:,0] - cfield[0], cents[:,1] - cfield[1])
            w = masses / (np.sum(masses) + eps)
            return float(np.sum(w * d) / diag)
        # We need masses, so re-extract from cents arrays length
        # (caller already computed masses; pass them as closure variables)
        # This helper currently only computes L1; return placeholder NaNs for the other two values
        return L1, float("nan"), float("nan")

    # Prep fields (2D, optional smoothing)
    hr2  = _prep(hr)
    gen2 = _prep(gen) if gen is not None else None
    lr2  = _prep(lr)  if lr  is not None else None

    # Common helpers to compute SAL components for a pair
    def _sal_pair(ref: np.ndarray, tst: np.ndarray) -> Dict[str, float]:
        # A
        A = _amplitude(float(np.nanmean(ref)), float(np.nanmean(tst)))

        # thresholds per-field (most common; avoids bias when climatologies differ)
        thr_ref = _threshold(ref)
        thr_tst = _threshold(tst)

        # objects
        _, masses_ref, cents_ref = _objects_and_masses(ref, thr_ref)
        _, masses_tst, cents_tst = _objects_and_masses(tst, thr_tst)

        # S: peakedness based on object mass dominance
        P_ref = _peakedness(masses_ref, peakedness)
        P_tst = _peakedness(masses_tst, peakedness)
        S = 2.0 * (P_tst - P_ref) / (P_tst + P_ref + eps)

        # L = L1 + L2
        # L1: CoM distance of full fields (intensity-weighted)
        diag = float(math.hypot(*ref.shape)) + eps
        c_ref = _com(ref); c_tst = _com(tst)
        L1 = float(math.hypot(*(c_tst - c_ref)) / diag) if np.all(np.isfinite([*c_ref, *c_tst])) else np.nan

        # L2: scatter difference of object centroids around their own field CoM (mass-weighted)
        def _scatter(cfield: np.ndarray, cents: np.ndarray, masses: np.ndarray) -> float:
            if cents.size == 0 or masses.size == 0 or not np.all(np.isfinite(cfield)):
                return np.nan
            d = np.hypot(cents[:,0] - cfield[0], cents[:,1] - cfield[1])
            w = masses / (np.sum(masses) + eps)
            return float(np.sum(w * d) / diag)

        scat_ref = _scatter(c_ref, cents_ref, masses_ref)
        scat_tst = _scatter(c_tst, cents_tst, masses_tst)
        if np.isfinite(scat_ref) and np.isfinite(scat_tst):
            L2 = 2.0 * abs(scat_tst - scat_ref)  # already normalized by diag inside _scatter
        else:
            L2 = np.nan

        L = (L1 + L2) if (np.isfinite(L1) and np.isfinite(L2)) else np.nan

        SAL = float(np.sqrt(A*A + S*S + (L*L if np.isfinite(L) else 0.0)))
        return {"A": A, "S": S, "L": L, "SAL": SAL}

    out: Dict[str, Dict[str, float]] = {}
    if gen2 is not None:
        out["GEN_vs_HR"] = _sal_pair(hr2, gen2)
    if lr2 is not None:
        out["LR_vs_HR"]  = _sal_pair(hr2, lr2)
    return out