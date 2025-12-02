# sbgm/evaluate/evaluate_prcp/overlay_utils.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np

def resolve_baseline_dirs(sample_root: str | Path,
                          types: Tuple[str, ...],
                          split: str,
                          eval_type: str) -> Dict[str, Path]:
    """Return {baseline_type: /.../evaluation/baselines/<type>/<split>/prcp/<eval_type>/tables} that exist."""
    out: Dict[str, Path] = {}
    sr = Path(sample_root)
    for t in types:
        p = sr / "evaluation" / "baselines" / t / split / "prcp" / eval_type / "tables"
        if p.exists():
            out[t] = p
    return out

def load_npz_if_exists(path: Path, name: str) -> Optional[dict]:
    """Load <path>/<name>.npz → dict of arrays, or None if missing."""
    f = path / f"{name}.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    return {k: d[k] for k in d.files}

def load_csv_if_exists(path: Path, name: str) -> Optional[np.ndarray]:
    f = path / f"{name}.csv"
    if not f.exists():
        return None
    return np.genfromtxt(f, delimiter=",", names=True)

def load_txt_if_exists(path: Path, name: str) -> Optional[str]:
    f = path / f"{name}.txt"
    if not f.exists():
        return None
    return f.read_text()