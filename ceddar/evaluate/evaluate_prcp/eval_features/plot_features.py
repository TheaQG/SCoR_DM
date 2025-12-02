# sbgm/evaluate/evaluate_prcp/eval_features/plot_features.py
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scor_dm.evaluate.evaluate_prcp.plot_utils import _ensure_dir, _nice, _savefig
from scor_dm.variable_utils import get_color_for_model

SET_DPI = 300

def plot_features_all(figdir: Path, group: str, sal_metrics: dict) -> None:
    """
    Master plotting entry point for SAL decomposition.
    Shows GEN (PMM), optional GEN (ensemble mean) with error bars, and LR.
    """
    _nice()
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.2))
    bars = ["A", "S", "L"]
    x = np.arange(len(bars))
    width = 0.25

    col_lr = get_color_for_model("LR")
    col_pmm = get_color_for_model("PMM")          # PMM color
    col_gen_ens = get_color_for_model("ENSEMBLE")                          # reuse color for ens mean

    # Sources
    gen = [sal_metrics.get("GEN_vs_HR", {}).get(b, np.nan) for b in bars]
    lr  = [sal_metrics.get("LR_vs_HR", {}).get(b, np.nan) for b in bars]

    # Ensemble mean + std (optional)
    gen_ens = [sal_metrics.get("GEN_ENS_vs_HR", {}).get(b, np.nan) for b in bars]
    gen_ens_std = [sal_metrics.get("GEN_ENS_std", {}).get(b, np.nan) for b in bars]
    has_ens = np.isfinite(np.array(gen_ens, dtype=float)).any()

    # Layout: GEN_ENS (left, with error), GEN PMM (middle), LR (right)
    if has_ens:
        ax.bar(x - width, gen_ens, width,
               label="GEN (ens mean)", color=col_gen_ens,
               edgecolor="black", linewidth=0.7, alpha=0.85,
               yerr=gen_ens_std, capsize=2)
        ax.bar(x, gen, width,
               label="PMM", color=col_pmm,
               edgecolor="black", linewidth=0.7, alpha=0.85)
        ax.bar(x + width, lr, width,
               label="LR", color=col_lr,
               edgecolor="black", linewidth=0.7, alpha=0.85)
    else:
        # Original two-bar layout if no ensemble stats present
        ax.bar(x - width/2, gen, width,
               label="GEN", color=col_pmm)
        ax.bar(x + width/2, lr, width,
               label="LR", color=col_lr)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(bars)
    ax.set_ylabel("Normalized difference")
    ax.set_title(f"SAL decomposition ({group})" + (" — ensemble" if has_ens else ""))
    ax.legend(frameon=True)
    ax.grid(True, ls=":", alpha=0.6)

    fig.tight_layout()
    _ensure_dir(figdir)
    _savefig(fig, figdir / f"features_sal_{group}.png", dpi=SET_DPI)
    plt.close(fig)