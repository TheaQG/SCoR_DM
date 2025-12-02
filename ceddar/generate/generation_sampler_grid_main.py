import os
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scor_dm.training_utils import get_model, get_final_gen_dataloader
from generate.generation import GenerationRunner, GenerationConfig
from scor_dm.utils import get_model_string

logger = logging.getLogger(__name__)

def _resolve_sampler_out_dir(cfg) -> Path:
    """
    Base dir for sampler grid runs:
        <paths.sample_dir>/generation/<model_name>/sampler_grid/
    """
    model_name_str = get_model_string(cfg)

    # Try attribute access first, then dict-style (OmegaConf supports both)
    sample_dir = None
    if hasattr(cfg, "paths") and hasattr(cfg.paths, "sample_dir"):
        sample_dir = cfg.paths.sample_dir
    elif isinstance(cfg, dict) and "paths" in cfg and "sample_dir" in cfg["paths"]:
        sample_dir = cfg["paths"]["sample_dir"]
    else:
        raise KeyError("Could not resolve cfg.paths.sample_dir")

    base = Path(sample_dir) / "generation" / model_name_str / "sampler_grid"
    base.mkdir(parents=True, exist_ok=True)
    return base

def _get_dict(cfg, key: str, default: Any = None):
    """Small helper to read nested config robustly for dict/OmegaConf."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)

def _build_generation_config(cfg, out_root: Path) -> GenerationConfig:
    """
    Build a GenerationConfig from cfg for the current (already mutated) edm params.
    We assume cfg['edm'] has been updated with rho, S_churn, sigma_min/max, etc.
    """
    full_gen_eval = _get_dict(cfg, "full_gen_eval", {})
    edm = _get_dict(cfg, "edm", {})

    # Ensemble size: prefer full_gen_eval.ensemble_size, fall back to data_handling.n_gen_samples, else 32
    if isinstance(cfg, dict):
        dh = cfg.get("data_handling", {})
        M = int(full_gen_eval.get("ensemble_size", dh.get("n_gen_samples", 32)))
    else:
        dh = getattr(cfg, "data_handling", {})
        M = int(getattr(full_gen_eval, "ensemble_size", getattr(dh, "n_gen_samples", 32)))

    return GenerationConfig(
        output_root=str(out_root),
        ensemble_size=M,
        sampler_steps=int(edm.get("sampling_steps", 40)),
        seed=int(full_gen_eval.get("seed", 1234)) if isinstance(full_gen_eval, dict) else int(getattr(full_gen_eval, "seed", 1234)),
        use_edm=bool(edm.get("enabled", True)),
        sigma_min=float(edm.get("sigma_min", 0.002)),
        sigma_max=float(edm.get("sigma_max", 80.0)),
        rho=float(edm.get("rho", 7.0)),
        S_churn=float(edm.get("S_churn", 0.0)),
        S_min=float(edm.get("S_min", 0.0)),
        S_max=float(edm.get("S_max", float("inf"))),
        S_noise=float(edm.get("S_noise", 1.0)),
        predict_residual=bool(edm.get("predict_residual", False)),
        save_space="physical",
        max_dates=int(full_gen_eval.get("max_dates", -1)) if isinstance(full_gen_eval, dict) else int(getattr(full_gen_eval, "max_dates", -1)),
    )

def _as_float_list(x, fallback: float | None = None):
    """Normalize config entry to a list[float]."""
    if x is None:
        return [] if fallback is None else [float(fallback)]
    if isinstance(x, (float, int)):
        return [float(x)]
    return [float(v) for v in x]


# Helper to enumerate all sampler combos with indices
def _enumerate_sampler_combos(rho_grid, S_churn_grid, sigma_scale_grid):
    """
    Enumerate all (rho, S_churn, sigma_scale) combinations with a flat index.
    Returns a list of dicts: [{"idx": i, "rho": ..., "S_churn": ..., "sigma_scale": ...}, ...]
    """
    combos = []
    idx = 0
    for rho in rho_grid:
        for S_churn in S_churn_grid:
            for sigma_scale in sigma_scale_grid:
                combos.append(
                    {
                        "idx": idx,
                        "rho": float(rho),
                        "S_churn": float(S_churn),
                        "sigma_scale": float(sigma_scale),
                    }
                )
                idx += 1
    return combos


def generation_sampler_grid_main(cfg):
    """
    Generate ensembles across a grid of (rho, S_churn, sigma_scale) values.

    For each combination, outputs go to:
      <sample_dir>/generation/<model_name>/sampler_grid/
          rho=<rho>_Schurn=<S_churn>_sigscale=<sigma_scale>/
    """

    # ----------------------- Seed -----------------------
    full_gen_eval = _get_dict(cfg, "full_gen_eval", {})
    if isinstance(full_gen_eval, dict):
        seed = int(full_gen_eval.get("seed", 1234))
    else:
        seed = int(getattr(full_gen_eval, "seed", 1234))

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # ----------------------- Device -----------------------
    training_cfg = _get_dict(cfg, "training", {})
    if isinstance(training_cfg, dict):
        device = training_cfg.get("device", "cpu")
    else:
        device = getattr(training_cfg, "device", "cpu")
    if device is None:
        device = "cpu"

    # ----------------------- Model & checkpoint -----------------------
    model, ckpt_dir, ckpt_name = get_model(cfg)
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    ckpt = torch.load(ckpt_path, map_location=device)
    if "network_params" not in ckpt:
        raise KeyError(f"Checkpoint missing 'network_params': {ckpt_path}")
    model.load_state_dict(ckpt["network_params"])
    model.eval()
    logger.info(f"[generation_sampler_grid_main] Loaded checkpoint: {ckpt_path}")

    # ----------------------- Data (deterministic, split-aware) -----------------------
    # Decide which temporal split to generate for: train / val / test
    if isinstance(full_gen_eval, dict):
        split_cfg = str(full_gen_eval.get("split", "test")).lower()
    else:
        split_cfg = str(getattr(full_gen_eval, "split", "test")).lower()

    if split_cfg in ("val", "valid", "validation"):
        split_for_dataset = "valid"
    elif split_cfg == "train":
        split_for_dataset = "train"
    else:
        split_for_dataset = "test"

    # Attach split info to cfg.data_handling (for bookkeeping; batching is handled in get_final_gen_dataloader)
    if isinstance(cfg, dict):
        cfg.setdefault("data_handling", {})
        dh = cfg["data_handling"]
        dh["split"] = split_for_dataset
        dh["shuffle"] = False
        dh["drop_last"] = False
    else:
        if not hasattr(cfg, "data_handling") or cfg.data_handling is None:
            cfg.data_handling = {}
        cfg.data_handling["split"] = split_for_dataset
        cfg.data_handling["shuffle"] = False
        cfg.data_handling["drop_last"] = False

    logger.info(f"[generation_sampler_grid_main] Using data split='{split_for_dataset}' for sampler grid generation")

    gen_dataloader = get_final_gen_dataloader(cfg, split=split_for_dataset)

    # ----------------------- Sampler grid -----------------------
    sampler_grid = None
    if isinstance(full_gen_eval, dict):
        sampler_grid = full_gen_eval.get("sampler_grid", {})
    else:
        sampler_grid = getattr(full_gen_eval, "sampler_grid", {})

    edm_cfg = _get_dict(cfg, "edm", {})

    rho_grid = _as_float_list(
        sampler_grid.get("rho", None) if isinstance(sampler_grid, dict) else getattr(sampler_grid, "rho", None),
        fallback=edm_cfg.get("rho", 7.0),
    )
    S_churn_grid = _as_float_list(
        sampler_grid.get("S_churn", None) if isinstance(sampler_grid, dict) else getattr(sampler_grid, "S_churn", None),
        fallback=edm_cfg.get("S_churn", 0.0),
    )
    sigma_scale_grid = _as_float_list(
        sampler_grid.get("sigma_scale", None) if isinstance(sampler_grid, dict) else getattr(sampler_grid, "sigma_scale", None),
        fallback=1.0,
    )

    logger.info(f"[generation_sampler_grid_main] rho grid        = {rho_grid}")
    logger.info(f"[generation_sampler_grid_main] S_churn grid    = {S_churn_grid}")
    logger.info(f"[generation_sampler_grid_main] sigma_scale grid = {sigma_scale_grid}")

    base_sigma_min = float(edm_cfg.get("sigma_min", 0.002))
    base_sigma_max = float(edm_cfg.get("sigma_max", 80.0))

    # ----------------------- Enumerate and optionally slice combos -----------------------
    all_combos = _enumerate_sampler_combos(rho_grid, S_churn_grid, sigma_scale_grid)
    n_combos = len(all_combos)

    start_idx_env = os.environ.get("SAMPLER_COMBO_INDEX_START", None)
    end_idx_env = os.environ.get("SAMPLER_COMBO_INDEX_END", None)

    if start_idx_env is not None or end_idx_env is not None:
        start_idx = int(start_idx_env) if start_idx_env is not None else 0
        end_idx = int(end_idx_env) if end_idx_env is not None else (n_combos - 1)
        selected_combos = [c for c in all_combos if start_idx <= c["idx"] <= end_idx]
        logger.info(
            "[generation_sampler_grid_main] Restricting combos via env: "
            "SAMPLER_COMBO_INDEX_START=%s, SAMPLER_COMBO_INDEX_END=%s -> %d / %d combos",
            start_idx_env,
            end_idx_env,
            len(selected_combos),
            n_combos,
        )
    else:
        selected_combos = all_combos
        logger.info("[generation_sampler_grid_main] Using all %d sampler combos.", n_combos)
    
    # ----------------------- Base output -----------------------
    base_out = _resolve_sampler_out_dir(cfg)

    # ----------------------- Loop over selected combos -----------------------
    for combo in selected_combos:
        combo_idx = combo["idx"]
        rho = combo["rho"]
        S_churn = combo["S_churn"]
        sigma_scale_f = combo["sigma_scale"]

        sigma_min_eff = base_sigma_min * sigma_scale_f
        sigma_max_eff = base_sigma_max * sigma_scale_f

        # Mutate cfg['edm'] so GenerationRunner/edm_sampler see the correct parameters
        if isinstance(cfg, dict):
            cfg.setdefault("edm", {})
            edm_mut = cfg["edm"]
            edm_mut["rho"] = float(rho)
            edm_mut["S_churn"] = float(S_churn)
            edm_mut["sigma_min"] = float(sigma_min_eff)
            edm_mut["sigma_max"] = float(sigma_max_eff)
            edm_mut["sigma_scale"] = sigma_scale_f  # optional, for logging downstream
        else:
            # OmegaConf-style
            if "edm" not in cfg:
                cfg["edm"] = {}
            cfg["edm"]["rho"] = float(rho)
            cfg["edm"]["S_churn"] = float(S_churn)
            cfg["edm"]["sigma_min"] = float(sigma_min_eff)
            cfg["edm"]["sigma_max"] = float(sigma_max_eff)
            cfg["edm"]["sigma_scale"] = sigma_scale_f

        logger.info(
            "[generation_sampler_grid_main] Combo idx=%d: rho=%.2f | S_churn=%.2f | sigma_scale=%.2f "
            "=> sigma_min=%.4g, sigma_max=%.4g",
            combo_idx,
            rho,
            S_churn,
            sigma_scale_f,
            sigma_min_eff,
            sigma_max_eff,
        )

        # Subdirectory for this combo
        subdir = base_out / f"rho={rho:.2f}_Schurn={S_churn:.2f}_sigscale={sigma_scale_f:.2f}"
        subdir.mkdir(parents=True, exist_ok=True)

        # Build GenerationConfig for this combo
        gen_cfg = _build_generation_config(cfg, subdir)

        # Rebuild dataloader inside the loop to avoid any subtle statefulness
        gen_dataloader = get_final_gen_dataloader(cfg, split=split_for_dataset)

        # Run generator
        logger.info(f"[generation_sampler_grid_main] Generating for combo idx={combo_idx} -> {subdir}")
        try:
            runner = GenerationRunner(
                model=model,
                cfg=cfg,
                device=device,
                out_root=subdir,
                gen_config=gen_cfg,
            )
        except TypeError as e:
            logger.warning(
                "[generation_sampler_grid_main] GenerationRunner signature issue, retrying without out_root: %s",
                e,
            )
            runner = GenerationRunner(  # type: ignore
                model=model,
                cfg=cfg,
                device=device,
                gen_config=gen_cfg,
            )
        runner.run(gen_dataloader)

    logger.info(f"[generation_sampler_grid_main] Done. Outputs at: {base_out}")
    return base_out