# sbgm/logging_utils.py
from __future__ import annotations
import datetime, hashlib, json, logging, logging.config, os, subprocess, warnings
from omegaconf import OmegaConf
import logging

# --------- Helpers to make stable run names / hashes ----------
def cfg_hash(cfg) -> str:
    yml = OmegaConf.to_yaml(cfg, resolve=True) # Resolve ${env:...} etc
    return hashlib.md5(yml.encode()).hexdigest()[:8]

def make_run_name(exp_name: str, cfg_hash_str: str) -> str:
    # UTC timestamp keeps filenames sortable
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"{exp_name}__{cfg_hash_str}__{ts}"

def ensure_run_dir(base_dir: str, model_name: str) -> str:
    run_dir = os.path.join(base_dir, model_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def _to_py(obj):
    """ Convert OmegaConf nodes (DictConfig/ListConfig) to plain Python dict/list """
    return OmegaConf.to_container(obj, resolve=True)

# --------- Filter to inject run_id into every record -----------
# Extend the filter to include SLURM job info
class RunContextFilter(logging.Filter):
    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id
        self.job_id = os.environ.get("SLURM_JOB_ID", "N/A")
        self.node = os.environ.get("SLURM_NODENAME", "N/A")
    def filter(self, record: logging.LogRecord) -> bool:
        # attach if absent so formatters using %(run_id)s never break
        if not hasattr(record, "run_id"):
            record.run_id = self.run_id
        if not hasattr(record, "job_id"):
            record.job_id = self.job_id
        if not hasattr(record, "node"):
            record.node = self.node
        return True

# --------- DictConfig-based setup (root logger) ----------------
def setup_logging(run_dir: str, run_name: str, file_level: str = "INFO", console_level: str = "WARNING") -> str:
    log_path = os.path.join(run_dir, f"{run_name}.log")
    fmt = "%(asctime)s | %(levelname)-7s | %(run_id)s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    cfg = {
        "version": 1,
        "disable_existing_loggers": False,   # keep 3rd-party loggers working
        "formatters": {
            "std": {"format": fmt, "datefmt": datefmt}
        },
        "handlers": {
            "console": {"class": "logging.StreamHandler", "level": console_level, "formatter": "std"},
            "file": {
                "class": "logging.FileHandler",
                "level": file_level,
                "formatter": "std",
                "filename": log_path,
                "mode": "w",
                "encoding": "utf-8",
            },
        },
        "loggers": {
            # Tame a few noisy libs (adjust as needed)
            "matplotlib": {"level": "WARNING", "propagate": True},
            "urllib3": {"level": "WARNING", "propagate": True},
            "PIL": {"level": "WARNING", "propagate": True},
        },
        "root": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
        },
    }

    logging.config.dictConfig(cfg)

    # attach run_id to *both* handlers
    rc_filter = RunContextFilter(run_name)
    root = logging.getLogger()
    for h in root.handlers:
        h.addFilter(rc_filter)

    # capture warnings into logging
    logging.captureWarnings(True)

    return log_path





# --------- Manifest (RUN.json / RUN.md) ------------------------
def write_run_manifest(run_dir: str, run_name: str, cfg, model_name: str) -> None:
    # try:
    #     git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    # except Exception:
    #     git_commit = "N/A"
    
    os.makedirs(run_dir, exist_ok=True)

    # 1) Keep a full resolved snapshot (human-readable)
    with open(os.path.join(run_dir, "RUN_config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    # 2) Build a compact JSON manifest using ONLY plain Python containers
    cfg_py = _to_py(cfg) # <-- Everything below is plain dict/list/str/float/...

    manifest = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "run_name": run_name,
        "model_name": model_name,
        "cfg_hash": cfg_hash(cfg),
        # "git_commit": git_commit,
        "paths": cfg_py.get("paths", {}), # type: ignore
        "experiment": cfg_py.get("experiment", {}), # type: ignore
        # keep it lightweight; expand if you want:
        "training": cfg_py.get("training", {}), # type: ignore
        "edm": cfg_py.get("edm", {}), # type: ignore
        "highres": cfg_py.get("highres", {}), # type: ignore
        "lowres": cfg_py.get("lowres", {}), # type: ignore
        # Add others as needed
    }

    # 1) Compact JSON manifest for quick inspection/scripts
    with open(os.path.join(run_dir, "RUN.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    # 2) Pretty-printed markdown manifest for humans
    with open(os.path.join(run_dir, "RUN.md"), "w") as f:
        f.write(f"# Run manifest: {run_name}\n\n```\n{json.dumps(manifest, indent=2)}\n```\n")
    

# --------- Pretty banners -------------------------------------
def log_banner(title: str, logger: logging.Logger | None = None, char: str = "═") -> None:
    lg = logger or logging.getLogger(__name__)
    line = char * (len(title) + 10)
    lg.info("\n%s\n   %s\n%s", line, title, line)