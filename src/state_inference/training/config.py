"""Bounded pilot-training configuration, separate from smoke/data schemas."""
from pathlib import Path
import math
import re
import yaml

FIELDS = {"experiment_id", "seed", "output_dir", "model_config", "source_file", "source_sha256", "source_revision",
          "train_roots", "validation_roots", "per_root", "max_tokens", "rank", "optimizer_steps",
          "gradient_accumulation", "learning_rate", "weight_decay", "max_wall_seconds", "train_eval_limit"}


def load_training_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if not isinstance(cfg, dict) or set(cfg) != FIELDS:
        raise ValueError("Unexpected or missing pilot configuration fields")
    if not isinstance(cfg["experiment_id"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", cfg["experiment_id"]):
        raise ValueError("Unsafe experiment ID")
    bounds = {"seed": (0, 2**32-1), "train_roots": (1, 2000), "validation_roots": (1, 500), "per_root": (1, 16),
              "max_tokens": (16, 1024), "rank": (1, 16), "optimizer_steps": (1, 300),
              "gradient_accumulation": (1, 8), "max_wall_seconds": (1, 1800), "train_eval_limit": (1, 256)}
    for key, (lo, hi) in bounds.items():
        if type(cfg[key]) is not int or not lo <= cfg[key] <= hi:
            raise ValueError(f"Invalid bounded integer {key}")
    for key in ("output_dir", "model_config", "source_file"):
        if not isinstance(cfg[key], str) or not cfg[key].strip():
            raise ValueError(f"Invalid path {key}")
    for key, length in (("source_revision", 40), ("source_sha256", 64)):
        if not isinstance(cfg[key], str) or not re.fullmatch(f"[0-9a-f]{{{length}}}", cfg[key]):
            raise ValueError(f"Pin {key}")
    for key in ("learning_rate", "weight_decay"):
        if type(cfg[key]) not in (float, int) or not math.isfinite(cfg[key]):
            raise ValueError(f"Invalid numeric {key}")
    if not 0 < cfg["learning_rate"] <= 0.01 or not 0 <= cfg["weight_decay"] <= 1:
        raise ValueError("Unsafe optimizer settings")
    return cfg
