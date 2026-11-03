"""Separate strict schema for the data milestone, without changing smoke configs."""
from dataclasses import dataclass, asdict
from pathlib import Path
import re
import yaml


@dataclass(frozen=True)
class DataConfig:
    experiment_id: str
    seed: int
    output_dir: str
    processed_dir: str
    raw_dir: str
    synthetic_id_roots: int
    synthetic_ood_roots: int
    gsm8k: dict | None

    def to_dict(self):
        return asdict(self)


def load_data_config(path):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    fields = set(DataConfig.__dataclass_fields__)
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"Data configuration requires exactly {sorted(fields)}")
    if not isinstance(value["experiment_id"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value["experiment_id"]):
        raise ValueError("Unsafe experiment_id")
    if type(value["seed"]) is not int or not 0 <= value["seed"] < 2**32:
        raise ValueError("Invalid seed")
    for key in ("output_dir", "processed_dir", "raw_dir"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError(f"{key} must be a nonempty path")
    for key in ("synthetic_id_roots", "synthetic_ood_roots"):
        if type(value[key]) is not int or not 1 <= value[key] <= 10000:
            raise ValueError(f"Invalid {key}")
    gsm = value["gsm8k"]
    if gsm is not None:
        if not isinstance(gsm, dict) or set(gsm) != {"revision", "sha256"}:
            raise ValueError("gsm8k requires revision and sha256")
        if not isinstance(gsm["revision"], str) or not re.fullmatch(r"[0-9a-f]{40}", gsm["revision"]):
            raise ValueError("Pin GSM8K to a full Git commit")
        if not isinstance(gsm["sha256"], dict) or set(gsm["sha256"]) != {"train.jsonl", "test.jsonl", "LICENSE"}:
            raise ValueError("Pin train, test and license file hashes")
        if any(not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v) for v in gsm["sha256"].values()):
            raise ValueError("Invalid source SHA256")
    return DataConfig(**value)
