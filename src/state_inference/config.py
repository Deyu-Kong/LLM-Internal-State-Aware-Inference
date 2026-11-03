"""Validated YAML configuration for infrastructure-only smoke runs."""
from dataclasses import asdict, dataclass
from pathlib import Path
import re
import yaml


@dataclass(frozen=True)
class Config:
    experiment_id: str
    seed: int
    output_dir: str

    def to_dict(self):
        return asdict(self)


def load_config(path: str | Path) -> Config:
    with Path(path).open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict) or set(value) != {"experiment_id", "seed", "output_dir"}:
        raise ValueError("Configuration requires exactly experiment_id, seed, output_dir")
    name, seed, output = (value[k] for k in ("experiment_id", "seed", "output_dir"))
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
        raise ValueError("experiment_id must be a safe identifier (1–80 characters)")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if not isinstance(output, str) or not output.strip():
        raise ValueError("output_dir must be a nonempty path")
    return Config(name, seed, output)
