"""Version and runtime provenance without loading GPU libraries."""
import importlib.metadata
import platform
import subprocess
from pathlib import Path


def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_state():
    try:
        root = Path(__file__).resolve().parents[2]
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True).stdout)
        return {"commit": sha, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def environment():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": {name: version(name) for name in ("state-inference", "PyYAML", "torch", "transformers", "peft", "datasets", "vllm")},
        "cuda_version": None,
        "gpu_information": None,
        "gpu_status": "not_inspected_cpu_only_smoke",
    }
