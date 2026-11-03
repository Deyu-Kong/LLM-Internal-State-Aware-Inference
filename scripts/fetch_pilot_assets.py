"""Fetch public, pinned pilot assets on a network-capable host; no CUDA imports."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

from state_inference.cli import write_json
from state_inference.data.sources import sha256_file
from state_inference.data.prm800k import REVISION, SOURCE_SHA256


def acquire(url, path, expected):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = None
        try:
            with urllib.request.urlopen(url, timeout=60) as remote, tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                total = 0
                while block := remote.read(1024 * 1024):
                    total += len(block)
                    if total > expected["bytes"]:
                        raise ValueError("Asset exceeds pinned byte size")
                    stream.write(block)
            verify(temporary, expected)
            os.link(temporary, path)
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)
    verify(path, expected)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def verify(path, expected):
    if path.stat().st_size != expected["bytes"]:
        raise ValueError(f"Asset size mismatch: {path.name}")
    if "sha256" in expected:
        valid = sha256_file(path) == expected["sha256"]
    else:
        payload = path.read_bytes()
        valid = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest() == expected["git_blob"]
    if not valid:
        raise ValueError(f"Asset hash mismatch: {path.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default="configs/models/smollm2_135m.json")
    args = parser.parse_args()
    cfg = json.loads(Path(args.model_config).read_text())
    assets = []
    for name, expected in cfg["files"].items():
        url = f"https://huggingface.co/{cfg['model_id']}/resolve/{cfg['revision']}/{name}?download=true"
        assets.append(acquire(url, Path(cfg["local_dir"]) / name, expected))
        print("verified", name, flush=True)
    data_path = Path("data/raw/prm800k") / REVISION / "phase1_train.jsonl"
    url = f"https://media.githubusercontent.com/media/openai/prm800k/{REVISION}/prm800k/data/phase1_train.jsonl"
    source = acquire(url, data_path, {"bytes": 7900236, "sha256": SOURCE_SHA256})
    write_json(Path(cfg["local_dir"]) / "asset_manifest.json", {"model_id": cfg["model_id"], "revision": cfg["revision"], "files": assets})
    print("verified PRM800K", source["sha256"], flush=True)


if __name__ == "__main__":
    main()
