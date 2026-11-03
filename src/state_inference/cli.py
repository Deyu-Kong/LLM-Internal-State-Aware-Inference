"""CPU-only smoke command with immutable run directories and JSONL logging."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import yaml
from .config import load_config
from .manifest import environment, git_state


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_smoke(config_path):
    config = load_config(config_path)
    resolved = config.to_dict()
    digest = hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest()
    run_dir = Path(config.output_dir) / config.experiment_id
    run_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    env = environment()
    random.seed(config.seed)
    # Use a local RNG: the smoke output must not depend on unrelated callers.
    sample = random.Random(config.seed).random()
    (run_dir / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8")
    dataset = {"name": "none", "version": None, "status": "not_used_cpu_smoke", "split": None}
    write_json(run_dir / "dataset_manifest.json", dataset)
    log_path = run_dir / "events.jsonl"
    with log_path.open("w", encoding="utf-8") as log:
        def emit(event, **fields):
            log.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "event": event, "experiment_id": config.experiment_id, **fields}) + "\n")
            log.flush()
        emit("run_started", seed=config.seed, environment=env)
        result = {"status": "success", "task": "infrastructure_smoke", "sample": sample}
        write_json(run_dir / "result.json", result)
        elapsed = time.perf_counter() - start
        manifest = {
            "schema_version": 1, "experiment_id": config.experiment_id,
            "status": "success", "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": elapsed, "seed": config.seed,
            "git": git_state(), "environment": env,
            "config_sha256": digest, "config": resolved,
            "model": {"name": None, "revision": None},
            "tokenizer_revision": None, "adapter_revision": None,
            "dataset_manifest": "dataset_manifest.json",
            "batch_size": None, "input_length": None, "output_length": None,
            "sampling_parameters": {},
            "reproduce_argv": ["state-inference", "smoke", "--config", "config.yaml"],
            "reproduce_note": "Run from a fresh directory with the copied config.yaml; output paths are relative to the working directory.",
        }
        write_json(run_dir / "manifest.json", manifest)
        emit("run_finished", status="success", runtime_seconds=elapsed)
    return run_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke", help="Check infrastructure without models or GPUs")
    smoke.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    try:
        output = run_smoke(args.config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(output)
    return 0
