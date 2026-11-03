"""Reproducible data artifacts, counts and split audits (no model execution)."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import yaml
from state_inference.cli import write_json
from state_inference.manifest import environment, git_state
from .config import load_data_config
from .gsm8k import prepare_gsm8k
from .sources import gsm8k_sources, sha256_file
from .splits import SPLITS, assert_no_leakage
from .synthetic import VERSION, generate_synthetic, validate_trace


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def prepare_data(config_path, allow_download=False):
    config = load_data_config(config_path)
    output = Path(config.output_dir) / config.experiment_id
    processed = Path(config.processed_dir) / config.experiment_id
    if output.resolve() == processed.resolve() or output.resolve() in processed.resolve().parents or processed.resolve() in output.resolve().parents:
        raise ValueError("Run metadata and processed directories must be separate")
    if output.exists() or processed.exists():
        raise FileExistsError("Run or processed directory exists; choose a new experiment_id")
    output.mkdir(parents=True, exist_ok=False)
    resolved = config.to_dict()
    (output / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8")
    env = environment()
    env["gpu_status"] = "not_used_cpu_data_preparation"
    manifest = {"schema_version": 1, "experiment_id": config.experiment_id, "seed": config.seed,
                "git": git_state(), "environment": env, "config": resolved,
                "config_sha256": hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest(),
                "model": None, "tokenizer_revision": None, "adapter_revision": None,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "reproduce_argv": ["python", "scripts/prepare_data.py", "--config", str(config_path)] + (["--download"] if allow_download else []),
                "dataset_manifest": "dataset_manifest.json"}
    with (output / "events.jsonl").open("w", encoding="utf-8") as log:
        def emit(event, **fields):
            log.write(json.dumps({"event": event, "timestamp": datetime.now(timezone.utc).isoformat(), **fields}) + "\n")
            log.flush()
        emit("run_started", seed=config.seed, environment=env)
        start = time.perf_counter()
        try:
            synthetic = generate_synthetic(config.synthetic_id_roots, config.synthetic_ood_roots, config.seed)
            for row in synthetic:
                validate_trace(row)
            datasets = {"synthetic": synthetic}
            sources, excluded, raw_counts = [], [], {}
            if config.gsm8k is not None:
                sources = gsm8k_sources(config.raw_dir, config.gsm8k, allow_download)
                paths = {s["name"]: Path(s["path"]) for s in sources}
                train, test = read_jsonl(paths["train.jsonl"]), read_jsonl(paths["test.jsonl"])
                raw_counts = {"train": len(train), "test": len(test)}
                gsm, excluded = prepare_gsm8k(train, test, config.seed)
                datasets["gsm8k"] = gsm
                (output / "LICENSE.gsm8k.txt").write_bytes(paths["LICENSE"].read_bytes())
                emit("sources_verified", raw_counts=raw_counts, excluded_training_overlaps=len(excluded))
            all_records = [row for rows in datasets.values() for row in rows]
            audit = assert_no_leakage(all_records)
            processed.mkdir(parents=True, exist_ok=False)
            files, counts = [], {}
            for name, rows in datasets.items():
                folder = processed / name
                folder.mkdir()
                counts[name] = {}
                for split in SPLITS:
                    subset = [row for row in rows if row["split"] == split]
                    path = folder / f"{split}.jsonl"
                    write_jsonl(path, subset)
                    counts[name][split] = len(subset)
                    files.append({"path": str(path), "dataset": name, "split": split,
                                  "rows": len(subset), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
            assignments = {}
            for row in all_records:
                item = assignments.setdefault(row["root_id"], {"root_id": row["root_id"], "split": row["split"], "samples": 0})
                item["samples"] += 1
            assignment_path = processed / "split_assignments.jsonl"
            write_jsonl(assignment_path, [assignments[k] for k in sorted(assignments)])
            exclusion_path = processed / "excluded_training_overlaps.jsonl"
            write_jsonl(exclusion_path, excluded)
            # Only synthetic examples are redistributed here, never raw benchmark text.
            examples = output / "synthetic_examples.jsonl"
            write_jsonl(examples, synthetic[:4])
            dataset_manifest = {
                "schema_version": 1, "synthetic_version": VERSION,
                "gsm8k_revision": config.gsm8k["revision"] if config.gsm8k else None,
                "sources": sources, "raw_gsm8k_counts": raw_counts, "counts": counts, "files": files,
                "split_policy": "seeded SHA256-ranked roots; synthetic ID 80/10/10; GSM8K official train 90/10; official test reserved; synthetic OOD held out",
                "question_normalization": "Unicode NFKC + whitespace collapse + casefold; semantic near-duplicates not detected",
                "audit": audit, "excluded_training_overlaps": len(excluded),
                "split_assignments": {"path": str(assignment_path), "sha256": sha256_file(assignment_path)},
                "exclusion_file": {"path": str(exclusion_path), "sha256": sha256_file(exclusion_path)},
                "label_policy": "exact synthetic replay labels; GSM8K reference answer only, process/outcome predictions unknown; no pseudo-targets without generated traces/tokenizer",
                "synthetic_variants": dict(Counter(row["variant"] for row in synthetic)),
                "synthetic_outcomes": {"correct": sum(row["final_answer_correct"] for row in synthetic),
                                       "incorrect": sum(not row["final_answer_correct"] for row in synthetic)},
                "tokenizer_revision": None, "generated_by_llm": False,
            }
            write_json(output / "dataset_manifest.json", dataset_manifest)
            summary = {"status": "success", "counts": counts, "audit": audit,
                       "excluded_training_overlaps": len(excluded), "synthetic_samples": len(synthetic),
                       "gsm8k_process_labels": "unavailable", "gpu_used": False, "training_performed": False}
            write_json(output / "result.json", summary)
            emit("data_prepared", **summary)
            manifest["status"] = "success"
        except Exception as exc:
            manifest["status"] = "failed"
            emit("run_failed", error_type=type(exc).__name__, message=str(exc))
            raise
        finally:
            manifest.update(runtime_seconds=time.perf_counter() - start, finished_at=datetime.now(timezone.utc).isoformat())
            write_json(output / "manifest.json", manifest)
            emit("run_finished", status=manifest["status"], runtime_seconds=manifest["runtime_seconds"])
    return output
