"""Run a bounded PRM800K local-step pilot; GPU guard precedes torch imports."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import os
import random
import time
import yaml

from state_inference.cli import write_json
from state_inference.data.prm800k import extract_candidates, choose_subset, model_text
from state_inference.data.preparation import read_jsonl, write_jsonl
from state_inference.data.sources import sha256_file
from state_inference.manifest import environment, git_state
from state_inference.runtime.devices import select_idle_gpu
from state_inference.training.config import load_training_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1))
    parser.add_argument("--preflight", action="store_true", help="CPU-only one-backward check, no optimizer steps")
    args = parser.parse_args()
    cfg = load_training_config(args.config)
    if args.preflight and args.gpu is not None:
        parser.error("Preflight must use CPU")
    gpu = select_idle_gpu(args.gpu) if args.gpu is not None else None
    if gpu is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from state_inference.kv_probe import SingleKVProbe
    from state_inference.training.pilot import (frozen_digest, prefix_cache, evaluate, example_logits,
        length_position_baseline, trainable_state, load_trainable_state)
    from state_inference.kv_probe.single_probe import fork_dynamic_cache
    from state_inference.metrics.binary import binary_metrics
    from fetch_pilot_assets import verify

    torch.set_num_threads(2)
    torch.manual_seed(cfg["seed"])
    if gpu:
        assert torch.cuda.device_count() == 1
        torch.cuda.reset_peak_memory_stats()
    device = "cuda:0" if gpu else "cpu"
    name = cfg["experiment_id"] + ("-preflight" if args.preflight else "")
    output = Path(cfg["output_dir"]) / name
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))
    env = environment()
    env.update(cuda_version=torch.version.cuda, gpu_information=gpu, gpu_status="single_authorized_gpu" if gpu else "cpu_only")
    manifest = {"schema_version": 1, "experiment_id": name, "config": cfg, "seed": cfg["seed"],
                "git": git_state(), "environment": env, "mode": "preflight" if args.preflight else "training",
                "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
                "model": None, "tokenizer_revision": None, "adapter_revision": None,
                "dataset_manifest": "dataset_manifest.json", "dtype": "float32", "batch_size": 1,
                "effective_batch_size": cfg["gradient_accumulation"], "input_length_limit": cfg["max_tokens"],
                "output_length": 1, "sampling_parameters": {}, "started_at": datetime.now(timezone.utc).isoformat(),
                "reproduce_argv": ["python", "scripts/train_probe.py", "--config", args.config] + (["--preflight"] if args.preflight else []) + (["--gpu", str(args.gpu)] if gpu else [])}
    def sync():
        if gpu:
            torch.cuda.synchronize()
    with (output / "events.jsonl").open("w") as log:
        def emit(event, **fields):
            record = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps({"event": event, **fields}), flush=True)
        start = time.perf_counter()
        emit("run_started", seed=cfg["seed"], environment=env)
        try:
            if sha256_file(cfg["source_file"]) != cfg["source_sha256"]:
                raise ValueError("PRM800K source hash mismatch")
            model_cfg = json.loads(Path(cfg["model_config"]).read_text())
            assets = []
            for filename, expected in model_cfg["files"].items():
                path = Path(model_cfg["local_dir"]) / filename
                verify(path, expected)
                assets.append({"file": filename, "sha256": sha256_file(path), "bytes": path.stat().st_size})
            write_json(output / "model_assets.json", {**model_cfg, "verified_files": assets})
            tokenizer = AutoTokenizer.from_pretrained(model_cfg["local_dir"], local_files_only=True)
            source_rows = read_jsonl(Path(cfg["source_file"]))
            candidates, audit = extract_candidates(source_rows)
            subset = choose_subset(candidates, seed=cfg["seed"], train_roots=cfg["train_roots"],
                validation_roots=cfg["validation_roots"], per_root=cfg["per_root"],
                max_train=cfg["train_roots"]*cfg["per_root"], max_validation=cfg["validation_roots"]*cfg["per_root"])
            selected_ids = [{"sample_id": r["sample_id"], "root_id": r["root_id"], "split": r["split"]} for r in subset]
            examples, excluded = [], []
            for row in subset:
                tokens = tokenizer.encode(model_text(row), add_special_tokens=True)
                if len(tokens) > cfg["max_tokens"]:
                    excluded.append({"sample_id": row["sample_id"], "split": row["split"], "tokens": len(tokens), "reason": "over_length_no_truncation"})
                else:
                    examples.append({**row, "input_ids": tokens})
            train = [r for r in examples if r["split"] == "train"]
            validation = [r for r in examples if r["split"] == "validation"]
            if set(r["root_id"] for r in train) & set(r["root_id"] for r in validation):
                raise ValueError("Root leakage")
            if any(set(r["label"] for r in rows) != {0, 1} for rows in (train, validation)):
                raise ValueError("Both classes are required in both splits")
            for split, rows in (("train", train), ("validation", validation)):
                write_jsonl(output / f"{split}_samples.jsonl", rows)
            write_jsonl(output / "over_length_exclusions.jsonl", excluded)
            write_jsonl(output / "selected_ids.jsonl", selected_ids)
            data_manifest = {"source": "openai/prm800k phase1_train", "revision": cfg["source_revision"],
                "sha256": cfg["source_sha256"], "audit": audit, "label_policy": "+1 vs -1; neutral/uncertain excluded; human-corrected chosen history only; conflicts removed",
                "counts": {split: {"samples": len(rows), "roots": len(set(r['root_id'] for r in rows)),
                                   "positives": sum(r['label'] for r in rows), "negatives": sum(1-r['label'] for r in rows)}
                           for split, rows in (("train", train), ("validation", validation))},
                "over_length_exclusions": len(excluded), "max_tokens": cfg["max_tokens"], "truncation": False,
                "model_input_fields": ["problem", "prefix_steps", "candidate_step"], "official_test_used": False,
                "files": {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size} for p in output.glob("*.jsonl") if p.name != "events.jsonl"}}
            write_json(output / "dataset_manifest.json", data_manifest)
            emit("dataset_ready", counts=data_manifest["counts"], over_length_exclusions=len(excluded))
            backbone = AutoModelForCausalLM.from_pretrained(model_cfg["local_dir"], local_files_only=True,
                                                          dtype=torch.float32, attn_implementation="eager").eval()
            probe = SingleKVProbe(backbone, rank=cfg["rank"], init_token_id=tokenizer.eos_token_id, output_logits=True).to(device)
            backbone.config.to_json_file(output / "model_config.json")
            manifest.update(model={"name": model_cfg["model_id"], "revision": model_cfg["revision"]},
                            tokenizer_revision=model_cfg["revision"], adapter_revision="checkpoint.pt after training")
            before_frozen = frozen_digest(probe)
            initial = trainable_state(probe)
            ntrain = sum(p.numel() for p in probe.parameters() if p.requires_grad)
            # Preserve a real prefix and normal-generation continuation reference across training.
            cache = prefix_cache(probe, train[0]["input_ids"], device)
            cache_copy = [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers]
            with torch.no_grad():
                base_logits = backbone(torch.tensor([[tokenizer.eos_token_id]], device=device),
                    past_key_values=fork_dynamic_cache(cache, backbone.config), use_cache=True).logits.clone()
            if args.preflight:
                score = probe(cache)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(score, torch.tensor([train[0]["label"]], device=device, dtype=torch.float32))
                loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in probe.parameters() if p.requires_grad)
                result = {"status": "success", "optimizer_steps": 0, "loss": loss.item(), "trainable_parameters": ntrain}
            else:
                eval_train = train[:cfg["train_eval_limit"]]
                prior = sum(r["label"] for r in train)/len(train)
                baseline_constant = binary_metrics([r["label"] for r in validation], [prior]*len(validation))
                baseline_length, baseline_fit = length_position_baseline(train, validation)
                before_train, _ = evaluate(probe, eval_train, device)
                before_val, before_scores = evaluate(probe, validation, device)
                emit("before_training", train=before_train, validation=before_val,
                     constant_baseline=baseline_constant, length_position_baseline=baseline_length)
                optimizer = torch.optim.AdamW([p for p in probe.parameters() if p.requires_grad],
                                              lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
                negatives = sum(1-r["label"] for r in train)
                positives = len(train) - negatives
                pos_weight = torch.tensor([negatives/positives], device=device)
                rng, order, offset, updates = random.Random(cfg["seed"]), [], 0, 0
                sync()
                training_start = time.perf_counter()
                for update in range(cfg["optimizer_steps"]):
                    if time.perf_counter() - start > cfg["max_wall_seconds"]:
                        emit("wall_time_limit", completed_steps=updates)
                        break
                    probe.train()
                    optimizer.zero_grad(set_to_none=True)
                    total = 0.0
                    for _ in range(cfg["gradient_accumulation"]):
                        if offset >= len(order):
                            order = list(range(len(train)))
                            rng.shuffle(order)
                            offset = 0
                        row = train[order[offset]]
                        offset += 1
                        logits = example_logits(probe, row, device)
                        target = torch.tensor([row["label"]], device=device, dtype=torch.float32)
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
                        if not torch.isfinite(loss):
                            raise ValueError("Non-finite training loss")
                        (loss/cfg["gradient_accumulation"]).backward()
                        total += loss.item()/cfg["gradient_accumulation"]
                    norm = torch.nn.utils.clip_grad_norm_([p for p in probe.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    updates += 1
                    if updates == 1 or updates % 10 == 0:
                        emit("optimizer_step", step=updates, weighted_bce=total, gradient_norm=float(norm.item()), elapsed_seconds=time.perf_counter()-training_start)
                sync()
                training_seconds = time.perf_counter()-training_start
                after_train, _ = evaluate(probe, eval_train, device)
                after_val, after_scores = evaluate(probe, validation, device)
                final = trainable_state(probe)
                changed = [name for name in final if not torch.equal(initial[name], final[name])]
                assert updates > 0 and changed
                torch.save({"trainable": final, "model": model_cfg["model_id"], "revision": model_cfg["revision"],
                            "rank": cfg["rank"], "output_logits": True, "seed": cfg["seed"], "optimizer_steps": updates}, output / "checkpoint.pt")
                loaded = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
                load_trainable_state(probe, loaded["trainable"])
                with torch.no_grad():
                    restored_score = float(probe(prefix_cache(probe, validation[0]["input_ids"], device)).sigmoid().item())
                assert abs(restored_score-after_scores[0]) < 1e-6
                write_jsonl(output / "validation_predictions.jsonl", [{"sample_id": r["sample_id"], "root_id": r["root_id"],
                    "label": r["label"], "untrained_score": a, "trained_score": b} for r,a,b in zip(validation,before_scores,after_scores)])
                result = {"status": "success", "optimizer_steps": updates, "training_seconds": training_seconds,
                          "trainable_parameters": ntrain, "changed_trainable_tensors": len(changed),
                          "class_pos_weight": float(pos_weight.item()), "before": {"train": before_train, "validation": before_val},
                          "after": {"train": after_train, "validation": after_val}, "train_eval_samples": len(eval_train),
                          "baselines": {"constant": baseline_constant, "length_position": baseline_length, "length_position_fit": baseline_fit},
                          "checkpoint": {"file": "checkpoint.pt", "bytes": (output/'checkpoint.pt').stat().st_size, "sha256": sha256_file(output/'checkpoint.pt')},
                          "checkpoint_reload_verified": True, "single_seed_pilot": True}
            assert before_frozen == frozen_digest(probe), "Frozen backbone changed"
            for layer, (k,v) in zip(cache.layers,cache_copy):
                torch.testing.assert_close(layer.keys,k,rtol=0,atol=0)
                torch.testing.assert_close(layer.values,v,rtol=0,atol=0)
            with torch.no_grad():
                after_logits = backbone(torch.tensor([[tokenizer.eos_token_id]], device=device),
                    past_key_values=fork_dynamic_cache(cache, backbone.config), use_cache=True).logits
            torch.testing.assert_close(base_logits,after_logits,rtol=0,atol=0)
            result.update(frozen_backbone_unchanged=True, generator_cache_unchanged=True, generator_logits_unchanged=True)
            write_json(output / "result.json", result)
            emit("run_succeeded", optimizer_steps=result["optimizer_steps"])
            manifest["status"] = "success"
        except Exception as exc:
            manifest["status"] = "failed"
            emit("run_failed", error_type=type(exc).__name__, message=str(exc))
            raise
        finally:
            sync()
            manifest.update(runtime_seconds=time.perf_counter()-start, finished_at=datetime.now(timezone.utc).isoformat(),
                            peak_gpu_memory_allocated_bytes=torch.cuda.max_memory_allocated() if gpu else None)
            write_json(output / "manifest.json", manifest)
            emit("run_finished", status=manifest["status"], runtime_seconds=manifest["runtime_seconds"])


if __name__ == "__main__":
    main()
