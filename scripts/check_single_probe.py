"""One forward/backward and diagnostic timings; never performs optimizer updates."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

from state_inference.cli import write_json
from state_inference.config import load_config
from state_inference.manifest import environment, git_state
from state_inference.runtime.devices import select_idle_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1))
    args = parser.parse_args()
    config = load_config(args.config)
    gpu = select_idle_gpu(args.gpu) if args.gpu is not None else None
    if not gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    import yaml
    from state_inference.kv_probe import SingleKVProbe, linear_pseudo_target
    from state_inference.kv_probe.single_probe import fork_dynamic_cache

    torch.set_num_threads(2)
    torch.manual_seed(config.seed)
    device = "cuda:0" if gpu else "cpu"
    if gpu:
        assert torch.cuda.device_count() == 1
        torch.cuda.reset_peak_memory_stats()
    directory = Path(config.output_dir) / config.experiment_id
    directory.mkdir(parents=True, exist_ok=False)
    resolved = config.to_dict()
    (directory / "config.yaml").write_text(yaml.safe_dump(resolved), encoding="utf-8")
    settings = {"lora_rank": 2, "lora_scaling": 1, "lora_targets": ["q_proj", "k_proj", "v_proj"],
                "probe_init_token_id": 1, "head": "32 -> SiLU -> 32 -> SiLU -> 1 -> sigmoid",
                "train_probe_embedding": True, "cache_isolation": "detached full clone on same device",
                "warmup_iterations": 3, "timing_iterations": 10, "optimizer_steps": 0}
    write_json(directory / "probe_config.json", settings)
    model_config = LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                               num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                               max_position_embeddings=64, attention_dropout=0.0)
    model_config._attn_implementation = "eager"
    backbone = LlamaForCausalLM(model_config).to(device).eval()
    probe = SingleKVProbe(backbone, rank=settings["lora_rank"], init_token_id=1)
    write_json(directory / "model_config.json", model_config.to_dict())
    dataset = {"name": "synthetic_probe_gradient_fixture", "version": 1, "prompt_ids": [1, 2],
               "response_ids": [3, 4, 5, 6], "prefix_response_position": 2,
               "outcome": 1, "outcome_is_arbitrary_fixture": True,
               "target_kind": "outcome_derived_linear_pseudo_confidence_not_process_label",
               "split": "mechanism_check_only", "used_for_training": False}
    write_json(directory / "dataset_manifest.json", dataset)
    source = Path("data/manifests/otv_sources.json")
    (directory / "otv_sources.json").write_bytes(source.read_bytes())
    env = environment()
    env.update(cuda_version=torch.version.cuda, gpu_information=gpu,
               gpu_status="single_authorized_gpu" if gpu else "cpu_only")
    command = ["python", "scripts/check_single_probe.py", "--config", args.config]
    if gpu:
        command += ["--gpu", str(args.gpu)]
    manifest = {"schema_version": 1, "experiment_id": config.experiment_id, "git": git_state(),
                "environment": env, "seed": config.seed, "config": resolved,
                "config_sha256": hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest(),
                "model": {"name": "random_tiny_llama", "revision": "model_config.json + seed + package versions", "pretrained": False},
                "adapter_revision": "random initialization; probe_config.json + seed + Git commit",
                "tokenizer_revision": None, "dataset_manifest": "dataset_manifest.json",
                "probe_config": "probe_config.json", "batch_size": 1, "input_length": 4,
                "output_length": 1, "sampling_parameters": {"do_sample": False},
                "dtype": "float32", "reproduce_argv": command,
                "started_at": datetime.now(timezone.utc).isoformat()}
    def sync():
        if gpu:
            torch.cuda.synchronize()
    def measured(fn):
        sync()
        start = time.perf_counter()
        value = fn()
        sync()
        return value, (time.perf_counter() - start) * 1000
    with (directory / "events.jsonl").open("w", encoding="utf-8") as log:
        def emit(event, **fields):
            log.write(json.dumps({"event": event, "timestamp": datetime.now(timezone.utc).isoformat(), **fields}) + "\n")
            log.flush()
        emit("run_started", seed=config.seed, environment=env)
        start = time.perf_counter()
        try:
            with torch.no_grad():
                prefix = backbone(torch.tensor([[1, 2, 3, 4]], device=device), use_cache=True).past_key_values
            cache_before = [(layer.keys.clone(), layer.values.clone()) for layer in prefix.layers]
            weights_before = {name: p.detach().clone() for name, p in probe.named_parameters()}
            target = torch.tensor([linear_pseudo_target(1, 2, 4)], device=device)
            probe.train()
            score, forward_ms = measured(lambda: probe(prefix))
            loss = torch.nn.functional.mse_loss(score, target)
            _, backward_ms = measured(loss.backward)
            grads = {}
            for name, p in probe.named_parameters():
                torch.testing.assert_close(p, weights_before[name], rtol=0, atol=0)
                if p.requires_grad:
                    assert p.grad is not None and torch.isfinite(p.grad).all(), name
                    grads[name] = float(p.grad.norm().item())
                else:
                    assert p.grad is None, name
            assert any(value > 0 for name, value in grads.items() if name.endswith("lora_b"))
            assert grads["probe_embedding"] > 0
            for layer, (k, v) in zip(prefix.layers, cache_before):
                torch.testing.assert_close(layer.keys, k, rtol=0, atol=0)
                torch.testing.assert_close(layer.values, v, rtol=0, atol=0)
            # Continue from an untouched independent baseline and the probed generator.
            reference = fork_dynamic_cache(prefix, backbone.config)
            probe.eval()
            with torch.no_grad():
                ids = torch.tensor([[5]], device=device)
                for _ in range(3):
                    baseline = backbone(ids, past_key_values=reference, use_cache=True)
                    observed = backbone(ids, past_key_values=prefix, use_cache=True)
                    torch.testing.assert_close(baseline.logits, observed.logits, rtol=0, atol=0)
                    reference, prefix = baseline.past_key_values, observed.past_key_values
                    ids = baseline.logits[:, -1].argmax(-1, keepdim=True)
                for a, b in zip(reference.layers, prefix.layers):
                    torch.testing.assert_close(a.keys, b.keys, rtol=0, atol=0)
                    torch.testing.assert_close(a.values, b.values, rtol=0, atol=0)
                # Timing uses the original 4-token prefix, independent of continuation.
                timing_cache = backbone(torch.tensor([[1, 2, 3, 4]], device=device), use_cache=True).past_key_values
                for _ in range(settings["warmup_iterations"]):
                    probe(timing_cache)
                timings = {"cache_fork_ms": [], "private_probe_forward_ms": [], "end_to_end_probe_ms": []}
                for _ in range(settings["timing_iterations"]):
                    branch, fork_ms = measured(lambda: fork_dynamic_cache(timing_cache, backbone.config))
                    _, private_ms = measured(lambda: probe._forward_private_cache(branch))
                    _, total_ms = measured(lambda: probe(timing_cache))
                    timings["cache_fork_ms"].append(fork_ms)
                    timings["private_probe_forward_ms"].append(private_ms)
                    timings["end_to_end_probe_ms"].append(total_ms)
            result = {"status": "success", "score": float(score.detach().item()), "target": float(target.item()),
                      "loss": float(loss.detach().item()), "target_kind": dataset["target_kind"],
                      "trainable_parameters": sum(p.numel() for p in probe.parameters() if p.requires_grad),
                      "total_parameters": sum(p.numel() for p in probe.parameters()),
                      "gradient_norms": grads, "generator_cache_unchanged": True,
                      "continuation_logits_and_cache_equal": True, "all_weights_unchanged": True,
                      "optimizer_steps": 0, "forward_with_autograd_ms": forward_ms, "backward_ms": backward_ms,
                      "timing_samples": timings,
                      "timing_mean_ms": {k: statistics.mean(v) for k, v in timings.items()},
                      "note": "Tiny synthetic diagnostic only; not official reproduction, calibrated confidence, or production benchmark"}
            write_json(directory / "result.json", result)
            emit("checks_passed", loss=result["loss"], trainable_parameters=result["trainable_parameters"], optimizer_steps=0)
            manifest["status"] = "success"
        except Exception:
            manifest["status"] = "failed"
            raise
        finally:
            sync()
            manifest.update(runtime_seconds=time.perf_counter() - start,
                            finished_at=datetime.now(timezone.utc).isoformat(),
                            peak_gpu_memory_allocated_bytes=torch.cuda.max_memory_allocated() if gpu else None)
            write_json(directory / "manifest.json", manifest)
            emit("run_finished", status=manifest["status"], runtime_seconds=manifest["runtime_seconds"])
    print(directory)


if __name__ == "__main__":
    main()
