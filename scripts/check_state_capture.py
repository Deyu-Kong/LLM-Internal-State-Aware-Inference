"""Short synthetic state-capture validation. No downloads, training or benchmarks."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

from state_inference.cli import write_json
from state_inference.config import load_config
from state_inference.manifest import environment, git_state
from state_inference.runtime.devices import select_idle_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1), help="Omit to run on CPU")
    args = parser.parse_args()
    config = load_config(args.config)
    gpu = select_idle_gpu(args.gpu) if args.gpu is not None else None
    if gpu is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Device restriction is applied before importing any CUDA-capable package.
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    import yaml
    from state_inference.state_capture import capture_states

    torch.set_num_threads(2)
    torch.manual_seed(config.seed)
    device = "cuda:0" if gpu else "cpu"
    if gpu:
        assert torch.cuda.device_count() == 1
        torch.cuda.reset_peak_memory_stats()
    output = Path(config.output_dir) / config.experiment_id
    output.mkdir(parents=True, exist_ok=False)
    resolved = config.to_dict()
    (output / "config.yaml").write_text(yaml.safe_dump(resolved), encoding="utf-8")
    model_config = LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                               num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                               max_position_embeddings=64, attention_dropout=0.0)
    model_config._attn_implementation = "eager"
    model = LlamaForCausalLM(model_config).to(device).eval()
    write_json(output / "model_config.json", model_config.to_dict())
    write_json(output / "dataset_manifest.json", {"name": "synthetic_token_ids", "version": 1, "input_ids": [[1, 2, 3, 4]], "split": "smoke", "training": False})
    env = environment()
    env.update(cuda_version=torch.version.cuda, gpu_information=gpu,
               gpu_status="single_authorized_gpu" if gpu else "cpu_only")
    command = ["python", "scripts/check_state_capture.py", "--config", args.config]
    if gpu:
        command += ["--gpu", str(args.gpu)]
    manifest = {"schema_version": 1, "experiment_id": config.experiment_id, "git": git_state(),
                "seed": config.seed, "environment": env, "config": resolved,
                "config_sha256": hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest(),
                "model": {"name": "random_tiny_llama", "revision": "model_config.json + seed + pinned transformers version", "pretrained": False},
                "tokenizer_revision": None, "adapter_revision": None,
                "dataset_manifest": "dataset_manifest.json", "batch_size": 1,
                "input_length": 4, "output_length": 4,
                "sampling_parameters": {"do_sample": False}, "dtype": "float32",
                "warmup_iterations": 0, "benchmark": False,
                "reproduce_argv": command, "started_at": datetime.now(timezone.utc).isoformat()}
    def sync():
        if gpu:
            torch.cuda.synchronize()
    with (output / "events.jsonl").open("w", encoding="utf-8") as log:
        def emit(event, **fields):
            log.write(json.dumps({"event": event, "timestamp": datetime.now(timezone.utc).isoformat(), **fields}) + "\n")
            log.flush()
        emit("run_started", seed=config.seed, environment=env)
        sync()
        start = time.perf_counter()
        try:
            def decode(observe):
                ids = torch.tensor([[1, 2, 3, 4]], device=device)
                cache, logits, tokens, snapshots = None, [], [], []
                with torch.inference_mode():
                    for _ in range(4):
                        out = model(ids, past_key_values=cache, use_cache=True, output_hidden_states=observe)
                        if observe:
                            snapshots.append(capture_states(out, hidden_layers=[0, -1], kv_layers=[0, 1], positions=[-1], kv_positions=[0, -1]))
                        logits.append(out.logits[:, -1].clone())
                        ids = out.logits[:, -1].argmax(-1, keepdim=True)
                        tokens.append(ids.item())
                        cache = out.past_key_values
                return logits, tokens, cache, snapshots
            plain = decode(False)
            observed = decode(True)
            for a, b in zip(plain[0], observed[0]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            assert plain[1] == observed[1]
            for a, b in zip(plain[2].layers, observed[2].layers):
                torch.testing.assert_close(a.keys, b.keys, rtol=0, atol=0)
                torch.testing.assert_close(a.values, b.values, rtol=0, atol=0)
            snapshot = observed[3][0]
            assert snapshot.logits.shape == (1, 1, 32)
            assert snapshot.hidden_states[2].shape == (1, 1, 32)
            assert snapshot.keys[0].shape == snapshot.values[0].shape == (1, 2, 2, 8)
            write_json(output / "result.json", {"status": "success", "tokens": observed[1],
                       "exact_logits_equal": True, "exact_cache_equal": True,
                       "shapes": {"logits": list(snapshot.logits.shape), "hidden": list(snapshot.hidden_states[2].shape), "kv": list(snapshot.keys[0].shape)},
                       "note": "Synthetic correctness check only; no model quality or speed claim"})
            manifest["status"] = "success"
        except Exception:
            manifest["status"] = "failed"
            raise
        finally:
            sync()
            manifest.update(runtime_seconds=time.perf_counter() - start,
                            finished_at=datetime.now(timezone.utc).isoformat(),
                            peak_gpu_memory_allocated_bytes=torch.cuda.max_memory_allocated() if gpu else None)
            write_json(output / "manifest.json", manifest)
            emit("run_finished", status=manifest["status"], runtime_seconds=manifest["runtime_seconds"])
    print(output)


if __name__ == "__main__":
    main()
