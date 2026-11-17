"""Short real-model mechanics check; synthetic labels are not scientific evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    # Visibility must be constrained by the launcher before importing torch.
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible.startswith('GPU-') or ',' in visible:
        raise RuntimeError('Select one verified idle GPU by UUID before launching.')
    import torch
    from transformers import AutoTokenizer
    from otv import OTVVerifier
    from otv.data import encode_trace

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    import faulthandler
    faulthandler.dump_traceback_later(120, repeat=True)
    start = time.perf_counter()
    print('Loading frozen reasoner and verifier', flush=True)
    model = OTVVerifier.from_pretrained(args.model, device='cuda:0', dtype=torch.bfloat16,
                                      head_dropout=0).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    encoded = encode_trace(tokenizer, 'What is 12 + 15?',
                           'Adding twelve and fifteen gives twenty-seven. The answer is 27.')
    ids = torch.tensor([encoded['input_ids']], device='cuda:0')
    positions = [encoded['response_start'], ids.shape[1] - 3, ids.shape[1]]

    def digest(tensors):
        checksum = hashlib.sha256()
        for tensor in tensors:
            checksum.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
        return checksum.hexdigest()

    frozen = [p for p in model.parameters() if not p.requires_grad]
    print('Hashing all frozen weights before check', flush=True)
    before_weights = digest(frozen)
    print('Checking parallel probes, adapter gating, and future masking', flush=True)
    with torch.no_grad():
        cache = model.prefill(ids)
        before_cache = digest([t for layer in cache.layers for t in layer])
        parallel = model(cache, positions)
        independent = torch.cat([model(model.prefill(ids[:, :p]), [p]) for p in positions], dim=1)
        torch.testing.assert_close(parallel, independent, atol=3e-3, rtol=3e-3)
        # Nonzero B makes the check sensitive to active adapter pathways.
        for name, parameter in model.named_parameters():
            if 'lora_B' in name:
                parameter.normal_(std=0.001)
        active = model(cache, positions)
        active_independent = torch.cat([
            model(model.prefill(ids[:, :p]), [p]) for p in positions], dim=1)
        torch.testing.assert_close(active, active_independent, atol=3e-3, rtol=3e-3)
        changed = ids.clone()
        changed[:, positions[0]:] = tokenizer.eos_token_id
        torch.testing.assert_close(model(model.prefill(changed), [positions[0]]),
                                   active[:, :1], atol=3e-3, rtol=3e-3)
    model.train()
    print('Checking backward and one synthetic optimizer step', flush=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    prediction = model(cache, positions)
    # Synthetic target checks the differentiable pathway only.
    loss = (prediction.float() - 1).square().mean()
    loss.backward()
    assert model.truth_embedding.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for n, p in model.named_parameters() if 'lora_B' in n)
    assert all(p.grad is None for p in frozen)
    optimizer.step()
    assert before_cache == digest([t for layer in cache.layers for t in layer])
    print('Hashing all frozen weights after check', flush=True)
    assert before_weights == digest(frozen)
    torch.cuda.synchronize()
    result = dict(status='passed', scope='mechanics_only_not_effectiveness',
                  model=args.model, gpu_uuid=visible, tokens=ids.shape[1],
                  positions=positions, max_parallel_difference=(parallel-independent).abs().max().item(),
                  frozen_weights_unchanged=True, original_cache_unchanged=True,
                  synthetic_loss=loss.item(), seconds=time.perf_counter()-start,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  torch_version=torch.__version__)
    (output / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    main()
