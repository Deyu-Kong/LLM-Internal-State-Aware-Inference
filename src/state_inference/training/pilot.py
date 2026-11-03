"""Training primitives for an unpadded, sequential single-probe pilot."""
import hashlib
import math
import torch
from state_inference.metrics.binary import binary_metrics


def frozen_digest(probe):
    digest = hashlib.sha256()
    for name, param in probe.named_parameters():
        if not param.requires_grad:
            digest.update(name.encode())
            digest.update(param.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def prefix_cache(probe, tokens, device):
    ids = torch.tensor([tokens], device=device)
    with torch.no_grad():
        return probe.model.model(ids, use_cache=True).past_key_values


def example_logits(probe, example, device):
    return probe(prefix_cache(probe, example["input_ids"], device))


def evaluate(probe, examples, device):
    probe.eval()
    scores = []
    with torch.no_grad():
        for example in examples:
            scores.append(float(example_logits(probe, example, device).sigmoid().item()))
    labels = [row["label"] for row in examples]
    return binary_metrics(labels, scores), scores


def length_position_baseline(train, validation):
    """Fit a tiny CPU logistic baseline using training-only feature statistics."""
    def features(rows):
        return torch.tensor([[math.log1p(len(r["input_ids"])), math.log1p(r["step_index"])] for r in rows], dtype=torch.float32)
    x, v = features(train), features(validation)
    mean, scale = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-6)
    x, v = (x-mean)/scale, (v-mean)/scale
    weight = torch.zeros(2, requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    y = torch.tensor([r["label"] for r in train], dtype=torch.float32)
    optimizer = torch.optim.Adam([weight, bias], lr=0.05)
    for _ in range(200):
        optimizer.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(x @ weight + bias, y).backward()
        optimizer.step()
    with torch.no_grad():
        scores = (v @ weight + bias).sigmoid().tolist()
    return binary_metrics([r["label"] for r in validation], scores), {
        "mean": mean.tolist(), "scale": scale.tolist(), "weight": weight.tolist(), "bias": bias.item()}


def trainable_state(probe):
    return {name: p.detach().cpu().clone() for name, p in probe.named_parameters() if p.requires_grad}


def load_trainable_state(probe, state):
    parameters = {name: p for name, p in probe.named_parameters() if p.requires_grad}
    if parameters.keys() != state.keys():
        raise ValueError("Adapter checkpoint parameter names do not match")
    with torch.no_grad():
        for name, param in parameters.items():
            param.copy_(state[name].to(param.device))
