"""Copy selected states from an existing forward output, never mutate its cache."""
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StateSnapshot:
    # Hidden layer 0 is embeddings; KV layer 0 is the first transformer block.
    hidden_states: dict[int, Any]
    logits: Any
    keys: dict[int, Any]
    values: dict[int, Any]


def _indices(indices, length, label):
    indices = tuple(indices)
    if not indices or any(type(i) is not int or not -length <= i < length for i in indices):
        raise ValueError(f"{label} must contain indices in [-{length}, {length})")
    return tuple(i % length for i in indices)


def capture_states(output, *, hidden_layers, kv_layers, positions, kv_positions=None):
    """Snapshot to CPU, detached and cloned; selection never writes source tensors.

    positions index the current forward's hidden/logit sequence. kv_positions index
    the entire cached sequence (defaults to positions). Supports DynamicCache.layers
    and legacy tuple caches. Attention weights and masked/sliding caches are out of scope.
    """
    if output.hidden_states is None or output.past_key_values is None:
        raise ValueError("Forward must enable output_hidden_states=True and use_cache=True")
    pos = _indices(positions, output.logits.shape[1], "positions")
    hs = _indices(hidden_layers, len(output.hidden_states), "hidden_layers")
    cache = output.past_key_values
    if hasattr(cache, "layers"):
        pairs = [(layer.keys, layer.values) for layer in cache.layers]
    else:
        pairs = list(cache)
    layers = _indices(kv_layers, len(pairs), "kv_layers")

    def copy(tensor):
        return tensor.detach().to("cpu").clone()

    keys, values = {}, {}
    for layer in layers:
        k, v = pairs[layer]
        kp = _indices(kv_positions if kv_positions is not None else positions, k.shape[-2], "kv_positions")
        keys[layer] = copy(k[:, :, kp, :])
        values[layer] = copy(v[:, :, kp, :])
    return StateSnapshot(
        {layer: copy(output.hidden_states[layer][:, pos, :]) for layer in hs},
        copy(output.logits[:, pos, :]), keys, values,
    )
