"""Sequential, unpadded Llama single-probe correctness prototype.

This implementation follows the paper's single-token equations, with explicitly
chosen embedding/head initialization. Official code was unavailable at review time.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import math

import torch
from torch import nn
from transformers import DynamicCache
from transformers.cache_utils import DynamicLayer


class GatedLoRALinear(nn.Module):
    """Frozen linear plus an unscaled rank-r delta, active only in probe context."""

    def __init__(self, base, rank, gate):
        super().__init__()
        if type(rank) is not int or not 1 <= rank <= min(base.in_features, base.out_features):
            raise ValueError("rank must be a positive integer no greater than projection dimensions")
        self.base = base
        self.base.requires_grad_(False)
        self.gate = gate
        self.lora_a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, value):
        result = self.base(value)
        if self.gate.get():
            result = result + (value @ self.lora_a.T) @ self.lora_b.T
        return result


def linear_pseudo_target(outcome, position, response_length):
    """Outcome-derived heuristic, never a ground-truth process correctness label."""
    if type(outcome) is not int or outcome not in (0, 1):
        raise ValueError("outcome must be integer 0 or 1")
    if type(response_length) is not int or response_length < 1:
        raise ValueError("response_length must be positive")
    if type(position) is not int or not 1 <= position <= response_length:
        raise ValueError("position must index a response token (1-based)")
    return 0.5 + (outcome - 0.5) * position / response_length


def fork_dynamic_cache(cache, config):
    """Clone and detach every KV tensor on-device before any verifier forward.

    Deliberately O(context length) in memory/time, including cached inference tensors.
    Reject sliding/static/offloaded caches rather than silently changing semantics.
    """
    if type(cache) is not DynamicCache or len(cache.layers) != config.num_hidden_layers:
        raise ValueError("Expected a fully populated ordinary DynamicCache")
    if getattr(cache, "offloading", False):
        raise ValueError("Offloaded caches are unsupported")
    if any(type(layer) is not DynamicLayer or layer.keys is None or layer.values is None
           for layer in cache.layers):
        raise ValueError("Only initialized full-attention DynamicLayer caches are supported")
    lengths = {layer.keys.shape[-2] for layer in cache.layers}
    if len(lengths) != 1 or next(iter(lengths)) == 0:
        raise ValueError("Cache must contain a nonempty prefix of equal length at every layer")
    branch = DynamicCache(config=config)
    for i, layer in enumerate(cache.layers):
        branch.update(layer.keys.detach().clone(), layer.values.detach().clone(), i)
    return branch


class SingleKVProbe(nn.Module):
    """A shared frozen Llama backbone with gated Q/K/V adapters and one probe.

    Only equal-length, unpadded full-attention prefixes are supported. Run sequentially;
    this is not a serving/concurrency implementation. Calling model directly outside
    verification uses the original frozen projections, including after adapter updates.
    """

    def __init__(self, model, rank=2, init_token_id=1):
        super().__init__()
        if model.config.model_type != "llama":
            raise ValueError("This prototype is validated only for Llama")
        if type(init_token_id) is not int or not 0 <= init_token_id < model.config.vocab_size:
            raise ValueError("init_token_id must index the frozen embedding table")
        # Validate before mutating the supplied model.
        projections = [getattr(layer.self_attn, name) for layer in model.model.layers
                       for name in ("q_proj", "k_proj", "v_proj")]
        if any(not isinstance(p, nn.Linear) for p in projections):
            raise ValueError("Projections must be unwrapped nn.Linear modules")
        if type(rank) is not int or not 1 <= rank <= min(min(p.in_features, p.out_features) for p in projections):
            raise ValueError("Invalid LoRA rank")
        self.model = model
        self.model.requires_grad_(False)
        self._gate = ContextVar("single_kv_probe_gate", default=False)
        for layer in self.model.model.layers:
            for name in ("q_proj", "k_proj", "v_proj"):
                setattr(layer.self_attn, name, GatedLoRALinear(getattr(layer.self_attn, name), rank, self._gate))
        self.probe_embedding = nn.Parameter(model.get_input_embeddings().weight[init_token_id].detach().clone())
        width = model.config.hidden_size
        self.head = nn.Sequential(nn.Linear(width, width), nn.SiLU(),
                                  nn.Linear(width, width), nn.SiLU(),
                                  nn.Linear(width, 1), nn.Sigmoid())
        self.head.to(device=self.probe_embedding.device, dtype=self.probe_embedding.dtype)
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()  # Keep backbone dropout off, while retaining adapter autograd.
        return self

    @contextmanager
    def _verification(self):
        token = self._gate.set(True)
        try:
            yield
        finally:
            self._gate.reset(token)

    def forward(self, generator_cache):
        branch = fork_dynamic_cache(generator_cache, self.model.config)
        return self._forward_private_cache(branch)

    def _forward_private_cache(self, private_cache):
        """Internal timing helper: consumes a PRIVATE cache and appends the probe KV.

        External callers should use forward(generator_cache), which always forks.
        """
        batch = private_cache.layers[0].keys.shape[0]
        embedding = self.probe_embedding.view(1, 1, -1).expand(batch, 1, -1)
        with self._verification():
            output = self.model.model(inputs_embeds=embedding, past_key_values=private_cache,
                                      use_cache=True, return_dict=True)
        return self.head(output.last_hidden_state[:, -1]).squeeze(-1)
