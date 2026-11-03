"""Mechanism checks; these use no trained weights or research labels."""
import copy
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM
from state_inference.kv_probe import SingleKVProbe, linear_pseudo_target
from state_inference.kv_probe.single_probe import fork_dynamic_cache


def model():
    torch.manual_seed(42)
    cfg = LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=64, attention_dropout=0.0)
    cfg._attn_implementation = "eager"
    return LlamaForCausalLM(cfg).eval()


def prefix(backbone):
    with torch.no_grad():
        return backbone(torch.tensor([[1, 2, 3, 4]]), use_cache=True).past_key_values


def test_backward_and_frozen_base():
    probe = SingleKVProbe(model())
    cache = prefix(probe.model)
    before = [(x.keys.clone(), x.values.clone()) for x in cache.layers]
    score = probe(cache)
    assert score.shape == (1,)
    assert 0 <= score.item() <= 1
    torch.nn.functional.mse_loss(score, torch.tensor([0.75])).backward()
    trainable = {name: param for name, param in probe.named_parameters() if param.requires_grad}
    assert trainable
    assert all(name == "probe_embedding" or name.startswith("head.") or ".lora_" in name for name in trainable)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable.values())
    assert probe.probe_embedding.grad.norm() > 0
    assert sum(p.grad.norm() for n, p in trainable.items() if n.endswith("lora_b")) > 0
    # B starts at zero, hence A's first backward is exactly zero; this is expected.
    assert all(torch.count_nonzero(p.grad) == 0 for n, p in trainable.items() if n.endswith("lora_a"))
    assert all(p.grad is None for p in probe.parameters() if not p.requires_grad)
    for layer, (k, v) in zip(cache.layers, before):
        torch.testing.assert_close(layer.keys, k, rtol=0, atol=0)
        torch.testing.assert_close(layer.values, v, rtol=0, atol=0)
    assert cache.get_seq_length() == 4


def test_nonzero_adapters_do_not_change_generator_or_continuation():
    backbone = model()
    reference = copy.deepcopy(backbone)
    probe = SingleKVProbe(backbone)
    with torch.no_grad():
        for name, p in probe.named_parameters():
            if name.endswith("lora_b"):
                p.fill_(0.2)
    cache = prefix(backbone)
    plain_cache = prefix(reference)
    score = probe(cache)
    score.sum().backward()
    assert any(p.grad.norm() > 0 for n, p in probe.named_parameters() if n.endswith("lora_a"))
    probe.train()
    assert not backbone.training
    with torch.no_grad():
        ids = torch.tensor([[5]])
        for _ in range(3):
            plain = reference(ids, past_key_values=plain_cache, use_cache=True)
            observed = backbone(ids, past_key_values=cache, use_cache=True)
            torch.testing.assert_close(plain.logits, observed.logits, rtol=0, atol=0)
            ids = plain.logits[:, -1].argmax(-1, keepdim=True)
            plain_cache, cache = plain.past_key_values, observed.past_key_values
    for a, b in zip(plain_cache.layers, cache.layers):
        torch.testing.assert_close(a.keys, b.keys, rtol=0, atol=0)
        torch.testing.assert_close(a.values, b.values, rtol=0, atol=0)


def test_attention_matches_full_sequence_with_zero_delta():
    backbone = model()
    probe = SingleKVProbe(backbone)
    cache = prefix(backbone)
    private = fork_dynamic_cache(cache, backbone.config)
    embed = probe.probe_embedding.reshape(1, 1, -1)
    with probe._verification():
        one = backbone.model(inputs_embeds=embed, past_key_values=private,
                             use_cache=True, output_attentions=True)
    full_input = torch.cat([backbone.get_input_embeddings()(torch.tensor([[1, 2, 3, 4]])), embed], dim=1)
    full = backbone.model(inputs_embeds=full_input, use_cache=False, output_attentions=True)
    torch.testing.assert_close(one.last_hidden_state, full.last_hidden_state[:, -1:], rtol=1e-5, atol=1e-6)
    assert one.attentions[0].shape == (1, 4, 1, 5)
    torch.testing.assert_close(one.attentions[0], full.attentions[0][:, :, -1:], rtol=1e-5, atol=1e-6)
    assert (one.attentions[0] > 0).all()  # all four prefix tokens AND probe self
    assert private.get_seq_length() == 5
    assert cache.get_seq_length() == 4


def test_inference_cache_is_detached_and_independent():
    backbone = model()
    with torch.inference_mode():
        cache = backbone(torch.tensor([[1, 2, 3, 4]]), use_cache=True).past_key_values
    probe = SingleKVProbe(backbone)
    copied = fork_dynamic_cache(cache, backbone.config)
    for original, fork in zip(cache.layers, copied.layers):
        assert original.keys.data_ptr() != fork.keys.data_ptr()
        assert not fork.keys.is_inference()
        assert not fork.keys.requires_grad
    probe(cache).sum().backward()
    assert probe.probe_embedding.grad is not None


def test_gate_restored_after_exception(monkeypatch):
    probe = SingleKVProbe(model())
    cache = prefix(probe.model)
    def fail(*args, **kwargs):
        raise RuntimeError("injected")
    monkeypatch.setattr(probe.model.model, "forward", fail)
    with pytest.raises(RuntimeError, match="injected"):
        probe(cache)
    assert not probe._gate.get()
    assert cache.get_seq_length() == 4


def test_bad_cache_and_double_wrapping_rejected():
    backbone = model()
    probe = SingleKVProbe(backbone)
    with pytest.raises(ValueError):
        probe(DynamicCache(config=backbone.config))
    with pytest.raises(ValueError):
        SingleKVProbe(backbone)


@pytest.mark.parametrize("args,expected", [((1, 1, 4), 0.625), ((0, 4, 4), 0.0), ((1, 4, 4), 1.0)])
def test_pseudo_target(args, expected):
    assert linear_pseudo_target(*args) == expected


@pytest.mark.parametrize("args", [(True, 1, 4), (2, 1, 4), (1, 0, 4), (1, 5, 4), (1, 1, 0)])
def test_invalid_target(args):
    with pytest.raises(ValueError):
        linear_pseudo_target(*args)
