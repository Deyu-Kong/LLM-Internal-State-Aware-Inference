import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaForCausalLM
from state_inference.kv_probe import SingleKVProbe
from state_inference.training.pilot import frozen_digest, prefix_cache, trainable_state, load_trainable_state


def test_real_optimizer_updates_only_probe_and_checkpoint_reload(tmp_path):
    torch.manual_seed(42)
    cfg = LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, attention_dropout=0.0)
    cfg._attn_implementation = "eager"
    probe = SingleKVProbe(LlamaForCausalLM(cfg), output_logits=True)
    before_base = frozen_digest(probe)
    before = trainable_state(probe)
    cache = prefix_cache(probe, [1,2,3], "cpu")
    copied = cache.layers[0].keys.clone()
    optimizer = torch.optim.AdamW([p for p in probe.parameters() if p.requires_grad], lr=0.01, weight_decay=0)
    for _ in range(3):
        optimizer.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(probe(cache), torch.tensor([1.0])).backward()
        optimizer.step()
    assert frozen_digest(probe) == before_base
    after = trainable_state(probe)
    assert any(not torch.equal(before[k],after[k]) for k in after)
    torch.testing.assert_close(copied,cache.layers[0].keys,rtol=0,atol=0)
    score = probe(cache).detach()
    p = tmp_path / "adapter.pt"
    torch.save(after,p)
    load_trainable_state(probe,before)
    assert not torch.equal(score,probe(cache))
    load_trainable_state(probe,torch.load(p,weights_only=True))
    torch.testing.assert_close(score,probe(cache),rtol=0,atol=0)
