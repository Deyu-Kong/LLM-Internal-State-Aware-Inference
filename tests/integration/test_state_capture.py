import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaForCausalLM
from state_inference.state_capture import capture_states


def tiny_model(seed=42):
    torch.manual_seed(seed)
    config = LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=64, attention_dropout=0.0)
    config._attn_implementation = "eager"
    return LlamaForCausalLM(config).eval()


def test_shapes_and_independent_snapshots():
    model = tiny_model()
    with torch.inference_mode():
        out = model(torch.tensor([[1, 2, 3, 4]]), use_cache=True, output_hidden_states=True)
        snap = capture_states(out, hidden_layers=[0, -1], kv_layers=[0, 1], positions=[0, -1])
        assert snap.logits.shape == (1, 2, 32)
        assert snap.hidden_states[2].shape == (1, 2, 32)
        assert snap.keys[0].shape == snap.values[0].shape == (1, 2, 2, 8)
        saved = out.hidden_states[-1].clone()
        snap.hidden_states[2].zero_()
        torch.testing.assert_close(saved, out.hidden_states[-1], rtol=0, atol=0)
        snap.keys[0].zero_()
        again = capture_states(out, hidden_layers=[-1], kv_layers=[0], positions=[0, -1])
        assert torch.count_nonzero(again.keys[0]) > 0
        assert not again.logits.requires_grad


def test_cached_decode_unchanged():
    model = tiny_model()
    def decode(observe):
        cache, ids, logits, snapshots = None, torch.tensor([[1, 2, 3, 4]]), [], []
        with torch.inference_mode():
            for _ in range(4):
                out = model(ids, past_key_values=cache, use_cache=True, output_hidden_states=observe)
                if observe:
                    snapshots.append(capture_states(out, hidden_layers=[-1], kv_layers=[0], positions=[-1], kv_positions=[0, -1]))
                logits.append(out.logits[:, -1].clone())
                ids = out.logits[:, -1].argmax(-1, keepdim=True)
                cache = out.past_key_values
        return logits, cache
    plain, plain_cache = decode(False)
    observed, observed_cache = decode(True)
    for a, b in zip(plain, observed):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(plain_cache.layers, observed_cache.layers):
        torch.testing.assert_close(a.keys, b.keys, rtol=0, atol=0)
        torch.testing.assert_close(a.values, b.values, rtol=0, atol=0)


@pytest.mark.parametrize("selection", [{"hidden_layers": [99]}, {"kv_layers": [2]}, {"positions": [4]}, {"positions": []}, {"kv_positions": [99]}])
def test_invalid_selection(selection):
    with torch.inference_mode():
        out = tiny_model()(torch.tensor([[1, 2, 3, 4]]), use_cache=True, output_hidden_states=True)
    args = dict(hidden_layers=[-1], kv_layers=[0], positions=[-1])
    args.update(selection)
    with pytest.raises(ValueError):
        capture_states(out, **args)
