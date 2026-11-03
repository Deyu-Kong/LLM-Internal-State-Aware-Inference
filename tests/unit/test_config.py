import pytest
from state_inference.config import load_config


def test_example():
    config = load_config("configs/smoke.yaml")
    assert config.seed == 42
    assert config.experiment_id == "local-smoke-001"


@pytest.mark.parametrize("text", ["[]", "", "experiment_id: x", "experiment_id: ../escape\nseed: 1\noutput_dir: x", "experiment_id: x\nseed: true\noutput_dir: x", "experiment_id: x\nseed: -1\noutput_dir: x", "experiment_id: x\nseed: 1\noutput_dir: ''", "experiment_id: x\nseed: 1\noutput_dir: x\nunknown: y"])
def test_invalid(tmp_path, text):
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_config(path)
