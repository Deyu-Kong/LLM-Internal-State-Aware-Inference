import hashlib
import json
from pathlib import Path
import subprocess
import sys
import yaml
from state_inference.cli import run_smoke
import pytest


def test_cli_and_artifacts(tmp_path):
    config = tmp_path / "input.yaml"
    config.write_text(yaml.safe_dump({"experiment_id": "test", "seed": 7, "output_dir": str(tmp_path / "runs")}))
    process = subprocess.run([sys.executable, "-m", "state_inference", "smoke", "--config", str(config)], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    output = Path(process.stdout.strip())
    manifest = json.loads((output / "manifest.json").read_text())
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert manifest["runtime_seconds"] >= 0
    assert manifest["seed"] == events[0]["seed"] == 7
    assert events[0]["environment"]["python"]
    assert events[-1]["runtime_seconds"] == manifest["runtime_seconds"]
    assert manifest["environment"]["gpu_status"] == "not_inspected_cpu_only_smoke"
    assert json.loads((output / "dataset_manifest.json").read_text())["version"] is None
    saved_config = yaml.safe_load((output / "config.yaml").read_text())
    assert manifest["config_sha256"] == hashlib.sha256(json.dumps(saved_config, sort_keys=True).encode()).hexdigest()
    before = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_smoke(config)
    assert (output / "manifest.json").read_bytes() == before
    saved_config["output_dir"] = str(tmp_path / "second")
    config.write_text(yaml.safe_dump(saved_config))
    second = run_smoke(config)
    assert (second / "result.json").read_bytes() == (output / "result.json").read_bytes()


def test_invalid_cli_creates_no_run(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text("[]")
    result = subprocess.run([sys.executable, "-m", "state_inference", "smoke", "--config", str(config)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 1
    assert "error:" in result.stderr
    assert list(tmp_path.iterdir()) == [config]
