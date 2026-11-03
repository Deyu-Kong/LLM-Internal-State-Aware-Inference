import hashlib
import json
from pathlib import Path
import pytest
import yaml
from state_inference.data.config import load_data_config
from state_inference.data.preparation import prepare_data
from state_inference.data.sources import gsm8k_sources


def configuration(tmp_path, name="run"):
    return {"experiment_id": name, "seed": 42, "output_dir": str(tmp_path / "runs"),
            "processed_dir": str(tmp_path / "processed"), "raw_dir": str(tmp_path / "raw"),
            "synthetic_id_roots": 20, "synthetic_ood_roots": 5, "gsm8k": None}


def test_offline_pipeline_reproducible_hashes_and_no_overwrite(tmp_path):
    value = configuration(tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(value))
    output = prepare_data(path)
    dataset = json.loads((output / "dataset_manifest.json").read_text())
    assert dataset["audit"] == {"unique_roots": 25, "unique_questions": 25, "samples": 100}
    for file in dataset["files"]:
        assert hashlib.sha256(Path(file["path"]).read_bytes()).hexdigest() == file["sha256"]
    before = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        prepare_data(path)
    assert (output / "manifest.json").read_bytes() == before
    value["experiment_id"] = "second"
    path.write_text(yaml.safe_dump(value))
    second = json.loads((prepare_data(path) / "dataset_manifest.json").read_text())
    assert [f["sha256"] for f in dataset["files"]] == [f["sha256"] for f in second["files"]]
    assert dataset["split_assignments"]["sha256"] == second["split_assignments"]["sha256"]
    log = [json.loads(x) for x in (output / "events.jsonl").read_text().splitlines()]
    assert log[0]["seed"] == 42
    assert log[-1]["status"] == "success"
    assert log[-1]["runtime_seconds"] >= 0


def test_verified_download_and_corrupted_cache_rejected(tmp_path, monkeypatch):
    payload = b'{"fixture": true}\n'
    digest = hashlib.sha256(payload).hexdigest()
    settings = {"revision": "a" * 40, "sha256": {"train.jsonl": digest}}
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, n): return payload
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: Response())
    with pytest.raises(FileNotFoundError):
        gsm8k_sources(tmp_path, settings)
    source = gsm8k_sources(tmp_path, settings, allow_download=True)[0]
    assert source["sha256"] == digest
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("cache should avoid network"))
    gsm8k_sources(tmp_path, settings)
    cached = Path(source["path"])
    cached.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        gsm8k_sources(tmp_path, settings, allow_download=True)
    assert cached.read_bytes() == b"corrupted"


def test_source_mismatch_not_published(tmp_path, monkeypatch):
    import io
    import urllib.request
    settings = {"revision": "a" * 40, "sha256": {"train.jsonl": "0" * 64}}
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b"wrong"))
    with pytest.raises(ValueError, match="mismatch"):
        gsm8k_sources(tmp_path, settings, allow_download=True)
    assert not list(tmp_path.rglob("train.jsonl"))


def test_failed_run_is_logged(tmp_path):
    value = configuration(tmp_path)
    value["gsm8k"] = {"revision": "a" * 40, "sha256": {name: "0" * 64 for name in ["train.jsonl", "test.jsonl", "LICENSE"]}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(FileNotFoundError):
        prepare_data(path)
    manifest = json.loads((tmp_path / "runs/run/manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "run_failed" in (tmp_path / "runs/run/events.jsonl").read_text()


@pytest.mark.parametrize("field,bad", [("seed", True), ("experiment_id", "../escape"), ("synthetic_id_roots", 0), ("gsm8k", {"revision": "main", "sha256": {}})])
def test_invalid_data_config(tmp_path, field, bad):
    value = configuration(tmp_path)
    value[field] = bad
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError):
        load_data_config(path)
