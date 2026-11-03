"""Safety checks run without importing torch or initializing CUDA."""
import runpy
import subprocess
from pathlib import Path
import pytest

select = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/check_state_capture.py"))["select_idle_gpu"]


@pytest.mark.parametrize("index", [-1, 2, 7])
def test_other_gpus_rejected(index):
    with pytest.raises(ValueError, match="authorized"):
        select(index)


@pytest.mark.parametrize("memory,utilization,pids", [("1", "0", ""), ("0", "5", ""), ("0", "0", "123")])
def test_occupied_gpu_rejected(monkeypatch, memory, utilization, pids):
    def run(args, **kwargs):
        text = pids if "--query-compute-apps=pid" in args else f"GPU-example, A800, {memory}, {utilization}, 595"
        return subprocess.CompletedProcess(args, 0, text)
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(RuntimeError, match="occupied"):
        select(0)


def test_idle_gpu_uses_uuid(monkeypatch):
    def run(args, **kwargs):
        assert args[args.index("-i") + 1] == "1"
        text = "" if "--query-compute-apps=pid" in args else "GPU-example, A800, 0, 0, 595"
        return subprocess.CompletedProcess(args, 0, text)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert select(1)["physical_index"] == 1
    import os
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-example"
