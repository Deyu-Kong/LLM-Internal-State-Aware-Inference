"""Shared GPU guard: query occupancy and select an authorized physical GPU UUID."""
import os
import subprocess


def select_idle_gpu(index):
    if index not in (0, 1):
        raise ValueError("Only physical GPU 0 or 1 is authorized")
    result = subprocess.run([
        "nvidia-smi", "-i", str(index),
        "--query-gpu=uuid,name,memory.used,utilization.gpu,driver_version",
        "--format=csv,noheader,nounits",
    ], check=True, capture_output=True, text=True)
    uuid, name, memory, utilization, driver = [x.strip() for x in result.stdout.strip().split(",")]
    apps = subprocess.run(["nvidia-smi", "-i", str(index), "--query-compute-apps=pid", "--format=csv,noheader"], check=True, capture_output=True, text=True)
    if int(memory) != 0 or int(utilization) != 0 or apps.stdout.strip():
        raise RuntimeError(f"GPU {index} is occupied; refusing to run")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    return {"physical_index": index, "uuid": uuid, "name": name, "driver": driver}

