#!/usr/bin/env bash
set -euo pipefail
cd /data1/kdy/Project/otv-probe-pilot/source
export CUDA_VISIBLE_DEVICES=GPU-57936391-3548-4e9b-3a76-171251751d17
export HF_HUB_OFFLINE=1
export PYTHONPATH=src
export OMP_NUM_THREADS=2
MODEL=/data1/kdy/LLMs/Qwen2.5-Coder-1.5B-Instruct
TRACES=../runs/small-traces-001
sha256sum scripts/run_probe_pilot_train.py scripts/analyze_active_probe_pilot.py src/otv/*.py > ../provenance/small-training-source-sha256.txt
timeout 1800 ../.venv-runtime/bin/python scripts/run_probe_pilot_train.py train --mode otv --model "$MODEL" --traces "$TRACES" --output ../runs/small-otv-001 --epochs 3 > ../logs/small-otv-001.log 2>&1
sleep 2
timeout 1800 ../.venv-runtime/bin/python scripts/run_probe_pilot_train.py train --mode no_lora --model "$MODEL" --traces "$TRACES" --output ../runs/small-no-lora-001 --epochs 3 > ../logs/small-no-lora-001.log 2>&1
sleep 2
timeout 900 ../.venv-runtime/bin/python scripts/run_probe_pilot_train.py hidden --model "$MODEL" --traces "$TRACES" --output ../runs/small-hidden-001 --epochs 3 > ../logs/small-hidden-001.log 2>&1
../.venv-runtime/bin/python scripts/analyze_active_probe_pilot.py --otv ../runs/small-otv-001 --no-lora ../runs/small-no-lora-001 --hidden ../runs/small-hidden-001 --output ../runs/small-comparison-001.json
printf 'complete\n' > ../runs/small-pipeline-001-complete.txt
