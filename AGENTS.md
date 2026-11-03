# Research workspace instructions

Read `llm_internal_state_research_implementation_plan.md` before changing research methods. Implement one milestone at a time; do not implement WP1, WP2 and WP3 simultaneously. Never invent experimental results.

## Remote compute policy

- Prefer `ssh a800`; fallback is `ssh node32`.
- Only physical GPU 0 and GPU 1 are authorized. Never use GPU 2 or higher.
- Check occupancy immediately before each run. If the selected GPU is occupied, stop or use the other authorized idle GPU; never terminate other processes.
- Restrict CUDA visibility by the selected GPU UUID before importing torch. Use only one idle GPU for small checks.
- Do not modify shared environments, GPU settings, services, or other users' files. Use a project-local environment.
- Estimate duration, VRAM and output storage before long runs. No GPU training without a separately scoped task.

## Validation

Use `uv run --locked pytest -q` for the lightweight scaffold. State capture tests additionally require `requirements/state-capture.txt`; they skip when those optional dependencies are absent. Run the state capture checks on CPU before any short GPU validation. Record configuration, commit, dependency versions, dataset provenance, logs and exact commands for results.
