# local-llm-training-k8s
Production-style distributed training framework for decoder-only transformer models, built on PyTorch DDP and Kubernetes (kind), with MLflow tracking and pluggable model/data adapters


## Goal
- Minimal-yet-realistic training platform focused on CPU-based DDP correctness and reproducibility.
- Pluggable architecture: `ModelAdapter`, `DataModule`, trainer separated by contracts.
- Local K8s via kind and IndexedJob for multi-pod DDP; MLflow for params/metrics/artifacts; checkpoints are resumable.

## Quickstart (v0.1 skeleton)
1) Install deps (uses `uv` by default):
   ```bash
   uv sync
   ```
2) Lint & type check:
   ```bash
   make lint
   ```
3) Run tests:
   ```bash
   make test
   ```
4) CLI smoke:
   ```bash
   python -m llmtrain --help
   ```

## Repo layout (early milestone)
- `src/llmtrain/`: package code
  - `cli.py` / `__main__.py`: CLI entrypoint
  - `config/`, `registry/`, `models/`, `data/`, `training/`, `utils/`: scaffolds for upcoming milestones
- `tests/`: pytest suite (includes CLI smoke)
- `pyproject.toml`: project + tool config (ruff, mypy, pytest)
- `.pre-commit-config.yaml`: ruff/mypy hooks
- `Makefile`: `format`, `lint`, `test`

## Tooling
- `ruff` for lint/format, `mypy` for types, `pytest` for tests, `pre-commit` for hooks.
- `uv` manages the environment/lockfile.
