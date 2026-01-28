# local-llm-training-k8s
Production-style distributed training framework for decoder-only transformer models, built on PyTorch DDP and Kubernetes (kind), with MLflow tracking and pluggable model/data adapters


## Goal
- Minimal-yet-realistic training platform focused on CPU-based DDP correctness and reproducibility.
- Pluggable architecture: `ModelAdapter`, `DataModule`, trainer separated by contracts.
- Local K8s via kind and IndexedJob for multi-pod DDP; MLflow for params/metrics/artifacts; checkpoints are resumable.

## Quickstart (v0.2 config-driven CLI)
1) Install deps (uses `uv` with dev extras for tooling):
   ```bash
   uv sync --extra dev
   ```
2) Install pre-commit hooks:
   ```bash
   pre-commit install
   ```
3) Format code:
   ```bash
   make format
   ```
3) Lint & type check:
   ```bash
   make lint
   ```
4) Run tests:
   ```bash
   make test
   ```
5) CLI help:
   ```bash
   python -m llmtrain --help
   ```
6) Validate a config:
   ```bash
   python -m llmtrain validate --config configs/presets/example.yaml
   ```
7) Inspect the resolved config (defaults materialized):
   ```bash
   python -m llmtrain print-config --config configs/presets/example.yaml
   ```
8) Prepare a run (creates `runs/<run_id>/` with config + metadata):
   ```bash
   python -m llmtrain train --config configs/presets/example.yaml
   ```

Notes:
- `train` currently runs a short dry-run loop (fake data) and emits summary output; full training is wired in later milestones.
- `--dry-run` is accepted but currently a no-op (the dry-run loop always executes).
- Use `--json` on any command for machine-readable output.

## Config structure (v0.2)
Configs are YAML files validated by Pydantic with strict fields. Example presets live in `configs/presets/`.

```yaml
schema_version: 1
run:         # run identity & reproducibility
model:       # model architecture
data:        # dataset + splits
trainer:     # optimizer + loop pacing
ddp:         # distributed runtime hints
mlflow:      # tracking integration
logging:     # stdout/file logging
output:      # run directory + persistence
```

### Section details
- `schema_version`: integer schema version (currently `1`).
- `run`: `name`, `seed`, `device`, `deterministic`, `notes`.
- `model`: `name`, `init`, `block_size`, `d_model`, `n_layers`, `n_heads`, `d_ff`, `dropout`, `tie_embeddings`, `vocab_size`.
- `data`: `name`, `cache_dir`, `num_workers`, `train_split`, `val_split`, optional `dataset_name`, `dataset_config`, `text_column`.
- `trainer`: `max_steps`, `micro_batch_size`, `grad_accum_steps`, `lr`, `weight_decay`, `warmup_steps`, `max_grad_norm`, `log_every_steps`, `eval_every_steps`, `save_every_steps`.
- `ddp`: `enabled`, `backend`, `init_method`, `timeout_sec`, `find_unused_parameters`, optional `rank`, `world_size`, `local_rank`, `master_addr`, `master_port`.
- `mlflow`: `enabled`, `tracking_uri`, `experiment`, optional `run_name`, `log_models`.
- `logging`: `level`, `json_output`, `log_to_file`, `file_name`.
- `output`: `root_dir`, optional `run_id`, `save_config_copy`, `save_meta_json`.

### Output layout
When running `train`, a run directory is created under `output.root_dir`:
- `runs/<run_id>/config.yaml`: resolved config (with defaults).
- `runs/<run_id>/meta.json`: metadata snapshot (git SHA, argv, env).
- `runs/<run_id>/logs/`: log files when `logging.log_to_file=true`.

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
