# local-llm-training-k8s
Production-style distributed training framework for decoder-only transformers, targeting correctness-first CPU training today and a local Kubernetes DDP pipeline by v1.2.

## Goal
- Build a realistic, test-driven training stack for decoder-only GPT-style models.
- Keep it modular: `ModelAdapter`, `DataModule`, and `Trainer` are contract-based.
- Reach "one command" local K8s training with checkpoints, metrics, and reproducible runs.

## What exists today (v0.6)
- Real single-process training loop with gradient accumulation and LR schedule.
- Checkpointing every `save_every_steps` with resume via `--resume`.
- Periodic evaluation with `val/*` metrics and `final_val_loss` summary fields.
- Config-driven CLI with strict validation and JSON output support.
- Deterministic run directories with config + metadata snapshots.

## High-level roadmap
- **v0.6**: evaluation loop and validation metrics.
- **v0.7**: MLflow experiment tracking (optional dependency).
- **v0.8**: real GPT decoder model (causal attention).
- **v0.9**: real data pipeline (HuggingFace datasets + tokenizer).
- **v1.0**: Distributed Data Parallel on a single machine.
- **v1.1**: Kubernetes `kind` + IndexedJob orchestration.
- **v1.2**: production hardening (signals, CI, docs polish).

## Quickstart (v0.6 config-driven CLI)
1) Install deps (uses `uv`):
   ```bash
   uv sync
   ```
2) Optional tooling for lint/test:
   ```bash
   uv sync --extra dev
   pre-commit install
   ```
3) Optional MLflow dependency (for v0.7+ tracking):
   ```bash
   uv sync --extra mlflow
   ```
   Or (editable install without `uv`):
   ```bash
   pip install -e '.[mlflow]'
   ```
4) Run lint & tests (optional but recommended):
   ```bash
   make lint
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
8) Train (creates `runs/<run_id>/` with config + metadata + checkpoints):
   ```bash
   python -m llmtrain train --config configs/presets/example.yaml
   ```
9) Resume from the latest checkpoint in a run:
   ```bash
   python -m llmtrain train --config configs/presets/example.yaml --resume <run_id>
   ```
10) Resume from a specific checkpoint file:
   ```bash
   python -m llmtrain train --config configs/presets/example.yaml --resume runs/<run_id>/checkpoints/step_20.pt
   ```

Notes:
- `--dry-run` runs a forward-only sanity check (no optimization).
- Use `--json` on any command for machine-readable output.

## Config structure (v0.5)
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
- `runs/<run_id>/checkpoints/`: periodic checkpoints (`step_<N>.pt`).

## Repo layout (current)
- `src/llmtrain/`: package code
  - `cli.py` / `__main__.py`: CLI entrypoint
  - `training/`: single-process trainer + checkpointing
  - `models/`, `data/`: dummy adapters for fast CPU smoke tests
  - `config/`, `registry/`, `utils/`: contracts and runtime plumbing
- `tests/`: pytest suite (includes CLI smoke)
- `pyproject.toml`: project + tool config (ruff, mypy, pytest)
- `.pre-commit-config.yaml`: ruff/mypy hooks
- `Makefile`: `format`, `lint`, `test`

## Tooling
- `ruff` for lint/format, `mypy` for types, `pytest` for tests, `pre-commit` for hooks.
- `uv` manages the environment/lockfile.

## Development process

This project is built with AI-assisted developer tooling (e.g., Cursor / LLM copilots) as a productivity multiplier for iteration, refactoring, and documentation.

All architecture, trade-offs, and final implementations are reviewed and owned by me. The codebase is validated via automated linting, type-checking, and tests, with a commit history intended to remain readable and reviewable.


