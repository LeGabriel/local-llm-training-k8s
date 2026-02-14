# local-llm-training-k8s
Production-style distributed training framework for decoder-only transformers, targeting correctness-first CPU training today and a local Kubernetes DDP pipeline by v1.2.

## Goal
- Build a realistic, test-driven training stack for decoder-only GPT-style models.
- Keep it modular: `ModelAdapter`, `DataModule`, and `Trainer` are contract-based.
- Reach "one command" local K8s training with checkpoints, metrics, and reproducible runs.

## What exists today (v1.0)
- Real decoder-only GPT model adapter (`gpt`) with causal self-attention.
- Fast smoke-path model adapter (`dummy_gpt`) remains available.
- GPT tokenizer wiring via `tiktoken` (`gpt2` encoding).
- Real text data pipeline via Hugging Face datasets (`hf_text`).
- Real single-process training loop with gradient accumulation and LR schedule.
- **Distributed Data Parallel (DDP)** multi-process training via `torchrun`.
- Checkpointing every `save_every_steps` with resume via `--resume`.
- Periodic evaluation with `val/*` metrics and `final_val_loss` summary fields.
- Config-driven CLI with strict validation and JSON output support.
- Deterministic run directories with config + metadata snapshots.

## High-level roadmap
- **v0.5**: checkpointing & resume.
- **v0.6**: evaluation loop and validation metrics.
- **v0.7**: MLflow experiment tracking (optional dependency).
- **v0.8**: real GPT decoder model (causal attention).
- **v0.9**: real data pipeline (HuggingFace datasets + tokenizer).
- **v1.0**: Distributed Data Parallel on a single machine. **(current)**
- **v1.1**: Kubernetes `kind` + IndexedJob orchestration.
- **v1.2**: production hardening (signals, CI, docs polish).

## Quickstart (v0.9 config-driven CLI)
1) Install deps (uses `uv`):
   ```bash
   uv sync
   ```
2) Optional tooling for lint/test:
   ```bash
   uv sync --extra dev
   pre-commit install
   ```
3) Real-data dependencies:
   - `datasets` and `tiktoken` are core dependencies and are installed by default with `uv sync`.
   - `gpt_wikitext` works without any additional install extra.
4) Optional MLflow dependency (for v0.7+ tracking):
   ```bash
   uv sync --extra mlflow
   ```
   Or (editable install without `uv`):
   ```bash
   pip install -e '.[mlflow]'
   ```
5) Run lint & tests (optional but recommended):
   ```bash
   make lint
   make test
   ```
6) CLI help:
   ```bash
   python -m llmtrain --help
   ```
7) Validate a config:
   ```bash
   python -m llmtrain validate --config configs/presets/example.yaml
   ```
8) Inspect the resolved config (defaults materialized):
   ```bash
   python -m llmtrain print-config --config configs/presets/example.yaml
   ```
9) Train (creates `runs/<run_id>/` with config + metadata + checkpoints):
   ```bash
   python -m llmtrain train --config configs/presets/example.yaml
   ```
10) Resume from the latest checkpoint in a run:
   ```bash
   python -m llmtrain train --config configs/presets/example.yaml --resume <run_id>
   ```
11) Resume from a specific checkpoint file:
   ```bash
   python -m llmtrain train --config configs/presets/example.yaml --resume runs/<run_id>/checkpoints/step_20.pt
   ```

### GPT smoke preset (v0.8)

Use the v0.8 GPT decoder preset for a fast end-to-end smoke run:

```bash
python -m llmtrain train --config configs/presets/gpt_smoke.yaml
```

To inspect or validate the same preset before training:

```bash
python -m llmtrain print-config --config configs/presets/gpt_smoke.yaml
python -m llmtrain validate --config configs/presets/gpt_smoke.yaml
```

Notes:
- `--dry-run` runs a forward-only sanity check (no optimization).
- Use `--json` on any command for machine-readable output.

### GPT + WikiText preset (v0.9 real-data)

Use the v0.9 real-data preset to run GPT training on Hugging Face WikiText:

```bash
python -m llmtrain train --config configs/presets/gpt_wikitext.yaml
```

Inspect or validate before launching:

```bash
python -m llmtrain print-config --config configs/presets/gpt_wikitext.yaml
python -m llmtrain validate --config configs/presets/gpt_wikitext.yaml
```

Real-data behavior:
- `data.name: hf_text` loads `wikitext/wikitext-2-raw-v1` and reads from `data.text_column`.
- Tokenization uses the GPT tokenizer (`tiktoken` `gpt2` encoding), then builds next-token training windows.
- Processed/downloaded dataset artifacts are cached under `data.cache_dir` (preset default: `.cache/datasets`).
- The first run may take longer because dataset files are downloaded and processed.

Slow integration test coverage:
- `tests/test_hf_text_integration.py` includes `@pytest.mark.slow` real-data CLI training.
- Run only slow tests with:
  ```bash
  pytest -m slow
  ```
- This test asserts finite losses and a lower final train loss versus first-step loss.

## MLflow tracking (v0.7)

Install the optional MLflow dependency, then run the MLflow preset:

```bash
uv sync --extra mlflow
python -m llmtrain train --config configs/presets/ddp_mlflow.yaml
```

The `ddp_mlflow` preset is configured with a local SQLite backend URI
(`sqlite:///./mlflow.db`). After training, open the MLflow UI with:

```bash
mlflow ui --backend-store-uri sqlite:///./mlflow.db
```

You should see the run with logged params, metrics (`train/loss`, `train/lr`,
`val/loss`), and artifacts (`config.yaml`, `meta.json`).

## Distributed Training — DDP (v1.0)

v1.0 adds single-machine, multi-process Distributed Data Parallel (DDP) training
via PyTorch's `torchrun` launcher. Each process gets its own copy of the model;
gradients are synchronized automatically across processes after each accumulation
window.

### Quick start

Run a 2-process DDP smoke test with the provided Makefile target:

```bash
make train-ddp
```

This is equivalent to:

```bash
torchrun --nproc_per_node=2 -m llmtrain train --config configs/presets/ddp_smoke.yaml
```

### How it works

1. Set `ddp.enabled: true` in your config YAML (see `configs/presets/ddp_smoke.yaml`
   for a minimal example).
2. Launch with `torchrun` (or `python -m torch.distributed.run`). `torchrun` sets
   the `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR`, and `MASTER_PORT`
   environment variables automatically.
3. The CLI calls `setup_ddp()` which initializes the process group and returns a
   `DDPState`. The model is wrapped in `DistributedDataParallel` and gradient
   synchronization is deferred until the final micro-batch of each accumulation
   window via `model.no_sync()`.
4. Only rank 0 writes checkpoints, config copies, metadata, MLflow logs, and the
   JSON summary. All ranks log to stdout for debugging visibility.
5. On exit, `teardown_ddp()` destroys the process group.

### Limitations

- **gloo backend only (CPU)**: The current implementation uses the `gloo` backend
  for CPU-based training. GPU/NCCL support is not yet wired.
- **Single machine**: `torchrun --nproc_per_node=N` launches N processes on one
  host. Multi-node DDP is planned for v1.1 (Kubernetes).

### Slow integration test

The DDP integration test spawns a real 2-process `torchrun` run and is marked
`@pytest.mark.slow`:

```bash
pytest -m slow -k test_ddp_two_process_torchrun
```

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
  - `distributed/`: DDP setup/teardown utilities and `DDPState`
  - `models/`: GPT + dummy model adapters
  - `data/`: HF real-text + dummy data modules
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


