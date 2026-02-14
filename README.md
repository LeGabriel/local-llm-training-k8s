# local-llm-training-k8s
Production-style distributed training framework for decoder-only transformers, running correctness-first CPU training on a local Kubernetes cluster via `kind` and IndexedJob.

## Goal
- Build a realistic, test-driven training stack for decoder-only GPT-style models.
- Keep it modular: `ModelAdapter`, `DataModule`, and `Trainer` are contract-based.
- Reach "one command" local K8s training with checkpoints, metrics, and reproducible runs.

## What exists today (v1.1)
- Real decoder-only GPT model adapter (`gpt`) with causal self-attention.
- Fast smoke-path model adapter (`dummy_gpt`) remains available.
- GPT tokenizer wiring via `tiktoken` (`gpt2` encoding).
- Real text data pipeline via Hugging Face datasets (`hf_text`).
- Real single-process training loop with gradient accumulation and LR schedule.
- **Distributed Data Parallel (DDP)** multi-process training via `torchrun`.
- **Kubernetes orchestration** via `kind` cluster and IndexedJob with multi-pod DDP.
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
- **v1.0**: Distributed Data Parallel on a single machine.
- **v1.1**: Kubernetes `kind` + IndexedJob orchestration. **(current)**
- **v1.2**: K8s runs persistence - checkpoints, run logs, and MLflow tracking back to the local host machine.

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
  host. For multi-pod DDP on Kubernetes, see the v1.1 section below.

### Slow integration test

The DDP integration test spawns a real 2-process `torchrun` run and is marked
`@pytest.mark.slow`:

```bash
pytest -m slow -k test_ddp_two_process_torchrun
```

## Kubernetes Training — IndexedJob (v1.1)

v1.1 brings Kubernetes into the picture: using `kind` (Kubernetes-in-Docker), the
framework runs multi-pod DDP training on a local machine, proving the K8s
deployment model. The core application code requires **no changes** -- `setup_ddp()`
already reads `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR`, and `MASTER_PORT`
from environment variables; the K8s manifests just set them differently.

### Prerequisites

Install the following tools (in addition to the standard Python/`uv` toolchain):

| Tool | Purpose |
|------|---------|
| [Docker](https://docs.docker.com/get-docker/) | Build the training container image |
| [kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation) | Create a local Kubernetes cluster in Docker |
| [kubectl](https://kubernetes.io/docs/tasks/tools/) | Interact with the Kubernetes cluster |

### Quick start

Run the full pipeline in one shot:

```bash
make k8s-cluster k8s-build k8s-train k8s-logs
```

Or use the all-in-one convenience target:

```bash
make k8s-full
```

Individual targets:

```bash
make k8s-cluster          # create kind cluster
make k8s-build            # build Docker image and load into kind
make k8s-train            # apply manifests and wait for job completion
make k8s-logs             # fetch logs from all training pods
make k8s-clean            # delete K8s resources (job, service, configmap, rbac)
make k8s-cluster-delete   # tear down the kind cluster entirely
```

### How it works — IndexedJob pattern

The training workload is deployed as a Kubernetes
[IndexedJob](https://kubernetes.io/docs/concepts/workloads/controllers/job/#completion-mode)
with `completions: 2` and `parallelism: 2`. Each pod receives a unique
`JOB_COMPLETION_INDEX` (0, 1, ...) from the IndexedJob controller.

An entrypoint shell script (`k8s/entrypoint.sh`) bootstraps the DDP environment:

1. Sets `RANK` and `LOCAL_RANK` from `JOB_COMPLETION_INDEX`.
2. `WORLD_SIZE` and `MASTER_PORT` are static env vars defined in the Job spec.
3. Resolves `MASTER_ADDR` dynamically:
   - **Rank 0** uses its own pod IP (injected via the Kubernetes downward API as
     `POD_IP`).
   - **Non-zero ranks** query the Kubernetes API (using in-cluster ServiceAccount
     credentials) to find the pod with completion index 0 and extract its `podIP`.
     Retries with a short backoff until the rank-0 pod is available.
4. Execs into `python -m llmtrain train --config /config/train.yaml`.

The application code is completely unaware of Kubernetes -- it sees the same
environment variables that `torchrun` would provide.

### K8s manifest overview

| File | Purpose |
|------|---------|
| `k8s/Dockerfile` | Container image: `python:3.11-slim` with the package + `curl`/`jq` |
| `k8s/kind-config.yaml` | Single control-plane node cluster spec |
| `k8s/rbac.yaml` | ServiceAccount + Role + RoleBinding (get/list pods for master discovery) |
| `k8s/configmap.yaml` | Training config (`train.yaml`) embedded as a ConfigMap |
| `k8s/service.yaml` | Headless Service (`clusterIP: None`) for DNS-based peer discovery |
| `k8s/job.yaml` | IndexedJob manifest (`parallelism=2`, `completions=2`) |
| `k8s/entrypoint.sh` | Shell script: derives DDP env vars and launches training |
| `k8s/test_e2e.sh` | End-to-end test script (not run by `make test`) |

### Artifacts and output

Only rank 0 writes checkpoints, config copies, metadata, and the JSON summary
(unchanged from v1.0). Artifacts stay inside the rank-0 pod container under
`/app/runs/` and can be retrieved via:

```bash
kubectl cp <rank-0-pod>:/app/runs ./runs
```

Shared persistent volumes (PVCs) are out of scope for v1.1.

### Debugging

```bash
kubectl get pods -l app=llmtrain              # check pod status
kubectl describe pod <pod-name>               # inspect events, env, mounts
kubectl logs <pod-name>                       # read a specific pod's output
kubectl logs -l app=llmtrain --prefix         # tail all training pods
kubectl get endpoints llmtrain-headless       # verify headless Service endpoints
```

Common failure modes:
- **MASTER_ADDR resolution failure**: RBAC misconfiguration or rank-0 pod not yet running. Check `kubectl logs` for the entrypoint retry messages.
- **WORLD_SIZE mismatch**: The Job's `completions`/`parallelism` must match the `WORLD_SIZE` env var.
- **RBAC 403 Forbidden**: The ServiceAccount lacks get/list permissions on pods.
- **DNS issues**: Headless Service label selector (`app: llmtrain`) doesn't match pod labels. Verify with `kubectl get endpoints`.

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
- `k8s/`: Kubernetes manifests and tooling
  - `Dockerfile`: training container image
  - `kind-config.yaml`: local cluster spec
  - `rbac.yaml`, `configmap.yaml`, `service.yaml`, `job.yaml`: K8s resources
  - `entrypoint.sh`: DDP env var bootstrap for IndexedJob pods
  - `test_e2e.sh`: end-to-end K8s test script
- `tests/`: pytest suite (includes CLI smoke)
- `pyproject.toml`: project + tool config (ruff, mypy, pytest)
- `.pre-commit-config.yaml`: ruff/mypy hooks
- `Makefile`: `format`, `lint`, `test`, `k8s-*` targets

## Tooling
- `ruff` for lint/format, `mypy` for types, `pytest` for tests, `pre-commit` for hooks.
- `uv` manages the environment/lockfile.

## Development process

This project is built with AI-assisted developer tooling (e.g., Cursor / LLM copilots) as a productivity multiplier for iteration, refactoring, and documentation.

All architecture, trade-offs, and final implementations are reviewed and owned by me. The codebase is validated via automated linting, type-checking, and tests, with a commit history intended to remain readable and reviewable.


