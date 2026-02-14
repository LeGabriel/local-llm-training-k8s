.PHONY: format lint test train-ddp train-gpt-ddp mlflow

format:
	uv run ruff format src tests

lint:
	uv run ruff check src tests
	uv run mypy --config-file=pyproject.toml src/llmtrain tests

test:
	uv run pytest

train-ddp:
	uv run torchrun --nproc_per_node=2 -m llmtrain train --config configs/presets/ddp_smoke.yaml

train-gpt-ddp:
	uv run torchrun --nproc_per_node=2 -m llmtrain train --config configs/presets/gpt_wikitext_ddp.yaml

mlflow:
	uv run mlflow ui --backend-store-uri sqlite:///./mlflow.db
