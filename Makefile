.PHONY: format lint test

format:
	uv run ruff format src tests

lint:
	uv run ruff check src tests
	uv run mypy --config-file=pyproject.toml src/llmtrain tests

test:
	uv run pytest
