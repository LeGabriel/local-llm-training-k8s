PYTHON ?= python
export PYTHONPATH := $(PWD)/src

.PHONY: format lint test

format:
	$(PYTHON) -m ruff format src tests

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m mypy src

test:
	$(PYTHON) -m pytest

