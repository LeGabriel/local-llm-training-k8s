from __future__ import annotations

import json
import logging

import pytest

from llmtrain.utils.logging import configure_logging


def test_configure_logging_emits_json(capsys: pytest.CaptureFixture[str]) -> None:
    logger = configure_logging(level=logging.INFO, name="llmtrain.test")

    logger.info("hello")

    captured = capsys.readouterr().out.strip()
    payload = json.loads(captured)

    assert payload["message"] == "hello"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "llmtrain.test"
    assert "timestamp" in payload
