from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO


class JsonFormatter(logging.Formatter):
    """Render log records as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003 - keep stdlib name
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


def _build_formatter(*, json_output: bool) -> logging.Formatter:
    if json_output:
        return JsonFormatter()
    return logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")


def configure_logging(
    *,
    level: int = logging.INFO,
    name: str = "llmtrain",
    json_output: bool = True,
    log_to_file: bool = False,
    file_name: str = "train.log",
    stream: TextIO | None = None,
) -> logging.Logger:
    """Configure stdout (and optional file) logging for llmtrain."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = _build_formatter(json_output=json_output)
    target_stream = sys.stdout if stream is None else stream

    stdout_handler = next(
        (
            handler
            for handler in logger.handlers
            if isinstance(handler, logging.StreamHandler)
            and getattr(handler, "stream", None) is target_stream
        ),
        None,
    )
    if stdout_handler is None:
        stdout_handler = logging.StreamHandler(target_stream)
        logger.addHandler(stdout_handler)
    stdout_handler.setFormatter(formatter)

    if log_to_file:
        file_path = Path(file_name)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = next(
            (
                handler
                for handler in logger.handlers
                if isinstance(handler, logging.FileHandler)
                and Path(handler.baseFilename) == file_path
            ),
            None,
        )
        if file_handler is None:
            for handler in list(logger.handlers):
                if isinstance(handler, logging.FileHandler):
                    logger.removeHandler(handler)
                    handler.close()
            file_handler = logging.FileHandler(file_path, encoding="utf-8")
            logger.addHandler(file_handler)
        file_handler.setFormatter(formatter)
    else:
        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)
                handler.close()

    logger.propagate = False
    return logger
