from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from llmtrain import __version__
from llmtrain.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the llmtrain CLI."""
    parser = argparse.ArgumentParser(
        prog="llmtrain",
        description=(
            "Utilities for orchestrating distributed LLMtraining on local Kubernetes clusters."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show the llmtrain version and exit.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase log verbosity (repeatable).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entrypoint for ``python -m llmtrain``."""
    parser = build_parser()
    args = parser.parse_args(argv)

    level = logging.INFO if args.verbose == 0 else logging.DEBUG
    logger = configure_logging(level=level)
    logger.debug("llmtrain CLI invoked", extra={"argv": list(argv) if argv else []})
    return 0
