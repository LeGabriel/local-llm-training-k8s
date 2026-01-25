from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence

import yaml

from llmtrain import __version__
from llmtrain.config.loader import ConfigLoadError, load_and_validate_config
from llmtrain.utils.logging import configure_logging
from llmtrain.utils.metadata import generate_meta, write_meta_json
from llmtrain.utils.run_dir import create_run_directory, write_resolved_config
from llmtrain.utils.run_id import generate_run_id
from llmtrain.utils.summary import format_run_summary


def _emit_config_error(error: ConfigLoadError, *, json_output: bool) -> None:
    if json_output:
        payload = {
            "status": "error",
            "message": error.message,
            "details": error.details,
            "errors": error.errors,
        }
        print(json.dumps(payload, indent=2), file=sys.stderr)
        return

    print(f"Config error: {error.message}", file=sys.stderr)
    if error.details:
        print(error.details, file=sys.stderr)


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

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file.",
    )
    common.add_argument(
        "--run-id",
        help="Override the run ID for the training command.",
    )
    common.add_argument(
        "--dry-run",
        action="store_true",
        help="Stop after run setup (no training execution).",
    )
    common.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON output.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("train", parents=[common], help="Prepare a training run.")
    subparsers.add_parser("validate", parents=[common], help="Validate a config file.")
    subparsers.add_parser(
        "print-config",
        parents=[common],
        help="Print the resolved config with defaults.",
    )

    return parser


def _handle_validate(args: argparse.Namespace) -> int:
    try:
        load_and_validate_config(args.config)
    except ConfigLoadError as exc:
        _emit_config_error(exc, json_output=args.json)
        return 2

    if args.json:
        print(json.dumps({"status": "ok"}, indent=2))
    else:
        print("Config validation succeeded.")
    return 0


def _handle_print_config(args: argparse.Namespace) -> int:
    try:
        config, _, _ = load_and_validate_config(args.config)
    except ConfigLoadError as exc:
        _emit_config_error(exc, json_output=args.json)
        return 2

    payload = config.model_dump()
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(yaml.safe_dump(payload, sort_keys=False), end="")
    return 0


def _handle_train(args: argparse.Namespace) -> int:
    try:
        config, raw_path, resolved_path = load_and_validate_config(args.config)
    except ConfigLoadError as exc:
        _emit_config_error(exc, json_output=args.json)
        return 2

    root_dir = config.output.root_dir
    run_id = args.run_id or config.output.run_id or generate_run_id(config.run.name, root_dir)
    run_dir = create_run_directory(root_dir, run_id)

    if config.output.save_config_copy:
        write_resolved_config(run_dir, config)
    if config.output.save_meta_json:
        meta = generate_meta(
            run_id=run_id,
            run_name=config.run.name,
            config_path=raw_path,
            resolved_config_path=str(resolved_path),
        )
        write_meta_json(run_dir, meta)

    summary = format_run_summary(
        config=config,
        run_id=run_id,
        run_dir=run_dir,
        json_output=args.json,
    )

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(summary)

    if args.dry_run:
        return 0

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entrypoint for ``python -m llmtrain``."""
    parser = build_parser()
    args = parser.parse_args(argv)

    level = logging.INFO if args.verbose == 0 else logging.DEBUG
    logger = configure_logging(level=level)
    logger.debug("llmtrain CLI invoked", extra={"argv": list(argv) if argv else []})
    if args.command == "validate":
        return _handle_validate(args)
    if args.command == "print-config":
        return _handle_print_config(args)
    if args.command == "train":
        return _handle_train(args)

    logger.error("Unknown command", extra={"command": args.command})
    return 1
