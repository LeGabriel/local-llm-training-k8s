from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

import yaml

from llmtrain import __version__
from llmtrain.config.loader import ConfigLoadError, load_and_validate_config
from llmtrain.config.schemas import LoggingConfig, RunConfig
from llmtrain.registry import initialize_registries
from llmtrain.registry.data import RegistryError as DataRegistryError
from llmtrain.registry.data import get_data_module
from llmtrain.registry.models import RegistryError as ModelRegistryError
from llmtrain.registry.models import get_model_adapter
from llmtrain.tracking import MLflowTracker, NullTracker, Tracker
from llmtrain.training import Trainer
from llmtrain.training.dry_run import run_dry_run
from llmtrain.utils.logging import configure_logging
from llmtrain.utils.metadata import generate_meta, write_meta_json
from llmtrain.utils.run_dir import create_run_directory, write_resolved_config
from llmtrain.utils.run_id import generate_run_id
from llmtrain.utils.summary import format_run_summary

LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


def _configure_logger(
    config_logging: LoggingConfig,
    *,
    verbose: int,
    log_dir: Path | None = None,
    stream: TextIO | None = None,
) -> logging.Logger:
    level = LOG_LEVELS.get(config_logging.level, logging.INFO)
    if verbose > 0:
        level = logging.DEBUG

    file_name = config_logging.file_name
    if log_dir is not None:
        file_name = str(log_dir / file_name)

    return configure_logging(
        level=level,
        json_output=config_logging.json_output,
        log_to_file=config_logging.log_to_file,
        file_name=file_name,
        stream=stream,
    )


def _emit_config_error(error: ConfigLoadError, *, json_output: bool) -> None:
    if json_output:
        payload = {
            "status": "error",
            "message": error.message,
            "details": error.details,
            "errors": error.errors,
        }
        print(json.dumps(payload, indent=2, default=str), file=sys.stderr)
        return

    print(f"Config error: {error.message}", file=sys.stderr)
    if error.details:
        print(error.details, file=sys.stderr)


def _create_tracker(config: RunConfig, logger: logging.Logger) -> Tracker:
    mlflow_cfg = config.mlflow
    if not mlflow_cfg.enabled:
        return NullTracker()

    try:
        return MLflowTracker(
            tracking_uri=mlflow_cfg.tracking_uri,
            experiment=mlflow_cfg.experiment,
            run_name=mlflow_cfg.run_name,
        )
    except RuntimeError as exc:
        logger.warning("MLflow unavailable; falling back to NullTracker: %s", exc)
        return NullTracker()


def _log_run_artifacts(tracker: Tracker, run_dir: Path) -> None:
    for artifact_name in ("config.yaml", "meta.json"):
        artifact_path = run_dir / artifact_name
        if artifact_path.exists():
            tracker.log_artifact(artifact_path, artifact_path="artifacts")


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
    train_parser = subparsers.add_parser("train", parents=[common], help="Prepare a training run.")
    train_parser.add_argument(
        "--resume",
        default=None,
        help="Resume from a run_id or checkpoint path.",
    )
    subparsers.add_parser("validate", parents=[common], help="Validate a config file.")
    subparsers.add_parser(
        "print-config",
        parents=[common],
        help="Print the resolved config with defaults.",
    )

    return parser


def _handle_validate(args: argparse.Namespace) -> int:
    try:
        config, _, _ = load_and_validate_config(args.config)
    except ConfigLoadError as exc:
        _emit_config_error(exc, json_output=args.json)
        return 2
    _configure_logger(
        config.logging,
        verbose=args.verbose,
        stream=sys.stderr if args.json else None,
    )

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
    _configure_logger(
        config.logging,
        verbose=args.verbose,
        stream=sys.stderr if args.json else None,
    )

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
    logger = _configure_logger(
        config.logging,
        verbose=args.verbose,
        log_dir=run_dir / "logs",
        stream=sys.stderr if args.json else None,
    )

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

    initialize_registries()
    try:
        get_model_adapter(config.model.name)
        get_data_module(config.data.name)
    except (ModelRegistryError, DataRegistryError) as exc:
        _emit_config_error(ConfigLoadError(str(exc)), json_output=args.json)
        return 2

    tracker = _create_tracker(config, logger)
    try:
        tracker.start_run(run_name=config.mlflow.run_name or run_id)

        if args.dry_run:
            # --- Dry-run path (forward-only sanity check) ---
            dry_run_logger = logger
            if args.json:
                dry_run_logger = logging.getLogger("llmtrain.dry_run")
                dry_run_logger.setLevel(logger.level)
                dry_run_logger.handlers.clear()
                handler = logging.StreamHandler(sys.stderr)
                handler.setFormatter(logger.handlers[0].formatter if logger.handlers else None)
                dry_run_logger.addHandler(handler)
                dry_run_logger.propagate = False

            try:
                dry_run_result = run_dry_run(config, logger=dry_run_logger)
            except Exception as exc:
                print(f"Dry-run failed: {exc}", file=sys.stderr)
                return 1

            summary = format_run_summary(
                config=config,
                run_id=run_id,
                run_dir=run_dir,
                json_output=args.json,
                resolved_model_adapter=dry_run_result.resolved_model_adapter,
                resolved_data_module=dry_run_result.resolved_data_module,
                dry_run_steps_executed=dry_run_result.steps_executed,
            )
        else:
            # --- Full training path ---
            if args.json:
                train_logger = logging.getLogger("llmtrain.training.trainer")
                train_logger.setLevel(logger.level)
                train_logger.handlers.clear()
                handler = logging.StreamHandler(sys.stderr)
                handler.setFormatter(logger.handlers[0].formatter if logger.handlers else None)
                train_logger.addHandler(handler)
                train_logger.propagate = False

            resume_from = getattr(args, "resume", None)
            if resume_from is not None:
                logger.info("Resuming from: %s", resume_from)

            try:
                trainer = Trainer(config, run_dir=run_dir, tracker=tracker)
                train_result = trainer.fit(resume_from=resume_from)
            except Exception as exc:
                print(f"Training failed: {exc}", file=sys.stderr)
                return 1

            summary = format_run_summary(
                config=config,
                run_id=run_id,
                run_dir=run_dir,
                json_output=args.json,
                train_result=train_result,
                resumed_from=resume_from,
            )

        _log_run_artifacts(tracker, run_dir)

        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print(summary)
    finally:
        tracker.end_run()

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entrypoint for ``python -m llmtrain``."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate":
        return _handle_validate(args)
    if args.command == "print-config":
        return _handle_print_config(args)
    if args.command == "train":
        return _handle_train(args)

    print(f"Unknown command: {args.command}", file=sys.stderr)
    return 1
