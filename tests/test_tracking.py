from __future__ import annotations

from pathlib import Path

from llmtrain.tracking import NullTracker


def test_null_tracker_methods_are_callable_without_error() -> None:
    tracker = NullTracker()

    tracker.start_run(run_name="smoke-run")
    tracker.log_params({"batch_size": 8, "use_amp": False})
    tracker.log_metrics({"train/loss": 1.23, "train/lr": 1e-3}, step=1)
    tracker.log_artifact(Path("config.yaml"), artifact_path="configs")
    tracker.end_run()
