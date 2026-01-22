from __future__ import annotations

import subprocess
import sys

import llmtrain


def test_imports() -> None:
    assert llmtrain.__version__


def test_cli_help_exits_zero() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "llmtrain", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()
