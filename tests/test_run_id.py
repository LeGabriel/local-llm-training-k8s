from __future__ import annotations

import subprocess
from datetime import datetime, tzinfo
from pathlib import Path

import pytest
from _pytest.monkeypatch import MonkeyPatch

import llmtrain.utils.run_id as run_id


class _FixedDatetime:
    @staticmethod
    def now(tz: tzinfo | None = None) -> datetime:
        dt = datetime(2024, 1, 2, 3, 4, 5)
        return dt if tz is None else dt.replace(tzinfo=tz)


def test_slugify_run_name_basic() -> None:
    assert run_id.slugify_run_name("My Run!!") == "my_run"
    assert run_id.slugify_run_name("  ---  ") == "run"


def test_slugify_run_name_truncates() -> None:
    long_name = "a" * 100
    assert run_id.slugify_run_name(long_name) == "a" * 40


def test_slugify_run_name_normalizes_separators() -> None:
    assert run_id.slugify_run_name("  Foo---Bar___Baz  ") == "foo_bar_baz"


def test_generate_run_id_format(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(run_id, "datetime", _FixedDatetime)
    monkeypatch.setattr(run_id, "_get_short_git_sha", lambda: "abc123")

    run_id_value = run_id.generate_run_id("My Run")
    assert run_id_value == "20240102_030405_abc123_my_run"


def test_get_short_git_sha_fallback(monkeypatch: MonkeyPatch) -> None:
    def _raise(*_args: object, **_kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, ["git", "rev-parse"])

    monkeypatch.setattr(run_id.subprocess, "run", _raise)
    assert run_id._get_short_git_sha() == "nogit"


def test_get_short_git_sha_empty_stdout(monkeypatch: MonkeyPatch) -> None:
    class _Result:
        stdout = ""

    monkeypatch.setattr(run_id.subprocess, "run", lambda *_args, **_kwargs: _Result())
    assert run_id._get_short_git_sha() == "nogit"


def test_generate_run_id_collision_suffix(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(run_id, "datetime", _FixedDatetime)
    monkeypatch.setattr(run_id, "_get_short_git_sha", lambda: "abc123")

    base_run_id = run_id.generate_run_id("My Run")
    (tmp_path / base_run_id).mkdir(parents=True)

    resolved = run_id.generate_run_id("My Run", root_dir=tmp_path)
    assert resolved == f"{base_run_id}__01"


def test_generate_run_id_collision_suffix_increments(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(run_id, "datetime", _FixedDatetime)
    monkeypatch.setattr(run_id, "_get_short_git_sha", lambda: "abc123")

    base_run_id = run_id.generate_run_id("My Run")
    (tmp_path / base_run_id).mkdir(parents=True)
    (tmp_path / f"{base_run_id}__01").mkdir(parents=True)

    resolved = run_id.generate_run_id("My Run", root_dir=tmp_path)
    assert resolved == f"{base_run_id}__02"


def test_generate_run_id_collision_limit(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(run_id, "datetime", _FixedDatetime)
    monkeypatch.setattr(run_id, "_get_short_git_sha", lambda: "abc123")

    base_run_id = run_id.generate_run_id("My Run")
    (tmp_path / base_run_id).mkdir(parents=True)
    for suffix in range(1, 100):
        (tmp_path / f"{base_run_id}__{suffix:02d}").mkdir(parents=True)

    with pytest.raises(RuntimeError):
        run_id.generate_run_id("My Run", root_dir=tmp_path)
