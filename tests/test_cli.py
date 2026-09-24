"""Tests for oz-viewer CLI commands."""

from __future__ import annotations

import shutil
import sys
import types
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

if TYPE_CHECKING:
    from pathlib import Path

from oz_viewer._cli import app
from oz_viewer._perf import (
    StartupPerfTracer,
    configure_perf_logging,
    perf_enabled_from_env,
)

runner = CliRunner()


# ---------------------------------------------------------------------------
# Stage 1 — validate command
# ---------------------------------------------------------------------------


def test_validate_valid_image(write_demo_ome):
    path = write_demo_ome("image")
    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == 0
    assert "✓ Valid OME-Zarr" in result.output


def test_validate_valid_plate(write_demo_ome):
    path = write_demo_ome("plate")
    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == 0
    assert "Plate" in result.output


def test_validate_verbose_image(write_demo_ome):
    path = write_demo_ome("image")
    result = runner.invoke(app, ["validate", "--verbose", str(path)])
    assert result.exit_code == 0
    assert "Image(" in result.output


def test_validate_verbose_plate(write_demo_ome):
    path = write_demo_ome("plate")
    result = runner.invoke(app, ["validate", "--verbose", str(path)])
    assert result.exit_code == 0
    assert "Plate(" in result.output
    assert "PlateDef(" in result.output


def _break_store(path: Path) -> None:
    """Remove the first level subdirectory to break a demo OME-Zarr store."""
    from yaozarrs import validate_zarr_store

    group = validate_zarr_store(str(path))
    meta = group.ome_metadata()
    level_path = meta.multiscales[0].datasets[-1].path
    shutil.rmtree(path / level_path)


def test_validate_invalid_store(write_demo_ome):
    path = write_demo_ome("image")
    _break_store(path)
    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == 1
    assert "✗ Validation failed" in result.output


def test_validate_nonexistent_store(tmp_path):
    path = tmp_path / "nonexistent.zarr"
    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == 2


def test_validate_verbose_does_not_run_on_failure(write_demo_ome):
    path = write_demo_ome("image")
    _break_store(path)
    result = runner.invoke(app, ["validate", "--verbose", str(path)])
    assert result.exit_code == 1
    assert "Metadata Model" not in result.output


def test_no_args_shows_help():
    result = runner.invoke(app, [])
    # typer exits with 2 for no-args help; just check output content
    assert "validate" in result.output


# ---------------------------------------------------------------------------
# Stage 2 — ping command
# ---------------------------------------------------------------------------


def test_ping_local_store(write_demo_ome):
    path = write_demo_ome("image")
    result = runner.invoke(app, ["ping", str(path)])
    assert result.exit_code == 0
    assert "✓ Complete" in result.output


def test_ping_default_n_fetch(write_demo_ome):
    path = write_demo_ome("image")
    result = runner.invoke(app, ["ping", str(path)])
    assert result.exit_code == 0
    assert "5" in result.output


def test_ping_timeout(write_demo_ome, slow_http_store):
    path = write_demo_ome("image")
    url = slow_http_store(path)
    result = runner.invoke(app, ["ping", url, "--timeout", "0", "--n-fetch", "1"])
    assert result.exit_code == 0
    assert "Timeouts" in result.output


def test_ping_invalid_store(tmp_path):
    path = tmp_path / "nonexistent.zarr"
    result = runner.invoke(app, ["ping", str(path)])
    assert result.exit_code == 2


def test_ortho_perf_startup_flag_enables_tracer(tmp_path, monkeypatch):
    zarr_path = tmp_path / "demo.zarr"
    zarr_path.mkdir()

    captured: dict[str, object] = {}

    def _fake_launch(zarr_uri, theme="dark", perf=None, channel_axis=None, **_):
        captured["zarr_uri"] = zarr_uri
        captured["theme"] = theme
        captured["perf"] = perf

    monkeypatch.setitem(
        sys.modules,
        "oz_viewer.viewer",
        types.SimpleNamespace(launch_orthoviewer=_fake_launch),
    )

    result = runner.invoke(app, ["ortho", str(zarr_path), "--perf-startup"])
    assert result.exit_code == 0
    assert captured["theme"] == "dark"
    assert captured["zarr_uri"] == f"file://{zarr_path.resolve()}"
    assert captured["perf"] is not None
    assert captured["perf"].enabled is True
    assert captured["perf"].show_table is False


def test_ortho_perf_table_flag_sets_tracer_show_table(tmp_path, monkeypatch):
    zarr_path = tmp_path / "demo.zarr"
    zarr_path.mkdir()

    captured: dict[str, object] = {}

    def _fake_launch(zarr_uri, theme="dark", perf=None, channel_axis=None, **_):
        captured["perf"] = perf

    monkeypatch.setitem(
        sys.modules,
        "oz_viewer.viewer",
        types.SimpleNamespace(launch_orthoviewer=_fake_launch),
    )

    result = runner.invoke(
        app,
        ["ortho", str(zarr_path), "--perf-startup", "--perf-table"],
    )
    assert result.exit_code == 0
    assert captured["perf"] is not None
    assert captured["perf"].show_table is True


def test_ortho_perf_table_title_sets_tracer_title(tmp_path, monkeypatch):
    zarr_path = tmp_path / "demo.zarr"
    zarr_path.mkdir()

    captured: dict[str, object] = {}

    def _fake_launch(zarr_uri, theme="dark", perf=None, channel_axis=None, **_):
        captured["perf"] = perf

    monkeypatch.setitem(
        sys.modules,
        "oz_viewer.viewer",
        types.SimpleNamespace(launch_orthoviewer=_fake_launch),
    )

    result = runner.invoke(
        app,
        [
            "ortho",
            str(zarr_path),
            "--perf-startup",
            "--perf-table",
            "--perf-table-title",
            "Custom Perf Title",
        ],
    )
    assert result.exit_code == 0
    assert captured["perf"] is not None
    assert captured["perf"].table_title == "Custom Perf Title"


def test_ortho_perf_env_enables_tracer(tmp_path, monkeypatch):
    zarr_path = tmp_path / "demo.zarr"
    zarr_path.mkdir()

    captured: dict[str, object] = {}

    def _fake_launch(zarr_uri, theme="dark", perf=None, channel_axis=None, **_):
        captured["perf"] = perf

    monkeypatch.setitem(
        sys.modules,
        "oz_viewer.viewer",
        types.SimpleNamespace(launch_orthoviewer=_fake_launch),
    )
    monkeypatch.setenv("OZ_VIEWER_PERF", "1")

    result = runner.invoke(app, ["ortho", str(zarr_path)])
    assert result.exit_code == 0
    assert captured["perf"] is not None
    assert captured["perf"].enabled is True


def test_perf_enabled_from_env_truthy(monkeypatch):
    monkeypatch.setenv("OZ_VIEWER_PERF", "true")
    assert perf_enabled_from_env() is True


def test_perf_enabled_from_env_falsey(monkeypatch):
    monkeypatch.setenv("OZ_VIEWER_PERF", "0")
    assert perf_enabled_from_env() is False


def test_startup_perf_tracer_writes_to_file(tmp_path):
    log_file = tmp_path / "perf.log"
    configure_perf_logging(enabled=True, log_file=str(log_file))

    tracer = StartupPerfTracer(enabled=True)
    tracer.mark("startup.step", stage="test")

    content = log_file.read_text()
    assert "[oz_viewer.perf]" in content
    assert "step=startup.step" in content
    assert "stage=test" in content


def test_startup_perf_tracer_rich_table(capsys):
    configure_perf_logging(enabled=True)
    tracer = StartupPerfTracer(enabled=True, show_table=True)
    tracer.mark("first")
    tracer.mark("second", phase="io")

    tracer.report_rich_table(title="Perf Test")

    stderr = capsys.readouterr().err
    assert "Perf Test" in stderr
    assert "Step (ms)" in stderr
    assert "Cumulative (ms)" in stderr
    assert "first" in stderr
    assert "second" in stderr


def test_startup_perf_tracer_rich_table_uses_default_title(capsys):
    configure_perf_logging(enabled=True)
    tracer = StartupPerfTracer(
        enabled=True,
        show_table=True,
        table_title="Tracer Default Title",
    )
    tracer.mark("only")

    tracer.report_rich_table()

    stderr = capsys.readouterr().err
    assert "Tracer Default Title" in stderr


# ---------------------------------------------------------------------------
# --infer-multiscale-translations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "launcher"),
    [("view", "launch_viewer"), ("ortho", "launch_orthoviewer")],
)
@pytest.mark.parametrize("flag", [True, False])
def test_infer_multiscale_translations_flag_reaches_launcher(
    tmp_path, monkeypatch, command, launcher, flag
):
    zarr_path = tmp_path / "demo.zarr"
    zarr_path.mkdir()
    captured: dict[str, object] = {}

    def _fake_launch(zarr_uri, **kwargs):
        captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "oz_viewer.viewer",
        types.SimpleNamespace(**{launcher: _fake_launch}),
    )

    args = [command, str(zarr_path)]
    if flag:
        args.append("--infer-multiscale-translations")
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert captured["infer_multiscale_translations"] is flag
