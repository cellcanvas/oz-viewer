"""CLI entry point for oz-viewer."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from oz_viewer._display import (
    make_console,
    make_ping_progress,
    print_download_complete,
    print_error_panel,
    print_metadata_panel,
    print_ping_header,
    print_ping_results,
    print_success_panel,
)
from oz_viewer._perf import (
    StartupPerfTracer,
    configure_perf_logging,
    perf_enabled_from_env,
)

app = typer.Typer(
    name="oz-viewer",
    help="Validate and inspect OME-Zarr stores.",
    no_args_is_help=True,
)


@app.command()
def validate(
    path: Annotated[str, typer.Argument(help="Path or URI to the OME-Zarr store.")],
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Pretty-print the full metadata model after successful validation.",
        ),
    ] = False,
) -> None:
    """Validate an OME-Zarr store."""
    from yaozarrs import validate_zarr_store
    from yaozarrs._storage import StorageValidationError

    console = make_console()
    try:
        group = validate_zarr_store(path)
    except StorageValidationError as e:
        print_error_panel(path, e, console)
        raise typer.Exit(code=1) from None
    except ImportError as e:
        print(str(e), file=sys.stderr)
        raise typer.Exit(code=2) from None
    except Exception as e:
        print(str(e), file=sys.stderr)
        raise typer.Exit(code=2) from None

    print_success_panel(path, group, console)
    if verbose:
        print_metadata_panel(group.ome_metadata(), console)


@app.command()
def ping(
    path: Annotated[str, typer.Argument(help="Path or URI to the OME-Zarr store.")],
    n_fetch: Annotated[
        int,
        typer.Option(
            "--n-fetch",
            help="Number of chunk fetches to average over.",
            min=1,
        ),
    ] = 5,
    timeout: Annotated[
        float,
        typer.Option(
            "--timeout",
            help="Per-fetch timeout in seconds.",
            min=0.0,
        ),
    ] = 10.0,
) -> None:
    """Measure chunk fetch latency for an OME-Zarr store."""
    from yaozarrs import validate_zarr_store

    from oz_viewer._ping import build_chunk_info, run_fetches

    console = make_console()
    try:
        group = validate_zarr_store(path)
    except Exception as e:
        print(str(e), file=sys.stderr)
        raise typer.Exit(code=2) from None

    chunk_info = build_chunk_info(group, group.ome_metadata())
    print_ping_header(path, chunk_info, n_fetch, timeout, console)

    progress = make_ping_progress(console)
    with progress:
        task_id = progress.add_task("Fetching chunks…", total=n_fetch)
        result = run_fetches(chunk_info, n_fetch, timeout, progress, task_id)

    print_ping_results(chunk_info, result, console)


def _resolve_zarr_uri(path: str) -> str:
    """Convert a local path to a file:// URI; pass remote URIs through unchanged."""
    if "://" in path:
        return path
    local = Path(path)
    if not local.exists():
        typer.echo(f"Error: OME-Zarr store not found at '{local}'", err=True)
        raise typer.Exit(code=1)
    return f"file://{local.resolve()}"


@app.command()
def ortho(
    path: Annotated[
        str | None,
        typer.Argument(
            help="Path or URI to the OME-Zarr store (local path, s3://, gs://, https://).",
            show_default=False,
        ),
    ] = None,
    path_option: Annotated[
        str | None,
        typer.Option(
            "--path",
            help=(
                "Path or URI to the OME-Zarr store"
                " (alternative to positional argument)."
            ),
            show_default=False,
        ),
    ] = None,
    make_example: Annotated[
        bool,
        typer.Option(
            "--make-example",
            help="Create a synthetic anisotropic OME-Zarr and open it in the viewer.",
        ),
    ] = False,
    multichannel: Annotated[
        int | None,
        typer.Option(
            "--multichannel",
            help=(
                "Dimension index to treat as the channel axis, enabling multichannel"
                " mode. If omitted, single-channel mode is used."
            ),
            show_default=False,
        ),
    ] = None,
    infer_multiscale_translations: Annotated[
        bool,
        typer.Option(
            "--infer-multiscale-translations",
            help=(
                "Assume the coarser pyramid levels were downsampled centre-aligned"
                " (e.g. block averages) and add the half-voxel offsets that implies"
                " when the store declares no level translations. Wrong for"
                " pyramids made by striding. Ignored, with a warning, when the"
                " store declares translations."
            ),
        ),
    ] = False,
    theme: Annotated[
        str,
        typer.Option(
            "--theme",
            help=(
                "Theme name to apply. Run 'oz-viewer theme list' to see"
                " available themes."
            ),
        ),
    ] = "dark",
    perf_startup: Annotated[
        bool,
        typer.Option(
            "--perf-startup/--no-perf-startup",
            help="Enable startup performance logging diagnostics.",
        ),
    ] = False,
    perf_log_file: Annotated[
        Path | None,
        typer.Option(
            "--perf-log-file",
            help="Write startup performance logs to a file instead of stderr.",
            show_default=False,
        ),
    ] = None,
    perf_table: Annotated[
        bool,
        typer.Option(
            "--perf-table/--no-perf-table",
            help="Display startup timings as a Rich table at startup completion.",
        ),
    ] = False,
    perf_table_title: Annotated[
        str,
        typer.Option(
            "--perf-table-title",
            help="Title used for the startup performance Rich table.",
        ),
    ] = "Orthoviewer startup timings",
) -> None:
    """Open an OME-Zarr store in the 4-panel orthoviewer."""
    from oz_viewer.viewer import launch_orthoviewer

    perf_enabled = perf_startup or perf_enabled_from_env()
    configure_perf_logging(
        enabled=perf_enabled,
        log_file=str(perf_log_file) if perf_log_file is not None else None,
    )
    perf = StartupPerfTracer(
        enabled=perf_enabled,
        show_table=perf_table,
        table_title=perf_table_title,
    )
    perf.mark("cli.ortho.start", make_example=make_example, theme=theme)

    if make_example:
        from oz_viewer.data._blobs import make_example_zarr

        zarr_path = make_example_zarr()
        zarr_uri = f"file://{zarr_path}"
        perf.mark("cli.ortho.example_created", zarr_path=zarr_path)
    else:
        raw = path or path_option
        if raw is None:
            typer.echo(
                "Error: provide a path as a positional argument, via --path, "
                "or use --make-example.",
                err=True,
            )
            raise typer.Exit(code=1)
        if path is not None and path_option is not None:
            typer.echo(
                "Error: provide the path as a positional argument or --path, not both.",
                err=True,
            )
            raise typer.Exit(code=1)
        zarr_uri = _resolve_zarr_uri(raw)
        perf.mark("cli.ortho.uri_resolved", zarr_uri=zarr_uri)

    perf.mark("cli.ortho.launch")
    launch_orthoviewer(
        zarr_uri,
        channel_axis=multichannel,
        infer_multiscale_translations=infer_multiscale_translations,
        theme=theme,
        perf=perf,
    )


@app.command()
def view(
    path: Annotated[
        str | None,
        typer.Argument(
            help="Path or URI to the OME-Zarr store (local path, s3://, gs://, https://).",
            show_default=False,
        ),
    ] = None,
    path_option: Annotated[
        str | None,
        typer.Option(
            "--path",
            help=(
                "Path or URI to the OME-Zarr store"
                " (alternative to positional argument)."
            ),
            show_default=False,
        ),
    ] = None,
    make_example: Annotated[
        bool,
        typer.Option(
            "--make-example",
            help="Create a synthetic anisotropic OME-Zarr and open it in the viewer.",
        ),
    ] = False,
    multichannel: Annotated[
        int | None,
        typer.Option(
            "--multichannel",
            help=(
                "Dimension index to treat as the channel axis, enabling multichannel"
                " mode. If omitted, the channel axis is auto-detected from the"
                " OME-Zarr metadata when present."
            ),
            show_default=False,
        ),
    ] = None,
    infer_multiscale_translations: Annotated[
        bool,
        typer.Option(
            "--infer-multiscale-translations",
            help=(
                "Assume the coarser pyramid levels were downsampled centre-aligned"
                " (e.g. block averages) and add the half-voxel offsets that implies"
                " when the store declares no level translations. Wrong for"
                " pyramids made by striding. Ignored, with a warning, when the"
                " store declares translations."
            ),
        ),
    ] = False,
    theme: Annotated[
        str,
        typer.Option(
            "--theme",
            help=(
                "Theme name to apply. Run 'oz-viewer theme list' to see"
                " available themes."
            ),
        ),
    ] = "dark",
    perf_startup: Annotated[
        bool,
        typer.Option(
            "--perf-startup/--no-perf-startup",
            help="Enable startup performance logging diagnostics.",
        ),
    ] = False,
    perf_log_file: Annotated[
        Path | None,
        typer.Option(
            "--perf-log-file",
            help="Write startup performance logs to a file instead of stderr.",
            show_default=False,
        ),
    ] = None,
    perf_table: Annotated[
        bool,
        typer.Option(
            "--perf-table/--no-perf-table",
            help="Display startup timings as a Rich table at startup completion.",
        ),
    ] = False,
    perf_table_title: Annotated[
        str,
        typer.Option(
            "--perf-table-title",
            help="Title used for the startup performance Rich table.",
        ),
    ] = "Viewer startup timings",
) -> None:
    """Open an OME-Zarr store in the 2D / 3D toggle viewer."""
    from oz_viewer.viewer import launch_viewer

    perf_enabled = perf_startup or perf_enabled_from_env()
    configure_perf_logging(
        enabled=perf_enabled,
        log_file=str(perf_log_file) if perf_log_file is not None else None,
    )
    perf = StartupPerfTracer(
        enabled=perf_enabled,
        show_table=perf_table,
        table_title=perf_table_title,
    )
    perf.mark("cli.view.start", make_example=make_example, theme=theme)

    if make_example:
        from oz_viewer.data._blobs import make_example_zarr

        zarr_path = make_example_zarr()
        zarr_uri = f"file://{zarr_path}"
        perf.mark("cli.view.example_created", zarr_path=zarr_path)
    else:
        raw = path or path_option
        if raw is None:
            typer.echo(
                "Error: provide a path as a positional argument, via --path, "
                "or use --make-example.",
                err=True,
            )
            raise typer.Exit(code=1)
        if path is not None and path_option is not None:
            typer.echo(
                "Error: provide the path as a positional argument or --path, not both.",
                err=True,
            )
            raise typer.Exit(code=1)
        zarr_uri = _resolve_zarr_uri(raw)
        perf.mark("cli.view.uri_resolved", zarr_uri=zarr_uri)

    perf.mark("cli.view.launch")
    launch_viewer(
        zarr_uri,
        channel_axis=multichannel,
        infer_multiscale_translations=infer_multiscale_translations,
        theme=theme,
        perf=perf,
    )


@app.command(name="theme")
def theme_cmd(
    action: Annotated[
        str,
        typer.Argument(help="Action to perform. Currently supports: list"),
    ],
) -> None:
    """Manage oz-viewer themes."""
    if action == "list":
        from oz_viewer.theme import list_themes

        for name in list_themes():
            typer.echo(name)
    else:
        typer.echo(f"Unknown action {action!r}. Available actions: list", err=True)
        raise typer.Exit(code=1)


@app.command()
def download(
    url: Annotated[
        str,
        typer.Argument(
            help="Source URL: s3://<bucket>/path/data.zarr or https://host/path/data.zarr"
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Local output directory (default: basename of URL).",
            show_default=False,
        ),
    ] = None,
    concurrency: Annotated[
        int,
        typer.Option(
            "--concurrency",
            "-c",
            help="Maximum concurrent chunk transfers.",
            min=1,
        ),
    ] = 32,
    no_validate: Annotated[
        bool,
        typer.Option(
            "--no-validate",
            help="Skip OME-Zarr validation after download.",
        ),
    ] = False,
    anon: Annotated[
        bool,
        typer.Option(
            "--anon/--no-anon",
            help="Anonymous S3 access (default: --anon).",
        ),
    ] = True,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Remove and replace an existing output directory.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help=(
                "Enumerate keys and probe the first 3, but do not write anything. "
                "Useful for diagnosing URL or auth issues."
            ),
        ),
    ] = False,
) -> None:
    """Download an OME-Zarr store from S3 or HTTPS to a local directory."""
    import asyncio
    import shutil
    from urllib.parse import urlparse

    from yaozarrs import validate_zarr_store
    from yaozarrs._storage import StorageValidationError

    from oz_viewer._download import download as _download

    parsed = urlparse(url.rstrip("/"))
    scheme = parsed.scheme.lower()
    if scheme not in ("s3", "https", "http"):
        typer.echo(
            f"Error: unsupported URL scheme {scheme!r}. "
            "Use s3://, https://, or http://.",
            err=True,
        )
        raise typer.Exit(code=1)

    default_name = url.rstrip("/").split("/")[-1] or "downloaded.zarr"
    output_dir: Path = output if output is not None else Path(default_name)

    if output_dir.exists():
        if overwrite:
            typer.echo(f"Removing existing {output_dir} ...")
            shutil.rmtree(output_dir)
        else:
            typer.echo(
                f"Error: {output_dir} already exists. Pass --overwrite to replace it.",
                err=True,
            )
            raise typer.Exit(code=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    console = make_console()

    try:
        asyncio.run(_download(url, output_dir, concurrency, anon, dry_run, console))
    except (KeyboardInterrupt, asyncio.CancelledError):
        typer.echo("\nInterrupted — cleaning up partial output ...", err=True)
        shutil.rmtree(output_dir, ignore_errors=True)
        raise typer.Exit(code=1) from None
    except Exception as exc:
        typer.echo(f"\nFailed: {exc}", err=True)
        typer.echo("Cleaning up partial output ...", err=True)
        shutil.rmtree(output_dir, ignore_errors=True)
        raise typer.Exit(code=1) from None

    if dry_run:
        return

    print_download_complete(output_dir, console)

    if not no_validate:
        try:
            validate_zarr_store(str(output_dir))
            console.print("[bold green]✓ Valid OME-Zarr store.[/bold green]")
        except StorageValidationError as exc:
            print_error_panel(str(output_dir), exc, console)
            raise typer.Exit(code=1) from None
        except Exception as exc:
            typer.echo(f"⚠  Validation raised an unexpected error: {exc}", err=True)


if __name__ == "__main__":
    app()
