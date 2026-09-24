"""Shared pytest fixtures for oz-viewer tests."""

from __future__ import annotations

import functools
import gc
import sys
import threading
import time
import weakref
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import TYPE_CHECKING, Literal

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from pathlib import Path


@pytest.fixture(autouse=True)
def _close_cellier_controllers(monkeypatch):
    """Close every ``CellierController`` a test creates.

    A controller owns render canvases and GPU resources that the GUI backend
    holds, not Python refcounting.  A test that only closes its window leaves
    them to the garbage collector, which may run their teardown at any later
    moment -- including while a later test writes an example store with
    zarr's I/O threads (and, through ome-zarr-py, a dask thread pool) busy,
    which has deadlocked and segfaulted.  Mirrors cellier's own test
    teardown, and then collects here so that nothing the test left behind is
    finalized at such a moment.
    """
    from cellier.controller import CellierController

    created: list[weakref.ref] = []
    original_init = CellierController.__init__

    def _tracking_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        created.append(weakref.ref(self))

    monkeypatch.setattr(CellierController, "__init__", _tracking_init)

    yield

    for ref in created:
        controller = ref()
        if controller is None:
            continue
        try:
            controller.close()
        except Exception:
            # Teardown must not turn a passing test into an error.
            pass

    # Qt deletes a closed widget only when the event loop next runs.
    widgets_module = sys.modules.get("PySide6.QtWidgets")
    app = None
    if widgets_module is not None:
        app = widgets_module.QApplication.instance()
        if app is not None:
            app.processEvents()

    # Finalize the test's leftovers here, at a quiet point, rather than
    # whenever the collector next runs -- which can be while the next test
    # writes its example store with zarr's I/O threads busy.
    gc.collect()
    if app is not None:
        app.processEvents()


def _write_demo_image(path: Path) -> None:
    """Write a small demo OME-Zarr 0.5 image with plain zarr.

    ``c, z, y, x`` = ``(1, 1, 64, 64)`` uint16 random data, two levels with
    y and x halved, chunks ``(1, 1, 32, 32)`` -- the layout yaozarrs' demo
    writer produces.  Written directly rather than through that writer, which
    builds the pyramid with ome-zarr-py and so runs a dask thread pool inside
    the test process (see ``_close_cellier_controllers``).
    """
    import numpy as np
    import zarr

    rng = np.random.default_rng(42)
    level_0 = rng.integers(0, 1000, size=(1, 1, 64, 64), dtype=np.uint16)
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    datasets = []
    for level in range(2):
        factor = 2**level
        data = level_0[..., ::factor, ::factor]
        array = root.create_array(
            f"s{level}",
            shape=data.shape,
            chunks=(1, 1, 32, 32),
            dtype=data.dtype,
            dimension_names=("c", "z", "y", "x"),
        )
        array[:] = data
        datasets.append(
            {
                "path": f"s{level}",
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, 1.0, factor, factor]}
                ],
            }
        )
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "name": "demo",
                "axes": [
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space"},
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ],
                "datasets": datasets,
            }
        ],
    }


@pytest.fixture
def write_demo_ome(tmp_path: Path) -> Callable[[Literal["image", "plate"]], Path]:
    """Return a factory that writes demo OME-Zarr stores to tmp_path.

    The image is written with plain zarr (:func:`_write_demo_image`).  The
    plate still comes from yaozarrs' demo writer (ome-zarr-py, and so dask);
    the test is skipped if that is not installed.
    """

    def _factory(store_type: Literal["image", "plate"] = "image") -> Path:
        if store_type == "image":
            path = tmp_path / "demo_image.zarr"
            _write_demo_image(path)
        elif store_type == "plate":
            try:
                from yaozarrs._demo_data import write_ome_plate
            except ImportError:
                pytest.skip("ome-zarr is required for the demo plate fixture")
            path = tmp_path / "demo_plate.zarr"
            write_ome_plate(path)
        else:
            raise ValueError(f"Unknown store_type: {store_type!r}")
        return path

    return _factory


@pytest.fixture
def slow_http_store(tmp_path: Path) -> Generator[Callable[[Path], str], None, None]:
    """Serve tmp_path over HTTP with a per-request delay.

    Starts a local HTTP server that sleeps 100 ms before each response, making
    a ``--timeout 0`` ping reliably time out without any monkeypatching.

    Yields a callable that converts a local path under tmp_path to its HTTP URL.
    """

    class _SlowHandler(SimpleHTTPRequestHandler):
        def do_GET(self) -> None:
            time.sleep(0.1)
            super().do_GET()

        def log_message(self, *args: object) -> None:
            pass

    handler = functools.partial(_SlowHandler, directory=str(tmp_path))
    server = HTTPServer(("127.0.0.1", 0), handler)
    port: int = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def to_url(local_path: Path) -> str:
        return f"http://127.0.0.1:{port}/{local_path.relative_to(tmp_path)}"

    yield to_url

    server.shutdown()
