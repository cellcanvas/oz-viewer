"""Shared pytest fixtures for oz-viewer tests."""

from __future__ import annotations

import functools
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
    moment on any thread -- including inside zarr's I/O thread while a later
    test writes an example store, which then deadlocks.  Mirrors cellier's own
    test teardown.
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
    if widgets_module is not None:
        app = widgets_module.QApplication.instance()
        if app is not None:
            app.processEvents()


@pytest.fixture
def write_demo_ome(tmp_path: Path) -> Callable[[Literal["image", "plate"]], Path]:
    """Return a factory that writes demo OME-Zarr stores to tmp_path.

    Skips the test if zarr or ome-zarr are not installed.
    """
    try:
        from yaozarrs._demo_data import write_ome_image, write_ome_plate
    except ImportError:
        pytest.skip("zarr and ome-zarr are required for demo data fixtures")

    def _factory(store_type: Literal["image", "plate"] = "image") -> Path:
        if store_type == "image":
            path = tmp_path / "demo_image.zarr"
            write_ome_image(path)
        elif store_type == "plate":
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
