"""Single-panel 2D / 3D toggle viewer for OME-Zarr images.

Rebuilt on top of :mod:`cellier.convenience`, so the same builder runs under
both ``gui="qt"`` (desktop / CLI) and ``gui="anywidget"`` (Jupyter / marimo).

The image control (both single and composite modes, and the switch between
them) comes from cellier's cross-toolkit ``AppearanceControls`` dock and the
2D/3D toggle from the dims control embedded in the canvas view; this module
only supplies the OME-Zarr-specific geometry (see
:mod:`oz_viewer.viewer._geometry`) and the Qt-specific launch niceties
oz-viewer cares about (theme, fsspec loop, asyncio exception handling, startup
perf tracing).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Literal

from oz_viewer.viewer._geometry import _ViewerGeometry, extract_viewer_geometry
from oz_viewer.viewer._image import controller_render_config, image_visual_kwargs
from oz_viewer.viewer._utils import (
    _asyncio_exception_handler,
    _ensure_qt_app,
    _perf_mark,
    _sidecar_options,
)

if TYPE_CHECKING:
    from cellier.convenience import Viewer
    from cellier.convenience.layout._spec import Layout
    from cellier.data.image import OMEZarrImageDataStore

    from oz_viewer._perf import StartupPerfTracer

_INITIAL_LOD_BIAS = 1.5


# ---------------------------------------------------------------------------
# Layer 1: build the convenience Viewer (no launch, no Qt event loop)
# ---------------------------------------------------------------------------


def build_viewer(
    data_store: OMEZarrImageDataStore,
    geometry: _ViewerGeometry,
    *,
    gui: Literal["qt", "anywidget"] = "qt",
) -> Viewer:
    """Build a :class:`cellier.convenience.Viewer` for an OME-Zarr store.

    Adds one multiscale image visual.  When *geometry* found a channel axis
    the visual starts in composite mode; the image control switches it to
    single mode (and back) at runtime.

    Parameters
    ----------
    data_store : OMEZarrImageDataStore
        The backing multiscale store.
    geometry : _ViewerGeometry
        Metadata extracted by :func:`extract_viewer_geometry`.
    gui : "qt" or "anywidget"
        Which toolkit the viewer renders into.

    Returns
    -------
    cellier.convenience.Viewer
    """
    from cellier.convenience import Viewer

    viewer = Viewer(
        geometry.world,
        dim="2d",
        render_config=controller_render_config(),
        gui=gui,
    )
    viewer.controller.camera_reslice_enabled = True
    viewer.controller.camera_settle_threshold_s = 0.3

    viewer.add_image_multiscale(
        data_store,
        name="volume",
        **image_visual_kwargs(geometry, render_mode="mip", lod_bias=_INITIAL_LOD_BIAS),
    )

    # Center the spatial axes at the volume midpoint; extra axes such as
    # channel keep their default position of 0.  update_slice_indices merges.
    viewer.controller.update_slice_indices(
        viewer.scene.id, geometry.center_slice_indices()
    )

    return viewer


def build_viewer_layout(
    viewer: Viewer,
    geometry: _ViewerGeometry,
    *,
    min_canvas_size: tuple[int, int] | None = None,
) -> tuple[Layout, object]:
    """Build the canvas widget and the :class:`Layout` for *viewer*.

    Parameters
    ----------
    viewer : Viewer
        The viewer to build a layout for.
    geometry : _ViewerGeometry
        The panel geometry describing the canvas layout.
    min_canvas_size : tuple[int, int] or None, optional
        Minimum CSS pixel size ``(width, height)`` for the anywidget canvas.
        Ignored for the Qt gui. Defaults to cellier's built-in ``(600, 600)``
        when ``None``.

    Returns
    -------
    tuple[Layout, canvas_view]
        The layout spec plus the canvas view/widget (kept by the caller so it
        can install a paint tracker or avoid GC).
    """
    from cellier.convenience import AppearanceControls, Layout
    from cellier.convenience.gui import build_canvas_widget

    canvas_view = build_canvas_widget(
        viewer,
        geometry.axis_values,
        depth_range_3d=geometry.depth_range,
        canvas_size=min_canvas_size,
    )

    # Left dock: the image control, which holds both modes and the switch
    # between them.  The 2D/3D toggle needs no dock of its own -- cellier
    # embeds it in the canvas view's dims control.
    layout = Layout(center=canvas_view, left_dock=AppearanceControls())

    return layout, canvas_view


# ---------------------------------------------------------------------------
# Layer 2: blocking Qt launcher (scripts / CLI)
# ---------------------------------------------------------------------------


def launch_viewer(
    zarr_uri: str,
    theme: str = "dark",
    *,
    channel_axis: int | None = None,
    infer_multiscale_translations: bool = False,
    perf: StartupPerfTracer | None = None,
    gui: Literal["qt", "anywidget"] = "qt",
) -> None:
    """Open a viewer window and block until it is closed.

    Creates a ``QApplication`` if one does not already exist, then runs the
    Qt + asyncio event loop via ``QtAsyncio``.  Intended for scripts and the
    CLI.  For notebook (anywidget) use, call :func:`display_viewer`; for
    interactive Qt (IPython) use :func:`viewer`.

    Parameters
    ----------
    zarr_uri : str
        Path or URI to the OME-Zarr store.
    theme : str
        Registered theme name.  Defaults to ``"dark"``.
    channel_axis : int or None, optional
        Axis index to treat as the channel dimension.  Auto-detected from the
        OME-Zarr metadata when ``None``.
    infer_multiscale_translations : bool, optional
        Give the coarser levels the half-voxel offsets of centre-aligned
        downsampling (e.g. block averages) when the store declares no level
        translations; warns and keeps the declared ones otherwise.  Default
        ``False``: wrong for pyramids made by striding.
    perf : StartupPerfTracer or None, optional
        Optional startup performance tracer.
    gui : "qt" or "anywidget"
        Toolkit.  ``"anywidget"`` is not launchable from a script/CLI; use
        :func:`display_viewer` in a notebook instead.
    """
    if gui == "anywidget":
        raise ValueError(
            "gui='anywidget' cannot be launched from a script or the CLI. "
            "In a Jupyter/marimo notebook call "
            "oz_viewer.viewer.display_viewer(zarr_uri) instead."
        )

    import sys

    import fsspec.asyn as _fsspec_asyn
    import PySide6.QtAsyncio as QtAsyncio
    from PySide6.QtWidgets import QApplication

    from oz_viewer.theme import apply_theme

    # Ensure fsspec's background event loop exists before QtAsyncio takes over
    # the main loop; required for remote (s3/https) OME-Zarr stores.
    _fsspec_asyn.get_loop()

    _perf_mark(perf, "viewer.launch.start", theme=theme)
    app = QApplication.instance() or QApplication([sys.argv[0]])
    apply_theme(app, theme)
    _perf_mark(perf, "viewer.launch.qapp_ready")

    QtAsyncio.run(
        _run_viewer_async(
            zarr_uri,
            channel_axis=channel_axis,
            infer_multiscale_translations=infer_multiscale_translations,
            perf=perf,
        ),
        handle_sigint=True,
    )


async def _run_viewer_async(
    zarr_uri: str,
    *,
    channel_axis: int | None = None,
    infer_multiscale_translations: bool = False,
    perf: StartupPerfTracer | None = None,
) -> None:
    """Build, show, and keep the viewer alive until the window closes."""
    from PySide6.QtWidgets import QApplication

    asyncio.get_event_loop().set_exception_handler(_asyncio_exception_handler)
    _perf_mark(perf, "viewer.async.start")

    holder = _build_and_show_viewer_qt(
        zarr_uri,
        channel_axis=channel_axis,
        infer_multiscale_translations=infer_multiscale_translations,
        perf=perf,
    )
    _perf_mark(perf, "viewer.async.build_complete")

    app = QApplication.instance()
    close_event = asyncio.Event()
    app.aboutToQuit.connect(close_event.set)
    await close_event.wait()
    # Keep the holder referenced until the loop ends so the window/controls are
    # not garbage collected mid-session.
    del holder


class _ViewerHandle:
    """Keeps strong references to the live viewer objects (prevents GC)."""

    def __init__(self, viewer, window, canvas_view, geometry) -> None:
        self.viewer = viewer
        self.window = window
        self.canvas_view = canvas_view
        self.geometry = geometry
        self._perf_objects: tuple | None = None


def _build_and_show_viewer_qt(
    zarr_uri: str,
    *,
    channel_axis: int | None = None,
    infer_multiscale_translations: bool = False,
    perf: StartupPerfTracer | None = None,
) -> _ViewerHandle:
    """Build the Qt viewer window, show it, and arm first-frame startup."""
    # ``render_qt`` / ``_init_view`` are cellier-internal, but oz-viewer keeps
    # its own QtAsyncio loop (for the fsspec loop + asyncio exception handler +
    # perf tracer) rather than cellier's blocking ``launch``, so it drives the
    # same window build and first-frame arming that ``show``/``launch`` do.
    from cellier.convenience._launch import _init_view
    from cellier.convenience.layout._qt_renderer import render_qt

    _perf_mark(perf, "viewer.build.start")
    data_store, geometry = extract_viewer_geometry(
        zarr_uri,
        channel_axis=channel_axis,
        infer_multiscale_translations=infer_multiscale_translations,
        perf=perf,
    )
    viewer = build_viewer(data_store, geometry, gui="qt")
    _perf_mark(perf, "viewer.build.model_ready")

    layout, canvas_view = build_viewer_layout(viewer, geometry)
    window = render_qt(layout, viewer)
    window.setWindowTitle("OME-Zarr Viewer")
    _perf_mark(perf, "viewer.build.window_ready")

    handle = _ViewerHandle(viewer, window, canvas_view, geometry)
    if perf is not None and perf.enabled:
        _install_paint_tracker(handle, canvas_view, perf)

    window.show()
    _perf_mark(perf, "viewer.window.show_called")
    _init_view(viewer, fit="ready")
    return handle


def _install_paint_tracker(
    handle: _ViewerHandle,
    canvas_view: object,
    perf: StartupPerfTracer,
) -> None:
    """Record first-paint and a settle timing on the canvas widget."""
    from PySide6.QtCore import QEvent, QObject, QTimer

    widget = canvas_view.widget

    settled_timer = QTimer()
    settled_timer.setSingleShot(True)
    settled_timer.setInterval(300)

    def _on_settled() -> None:
        _perf_mark(perf, "viewer.canvas.startup_settled", quiet_ms=300)
        perf.report_rich_table()

    settled_timer.timeout.connect(_on_settled)

    class _PaintTracker(QObject):
        def eventFilter(self, watched, event):
            if event.type() == QEvent.Type.Paint:
                _perf_mark(perf, "viewer.canvas.first_paint")
                settled_timer.start()
                widget.removeEventFilter(self)
            return False

    paint_tracker = _PaintTracker()
    widget.installEventFilter(paint_tracker)
    handle._perf_objects = (paint_tracker, settled_timer)


# ---------------------------------------------------------------------------
# Layer 3: non-blocking entry points (interactive Qt + notebook anywidget)
# ---------------------------------------------------------------------------


def viewer(
    zarr_uri: str,
    theme: str = "dark",
    *,
    channel_axis: int | None = None,
    infer_multiscale_translations: bool = False,
) -> _ViewerHandle:
    """Open a Qt viewer window without blocking (IPython / interactive).

    The Qt event loop must already be running or be startable via IPython's
    ``enable_gui``.  For scripts use :func:`launch_viewer`; for notebooks use
    :func:`display_viewer`.

    Returns
    -------
    _ViewerHandle
        Keep a reference to prevent garbage collection.
    """
    from PySide6.QtWidgets import QApplication

    from oz_viewer.theme import apply_theme

    app = _ensure_qt_app()
    if app is None:
        raise RuntimeError(
            "No Qt event loop is running. "
            "Use launch_viewer() for scripts, display_viewer() for notebooks, "
            "or run inside IPython/Jupyter."
        )
    apply_theme(QApplication.instance(), theme)
    return _build_and_show_viewer_qt(
        zarr_uri,
        channel_axis=channel_axis,
        infer_multiscale_translations=infer_multiscale_translations,
    )


def display_viewer(
    zarr_uri: str,
    *,
    channel_axis: int | None = None,
    infer_multiscale_translations: bool = False,
    sidecar: bool = False,
    min_canvas_size: tuple[int, int] | None = None,
):
    """Build and present an anywidget viewer in a notebook (Jupyter / marimo).

    The notebook counterpart of :func:`launch_viewer`.  Returns the cellier
    ``DisplayHandle`` (Jupyter) or the host-native renderable (marimo).

    Parameters
    ----------
    zarr_uri : str
        Path or URI to the OME-Zarr store.
    channel_axis : int or None, optional
        Axis index to treat as the channel dimension.  Auto-detected when
        ``None``.
    infer_multiscale_translations : bool, optional
        Give the coarser levels the half-voxel offsets of centre-aligned
        downsampling (e.g. block averages) when the store declares no level
        translations; warns and keeps the declared ones otherwise.  Default
        ``False``: wrong for pyramids made by striding.
    sidecar : bool
        Present the viewer in a ``jupyterlab-sidecar`` tab instead of below
        the cell.  Requires the optional ``sidecar`` package (raises
        ``ImportError`` with an install hint if missing) and the Jupyter host
        (raises ``RuntimeError`` under marimo, which already tabs its cell
        output). The returned ``DisplayHandle`` owns the tab -- call
        ``handle.close()`` before re-running the cell, or the old tab is left
        open alongside the new one.
    min_canvas_size : tuple[int, int] or None, optional
        Minimum CSS pixel size ``(width, height)`` for the canvas. The canvas
        column won't shrink below this width as the notebook/dock is resized.
        Defaults to cellier's built-in ``(600, 600)`` when ``None``.
    """
    from cellier.convenience import display

    data_store, geometry = extract_viewer_geometry(
        zarr_uri,
        channel_axis=channel_axis,
        infer_multiscale_translations=infer_multiscale_translations,
    )
    v = build_viewer(data_store, geometry, gui="anywidget")
    layout, _canvas_view = build_viewer_layout(
        v, geometry, min_canvas_size=min_canvas_size
    )
    return display(
        v,
        layout,
        fit="ready",
        sidecar=_sidecar_options(sidecar, "OME-Zarr Viewer"),
    )
