"""Single-panel 2D / 3D toggle viewer for OME-Zarr images.

Rebuilt on top of :mod:`cellier.convenience`, so the same builder runs under
both ``gui="qt"`` (desktop / CLI) and ``gui="anywidget"`` (Jupyter / marimo).
Appearance controls and per-channel controls are provided by cellier's
cross-toolkit ``Layout`` docks and the 2D/3D toggle by the dims control
embedded in the canvas view; this module only supplies the
OME-Zarr-specific geometry (see :mod:`oz_viewer.viewer._geometry`) and the
Qt-specific launch niceties oz-viewer cares about (theme, fsspec loop, asyncio
exception handling, startup perf tracing).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Literal

from oz_viewer.viewer._geometry import _ViewerGeometry, extract_viewer_geometry
from oz_viewer.viewer._utils import (
    _asyncio_exception_handler,
    _ensure_qt_app,
    _perf_mark,
    _sidecar_options,
)
from oz_viewer.viewer._widgets import _DEFAULT_COLORMAPS

if TYPE_CHECKING:
    from cellier.convenience import Viewer
    from cellier.convenience.layout._spec import Layout
    from cellier.data.image import OMEZarrImageDataStore

    from oz_viewer._perf import StartupPerfTracer

# Appearance fields exposed in the single-channel appearance dock, in order.
_APPEARANCE_FIELDS = [
    "color_map",
    "clim",
    "render_mode",
    "iso_threshold",
    "attenuation",
    "lod_bias",
]
# Per-channel fields exposed in the multichannel dock, in order.
_CHANNEL_FIELDS = ["visible", "color_map", "clim", "opacity"]

_INITIAL_LOD_BIAS = 1.5


def _render_config() -> object:
    """The controller render-pipeline config used by both viewers."""
    from cellier.render import (
        RenderManagerConfig,
        SlicingConfig,
        TemporalAccumulationConfig,
    )

    return RenderManagerConfig(
        slicing=SlicingConfig(batch_size=32, render_every=4),
        temporal=TemporalAccumulationConfig(enabled=False),
    )


def _visual_render_config() -> object:
    """LOD / GPU-budget config for the single-panel multiscale visual."""
    from cellier.visuals import MultiscaleImageRenderConfig

    return MultiscaleImageRenderConfig(
        block_size=32,
        gpu_budget_bytes=2048 * 1024**2,
        gpu_budget_bytes_2d=64 * 1024**2,
    )


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

    Chooses a single-channel or multichannel visual at build time based on
    whether *geometry* found a channel axis (D1 in the conversion plan): there
    is no runtime single<->multichannel toggle.

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
        render_config=_render_config(),
        gui=gui,
    )
    viewer.controller.camera_reslice_enabled = True
    viewer.controller.camera_settle_threshold_s = 0.3

    if geometry.channel_axis is not None:
        _add_multichannel_visual(viewer, data_store, geometry)
    else:
        _add_single_channel_visual(viewer, data_store, geometry)

    # Center the sliced spatial axis (e.g. Z) at the volume midpoint; extra
    # axes such as channel keep their default position of 0.
    center = geometry.center_slice_indices()
    current = dict(viewer.scene.dims.selection.slice_indices)
    updated = {a: center[a] for a in center if a in current}
    if updated:
        current.update(updated)
        viewer.controller.update_slice_indices(viewer.scene.id, current)

    return viewer


def _add_single_channel_visual(
    viewer: Viewer,
    data_store: OMEZarrImageDataStore,
    geometry: _ViewerGeometry,
) -> None:
    """Add a single-channel multiscale image visual + appearance controls."""
    from cellier.convenience import MultiscaleImageControlsConfig
    from cellier.visuals import MultiscaleImageAppearance

    clim_max = geometry.initial_clim_max
    viewer.add_image_multiscale(
        data_store,
        appearance=MultiscaleImageAppearance(
            color_map="viridis",
            clim=(0.0, clim_max),
            lod_bias=_INITIAL_LOD_BIAS,
            iso_threshold=clim_max / 2.0,
            render_mode="mip",
            attenuation=1.0,
        ),
        name="volume",
        render_config=_visual_render_config(),
        transform=geometry.voxel_to_world,
        controls=MultiscaleImageControlsConfig(
            appearance=_APPEARANCE_FIELDS,
            colormap_names=_DEFAULT_COLORMAPS,
            clim_range=geometry.clim_range,
        ),
    )


def _add_multichannel_visual(
    viewer: Viewer,
    data_store: OMEZarrImageDataStore,
    geometry: _ViewerGeometry,
) -> None:
    """Add a multichannel multiscale visual + per-channel controls.

    The channel axis is moved to ``stacked_axes`` so it renders as a stack (all
    channels at once) and no redundant dims slider appears for it -- the
    ``ChannelControls`` dock owns per-channel visibility instead.
    """
    from cellier.convenience import ChannelControlsConfig
    from cellier.visuals import ChannelAppearance

    clim_max = geometry.initial_clim_max
    channel_axis = geometry.channel_axis
    assert channel_axis is not None
    n = geometry.n_channels
    # Start every channel visible; the ChannelControls dock toggles from there.
    channels = {
        i: ChannelAppearance(
            color_map=_DEFAULT_COLORMAPS[i % len(_DEFAULT_COLORMAPS)],
            clim=(0.0, clim_max),
            visible=True,
        )
        for i in range(n)
    }
    # Raise the channel-node budget to cover every channel so construction and
    # the ChannelControls dock never exceed the cap for real (few-channel) data.
    max_channels = max(8, n)
    viewer.add_multichannel_image_multiscale(
        data_store,
        channel_axis=channel_axis,
        channels=channels,
        name="multichannel_volume",
        render_config=_visual_render_config(),
        transform=geometry.voxel_to_world,
        max_channels_2d=max_channels,
        max_channels_3d=max_channels,
        controls=ChannelControlsConfig(
            fields=_CHANNEL_FIELDS,
            colormap_names=_DEFAULT_COLORMAPS,
            clim_range=geometry.clim_range,
        ),
    )

    # Stack the channel axis: drop it from slice_indices and mark it stacked so
    # the multichannel visual renders every channel and no slider is shown.
    current = dict(viewer.scene.dims.selection.slice_indices)
    current.pop(channel_axis, None)
    viewer.controller.update_slice_indices(viewer.scene.id, current)
    viewer.controller.set_stacked_axes(viewer.scene.id, (channel_axis,))


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
    from cellier.convenience import AppearanceControls, ChannelControls, Layout
    from cellier.convenience.gui import build_canvas_widget

    canvas_view = build_canvas_widget(
        viewer,
        geometry.axis_values,
        depth_range_3d=geometry.depth_range,
        canvas_size=min_canvas_size,
    )

    # Left dock: per-channel controls for multichannel data, otherwise the
    # single-channel appearance panel.  The 2D/3D toggle needs no dock of its
    # own -- cellier embeds it in the canvas view's dims control.
    if geometry.channel_axis is not None:
        left: object = ChannelControls()
    else:
        left = AppearanceControls()

    layout = Layout(center=canvas_view, left_dock=left)
    return layout, canvas_view


# ---------------------------------------------------------------------------
# Layer 2: blocking Qt launcher (scripts / CLI)
# ---------------------------------------------------------------------------


def launch_viewer(
    zarr_uri: str,
    theme: str = "dark",
    *,
    channel_axis: int | None = None,
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
        _run_viewer_async(zarr_uri, channel_axis=channel_axis, perf=perf),
        handle_sigint=True,
    )


async def _run_viewer_async(
    zarr_uri: str,
    *,
    channel_axis: int | None = None,
    perf: StartupPerfTracer | None = None,
) -> None:
    """Build, show, and keep the viewer alive until the window closes."""
    from PySide6.QtWidgets import QApplication

    asyncio.get_event_loop().set_exception_handler(_asyncio_exception_handler)
    _perf_mark(perf, "viewer.async.start")

    holder = _build_and_show_viewer_qt(zarr_uri, channel_axis=channel_axis, perf=perf)
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
        zarr_uri, channel_axis=channel_axis, perf=perf
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
    return _build_and_show_viewer_qt(zarr_uri, channel_axis=channel_axis)


def display_viewer(
    zarr_uri: str,
    *,
    channel_axis: int | None = None,
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

    data_store, geometry = extract_viewer_geometry(zarr_uri, channel_axis=channel_axis)
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
