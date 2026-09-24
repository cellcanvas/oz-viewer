"""OME-Zarr orthoviewer: 4-panel viewer (XY, XZ, YZ, 3D).

Rebuilt on :class:`cellier.convenience.OrthoViewer`, so the same builder runs
under both ``gui="qt"`` (desktop / CLI) and ``gui="anywidget"`` (Jupyter /
marimo).  The four pre-wired panels, cross-panel axis sync, the image fanned
out to every panel, and the image controls (``AppearanceControls``: a
selector between one control for the three 2D views and one for the 3D view,
each with its own single/composite switch) come from the convenience API;
the OME-Zarr-specific geometry (see :mod:`oz_viewer.viewer._geometry`) and
the 3D overlays (see :mod:`oz_viewer.viewer._ortho_overlays`) are
re-attached on top.

The overlay controls (orientation gizmo toggle, slice-plane opacity) have no
convenience equivalent, so they are a Qt-only dock (see
:mod:`oz_viewer.viewer._ortho_controls`) that is omitted in the anywidget path.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from oz_viewer.viewer._geometry import _ViewerGeometry, extract_viewer_geometry
from oz_viewer.viewer._image import controller_render_config, image_visual_kwargs
from oz_viewer.viewer._ortho_overlays import (
    _PLANE_COLOR_XY,
    _PLANE_COLOR_XZ,
    _PLANE_COLOR_YZ,
    OrthoOverlays,
    attach_ortho_overlays,
)
from oz_viewer.viewer._utils import (
    _asyncio_exception_handler,
    _ensure_qt_app,
    _perf_mark,
    _sidecar_options,
)

if TYPE_CHECKING:
    from cellier.convenience import OrthoViewer
    from cellier.data.image import OMEZarrImageDataStore

    from oz_viewer._perf import StartupPerfTracer


@dataclass
class _OrthoBuild:
    """The convenience OrthoViewer plus the handles overlays/controls need."""

    viewer: OrthoViewer
    #: The image's panel siblings, keyed ``xy``/``xz``/``yz``/``vol``.
    visuals: dict
    vol_visual_id: object
    overlays: OrthoOverlays | None = None


# ---------------------------------------------------------------------------
# Layer 1: build the convenience OrthoViewer (no launch, no Qt event loop)
# ---------------------------------------------------------------------------


def build_ortho_viewer(
    data_store: OMEZarrImageDataStore,
    geometry: _ViewerGeometry,
    *,
    gui: Literal["qt", "anywidget"] = "qt",
) -> _OrthoBuild:
    """Build a :class:`cellier.convenience.OrthoViewer` for an OME-Zarr store.

    Fans one multiscale image out to every panel.  When *geometry* found a
    channel axis the image starts in composite mode; the image controls
    switch the 2D views and the 3D view to single mode (and back) at runtime,
    each on its own.

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
    _OrthoBuild
    """
    from cellier.convenience import OrthoViewer

    if geometry.spatial_ndim != 3:
        raise ValueError(
            "The orthoviewer requires exactly 3 spatial axes, but the store has "
            f"{geometry.spatial_ndim} (axes={geometry.axis_names!r}, "
            f"channel_axis={geometry.channel_axis!r}). Use the single-panel "
            "viewer for lower-dimensional data."
        )

    viewer = OrthoViewer(
        geometry.world,
        spatial_axes=geometry.spatial_axes,
        render_config=controller_render_config(),
        gui=gui,
    )
    viewer.controller.camera_reslice_enabled = True
    viewer.controller.camera_settle_threshold_s = 0.3

    build = _add_image_visuals(viewer, data_store, geometry)
    _center_ortho_slices(viewer, geometry)
    return build


def _add_image_visuals(
    viewer: OrthoViewer,
    data_store: OMEZarrImageDataStore,
    geometry: _ViewerGeometry,
) -> _OrthoBuild:
    """Fan one multiscale image out to every panel.

    They start with one appearance.  cellier then gives the three 2D panels
    one image control and the ``vol`` panel another, so the 2D views and the
    3D view can differ from there on.  The ``vol`` panel's level-of-detail
    policy differs from the start: it is pinned to the coarsest level and not
    frustum-culled.  Those appearance fields are not in the control, so it
    never overwrites them.
    """
    controller = viewer.controller
    visuals = viewer.add_image_multiscale(
        data_store,
        name="image",
        **image_visual_kwargs(geometry, render_mode="iso", lod_bias=1.0),
    )

    vol_visual = visuals["vol"]
    controller.update_appearance_field(
        vol_visual.id, "force_level", data_store.n_levels - 1
    )
    controller.update_appearance_field(vol_visual.id, "frustum_cull", False)
    vol_visual.aabb.enabled = True
    vol_visual.aabb.color = "#ff00ff"

    return _OrthoBuild(viewer=viewer, visuals=visuals, vol_visual_id=vol_visual.id)


def _center_ortho_slices(viewer: OrthoViewer, geometry: _ViewerGeometry) -> None:
    """Center each panel's sliced spatial axis at the volume midpoint.

    Applied from the OME-Zarr metadata rather than via
    ``OrthoViewer.center_slices`` so the midpoint agrees with
    :attr:`_ViewerGeometry.axis_values`, which spans voxel centres rather than
    voxel edges.  Extra axes (channel) keep position 0.
    """
    center = geometry.center_slice_indices()
    for scene in viewer.scenes.values():
        # Panels after the first are usually already mirrored by the axis sync.
        current = scene.dims.selection.slice_indices
        if any(current.get(a) != v for a, v in center.items()):
            viewer.controller.update_slice_indices(scene.id, center)


# ---------------------------------------------------------------------------
# Layer 2: canvases + layout + cross-toolkit 2D axis overlays
# ---------------------------------------------------------------------------


def build_ortho_layout(
    build: _OrthoBuild,
    geometry: _ViewerGeometry,
    *,
    min_canvas_size: tuple[int, int] | None = None,
) -> tuple:
    """Build the 2x2 grid, the 2D axis overlays, and the :class:`Layout`.

    Parameters
    ----------
    build : _OrthoBuild
        The built orthoviewer bundle (viewer, scenes, and canvas ids).
    geometry : _ViewerGeometry
        The panel geometry describing the 2x2 grid layout.
    min_canvas_size : tuple[int, int] or None, optional
        Minimum CSS pixel size ``(width, height)`` for each anywidget canvas
        panel. Ignored for the Qt gui. Defaults to cellier's built-in
        ``(600, 600)`` when ``None``.

    Returns
    -------
    tuple[Layout, grid]
        The layout spec plus the grid widget/view (kept by the caller so it is
        not garbage collected and so a paint tracker can be installed).
    """
    from cellier.convenience import AppearanceControls, Layout
    from cellier.convenience.gui import build_ortho_grid_widget

    viewer = build.viewer
    grid = build_ortho_grid_widget(
        viewer,
        geometry.axis_values,
        depth_range_3d=geometry.depth_range,
        canvas_size=min_canvas_size,
    )
    # Screen-space 2D orientation axes on each slice panel (both toolkits).
    _add_2d_axis_overlays(viewer)

    # The image controls: a selector between the 2D views (xy, xz, yz driven
    # together) and the 3D view.
    layout = Layout(center=grid, left_dock=AppearanceControls())
    return layout, grid


def _add_2d_axis_overlays(viewer: OrthoViewer) -> None:
    """Add screen-space ``CenteredAxes2D`` overlays to the three 2D panels."""
    from cellier.visuals import CenteredAxes2D, CenteredAxes2DAppearance

    controller = viewer.controller
    scenes = viewer.scenes
    label_color = (1.0, 0.0, 1.0, 1.0)
    specs = (
        ("xy", "xy_axes", "Y", "X", _PLANE_COLOR_XZ, _PLANE_COLOR_YZ),
        ("xz", "xz_axes", "Z", "X", _PLANE_COLOR_XY, _PLANE_COLOR_YZ),
        ("yz", "yz_axes", "Z", "Y", _PLANE_COLOR_XY, _PLANE_COLOR_XZ),
    )
    for key, name, label_a, label_b, color_a, color_b in specs:
        canvas_ids = controller.get_canvas_ids(scenes[key].id)
        if not canvas_ids:
            continue
        controller.add_canvas_overlay(
            canvas_ids[0],
            CenteredAxes2D(
                name=name,
                axis_a_direction=(0.0, 1.0, 0.0),
                axis_a_label=label_a,
                axis_b_direction=(1.0, 0.0, 0.0),
                axis_b_label=label_b,
                appearance=CenteredAxes2DAppearance(
                    axis_a_color=(*color_a, 1.0),
                    axis_b_color=(*color_b, 1.0),
                    label_color=label_color,
                ),
            ),
        )


# ---------------------------------------------------------------------------
# Layer 3: blocking Qt launcher (scripts / CLI)
# ---------------------------------------------------------------------------


def launch_orthoviewer(
    zarr_uri: str,
    theme: str = "dark",
    *,
    channel_axis: int | None = None,
    infer_multiscale_translations: bool = False,
    perf: StartupPerfTracer | None = None,
    gui: Literal["qt", "anywidget"] = "qt",
) -> None:
    """Open an orthoviewer window and block until it is closed.

    Creates a ``QApplication`` if one does not already exist, then runs the
    Qt + asyncio event loop via ``QtAsyncio``.  Intended for scripts and the
    CLI.  For notebook (anywidget) use, call :func:`display_orthoviewer`; for
    interactive Qt (IPython) use :func:`orthoviewer`.

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
        :func:`display_orthoviewer` in a notebook instead.
    """
    if gui == "anywidget":
        raise ValueError(
            "gui='anywidget' cannot be launched from a script or the CLI. "
            "In a Jupyter/marimo notebook call "
            "oz_viewer.viewer.display_orthoviewer(zarr_uri) instead."
        )

    import sys

    import fsspec.asyn as _fsspec_asyn
    import PySide6.QtAsyncio as QtAsyncio
    from PySide6.QtWidgets import QApplication

    from oz_viewer.theme import apply_theme

    # Ensure fsspec's background loop exists before QtAsyncio owns the main loop.
    _fsspec_asyn.get_loop()

    _perf_mark(perf, "viewer.launch.start", theme=theme)
    app = QApplication.instance() or QApplication([sys.argv[0]])
    apply_theme(app, theme)
    _perf_mark(perf, "viewer.launch.qapp_ready")

    QtAsyncio.run(
        _run_orthoviewer_async(
            zarr_uri,
            channel_axis=channel_axis,
            infer_multiscale_translations=infer_multiscale_translations,
            perf=perf,
        ),
        handle_sigint=True,
    )


async def _run_orthoviewer_async(
    zarr_uri: str,
    *,
    channel_axis: int | None = None,
    infer_multiscale_translations: bool = False,
    perf: StartupPerfTracer | None = None,
) -> None:
    """Build, show, and keep the orthoviewer alive until the window closes."""
    from PySide6.QtWidgets import QApplication

    asyncio.get_event_loop().set_exception_handler(_asyncio_exception_handler)
    _perf_mark(perf, "viewer.async.start")

    handle = _build_and_show_ortho_qt(
        zarr_uri,
        channel_axis=channel_axis,
        infer_multiscale_translations=infer_multiscale_translations,
        perf=perf,
    )
    _perf_mark(perf, "viewer.async.build_complete")

    app = QApplication.instance()
    close_event = asyncio.Event()
    app.aboutToQuit.connect(close_event.set)
    app.aboutToQuit.connect(handle.close)
    await close_event.wait()
    del handle


@dataclass
class _OrthoHandle:
    """Keeps strong references to the live orthoviewer objects (prevents GC)."""

    build: _OrthoBuild
    window: object
    grid: object
    controls: object | None
    geometry: _ViewerGeometry
    _perf_objects: tuple | None = field(default=None)

    def close(self) -> None:
        if self.controls is not None:
            self.controls.close()
        overlays = self.build.overlays
        if overlays is not None:
            overlays.close()
        grid = self.grid
        if grid is not None and hasattr(grid, "close"):
            grid.close()


def _build_and_show_ortho_qt(
    zarr_uri: str,
    *,
    channel_axis: int | None = None,
    infer_multiscale_translations: bool = False,
    perf: StartupPerfTracer | None = None,
) -> _OrthoHandle:
    """Build the Qt orthoviewer window, show it, and arm first-frame startup."""
    # ``render_qt`` / ``_init_view`` are cellier-internal, but oz-viewer keeps
    # its own QtAsyncio loop (fsspec loop + asyncio handler + perf tracer)
    # rather than cellier's blocking ``launch``.
    from cellier.convenience._launch import _init_view
    from cellier.convenience.layout._qt_renderer import render_qt

    _perf_mark(perf, "viewer.build.start")
    data_store, geometry = extract_viewer_geometry(
        zarr_uri,
        channel_axis=channel_axis,
        infer_multiscale_translations=infer_multiscale_translations,
        perf=perf,
    )
    build = build_ortho_viewer(data_store, geometry, gui="qt")
    _perf_mark(perf, "viewer.build.model_ready")

    build.overlays = attach_ortho_overlays(
        build.viewer, geometry, vol_visual_id=build.vol_visual_id
    )
    _perf_mark(perf, "viewer.build.overlays_ready")

    layout, grid = build_ortho_layout(build, geometry)
    window = render_qt(layout, build.viewer)
    window.setWindowTitle("OME-Zarr Orthoviewer")
    _perf_mark(perf, "viewer.build.window_ready")

    controls = _attach_qt_controls(build, window)
    _perf_mark(perf, "viewer.build.controls_ready")

    handle = _OrthoHandle(
        build=build,
        window=window,
        grid=grid,
        controls=controls,
        geometry=geometry,
    )
    if perf is not None and perf.enabled:
        _install_ortho_paint_tracker(handle, grid, perf)

    window.show()
    _perf_mark(perf, "viewer.window.show_called")
    _init_view(build.viewer, fit="ready")
    return handle


def _attach_qt_controls(build: _OrthoBuild, window) -> object | None:
    """Bolt oz's Qt-only overlay panel onto the rendered ``QMainWindow``.

    The image controls are already in the layout's left ``AppearanceControls``
    dock; this adds the overlay toggles on the right.
    """
    from PySide6 import QtWidgets
    from PySide6.QtCore import Qt

    from oz_viewer.viewer._ortho_controls import build_overlay_controls_panel

    if build.overlays is None:
        return None
    controls = build_overlay_controls_panel(build.viewer.controller, build.overlays)

    dock = QtWidgets.QDockWidget("Overlays", window)
    dock.setWidget(controls.widget)
    dock.setFeatures(
        QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
        | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
    )
    window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
    return controls


def _install_ortho_paint_tracker(
    handle: _OrthoHandle,
    grid: object,
    perf: StartupPerfTracer,
) -> None:
    """Record first-paint across all four canvases plus a settle timing."""
    from PySide6.QtCore import QEvent, QObject, QTimer

    canvases = getattr(grid, "canvases", {})
    widgets = {key: cw.widget for key, cw in canvases.items()}
    if not widgets:
        return

    paint_seen: set[str] = set()
    total = len(widgets)

    settled_timer = QTimer()
    settled_timer.setSingleShot(True)
    settled_timer.setInterval(300)

    def _on_settled() -> None:
        if len(paint_seen) == total:
            _perf_mark(perf, "viewer.canvas.startup_settled", quiet_ms=300)
            perf.report_rich_table()

    settled_timer.timeout.connect(_on_settled)

    class _CanvasPaintTracker(QObject):
        def eventFilter(self, watched, event):
            if event.type() != QEvent.Type.Paint:
                return False
            for name, widget in widgets.items():
                if watched is widget and name not in paint_seen:
                    paint_seen.add(name)
                    _perf_mark(
                        perf,
                        "viewer.canvas.first_paint",
                        scene=name,
                        n_seen=len(paint_seen),
                        n_total=total,
                    )
                    settled_timer.start()
                    break
            return False

    tracker = _CanvasPaintTracker()
    for widget in widgets.values():
        widget.installEventFilter(tracker)
    handle._perf_objects = (tracker, settled_timer)


# ---------------------------------------------------------------------------
# Layer 4: non-blocking entry points (interactive Qt + notebook anywidget)
# ---------------------------------------------------------------------------


def orthoviewer(
    zarr_uri: str,
    theme: str = "dark",
    *,
    channel_axis: int | None = None,
    infer_multiscale_translations: bool = False,
) -> _OrthoHandle:
    """Open a Qt orthoviewer window without blocking (IPython / interactive).

    The Qt event loop must already be running or be startable via IPython's
    ``enable_gui``.  For scripts use :func:`launch_orthoviewer`; for notebooks
    use :func:`display_orthoviewer`.

    Returns
    -------
    _OrthoHandle
        Keep a reference to prevent garbage collection.
    """
    from PySide6.QtWidgets import QApplication

    from oz_viewer.theme import apply_theme

    app = _ensure_qt_app()
    if app is None:
        raise RuntimeError(
            "No Qt event loop is running. "
            "Use launch_orthoviewer() for scripts, display_orthoviewer() for "
            "notebooks, or run inside IPython/Jupyter."
        )
    apply_theme(QApplication.instance(), theme)
    return _build_and_show_ortho_qt(
        zarr_uri,
        channel_axis=channel_axis,
        infer_multiscale_translations=infer_multiscale_translations,
    )


def display_orthoviewer(
    zarr_uri: str,
    *,
    channel_axis: int | None = None,
    infer_multiscale_translations: bool = False,
    sidecar: bool = False,
    min_canvas_size: tuple[int, int] | None = None,
):
    """Build and present an anywidget orthoviewer in a notebook (Jupyter / marimo).

    The notebook counterpart of :func:`launch_orthoviewer`.  Returns the cellier
    ``DisplayHandle`` (Jupyter) or the host-native renderable (marimo).  The
    Qt-only overlay panel is omitted; the four synced panels, the 3D overlays,
    and the image control dock render normally.

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
        Present the orthoviewer in a ``jupyterlab-sidecar`` tab instead of
        below the cell.  Requires the optional ``sidecar`` package (raises
        ``ImportError`` with an install hint if missing) and the Jupyter host
        (raises ``RuntimeError`` under marimo, which already tabs its cell
        output). The returned ``DisplayHandle`` owns the tab -- call
        ``handle.close()`` before re-running the cell, or the old tab is left
        open alongside the new one.
    min_canvas_size : tuple[int, int] or None, optional
        Minimum CSS pixel size ``(width, height)`` for each of the four grid
        panels. Defaults to cellier's built-in ``(600, 600)`` when ``None``.
    """
    from cellier.convenience import display

    data_store, geometry = extract_viewer_geometry(
        zarr_uri,
        channel_axis=channel_axis,
        infer_multiscale_translations=infer_multiscale_translations,
    )
    build = build_ortho_viewer(data_store, geometry, gui="anywidget")
    build.overlays = attach_ortho_overlays(
        build.viewer, geometry, vol_visual_id=build.vol_visual_id
    )
    layout, _grid = build_ortho_layout(build, geometry, min_canvas_size=min_canvas_size)
    return display(
        build.viewer,
        layout,
        fit="ready",
        sidecar=_sidecar_options(sidecar, "OME-Zarr Orthoviewer"),
    )
