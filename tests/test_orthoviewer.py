"""Headless build tests for the convenience-based orthoviewer.

These exercise the full Qt build path (``_build_and_show_ortho_qt``) offscreen:
the four synced panels, the build-time single/multichannel choice, the
re-attached 3D overlays, and the Qt control docks.  A live GPU render is not
exercised (there is no display in CI), only construction.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    return app


def _dock_titles(window) -> list[str]:
    from PySide6.QtWidgets import QDockWidget

    return [d.windowTitle() for d in window.findChildren(QDockWidget)]


def _dock_control_names(window, title: str) -> set[str]:
    """Every group-box title and label inside the dock called *title*.

    See the twin helper in ``test_viewer.py``: a dock title alone does not show
    that cellier put any controls in it.
    """
    from PySide6.QtWidgets import QCheckBox, QDockWidget, QGroupBox, QLabel

    dock = next(d for d in window.findChildren(QDockWidget) if d.windowTitle() == title)
    names = {g.title() for g in dock.findChildren(QGroupBox)}
    names |= {w.text() for w in dock.findChildren(QLabel)}
    names |= {w.text() for w in dock.findChildren(QCheckBox)}
    return {n for n in names if n}


def test_single_channel_ortho_build(qapp, tmp_path):
    """Blobs (z,y,x) -> single-channel ortho: 4 panels, overlays, Rendering dock."""
    from oz_viewer.data._blobs import make_example_zarr
    from oz_viewer.viewer._orthoviewer import _build_and_show_ortho_qt

    zarr_path = make_example_zarr(output_path=tmp_path / "blobs.ome.zarr")
    handle = _build_and_show_ortho_qt(f"file://{zarr_path}")
    try:
        build = handle.build
        assert build.is_multichannel is False
        assert set(build.visuals) == {"xy", "xz", "yz", "vol"}
        assert set(build.viewer.scenes) == {"xy", "xz", "yz", "vol"}

        # Overlays attached: plane + three orientation-axis meshes, ISO profile.
        overlays = build.overlays
        assert overlays is not None
        assert len(overlays.axis_visual_ids) == 3
        assert overlays.transparency_manager.current_mode == "iso"

        # A dims change reaches the overlays: the slice plane and the gizmo
        # follow the XY panel's slider.  Off the voxel grid on purpose -- slice
        # positions are floats, and an integer is where rounding rules agree.
        controller = build.viewer.controller
        z_axis = handle.geometry.spatial_axes[0]
        controller.update_slice_indices(build.viewer.scenes["xy"].id, {z_axis: 12.3})
        assert overlays.plane_store.positions[0, 0] == pytest.approx(12.3)
        gizmo = controller.get_visual_model(overlays.axis_visual_ids[0])
        assert gizmo.transform.translation[z_axis] == pytest.approx(12.3)

        # Qt-only appearance panel on the left dock; no channel dock.
        assert "Rendering" in _dock_titles(handle.window)
    finally:
        handle.close()


def test_multichannel_ortho_build(qapp, write_demo_ome):
    """Demo image (c,z,y,x) -> multichannel ortho: channel stacked, both docks."""
    from oz_viewer.viewer._orthoviewer import _build_and_show_ortho_qt

    zarr_path = write_demo_ome("image")
    handle = _build_and_show_ortho_qt(f"file://{zarr_path}")
    try:
        build = handle.build
        assert build.is_multichannel is True
        assert set(build.visuals) == {"xy", "xz", "yz", "vol"}

        # Channel axis (0) is stacked, not sliced, so no channel slider remains.
        for scene in build.viewer.scenes.values():
            assert 0 not in scene.dims.selection.slice_indices

        # Multichannel volume is locked to MIP; overlays present.
        assert build.overlays.transparency_manager.current_mode == "mip"

        # The overlay meshes are 3-D (z, y, x) and broadcast over the channel
        # axis, so they exist at every channel without per-channel updates.
        controller = build.viewer.controller
        world = build.viewer.scenes["vol"].dims.world_coordinate_system
        overlay_ids = (build.overlays.plane_visual.id, *build.overlays.axis_visual_ids)
        for visual_id in overlay_ids:
            transform = controller.get_visual_model(visual_id).transform
            assert transform.broadcast_axes == frozenset({world.axes[0].id})

        # A channel slider steps through channels; spatial sliders are free.
        from cellier.convenience import ContinuousAxisValues, DiscreteAxisValues

        axis_values = handle.geometry.axis_values
        assert axis_values[0] == DiscreteAxisValues(
            values=tuple(float(i) for i in range(handle.geometry.n_channels))
        )
        assert all(isinstance(axis_values[a], ContinuousAxisValues) for a in (1, 2, 3))

        # ChannelControls dock ("Left") plus the Qt-only volume group ("Volume").
        titles = _dock_titles(handle.window)
        assert "Left" in titles
        assert "Volume" in titles

        # The channel dock drives all four panels' sibling visuals, so it must
        # actually hold one group per channel -- not just exist.
        names = _dock_control_names(handle.window, "Left")
        assert {f"Channel {i}" for i in range(handle.geometry.n_channels)} <= names
    finally:
        handle.close()
