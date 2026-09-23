"""Headless build tests for the convenience-based orthoviewer.

These exercise the full Qt build path (``_build_and_show_ortho_qt``) offscreen:
the four synced panels, the image's starting mode and the runtime switch
between modes, the re-attached 3D overlays, and the control docks.  A live GPU
render is not
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
    """Blobs (z,y,x) -> single-mode ortho: 4 panels, overlays, both docks."""
    from oz_viewer.data._blobs import make_example_zarr
    from oz_viewer.viewer._orthoviewer import _build_and_show_ortho_qt

    zarr_path = make_example_zarr(output_path=tmp_path / "blobs.ome.zarr")
    handle = _build_and_show_ortho_qt(f"file://{zarr_path}")
    try:
        build = handle.build
        assert set(build.visuals) == {"xy", "xz", "yz", "vol"}
        assert set(build.viewer.scenes) == {"xy", "xz", "yz", "vol"}
        assert not any(v.composite for v in build.visuals.values())

        # One linked appearance; only the vol panel's LOD policy differs.
        assert build.visuals["vol"].appearance.force_level is not None
        assert build.visuals["vol"].appearance.frustum_cull is False
        assert build.visuals["xy"].appearance.force_level is None

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

        # The image control (both toolkits) plus the Qt-only overlay panel.
        titles = _dock_titles(handle.window)
        assert "Left" in titles
        assert "Overlays" in titles
        names = _dock_control_names(handle.window, "Left")
        assert {"Contrast", "Colormap", "Render mode", "LOD bias"} <= names
    finally:
        handle.close()


def test_multichannel_ortho_build(qapp, write_demo_ome):
    """Demo image (c,z,y,x) -> composite ortho, switchable to single at runtime."""
    from oz_viewer.viewer._orthoviewer import _build_and_show_ortho_qt

    zarr_path = write_demo_ome("image")
    handle = _build_and_show_ortho_qt(f"file://{zarr_path}")
    try:
        build = handle.build
        viewer = build.viewer
        controller = viewer.controller
        assert set(build.visuals) == {"xy", "xz", "yz", "vol"}
        assert all(v.composite for v in build.visuals.values())

        # A composited channel axis (0) has no slider on any panel.
        for scene in viewer.scenes.values():
            assert 0 not in scene.slider_axes

        # Composite channels render MIP, so the overlays use the MIP profile.
        manager = build.overlays.transparency_manager
        assert manager.current_mode == "mip"

        # The overlay meshes are 3-D (z, y, x) and broadcast over the channel
        # axis, so they exist at every channel without per-channel updates.
        world = viewer.scenes["vol"].dims.world_coordinate_system
        overlay_ids = (build.overlays.plane_visual.id, *build.overlays.axis_visual_ids)
        for visual_id in overlay_ids:
            transform = controller.get_visual_model(visual_id).transform
            assert transform.broadcast_axes == frozenset({world.axes[0].id})

        # A channel slider steps through channels; spatial sliders are free.
        from cellier.convenience import ContinuousAxisValues, DiscreteAxisValues

        axis_values = handle.geometry.axis_values
        assert axis_values[0] == DiscreteAxisValues(
            values=tuple(float(i) for i in range(handle.geometry.n_channels)),
            labels=handle.geometry.channel_labels,
        )
        assert all(isinstance(axis_values[a], ContinuousAxisValues) for a in (1, 2, 3))

        # The image control on the left drives all four panels; the overlay
        # panel on the right.
        titles = _dock_titles(handle.window)
        assert "Left" in titles
        assert "Overlays" in titles
        names = _dock_control_names(handle.window, "Left")
        assert {f"Channel {i}" for i in range(handle.geometry.n_channels)} <= names
        assert "Composite channels" in names

        # Runtime switch to single mode: every panel follows, the channel axis
        # gets a slider, and the manager picks up the single ISO render mode
        # and applies its volume profile.
        vol_id = build.vol_visual_id
        viewer.set_image_composite(build.visuals, False)
        assert not any(v.composite for v in build.visuals.values())
        for scene in viewer.scenes.values():
            assert 0 in scene.slider_axes
        assert manager.current_mode == "iso"
        vol = controller.get_visual_model(vol_id)
        assert vol.single.opacity == pytest.approx(manager.current_profile.opacity)
        assert vol.appearance.transparency_mode == "weighted_blend"

        # An opacity edit through the image group is remembered per mode.
        viewer.update_image_single_field(build.visuals, "opacity", 0.6)
        assert manager.current_profile.opacity == pytest.approx(0.6)

        # Back to composite: blending returns to cellier's default, and the
        # remembered ISO profile is re-applied on the next switch to single.
        viewer.set_image_composite(build.visuals, True)
        assert manager.current_mode == "mip"
        assert vol.appearance.transparency_mode is None
        viewer.set_image_composite(build.visuals, False)
        assert vol.single.opacity == pytest.approx(0.6)
    finally:
        handle.close()
