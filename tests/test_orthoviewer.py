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


def _dock_selector_items(window, title: str) -> list[list[str]]:
    """The items of every combo box inside the dock called *title*."""
    from PySide6.QtWidgets import QComboBox, QDockWidget

    dock = next(d for d in window.findChildren(QDockWidget) if d.windowTitle() == title)
    return [
        [combo.itemText(i) for i in range(combo.count())]
        for combo in dock.findChildren(QComboBox)
    ]


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

        # The 3D view draws the 2D views' slices on their planes, so it picks
        # its level of detail from the camera as they do.
        vol = build.visuals["vol"]
        assert vol.single.render_mode == "plane"
        assert vol.appearance.force_level is None
        assert vol.appearance.frustum_cull is True
        assert build.visuals["xy"].appearance.force_level is None

        # Overlays attached: three orientation-axis meshes, plane profile.
        overlays = build.overlays
        assert overlays is not None
        assert len(overlays.axis_visual_ids) == 3
        assert overlays.transparency_manager.current_mode == "plane"

        # cellier ties three outlined render planes to the 2D sliders; they
        # start on a voxel centre (z is 200 voxels of 5 um: 497.5 is a face).
        planes = build.viewer.plane_controller
        assert planes.mode == "slices"
        assert len(vol.render_planes) == 3
        assert all(plane.outline.enabled for plane in vol.render_planes)
        assert planes.positions()[0] == pytest.approx(500.0)

        # A dims change reaches both: the slice plane and the gizmo follow
        # the XY panel's slider.  Off the voxel grid on purpose -- slice
        # positions are floats, and an integer is where rounding rules agree.
        controller = build.viewer.controller
        z_axis = handle.geometry.spatial_axes[0]
        controller.update_slice_indices(build.viewer.scenes["xy"].id, {z_axis: 12.3})
        assert vol.render_planes[0].origin[0] == pytest.approx(12.3)
        gizmo = controller.get_visual_model(overlays.axis_visual_ids[0])
        assert gizmo.transform.translation[z_axis] == pytest.approx(12.3)

        # The image control (both toolkits) plus the Qt-only overlay panel.
        titles = _dock_titles(handle.window)
        assert "Left" in titles
        assert "Overlays" in titles
        names = _dock_control_names(handle.window, "Left")
        assert {"Contrast", "Colormap", "Render mode", "Settled bias"} <= names
        assert "Data fetch status" in names

        # One control for the three 2D views and one for the 3D view, chosen
        # with the dock's selector.
        from cellier.convenience.layout._shared import appearance_targets

        assert ["image (2D views)", "image (3D view)"] in _dock_selector_items(
            handle.window, "Left"
        )
        views_2d, view_3d = appearance_targets(build.viewer)
        assert views_2d.visual_ids == [build.visuals[k].id for k in ("xy", "xz", "yz")]
        assert view_3d.visual_ids == [build.vol_visual_id]

        # Only the 3D view's control lists the render planes.
        from PySide6.QtWidgets import QComboBox, QDockWidget

        assert "Render planes" not in names
        dock = next(
            d
            for d in handle.window.findChildren(QDockWidget)
            if d.windowTitle() == "Left"
        )
        (selector,) = [
            c for c in dock.findChildren(QComboBox) if c.findText(view_3d.label) >= 0
        ]
        selector.setCurrentIndex(selector.findText(view_3d.label))
        assert "Render planes" in _dock_control_names(handle.window, "Left")
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

        # Composite channels are drawn on the slice planes, so the overlays
        # use the plane profile.
        manager = build.overlays.transparency_manager
        assert manager.current_mode == "plane"

        # The overlay meshes are 3-D (z, y, x) and broadcast over the channel
        # axis, so they exist at every channel without per-channel updates.
        world = viewer.scenes["vol"].dims.world_coordinate_system
        for visual_id in build.overlays.axis_visual_ids:
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

        # The image controls on the left (the 2D views' control is shown
        # first); the overlay panel on the right.
        titles = _dock_titles(handle.window)
        assert "Left" in titles
        assert "Overlays" in titles
        names = _dock_control_names(handle.window, "Left")
        assert {f"Channel {i}" for i in range(handle.geometry.n_channels)} <= names
        assert "Composite channels" in names

        # Runtime switch to single mode: every panel follows, the channel axis
        # gets a slider, and the manager follows the single render mode.
        vol_id = build.vol_visual_id
        viewer.set_image_composite(build.visuals, False)
        assert not any(v.composite for v in build.visuals.values())
        for scene in viewer.scenes.values():
            assert 0 in scene.slider_axes
        assert manager.current_mode == "plane"
        vol = controller.get_visual_model(vol_id)
        assert vol.appearance.force_level is None

        # A volume render mode applies its profile and pins the volume to the
        # coarsest level.
        controller.update_single_appearance_field(vol_id, "render_mode", "iso")
        assert manager.current_mode == "iso"
        assert vol.single.opacity == pytest.approx(manager.current_profile.opacity)
        assert vol.appearance.transparency_mode == "weighted_blend"
        assert vol.appearance.force_level == len(vol.level_transforms) - 1
        assert vol.appearance.frustum_cull is False

        # An opacity edit through the image group is remembered per mode.
        viewer.update_image_single_field(build.visuals, "opacity", 0.6)
        assert manager.current_profile.opacity == pytest.approx(0.6)

        # Back to composite: blending returns to cellier's default, and the
        # remembered ISO profile is re-applied on the next switch to single.
        viewer.set_image_composite(build.visuals, True)
        assert manager.current_mode == "plane"
        assert vol.appearance.transparency_mode is None
        assert vol.appearance.force_level is None
        viewer.set_image_composite(build.visuals, False)
        assert vol.single.opacity == pytest.approx(0.6)
    finally:
        handle.close()


def test_the_2d_and_3d_views_are_edited_separately(qapp, tmp_path):
    """A contrast edit in the dock's 2D views control leaves the volume alone."""
    from PySide6.QtWidgets import QDockWidget
    from superqt import QLabeledDoubleRangeSlider

    from oz_viewer.data._blobs import make_example_zarr
    from oz_viewer.viewer._orthoviewer import _build_and_show_ortho_qt

    zarr_path = make_example_zarr(output_path=tmp_path / "blobs.ome.zarr")
    handle = _build_and_show_ortho_qt(f"file://{zarr_path}")
    try:
        build = handle.build
        vol_clim = tuple(build.visuals["vol"].single.clim)
        dock = next(
            d
            for d in handle.window.findChildren(QDockWidget)
            if d.windowTitle() == "Left"
        )
        # The selector starts on the 2D views; the 3D view's control is built
        # but not shown.
        (contrast,) = [
            s
            for s in dock.findChildren(QLabeledDoubleRangeSlider)
            if s.isVisibleTo(dock)
        ]

        contrast.setValue((10.0, 20.0))

        for key in ("xy", "xz", "yz"):
            assert tuple(build.visuals[key].single.clim) == pytest.approx((10.0, 20.0))
        assert tuple(build.visuals["vol"].single.clim) == vol_clim
    finally:
        handle.close()


def test_the_orientation_axes_keep_their_opacity_across_render_modes(qapp, tmp_path):
    """Only the blending of the orientation meshes follows the render mode."""
    from PySide6.QtWidgets import QCheckBox, QDockWidget

    from oz_viewer.data._blobs import make_example_zarr
    from oz_viewer.viewer._orthoviewer import _build_and_show_ortho_qt

    zarr_path = make_example_zarr(output_path=tmp_path / "blobs.ome.zarr")
    handle = _build_and_show_ortho_qt(f"file://{zarr_path}")
    try:
        build = handle.build
        controller = build.viewer.controller
        overlays = build.overlays
        mesh_ids = overlays.axis_visual_ids

        def appearances():
            return [controller.get_visual_model(v).appearance for v in mesh_ids]

        assert [a.opacity for a in appearances()] == pytest.approx([0.8] * 3)
        assert all(a.depth_test for a in appearances())

        vol = build.vol_visual_id
        controller.update_single_appearance_field(vol, "render_mode", "mip")
        assert overlays.transparency_manager.current_mode == "mip"
        assert [a.opacity for a in appearances()] == pytest.approx([0.8] * 3)
        assert not any(a.depth_test for a in appearances())

        controller.update_single_appearance_field(vol, "render_mode", "plane")
        assert overlays.transparency_manager.current_mode == "plane"
        assert all(a.depth_test for a in appearances())

        # The Qt-only panel toggles them.
        dock = next(
            d
            for d in handle.window.findChildren(QDockWidget)
            if d.windowTitle() == "Overlays"
        )
        (toggle,) = dock.findChildren(QCheckBox)
        toggle.setChecked(False)
        assert not any(a.visible for a in appearances())
    finally:
        handle.close()
