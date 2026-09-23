"""Headless build tests for the convenience-based single-panel viewer.

These exercise the full Qt build path (``_build_and_show_viewer_qt``) offscreen:
the image's starting mode (composite when the store has a channel axis), the
runtime switch between modes, and the resulting control dock.  A
live GPU render is not exercised (there is no display in CI), only construction.
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

    A dock title alone only proves cellier added a dock; these are the names of
    the controls actually inside it.  ``build_viewer`` used to hand cellier a
    plain dict for ``controls=``, which produced an empty spec list and no dock
    at all -- and would have produced an empty dock had cellier docked it -- so
    the docks are checked by content, not just by existence.
    """
    from PySide6.QtWidgets import QCheckBox, QDockWidget, QGroupBox, QLabel

    dock = next(d for d in window.findChildren(QDockWidget) if d.windowTitle() == title)
    names = {g.title() for g in dock.findChildren(QGroupBox)}
    names |= {w.text() for w in dock.findChildren(QLabel)}
    names |= {w.text() for w in dock.findChildren(QCheckBox)}
    return {n for n in names if n}


def test_single_channel_viewer_build(qapp, tmp_path):
    """Blobs (z,y,x) -> single-channel viewer with the appearance dock."""
    from oz_viewer.data._blobs import make_example_zarr
    from oz_viewer.viewer._viewer import _build_and_show_viewer_qt

    zarr_path = make_example_zarr(output_path=tmp_path / "blobs.ome.zarr")
    handle = _build_and_show_viewer_qt(f"file://{zarr_path}")
    try:
        assert handle.geometry.channel_axis is None
        # The 2D/3D toggle lives in the canvas view's dims control, not a dock,
        # so the appearance panel is the only dock.
        assert _dock_titles(handle.window) == ["Left"]

        # The image fields resolved to controls, plus the bounding box cellier
        # adds to any configured appearance panel.
        names = _dock_control_names(handle.window, "Left")
        assert {
            "Contrast",
            "Colormap",
            "Render mode",
            "LOD bias",
            "Bounding box",
        } <= names
    finally:
        handle.window.close()


def test_multichannel_viewer_build(qapp, write_demo_ome):
    """Demo image (c,z,y,x) -> composite image, switchable to single at runtime."""
    from oz_viewer.viewer._viewer import _build_and_show_viewer_qt

    zarr_path = write_demo_ome("image")
    handle = _build_and_show_viewer_qt(f"file://{zarr_path}")
    try:
        assert handle.geometry.channel_axis == 0
        scene = handle.viewer.scene
        (visual,) = scene.visuals
        assert visual.composite is True
        assert set(visual.channels) == set(range(handle.geometry.n_channels))

        # A composited channel axis has no slider; every axis keeps a position.
        assert 0 not in scene.slider_axes
        assert 0 in scene.dims.selection.slice_indices

        # One image control holding the composite page and the mode switch.
        assert _dock_titles(handle.window) == ["Left"]
        names = _dock_control_names(handle.window, "Left")
        n_channels = handle.geometry.n_channels
        assert {f"Channel {i}" for i in range(n_channels)} <= names
        assert "Composite channels" in names

        # The single/composite choice is no longer build-time only: single
        # mode gives the channel axis a slider back.
        handle.viewer.controller.set_image_composite(visual.id, False)
        assert visual.composite is False
        assert 0 in scene.slider_axes
    finally:
        handle.window.close()


def test_composite_holds_at_most_cellier_default_channels(tmp_path, write_demo_ome):
    """A store with more channels than cellier's cap composites the first ones."""
    from cellier.visuals import MultiscaleImageVisual

    from oz_viewer.viewer._geometry import extract_viewer_geometry
    from oz_viewer.viewer._image import image_visual_kwargs

    _, geometry = extract_viewer_geometry(
        f"file://{write_demo_ome('image')}", print_summary=False
    )
    cap = MultiscaleImageVisual.model_fields["max_channels"].default
    kwargs = image_visual_kwargs(
        geometry._replace(n_channels=cap + 2, channel_labels=None),
        render_mode="mip",
        lod_bias=1.0,
    )
    assert kwargs["composite"] is True
    assert sorted(kwargs["channels"]) == list(range(cap))
    assert "max_channels" not in kwargs


@pytest.mark.parametrize("fixture_name", ["blobs", "demo"])
def test_controls_configs_are_config_objects(
    qapp, tmp_path, write_demo_ome, fixture_name
):
    """``controls=`` must be a config dataclass, never a plain dict.

    cellier stores whatever ``add_*`` was handed and dispatches on its type, so
    a dict is accepted, recorded, and then silently ignored at render time --
    no dock, no error, no warning.  This is the cheap, Qt-window-free guard for
    that: it fails on the config object, not on a missing widget.
    """
    from cellier.convenience import BaseControlsConfig

    from oz_viewer.viewer._geometry import extract_viewer_geometry
    from oz_viewer.viewer._viewer import build_viewer

    if fixture_name == "blobs":
        from oz_viewer.data._blobs import make_example_zarr

        zarr_path = make_example_zarr(output_path=tmp_path / "blobs.ome.zarr")
    else:
        zarr_path = write_demo_ome("image")

    data_store, geometry = extract_viewer_geometry(f"file://{zarr_path}")
    viewer = build_viewer(data_store, geometry, gui="qt")

    configs = list(viewer._controls_configs.values())
    assert configs, "no controls were configured for the visual"
    assert all(isinstance(c, BaseControlsConfig) for c in configs)
    # One image control config covers single and composite modes alike.
    assert [type(c).__name__ for c in configs] == ["MultiscaleImageControlsConfig"]
