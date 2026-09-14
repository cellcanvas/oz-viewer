"""Qt-only control panels for the orthoviewer.

The convenience ``Layout``/renderer control docks cover the multichannel
``ChannelControls`` (cross-toolkit) but have no appearance dock for an
``OrthoViewer`` (their appearance builder needs a single-scene ``viewer.scene``).
oz-viewer therefore builds its single-channel appearance panel and the
render/opacity groups here, in Qt only, and bolts them onto the rendered
``QMainWindow``.  In the anywidget path these panels are simply omitted
(accepted feature loss -- see conversion plan risks); the canvases, slicing,
overlays, and multichannel controls still work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from oz_viewer.viewer._ortho_overlays import _TRANSPARENCY_MODES, _make_plane_colors
from oz_viewer.viewer._widgets import (
    _DEFAULT_COLORMAPS,
    _MultiVisualClimSlider,
    _MultiVisualColormapCombo,
    _MultiVisualLodBiasSlider,
)

if TYPE_CHECKING:
    from oz_viewer.viewer._geometry import _ViewerGeometry
    from oz_viewer.viewer._ortho_overlays import OrthoOverlays


@dataclass
class _OrthoControlsHandle:
    """The Qt controls panel widget plus the sub-widgets to close on teardown."""

    widget: object
    _closables: list = field(default_factory=list)

    def close(self) -> None:
        for w in self._closables:
            try:
                w.close()
            except (RuntimeError, ValueError):
                pass


def _plane_opacity_group(controller, overlays: OrthoOverlays, initial_opacity: float):
    """A 'Slice opacity' group driving the plane + orientation-mesh opacity."""
    from PySide6 import QtWidgets
    from PySide6.QtCore import Qt
    from superqt import QLabeledDoubleSlider

    box = QtWidgets.QGroupBox("Slice overlay opacity")
    layout = QtWidgets.QVBoxLayout(box)
    slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
    slider.setRange(0.0, 1.0)
    slider.setSingleStep(0.05)
    slider.setDecimals(2)
    slider.setValue(initial_opacity)

    plane_visual = overlays.plane_visual
    plane_store = overlays.plane_store
    axis_visual_ids = overlays.axis_visual_ids

    def _on_changed(value: float) -> None:
        controller.update_appearance_field(plane_visual.id, "opacity", value)
        plane_store.colors = _make_plane_colors(value)
        controller.reslice_visual(plane_visual.id)
        for vid in axis_visual_ids:
            controller.update_appearance_field(vid, "opacity", value)

    slider.valueChanged.connect(_on_changed)
    layout.addWidget(slider)
    return box


def build_sc_controls_panel(
    controller,
    visuals: dict,
    geometry: _ViewerGeometry,
    overlays: OrthoOverlays | None,
) -> _OrthoControlsHandle:
    """Build the single-channel appearance + overlay control panel (Qt only).

    Parameters
    ----------
    controller : cellier.controller.CellierController
        The orthoviewer's controller.
    visuals : dict
        Per-panel single-channel image visuals keyed ``xy``/``xz``/``yz``/``vol``.
    geometry : _ViewerGeometry
        OME-Zarr geometry (clim range, slider decimals).
    overlays : OrthoOverlays or None
        The attached 3D overlays; ``None`` disables the overlay control groups.
    """
    from cellier.gui.qt.visuals import (
        QtClimRangeSlider,
        QtColormapCombo,
        QtVolumeRenderControls,
    )
    from PySide6 import QtWidgets
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QCheckBox

    clim_range = geometry.clim_range
    decimals = geometry.slider_decimals
    closables: list = []

    panel = QtWidgets.QWidget()
    panel.setFixedWidth(300)
    root = QtWidgets.QVBoxLayout(panel)
    root.setAlignment(Qt.AlignmentFlag.AlignTop)

    # ── 2D rendering (drives all three 2D panels at once) ──────────────────
    group_2d = QtWidgets.QGroupBox("2D rendering")
    layout_2d = QtWidgets.QVBoxLayout(group_2d)

    ids_2d = [visuals[k].id for k in ("xy", "xz", "yz") if visuals.get(k) is not None]
    xy_app = visuals["xy"].appearance

    clim_2d = _MultiVisualClimSlider(
        ids_2d, clim_range=clim_range, initial_clim=xy_app.clim, decimals=decimals
    )
    controller.connect_widget(clim_2d, subscription_specs=clim_2d.subscription_specs())
    closables.append(clim_2d)
    _add_group(layout_2d, "Contrast limits", clim_2d.widget)

    cmap_2d = _MultiVisualColormapCombo(ids_2d, initial_colormap=xy_app.color_map)
    controller.connect_widget(cmap_2d, subscription_specs=cmap_2d.subscription_specs())
    closables.append(cmap_2d)
    _add_group(layout_2d, "Colormap", cmap_2d.widget)

    lod_2d = _MultiVisualLodBiasSlider(ids_2d, initial_lod_bias=xy_app.lod_bias)
    controller.connect_widget(lod_2d, subscription_specs=lod_2d.subscription_specs())
    closables.append(lod_2d)
    _add_group(layout_2d, "Fine-coarse tile bias", lod_2d.widget)

    root.addWidget(group_2d)

    # ── 3D rendering (vol panel) ───────────────────────────────────────────
    vol_visual = visuals.get("vol")
    if vol_visual is not None:
        vol_id = vol_visual.id
        vol_app = vol_visual.appearance
        group_3d = QtWidgets.QGroupBox("3D rendering")
        layout_3d = QtWidgets.QVBoxLayout(group_3d)

        clim_3d = QtClimRangeSlider(
            vol_id, clim_range=clim_range, initial_clim=vol_app.clim, decimals=decimals
        )
        controller.connect_widget(
            clim_3d, subscription_specs=clim_3d.subscription_specs()
        )
        closables.append(clim_3d)
        _add_group(layout_3d, "Contrast limits", clim_3d.widget)

        cmap_3d = QtColormapCombo(vol_id, initial_colormap=vol_app.color_map)
        cmap_3d.add_colormaps(_DEFAULT_COLORMAPS)
        controller.connect_widget(
            cmap_3d, subscription_specs=cmap_3d.subscription_specs()
        )
        closables.append(cmap_3d)
        _add_group(layout_3d, "Colormap", cmap_3d.widget)

        render_3d = QtVolumeRenderControls(
            vol_id,
            dtype_max=clim_range[1],
            initial_render_mode=vol_app.render_mode,
            initial_threshold=vol_app.iso_threshold,
            decimals=decimals,
        )
        controller.connect_widget(
            render_3d, subscription_specs=render_3d.subscription_specs()
        )
        closables.append(render_3d)
        _add_group(layout_3d, "Render mode", render_3d.widget)
        # The transparency manager observes the appearance model's render_mode
        # directly (see attach_ortho_overlays), so no widget signal is wired.

        if overlays is not None:
            orient_cb = QCheckBox("Show 3D orientation axes")
            orient_cb.setChecked(True)

            def _on_orient_toggled(checked: bool) -> None:
                for vid in overlays.axis_visual_ids:
                    controller.set_visual_visible(vid, checked)

            orient_cb.toggled.connect(_on_orient_toggled)
            layout_3d.addWidget(orient_cb)

        root.addWidget(group_3d)

    # ── Slice overlay opacity ──────────────────────────────────────────────
    if overlays is not None:
        from oz_viewer.viewer._ortho_overlays import _INITIAL_PLANE_OPACITY

        root.addWidget(
            _plane_opacity_group(controller, overlays, _INITIAL_PLANE_OPACITY)
        )

    root.addStretch()
    return _OrthoControlsHandle(widget=panel, _closables=closables)


def build_mc_controls_panel(
    controller,
    geometry: _ViewerGeometry,
    overlays: OrthoOverlays | None,
) -> _OrthoControlsHandle:
    """Build the multichannel 3D render + opacity panel (Qt only).

    The per-channel controls come from the cross-toolkit ``ChannelControls``
    dock; this panel adds the volume transparency/opacity and slice-opacity
    groups that have no convenience equivalent yet.
    """
    from PySide6 import QtWidgets
    from PySide6.QtCore import Qt
    from superqt import QLabeledDoubleSlider

    panel = QtWidgets.QWidget()
    panel.setFixedWidth(300)
    root = QtWidgets.QVBoxLayout(panel)
    root.setAlignment(Qt.AlignmentFlag.AlignTop)

    if overlays is None:
        root.addStretch()
        return _OrthoControlsHandle(widget=panel)

    tm = overlays.transparency_manager

    group = QtWidgets.QGroupBox("3D rendering")
    layout = QtWidgets.QVBoxLayout(group)

    mode_box = QtWidgets.QGroupBox("Transparency mode")
    mode_layout = QtWidgets.QVBoxLayout(mode_box)
    mode_combo = QtWidgets.QComboBox()
    for m in _TRANSPARENCY_MODES:
        mode_combo.addItem(m)
    mode_combo.setCurrentText(tm.current_profile.transparency_mode)
    mode_combo.currentTextChanged.connect(tm.update_transparency_mode)
    mode_layout.addWidget(mode_combo)
    layout.addWidget(mode_box)

    opacity_box = QtWidgets.QGroupBox("Volume opacity")
    opacity_layout = QtWidgets.QVBoxLayout(opacity_box)
    opacity_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
    opacity_slider.setRange(0.0, 1.0)
    opacity_slider.setSingleStep(0.05)
    opacity_slider.setDecimals(2)
    opacity_slider.setValue(tm.current_profile.opacity)
    opacity_slider.valueChanged.connect(tm.update_opacity)
    opacity_layout.addWidget(opacity_slider)
    layout.addWidget(opacity_box)

    root.addWidget(group)

    from oz_viewer.viewer._ortho_overlays import _INITIAL_PLANE_OPACITY

    root.addWidget(_plane_opacity_group(controller, overlays, _INITIAL_PLANE_OPACITY))
    root.addStretch()
    return _OrthoControlsHandle(widget=panel)


def _add_group(parent_layout, title: str, widget) -> None:
    from PySide6 import QtWidgets

    box = QtWidgets.QGroupBox(title)
    QtWidgets.QVBoxLayout(box).addWidget(widget)
    parent_layout.addWidget(box)
