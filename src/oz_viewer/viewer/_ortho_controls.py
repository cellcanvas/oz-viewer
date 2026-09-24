"""Qt-only overlay control panel for the orthoviewer.

The image itself is controlled by cellier's cross-toolkit
``AppearanceControls`` dock (one control for the 2D views, one for the 3D
view).
The orthoviewer's 3D overlays (the slice planes and the orientation gizmo) are
plain meshes oz-viewer adds to the ``vol`` scene only, and the convenience
docks have no public way to configure controls for them, so their panel is
built here, in Qt only, and bolted onto the rendered ``QMainWindow``.  In the
anywidget path this panel is omitted; the overlays still render.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from oz_viewer.viewer._ortho_overlays import _INITIAL_PLANE_OPACITY, _make_plane_colors

if TYPE_CHECKING:
    from oz_viewer.viewer._ortho_overlays import OrthoOverlays


@dataclass
class _OrthoControlsHandle:
    """The Qt controls panel widget."""

    widget: object

    def close(self) -> None:
        """Nothing to unsubscribe: the panel only writes to the controller."""


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
    manager = overlays.transparency_manager

    def _on_changed(value: float) -> None:
        # The manager owns the meshes' opacity so it survives render-mode
        # switches; the plane's face colours carry it too.
        manager.overlay_opacity = value
        plane_store.colors = _make_plane_colors(value)
        controller.reslice_visual(plane_visual.id)

    slider.valueChanged.connect(_on_changed)
    layout.addWidget(slider)
    return box


def build_overlay_controls_panel(
    controller,
    overlays: OrthoOverlays,
) -> _OrthoControlsHandle:
    """Build the orientation-axes toggle + slice-opacity panel (Qt only).

    Parameters
    ----------
    controller : cellier.controller.CellierController
        The orthoviewer's controller.
    overlays : OrthoOverlays
        The attached 3D overlays.
    """
    from PySide6 import QtWidgets
    from PySide6.QtCore import Qt

    panel = QtWidgets.QWidget()
    panel.setFixedWidth(300)
    root = QtWidgets.QVBoxLayout(panel)
    root.setAlignment(Qt.AlignmentFlag.AlignTop)

    orient_cb = QtWidgets.QCheckBox("Show 3D orientation axes")
    orient_cb.setChecked(True)

    def _on_orient_toggled(checked: bool) -> None:
        for vid in overlays.axis_visual_ids:
            controller.set_visual_visible(vid, checked)

    orient_cb.toggled.connect(_on_orient_toggled)
    root.addWidget(orient_cb)

    root.addWidget(_plane_opacity_group(controller, overlays, _INITIAL_PLANE_OPACITY))
    root.addStretch()
    return _OrthoControlsHandle(widget=panel)
