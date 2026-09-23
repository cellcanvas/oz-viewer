"""Toolkit-independent 3D overlays for the convenience-based orthoviewer.

The orthoviewer's three 3D overlays -- the volume transparency manager, the
slice-plane meshes, and the orientation gizmo -- all speak only *controller +
scene id + visual id + mesh store*, never Qt.  This module holds them (ported
verbatim from the old hand-built orthoviewer) plus :func:`attach_ortho_overlays`,
which wires them on top of a :class:`cellier.convenience.OrthoViewer`.

Because everything is model/controller-level, these overlays render in both Qt
and anywidget with no per-toolkit code (conversion plan section 4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np

if TYPE_CHECKING:
    from cellier.convenience import OrthoViewer

    from oz_viewer.viewer._geometry import _ViewerGeometry

# Colors matching the 2D panel slider gradients (blue=XY, green=XZ, orange=YZ).
_PLANE_COLOR_XY = (0.33, 0.33, 1.00)  # blue
_PLANE_COLOR_XZ = (0.23, 0.67, 0.23)  # green
_PLANE_COLOR_YZ = (0.80, 0.40, 0.00)  # orange

_AXIS_3D_LENGTH_FRACTION: float = 0.12
_AXIS_3D_CUBE_SIDE_FRACTION: float = 0.024
_AXIS_3D_PRISM_CROSS_SECTION_FRACTION: float = 0.020
_AXIS_3D_CUBE_COLOUR: tuple[float, float, float, float] = (0.75, 0.75, 0.75, 1.0)
_N_FACES_PER_BOX: int = 12

_INITIAL_PLANE_OPACITY: float = 1.0


# ---------------------------------------------------------------------------
# Volume transparency profiles
# ---------------------------------------------------------------------------


@dataclass
class _VolTransparencyProfile:
    transparency_mode: str
    opacity: float


_ISO_DEFAULT_TRANSPARENCY = _VolTransparencyProfile("weighted_blend", 0.3)
_MIP_DEFAULT_TRANSPARENCY = _VolTransparencyProfile("weighted_blend", 1.0)


@dataclass
class _VisualRenderProfile:
    render_order: int
    depth_test: bool
    depth_write: bool
    transparency_mode: str
    opacity: float


_ISO_PLANE_PROFILE = _VisualRenderProfile(
    render_order=0,
    depth_test=True,
    depth_write=True,
    transparency_mode="blend",
    opacity=1.0,
)
_MIP_PLANE_PROFILE = _VisualRenderProfile(
    render_order=1,
    depth_test=False,
    depth_write=True,
    transparency_mode="weighted_blend",
    opacity=0.99,
)
_ISO_AXES_PROFILE = _VisualRenderProfile(
    render_order=1,
    depth_test=True,
    depth_write=True,
    transparency_mode="blend",
    opacity=1.0,
)
_MIP_AXES_PROFILE = _VisualRenderProfile(
    render_order=2,
    depth_test=False,
    depth_write=False,
    transparency_mode="blend",
    opacity=1.0,
)


# ---------------------------------------------------------------------------
# Volume transparency manager
# ---------------------------------------------------------------------------


def _is_mip(render_mode: object) -> bool:
    """Whether *render_mode* is a maximum-intensity variant."""
    return str(render_mode) in ("mip", "attenuated_mip")


class _VolTransparencyManager:
    """Keeps the volume and the overlay meshes blending per render mode.

    The effective render mode is the ``vol`` image's: its single appearance's
    ``render_mode`` in single mode; in composite mode ``"iso"`` if any held
    channel renders an isosurface, otherwise ``"mip"``.  It is re-read from
    the model whenever a render mode or the image's single/composite mode
    changes, whatever the origin (Qt, anywidget, programmatic).

    In single mode the volume's opacity and blending follow a per-render-mode
    profile, and edits made through the image control are remembered in the
    current profile.  A composite hands blending back to cellier (its default
    follows the mode) and keeps the channels' own opacities; only the meshes
    follow the render mode there.

    Subscribes through ``controller.connect_widget`` (the widget contract:
    ``_id``, ``changed``, ``closed``), so the bus rewires it if the image's
    ``single`` or ``channels`` models are replaced; :meth:`close` unsubscribes.
    """

    from psygnal import Signal

    #: Required by ``connect_widget``; the manager writes through the
    #: controller directly, so this never emits.
    changed = Signal(object)
    closed = Signal()

    def __init__(
        self,
        controller,
        vol_visual_id,
        *,
        plane_visual_id=None,
        axis_visual_ids: list | None = None,
    ) -> None:
        self._id = uuid4()
        self._controller = controller
        self._vol_visual_id = vol_visual_id
        self._plane_visual_id = plane_visual_id
        self._axis_visual_ids: list = axis_visual_ids or []
        self._vol_profiles: dict[str, _VolTransparencyProfile] = {
            "iso": _VolTransparencyProfile(
                transparency_mode=_ISO_DEFAULT_TRANSPARENCY.transparency_mode,
                opacity=_ISO_DEFAULT_TRANSPARENCY.opacity,
            ),
            "mip": _VolTransparencyProfile(
                transparency_mode=_MIP_DEFAULT_TRANSPARENCY.transparency_mode,
                opacity=_MIP_DEFAULT_TRANSPARENCY.opacity,
            ),
        }
        self._plane_profiles: dict[str, _VisualRenderProfile] = {
            "iso": _ISO_PLANE_PROFILE,
            "mip": _MIP_PLANE_PROFILE,
        }
        self._axes_profiles: dict[str, _VisualRenderProfile] = {
            "iso": _ISO_AXES_PROFILE,
            "mip": _MIP_AXES_PROFILE,
        }
        self._current_mode = self._read_mode()
        controller.connect_widget(self, subscription_specs=self._subscription_specs())

    @property
    def current_mode(self) -> str:
        return self._current_mode

    @property
    def current_profile(self) -> _VolTransparencyProfile:
        return self._vol_profiles[self._current_mode]

    def _vol_visual(self):
        return self._controller.get_visual_model(self._vol_visual_id)

    def _read_mode(self) -> str:
        visual = self._vol_visual()
        if visual.composite:
            modes = [ch.render_mode for ch in visual.channels.values()]
            return "mip" if all(_is_mip(m) for m in modes) else "iso"
        return "mip" if _is_mip(visual.single.render_mode) else "iso"

    def _subscription_specs(self) -> list:
        from cellier.events import (
            AppearanceChangedEvent,
            ChannelAppearanceChangedEvent,
            ImageCompositeChangedEvent,
            SingleAppearanceChangedEvent,
            SubscriptionSpec,
        )

        vid = self._vol_visual_id
        return [
            SubscriptionSpec(SingleAppearanceChangedEvent, self._on_single, vid),
            SubscriptionSpec(ChannelAppearanceChangedEvent, self._on_channel, vid),
            SubscriptionSpec(ImageCompositeChangedEvent, self._on_composite, vid),
            SubscriptionSpec(AppearanceChangedEvent, self._on_appearance, vid),
        ]

    # -- bus handlers --------------------------------------------------------

    def _on_single(self, event) -> None:
        # field_name is None when the whole single model was replaced.
        if event.field_name in (None, "render_mode"):
            self._refresh_mode()
        elif (
            event.field_name == "opacity"
            and event.source_id != self._id
            and not self._vol_visual().composite
        ):
            self.current_profile.opacity = float(event.new_value)

    def _on_channel(self, event) -> None:
        if event.field_name in (None, "render_mode"):
            self._refresh_mode()

    def _on_composite(self, event) -> None:
        # Always re-apply: entering single mode restores the volume profile,
        # entering composite mode hands blending back to cellier.
        self._current_mode = self._read_mode()
        self.apply()

    def _on_appearance(self, event) -> None:
        if (
            event.field_name == "transparency_mode"
            and event.source_id != self._id
            and event.new_value is not None
            and not self._vol_visual().composite
        ):
            self.current_profile.transparency_mode = str(event.new_value)

    # -- application ---------------------------------------------------------

    def _refresh_mode(self) -> None:
        new_mode = self._read_mode()
        if new_mode != self._current_mode:
            self._current_mode = new_mode
            self.apply()

    def _apply_profile_to_mesh(self, visual_id, profile: _VisualRenderProfile) -> None:
        if visual_id is None:
            return
        c = self._controller
        c.update_appearance_field(visual_id, "render_order", profile.render_order)
        c.update_appearance_field(visual_id, "depth_test", profile.depth_test)
        c.update_appearance_field(visual_id, "depth_write", profile.depth_write)
        c.update_appearance_field(
            visual_id, "transparency_mode", profile.transparency_mode
        )
        c.update_appearance_field(visual_id, "opacity", profile.opacity)

    def _apply_vol_profile(self) -> None:
        """Apply the current volume profile, or hand blending back in composite.

        Writes are stamped with the manager's id so the bus handlers can tell
        them apart from user edits.
        """
        c = self._controller
        vid = self._vol_visual_id
        if self._vol_visual().composite:
            # cellier's defaults: blending follows the mode (None), and the
            # composite manages depth between its channel volumes itself.
            c.update_appearance_field(
                vid, "transparency_mode", None, source_id=self._id
            )
            c.update_appearance_field(vid, "depth_write", True, source_id=self._id)
            return
        vol = self.current_profile
        depth_write = vol.transparency_mode == "weighted_solid"
        c.update_appearance_field(
            vid, "transparency_mode", vol.transparency_mode, source_id=self._id
        )
        c.update_appearance_field(vid, "depth_write", depth_write, source_id=self._id)
        c.update_single_appearance_field(
            vid, "opacity", vol.opacity, source_id=self._id
        )

    def apply(self) -> None:
        self._apply_vol_profile()
        plane_profile = self._plane_profiles[self._current_mode]
        self._apply_profile_to_mesh(self._plane_visual_id, plane_profile)
        axes_profile = self._axes_profiles[self._current_mode]
        for vid in self._axis_visual_ids:
            self._apply_profile_to_mesh(vid, axes_profile)

    def close(self) -> None:
        """Unsubscribe from the controller bus."""
        self.closed.emit()


# ---------------------------------------------------------------------------
# Geometry helpers (pure numpy + MeshMemoryStore, no Qt, no viewer coupling)
# ---------------------------------------------------------------------------


def _box_faces_geometry(
    centre_zyx: np.ndarray,
    half_extents_zyx: np.ndarray,
    vertex_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    cz, cy, cx = float(centre_zyx[0]), float(centre_zyx[1]), float(centre_zyx[2])
    hz, hy, hx = (
        float(half_extents_zyx[0]),
        float(half_extents_zyx[1]),
        float(half_extents_zyx[2]),
    )
    z0, z1 = cz - hz, cz + hz
    y0, y1 = cy - hy, cy + hy
    x0, x1 = cx - hx, cx + hx

    positions = np.array(
        [
            [z0, y0, x0],
            [z0, y1, x0],
            [z0, y1, x1],
            [z0, y0, x1],  # Face 0: -Z
            [z1, y0, x0],
            [z1, y0, x1],
            [z1, y1, x1],
            [z1, y1, x0],  # Face 1: +Z
            [z0, y0, x0],
            [z0, y0, x1],
            [z1, y0, x1],
            [z1, y0, x0],  # Face 2: -Y
            [z0, y1, x0],
            [z1, y1, x0],
            [z1, y1, x1],
            [z0, y1, x1],  # Face 3: +Y
            [z0, y0, x0],
            [z1, y0, x0],
            [z1, y1, x0],
            [z0, y1, x0],  # Face 4: -X
            [z0, y0, x1],
            [z0, y1, x1],
            [z1, y1, x1],
            [z1, y0, x1],  # Face 5: +X
        ],
        dtype=np.float32,
    )

    base_indices = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [8, 9, 10],
            [8, 10, 11],
            [12, 13, 14],
            [12, 14, 15],
            [16, 17, 18],
            [16, 18, 19],
            [20, 21, 22],
            [20, 22, 23],
        ],
        dtype=np.int32,
    )
    return positions, base_indices + vertex_offset


def _make_axis_set_geometry(
    axis_a: int,
    axis_b: int,
    axis_length: float,
    cube_side: float,
    prism_cross_section: float,
) -> tuple[np.ndarray, np.ndarray]:
    half_cube = cube_side / 2.0
    half_length = axis_length / 2.0
    half_cross = prism_cross_section / 2.0

    origin = np.zeros(3, dtype=np.float64)
    cube_half_extents = np.full(3, half_cube, dtype=np.float64)
    cube_positions, cube_indices = _box_faces_geometry(origin, cube_half_extents, 0)

    centre_a = np.zeros(3, dtype=np.float64)
    centre_a[axis_a] = half_cube + half_length
    half_extents_a = np.full(3, half_cross, dtype=np.float64)
    half_extents_a[axis_a] = half_length
    prism_a_positions, prism_a_indices = _box_faces_geometry(
        centre_a, half_extents_a, 24
    )

    centre_b = np.zeros(3, dtype=np.float64)
    centre_b[axis_b] = half_cube + half_length
    half_extents_b = np.full(3, half_cross, dtype=np.float64)
    half_extents_b[axis_b] = half_length
    prism_b_positions, prism_b_indices = _box_faces_geometry(
        centre_b, half_extents_b, 48
    )

    positions = np.concatenate([cube_positions, prism_a_positions, prism_b_positions])
    indices = np.concatenate([cube_indices, prism_a_indices, prism_b_indices])
    return positions, indices


def _make_axis_set_face_colors(
    axis_a_color_rgb: tuple[float, float, float],
    axis_b_color_rgb: tuple[float, float, float],
) -> np.ndarray:
    cube_color = np.array(_AXIS_3D_CUBE_COLOUR, dtype=np.float32)
    color_a = np.array([*axis_a_color_rgb, 1.0], dtype=np.float32)
    color_b = np.array([*axis_b_color_rgb, 1.0], dtype=np.float32)
    return np.concatenate(
        [
            np.tile(cube_color, (_N_FACES_PER_BOX, 1)),
            np.tile(color_a, (_N_FACES_PER_BOX, 1)),
            np.tile(color_b, (_N_FACES_PER_BOX, 1)),
        ]
    )


def _spatial_mesh_store(
    positions_zyx: np.ndarray,
    indices: np.ndarray,
    colors: np.ndarray,
    name: str,
    *,
    world,
    spatial_axes: tuple[int, int, int],
):
    """A face-coloured mesh store over the three spatial world axes only.

    Its axes copy the world's spatial axes (name, type, unit) with fresh ids,
    so :func:`_mesh_to_world` can map them one to one.  The coordinate system
    is built here rather than derived when the mesh is added, because the
    transform that places the mesh is built against it first.
    """
    from cellier.data.mesh import MeshMemoryStore
    from cellier.transform import Axis, DataCoordinateSystem

    zyx = [world.axes[axis] for axis in spatial_axes]
    system = DataCoordinateSystem(
        name=f"{name}_data",
        axes=tuple(
            Axis(name=axis.name, axis_type=axis.axis_type, unit=axis.unit)
            for axis in zyx
        ),
        datastore_id=uuid4(),
    )
    return MeshMemoryStore(
        positions=positions_zyx,
        indices=indices,
        colors=colors,
        colors_layout="face",
        name=name,
        data_coordinate_systems=[system],
    )


def _mesh_to_world(
    store,
    world,
    spatial_axes: tuple[int, int, int],
    translation_zyx=(0.0, 0.0, 0.0),
):
    """Place a 3-D ``(z, y, x)`` overlay mesh in the world.

    The mesh has only the three spatial axes; every other world axis (channel,
    time, ...) is broadcast, so the overlay exists at every position along
    them and needs no update when their sliders move.
    """
    from cellier.transform import AffineTransform

    data = store.data_coordinate_system
    return AffineTransform.from_axis_map(
        data,
        world,
        axis_map={
            data.axes[i].id: world.axes[axis].id for i, axis in enumerate(spatial_axes)
        },
        translation={
            data.axes[i].id: float(value) for i, value in enumerate(translation_zyx)
        },
        broadcast_output_axes=[
            world.axes[axis].id
            for axis in range(world.ndim)
            if axis not in spatial_axes
        ],
        name="overlay_to_world",
    )


def _make_axis_meshes(
    controller,
    vol_scene,
    initial_centre_zyx: np.ndarray,
    world_min_extent: float,
    *,
    spatial_axes: tuple[int, int, int],
) -> tuple:
    from cellier.visuals import MeshFlatAppearance

    world = vol_scene.dims.world_coordinate_system
    color_z = _PLANE_COLOR_XY
    color_y = _PLANE_COLOR_XZ
    color_x = _PLANE_COLOR_YZ

    view_specifications = [
        ("xy_axis_set", 1, 2, color_y, color_x),
        ("xz_axis_set", 0, 2, color_z, color_x),
        ("yz_axis_set", 0, 1, color_z, color_y),
    ]

    axis_length = _AXIS_3D_LENGTH_FRACTION * world_min_extent
    cube_side = _AXIS_3D_CUBE_SIDE_FRACTION * world_min_extent
    prism_cross_section = _AXIS_3D_PRISM_CROSS_SECTION_FRACTION * world_min_extent

    axis_stores = []
    axis_visuals = []
    for view_name, axis_a, axis_b, color_a, color_b in view_specifications:
        positions, indices = _make_axis_set_geometry(
            axis_a, axis_b, axis_length, cube_side, prism_cross_section
        )
        face_colors = _make_axis_set_face_colors(color_a, color_b)
        store = _spatial_mesh_store(
            positions,
            indices,
            face_colors,
            view_name,
            world=world,
            spatial_axes=spatial_axes,
        )
        initial_transform = _mesh_to_world(
            store, world, spatial_axes, initial_centre_zyx
        )
        appearance = MeshFlatAppearance(
            color_mode="face",
            side="both",
            opacity=1.0,
            render_order=1,
            depth_test=True,
            depth_write=True,
            depth_compare="<=",
            transparency_mode="blend",
        )
        visual = controller.add_mesh(
            data=store,
            scene_id=vol_scene.id,
            appearance=appearance,
            name=view_name,
            transform=initial_transform,
        )
        axis_stores.append(store)
        axis_visuals.append(visual)

    return tuple(axis_visuals), tuple(axis_stores)


def _make_plane_positions(
    z_world: float,
    y_world: float,
    x_world: float,
    world_max_zyx: np.ndarray,
) -> np.ndarray:
    """``(12, 3)`` ``(z, y, x)`` vertices of the three slice planes."""
    wz = float(world_max_zyx[0])
    wy = float(world_max_zyx[1])
    wx = float(world_max_zyx[2])
    z, y, x = float(z_world), float(y_world), float(x_world)

    positions = np.zeros((12, 3), dtype=np.float32)

    # XY plane (constant Z = z)
    positions[0:4, 0] = z
    positions[0:4, 1] = [0.0, wy, wy, 0.0]
    positions[0:4, 2] = [0.0, 0.0, wx, wx]

    # XZ plane (constant Y = y)
    positions[4:8, 0] = [0.0, wz, wz, 0.0]
    positions[4:8, 1] = y
    positions[4:8, 2] = [0.0, 0.0, wx, wx]

    # YZ plane (constant X = x)
    positions[8:12, 0] = [0.0, wz, wz, 0.0]
    positions[8:12, 1] = [0.0, 0.0, wy, wy]
    positions[8:12, 2] = x

    return positions


def _make_plane_colors(opacity: float) -> np.ndarray:
    a = float(opacity)
    return np.array(
        [
            [*_PLANE_COLOR_XY, a],
            [*_PLANE_COLOR_XY, a],
            [*_PLANE_COLOR_XZ, a],
            [*_PLANE_COLOR_XZ, a],
            [*_PLANE_COLOR_YZ, a],
            [*_PLANE_COLOR_YZ, a],
        ],
        dtype=np.float32,
    )


def _make_plane_mesh(
    controller,
    vol_scene,
    z_world: float,
    y_world: float,
    x_world: float,
    world_max_zyx: np.ndarray,
    initial_opacity: float = 0.4,
    *,
    spatial_axes: tuple[int, int, int],
):
    from cellier.visuals import MeshFlatAppearance

    world = vol_scene.dims.world_coordinate_system
    positions = _make_plane_positions(z_world, y_world, x_world, world_max_zyx)
    colors = _make_plane_colors(initial_opacity)
    indices = np.array(
        [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [8, 9, 10], [8, 10, 11]],
        dtype=np.int32,
    )

    store = _spatial_mesh_store(
        positions,
        indices,
        colors,
        "slice_planes",
        world=world,
        spatial_axes=spatial_axes,
    )
    appearance = MeshFlatAppearance(
        color_mode="face", side="both", opacity=initial_opacity, wireframe=False
    )
    visual = controller.add_mesh(
        data=store,
        scene_id=vol_scene.id,
        appearance=appearance,
        name="slice_planes",
        transform=_mesh_to_world(store, world, spatial_axes),
    )
    return store, visual


class _PlaneUpdater:
    def __init__(
        self,
        controller,
        plane_store,
        plane_visual,
        world_max_zyx,
        *,
        spatial_axes: tuple[int, int, int],
    ) -> None:
        self._id = uuid4()
        self._controller = controller
        self._plane_store = plane_store
        self._plane_visual = plane_visual
        self._world_max_zyx = world_max_zyx
        self._spatial_axes = spatial_axes

        # Read initial slice positions from the (z, y, x) positions array.
        positions = plane_store.positions
        self._z_world = float(positions[0, 0])  # XY plane vertex 0: Z
        self._y_world = float(positions[4, 1])  # XZ plane vertex 4: Y
        self._x_world = float(positions[8, 2])  # YZ plane vertex 8: X

    def _update(self) -> None:
        self._plane_store.positions = _make_plane_positions(
            self._z_world, self._y_world, self._x_world, self._world_max_zyx
        )
        self._controller.reslice_visual(self._plane_visual.id)

    def on_xy_dims_changed(self, event) -> None:
        sz0 = self._spatial_axes[0]
        if sz0 in event.slice_indices:
            self._z_world = float(event.slice_indices[sz0])
            self._update()

    def on_xz_dims_changed(self, event) -> None:
        sz1 = self._spatial_axes[1]
        if sz1 in event.slice_indices:
            self._y_world = float(event.slice_indices[sz1])
            self._update()

    def on_yz_dims_changed(self, event) -> None:
        sz2 = self._spatial_axes[2]
        if sz2 in event.slice_indices:
            self._x_world = float(event.slice_indices[sz2])
            self._update()


class _OrientationUpdater:
    def __init__(
        self,
        controller,
        xy_axis_visual,
        xz_axis_visual,
        yz_axis_visual,
        world_max_zyx: np.ndarray,
        *,
        axis_stores: tuple,
        world,
        spatial_axes: tuple[int, int, int],
    ):
        self._id = uuid4()
        self._controller = controller
        self._xy_axis_visual_id = xy_axis_visual.id
        self._xz_axis_visual_id = xz_axis_visual.id
        self._yz_axis_visual_id = yz_axis_visual.id
        self._axis_stores = axis_stores
        self._world = world
        self._spatial_axes = spatial_axes

        mid = world_max_zyx / 2.0
        self._z_world = float(mid[0])
        self._y_world = float(mid[1])
        self._x_world = float(mid[2])
        # (z, y, x) centre vectors, one per 2D panel.
        self._xy_centre = np.array(
            [self._z_world, self._y_world, self._x_world], dtype=np.float64
        )
        self._xz_centre = self._xy_centre.copy()
        self._yz_centre = self._xy_centre.copy()

    def _update_3d(self) -> None:
        for visual_id, store, centre_zyx in zip(
            (self._xy_axis_visual_id, self._xz_axis_visual_id, self._yz_axis_visual_id),
            self._axis_stores,
            (self._xy_centre, self._xz_centre, self._yz_centre),
            strict=True,
        ):
            self._controller.set_visual_transform(
                visual_id,
                _mesh_to_world(store, self._world, self._spatial_axes, centre_zyx),
                reslice=False,
            )

    def on_xy_camera_changed(self, event) -> None:
        p = event.camera_state.position
        # p[0] → X world, p[1] → Y world (canvas horizontal/vertical convention)
        self._xy_centre = np.array([self._z_world, p[1], p[0]], dtype=np.float64)
        self._update_3d()

    def on_xz_camera_changed(self, event) -> None:
        p = event.camera_state.position
        # p[0] → X world, p[1] → Z world
        self._xz_centre = np.array([p[1], self._y_world, p[0]], dtype=np.float64)
        self._update_3d()

    def on_yz_camera_changed(self, event) -> None:
        p = event.camera_state.position
        # p[0] → Y world, p[1] → Z world
        self._yz_centre = np.array([p[1], p[0], self._x_world], dtype=np.float64)
        self._update_3d()

    def on_xy_dims_changed(self, event) -> None:
        sz0 = self._spatial_axes[0]
        if sz0 in event.slice_indices:
            self._z_world = float(event.slice_indices[sz0])
            self._xy_centre[0] = self._z_world
        self._update_3d()

    def on_xz_dims_changed(self, event) -> None:
        sz1 = self._spatial_axes[1]
        if sz1 in event.slice_indices:
            self._y_world = float(event.slice_indices[sz1])
            self._xz_centre[1] = self._y_world
        self._update_3d()

    def on_yz_dims_changed(self, event) -> None:
        sz2 = self._spatial_axes[2]
        if sz2 in event.slice_indices:
            self._x_world = float(event.slice_indices[sz2])
            self._yz_centre[2] = self._x_world
        self._update_3d()


# ---------------------------------------------------------------------------
# Overlay attachment on top of a convenience OrthoViewer
# ---------------------------------------------------------------------------


@dataclass
class OrthoOverlays:
    """Live handle bundling the orthoviewer's 3D overlays.

    Returned by :func:`attach_ortho_overlays`.  The launcher keeps a reference
    so the meshes/updaters are not garbage collected; :meth:`close`
    unsubscribes every owner from the controller bus.
    """

    transparency_manager: _VolTransparencyManager
    plane_visual: object
    plane_store: object
    axis_visual_ids: list
    _plane_updater: _PlaneUpdater
    _orientation_updater: _OrientationUpdater
    _owner_ids: list
    _controller: object

    def close(self) -> None:
        """Unsubscribe every overlay owner, the transparency manager included."""
        self.transparency_manager.close()
        for owner_id in self._owner_ids:
            self._controller.unsubscribe_owner(owner_id)


def attach_ortho_overlays(
    viewer: OrthoViewer,
    geometry: _ViewerGeometry,
    *,
    vol_visual_id,
) -> OrthoOverlays:
    """Attach the slice-plane, orientation, and transparency overlays.

    Builds the plane + orientation-gizmo meshes in the ``vol`` scene only,
    wires the updaters to the 2D panels' dims/camera events, and installs the
    volume transparency manager.  Everything is model/controller-level, so the
    overlays render identically under Qt and anywidget.

    Parameters
    ----------
    viewer : cellier.convenience.OrthoViewer
        The orthoviewer whose ``controller`` and ``scenes`` drive the overlays.
    geometry : _ViewerGeometry
        OME-Zarr geometry (spatial axes, world extents, channel axis).
    vol_visual_id :
        The image visual id in the ``vol`` scene the transparency manager
        follows; it tracks the image's single/composite mode live.

    Returns
    -------
    OrthoOverlays
    """
    controller = viewer.controller
    scenes = viewer.scenes
    vol_scene = scenes["vol"]

    spatial_axes = geometry.spatial_axes
    world_max_zyx = geometry.world_max_spatial

    center = geometry.center_slice_indices()
    z_mid = center[spatial_axes[0]]
    y_mid = center[spatial_axes[1]]
    x_mid = center[spatial_axes[2]]
    initial_centre_zyx = np.array([z_mid, y_mid, x_mid], dtype=np.float64)

    # --- orientation-gizmo axis meshes (vol scene only) ---
    (
        (xy_axis_visual, xz_axis_visual, yz_axis_visual),
        (xy_axis_store, xz_axis_store, yz_axis_store),
    ) = _make_axis_meshes(
        controller=controller,
        vol_scene=vol_scene,
        initial_centre_zyx=initial_centre_zyx,
        world_min_extent=float(world_max_zyx.min()),
        spatial_axes=spatial_axes,
    )
    axis_visual_ids = [xy_axis_visual.id, xz_axis_visual.id, yz_axis_visual.id]

    # --- slice-plane mesh (vol scene only) ---
    plane_store, plane_visual = _make_plane_mesh(
        controller,
        vol_scene,
        z_mid,
        y_mid,
        x_mid,
        world_max_zyx,
        initial_opacity=_INITIAL_PLANE_OPACITY,
        spatial_axes=spatial_axes,
    )

    plane_updater = _PlaneUpdater(
        controller=controller,
        plane_store=plane_store,
        plane_visual=plane_visual,
        world_max_zyx=world_max_zyx,
        spatial_axes=spatial_axes,
    )
    controller.on_dims_changed(
        scenes["xy"].id, plane_updater.on_xy_dims_changed, owner_id=plane_updater._id
    )
    controller.on_dims_changed(
        scenes["xz"].id, plane_updater.on_xz_dims_changed, owner_id=plane_updater._id
    )
    controller.on_dims_changed(
        scenes["yz"].id, plane_updater.on_yz_dims_changed, owner_id=plane_updater._id
    )

    # --- volume transparency manager ---
    # Follows the vol image's render mode and single/composite mode from any
    # origin (Qt, anywidget, programmatic) through the controller bus.
    transparency_manager = _VolTransparencyManager(
        controller,
        vol_visual_id,
        plane_visual_id=plane_visual.id,
        axis_visual_ids=axis_visual_ids,
    )
    transparency_manager.apply()

    # --- orientation updater (camera + dims driven) ---
    orient_updater = _OrientationUpdater(
        controller=controller,
        xy_axis_visual=xy_axis_visual,
        xz_axis_visual=xz_axis_visual,
        yz_axis_visual=yz_axis_visual,
        world_max_zyx=world_max_zyx,
        axis_stores=(xy_axis_store, xz_axis_store, yz_axis_store),
        world=vol_scene.dims.world_coordinate_system,
        spatial_axes=spatial_axes,
    )
    orient_owner = orient_updater._id
    controller.on_camera_changed(
        scenes["xy"].id, orient_updater.on_xy_camera_changed, owner_id=orient_owner
    )
    controller.on_camera_changed(
        scenes["xz"].id, orient_updater.on_xz_camera_changed, owner_id=orient_owner
    )
    controller.on_camera_changed(
        scenes["yz"].id, orient_updater.on_yz_camera_changed, owner_id=orient_owner
    )
    controller.on_dims_changed(
        scenes["xy"].id, orient_updater.on_xy_dims_changed, owner_id=orient_updater._id
    )
    controller.on_dims_changed(
        scenes["xz"].id, orient_updater.on_xz_dims_changed, owner_id=orient_updater._id
    )
    controller.on_dims_changed(
        scenes["yz"].id, orient_updater.on_yz_dims_changed, owner_id=orient_updater._id
    )

    # No channel following is needed: the overlay meshes are broadcast over
    # every non-spatial world axis (see _mesh_to_world), so moving a channel or
    # time slider leaves them in place.
    owner_ids = [plane_updater._id, orient_updater._id]

    # --- seed the orientation gizmo from post-fit camera state ---
    # The initial camera fit is deferred to the canvas first frame (fit="ready"),
    # so seed once every panel is ready rather than synchronously at build time.
    def _seed_cameras() -> None:
        from cellier.events import CameraChangedEvent

        for key, handler in (
            ("xy", orient_updater.on_xy_camera_changed),
            ("xz", orient_updater.on_xz_camera_changed),
            ("yz", orient_updater.on_yz_camera_changed),
        ):
            scene_id = scenes[key].id
            canvas_ids = controller.get_canvas_ids(scene_id)
            if not canvas_ids:
                continue
            canvas_view = controller.get_canvas_view(canvas_ids[0])
            handler(
                CameraChangedEvent(
                    source_id=canvas_view.canvas_id,
                    scene_id=scene_id,
                    camera_state=canvas_view.capture_camera_state(),
                )
            )

    viewer.on_ready(_seed_cameras)

    return OrthoOverlays(
        transparency_manager=transparency_manager,
        plane_visual=plane_visual,
        plane_store=plane_store,
        axis_visual_ids=axis_visual_ids,
        _plane_updater=plane_updater,
        _orientation_updater=orient_updater,
        _owner_ids=owner_ids,
        _controller=controller,
    )
