"""OME-Zarr metadata extraction shared by the convenience-based viewers.

This module holds the pure, Qt-free logic that turns an OME-Zarr store into the
handful of objects the cellier ``convenience`` API needs: the world coordinate
system, the voxel-to-world transform, which axis is the channel axis, the
spatial extents, the slider values, the contrast-limit range, and sensible
slider decimals.  It builds no viewer itself so it can be reused by both the
single-panel viewer and the orthoviewer.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np

from oz_viewer.viewer._utils import _dtype_clim_max, _dtype_decimals, _perf_mark

if TYPE_CHECKING:
    from cellier.convenience import ContinuousAxisValues, DiscreteAxisValues
    from cellier.data.image import OMEZarrImageDataStore
    from cellier.transform import AffineTransform, WorldCoordinateSystem

    from oz_viewer._perf import StartupPerfTracer


class _ViewerGeometry(NamedTuple):
    """Scene geometry derived from OME-Zarr metadata; no Qt objects."""

    axis_names: tuple[str, ...]
    spatial_indices: list[int]
    spatial_ndim: int
    channel_axis: int | None
    n_channels: int
    #: One name per channel from the store's ``omero`` metadata, or ``None``.
    channel_labels: tuple[str, ...] | None
    #: The world every scene shares, with the store's axis names, types and
    #: units.  Pass this exact object to the viewer: a transform names its
    #: endpoints by id, so an equal-looking world built elsewhere is rejected.
    world: WorldCoordinateSystem
    #: Level-0 voxel -> ``world``, scaled by the level-0 OME-Zarr scale.
    voxel_to_world: AffineTransform
    #: World units per level-0 voxel, for every axis.
    level_0_scale: np.ndarray
    #: World-space ``(shape - 1) * scale`` for every axis, used for centering.
    world_max_full: np.ndarray
    #: World-space ``(shape - 1) * scale`` for the spatial axes only.
    world_max_spatial: np.ndarray
    initial_clim_max: float
    clim_range: tuple[float, float]
    slider_decimals: int
    #: ``(near, far)`` clip distances derived from the world extents.
    depth_range: tuple[float, float]

    @property
    def spatial_axis_labels(self) -> tuple[str, ...]:
        """Axis labels of the spatial axes, in order."""
        return tuple(self.axis_names[i] for i in self.spatial_indices)

    @property
    def spatial_axes(self) -> tuple[int, ...]:
        """Global spatial axis indices as a tuple (``(z, y, x)`` when 3-D).

        The orthoviewer passes this to
        :class:`cellier.convenience.OrthoViewer` and to the overlay builders,
        which expect a length-3 ``(z, y, x)`` tuple.
        """
        return tuple(self.spatial_indices)

    @property
    def axis_values(self) -> dict[int, ContinuousAxisValues | DiscreteAxisValues]:
        """World-space slider values per axis, for the dims sliders.

        The channel axis is discrete -- one stop per channel, so its slider
        never rests between two channels -- and named by the ``omero`` channel
        labels when the store has them.  It only shows a slider while the image
        is in single mode; a composite draws every channel at once.  Every
        other axis (spatial, time) is
        continuous over ``[0, world_max]``, from the first voxel centre to the
        last.  Derived from the metadata rather than via
        ``cellier.convenience.axis_values_from_viewer``, which widens each axis
        by half a voxel at both ends.
        """
        from cellier.convenience import ContinuousAxisValues, DiscreteAxisValues

        values: dict[int, ContinuousAxisValues | DiscreteAxisValues] = {}
        for axis in range(len(self.axis_names)):
            if axis == self.channel_axis:
                step = float(self.level_0_scale[axis])
                values[axis] = DiscreteAxisValues(
                    values=tuple(i * step for i in range(self.n_channels)),
                    labels=self.channel_labels,
                )
            else:
                values[axis] = ContinuousAxisValues(
                    min=0.0, max=float(self.world_max_full[axis])
                )
        return values

    def center_slice_indices(self) -> dict[int, float]:
        """World-coordinate midpoints for each spatial axis.

        Not rounded: a slice position is a float world coordinate.  Extra
        (non-spatial) axes such as channel are intentionally omitted so they
        keep their default position of ``0``.
        """
        return {
            axis: float(self.world_max_full[axis]) / 2.0
            for axis in self.spatial_indices
        }


def extract_viewer_geometry(
    zarr_uri: str,
    *,
    channel_axis: int | None = None,
    perf: StartupPerfTracer | None = None,
    print_summary: bool = True,
    infer_multiscale_translations: bool = False,
) -> tuple[OMEZarrImageDataStore, _ViewerGeometry]:
    """Open an OME-Zarr store and extract the geometry the viewers need.

    Parameters
    ----------
    zarr_uri : str
        Path or URI to the OME-Zarr store.
    channel_axis : int or None, optional
        Axis index to treat as the channel dimension.  When ``None`` (default),
        the channel axis is auto-detected from the store's axis types.
    perf : StartupPerfTracer or None, optional
        Optional startup performance tracer.
    print_summary : bool, optional
        Print a Rich table summarizing the store's metadata.  Default ``True``.
    infer_multiscale_translations : bool, optional
        Assume the coarser levels were downsampled centre-aligned (e.g. block
        averages) and give each the half-voxel offset that implies, when the
        store declares no level translations.  See
        :func:`_with_inferred_level_translations`.  Default ``False``: wrong
        for pyramids made by striding.

    Returns
    -------
    tuple[OMEZarrImageDataStore, _ViewerGeometry]
    """
    from cellier.data.image import OMEZarrImageDataStore

    _perf_mark(perf, "geometry.start", zarr_uri=zarr_uri)
    data_store = OMEZarrImageDataStore.from_path(zarr_uri)
    _perf_mark(perf, "geometry.data_store_ready", n_levels=data_store.n_levels)

    translations: _TranslationSource = (
        "declared" if _has_level_translations(data_store) else "none"
    )
    if infer_multiscale_translations:
        if translations == "declared":
            warnings.warn(
                f"{zarr_uri} already declares multiscale level translations; "
                "--infer-multiscale-translations is ignored and the declared "
                "translations are used.",
                UserWarning,
                stacklevel=2,
            )
        else:
            data_store = _with_inferred_level_translations(data_store)
            translations = "inferred"
            _perf_mark(perf, "geometry.level_translations_inferred")

    # The level-0 coordinate system is the store's record of its axis names,
    # types and units, parsed from the NGFF metadata at construction.
    data_axes = data_store.data_coordinate_systems[0].axes
    n_dims = len(data_axes)

    # Detect the channel axis from the axis types when not explicitly set.
    if channel_axis is None:
        for idx, ax in enumerate(data_axes):
            if ax.axis_type == "channel":
                channel_axis = idx
                break

    spatial_indices: list[int] = [i for i in range(n_dims) if i != channel_axis]
    spatial_ndim = len(spatial_indices)

    # The level-0 data -> world scale: the multiscale's global scale composed
    # with the finest dataset's, as the store parsed it.
    level_0_scale_full = np.array(
        data_store.physical_scale or [1.0] * n_dims, dtype=np.float64
    )
    level_0_scale_spatial = level_0_scale_full[spatial_indices]

    vox_shape_full = np.array(data_store.level_shapes[0], dtype=np.float64)
    vox_shape_spatial = vox_shape_full[spatial_indices]
    world_extents_spatial = vox_shape_spatial * level_0_scale_spatial
    max_extent = float(world_extents_spatial.max())
    depth_range = (max(1.0, max_extent * 0.0001), max_extent * 10.0)

    world_max_full = (vox_shape_full - 1) * level_0_scale_full
    world_max_spatial = (vox_shape_spatial - 1) * level_0_scale_spatial

    if print_summary:
        _print_metadata_table(
            zarr_uri,
            data_store,
            channel_axis,
            spatial_ndim,
            level_0_scale_spatial,
            world_extents_spatial,
            depth_range,
            translations,
        )
        _perf_mark(perf, "geometry.metadata_printed")

    world, voxel_to_world = _world_and_transform(data_store, level_0_scale_full)

    initial_clim_max = _dtype_clim_max(data_store.dtype)
    slider_decimals = _dtype_decimals(data_store.dtype)
    n_channels = (
        int(data_store.level_shapes[0][channel_axis]) if channel_axis is not None else 0
    )
    labels = data_store.channel_labels
    channel_labels = (
        tuple(labels) if labels is not None and len(labels) == n_channels else None
    )

    geometry = _ViewerGeometry(
        axis_names=tuple(ax.name for ax in data_axes),
        spatial_indices=spatial_indices,
        spatial_ndim=spatial_ndim,
        channel_axis=channel_axis,
        n_channels=n_channels,
        channel_labels=channel_labels,
        world=world,
        voxel_to_world=voxel_to_world,
        level_0_scale=level_0_scale_full,
        world_max_full=world_max_full,
        world_max_spatial=world_max_spatial,
        initial_clim_max=initial_clim_max,
        clim_range=(0.0, initial_clim_max),
        slider_decimals=slider_decimals,
        depth_range=depth_range,
    )
    _perf_mark(perf, "geometry.ready")
    return data_store, geometry


#: Where the store's level translations came from, for the metadata table.
_TranslationSource = Literal["declared", "inferred", "none"]

#: Store fields cellier derives at construction, so a rebuilt store must not
#: carry them over: the id comes from the coordinate systems, and the level
#: transforms from the level scales and translations.
_DERIVED_STORE_FIELDS = frozenset({"id", "level_transforms", "store_type"})


def _has_level_translations(data_store: OMEZarrImageDataStore) -> bool:
    """Whether any level of *data_store* is offset from level 0."""
    return any(
        any(float(value) != 0.0 for value in translation)
        for translation in data_store.level_translations
    )


def _inferred_level_translations(
    level_scales: list[tuple[float, ...]],
) -> list[tuple[float, ...]]:
    """Half-voxel offsets of centre-aligned downsampling, per level and axis.

    A level-k voxel that averages ``f`` level-0 voxels along an axis starts at
    level-0 voxel 0, so its centre sits at level-0 index ``(f - 1) / 2``.  The
    result is in level-0 voxels, the units of ``level_translations``.  An axis
    that is not downsampled (``f == 1``) gets 0, and so does level 0.
    """
    return [tuple((float(f) - 1.0) / 2.0 for f in scale) for scale in level_scales]


def _with_inferred_level_translations(
    data_store: OMEZarrImageDataStore,
) -> OMEZarrImageDataStore:
    """Rebuild *data_store* with the centre-aligned level translations.

    Goes through the public constructor so cellier rebuilds the level
    transforms from the new translations; every other field is copied, so a
    field cellier adds later is carried over too.  Level 0 is untouched, so
    the store sits in the world exactly where it did.
    """
    fields = {
        name: getattr(data_store, name)
        for name in type(data_store).model_fields
        if name not in _DERIVED_STORE_FIELDS and name != "level_translations"
    }
    fields["level_translations"] = _inferred_level_translations(data_store.level_scales)
    return type(data_store)(**fields)


def _world_and_transform(
    data_store: OMEZarrImageDataStore,
    level_0_scale: np.ndarray,
) -> tuple[WorldCoordinateSystem, AffineTransform]:
    """Build the world system and the level-0 voxel -> world transform.

    The world takes the store's own axes -- names, types and units -- one to
    one, so every data axis maps onto the world axis of the same position and
    type.  The transform is built against the store's level-0 coordinate
    system and this world, which is what cellier checks when it is added.
    """
    from cellier.transform import AffineTransform, Axis, WorldCoordinateSystem

    level_0 = data_store.data_coordinate_systems[0]
    world = WorldCoordinateSystem(
        name="world",
        axes=tuple(
            Axis(name=ax.name, axis_type=ax.axis_type, unit=ax.unit)
            for ax in level_0.axes
        ),
    )
    voxel_to_world = AffineTransform.from_axis_map(
        level_0,
        world,
        axis_map={
            data_axis.id: world_axis.id
            for data_axis, world_axis in zip(level_0.axes, world.axes, strict=True)
        },
        scale={
            data_axis.id: float(scale)
            for data_axis, scale in zip(level_0.axes, level_0_scale, strict=True)
        },
        name="voxel_to_world",
    )
    return world, voxel_to_world


def _print_metadata_table(
    zarr_uri: str,
    data_store: OMEZarrImageDataStore,
    channel_axis: int | None,
    spatial_ndim: int,
    level_0_scale_spatial: np.ndarray,
    world_extents_spatial: np.ndarray,
    depth_range: tuple[float, float],
    translations: _TranslationSource = "none",
) -> None:
    """Print the startup metadata summary as a Rich table."""
    from rich.console import Console
    from rich.table import Table

    spatial_label = "spatial" if channel_axis is not None else "ZYX"[-spatial_ndim:]
    table = Table(
        title=f"OME-Zarr  [dim]{zarr_uri}[/dim]",
        show_header=False,
        box=None,
        padding=(0, 2),
    )
    table.add_column("Field", style="bold cyan", no_wrap=True)
    table.add_column("Value")
    table.add_row("dtype", str(data_store.dtype))
    data_axes = data_store.data_coordinate_systems[0].axes
    table.add_row("axes", "  ".join(ax.name for ax in data_axes))
    table.add_row("units", "  ".join(str(ax.unit) for ax in data_axes))
    table.add_row("levels", str(data_store.n_levels))
    for i, shape in enumerate(data_store.level_shapes):
        table.add_row(f"  level {i}", str(list(shape)))
    table.add_row(
        f"scale ({spatial_label})",
        "  ".join(f"{v:.4g}" for v in level_0_scale_spatial),
    )
    table.add_row(
        f"world extents ({spatial_label})",
        "  ".join(f"{v:.4g}" for v in world_extents_spatial),
    )
    table.add_row("depth range", f"near={depth_range[0]:.2f}  far={depth_range[1]:.0f}")
    table.add_row(
        "level translations",
        {
            "declared": "declared in the file",
            "inferred": "inferred: (factor - 1) / 2 level-0 voxels (centre-aligned)",
            "none": "none",
        }[translations],
    )
    Console().print(table)
