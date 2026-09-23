"""Tests for ``infer_multiscale_translations`` (the CLI's
``--infer-multiscale-translations``).

A pyramid downsampled centre-aligned (block averages) puts the centre of a
coarse voxel at level-0 index ``(f - 1) / 2``; a store that omits the
translation saying so draws each coarser level shifted toward the origin.  The
flag fills the offsets in when the store declares none, and warns and keeps
the store's own when it does.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

_SCALE_Z = 2.0
_SCALE_YX = 0.5
_N_LEVELS = 3


def _write_pyramid(path: Path, *, translations: bool) -> str:
    """Write a small (z, y, x) OME-Zarr whose y/x halve per level.

    With *translations*, each level declares the centre-aligned offset; without,
    it declares only a scale, like the ExpA dataset.
    """
    import zarr

    root = zarr.open_group(str(path), mode="w")
    data = np.arange(4 * 16 * 16, dtype=np.uint16).reshape(4, 16, 16)
    datasets = []
    for level in range(_N_LEVELS):
        factor = 2**level
        level_data = data[:, ::factor, ::factor]
        arr = root.create_array(
            f"s{level}", shape=level_data.shape, chunks=(4, 8, 8), dtype=np.uint16
        )
        arr[:] = level_data
        yx = _SCALE_YX * factor
        transforms: list[dict] = [{"type": "scale", "scale": [_SCALE_Z, yx, yx]}]
        if translations:
            offset = (factor - 1) / 2 * _SCALE_YX
            transforms.append(
                {"type": "translation", "translation": [0.0, offset, offset]}
            )
        datasets.append({"path": f"s{level}", "coordinateTransformations": transforms})
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "axes": [
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": datasets,
                "name": "pyramid",
            }
        ],
    }
    return f"file://{path.resolve()}"


# (f - 1) / 2 level-0 voxels on y and x; z is not downsampled.
_CENTRE_ALIGNED = [(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.0, 1.5, 1.5)]


def _extract(uri: str, **kwargs):
    from oz_viewer.viewer._geometry import extract_viewer_geometry

    return extract_viewer_geometry(uri, print_summary=False, **kwargs)


def test_without_the_flag_the_store_is_untouched(tmp_path):
    uri = _write_pyramid(tmp_path / "bare.zarr", translations=False)
    store, _ = _extract(uri)
    assert [tuple(t) for t in store.level_translations] == [(0.0, 0.0, 0.0)] * 3


def test_flag_infers_centre_aligned_offsets(tmp_path):
    uri = _write_pyramid(tmp_path / "bare.zarr", translations=False)
    plain_store, plain = _extract(uri)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        store, geometry = _extract(uri, infer_multiscale_translations=True)

    assert [tuple(t) for t in store.level_translations] == _CENTRE_ALIGNED
    # cellier rebuilt the level transforms from them (level-0 voxel units).
    assert store.level_transforms[2].translation[1] == pytest.approx(1.5)
    assert store.level_transforms[2].translation[0] == pytest.approx(0.0)

    # Everything else carries over, and level 0 -- hence the image's place in
    # the world -- does not move.
    assert store.zarr_path == plain_store.zarr_path
    assert store.level_scales == plain_store.level_scales
    assert store.level_shapes == plain_store.level_shapes
    np.testing.assert_array_equal(geometry.world_max_full, plain.world_max_full)
    np.testing.assert_array_equal(geometry.level_0_scale, plain.level_0_scale)


def test_flag_matches_what_a_complete_file_declares(tmp_path):
    """The inferred offsets equal what a file with the offsets states."""
    bare = _write_pyramid(tmp_path / "bare.zarr", translations=False)
    full = _write_pyramid(tmp_path / "full.zarr", translations=True)
    inferred, _ = _extract(bare, infer_multiscale_translations=True)
    declared, _ = _extract(full)
    assert [tuple(t) for t in inferred.level_translations] == [
        tuple(t) for t in declared.level_translations
    ]


def test_flag_warns_and_keeps_declared_translations(tmp_path):
    uri = _write_pyramid(tmp_path / "full.zarr", translations=True)
    declared, _ = _extract(uri)

    with pytest.warns(UserWarning, match="already declares multiscale level"):
        store, _ = _extract(uri, infer_multiscale_translations=True)

    assert [tuple(t) for t in store.level_translations] == [
        tuple(t) for t in declared.level_translations
    ]
