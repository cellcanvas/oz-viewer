"""Tests for the decimals oz-viewer hands cellier's widgets.

Cellier shows a fixed 2 decimals unless told otherwise; oz-viewer passes 0
for integer images' contrast and threshold, and per dims axis just enough to
resolve half a voxel.
"""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    ("scale", "expected"),
    [
        (5.0, 0),  # ExpA z
        (6.55, 0),  # ExpA y/x
        (2.0, 0),
        (1.0, 1),  # half a voxel is 0.5
        (0.29, 1),  # cells3d z
        (0.26, 1),  # cells3d y/x
        (0.2, 1),  # an exact power of ten is not pushed further
        (0.1, 2),
        (0.0, 2),  # not a usable scale: the dims slider's usual 2
        (float("nan"), 2),
    ],
)
def test_axis_decimals_resolve_half_a_voxel(scale, expected):
    from oz_viewer.viewer._geometry import _axis_decimals

    assert _axis_decimals(scale) == expected


def test_an_integer_image_gets_integer_contrast_and_axis_readouts(tmp_path):
    """The uint8 blobs example: 0 decimals for the controls and every axis."""
    from cellier.convenience import ContinuousAxisValues

    from oz_viewer.data._blobs import make_example_zarr
    from oz_viewer.viewer._geometry import extract_viewer_geometry
    from oz_viewer.viewer._image import image_visual_kwargs

    zarr_path = make_example_zarr(output_path=tmp_path / "blobs.ome.zarr")
    _, geometry = extract_viewer_geometry(f"file://{zarr_path}", print_summary=False)

    controls = image_visual_kwargs(geometry, render_mode="mip", lod_bias=1.0)[
        "controls"
    ]
    assert controls.decimals == 0

    axis_values = geometry.axis_values
    assert all(isinstance(v, ContinuousAxisValues) for v in axis_values.values())
    # z is 5 um and y/x 6.55 um per voxel: whole micrometres suffice.
    assert [axis_values[a].decimals for a in sorted(axis_values)] == [0, 0, 0]


def test_the_channel_axis_keeps_its_discrete_readout(write_demo_ome):
    """Only continuous axes get decimals; the channel axis shows its values."""
    from cellier.convenience import ContinuousAxisValues, DiscreteAxisValues

    from oz_viewer.viewer._geometry import extract_viewer_geometry

    _, geometry = extract_viewer_geometry(
        f"file://{write_demo_ome('image')}", print_summary=False
    )
    axis_values = geometry.axis_values
    assert isinstance(axis_values[geometry.channel_axis], DiscreteAxisValues)
    for axis in geometry.spatial_indices:
        spec = axis_values[axis]
        assert isinstance(spec, ContinuousAxisValues)
        assert spec.decimals == _expected(geometry.level_0_scale[axis])


def _expected(scale: float) -> int:
    from oz_viewer.viewer._geometry import _axis_decimals

    return _axis_decimals(float(scale))
