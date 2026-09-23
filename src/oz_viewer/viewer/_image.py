"""The multiscale image both viewers add, built from the OME-Zarr geometry.

Each viewer holds one image visual (fanned out to every panel on the
orthoviewer).  When the store has a channel axis the visual carries it and
starts in composite mode, drawing its channels at once; the image control's
mode switch flips it to single mode at runtime, where the channel axis gets a
slider instead.  Both modes are configured up front so either can be shown.

Also holds the render-pipeline configs the two viewers share.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from oz_viewer.viewer._widgets import _DEFAULT_COLORMAPS

if TYPE_CHECKING:
    from cellier.render import RenderManagerConfig
    from cellier.visuals import MultiscaleImageRenderConfig

    from oz_viewer.viewer._geometry import _ViewerGeometry

# Appearance fields shown in the image control.  Fields that only one mode
# has (e.g. a channel's ``visible``) appear on that mode's page.
_IMAGE_FIELDS = [
    "visible",
    "opacity",
    "color_map",
    "clim",
    "render_mode",
    "iso_threshold",
    "attenuation",
    "lod_bias",
]

# Composite channel colormaps, in channel order.  Composites blend additively
# by default, so each colormap starts from black.
_CHANNEL_COLORMAPS = ["green", "magenta", "cyan", "red"]


def controller_render_config() -> RenderManagerConfig:
    """The controller render-pipeline config used by both viewers."""
    from cellier.render import RenderManagerConfig, TemporalAccumulationConfig

    return RenderManagerConfig(temporal=TemporalAccumulationConfig(enabled=False))


def visual_render_config() -> MultiscaleImageRenderConfig:
    """LOD / GPU-budget config for the multiscale image visual.

    The 3D budget is only allocated by panels that render in 3D, so the
    orthoviewer's 2D panels can share this config.
    """
    from cellier.visuals import MultiscaleImageRenderConfig

    return MultiscaleImageRenderConfig(
        block_size=32,
        gpu_budget_bytes=2048 * 1024**2,
        gpu_budget_bytes_2d=64 * 1024**2,
    )


def _max_channels() -> int:
    """Cellier's default cap on the channels one image visual may hold."""
    from cellier.visuals import MultiscaleImageVisual

    return int(MultiscaleImageVisual.model_fields["max_channels"].default)


def image_visual_kwargs(
    geometry: _ViewerGeometry,
    *,
    render_mode: str,
    lod_bias: float,
) -> dict[str, Any]:
    """Keyword arguments for ``add_image_multiscale`` on either viewer.

    Parameters
    ----------
    geometry : _ViewerGeometry
        Metadata extracted by :func:`extract_viewer_geometry`.
    render_mode : str
        The single-mode 3D render mode, e.g. ``"mip"`` or ``"iso"``.
    lod_bias : float
        Initial level-of-detail bias.

    Returns
    -------
    dict[str, Any]
        Everything but the data store and the visual name.  With a channel
        axis the visual starts in composite mode holding the first
        ``max_channels`` channels (cellier's default); single mode can still
        step through every channel.
    """
    from cellier.convenience import MultiscaleImageControlsConfig
    from cellier.visuals import (
        MultiscaleImageAppearance,
        MultiscaleImageChannelAppearance,
        MultiscaleImageSingleAppearance,
    )

    clim_max = geometry.initial_clim_max
    labels = geometry.channel_labels
    kwargs: dict[str, Any] = {
        "appearance": MultiscaleImageAppearance(lod_bias=lod_bias, attenuation=1.0),
        "single": MultiscaleImageSingleAppearance(
            color_map="viridis",
            clim=(0.0, clim_max),
            iso_threshold=clim_max / 2.0,
            render_mode=render_mode,
        ),
        "render_config": visual_render_config(),
        "transform": geometry.voxel_to_world,
        "controls": MultiscaleImageControlsConfig(
            appearance=_IMAGE_FIELDS,
            colormap_names=_DEFAULT_COLORMAPS,
            clim_range=geometry.clim_range,
            channel_labels=dict(enumerate(labels)) if labels is not None else None,
        ),
    }
    if geometry.channel_axis is not None:
        n_held = min(geometry.n_channels, _max_channels())
        kwargs["channel_axis"] = geometry.channel_axis
        kwargs["composite"] = True
        kwargs["channels"] = {
            i: MultiscaleImageChannelAppearance(
                color_map=_CHANNEL_COLORMAPS[i % len(_CHANNEL_COLORMAPS)],
                clim=(0.0, clim_max),
                iso_threshold=clim_max / 2.0,
                render_mode="mip",
            )
            for i in range(n_held)
        }
    return kwargs
