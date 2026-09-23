# oz-viewer

[![License](https://img.shields.io/pypi/l/oz-viewer.svg?color=green)](https://github.com/kevinyamauchi/oz-viewer/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/oz-viewer.svg?color=green)](https://pypi.org/project/oz-viewer)
[![Python Version](https://img.shields.io/pypi/pyversions/oz-viewer.svg?color=green)](https://python.org)
[![CI](https://github.com/kevinyamauchi/oz-viewer/actions/workflows/ci.yml/badge.svg)](https://github.com/kevinyamauchi/oz-viewer/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/kevinyamauchi/oz-viewer/branch/main/graph/badge.svg)](https://codecov.io/gh/kevinyamauchi/oz-viewer)

> [!NOTE]
> This is an early prototype. Many things may not be working and the API will change.

A viewer for ome-zarr images.

## Installation

```sh
uv pip install "oz-viewer[examples] @ git+https://github.com/kevinyamauchi/oz-viewer.git"
```

## Download dataset

You can download an example dataset using the `download` CLI. For example, download the https://livingobjects.ebi.ac.uk/idr/zarr/v0.5/idr0066/ExpA_VIP_ASLM_on.zarr (1937, 2048, 2048) dataset used in the examples below with the following command (~10 GB):

```sh
oz-viewer download https://livingobjects.ebi.ac.uk/idr/zarr/v0.5/idr0066/ExpA_VIP_ASLM_on.zarr
```

## Single canvas viewer

You can load a v0.4 or v0.5 OME-Zarr file into a single-canvas 2d/3d viewer using the `oz-viewer view` CLI. See the example below. Replace the file path with the path to your image. This works for both local and remote data.

```sh
oz-viewer view path/to/image.ome.zarr
```

Example viewing a https://livingobjects.ebi.ac.uk/idr/zarr/v0.5/idr0066/ExpA_VIP_ASLM_on.zarr (1937, 2048, 2048), anisotropic voxels (file was on local SSD).

https://github.com/user-attachments/assets/59c31e89-db42-4cec-ae4f-373d997c227f

## Orthoviewer

You can load a v0.4 or v0.5 OME-Zarr file into an orthoviewer using the `oz-viewer ortho ` CLI. See the example below. Replace the file path with the path to your image. This works for both local and remote data.

```sh
oz-viewer ortho path/to/image.ome.zarr
```

Example viewing https://livingobjects.ebi.ac.uk/idr/zarr/v0.5/idr0066/ExpA_VIP_ASLM_on.zarr (1937, 2048, 2048), anisotropic voxels (file was on local SSD).

https://github.com/user-attachments/assets/a6c0cab9-0cd9-4fe0-80c2-c77207b4fd77

Images with a channel axis open as a composite of their channels (up to 4). Use the "Composite channels" checkbox in the image controls on the left to switch between the composite and single-channel rendering, where a slider steps through the channels. Example viewing the scikit-image cells3d (converted to ome-zarr) multichannel image

https://github.com/user-attachments/assets/b6655863-b8fb-4eea-be84-eb031283494e

## Jupyter Lab

Both the viewer and orthoviewer can be used in Jupyter Lab. See the `examples/viewer.ipynb` and `examples/orthoviewer.ipynb` notebooks for examples. The viewer can be rendered as a sidecar widget next to the notebook.

https://github.com/user-attachments/assets/f572a57e-ab92-4bf4-8b4c-e6af46c9f6ed

## Test dataset latency

The latency for fetching chunks greatly impacts the rendering performance of the viewer. High latency means that fetching chunks from the data store is slow and thus rendering feels slow. If the latency for fetching data from your image is high, you can offset it by increasing the lod_bias which causes the renderer to prefer loading lower resolution chunks. This reduces the aount of data that needs to be loaded, but sacrifices detail.

To measure the latency, you can use the `oz-viewer ping` CLI. See the example below. Replace the path with the path to your data. You can use both URL and file paths.

```sh
oz-viewer ping https://livingobjects.ebi.ac.uk/idr/share/ome2024-ngff-challenge/idr0066/ExpC_TPH2_left_cerebellum.zarr
```

To view the options, you can run

```sh
oz-viewer ping --help
```
  
## Orthoviewer startup performance diagnostics

Use the orthoviewer performance flags to print startup timings
with step and cumulative durations:

```sh
oz-viewer ortho /path/to/data.zarr --perf-startup --perf-table --perf-table-title "My Startup Profile"
```

## Development

The easiest way to get started is to use the [github cli](https://cli.github.com)
and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
gh repo fork kevinyamauchi/oz-viewer --clone
# or just
# gh repo clone kevinyamauchi/oz-viewer
cd oz-viewer
uv sync
```

Run tests:

```sh
uv run pytest
```

Lint files:

```sh
uv run pre-commit run --all-files
```
