# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Shared helpers for the raster (`RS_*`) function tests.

Each raster function's test module compares SedonaDB directly against a
rasterio reference computed inline: define the raster once as a numpy array +
GDAL geotransform, write it to a GeoTIFF with `write_geotiff`, run `RS_*`
through the SedonaDB connection, reconstruct the expected result from rasterio
primitives (resolving any engine-specific policy, like the fill outside a
rasterized geometry, to SedonaDB's), and compare with `assert_decoded_equal`.

This module holds the pieces shared across those modules — e.g. `DecodedRaster`
(a raster decoded to plain values), the `write_*` fixture writers, the
`decode_*` helpers, `assert_decoded_equal` / `assert_transform_and_nodata`,
`random_raster_data`, and small helpers (`dtype_min`, `approx_geotransform`,
`_is_nodata`).

Fixtures stay CRS-less (no CRS on either side, so nothing reprojects and
results stay bit-comparable) except where an encoder requires a real one —
then every side carries the same CRS.
"""

import math
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Tuple

import numpy as np


@dataclass
class DecodedRaster:
    """One raster decoded to plain values.

    `pixels` is `(band, rows, cols)`, `gdal_transform` is GDAL-order
    `(origin_x, scale_x, skew_x, origin_y, skew_y, scale_y)`, and `nodata`
    holds one sentinel per band (unpacked in the band's dtype).
    `compression` is the codec name of the decoded container when one was
    read (GeoTIFF decodes only); it is carried for encoder tests to assert
    on and deliberately not part of `assert_decoded_equal`.
    """

    pixels: "np.ndarray"
    gdal_transform: Tuple[float, ...]
    nodata: List[Any]
    compression: Optional[str] = None


def _is_nodata(sampled, nodata) -> bool:
    """Whether a sampled value equals the band nodata, NaN-aware (a NaN
    sentinel matches NaN pixels, which bare `==` never would)."""
    if nodata is None:
        return False
    if math.isnan(nodata):
        return bool(np.isnan(sampled))
    return bool(sampled == nodata)


def assert_decoded_equal(got: DecodedRaster, expected: DecodedRaster, *, context=""):
    """Strict raster comparison: exact pixels and dtype, geotransform to
    1e-12, nodata by value (None must match None, NaN matches NaN).
    `compression` is decode metadata, not content, and is not compared."""
    assert got is not None, f"got no raster: {context}"
    assert expected is not None, f"expected no raster: {context}"
    assert got.pixels.dtype == expected.pixels.dtype, context
    np.testing.assert_array_equal(got.pixels, expected.pixels, err_msg=str(context))
    assert got.gdal_transform == approx_geotransform(expected.gdal_transform), context
    assert len(got.nodata) == len(expected.nodata), context
    for got_nodata, expected_nodata in zip(got.nodata, expected.nodata):
        if expected_nodata is None:
            assert got_nodata is None, context
        elif isinstance(expected_nodata, float) and math.isnan(expected_nodata):
            assert got_nodata is not None and math.isnan(got_nodata), context
        else:
            assert got_nodata == expected_nodata, context


def approx_geotransform(value):
    """pytest.approx tight enough that only real georeferencing bugs pass it."""
    import pytest

    return pytest.approx(value, rel=1e-12, abs=1e-12)


def decode_raster(scalar) -> Optional[DecodedRaster]:
    """Decode one `sedona.raster` Arrow scalar to a `DecodedRaster` (None if NULL)."""
    if not scalar.is_valid:
        return None
    raster = scalar.as_py()
    return DecodedRaster(
        raster.to_numpy(),
        tuple(raster.transform),
        [band.nodata for band in raster.bands],
    )


def decode_geotiff(path) -> DecodedRaster:
    """Decode a GeoTIFF file to a `DecodedRaster` with rasterio."""
    with open(path, "rb") as f:
        return decode_geotiff_bytes(f.read())


def decode_geotiff_bytes(data: bytes) -> DecodedRaster:
    """Decode in-memory GeoTIFF bytes to a `DecodedRaster` with rasterio."""
    from rasterio.io import MemoryFile

    with MemoryFile(bytes(data)) as mem, mem.open() as src:
        return DecodedRaster(
            src.read(),
            tuple(src.transform.to_gdal()),
            list(src.nodatavals),
            compression=src.compression.value if src.compression else None,
        )


def write_geotiff(
    path, data: "np.ndarray", *, gdal_transform, nodata=None, crs=None
) -> None:
    """Write a `(bands, height, width)` array as a GeoTIFF.

    `gdal_transform` is GDAL-order `(origin_x, scale_x, skew_x, origin_y,
    skew_y, scale_y)`; `nodata` (optional) becomes the per-band nodata of
    every band. `crs` (optional) is any CRS rasterio accepts; parity fixtures
    stay CRS-less unless an engine requires one, and then use the same CRS
    everywhere so nothing reprojects.
    """
    import rasterio
    from rasterio.transform import Affine

    bands, height, width = data.shape
    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=str(data.dtype),
        transform=Affine.from_gdal(*gdal_transform),
        nodata=nodata,
        crs=crs,
    ) as dst:
        dst.write(data)


def write_random_geotiff(
    path, dtype, *, bands, height, width, gdal_transform, crs=None, nodata=None
) -> None:
    """Write a GeoTIFF of random `dtype` pixels on the given grid.

    Combines `random_raster_data` (dtype extremes planted in opposite corners)
    with `write_geotiff` — the input-raster fixture shape shared by raster warp
    parity tests. `gdal_transform`, `crs`, and `nodata` are as `write_geotiff`.
    """
    data = random_raster_data(dtype, bands=bands, height=height, width=width)
    write_geotiff(path, data, gdal_transform=gdal_transform, nodata=nodata, crs=crs)


def write_grid_geotiff(path, *, gdal_transform, width, height, crs=None) -> None:
    """Write a zeroed single-band GeoTIFF whose only role is to define a grid.

    Its pixels are never read — only its extent, resolution, and CRS matter,
    e.g. as the reference grid for `RS_ReprojectMatch`.
    """
    data = np.zeros((1, height, width), dtype="uint8")
    write_geotiff(path, data, gdal_transform=gdal_transform, crs=crs)


def assert_transform_and_nodata(got: DecodedRaster, expected: DecodedRaster) -> None:
    """Assert two decoded rasters share a geotransform (to 1e-12) and per-band
    nodata. A lighter check than `assert_decoded_equal` for tests that compare
    pixels separately — e.g. with a resampling tolerance."""
    assert got.gdal_transform == approx_geotransform(expected.gdal_transform)
    assert got.nodata == expected.nodata


def dtype_min(dtype):
    """The minimum representable value of a numpy dtype — SedonaDB's default
    nodata sentinel when neither an explicit value nor a band nodata exists."""
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        return float(np.finfo(dtype).min)
    return int(np.iinfo(dtype).min)


def random_raster_data(
    dtype,
    *,
    bands: int,
    height: int,
    width: int,
    seed: int = 42,
    plants: Optional[Mapping[Tuple[int, int], Any]] = None,
) -> "np.ndarray":
    """Random `(bands, height, width)` pixels with adversarial values planted.

    The dtype extremes always go in opposite corners (values that must
    round-trip through any operation that keeps them, and be overwritten by
    any that fills them). `plants` maps `(row, col)` to a value written into
    every band — use it to place values the test's geometry or nodata choices
    make meaningful.
    """
    rng = np.random.default_rng(seed)
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        data = ((rng.random((bands, height, width)) - 0.5) * 200.0).astype(dtype)
        info = np.finfo(dtype)
    else:
        info = np.iinfo(dtype)
        data = rng.integers(
            info.min, info.max, size=(bands, height, width), dtype=dtype, endpoint=True
        )
    data[:, 0, 0] = info.max
    data[:, -1, -1] = info.min
    for (row, col), value in (plants or {}).items():
        data[:, row, col] = value
    return data
