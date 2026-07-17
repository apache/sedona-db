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

"""Cross-engine parity testing for raster (`RS_*`) functions.

The raster sibling of the `DBEngine` framework in `sedonadb.testing`.
Geometry parity broadcasts one SQL string to every engine; raster engines
cannot share SQL — rasterio is not a SQL engine, and raster SQL dialects
disagree on function names and argument order — so parity here is
operation-level instead: `RasterEngine` exposes one method per operation
(`clip()`, ...) returning plain decoded values, and each engine translates
the operation into its own invocation.

Engines come in two kinds, which changes how tests compare results:

- **Dialect engines** (`SedonaDB`; a Sedona Spark engine belongs here too)
  execute the function under test and compare strictly — exact pixels, exact
  nodata, exact NULL/error behavior.
- **Reference engines** (`Rasterio`) reconstruct the operation from library
  primitives; engine-specific policy (like nodata precedence) is resolved by
  the caller before comparing.

Test fixtures follow one pattern: define the raster once as a numpy array +
GDAL geotransform, write it to a CRS-less GeoTIFF with `write_geotiff` (no
CRS on either side, so nothing reprojects and results stay bit-comparable),
and hand the same file to every engine.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Tuple

import numpy as np
import pyarrow as pa


@dataclass
class ClipResult:
    """One clipped raster decoded to plain values.

    `pixels` is `(band, rows, cols)`, `gdal_transform` is GDAL-order
    `(origin_x, scale_x, skew_x, origin_y, skew_y, scale_y)`, and `nodata`
    holds one sentinel per band (unpacked in the band's dtype).
    """

    pixels: "np.ndarray"
    gdal_transform: Tuple[float, ...]
    nodata: List[Any]


class RasterEngine(ABC):
    """Executes raster operations on one engine for cross-engine comparison."""

    @classmethod
    def create_or_skip(cls, *args, **kwargs):
        """Create this engine, or skip the calling test if it can't be built.

        If `SEDONADB_PYTHON_NO_SKIP_TESTS` is set the failure propagates
        instead, so CI can't silently skip.
        """
        try:
            return cls(*args, **kwargs)
        except Exception as e:
            if os.environ.get("SEDONADB_PYTHON_NO_SKIP_TESTS", "false") in (
                "true",
                "1",
            ):
                raise
            import pytest

            pytest.skip(f"Can't create {cls.__name__} raster engine: {e}")

    @abstractmethod
    def clip(
        self,
        path,
        geometry_wkt: str,
        *,
        band: int = 0,
        all_touched: bool = False,
        nodata: Optional[float] = None,
        crop: bool = True,
    ) -> Optional[ClipResult]:
        """Clip the GeoTIFF at `path` to a WKT geometry.

        `band` 0 selects all bands, otherwise the 1-based band. Returns None
        when the geometry selects no pixels and the engine's lenient behavior
        yields a NULL/absent result.
        """


class SedonaDB(RasterEngine):
    """Runs `RS_*` on a SedonaDB connection — the engine under test."""

    def __init__(self, con=None):
        import sedonadb

        self._con = con if con is not None else sedonadb.connect()

    @classmethod
    def create_or_skip(cls, *args, **kwargs):
        # Never skip on the engine under test: a construction failure here is
        # a bug, not a missing optional backend.
        return cls(*args, **kwargs)

    def clip(
        self, path, geometry_wkt, *, band=0, all_touched=False, nodata=None, crop=True
    ):
        (result,) = self.clip_rows(
            path,
            [
                {
                    "wkt": geometry_wkt,
                    "band": band,
                    "all_touched": all_touched,
                    "nodata": nodata,
                    "crop": crop,
                }
            ],
        )
        return result

    def clip_rows(self, path, rows) -> List[Optional[ClipResult]]:
        """Run `RS_Clip` once over an N-row table, one row per parameter combo.

        Options travel as table columns rather than literals so the kernel
        executes its real array path (literals constant-fold). Each row is a
        dict with keys `wkt`, `band`, `all_touched`, `nodata` (None = use the
        band's own), and `crop`. Returns one `ClipResult` (or None for NULL
        rows) per input row, in input order.
        """
        table = pa.table(
            {
                "idx": pa.array(range(len(rows)), type=pa.int64()),
                "path": pa.array([str(path)] * len(rows), type=pa.utf8()),
                "wkt": pa.array([r["wkt"] for r in rows], type=pa.utf8()),
                "band": pa.array([r["band"] for r in rows], type=pa.int32()),
                "all_touched": pa.array(
                    [r["all_touched"] for r in rows], type=pa.bool_()
                ),
                "nodata": pa.array([r["nodata"] for r in rows], type=pa.float64()),
                "crop": pa.array([r["crop"] for r in rows], type=pa.bool_()),
            }
        )
        df = self._con.create_data_frame(table)
        result = (
            df.select(
                "idx",
                r=df.path.funcs.rs_frompath().funcs.rs_clip(
                    df.band,
                    self._con.funcs.st_geomfromtext(df.wkt),
                    df.all_touched,
                    df.nodata,
                    df.crop,
                ),
            )
            .sort("idx")
            .to_arrow_table()["r"]
        )
        return [decode_raster(result[i]) for i in range(len(result))]


class Rasterio(RasterEngine):
    """Reference implementation composed from rasterio primitives.

    The window is `rasterio.features.geometry_window` (geometry bounds ∩
    raster extent, snapped outward to the grid), the selection is
    `rasterio.features.geometry_mask`, and pixels compose as "inside the
    geometry: source value verbatim; outside: `nodata`". This is deliberately
    not `rasterio.mask.mask`, which additionally reads the source masked and
    so remaps pixels *valued* at the band nodata to the output nodata — a
    policy Sedona (Spark and DB) does not follow.

    As a reference engine it takes no position on nodata precedence: `nodata`
    must be the already-resolved fill value, so `clip()` errors when it is
    None rather than guessing.

    One caveat on independence: both this engine and SedonaDB ultimately burn
    geometries with GDAL's rasterizer (rasterio bundles its own GDAL), so the
    pixel-center/all_touched selection rule itself is shared. The genuinely
    independent implementations under comparison are everything on top of the
    burn: window snapping, crop copy, transform shift, nodata handling, and
    band handling.
    """

    def __init__(self):
        import rasterio  # noqa: F401 — availability probe for create_or_skip
        import shapely  # noqa: F401

    def clip(
        self, path, geometry_wkt, *, band=0, all_touched=False, nodata=None, crop=True
    ):
        import rasterio
        import rasterio.features
        import rasterio.windows
        import shapely

        if nodata is None:
            raise ValueError(
                "Rasterio is a reference engine: pass the resolved nodata fill"
            )

        geom = shapely.from_wkt(geometry_wkt)
        with rasterio.open(str(path)) as src:
            if band < 0 or band > src.count:
                raise ValueError(
                    f"band {band} out of range for a {src.count}-band raster"
                )
            if crop:
                window = rasterio.features.geometry_window(src, [geom])
            else:
                window = rasterio.windows.Window(0, 0, src.width, src.height)
            transform = src.window_transform(window)
            data = src.read(window=window)
            inside = rasterio.features.geometry_mask(
                [geom],
                out_shape=(data.shape[1], data.shape[2]),
                transform=transform,
                all_touched=all_touched,
                invert=True,
            )
            # numpy raises for out-of-range fills but silently truncates
            # fractional ones (250.7 -> 250 on uint8); a lossy fill would make
            # the reference mirror the very bug it exists to catch.
            fill = np.asarray(nodata, dtype=data.dtype)
            if float(fill) != float(nodata):
                raise ValueError(
                    f"nodata {nodata} is not exactly representable as {data.dtype}"
                )
            pixels = np.where(inside, data, fill)

        if band != 0:
            pixels = pixels[band - 1 : band]
        return ClipResult(pixels, tuple(transform.to_gdal()), [nodata] * len(pixels))


def decode_raster(scalar) -> Optional[ClipResult]:
    """Decode one `sedona.raster` Arrow scalar to a `ClipResult` (None if NULL)."""
    if not scalar.is_valid:
        return None
    raster = scalar.as_py()
    return ClipResult(
        raster.to_numpy(),
        tuple(raster.transform),
        [band.nodata for band in raster.bands],
    )


def write_geotiff(path, data: "np.ndarray", *, gdal_transform, nodata=None) -> None:
    """Write a `(bands, height, width)` array as a CRS-less GeoTIFF.

    `gdal_transform` is GDAL-order `(origin_x, scale_x, skew_x, origin_y,
    skew_y, scale_y)`; `nodata` (optional) becomes the per-band nodata of
    every band.
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
    ) as dst:
        dst.write(data)


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
