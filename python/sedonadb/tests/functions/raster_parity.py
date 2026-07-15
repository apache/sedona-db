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

"""Helpers for validating RS_* functions against rasterio as a reference.

The pattern: define a raster once as a numpy array + GDAL geotransform, write
it to a GeoTIFF with rasterio, and run the function under test through both
engines — sedonadb reads the file back via ``RS_FromPath`` and rasterio reads
it directly — then compare pixel bytes, geotransform, and nodata exactly.
Fixtures carry no CRS so neither engine reprojects and results stay
bit-comparable.

rasterio is imported lazily inside each helper so importing this module does
not require it; test modules gate on ``pytest.importorskip("rasterio")``.
"""

import numpy as np
import pyarrow as pa


def dtype_min(dtype) -> float:
    """The minimum representable value of a numpy dtype — sedonadb's default
    nodata sentinel when neither an explicit value nor a band nodata exists."""
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        return float(np.finfo(dtype).min)
    return int(np.iinfo(dtype).min)


def write_geotiff(path, data: np.ndarray, *, gdal_transform, nodata=None) -> None:
    """Write a ``(bands, height, width)`` array as a CRS-less GeoTIFF.

    ``gdal_transform`` is GDAL-order ``(origin_x, scale_x, skew_x, origin_y,
    skew_y, scale_y)``; ``nodata`` (optional) becomes the per-band nodata of
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


def rasterio_clip(path, geom_wkt: str, *, all_touched, nodata, crop):
    """Reference clip composed from rasterio primitives over a GeoTIFF.

    The window is ``rasterio.features.geometry_window`` (geometry bounds ∩
    raster extent, snapped outward to the grid), the selection is
    ``rasterio.features.geometry_mask``, and pixels compose as "inside the
    geometry: source value verbatim; outside: ``nodata``". This is
    deliberately not ``rasterio.mask.mask``, which additionally reads the
    source masked and so remaps pixels *valued* at the band nodata to the
    output ``nodata`` — a policy choice about source-nodata pixels, separate
    from the geometry selection being validated here.

    ``nodata`` must be the already-resolved fill value (the caller applies the
    explicit-argument / band-nodata / dtype-minimum precedence). Returns
    ``(array(bands, h, w), gdal_transform)``.

    One caveat on independence: both engines ultimately burn the geometry with
    GDAL's rasterizer (sedonadb via the system GDAL, rasterio via its bundled
    one), so the pixel-center/all_touched selection rule itself is shared. The
    genuinely independent implementations under comparison are everything on
    top of the burn: window snapping, crop copy, transform shift, nodata
    precedence, and band handling.
    """
    import rasterio
    import rasterio.features
    import rasterio.windows
    import shapely

    geom = shapely.from_wkt(geom_wkt)
    with rasterio.open(str(path)) as src:
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
        fill = np.asarray(nodata, dtype=data.dtype)
        out = np.where(inside, data, fill)
    return out, tuple(transform.to_gdal())


def raster_to_numpy(raster):
    """Extract ``(array(bands, h, w), gdal_transform, [nodata per band])``."""
    nodatas = [band.nodata for band in raster.bands]
    return raster.to_numpy(), tuple(raster.transform), nodatas


def run_clip_rows(con, tiff_path, rows):
    """Run ``RS_Clip`` once over an N-row table, one row per parameter combo.

    Options arrive as table columns rather than literals so the kernel runs
    its real array execution path (literals constant-fold). Each row is a dict
    with keys ``wkt``, ``band``, ``all_touched``, ``nodata`` (None = use the
    band's own), and ``crop``. Returns a list of ``Raster`` (or None for NULL
    rows) in row order.
    """
    table = pa.table(
        {
            "idx": pa.array(range(len(rows)), type=pa.int64()),
            "path": pa.array([str(tiff_path)] * len(rows), type=pa.utf8()),
            "wkt": pa.array([r["wkt"] for r in rows], type=pa.utf8()),
            "band": pa.array([r["band"] for r in rows], type=pa.int32()),
            "all_touched": pa.array([r["all_touched"] for r in rows], type=pa.bool_()),
            "nodata": pa.array([r["nodata"] for r in rows], type=pa.float64()),
            "crop": pa.array([r["crop"] for r in rows], type=pa.bool_()),
        }
    )
    view = "raster_parity_clip_rows"
    con.create_data_frame(table).to_view(view, overwrite=True)
    try:
        result = con.sql(
            "SELECT RS_Clip(RS_FromPath(path), band, ST_GeomFromText(wkt),"
            f"               all_touched, nodata, crop) AS r FROM {view} ORDER BY idx"
        ).to_arrow_table()["r"]
    finally:
        con.drop_view(view)

    return [
        result[i].as_py() if result[i].is_valid else None for i in range(len(result))
    ]
