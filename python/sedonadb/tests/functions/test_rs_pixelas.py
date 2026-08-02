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

"""RS_PixelAsPoint / RS_PixelAsCentroid / RS_PixelAsPolygon parity.

Pixel coordinates are 1-based `(col, row)` and the point convention is the
pixel's upper-left corner; the rasterio reference computes the same
locations from the affine transform directly. A skewed geotransform is
always in the parameter set because skew is what separates correct affine
math from scale-only shortcuts. SedonaDB extrapolates out-of-bounds pixel
coordinates through the same affine math, so the out-of-bounds pixel runs
against the affine reference.
"""

import numpy as np
import pyarrow as pa
import pytest

from sedonadb.raster_testing import (
    approx_geotransform,
    random_raster_data,
    write_geotiff,
)

pytest.importorskip("rasterio")
shapely = pytest.importorskip("shapely")

TRANSFORMS = {
    "north-up": (100.0, 2.0, 0.0, 500.0, 0.0, -3.0),
    "skewed": (100.0, 2.0, 0.5, 500.0, 0.25, -3.0),
}
HEIGHT, WIDTH = 6, 7
# 1-based (col, row): the first pixel, an interior pixel, the last pixel,
# and an out-of-bounds pixel past both edges.
PIXELS = [(1, 1), (2, 3), (WIDTH, HEIGHT), (WIDTH + 2, HEIGHT + 2)]


@pytest.fixture(params=list(TRANSFORMS), ids=list(TRANSFORMS))
def tiff(request, tmp_path):
    path = tmp_path / f"pixelas_{request.param}.tif"
    write_geotiff(
        path,
        random_raster_data("uint8", bands=1, height=HEIGHT, width=WIDTH),
        gdal_transform=TRANSFORMS[request.param],
    )
    return path


def _sedonadb_pixel_geometry(con, function, path, col, row):
    """Run one `RS_PixelAs*` accessor and return the shapely geometry. Arguments
    travel as table columns so the kernel runs its real array path (literals
    constant-fold)."""
    df = con.create_data_frame(
        pa.table(
            {
                "path": pa.array([str(path)], pa.utf8()),
                "col": pa.array([int(col)], pa.int32()),
                "row": pa.array([int(row)], pa.int32()),
            }
        )
    )
    raster = df.path.funcs.rs_frompath()
    expr = getattr(raster.funcs, function)(df.col, df.row)
    wkt = df.select(g=expr.funcs.st_astext()).to_arrow_table()["g"][0].as_py()
    return shapely.from_wkt(wkt)


def _sedonadb_pixel_as_point(con, path, col, row):
    point = _sedonadb_pixel_geometry(con, "rs_pixelaspoint", path, col, row)
    return (point.x, point.y)


def _sedonadb_pixel_as_centroid(con, path, col, row):
    point = _sedonadb_pixel_geometry(con, "rs_pixelascentroid", path, col, row)
    return (point.x, point.y)


def _sedonadb_pixel_as_polygon(con, path, col, row):
    return _sedonadb_pixel_geometry(con, "rs_pixelaspolygon", path, col, row)


def _rasterio_pixel_as_point(path, col, row):
    import rasterio

    with rasterio.open(str(path)) as src:
        x, y = src.transform * (col - 1, row - 1)
        return (x, y)


def _rasterio_pixel_as_centroid(path, col, row):
    import rasterio

    with rasterio.open(str(path)) as src:
        x, y = src.transform * (col - 0.5, row - 0.5)
        return (x, y)


def _rasterio_pixel_as_polygon(path, col, row):
    import rasterio

    with rasterio.open(str(path)) as src:
        # Ring order matches both dialects: UL, UR, LR, LL (closed).
        corners = [
            (col - 1, row - 1),
            (col, row - 1),
            (col, row),
            (col - 1, row),
        ]
        return shapely.Polygon([src.transform * corner for corner in corners])


@pytest.mark.parametrize(("col", "row"), PIXELS)
def test_rs_pixelaspoint_matches_comparators(con, tiff, col, row):
    got = _sedonadb_pixel_as_point(con, tiff, col, row)
    assert got == approx_geotransform(_rasterio_pixel_as_point(tiff, col, row))


@pytest.mark.parametrize(("col", "row"), PIXELS)
def test_rs_pixelascentroid_matches_comparators(con, tiff, col, row):
    got = _sedonadb_pixel_as_centroid(con, tiff, col, row)
    assert got == approx_geotransform(_rasterio_pixel_as_centroid(tiff, col, row))


@pytest.mark.parametrize(("col", "row"), PIXELS)
def test_rs_pixelaspolygon_matches_comparators(con, tiff, col, row):
    got = _sedonadb_pixel_as_polygon(con, tiff, col, row)
    expected = _rasterio_pixel_as_polygon(tiff, col, row)
    # Coordinate-sequence comparison pins the shared ring convention
    # (UL, UR, LR, LL, closed), not just topological equality.
    np.testing.assert_allclose(
        shapely.get_coordinates(got),
        shapely.get_coordinates(expected),
        rtol=1e-12,
        atol=1e-12,
    )


def test_rs_pixelas_sql_text_smoke(con, tmp_path):
    """One SQL-text invocation per pixel function so the parser path stays
    covered (everything else in this module routes through the helpers).
    Pixel (2, 3) of the north-up grid: upper-left corner (102, 494), pixel
    extent x [102, 104], y [491, 494]."""
    path = tmp_path / "smoke.tif"
    write_geotiff(
        path,
        random_raster_data("uint8", bands=1, height=HEIGHT, width=WIDTH),
        gdal_transform=TRANSFORMS["north-up"],
    )

    expected = {
        "RS_PixelAsPoint": shapely.Point(102, 494),
        "RS_PixelAsCentroid": shapely.Point(103, 492.5),
        "RS_PixelAsPolygon": shapely.Polygon(
            [(102, 494), (104, 494), (104, 491), (102, 491)]
        ),
    }
    for function, want in expected.items():
        got = (
            con.sql(
                f"SELECT ST_AsText({function}(RS_FromPath($1), 2, 3)) AS g",
                params=(str(path),),
            )
            .to_arrow_table()["g"][0]
            .as_py()
        )
        assert shapely.equals_exact(shapely.from_wkt(got), want), (function, got)
