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

"""RS_MinConvexHull against a rasterio reference.

Each fixture is written to a GeoTIFF and read back through RS_FromPath. The
reference finds the data pixels with numpy and maps the outer corners of their
bounding cells through the file's affine transform. A skewed transform is in
the parameter set because skew is what separates mapping the four corners from
taking a world-space bounding box. The transforms and cell indices are small
dyadic values, so both sides compute every coordinate exactly and compare with
`==`.
"""

import numpy as np
import pyarrow as pa
import pytest

from sedonadb.raster_testing import write_geotiff

rasterio = pytest.importorskip("rasterio")
shapely = pytest.importorskip("shapely")

TRANSFORMS = {
    "north-up": (100.0, 2.0, 0.0, 500.0, 0.0, -3.0),
    "skewed": (100.0, 2.0, 0.5, 500.0, 0.25, -3.0),
}


def _rasterio_min_convex_hull(path, bands):
    with rasterio.open(path) as src:
        nodata = src.nodatavals[0]
        pixels = src.read(bands)
        affine = src.transform
    rows, cols = np.nonzero((pixels != nodata).any(axis=0))
    if len(rows) == 0:
        return None
    left, top, right, bottom = cols.min(), rows.min(), cols.max() + 1, rows.max() + 1
    ring = [(left, top), (right, top), (right, bottom), (left, bottom), (left, top)]
    return [affine * (int(col), int(row)) for col, row in ring]


def _sedonadb_min_convex_hull(con, path, band=None):
    """Arguments travel as table columns so the kernel runs its real array
    path (literals constant-fold)."""
    con.create_data_frame(
        pa.table({"path": [str(path)], "band": pa.array([band], pa.int32())})
    ).to_view("minconvexhull_src", overwrite=True)
    band_arg = "" if band is None else ", band"
    sql = (
        f"SELECT ST_AsText(RS_MinConvexHull(RS_FromPath(path){band_arg})) "
        "FROM minconvexhull_src"
    )
    wkt = con.sql(sql).to_arrow_table().column(0)[0].as_py()
    if wkt is None:
        return None
    return list(shapely.from_wkt(wkt).exterior.coords)


@pytest.fixture(params=list(TRANSFORMS), ids=list(TRANSFORMS))
def tiff(request, tmp_path):
    """A 7x6 two-band raster of nodata (0) with data in band 1 at rows 1-2,
    columns 2-3, and in band 2 at row 4, column 5."""
    data = np.zeros((2, 6, 7), dtype="uint8")
    data[0, 1, 3] = 10
    data[0, 2, 2] = 20
    data[1, 4, 5] = 30
    path = tmp_path / f"minconvexhull_{request.param}.tif"
    write_geotiff(path, data, gdal_transform=TRANSFORMS[request.param], nodata=0)
    return path


@pytest.mark.parametrize("band", [1, 2])
def test_rs_minconvexhull_one_band(con, tiff, band):
    expected = _rasterio_min_convex_hull(tiff, [band])
    assert _sedonadb_min_convex_hull(con, tiff, band) == expected


def test_rs_minconvexhull_all_bands(con, tiff):
    """Without a band, a cell holds data when any band holds data there."""
    expected = _rasterio_min_convex_hull(tiff, [1, 2])
    assert _sedonadb_min_convex_hull(con, tiff) == expected


def test_rs_minconvexhull_all_nodata_is_null(con, tmp_path):
    path = tmp_path / "all_nodata.tif"
    write_geotiff(
        path,
        np.zeros((1, 6, 7), dtype="uint8"),
        gdal_transform=TRANSFORMS["north-up"],
        nodata=0,
    )
    assert _rasterio_min_convex_hull(path, [1]) is None
    assert _sedonadb_min_convex_hull(con, path) is None


def test_rs_minconvexhull_band_out_of_range(con, tiff):
    with pytest.raises(Exception, match="RS_MinConvexHull"):
        _sedonadb_min_convex_hull(con, tiff, 3)
