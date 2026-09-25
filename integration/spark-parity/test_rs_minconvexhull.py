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

"""SedonaDB vs Sedona Spark parity for RS_MinConvexHull.

Each case writes one GeoTIFF of nodata with a few data pixels that both
engines read. The hull is the outer corners of the cells bounding the data
pixels, in the same ring order on both engines (upper-left, upper-right,
lower-right, lower-left), so the anchor is the exact WKT.

The skewed case needs exactly representable coefficients: SedonaDB maps a
corner as (x0 + col*sx) + row*kx and Sedona Spark's AffineTransform as
(col*sx + row*kx) + x0, which differ in the last bit for ordinary decimal
skews. Two differences from the 1.9.1 release the suite pins, both fixed on
Sedona's main branch by apache/sedona#3366, are xfails until the pin moves:
an all-nodata raster (NULL) and a NaN nodata value (matching NaN pixels).
"""

import numpy as np
import pytest

from sedonadb.raster_testing import write_geotiff
from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark

# Grids for the 7x6 fixtures: north-up by bbox, skewed by a raw GDAL transform
# (a bbox cannot express skew).
NORTH_UP = {"bbox": (100, 482, 114, 500)}
SKEWED = {"gdal_transform": (100.0, 2.0, 0.5, 500.0, 0.25, -3.0)}


def _engines(name, tmp_path, data, *, grid=NORTH_UP, nodata=0, crs=None):
    path = tmp_path / f"{name}.tif"
    write_geotiff(path, data, **grid, nodata=nodata, crs=crs)
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_raster_view(name, path)
    return sedona, spark


def _sparse():
    """A 7x6 raster of nodata (0) with data in band 1 at rows 1-2, columns 2-3,
    and in band 2 at row 4, column 5."""
    data = np.zeros((2, 6, 7), dtype="uint8")
    data[0, 1, 3] = 10
    data[0, 2, 2] = 20
    data[1, 4, 5] = 30
    return data


@pytest.mark.parametrize(
    "band_arg,expected",
    [
        pytest.param(
            ", 1",
            "POLYGON ((104 497, 108 497, 108 491, 104 491, 104 497))",
            id="band-1",
        ),
        pytest.param(
            ", 2",
            "POLYGON ((110 488, 112 488, 112 485, 110 485, 110 488))",
            id="band-2",
        ),
        # Without a band, a cell holds data when any band holds data there.
        pytest.param(
            "",
            "POLYGON ((104 497, 112 497, 112 485, 104 485, 104 497))",
            id="all-bands",
        ),
    ],
)
def test_rs_minconvexhull(band_arg, expected, tmp_path):
    sedona, spark = _engines("mch_src", tmp_path, _sparse())
    sql = f"SELECT RS_MinConvexHull(rast{band_arg}) FROM mch_src"
    compare(sql, sedona, spark, expected=expected)


def test_rs_minconvexhull_skewed(tmp_path):
    """On a skewed grid the ring follows the grid axes: each corner is the
    affine image of a cell corner, not a world-space bounding box."""
    sedona, spark = _engines("mch_skew_src", tmp_path, _sparse(), grid=SKEWED)
    compare(
        "SELECT RS_MinConvexHull(rast, 1) FROM mch_skew_src",
        sedona,
        spark,
        expected="POLYGON ((104.5 497.5, 108.5 498, 109.5 492, 105.5 491.5, 104.5 497.5))",
    )


def test_rs_minconvexhull_crs(tmp_path):
    """Both engines carry the raster's CRS on the hull (SedonaDB as an
    item-level CRS, Sedona Spark as a per-geometry SRID)."""
    sedona, spark = _engines("mch_crs_src", tmp_path, _sparse(), crs="EPSG:3857")
    hull = "POLYGON ((110 488, 112 488, 112 485, 110 485, 110 488))"
    compare(
        "SELECT RS_MinConvexHull(rast, 2) FROM mch_crs_src",
        sedona,
        spark,
        expected=[(("EPSG:3857", hull),)],
    )


def test_rs_minconvexhull_null_band(tmp_path):
    sedona, spark = _engines("mch_src", tmp_path, _sparse())
    sql = "SELECT RS_MinConvexHull(rast, CAST(NULL AS INT)) FROM mch_src"
    compare(sql, sedona, spark, expected=[(None,)])


def test_rs_minconvexhull_without_nodata(tmp_path):
    """A band with no nodata value holds data everywhere, so the hull is the
    whole grid (zeros included)."""
    sedona, spark = _engines(
        "mch_none_src", tmp_path, np.zeros((1, 6, 7), dtype="uint8"), nodata=None
    )
    compare(
        "SELECT RS_MinConvexHull(rast) FROM mch_none_src",
        sedona,
        spark,
        expected="POLYGON ((100 500, 114 500, 114 482, 100 482, 100 500))",
    )


def test_rs_minconvexhull_band_out_of_range(tmp_path):
    """Both engines refuse a band the raster does not have. Error types
    differ, so parity here is parity on refusal."""
    sedona, spark = _engines("mch_src", tmp_path, _sparse())
    sql = "SELECT RS_MinConvexHull(rast, 3) FROM mch_src"
    for eng in (sedona, spark):
        with pytest.raises(Exception):
            eng.result_to_tuples(eng.execute_and_collect(sql))


@pytest.mark.xfail(
    reason="Sedona Spark 1.9.1 builds a polygon from its Integer.MAX_VALUE "
    "sentinels when no pixel holds data; apache/sedona#3366 made it NULL "
    "after the release"
)
def test_rs_minconvexhull_all_nodata(tmp_path):
    sedona, spark = _engines(
        "mch_empty_src", tmp_path, np.zeros((1, 6, 7), dtype="uint8")
    )
    compare(
        "SELECT RS_MinConvexHull(rast) FROM mch_empty_src",
        sedona,
        spark,
        expected=[(None,)],
    )


@pytest.mark.xfail(
    reason="Sedona Spark 1.9.1 treats a NaN nodata value as matching no pixel, "
    "so the NaN margin counts as data; apache/sedona#3366 fixed it after the "
    "release"
)
def test_rs_minconvexhull_nan_nodata(tmp_path):
    """NaN pixels are nodata under a NaN nodata value, so the NaN margin is
    trimmed off."""
    data = np.full((1, 6, 7), np.nan, dtype="float32")
    data[0, 1, 3] = 1.5
    data[0, 2, 2] = 2.5
    sedona, spark = _engines("mch_nan_src", tmp_path, data, nodata=np.nan)
    compare(
        "SELECT RS_MinConvexHull(rast, 1) FROM mch_nan_src",
        sedona,
        spark,
        expected="POLYGON ((104 497, 108 497, 108 491, 104 491, 104 497))",
    )
