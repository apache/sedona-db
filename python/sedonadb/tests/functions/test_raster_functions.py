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

import pytest

from sedonadb.testing import SedonaDB


def query_value(con, expr):
    """Evaluate `expr` over the single example raster row and return the value."""
    table = con.sql(f"SELECT {expr} AS v FROM rasters").to_arrow_table()
    return table["v"][0].as_py()


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("RS_NumBands(RS_Example())", 3),
        ("RS_Width(RS_Example())", 64),
        ("RS_Height(RS_Example())", 32),
        ("RS_BandPixelType(RS_Example(), 1)", "UNSIGNED_8BITS"),
        ("RS_BandNoDataValue(RS_Example(), 1)", 127.0),
        ("RS_ScaleX(RS_Example())", 2.0),
        ("RS_ScaleY(RS_Example())", 2.0),
        ("RS_SkewX(RS_Example())", 1.0),
        ("RS_SkewY(RS_Example())", 1.0),
        ("RS_UpperLeftX(RS_Example())", 43.08),
        ("RS_UpperLeftY(RS_Example())", 79.07),
    ],
)
def test_rs_function(expr, expected):
    eng = SedonaDB()
    eng.assert_query_result(f"SELECT {expr}", expected)


# EPSG:3857 as WKT (carries an embedded EPSG authority) and a bespoke Lambert
# Conformal Conic WKT with no authority code anywhere.
WKT_3857 = (
    'PROJCS["WGS 84 / Pseudo-Mercator",GEOGCS["WGS 84",DATUM["WGS_1984",'
    'SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],'
    'AUTHORITY["EPSG","6326"]],AUTHORITY["EPSG","4326"]],'
    'PROJECTION["Mercator_1SP"],AUTHORITY["EPSG","3857"]]'
)
WKT_LCC_NO_AUTHORITY = (
    'PROJCS["Custom LCC",GEOGCS["WGS 84",DATUM["WGS_1984",'
    'SPHEROID["WGS 84",6378137,298.257223563]]],'
    'PROJECTION["Lambert_Conformal_Conic_2SP"],'
    'PARAMETER["standard_parallel_1",33],PARAMETER["standard_parallel_2",45],'
    'PARAMETER["latitude_of_origin",39],PARAMETER["central_meridian",-96],'
    'UNIT["metre",1]]'
)


# WKT1/WKT2 CRS strings round-trip through RS_SetCRS/RS_CRS unchanged, whether or
# not they carry an embedded authority.
@pytest.mark.parametrize("wkt", [WKT_3857, WKT_LCC_NO_AUTHORITY])
def test_rs_setcrs_wkt_roundtrips(wkt):
    eng = SedonaDB()
    eng.assert_query_result(f"SELECT RS_CRS(RS_SetCRS(RS_Example(), '{wkt}'))", wkt)


def test_rs_srid_from_wkt():
    """A WKT carrying an EPSG authority resolves to that SRID."""
    eng = SedonaDB()
    eng.assert_query_result(
        f"SELECT RS_SRID(RS_SetCRS(RS_Example(), '{WKT_3857}'))", 3857
    )


def test_rs_srid_from_authorityless_wkt_errors(con):
    """A WKT with no authority code anywhere has no SRID to extract."""
    with pytest.raises(Exception, match="SRID"):
        con.sql(
            f"SELECT RS_SRID(RS_SetCRS(RS_Example(), '{WKT_LCC_NO_AUTHORITY}'))"
        ).to_arrow_table()


def test_rs_ensureloaded(con, sedona_testing):
    path = sedona_testing / "data/raster/sentinel2.tif"
    t = con.sql("SELECT RS_FromPath($1) AS raster", params=(str(path),))
    tab = t.select(raster=t.raster.funcs.rs_ensureloaded()).to_arrow_table()
    r = tab["raster"][0].as_py()
    assert r.height == 512
    assert r.width == 512

    assert len(r.bands) == 1
    b = r.bands[0]
    assert b.shape == (512, 512)
    arr = b.to_numpy()
    assert arr.shape == (512, 512)
    assert arr.dtype == "uint16"
    assert arr[0, 0] == 2324


# RS_Example fills band `b` with the constant value `b`, except the top-left
# pixel (colX=1, rowY=1) which is set to the nodata value (127). Grid
# coordinates are 1-based. These also exercise the `needs_pixels` ->
# RS_EnsureLoaded planner path end to end (RS_Value reads materialised InDb
# bytes).
@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("RS_Value(raster, 2, 1)", 1.0),  # band 1, away from the nodata corner
        ("RS_Value(raster, 64, 32)", 1.0),  # bottom-right pixel, in bounds
        ("RS_Value(raster, 2, 1, 1)", 1.0),  # explicit band 1 == the default
        ("RS_Value(raster, 2, 1, 2)", 2.0),  # band 2
        ("RS_Value(raster, 2, 1, 3)", 3.0),  # band 3
        ("RS_Value(raster, 1, 1)", None),  # top-left pixel is nodata
        ("RS_Value(raster, 1, 1, 2)", None),  # nodata on band 2 too
        ("RS_Value(raster, 65, 1)", None),  # colX past the width (64)
        ("RS_Value(raster, 1, 33)", None),  # rowY past the height (32)
        ("RS_Value(raster, 0, 1)", None),  # colX 0 -> off the left edge
    ],
)
def test_rs_value_grid(raster_con, expr, expected):
    assert query_value(raster_con, expr) == expected


# Point sampling. (74.58, 110.57) is the centroid of pixel (10, 10) (0-based)
# in the raster's OGC:CRS84 space; the point and raster share a CRS so no
# reprojection happens. A point far outside the footprint yields NULL.
@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("RS_Value(raster, ST_SetCRS(ST_Point(74.58, 110.57), 'OGC:CRS84'))", 1.0),
        ("RS_Value(raster, ST_SetCRS(ST_Point(74.58, 110.57), 'OGC:CRS84'), 2)", 2.0),
        ("RS_Value(raster, ST_SetCRS(ST_Point(74.58, 110.57), 'OGC:CRS84'), 3)", 3.0),
        ("RS_Value(raster, ST_SetCRS(ST_Point(0.0, 0.0), 'OGC:CRS84'))", None),
    ],
)
def test_rs_value_point(raster_con, expr, expected):
    assert query_value(raster_con, expr) == expected
