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
"""SedonaDB vs Sedona Spark parity for RS_Value.

Point reads compare raw and exact on every probed dtype, including the
NULL for a nodata pixel and for a point outside the extent. A point on
a shared pixel corner resolves to the pixel to its lower right on both
engines. One module per RS_ function; xfails per the divergence-catalog
convention.
"""

import pytest

from sedonadb.raster_testing import random_raster_data
from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark


def _engines(name, tmp_path, **kwargs):
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(name, tmp_path / f"{name}.tif", **kwargs)
    return sedona, spark


def test_rs_value_planted(tmp_path):
    """A planted value reads back from the centre of its pixel, and the band
    argument addresses the band."""
    sedona, spark = _engines("val_src", tmp_path, plants={(1, 1): 42})
    sql = "SELECT RS_Value(rast, ST_GeomFromWKT('POINT(103 495.5)'), 1) FROM val_src"
    compare(sql, sedona, spark, expected=42.0)
    data = random_raster_data("uint8", bands=2, height=6, width=7, plants={(1, 1): 42})
    sql = "SELECT RS_Value(rast, ST_GeomFromWKT('POINT(105 491)'), 2) FROM val_src"
    compare(sql, sedona, spark, expected=float(data[1, 3, 2]))


def test_rs_value_nodata_pixel_is_null(tmp_path):
    """A pixel holding the band's nodata reads back NULL from both engines."""
    sedona, spark = _engines(
        "val_nd_src", tmp_path, nodata=200.0, plants={(2, 2): 200.0}
    )
    sql = "SELECT RS_Value(rast, ST_GeomFromWKT('POINT(105 492.5)'), 1) FROM val_nd_src"
    compare(sql, sedona, spark, expected=[(None,)])


def test_rs_value_outside_extent_is_null(tmp_path):
    """A point outside the raster's extent reads back NULL from both engines."""
    sedona, spark = _engines("val_out_src", tmp_path)
    sql = "SELECT RS_Value(rast, ST_GeomFromWKT('POINT(300 300)'), 1) FROM val_out_src"
    compare(sql, sedona, spark, expected=[(None,)])


def test_rs_value_float64_fraction(tmp_path):
    """A fractional float64 value reads back exactly."""
    sedona, spark = _engines(
        "val_f_src", tmp_path, dtype="float64", bands=1, plants={(1, 1): -0.5}
    )
    sql = "SELECT RS_Value(rast, ST_GeomFromWKT('POINT(103 495.5)'), 1) FROM val_f_src"
    compare(sql, sedona, spark, expected=-0.5)


def test_rs_value_pixel_corner(tmp_path):
    """A point exactly on the shared corner of four pixels resolves to the
    pixel to its lower right on both engines: (104, 494) sits where rows 1-2
    and columns 1-2 meet and reads pixel (row 2, col 2)."""
    sedona, spark = _engines("val_c_src", tmp_path)
    data = random_raster_data("uint8", bands=2, height=6, width=7)
    sql = "SELECT RS_Value(rast, ST_GeomFromWKT('POINT(104 494)'), 1) FROM val_c_src"
    compare(sql, sedona, spark, expected=float(data[0, 2, 2]))


@pytest.mark.parametrize(
    "wkt,pixel",
    [
        # The lower-right convention keeps the upper-left raster corner inside
        # (pixel (0, 0) holds the planted dtype max)...
        pytest.param("POINT(100 500)", (0, 0), id="raster-corner-upper-left"),
        # ...and top/left edge midpoints resolve inclusively.
        pytest.param("POINT(104 500)", (0, 2), id="top-edge"),
        pytest.param("POINT(100 494)", (2, 0), id="left-edge"),
    ],
)
def test_rs_value_boundary_inclusive(wkt, pixel, tmp_path):
    """Points on the raster's top/left boundary read the adjacent pixel on
    both engines."""
    sedona, spark = _engines("val_b_src", tmp_path)
    data = random_raster_data("uint8", bands=2, height=6, width=7)
    sql = f"SELECT RS_Value(rast, ST_GeomFromWKT('{wkt}'), 1) FROM val_b_src"
    compare(sql, sedona, spark, expected=float(data[0][pixel]))


@pytest.mark.parametrize(
    "wkt",
    [
        pytest.param("POINT(114 482)", id="raster-corner-lower-right"),
        pytest.param("POINT(114 500)", id="raster-corner-upper-right"),
        pytest.param("POINT(100 482)", id="raster-corner-lower-left"),
    ],
)
def test_rs_value_boundary_exclusive(wkt, tmp_path):
    """The other three raster corners are NULL on both engines — the
    lower-right convention steps outside the grid there, so the right and
    bottom boundaries are exclusive."""
    sedona, spark = _engines("val_bx_src", tmp_path)
    sql = f"SELECT RS_Value(rast, ST_GeomFromWKT('{wkt}'), 1) FROM val_bx_src"
    compare(sql, sedona, spark, expected=[(None,)])


def test_rs_value_non_point_rejected(tmp_path):
    """Both engines refuse a non-point geometry. Error types differ, so
    parity here is parity on refusal."""
    sedona, spark = _engines("val_l_src", tmp_path)
    sql = (
        "SELECT RS_Value(rast, ST_GeomFromWKT("
        "'LINESTRING(103 495.5, 105 492.5)'), 1) FROM val_l_src"
    )
    for eng in (sedona, spark):
        with pytest.raises(Exception):
            eng.result_to_tuples(eng.execute_and_collect(sql))


@pytest.mark.xfail(
    reason="SedonaDB requires the band argument on a multi-band raster; "
    "Sedona Spark's band-less form defaults to band 1"
)
def test_rs_value_two_arg_multiband(tmp_path):
    """The band-less form gets the same answer from both engines on a
    multi-band raster."""
    sedona, spark = _engines("val_2a_src", tmp_path)
    sql = "SELECT RS_Value(rast, ST_GeomFromWKT('POINT(103 495.5)')) FROM val_2a_src"
    compare(sql, sedona, spark)


@pytest.mark.xfail(
    reason="SedonaDB has no grid-coordinate overload of RS_Value (it raises "
    "'No kernel matching arguments'); Sedona Spark answers"
)
def test_rs_value_grid_coordinate_overload(tmp_path):
    """RS_Value(raster, colX, rowY, band) answers from both engines."""
    sedona, spark = _engines("val_g_src", tmp_path)
    sql = "SELECT RS_Value(rast, 2, 3, 1) FROM val_g_src"
    compare(sql, sedona, spark)


@pytest.mark.xfail(
    reason="SedonaDB rejects a point that carries a CRS when the raster has "
    "none; Sedona Spark computes as if the CRSs matched"
)
def test_rs_value_srid_point_on_crsless_raster(tmp_path):
    """A point with an SRID against a CRS-less raster gets the same treatment
    from both engines."""
    sedona, spark = _engines("val_s_src", tmp_path)
    sql = (
        "SELECT RS_Value(rast, ST_SetSRID(ST_GeomFromWKT("
        "'POINT(103 495.5)'), 4326), 1) FROM val_s_src"
    )
    compare(sql, sedona, spark)
