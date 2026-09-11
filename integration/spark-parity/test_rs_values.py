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
"""SedonaDB vs Sedona Spark parity for RS_Values.

A pure divergence catalog today: every spelling of RS_Values is accepted
by exactly one engine. SedonaDB reads sample locations from a single
(multi-)geometry; Sedona Spark reads them only from an ARRAY of
geometries — each rejects the other's spelling — and Sedona Spark
additionally has a coordinate-array overload SedonaDB lacks, whose
coordinates it reads 0-based (unlike its own 1-based RS_PixelAs*
functions). The shared `array(...)` these cases rely on is SedonaDB's
Spark-compat spelling of make_array, registered from datafusion-spark.

Sample locations on the standard seeded grid: (101 499) is pixel
(row 0, col 0) = 255 and (103 493) is pixel (row 2, col 1) = 112; the
coordinate arrays x = [1, 2], y = [1, 1] read 0-based land on 193
and 255.
"""

import pytest

from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark

GEOM_ARRAY = (
    "array(ST_GeomFromWKT('POINT (101 499)'), ST_GeomFromWKT('POINT (103 493)'))"
)


def _engines(name, tmp_path):
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(name, tmp_path / f"{name}.tif")
    return sedona, spark


@pytest.mark.parametrize(
    "band", [pytest.param("", id="bandless"), pytest.param(", 1", id="band-1")]
)
@pytest.mark.xfail(
    reason="SedonaDB has no array-of-geometries form of RS_Values (it raises "
    "'No kernel matching arguments'); Sedona Spark reads sample locations "
    "only from an array and answers [255.0, 112.0]"
)
def test_rs_values_geometry_array(band, tmp_path):
    """An array of point geometries samples the same values on both
    engines."""
    sedona, spark = _engines("vals_arr_src", tmp_path)
    sql = f"SELECT RS_Values(rast, {GEOM_ARRAY}{band}) FROM vals_arr_src"
    compare(sql, sedona, spark)


@pytest.mark.xfail(
    reason="SedonaDB samples a bare MULTIPOINT ([255.0, 112.0], one value "
    "per part); Sedona Spark rejects a non-array geometry argument "
    "(DATATYPE_MISMATCH — its geometry form takes array<geometry>)"
)
def test_rs_values_multipoint(tmp_path):
    """A bare MULTIPOINT samples the same values on both engines."""
    sedona, spark = _engines("vals_mp_src", tmp_path)
    sql = (
        "SELECT RS_Values(rast, "
        "ST_GeomFromWKT('MULTIPOINT ((101 499), (103 493))'), 1) FROM vals_mp_src"
    )
    compare(sql, sedona, spark)


@pytest.mark.xfail(
    reason="SedonaDB has no coordinate-array overload of RS_Values (it "
    "raises 'No kernel matching arguments'); Sedona Spark reads the arrays "
    "0-based — unlike its own 1-based RS_PixelAs* functions — and answers "
    "[193.0, 255.0]"
)
def test_rs_values_coordinate_arrays(tmp_path):
    """RS_Values(raster, xCoordinates, yCoordinates, band) answers from
    both engines."""
    sedona, spark = _engines("vals_xy_src", tmp_path)
    sql = "SELECT RS_Values(rast, array(1, 2), array(1, 1), 1) FROM vals_xy_src"
    compare(sql, sedona, spark)
