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
"""SedonaDB vs Sedona Spark parity for ST_GeometryN.

Both engines read the index 0-based — 0 answers the first element — which
is the opposite of PostGIS, so these cases pin the basis on both sides.
Negative indices are left out: SedonaDB returns NULL where Sedona Spark
raises from JTS.
"""

import pytest

from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark


@pytest.mark.parametrize(
    "geom,index,expected",
    [
        pytest.param("MULTIPOINT ((1 1), (2 2), (3 3))", 0, "POINT (1 1)", id="first"),
        pytest.param("MULTIPOINT ((1 1), (2 2), (3 3))", 2, "POINT (3 3)", id="last"),
        pytest.param("MULTIPOINT ((1 1), (2 2), (3 3))", 3, [(None,)], id="oob"),
        pytest.param(
            "MULTILINESTRING ((1 1, 2 2), (3 3, 4 4))",
            1,
            "LINESTRING (3 3, 4 4)",
            id="multilinestring",
        ),
        pytest.param(
            "MULTIPOLYGON (((0 0, 1 1, 0 1, 0 0)), ((5 5, 6 6, 5 6, 5 5)))",
            1,
            "POLYGON ((5 5, 6 6, 5 6, 5 5))",
            id="multipolygon",
        ),
        pytest.param(
            "GEOMETRYCOLLECTION (POINT (10 10), LINESTRING (20 20, 30 30))",
            1,
            "LINESTRING (20 20, 30 30)",
            id="collection",
        ),
        pytest.param("POINT (1 1)", 0, "POINT (1 1)", id="single_self"),
        pytest.param("POINT (1 1)", 1, [(None,)], id="single_oob"),
    ],
)
def test_st_geometryn(geom, index, expected):
    """A 0-based index addresses the same element on both engines, and a
    non-collection answers itself at index 0."""
    sedona, spark = SedonaDB(), SedonaSpark()
    sql = f"SELECT ST_GeometryN(ST_GeomFromWKT('{geom}'), {index})"
    compare(sql, sedona, spark, expected=expected)
