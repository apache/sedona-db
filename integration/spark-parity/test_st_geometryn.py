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

SedonaDB reads the index 1-based, as PostGIS does — 1 answers the first
element — while the pinned Sedona Spark 1.9.1 reads it 0-based. Every case
anchors the PostGIS answer; the ones where the engines diverge xfail until
apache/sedona#3403 ships in Sedona 2.0. Negative indices are left out:
Sedona Spark 1.9.1 raises from JTS on them.
"""

import pytest

from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark

SPARK_0_BASED = pytest.mark.xfail(
    reason="SedonaDB reads the index 1-based, as PostGIS does; the pinned "
    "Sedona Spark 1.9.1 reads it 0-based until apache/sedona#3403 ships in "
    "Sedona 2.0"
)

MULTIPOINT = "MULTIPOINT ((1 1), (2 2), (3 3))"


@pytest.mark.parametrize(
    "geom,index,expected",
    [
        pytest.param(MULTIPOINT, 1, "POINT (1 1)", id="first", marks=SPARK_0_BASED),
        pytest.param(MULTIPOINT, 3, "POINT (3 3)", id="last", marks=SPARK_0_BASED),
        pytest.param(MULTIPOINT, 4, [(None,)], id="oob"),
        pytest.param(MULTIPOINT, 0, [(None,)], id="zero", marks=SPARK_0_BASED),
        pytest.param(
            "MULTILINESTRING ((1 1, 2 2), (3 3, 4 4))",
            2,
            "LINESTRING (3 3, 4 4)",
            id="multilinestring",
            marks=SPARK_0_BASED,
        ),
        pytest.param(
            "MULTIPOLYGON (((0 0, 1 1, 0 1, 0 0)), ((5 5, 6 6, 5 6, 5 5)))",
            2,
            "POLYGON ((5 5, 6 6, 5 6, 5 5))",
            id="multipolygon",
            marks=SPARK_0_BASED,
        ),
        pytest.param(
            "GEOMETRYCOLLECTION (POINT (10 10), LINESTRING (20 20, 30 30))",
            2,
            "LINESTRING (20 20, 30 30)",
            id="collection",
            marks=SPARK_0_BASED,
        ),
        pytest.param(
            "POINT (1 1)", 1, "POINT (1 1)", id="single_self", marks=SPARK_0_BASED
        ),
        pytest.param("POINT (1 1)", 2, [(None,)], id="single_oob"),
    ],
)
def test_st_geometryn(geom, index, expected):
    """A 1-based index addresses the same element on both engines, and a
    non-collection answers itself at index 1."""
    sedona, spark = SedonaDB(), SedonaSpark()
    sql = f"SELECT ST_GeometryN(ST_GeomFromWKT('{geom}'), {index})"
    compare(sql, sedona, spark, expected=expected)
