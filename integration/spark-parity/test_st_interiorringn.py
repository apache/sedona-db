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
"""SedonaDB vs Sedona Spark parity for ST_InteriorRingN.

SedonaDB reads the ring index 1-based, as PostGIS does — 1 answers the first
hole — while the pinned Sedona Spark 1.9.1 reads it 0-based. Every case
anchors the PostGIS answer; the ones where the engines diverge xfail until
apache/sedona#3403 ships in Sedona 2.0. Negative indices are left out:
Sedona Spark 1.9.1 raises from JTS on them.
"""

import pytest

from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark

SPARK_0_BASED = pytest.mark.xfail(
    reason="SedonaDB reads the ring index 1-based, as PostGIS does; the pinned "
    "Sedona Spark 1.9.1 reads it 0-based until apache/sedona#3403 ships in "
    "Sedona 2.0"
)

TWO_HOLES = (
    "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0), "
    "(1 1, 1 2, 2 2, 2 1, 1 1), (4 4, 4 5, 5 5, 5 4, 4 4))"
)


@pytest.mark.parametrize(
    "geom,index,expected",
    [
        pytest.param(
            TWO_HOLES,
            1,
            "LINESTRING (1 1, 1 2, 2 2, 2 1, 1 1)",
            id="first",
            marks=SPARK_0_BASED,
        ),
        pytest.param(
            TWO_HOLES,
            2,
            "LINESTRING (4 4, 4 5, 5 5, 5 4, 4 4)",
            id="second",
            marks=SPARK_0_BASED,
        ),
        pytest.param(TWO_HOLES, 3, [(None,)], id="oob"),
        pytest.param(TWO_HOLES, 0, [(None,)], id="zero", marks=SPARK_0_BASED),
        pytest.param(
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", 1, [(None,)], id="no_holes"
        ),
        pytest.param("POINT (0 0)", 1, [(None,)], id="not_polygon"),
    ],
)
def test_st_interiorringn(geom, index, expected):
    """A 1-based index addresses the same hole on both engines."""
    sedona, spark = SedonaDB(), SedonaSpark()
    sql = f"SELECT ST_InteriorRingN(ST_GeomFromWKT('{geom}'), {index})"
    compare(sql, sedona, spark, expected=expected)
