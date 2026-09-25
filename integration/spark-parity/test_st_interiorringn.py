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

Both engines read the ring index 0-based — 0 answers the first hole — which
is the opposite of PostGIS, so these cases pin the basis on both sides.
Negative indices are left out: SedonaDB returns NULL where Sedona Spark
raises from JTS.
"""

import pytest

from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark

TWO_HOLES = (
    "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0), "
    "(1 1, 1 2, 2 2, 2 1, 1 1), (4 4, 4 5, 5 5, 5 4, 4 4))"
)


@pytest.mark.parametrize(
    "geom,index,expected",
    [
        pytest.param(TWO_HOLES, 0, "LINESTRING (1 1, 1 2, 2 2, 2 1, 1 1)", id="first"),
        pytest.param(TWO_HOLES, 1, "LINESTRING (4 4, 4 5, 5 5, 5 4, 4 4)", id="second"),
        pytest.param(TWO_HOLES, 2, [(None,)], id="oob"),
        pytest.param(
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", 0, [(None,)], id="no_holes"
        ),
        pytest.param("POINT (0 0)", 0, [(None,)], id="not_polygon"),
    ],
)
def test_st_interiorringn(geom, index, expected):
    """A 0-based index addresses the same hole on both engines."""
    sedona, spark = SedonaDB(), SedonaSpark()
    sql = f"SELECT ST_InteriorRingN(ST_GeomFromWKT('{geom}'), {index})"
    compare(sql, sedona, spark, expected=expected)
