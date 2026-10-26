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
"""SedonaDB vs Sedona Spark parity for ST_Affine.

Both engines take PostGIS' coefficient order — ST_Affine(geom, a, b, d, e,
xOff, yOff) in 2D and ST_Affine(geom, a, b, c, d, e, f, g, h, i, xOff, yOff,
zOff) in 3D, for x' = a*x + b*y (+ c*z) + xOff and so on. The matrices here
are not symmetric, so a transposed matrix gives a different answer; every case
anchors the PostGIS answer.
"""

import pytest

from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark


@pytest.mark.parametrize(
    "geom,coefficients,expected",
    [
        pytest.param("POINT (1 1)", "1, 2, 0, 1, 0, 0", "POINT (3 1)", id="shear"),
        pytest.param("POINT (1 2)", "2, 3, 5, 7, 1, 1", "POINT (9 20)", id="2d"),
        pytest.param(
            "LINESTRING (0 0, 1 1)",
            "2, 3, 5, 7, 1, 1",
            "LINESTRING (1 1, 6 13)",
            id="2d_line",
        ),
        pytest.param(
            "POINT Z (1 2 3)",
            "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12",
            "POINT Z (24 43 62)",
            id="3d",
        ),
    ],
)
def test_st_affine(geom, coefficients, expected):
    """The same coefficients move a geometry to the same place on both engines."""
    sedona, spark = SedonaDB(), SedonaSpark()
    sql = f"SELECT ST_Affine(ST_GeomFromWKT('{geom}'), {coefficients})"
    compare(sql, sedona, spark, expected=expected)
