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
from sedonadb.testing import geom_or_null, PostGIS, SedonaDB


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom1", "geom2", "density_frac", "expected"),
    [
        (None, None, None, None),
        ("POINT (0 0)", None, None, None),
        ("LINESTRING (0 0, 2 0)", "LINESTRING (0 1, 1 2, 2 1)", None, 1.4142135623730951),
        # Case with density fraction
        ("LINESTRING (0 0, 100 0)", "LINESTRING (0 0, 50 1, 100 0)", 0.5, 1.0),
        # Identical geometries
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", None, 0.0),
        # Disjoint geometries
        ("POINT (0 0)", "POINT (10 10)", None, 14.142135623730951),
    ],
)
def test_st_hausdorff_distance(eng, geom1, geom2, density_frac, expected):
    eng = eng.create_or_skip()

    if density_frac is None:
        query = f"SELECT ST_HausdorffDistance({geom_or_null(geom1)}, {geom_or_null(geom2)})"
    else:
        query = f"SELECT ST_HausdorffDistance({geom_or_null(geom1)}, {geom_or_null(geom2)}, {density_frac})"

    eng.assert_query_result(
        query,
        expected,
        numeric_epsilon=1e-8,
    )