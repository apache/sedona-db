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
from sedonadb.testing import BigQuery, PostGIS, SedonaDB, geog_or_null
import sedonadb

if "s2geography" not in sedonadb.__features__:
    pytest.skip("Python package built without s2geography", allow_module_level=True)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS, BigQuery])
@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        (None, None, None),
        # Single arg nulls are not handled by SedonaDB until upgrade
        # ("POINT (0 0)", None, None),
        # (None, "POINT (0 0)", None),
        ("POINT (0 0)", "POINT (0 0)", 0),
        (
            "POINT(-72.1235 42.3521)",
            "LINESTRING(-72.1260 42.45, -72.123 42.1546)",
            123.47576072749062,  # 123.80207675
        ),
        (
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            "POLYGON ((5 5, 6 5, 6 6, 5 6, 5 5))",
            628519.1549911008,  # 627129.50261075
        ),
    ],
)
def test_st_distance(eng, geom1, geom2, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Distance({geog_or_null(geom1)}, {geog_or_null(geom2)})",
        expected,
        numeric_epsilon=1e-2,
    )
