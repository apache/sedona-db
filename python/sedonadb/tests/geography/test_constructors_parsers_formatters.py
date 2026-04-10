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
from sedonadb.testing import SedonaDB, BigQuery, PostGIS, val_or_null


@pytest.mark.parametrize("eng", [SedonaDB, BigQuery, PostGIS])
@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        (None, None, None),
        (1, None, None),
        (None, 1, None),
        (1, 1, "POINT (1 1)"),
        (1.0, 1.0, "POINT (1 1)"),
        (10, -1.5, "POINT (10 -1.5)"),
    ],
)
def test_st_geogpoint(eng, x, y, expected):
    eng = eng.create_or_skip()
    if eng.name() != "postgis":
        eng.assert_query_result(
            f"SELECT ST_GeogPoint({val_or_null(x)}, {val_or_null(y)})", expected
        )
    else:
        eng.assert_query_result(
            f"SELECT ST_Point({val_or_null(x)}, {val_or_null(y)}) as geography",
            expected,
        )
