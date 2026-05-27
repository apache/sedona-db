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
import sedonadb
from sedonadb.testing import SedonaDB, BigQuery, geog_or_null

if "s2geography" not in sedonadb.__features__:
    pytest.skip("Python package built without s2geography", allow_module_level=True)


# S2_CellIdFromPoint tests - returns the S2 cell ID containing a point
@pytest.mark.parametrize("eng", [SedonaDB, BigQuery])
@pytest.mark.parametrize(
    ("geog", "expected"),
    [
        # Valid points
        pytest.param("POINT (0 0)", 1152921504606846977, id="point_origin"),
        pytest.param("POINT (0 1)", 1153451514845492609, id="point_1_degree"),
    ],
)
def test_s2_cellidfrompoint(eng, geog, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT S2_CellIdFromPoint({geog_or_null(geog)})",
        expected,
    )


# BigQuery doesn't handle empty input
@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geog", "expected"),
    [
        # Empties
        pytest.param("POINT EMPTY", None, id="empty_point"),
    ],
)
def test_s2_cellidfrompoint_empties(eng, geog, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT S2_CellIdFromPoint({geog_or_null(geog)})",
        expected,
    )


# S2_CoveringCellIds tests - returns a list of S2 cell IDs covering a geometry
# These are not stable across S2 versions/platforms so we just test the size here.
@pytest.mark.parametrize("eng", [SedonaDB, BigQuery])
@pytest.mark.parametrize(
    ("geog", "expected"),
    [
        # Empties return empty list
        pytest.param("POINT EMPTY", [], id="empty_point"),
        # Single point returns one cell
        pytest.param("POINT (0 0)", [1152921504606846977], id="point_origin"),
        # Linestring
        pytest.param(
            "LINESTRING (0 0, 1 1)",
            [
                384307168202282325,
                1152921504606846975,
                1152939096792891392,
                1152991873349976064,
                1153009465537069056,
                1153273348327735296,
                1153390629569429504,
                1921535841011411627,
            ],
            id="linestring",
        ),
    ],
)
def test_s2_coveringcellids(eng, geog, expected):
    eng = eng.create_or_skip()
    result = eng.execute_and_collect(
        f"SELECT S2_CoveringCellIds({geog_or_null(geog)})",
    )
    df = eng.result_to_pandas(result)
    assert len(df.iloc[0, 0]) == len(expected)
