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
from sedonadb.testing import SedonaDB, geog_or_null

if "s2geography" not in sedonadb.__features__:
    pytest.skip("Python package built without s2geography", allow_module_level=True)


# S2_CellIdFromPoint tests - returns the S2 cell ID containing a point
@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geog", "is_null_expected"),
    [
        # Nulls
        pytest.param(None, True, id="null_point"),
        # Empties
        pytest.param("POINT EMPTY", True, id="empty_point"),
        # Valid points
        pytest.param("POINT (0 0)", False, id="point_origin"),
        pytest.param("POINT (0 1)", False, id="point_1_degree"),
        pytest.param("POINT (-122.4194 37.7749)", False, id="point_sf"),
        pytest.param("POINT (180 0)", False, id="point_antimeridian"),
        pytest.param("POINT (0 90)", False, id="point_north_pole"),
        pytest.param("POINT (0 -90)", False, id="point_south_pole"),
    ],
)
def test_s2_cellidfrompoint(eng, geog, is_null_expected):
    eng = eng.create_or_skip()
    result = eng.execute_query(f"SELECT S2_CellIdFromPoint({geog_or_null(geog)})")
    if is_null_expected:
        assert result is None
    else:
        # Cell IDs should be non-zero 64-bit integers
        assert isinstance(result, int)
        assert result != 0


# Test that different points get different cell IDs
@pytest.mark.parametrize("eng", [SedonaDB])
def test_s2_cellidfrompoint_different_points(eng):
    eng = eng.create_or_skip()
    id1 = eng.execute_query("SELECT S2_CellIdFromPoint(ST_GeogFromText('POINT (0 0)'))")
    id2 = eng.execute_query("SELECT S2_CellIdFromPoint(ST_GeogFromText('POINT (0 1)'))")
    id3 = eng.execute_query("SELECT S2_CellIdFromPoint(ST_GeogFromText('POINT (1 0)'))")

    assert id1 != id2
    assert id1 != id3
    assert id2 != id3


# Test that nearby points get the same cell ID (at a coarse level)
@pytest.mark.parametrize("eng", [SedonaDB])
def test_s2_cellidfrompoint_consistency(eng):
    eng = eng.create_or_skip()
    # Same point should always return the same cell ID
    id1 = eng.execute_query("SELECT S2_CellIdFromPoint(ST_GeogFromText('POINT (0 0)'))")
    id2 = eng.execute_query("SELECT S2_CellIdFromPoint(ST_GeogFromText('POINT (0 0)'))")
    assert id1 == id2


# S2_CoveringCellIds tests - returns a list of S2 cell IDs covering a geometry
@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geog", "expected_count_min", "expected_count_max"),
    [
        # Empties return empty list
        pytest.param("POINT EMPTY", 0, 0, id="empty_point"),
        # Single point returns one cell
        pytest.param("POINT (0 0)", 1, 1, id="point_origin"),
        pytest.param("POINT (0 1)", 1, 1, id="point_1_degree"),
        # Linestrings may require multiple cells depending on length
        pytest.param("LINESTRING (0 0, 0.001 0.001)", 1, 4, id="linestring_short"),
        pytest.param(
            "LINESTRING (0 0, 100 50)", 2, 8, id="linestring_long"
        ),  # Spans multiple cells
        # Polygons may require multiple cells depending on size
        pytest.param(
            "POLYGON ((0 0, 0.001 0, 0.001 0.001, 0 0.001, 0 0))",
            1,
            4,
            id="polygon_tiny",
        ),
        pytest.param(
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
            1,
            8,
            id="polygon_medium",
        ),
    ],
)
def test_s2_coveringcellids(eng, geog, expected_count_min, expected_count_max):
    eng = eng.create_or_skip()
    result = eng.execute_query(f"SELECT S2_CoveringCellIds({geog_or_null(geog)})")
    if expected_count_min == 0 and expected_count_max == 0:
        # Empty geometry should return empty list or null
        assert result is None or (isinstance(result, list) and len(result) == 0)
    else:
        assert isinstance(result, list)
        assert expected_count_min <= len(result) <= expected_count_max, (
            f"Expected {expected_count_min}-{expected_count_max} cells, got {len(result)}"
        )
        # All cell IDs should be non-zero integers
        for cell_id in result:
            assert isinstance(cell_id, int)
            assert cell_id != 0


# Null input returns null
@pytest.mark.parametrize("eng", [SedonaDB])
def test_s2_coveringcellids_null(eng):
    eng = eng.create_or_skip()
    result = eng.execute_query("SELECT S2_CoveringCellIds(NULL)")
    assert result is None


# Test that coverings are deterministic
@pytest.mark.parametrize("eng", [SedonaDB])
def test_s2_coveringcellids_deterministic(eng):
    eng = eng.create_or_skip()
    cells1 = eng.execute_query(
        "SELECT S2_CoveringCellIds(ST_GeogFromText('POINT (0 0)'))"
    )
    cells2 = eng.execute_query(
        "SELECT S2_CoveringCellIds(ST_GeogFromText('POINT (0 0)'))"
    )
    assert cells1 == cells2


# Test that point covering cell ID matches S2_CellIdFromPoint
@pytest.mark.parametrize("eng", [SedonaDB])
def test_s2_coveringcellids_matches_cellidfrompoint(eng):
    eng = eng.create_or_skip()
    cell_id = eng.execute_query(
        "SELECT S2_CellIdFromPoint(ST_GeogFromText('POINT (0 0)'))"
    )
    covering_ids = eng.execute_query(
        "SELECT S2_CoveringCellIds(ST_GeogFromText('POINT (0 0)'))"
    )

    # The covering for a point should contain exactly one cell,
    # and it should be the same as the cell ID from S2_CellIdFromPoint
    assert isinstance(covering_ids, list)
    assert len(covering_ids) == 1
    assert covering_ids[0] == cell_id
