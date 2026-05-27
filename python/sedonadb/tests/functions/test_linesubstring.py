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
from sedonadb.testing import PostGIS, SedonaDB, geom_or_null, val_or_null


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "start", "end", "expected"),
    [
        ("LINESTRING EMPTY", 0.0, 1.0, None),
        ("LINESTRING EMPTY", None, 1.0, None),
        ("LINESTRING EMPTY", 0.0, None, None),
        (None, 0.0, 1.0, None),
        (
            "LINESTRING(0 0, 10 10)",
            None,
            1.0,
            None,
        ),  # Start fraction is NULL -> Output is NULL
        (
            "LINESTRING(0 0, 10 10)",
            0.0,
            None,
            None,
        ),  # End fraction is NULL -> Output is NULL
        (None, None, None, None),
        # Zero-length linestring
        ("LINESTRING (0 0, 0 0)", 0.0, 1.0, "POINT (0 0)"),
        # Single segment
        ("LINESTRING (0 0, 10 0)", 0.0, 1.0, "LINESTRING (0 0, 10 0)"),
        ("LINESTRING (0 0, 10 0)", 0.2, 0.8, "LINESTRING (2 0, 8 0)"),
        ("LINESTRING (0 0, 10 10)", 0.5, 0.5, "POINT (5 5)"),
        # Multi segment with degenerate edges
        ("LINESTRING (0 0, 0 0, 10 0)", 0.0, 1.0, "LINESTRING (0 0, 10 0)"),
        ("LINESTRING (0 0, 10 0, 10 0)", 0.0, 1.0, "LINESTRING (0 0, 10 0)"),
        # Multi segment (3 equal segments, each length 10, total 30)
        # (1) Entire linestring
        (
            "LINESTRING (0 0, 10 0, 10 10, 0 10)",
            0.0,
            1.0,
            "LINESTRING (0 0, 10 0, 10 10, 0 10)",
        ),
        # (2) Exactly the first segment (0 to 1/3)
        (
            "LINESTRING (0 0, 10 0, 10 10, 0 10)",
            0.0,
            1.0 / 3.0,
            "LINESTRING (0 0, 10 0)",
        ),
        # (3) Exactly the middle segment (1/3 to 2/3)
        (
            "LINESTRING (0 0, 10 0, 10 10, 0 10)",
            1.0 / 3.0,
            2.0 / 3.0,
            "LINESTRING (10 0, 10 10)",
        ),
        # (4) Exactly the last segment (2/3 to 1)
        (
            "LINESTRING (0 0, 10 0, 10 10, 0 10)",
            2.0 / 3.0,
            1.0,
            "LINESTRING (10 10, 0 10)",
        ),
        # (5) Part of the first segment (0 to 1/6 = halfway through first segment)
        (
            "LINESTRING (0 0, 10 0, 10 10, 0 10)",
            0.0,
            1.0 / 6.0,
            "LINESTRING (0 0, 5 0)",
        ),
        # (6) Part of the middle segment (0.4 to 0.6, both within middle segment)
        ("LINESTRING (0 0, 10 0, 10 10, 0 10)", 0.4, 0.6, "LINESTRING (10 2, 10 8)"),
        # (7) Part of the last segment (0.75 to 0.9, both within last segment)
        ("LINESTRING (0 0, 10 0, 10 10, 0 10)", 0.75, 0.9, "LINESTRING (7.5 10, 3 10)"),
        # (8) Part of first, entire middle, part of last (1/6 to 5/6)
        (
            "LINESTRING (0 0, 10 0, 10 10, 0 10)",
            1.0 / 6.0,
            5.0 / 6.0,
            "LINESTRING (5 0, 10 0, 10 10, 5 10)",
        ),
        # Z Case
        (
            "LINESTRING Z (0 0 0, 10 10 10)",
            0.5,
            0.8,
            "LINESTRING Z (5 5 5, 8 8 8)",
        ),
        # M Case
        (
            "LINESTRING M (0 10 20, 10 20 30)",
            0.0,
            0.5,
            "LINESTRING M (0 10 20, 5 15 25)",
        ),
        # ZM Case
        (
            "LINESTRING ZM (0 10 20 30, 10 20 30 40)",
            0.5,
            0.8,
            "LINESTRING ZM (5 15 25 35, 8 18 28 38)",
        ),
    ],
)
def test_st_line_substring(eng, geom, start, end, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_LineSubstring({geom_or_null(geom)}, {val_or_null(start)}, {val_or_null(end)})",
        expected,
    )
