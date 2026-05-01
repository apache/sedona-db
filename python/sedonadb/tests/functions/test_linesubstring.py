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
    ("geom", "start", "end", "expected"),
    [
        (None, 0.0, 1.0, None),
        ("LINESTRING EMPTY", 0.0, 1.0, "LINESTRING EMPTY"),

        ("LINESTRING (0 0, 10 0)", 0.2, 0.8, "LINESTRING (2 0, 8 0)"),

        ("LINESTRING (0 0, 10 10)", 0.3, 0.6, "LINESTRING (3 3, 6 6)"),

        ("LINESTRING (0 0, 10 10)", 0.5, 0.5, "POINT (5 5)"),

        ("LINESTRING (0 0, 10 0, 10 10)", 0.25, 0.75, "LINESTRING (5 0, 10 0, 10 5)"),

        (
            "LINESTRING Z (0 0 0, 10 10 10)",
            0.5,
            0.8,
            "LINESTRING Z (5 5 5, 8 8 8)",
        ),
    ],
)
def test_st_line_substring(eng, geom, start, end, expected):
    eng = eng.create_or_skip()

    if expected is not None:
        expected = expected.replace(", ", ",")
        expected = expected.replace(" (", "(")

        if isinstance(eng, PostGIS):
            
            expected = expected.replace("Z(", "Z (")
            expected = expected.replace("M(", "M (")
            expected = expected.replace("ZM(", "ZM (")

    eng.assert_query_result(
        f"SELECT ST_AsText(ST_LineSubstring({geom_or_null(geom)}, {start}, {end}))",
        expected,
    )