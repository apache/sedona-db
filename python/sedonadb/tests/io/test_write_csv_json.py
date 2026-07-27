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

import json
import tempfile
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

from sedonadb._lib import SedonaError


def test_to_csv_round_trip(con):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "out.csv"
        con.sql("SELECT 1 AS a, 'x' AS b UNION ALL SELECT 2, 'y'").to_csv(p)
        # A single file is written when the path ends with ".csv".
        assert p.is_file()
        out = con.read.csv(p).sort("a").to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}))


def test_to_csv_no_header(con):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "out.csv"
        con.sql("SELECT 1 AS a, 2 AS b").to_csv(p, has_header=False)
        assert p.read_text() == "1,2\n"


def test_to_csv_custom_delimiter(con):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "out.csv"
        con.sql("SELECT 1 AS a, 'x' AS b").to_csv(p, delimiter=";")
        assert p.read_text() == "a;b\n1;x\n"
        # And it round-trips when read back with the same delimiter.
        out = con.read.csv(p, delimiter=";").to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"a": [1], "b": ["x"]}))


def test_to_csv_bad_delimiter_raises(con):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "out.csv"
        with pytest.raises(SedonaError, match="single byte"):
            con.sql("SELECT 1 AS a").to_csv(p, delimiter=";;")


def test_to_csv_directory_output(con):
    # A path without a ".csv" suffix writes a directory of part file(s).
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "parts"
        con.sql("SELECT 1 AS a").to_csv(d)
        assert d.is_dir()
        assert list(d.glob("*.csv"))


def test_to_csv_geometry_raises(con):
    # CSV has no geometry representation, so a geometry column is a hard error
    # (rather than a silent opaque encoding), with a message naming the column.
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "geo.csv"
        df = con.sql("SELECT ST_Point(1.0, 2.0) AS geometry, 'a' AS name")
        with pytest.raises(SedonaError, match='geometry column.*"geometry"'):
            df.to_csv(p)


def test_to_json_geometry_raises(con):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "geo.json"
        df = con.sql("SELECT ST_Point(1.0, 2.0) AS geometry, 'a' AS name")
        with pytest.raises(SedonaError, match='geometry column.*"geometry"'):
            df.to_json(p)


def test_to_csv_nested_geometry_raises(con):
    # Geometry nested inside a list/struct (e.g. ST_Dump output) is caught too.
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "nested.csv"
        df = con.sql(
            "SELECT ST_Dump(ST_GeomFromText('MULTIPOINT (0 0, 1 1)')) AS parts"
        )
        with pytest.raises(SedonaError, match="geometry column"):
            df.to_csv(p)


def test_to_csv_geometry_as_text_ok(con):
    # The documented workaround: project geometry to text first.
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "wkt.csv"
        con.sql("SELECT ST_AsText(ST_Point(1.0, 2.0)) AS geometry").to_csv(p)
        assert p.read_text() == "geometry\nPOINT(1 2)\n"


def test_to_json_round_trip(con):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "out.json"
        con.sql("SELECT 1 AS a, 'x' AS b UNION ALL SELECT 2, 'y'").to_json(p)
        assert p.is_file()
        out = con.read.json(p).sort("a").to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}))


def test_to_json_ndjson_format(con):
    # to_json emits one JSON object per row (NDJSON): one object per line.
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "out.json"
        con.sql("SELECT 1 AS a, 'x' AS b UNION ALL SELECT 2, 'y'").to_json(p)
        lines = p.read_text().splitlines()
    assert len(lines) == 2
    assert sorted((json.loads(line) for line in lines), key=lambda r: r["a"]) == [
        {"a": 1, "b": "x"},
        {"a": 2, "b": "y"},
    ]
