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

import tempfile
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest


def test_read_csv_basic(con):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.csv"
        p.write_text("a,b\n1,x\n2,y\n")
        out = con.read.csv(p).sort("a").to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}))


def test_read_csv_no_header(con):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.csv"
        p.write_text("1,x\n2,y\n")
        out = con.read.csv(p, has_header=False).to_pandas()
    # Column names are auto-assigned when there's no header.
    assert len(out) == 2
    assert list(out.iloc[:, 0]) == [1, 2]


def test_read_csv_custom_delimiter(con):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.csv"
        p.write_text("a;b\n1;x\n2;y\n")
        out = con.read.csv(p, delimiter=";").sort("a").to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}))


def test_read_csv_multiple_paths(con):
    with tempfile.TemporaryDirectory() as td:
        p1 = Path(td) / "a.csv"
        p1.write_text("a,b\n1,x\n")
        p2 = Path(td) / "b.csv"
        p2.write_text("a,b\n2,y\n")
        out = con.read.csv([p1, p2]).sort("a").to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}))


def test_read_csv_bad_delimiter_raises(con):
    from sedonadb._lib import SedonaError

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.csv"
        p.write_text("a,b\n1,x\n")
        with pytest.raises(SedonaError, match="single byte"):
            con.read.csv(p, delimiter=";;")


def test_read_json_ndjson(con):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.json"
        p.write_text('{"a": 1, "b": "x"}\n{"a": 2, "b": "y"}\n')
        out = con.read.json(p).sort("a").to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}))


def test_read_json_multiple_paths(con):
    with tempfile.TemporaryDirectory() as td:
        p1 = Path(td) / "a.json"
        p1.write_text('{"a": 1}\n')
        p2 = Path(td) / "b.json"
        p2.write_text('{"a": 2}\n')
        out = con.read.json([p1, p2]).sort("a").to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"a": [1, 2]}))


def test_generic_read_guesses_csv_extension(con):
    # sd.read("x.csv") should route to the CSV reader by extension.
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.csv"
        p.write_text("a,b\n1,x\n")
        out = con.read(p).to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"a": [1], "b": ["x"]}))


def test_generic_read_csv_options_thread_through(con):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.csv"
        p.write_text("a;b\n1;x\n")
        out = con.read(p, options={"delimiter": ";"}).to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"a": [1], "b": ["x"]}))


def test_generic_read_guesses_json_extension(con):
    # sd.read("x.json") should route to the JSON reader by extension.
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.json"
        p.write_text('{"a": 1, "b": "x"}\n')
        out = con.read(p).to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"a": [1], "b": ["x"]}))
