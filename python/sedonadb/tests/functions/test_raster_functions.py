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

from sedonadb.testing import SedonaDB


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("RS_NumBands(RS_Example())", 3),
        ("RS_Width(RS_Example())", 64),
        ("RS_Height(RS_Example())", 32),
        ("RS_BandPixelType(RS_Example(), 1)", "UNSIGNED_8BITS"),
        ("RS_BandNoDataValue(RS_Example(), 1)", 127.0),
        ("RS_ScaleX(RS_Example())", 2.0),
        ("RS_ScaleY(RS_Example())", 2.0),
        ("RS_SkewX(RS_Example())", 1.0),
        ("RS_SkewY(RS_Example())", 1.0),
        ("RS_UpperLeftX(RS_Example())", 43.08),
        ("RS_UpperLeftY(RS_Example())", 79.07),
    ],
)
def test_rs_function(expr, expected):
    eng = SedonaDB()
    eng.assert_query_result(f"SELECT {expr}", expected)


def test_rs_ensureloaded(con, sedona_testing):
    path = sedona_testing / "data/raster/sentinel2.tif"
    t = con.sql("SELECT RS_FromPath($1) AS raster", params=(str(path),))
    tab = t.select(raster=t.raster.funcs.rs_ensureloaded()).to_arrow_table()
    r = tab["raster"][0].as_py()
    assert r.height == 512
    assert r.width == 512

    assert len(r.bands) == 1
    b = r.bands[0]
    assert b.shape == (512, 512)
    arr = b.to_numpy()
    assert arr.shape == (512, 512)
    assert arr.dtype == "uint16"
    assert arr[0, 0] == 2324
