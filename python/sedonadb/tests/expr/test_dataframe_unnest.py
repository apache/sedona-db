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

import pandas as pd
import pandas.testing as pdt
import pytest

from sedonadb._lib import SedonaError
from sedonadb.dataframe import DataFrame


def test_unnest_expands_list_to_rows(con):
    # Each list element becomes its own row; other columns are repeated.
    df = con.sql("SELECT 'a' AS label, [10, 20, 30] AS vals")
    out = df.unnest("vals").sort("vals").to_pandas()
    pdt.assert_frame_equal(
        out,
        pd.DataFrame({"label": ["a", "a", "a"], "vals": [10, 20, 30]}),
    )


def test_unnest_multiple_columns_parallel(con):
    # Multiple columns are expanded position-by-position, not cross-product.
    df = con.sql("SELECT [1, 2] AS a, [10, 20] AS b")
    out = df.unnest("a", "b").sort("a").to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"a": [1, 2], "b": [10, 20]}))


def test_unnest_st_dump_multigeometry(con):
    # The spatial explode pattern: ST_Dump yields a list of parts, unnest
    # turns each part into its own row.
    df = con.sql(
        "SELECT ST_Dump(ST_GeomFromText('MULTIPOINT(0 0, 1 1, 2 2)')) AS parts"
    )
    assert df.unnest("parts").count() == 3


def test_unnest_returns_lazy_dataframe(con):
    df = con.sql("SELECT [1, 2] AS vals")
    assert isinstance(df.unnest("vals"), DataFrame)


def test_unnest_no_args_raises(con):
    df = con.sql("SELECT [1, 2] AS vals")
    with pytest.raises(ValueError, match="at least one column"):
        df.unnest()


def test_unnest_non_str_raises(con):
    df = con.sql("SELECT [1, 2] AS vals")
    with pytest.raises(TypeError, match="expects str"):
        df.unnest(123)


def test_unnest_unknown_column_raises(con):
    df = con.sql("SELECT [1, 2] AS vals")
    with pytest.raises(SedonaError, match="No field named"):
        df.unnest("nope")


def test_unnest_geometry_column_raises(con):
    # A raw geometry (Binary) column can't be unnested directly; ST_Dump first.
    df = con.sql("SELECT ST_Point(0.0, 0.0) AS geom")
    with pytest.raises(SedonaError, match="unnest"):
        df.unnest("geom")
