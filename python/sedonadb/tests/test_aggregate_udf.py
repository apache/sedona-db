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
import pyarrow as pa
import pytest
from sedonadb import udf
from sedonadb.expr import col


def _mean_class():
    @udf.arrow_aggregate_udf(
        return_type=pa.float64(),
        input_types=[udf.NUMERIC],
        state_types=[pa.float64(), pa.int64()],
    )
    class my_mean:
        def __init__(self):
            self.total = 0.0
            self.count = 0

        def update(self, batch):
            for v in batch:
                if v.is_valid:
                    self.total += float(v.as_py())
                    self.count += 1

        def state(self):
            return (self.total, self.count)

        def merge(self, totals, counts):
            for i in range(len(totals)):
                self.total += totals[i].as_py()
                self.count += counts[i].as_py()

        def evaluate(self):
            return None if self.count == 0 else self.total / self.count

    return my_mean


def test_aggregate_udf_global_via_sql(con):
    con.register(_mean_class())
    con.create_data_frame(pd.DataFrame({"v": [1.0, 3.0, 5.0, 7.0]})).to_view(
        "t_global", overwrite=True
    )
    out = con.sql("SELECT my_mean(v) AS m FROM t_global").to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"m": [4.0]}))


def test_aggregate_udf_grouped(con):
    # Grouped aggregation exercises both update_batch and the merge path
    # across groups under the slow Accumulator-per-group fallback.
    con.register(_mean_class())
    df = con.create_data_frame(
        pd.DataFrame({"k": ["a", "a", "b", "b", "b"], "v": [1.0, 3.0, 2.0, 4.0, 6.0]})
    )
    out = df.group_by("k").agg(m=con.funcs.my_mean(col("v"))).sort("k").to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"k": ["a", "b"], "m": [2.0, 4.0]}))


def test_aggregate_udf_input_nulls_ignored(con):
    con.register(_mean_class())
    df = con.create_data_frame(pd.DataFrame({"v": [1.0, None, 3.0, None, 5.0]}))
    out = df.agg(m=con.funcs.my_mean(col("v"))).to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"m": [3.0]}))


def test_aggregate_udf_multi_batch_forces_merge(con):
    # The grouped test above relies on DataFusion choosing a two-phase
    # aggregate; that's incidental to the current plan shape. This test
    # forces the partial-state path with an input spanning many record
    # batches and asserts merge() actually ran, so state()/merge_batch()
    # coverage can't silently disappear if the planner changes.
    import pyarrow.compute as pc

    merge_calls = {"n": 0}

    @udf.arrow_aggregate_udf(
        return_type=pa.float64(),
        input_types=[udf.NUMERIC],
        state_types=[pa.float64(), pa.int64()],
    )
    class mean_with_probe:
        def __init__(self):
            self.total = 0.0
            self.count = 0

        def update(self, batch):
            total = pc.sum(batch).as_py()
            if total is not None:
                self.total += float(total)
            self.count += len(batch) - batch.null_count

        def state(self):
            return (self.total, self.count)

        def merge(self, totals, counts):
            merge_calls["n"] += 1
            for i in range(len(totals)):
                if totals[i].is_valid:
                    self.total += totals[i].as_py()
                    self.count += counts[i].as_py()

        def evaluate(self):
            return None if self.count == 0 else self.total / self.count

    con.register(mean_with_probe)
    # 50k rows >> the 8192-row default batch size; keys interleave so both
    # groups span every batch. k='a' holds the even values, k='b' the odds.
    n = 50_000
    df = con.create_data_frame(
        pa.table(
            {
                "k": ["a", "b"] * (n // 2),
                "v": [float(i) for i in range(n)],
            }
        )
    )
    out = (
        df.group_by("k")
        .agg(m=con.funcs.mean_with_probe(col("v")))
        .sort("k")
        .to_pandas()
    )
    pdt.assert_frame_equal(
        out,
        pd.DataFrame({"k": ["a", "b"], "m": [float(n - 2) / 2, float(n) / 2]}),
    )
    assert merge_calls["n"] > 0


def test_aggregate_udf_empty_input_returns_null(con):
    # No rows → count==0 → evaluate returns None → SQL NULL surfaces as
    # NaN in float64 pandas.
    con.register(_mean_class())
    con.create_data_frame(pd.DataFrame({"v": pd.Series([], dtype="float64")})).to_view(
        "t_empty", overwrite=True
    )
    out = con.sql("SELECT my_mean(v) AS m FROM t_empty").to_pandas()
    assert len(out) == 1
    assert pd.isna(out["m"].iloc[0])


def test_aggregate_udf_evaluate_returning_wrong_type_raises(con):
    # evaluate() must return a value compatible with return_type; returning
    # a string when float64 is declared fails when the wrapper builds the
    # 1-element pyarrow array for the result.
    @udf.arrow_aggregate_udf(
        return_type=pa.float64(),
        input_types=[udf.NUMERIC],
        state_types=[pa.int64()],
    )
    class wrong_return_type:
        def __init__(self):
            self.n = 0

        def update(self, batch):
            self.n += len(batch)

        def state(self):
            return (self.n,)

        def merge(self, counts):
            for i in range(len(counts)):
                self.n += counts[i].as_py()

        def evaluate(self):
            return "not a number"

    con.register(wrong_return_type)
    df = con.create_data_frame(pd.DataFrame({"v": [1.0, 2.0, 3.0]}))
    with pytest.raises(Exception, match="Could not convert"):
        df.agg(n=con.funcs.wrong_return_type(col("v"))).to_pandas()


def test_aggregate_udf_state_arity_mismatch_raises(con):
    # state() must return exactly len(state_types) elements; the wrapper
    # raises before handing the values to the engine.
    @udf.arrow_aggregate_udf(
        return_type=pa.int64(),
        input_types=[udf.NUMERIC],
        state_types=[pa.int64(), pa.int64()],
    )
    class wrong_state_arity:
        def __init__(self):
            self.n = 0

        def update(self, batch):
            self.n += len(batch)

        def state(self):
            return (self.n,)  # one element, but two declared

        def merge(self, a, b):
            for i in range(len(a)):
                self.n += a[i].as_py()

        def evaluate(self):
            return self.n

    con.register(wrong_state_arity)
    # A grouped multi-batch input forces a partial state() call.
    n = 20_000
    df = con.create_data_frame(
        pa.table({"k": ["a", "b"] * (n // 2), "v": [float(i) for i in range(n)]})
    )
    with pytest.raises(Exception, match=r"state\(\) returned 1 elements"):
        df.group_by("k").agg(m=con.funcs.wrong_state_arity(col("v"))).to_pandas()


@pytest.mark.parametrize("failing_method", ["update", "state", "merge", "evaluate"])
def test_aggregate_udf_exception_in_method_propagates(con, failing_method):
    # A Python exception raised in any accumulator method surfaces as an
    # error rather than crashing. The grouped multi-batch input ensures
    # state() and merge() are actually invoked (two-phase aggregation).
    @udf.arrow_aggregate_udf(
        return_type=pa.float64(),
        input_types=[udf.NUMERIC],
        state_types=[pa.float64(), pa.int64()],
    )
    class boom:
        def __init__(self):
            self.total = 0.0
            self.count = 0

        def update(self, batch):
            if failing_method == "update":
                raise RuntimeError("boom in update")
            self.count += len(batch)

        def state(self):
            if failing_method == "state":
                raise RuntimeError("boom in state")
            return (self.total, self.count)

        def merge(self, totals, counts):
            if failing_method == "merge":
                raise RuntimeError("boom in merge")
            for i in range(len(totals)):
                self.count += counts[i].as_py()

        def evaluate(self):
            if failing_method == "evaluate":
                raise RuntimeError("boom in evaluate")
            return self.total

    con.register(boom)
    n = 20_000
    df = con.create_data_frame(
        pa.table({"k": ["a", "b"] * (n // 2), "v": [float(i) for i in range(n)]})
    )
    with pytest.raises(Exception, match=f"boom in {failing_method}"):
        df.group_by("k").agg(m=con.funcs.boom(col("v"))).to_pandas()


def test_aggregate_udf_camel_case_name(con):
    # A CamelCase class name is exposed as snake_case for SQL / funcs.
    @udf.arrow_aggregate_udf(
        return_type=pa.int64(),
        input_types=[udf.NUMERIC],
        state_types=[pa.int64()],
    )
    class MyRowCount:
        def __init__(self):
            self.n = 0

        def update(self, batch):
            self.n += len(batch)

        def state(self):
            return (self.n,)

        def merge(self, counts):
            for i in range(len(counts)):
                self.n += counts[i].as_py()

        def evaluate(self):
            return self.n

    assert MyRowCount._name == "my_row_count"
    con.register(MyRowCount)
    df = con.create_data_frame(pd.DataFrame({"v": [1.0, 2.0, 3.0]}))
    out = df.agg(n=con.funcs.my_row_count(col("v"))).to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"n": [3]}))


def test_aggregate_udf_shapely_geometry(con):
    # Geometry in + geometry out: union all input points into one geometry.
    import shapely
    import geoarrow.pyarrow as ga

    @udf.arrow_aggregate_udf(
        return_type=ga.wkb(),
        input_types=[udf.GEOMETRY],
        state_types=[pa.binary()],
    )
    class geom_union:
        def __init__(self):
            self.acc = None

        def _absorb(self, wkbs):
            for wkb in wkbs:
                if wkb is None:
                    continue
                g = shapely.from_wkb(wkb)
                self.acc = g if self.acc is None else shapely.union(self.acc, g)

        def update(self, batch):
            self._absorb(batch.to_pylist())

        def state(self):
            return (None if self.acc is None else shapely.to_wkb(self.acc),)

        def merge(self, states):
            self._absorb(states.to_pylist())

        def evaluate(self):
            return None if self.acc is None else shapely.to_wkb(self.acc)

    con.register(geom_union)
    con.sql(
        "SELECT * FROM (VALUES (ST_Point(0.0, 0.0)), (ST_Point(1.0, 1.0))) AS t(g)"
    ).to_view("t_geom", overwrite=True)
    out = con.sql("SELECT geom_union(g) AS u FROM t_geom").to_pandas()
    expected = shapely.union(shapely.Point(0, 0), shapely.Point(1, 1))
    assert out["u"].iloc[0].equals(expected)
