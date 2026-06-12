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

from sedonadb.expr.expression import AggregateUdf, ScalarUdf
from sedonadb.functions import Functions


def test_random_geometry(con):
    df = con.funcs.table.sd_random_geometry("Point", 5, seed=99873)

    # Ensure we produce the correct number of rows
    assert df.count() == 5

    # Ensure the output is reproducible
    assert df.to_arrow_table() == df.to_arrow_table()


def test_ctx_expr_propagation(con):
    e = con.col("foofy")
    funcs_expr = Functions(con, expr=e)

    scalar_func = funcs_expr["st_area"]
    assert scalar_func._ctx is con
    assert scalar_func._expr is e

    scalar_func_call = scalar_func()
    assert scalar_func_call._ctx is con
    assert repr(scalar_func_call) == "Expr(st_area(foofy))"

    aggregate_func = funcs_expr["sum"]
    assert aggregate_func._ctx is con
    assert aggregate_func._expr is e

    aggregate_func_call = aggregate_func()
    assert aggregate_func_call._ctx is con
    assert repr(aggregate_func_call) == "Expr(sum(foofy))"


def test_funcs_dir(con):
    funcs_dir = dir(con.funcs)
    assert "st_area" in funcs_dir
    assert "st_buffer" in funcs_dir
    assert "st_intersection" in funcs_dir


def test_funcs_ipython_completions(con):
    completions = con.funcs._ipython_key_completions_()
    assert "st_area" in completions
    assert "st_buffer" in completions


def test_funcs_getitem_access(con):
    st_area = con.funcs["st_area"]
    assert isinstance(st_area, ScalarUdf)

    e = st_area(con.col("geom"))
    assert repr(e) == "Expr(st_area(geom))"

    sum_fn = con.funcs["sum"]
    assert isinstance(sum_fn, AggregateUdf)

    e = sum_fn(con.col("x"))
    assert repr(e) == "Expr(sum(x))"


def test_funcs_getattr_not_found_raises_attribute_error(con):
    with pytest.raises(AttributeError, match="Can't find scalar or aggregate function"):
        con.funcs.nonexistent_function_xyz


def test_funcs_getitem_not_found_raises_key_error(con):
    with pytest.raises(KeyError, match="Can't find scalar or aggregate function"):
        con.funcs["nonexistent_function_xyz"]


def test_scalar_udf_repr(con):
    st_area = con.funcs.st_area
    assert repr(st_area) == "ScalarUdf(st_area)"


def test_aggregate_udf_repr(con):
    st_collect_agg = con.funcs.st_collect_agg
    assert repr(st_collect_agg) == "AggregateUdf(st_collect_agg)"
