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

import pyarrow as pa
import pytest

from sedonadb.expr import Expr, col, lit


def test_col_returns_expr():
    e = col("x")
    assert isinstance(e, Expr)
    assert e._impl.variant_name() == "Column"
    assert "x" in repr(e)


def test_lit_from_python_scalar():
    e = lit(5)
    assert isinstance(e, Expr)
    assert e._impl.variant_name() == "Literal"
    assert "Int64(5)" in repr(e)


def test_lit_passthrough_for_existing_expr():
    e = col("x")
    assert lit(e) is e


def test_lit_from_pyarrow_scalar():
    arr = pa.array([42])
    e = lit(arr[0])
    assert "Int64(42)" in repr(e)


def test_lit_from_string():
    assert "Utf8" in repr(lit("hello"))


def test_lit_from_none():
    e = lit(None)
    assert "Null" in repr(e) or "NULL" in repr(e)


def test_alias():
    e = col("x").alias("y")
    assert "x AS y" in repr(e)


def test_alias_chain():
    e = col("x").alias("a").alias("b")
    # Either nested or last-wins; both encode the user intent.
    assert "b" in repr(e)


def test_cast_to_arrow_type():
    e = col("x").cast(pa.int32())
    assert "CAST(x AS Int32)" in repr(e)


def test_cast_to_string():
    e = col("x").cast(pa.string())
    assert "Utf8" in repr(e)


def test_cast_rejects_extension_type():
    import geoarrow.pyarrow as ga

    with pytest.raises(Exception, match="extension type"):
        col("x").cast(ga.wkb())


def test_is_null():
    assert "x IS NULL" in repr(col("x").is_null())


def test_is_not_null():
    assert "x IS NOT NULL" in repr(col("x").is_not_null())


def test_isin_python_scalars():
    e = col("x").isin([1, 2, 3])
    rep = repr(e)
    assert "IN" in rep
    assert "Int64(1)" in rep and "Int64(3)" in rep


def test_isin_with_expr_values():
    e = col("x").isin([lit(1), 2, lit(3)])
    rep = repr(e)
    assert "Int64(1)" in rep and "Int64(2)" in rep and "Int64(3)" in rep


def test_negate():
    assert "(- x)" in repr(col("x").negate())


def test_chain_alias_after_predicate():
    e = col("x").is_null().alias("missing")
    assert "missing" in repr(e)
    assert "IS NULL" in repr(e)


def test_expr_is_not_bound_to_dataframe():
    # Constructing an Expr referring to a non-existent column does not error.
    # Errors surface only at DataFrame consumption.
    e = col("nonexistent_column_xyz")
    assert "nonexistent_column_xyz" in repr(e)
