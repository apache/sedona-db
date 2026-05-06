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

# These tests pin the exact rendered form of each expression. Locking the
# Display output is intentional: it doubles as a regression test on how user
# expressions appear in error messages and `repr()` output, and any DataFusion
# upgrade that changes the rendering should be reviewed deliberately rather
# than auto-passing. If you find yourself loosening these assertions, add a
# replacement check on `_impl.variant_name()` so the structural meaning is
# still locked.

import pyarrow as pa
import pytest

from sedonadb.expr import Expr, col


def test_col_returns_expr():
    e = col("x")
    assert isinstance(e, Expr)
    assert e._impl.variant_name() == "Column"
    assert repr(e) == "Expr(x)"


def test_col_with_qualifier():
    e = col("x", "t")
    assert isinstance(e, Expr)
    assert e._impl.variant_name() == "Column"
    assert repr(e) == "Expr(t.x)"


def test_alias():
    e = col("x").alias("y")
    assert e._impl.variant_name() == "Alias"
    assert "x AS y" in repr(e)


def test_alias_chain():
    e = col("x").alias("a").alias("b")
    # Either nested or last-wins; in both cases the latest name must show.
    assert "b" in repr(e)


def test_cast_to_arrow_type():
    e = col("x").cast(pa.int32())
    assert e._impl.variant_name() == "Cast"
    assert "CAST(x AS Int32)" in repr(e)


def test_cast_to_string():
    e = col("x").cast(pa.string())
    assert "Utf8" in repr(e)


def test_cast_rejects_extension_type():
    import geoarrow.pyarrow as ga

    with pytest.raises(Exception, match="extension type"):
        col("x").cast(ga.wkb())


def test_is_null():
    e = col("x").is_null()
    assert e._impl.variant_name() == "IsNull"
    assert "x IS NULL" in repr(e)


def test_is_not_null():
    e = col("x").is_not_null()
    assert e._impl.variant_name() == "IsNotNull"
    assert "x IS NOT NULL" in repr(e)


def test_isin_python_scalars():
    # Plain Python scalars are coerced to literal expressions automatically.
    e = col("x").isin([1, 2, 3])
    assert e._impl.variant_name() == "InList"
    rep = repr(e)
    assert "IN" in rep
    assert "Int64(1)" in rep
    assert "Int64(3)" in rep


def test_isin_with_expr_values():
    # Mixed Expr + scalar input — Exprs pass through, scalars are coerced.
    e = col("x").isin([col("a"), 2])
    assert e._impl.variant_name() == "InList"
    rep = repr(e)
    assert "a" in rep
    assert "Int64(2)" in rep


def test_negate():
    e = col("x").negate()
    assert e._impl.variant_name() == "Negative"
    assert "(- x)" in repr(e)


def test_chain_alias_after_predicate():
    e = col("x").is_null().alias("missing")
    assert e._impl.variant_name() == "Alias"
    assert "missing" in repr(e)
    assert "IS NULL" in repr(e)


def test_expr_is_not_bound_to_dataframe():
    # Constructing an Expr referring to a non-existent column does not error.
    # Errors surface only at DataFrame consumption.
    e = col("nonexistent_column_xyz")
    assert "nonexistent_column_xyz" in repr(e)


def test_expr_init_rejects_wrong_type():
    # The Expr constructor should fail clearly when handed something that is
    # not an internal Expr handle.
    with pytest.raises(TypeError, match="InternalExpr"):
        Expr("not an internal expr")
    with pytest.raises(TypeError, match="InternalExpr"):
        Expr(42)
