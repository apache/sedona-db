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

# These tests assert structural properties of constructed expressions —
# primarily `variant_name()`, child variants reachable via `debug_string()`,
# and the presence of user-supplied identifiers (column names, literal
# values) inside the rendered representation. Where possible we avoid
# pinning exact substrings of DataFusion's `Display` formatting so the
# suite is not coupled to a specific DataFusion version.

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
    # Literal value should be represented somewhere in the repr; don't pin
    # DataFusion's exact format string.
    assert "5" in repr(e)


def test_lit_passthrough_for_existing_expr():
    e = col("x")
    assert lit(e) is e


def test_lit_from_pyarrow_scalar():
    arr = pa.array([42])
    e = lit(arr[0])
    assert e._impl.variant_name() == "Literal"
    assert "42" in repr(e)


def test_lit_from_string():
    e = lit("hello")
    assert e._impl.variant_name() == "Literal"
    assert "hello" in repr(e)


def test_lit_from_none():
    e = lit(None)
    assert e._impl.variant_name() == "Literal"


def test_alias():
    e = col("x").alias("y")
    assert e._impl.variant_name() == "Alias"
    # The new name must be reachable from the user's perspective; the
    # underlying column name should also still be visible.
    rep = repr(e)
    assert "y" in rep
    assert "x" in rep


def test_alias_chain():
    e = col("x").alias("a").alias("b")
    assert e._impl.variant_name() == "Alias"
    # Either nested or last-wins; in both cases the latest name must show.
    assert "b" in repr(e)


def test_cast_to_arrow_type():
    e = col("x").cast(pa.int32())
    assert e._impl.variant_name() == "Cast"
    assert "x" in repr(e)


def test_cast_to_string():
    e = col("x").cast(pa.string())
    assert e._impl.variant_name() == "Cast"


def test_cast_rejects_extension_type():
    import geoarrow.pyarrow as ga

    with pytest.raises(Exception, match="extension type"):
        col("x").cast(ga.wkb())


def test_is_null():
    e = col("x").is_null()
    assert e._impl.variant_name() == "IsNull"


def test_is_not_null():
    e = col("x").is_not_null()
    assert e._impl.variant_name() == "IsNotNull"


def test_isin_python_scalars():
    e = col("x").isin([1, 2, 3])
    assert e._impl.variant_name() == "InList"
    rep = repr(e)
    assert "x" in rep
    # Each value should still appear somewhere in the rendered form,
    # without pinning the exact wrapping that DataFusion uses.
    assert "1" in rep and "3" in rep


def test_isin_with_expr_values():
    e = col("x").isin([lit(1), 2, lit(3)])
    assert e._impl.variant_name() == "InList"
    rep = repr(e)
    assert "1" in rep and "2" in rep and "3" in rep


def test_negate():
    e = col("x").negate()
    assert e._impl.variant_name() == "Negative"


def test_chain_alias_after_predicate():
    e = col("x").is_null().alias("missing")
    assert e._impl.variant_name() == "Alias"
    assert "missing" in repr(e)


def test_expr_is_not_bound_to_dataframe():
    # Constructing an Expr referring to a non-existent column does not error.
    # Errors surface only at DataFrame consumption.
    e = col("nonexistent_column_xyz")
    assert "nonexistent_column_xyz" in repr(e)
