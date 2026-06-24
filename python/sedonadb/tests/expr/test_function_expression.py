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

from sedonadb.expr import Expr
from sedonadb.expr.expression import ScalarUdf, AggregateUdf

import shapely
import pytest


def test_scalar_st_function_returns_expr(con):
    st_geomfromwkt = con.funcs.st_geomfromwkt
    assert isinstance(st_geomfromwkt, ScalarUdf)

    e = st_geomfromwkt("POINT (0 1)")
    assert isinstance(e, Expr)
    assert repr(e) == 'Expr(st_geomfromwkt(Utf8("POINT (0 1)")))'

    # Also check piped function from literal
    e = con.lit("POINT (0 1)").funcs.st_geomfromwkt()
    assert repr(e) == 'Expr(st_geomfromwkt(Utf8("POINT (0 1)")))'


def test_scalar_st_function_alias_returns_expr(con):
    st_geomfromtext = con.funcs.st_geomfromtext
    assert isinstance(st_geomfromtext, ScalarUdf)

    e = st_geomfromtext("POINT (0 1)")
    assert isinstance(e, Expr)
    # st_geomfromtext internally maps to st_geomfromwkt
    assert repr(e) == 'Expr(st_geomfromwkt(Utf8("POINT (0 1)")))'


def test_scalar_st_function_with_column(con):
    st_area = con.funcs.st_area
    assert isinstance(st_area, ScalarUdf)

    e = st_area(con.col("geom"))
    assert isinstance(e, Expr)
    assert repr(e) == "Expr(st_area(geom))"

    # Also check piped function from column
    e = con.col("geom").funcs.st_geomfromwkt()
    assert repr(e) == "Expr(st_geomfromwkt(geom))"


def test_scalar_st_function_with_multiple_args(con):
    st_buffer = con.funcs.st_buffer
    assert isinstance(st_buffer, ScalarUdf)

    e = st_buffer(con.col("geom"), 10.0)
    assert isinstance(e, Expr)
    assert repr(e) == "Expr(st_buffer(geom, Float64(10)))"


def test_scalar_non_st_function_returns_expr(con):
    abs_fn = con.funcs.abs
    assert isinstance(abs_fn, ScalarUdf)

    e = abs_fn(con.col("x"))
    assert isinstance(e, Expr)
    assert repr(e) == "Expr(abs(x))"


def test_scalar_non_st_function_with_literal(con):
    sqrt_fn = con.funcs.sqrt
    assert isinstance(sqrt_fn, ScalarUdf)

    e = sqrt_fn(16.0)
    assert isinstance(e, Expr)
    assert repr(e) == "Expr(sqrt(Float64(16)))"


def test_aggregate_function_returns_expr(con):
    st_collect_agg = con.funcs.st_collect_agg
    assert isinstance(st_collect_agg, AggregateUdf)

    e = st_collect_agg(con.col("geom"))
    assert isinstance(e, Expr)
    assert repr(e) == "Expr(st_collect_agg(geom))"


def test_aggregate_function_envelope_agg(con):
    st_envelope_agg = con.funcs.st_envelope_agg
    assert isinstance(st_envelope_agg, AggregateUdf)

    e = st_envelope_agg(con.col("geom"))
    assert isinstance(e, Expr)
    assert repr(e) == "Expr(st_envelope_agg(geom))"


def test_function_expression_aliased(con):
    st_area = con.funcs.st_area
    e = st_area(con.col("geom")).alias("area")
    assert repr(e) == "Expr(st_area(geom) AS area)"


def test_function_expression_composed(con):
    st_geomfromwkt = con.funcs.st_geomfromwkt
    st_area = con.funcs.st_area

    e = st_area(st_geomfromwkt("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"))
    assert isinstance(e, Expr)
    assert (
        repr(e)
        == 'Expr(st_area(st_geomfromwkt(Utf8("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"))))'
    )


def test_geo_functions_accessor(con):
    pytest.importorskip("sedonadb_expr")

    # Check function as resolved from the geo accessor
    e = con.funcs.geo.as_text(con.col("foofy"))
    assert isinstance(e, Expr)
    assert repr(e) == "Expr(st_astext(foofy))"


def test_geo_methods_accessor(con):
    pytest.importorskip("sedonadb_expr")

    # Check piped function from literal via .geo accessor
    e = con.lit(shapely.Point(0, 1)).geo.as_text()
    assert (
        repr(e)
        == """Expr(st_astext(Binary("1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,63") FieldMetadata { inner: {"ARROW:extension:metadata": "{}", "ARROW:extension:name": "geoarrow.wkb"} }))"""
    )

    # Check piped function from Expr via .geo accessor
    e = con.col("foofy").geo.as_text()
    assert repr(e) == "Expr(st_astext(foofy))"


def test_raster_functions_accessor(con):
    pytest.importorskip("sedonadb_expr")

    # Check function as resolved from the rst accessor
    e = con.funcs.rst.height(con.col("raster_col"))
    assert isinstance(e, Expr)
    assert repr(e) == "Expr(rs_height(raster_col))"


def test_raster_methods_accessor(con):
    pytest.importorskip("sedonadb_expr")

    # Check piped function from Expr via .rst accessor
    e = con.col("raster_col").rst.height()
    assert repr(e) == "Expr(rs_height(raster_col))"

    # Check raster function with additional args
    e = con.col("raster_col").rst.band_no_data_value(1)
    assert repr(e) == "Expr(rs_bandnodatavalue(raster_col, Int64(1)))"
