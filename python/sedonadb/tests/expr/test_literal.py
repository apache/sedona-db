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

import numpy as np
import pyarrow as pa
import shapely
import geopandas
import pandas as pd
import geoarrow.pyarrow as ga
import geopandas.testing

from sedonadb.expr.literal import lit
from sedonadb.expr.expression import Expr
import pytest


def test_basic_python_literal():
    assert pa.array(lit(1)) == pa.array([1])
    assert pa.array(lit("one")) == pa.array(["one"])
    assert pa.array(lit(None)) == pa.array([None])


def test_already_arrow_literal():
    assert pa.array(lit(pa.array([1]))) == pa.array([1])


def test_arrow_scalar_literal():
    non_geo_array = pa.array([1])
    assert pa.array(lit(non_geo_array[0])) == pa.array([1])

    # Check non-null
    geo_array = ga.with_crs(ga.as_wkb(["POINT (0 1)"]), ga.OGC_CRS84)
    lit_array = pa.array(lit(geo_array[0]))
    assert lit_array.type.crs.to_json_dict()["id"] == {
        "authority": "OGC",
        "code": "CRS84",
    }

    # Check null (type and CRS should propagate)
    geo_array = ga.with_crs(ga.as_wkb(pa.array([None], pa.binary())), ga.OGC_CRS84)
    lit_array = pa.array(lit(geo_array[0]))
    assert lit_array.type.crs.to_json_dict()["id"] == {
        "authority": "OGC",
        "code": "CRS84",
    }


# We need to test all geometry types for shapely because these have all different
# Python class names depending on the geometry type
@pytest.mark.parametrize(
    "wkt",
    [
        "POINT (0 1)",
        "LINESTRING (0 0, 1 1, 2 0)",
        "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
        "MULTIPOINT ((0 0), (1 1))",
        "MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))",
        "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((2 2, 3 2, 3 3, 2 3, 2 2)))",
        "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 1))",
    ],
)
def test_shapely_literal(wkt):
    shapely_obj = shapely.from_wkt(wkt)
    literal = lit(shapely_obj)

    array = pa.array(literal)
    assert array == ga.as_wkb([wkt])


def test_shapely_linearring():
    shapely_obj = shapely.from_wkt("LINEARRING (0 0, 1 0, 0 1, 0 0)")
    literal = lit(shapely_obj)

    array = pa.array(literal)
    assert array == ga.as_wkb(["LINESTRING (0 0, 1 0, 0 1, 0 0)"])


def test_geopandas_literal():
    geoseries = geopandas.GeoSeries.from_wkt(["POINT (0 1)"], crs=3857)

    # Check GeoSeries literal
    literal = lit(geoseries)
    array = pa.array(literal)
    assert array.type.crs.to_json_dict()["id"] == {"authority": "EPSG", "code": 3857}

    geopandas.testing.assert_geoseries_equal(
        geopandas.GeoSeries.from_arrow(array), geoseries
    )

    # Check GeoDataFrame literal
    geodf = geopandas.GeoDataFrame({"geom": geoseries})
    literal = lit(geodf)
    array = pa.array(literal)
    assert array.type.crs.to_json_dict()["id"] == {"authority": "EPSG", "code": 3857}

    geopandas.testing.assert_geoseries_equal(
        geopandas.GeoSeries.from_arrow(array), geoseries
    )

    # Check GeoSeries literal where the first value was None (CRS and type should
    # still propagate)
    geoseries = geopandas.GeoSeries([None], crs=3857)
    literal = lit(geoseries)
    array = pa.array(literal)
    assert array.type.crs.to_json_dict()["id"] == {"authority": "EPSG", "code": 3857}

    geopandas.testing.assert_geoseries_equal(
        geopandas.GeoSeries.from_arrow(array), geoseries
    )


def test_pandas_literal():
    series = pd.Series([1])
    pd.testing.assert_series_equal(pa.array(lit(series)).to_pandas(), series)

    df = pd.DataFrame({"x": series})
    pd.testing.assert_series_equal(pa.array(lit(df)).to_pandas(), series)

    with pytest.raises(ValueError, match="with length != 1"):
        pa.array(lit(pd.Series([])))

    with pytest.raises(ValueError, match=r"with shape != \(1, 1\)"):
        pa.array(lit(pd.DataFrame({"x": []})))

    with pytest.raises(ValueError, match=r"with shape != \(1, 1\)"):
        pa.array(lit(pd.DataFrame({"x": [1], "y": [2]})))


def test_sedonadb_literal(con):
    df = con.sql("SELECT 1 as one")
    assert pa.array(lit(df)) == pa.array([1])

    with pytest.raises(ValueError, match="number of columns != 1"):
        df = con.sql("SELECT 1 as one, 2 as two")
        pa.array(lit(df))

    with pytest.raises(ValueError, match="size != 1 row"):
        df = con.sql("SELECT 1 as one WHERE false")
        pa.array(lit(df))


def test_crs_literal():
    import pyproj

    crs = pyproj.CRS("EPSG:26920")
    assert pa.array(lit(crs)) == pa.array([crs.to_json()])

    # Ensure this is also the case for whatever GeoSeries.crs returns
    geoseries = geopandas.GeoSeries.from_wkt(["POINT (0 1)"], crs="EPSG:26920")
    assert pa.array(lit(geoseries.crs)) == pa.array([crs.to_json()])

    # Make sure geoarrow.pyarrow CRSes also work here

    # A ProjjsonCrs
    ga_crs = ga.wkb().with_crs(crs).crs
    assert pa.array(lit(ga_crs)) == pa.array([crs.to_json()])

    # A StringCrs
    ga_crs = ga.wkb().with_crs("EPSG:26920").crs
    assert pa.array(lit(ga_crs)) == pa.array([crs.to_json()])


def test_literal_funcs(con):
    literal = con.lit(5.0)
    e = literal.funcs.sqrt()
    assert isinstance(e, Expr)
    assert repr(e) == "Expr(sqrt(Float64(5)))"


def test_contextless_literal():
    literal = lit(5.0)

    with pytest.raises(ValueError, match="Can't pipe Literal"):
        literal.funcs


def test_geoarrow_scalar_keeps_edge_type():
    # The scalar path rebuilt the type from CRS alone, so a spherical
    # (geography) scalar came back as planar geometry.
    spherical = ga.wkb().with_edge_type(ga.EdgeType.SPHERICAL).with_crs(ga.OGC_CRS84)
    for payload in (shapely.Point(1, 1).wkb, None):
        lit_array = pa.array(lit(pa.scalar(payload, spherical)))
        assert lit_array.type.edge_type == ga.EdgeType.SPHERICAL
        assert lit_array.type.crs.to_json_dict()["id"] == {
            "authority": "OGC",
            "code": "CRS84",
        }
        assert lit_array.null_count == (1 if payload is None else 0)


@pytest.mark.parametrize(
    "make",
    [
        lambda wkt: ga.as_wkt([wkt])[0],
        lambda wkt: ga.as_geoarrow([wkt])[0],
        lambda wkt: pa.scalar(shapely.from_wkt(wkt).wkb, ga.large_wkb()),
    ],
    ids=["wkt", "native", "large_wkb"],
)
def test_geoarrow_non_wkb_scalar_literal(make):
    # Only the WKB scalar class was registered; other GeoArrow scalars fell
    # through to pa.array([obj]) and failed. They all expose their WKB.
    scalar = make("POINT (1 2)")
    assert pa.array(lit(scalar)) == ga.as_wkb(["POINT (1 2)"])


@pytest.mark.parametrize(
    "type",
    [ga.wkt(), ga.point(), ga.wkb().with_crs(ga.OGC_CRS84)],
    ids=["wkt", "native", "wkb_with_crs"],
)
def test_geoarrow_null_scalar_literal(type):
    lit_array = pa.array(lit(pa.scalar(None, type)))
    assert lit_array.null_count == 1
    assert lit_array.type.extension_name == "geoarrow.wkb"
    if type.crs is not None:
        assert lit_array.type.crs.to_json_dict()["id"] == {
            "authority": "OGC",
            "code": "CRS84",
        }


def test_null_arrow_scalar_literal():
    # A null nested scalar is not accepted by pa.array([obj]); the typed
    # one-element array is.
    for type in (
        pa.list_(pa.int64()),
        pa.map_(pa.string(), pa.int64()),
        pa.struct([("a", pa.int32())]),
    ):
        lit_array = pa.array(lit(pa.scalar(None, type)))
        assert lit_array.type == type
        assert lit_array.null_count == 1
    # Valid nested scalars were already fine and still are.
    assert pa.array(lit(pa.scalar([1, 2]))).to_pylist() == [[1, 2]]


def test_pandas_missing_literal():
    assert pa.array(lit(pd.NA)) == pa.array([None])
    nat = pa.array(lit(pd.NaT))
    assert nat.type == pa.timestamp("ns")
    assert nat.null_count == 1


def test_pandas_timestamp_literal():
    # pa.array([Timestamp]) resolves at microseconds and silently drops
    # nanoseconds.
    stamp = pd.Timestamp("2026-01-01 00:00:00.000000001")
    lit_array = pa.array(lit(stamp))
    assert lit_array.type == pa.timestamp("ns")
    assert lit_array[0].as_py() == stamp

    aware = pd.Timestamp("2026-01-01 00:00:00.000000001", tz="US/Pacific")
    lit_array = pa.array(lit(aware))
    assert lit_array.type == pa.timestamp("ns", "US/Pacific")
    assert lit_array[0].as_py() == aware

    # A coarser-unit Timestamp keeps its unit rather than being forced to ns
    # (2500 is outside the ns range).
    coarse = pd.Timestamp("2500-01-01").as_unit("s")
    lit_array = pa.array(lit(coarse))
    assert lit_array.type == pa.timestamp("s")
    assert lit_array[0].as_py() == coarse


def test_pandas_timedelta_literal():
    lit_array = pa.array(lit(pd.Timedelta(1)))
    assert lit_array.type == pa.duration("ns")
    assert lit_array[0].as_py() == pd.Timedelta(1)


@pytest.mark.parametrize(
    "value,expected",
    [
        (
            np.datetime64("2500-01-01", "D"),
            pa.array([np.datetime64("2500-01-01T00:00:00", "s")]),
        ),
        (
            np.datetime64("2500", "Y"),
            pa.array([np.datetime64("2500-01-01T00:00:00", "s")]),
        ),
        (
            np.datetime64("2026-01-01T00:00:00.000000001", "ns"),
            pa.array([np.datetime64("2026-01-01T00:00:00.000000001", "ns")]),
        ),
        (np.datetime64(10**6, "fs"), pa.array([np.datetime64(1, "ns")])),
        (np.timedelta64(2, "D"), pa.array([np.timedelta64(2 * 86400, "s")])),
        (np.timedelta64(3, "W"), pa.array([np.timedelta64(3 * 7 * 86400, "s")])),
        (np.timedelta64(5, "ms"), pa.array([np.timedelta64(5, "ms")])),
    ],
    ids=["day", "year", "ns", "fs_exact", "td_day", "td_week", "td_ms"],
)
def test_numpy_temporal_literal(value, expected):
    # pyarrow only understands the four Arrow units; other units convert at a
    # lossless resolution rather than being rejected or forced to ns.
    assert pa.array(lit(value)) == expected


def test_numpy_temporal_nat_literal():
    # Explicit units: newer NumPy deprecates the unit-less ("generic") NaT.
    for value, type in (
        (np.datetime64("NaT", "ns"), pa.timestamp("ns")),
        (np.timedelta64("NaT", "ns"), pa.duration("ns")),
        (np.datetime64("NaT", "D"), pa.timestamp("ns")),
    ):
        lit_array = pa.array(lit(value))
        assert lit_array.type == type
        assert lit_array.null_count == 1


def test_numpy_temporal_literal_errors():
    # A timedelta in months or years has no fixed length.
    with pytest.raises(ValueError, match="unambiguous unit"):
        pa.array(lit(np.timedelta64(1, "M")))
    # A sub-nanosecond value with no exact nanosecond form.
    with pytest.raises(ValueError, match="loses precision"):
        pa.array(lit(np.datetime64(1, "fs")))
    # A coarse value whose seconds form overflows int64.
    with pytest.raises(OverflowError):
        pa.array(lit(np.timedelta64(2**62, "D")))


def test_numpy_masked_literal():
    assert pa.array(lit(np.ma.masked)) == pa.array([None])


def test_numpy_zero_dim_array_literal():
    # A 0-d array is one value, resolved as its typed scalar (dtype kept).
    lit_array = pa.array(lit(np.array(np.int32(5))))
    assert lit_array == pa.array([5], pa.int32())
    lit_array = pa.array(lit(np.array(np.datetime64("2500-01-01"))))
    assert lit_array == pa.array([np.datetime64("2500-01-01T00:00:00", "s")])
    # Arrays with dimensions keep resolving as a single list value.
    assert pa.array(lit(np.array([1, 2]))).to_pylist() == [[1, 2]]


def test_numpy_void_literal():
    assert pa.array(lit(np.void(b"ab"))) == pa.array([b"ab"])
    record = np.array([(3, 1.5)], dtype=[("c", "int16"), ("r", "float32")])[0]
    lit_array = pa.array(lit(record))
    assert lit_array.type == pa.struct([("c", pa.int16()), ("r", pa.float32())])
    assert lit_array.to_pylist() == [{"c": 3, "r": 1.5}]
