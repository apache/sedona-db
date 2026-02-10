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
import shapely
import geopandas
import pandas as pd
import geoarrow.pyarrow as ga
import geopandas.testing

from sedonadb.expr.literal import lit
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
