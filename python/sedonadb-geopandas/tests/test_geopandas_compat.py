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

import geopandas as gpd
import pytest

import sedonadb_geopandas as sgpd
from sedonadb_geopandas import GeoDataFrame, GeoSeries, Series


@pytest.fixture
def cities():
    return gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "pop": [100, 200, 300]},
        geometry=gpd.GeoSeries.from_wkt(["POINT (0 0)", "POINT (1 1)", "POINT (5 5)"]),
        crs="EPSG:4326",
    )


@pytest.fixture
def con_free_geom_frame():
    """A frame whose active geometry column is *not* the heuristic's pick.

    Two geometry columns, `geom` and `geometry`, with `geom` marked active — so
    anything that re-derives the geometry column instead of persisting it would
    wrongly land on `geometry`.
    """
    df = sgpd.default_context().sql(
        "SELECT ST_SetSRID(ST_Point(0.0, 0.0), 3857) AS geom, "
        "ST_SetSRID(ST_Point(9.0, 9.0), 3857) AS geometry, 1 AS a"
    )
    return GeoDataFrame(df, geometry="geom")


def assert_geopandas_expr_equal(gdf, op, *, sort_by):
    """Assert an operation gives the same result on GeoPandas and on the wrapper.

    Applies `op` to `gdf` (GeoPandas) and to `sgpd.from_geopandas(gdf)`, then
    compares the materialized results. Rows are sorted by `sort_by` and the
    index reset, since the relational engine preserves neither row order nor a
    row index. `check_crs` is left on, so CRS propagation is asserted too.
    Useful for throwing a corpus of GeoPandas ops at the wrapper.
    """
    from geopandas.testing import assert_geodataframe_equal

    expected = op(gdf).sort_values(sort_by).reset_index(drop=True)
    got = (
        op(sgpd.from_geopandas(gdf))
        .to_geopandas()
        .sort_values(sort_by)
        .reset_index(drop=True)
    )
    assert_geodataframe_equal(got, expected, check_like=True, check_crs=True)


def test_from_geopandas_returns_geodataframe(cities):
    gdf = sgpd.from_geopandas(cities)
    assert isinstance(gdf, GeoDataFrame)
    assert gdf.columns == ["name", "pop", "geometry"]
    assert len(gdf) == 3


def test_roundtrip_preserves_data(cities):
    out = sgpd.from_geopandas(cities).to_geopandas().sort_values("name")
    assert list(out["name"]) == ["A", "B", "C"]
    assert list(out["pop"]) == [100, 200, 300]
    assert out.geometry.to_wkt().tolist() == cities.geometry.to_wkt().tolist()


def test_filter_boolean_mask(cities):
    gdf = sgpd.from_geopandas(cities)
    out = gdf[gdf["pop"] > 150].to_geopandas().sort_values("name")
    # Same result as GeoPandas boolean-mask indexing.
    expected = cities[cities["pop"] > 150].sort_values("name")
    assert list(out["name"]) == list(expected["name"])


def test_filter_boolean_composition(cities):
    gdf = sgpd.from_geopandas(cities)
    out = gdf[(gdf["pop"] > 150) & (gdf["pop"] < 300)].to_geopandas()
    assert list(out["name"]) == ["B"]


def test_getitem_column_types(cities):
    gdf = sgpd.from_geopandas(cities)
    assert isinstance(gdf["geometry"], GeoSeries)
    assert isinstance(gdf["pop"], Series)
    # A non-geometry Series materializes to a plain pandas Series.
    assert sorted(gdf["pop"].to_pandas().tolist()) == [100, 200, 300]


def test_geoseries_centroid_of_points_is_identity(cities):
    gdf = sgpd.from_geopandas(cities)
    got = gdf.geometry.centroid.to_geopandas().to_wkt().tolist()
    assert got == cities.geometry.to_wkt().tolist()


def test_geoseries_buffer_area(cities):
    gdf = sgpd.from_geopandas(cities)
    areas = gdf.geometry.buffer(0.5).area.to_pandas().tolist()
    # A radius-0.5 buffer has area ~= pi/4; segmentation differs slightly from
    # GeoPandas, so compare approximately.
    assert areas == pytest.approx([0.785, 0.785, 0.785], abs=0.01)


def test_to_crs(cities):
    gdf = sgpd.from_geopandas(cities)
    web = gdf.to_crs("EPSG:3857")
    assert isinstance(web, GeoDataFrame)
    # `.crs` is read cheaply from the schema (SedonaDB's CRS representation).
    assert "3857" in str(web.crs)


def test_column_subset_keeps_geometry(cities):
    gdf = sgpd.from_geopandas(cities)
    sub = gdf[["name", "geometry"]]
    assert isinstance(sub, GeoDataFrame)
    assert sub.columns == ["name", "geometry"]
    # Geometry is still usable after subsetting.
    assert isinstance(sub.geometry, GeoSeries)


def test_filter_matches_geopandas(cities):
    # Same op applied to GeoPandas and to the wrapper yields the same result.
    assert_geopandas_expr_equal(cities, lambda df: df[df["pop"] > 150], sort_by="name")


def test_to_crs_matches_geopandas(cities):
    # Exercises CRS propagation through an operation (asserted via check_crs).
    assert_geopandas_expr_equal(
        cities, lambda df: df.to_crs("EPSG:3857"), sort_by="name"
    )


@pytest.mark.parametrize("source_crs", ["EPSG:4326", "EPSG:32633"])
def test_crs_propagates_from_projected_and_geographic(source_crs):
    # CRS survives the round trip from either a geographic or projected source.
    gdf = gpd.GeoDataFrame(
        {"name": ["A", "B"]},
        geometry=gpd.GeoSeries.from_wkt(["POINT (0 0)", "POINT (1 1)"]),
        crs=source_crs,
    )
    assert_geopandas_expr_equal(gdf, lambda df: df, sort_by="name")


def test_operand_accepts_literal(cities):
    # lit() is a supported escape hatch (and the way to carry a CRS).
    from sedonadb.expr import lit

    gdf = sgpd.from_geopandas(cities)
    out = gdf[gdf["pop"] > lit(150)].to_geopandas().sort_values("name")
    assert list(out["name"]) == ["B", "C"]


def test_repr_is_lazy(cities):
    # repr() must not execute the query (IDEs/consoles call it constantly).
    gdf = sgpd.from_geopandas(cities)
    assert (
        repr(gdf)
        == "GeoDataFrame(columns=['name', 'pop', 'geometry'], geometry='geometry')"
    )
    assert "GeoSeries" in repr(gdf.geometry)
    assert "Series" in repr(gdf["pop"])


def test_operand_rejects_arraylike(cities):
    gdf = sgpd.from_geopandas(cities)
    with pytest.raises(TypeError, match="array-like"):
        gdf["pop"] > cities["pop"]  # a real pandas Series


def test_invalid_geometry_column_raises(cities):
    df = sgpd.default_context().create_data_frame(cities)
    # A non-geometry column named as the geometry is rejected.
    with pytest.raises(ValueError, match="not a geometry column"):
        GeoDataFrame(df, geometry="pop")
    with pytest.raises(KeyError, match="not found"):
        GeoDataFrame(df, geometry="nope")


def test_no_geometry_frame(cities):
    # Dropping the geometry column yields a frame with no active geometry.
    # GeoPandas returns a plain DataFrame here, whose .geometry also raises.
    gdf = sgpd.from_geopandas(cities)
    plain = gdf[["name", "pop"]]
    assert plain.crs is None
    with pytest.raises(AttributeError, match="no active geometry"):
        plain.geometry
    assert not hasattr(cities[["name", "pop"]], "geometry")


def test_column_subset_persists_custom_geometry_name(con_free_geom_frame):
    # The active geometry column is persisted through a subset, not re-derived
    # (re-deriving would pick "geometry" over the active "geom").
    gdf = con_free_geom_frame
    assert gdf["geom"] is not None
    sub = gdf[["a", "geom"]]
    assert sub._geometry_name == "geom"
    assert sub.crs is not None


def test_to_geopandas_persists_custom_geometry_name(con_free_geom_frame):
    # A custom active geometry column survives the trip back to GeoPandas, even
    # when another column would win SedonaDB's primary-geometry heuristic.
    out = con_free_geom_frame.to_geopandas()
    assert out.geometry.name == "geom"


def test_cross_frame_series_raises(cities):
    # Combining Series from two different frames has no row alignment, so it
    # must error rather than silently produce wrong rows.
    a = sgpd.from_geopandas(cities)
    b = sgpd.from_geopandas(cities)
    with pytest.raises(ValueError, match="different DataFrames"):
        a["pop"] > b["pop"]


def test_slice_and_integer_keys(cities):
    gdf = sgpd.from_geopandas(cities)
    with pytest.raises(TypeError, match="Positional row slicing"):
        gdf[0:2]
    with pytest.raises(KeyError, match="column label"):
        gdf[0]


def test_head(cities):
    gdf = sgpd.from_geopandas(cities)
    assert len(gdf.head(2)) == 2
    # head() keeps the active geometry column.
    assert isinstance(gdf.head(2).geometry, GeoSeries)
