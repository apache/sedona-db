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

import sys

import geoarrow.pyarrow as ga
import geopandas as gpd
import numpy as np
import numpy.ma.mrecords as mrecords
import pandas as pd
import pyarrow as pa
import pytest
from geopandas.testing import assert_geodataframe_equal
from sedonadb.expr import lit
from shapely.geometry import MultiPoint, Point

import sedonadb_geopandas as sgpd
from sedonadb_geopandas import GeoDataFrame, GeoSeries, Series
from sedonadb_geopandas._frame import _is_missing
from sedonadb_geopandas._series import is_scalar


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


def test_setitem_from_series():
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    gdf = sgpd.from_geopandas(points)
    gdf["v2"] = gdf["v"]
    got = gdf.to_geopandas().sort_values("name")
    assert got["v2"].tolist() == points["v"].tolist()


def test_setitem_replaces_existing_column():
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    gdf = sgpd.from_geopandas(points)
    gdf["v"] = gdf["name"]
    assert sorted(gdf.to_geopandas()["v"]) == ["A", "B", "C"]
    assert gdf.columns.count("v") == 1


def test_setitem_scalar_broadcasts():
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    gdf = sgpd.from_geopandas(points)
    gdf["k"] = 7
    assert gdf.to_geopandas()["k"].tolist() == [7, 7, 7]


def test_setitem_geometry_column():
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    gdf = sgpd.from_geopandas(points)
    gdf["buffered"] = gdf.geometry.buffer(0.5)
    assert "buffered" in gdf.columns
    # The active geometry column is unchanged by adding another one.
    assert gdf.geometry._name == "geometry"


def test_setitem_makes_geometry_active_when_frame_had_none():
    # A frame that starts without geometry gains an active geometry column when
    # one is assigned.

    plain = GeoDataFrame(sgpd.default_context().sql("SELECT 1 AS a"))
    assert plain._geometry_name is None

    plain["geometry"] = Point(1, 2)
    assert plain._geometry_name == "geometry"
    assert plain.to_geopandas().geometry.to_wkt().tolist() == ["POINT (1 2)"]


def test_setitem_non_geometry_leaves_frame_without_geometry():
    plain = GeoDataFrame(sgpd.default_context().sql("SELECT 1.0 AS x"))
    plain["doubled"] = plain["x"]
    assert plain._geometry_name is None


def test_setitem_rejects_bad_inputs():
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    gdf = sgpd.from_geopandas(points)
    with pytest.raises(TypeError, match="must be a string"):
        gdf[0] = 1
    with pytest.raises(TypeError, match="isn't supported"):
        gdf["x"] = points["v"]
    # A Series from a different frame has no row alignment.
    other = sgpd.from_geopandas(points)
    with pytest.raises(ValueError, match="different"):
        gdf["y"] = other["v"]


def test_setitem_stale_series_raises():
    # Replacing a column rebinds the frame in a way earlier reads cannot
    # follow: a Series read beforehand would resolve to the new values, so it
    # is stale and raises.
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    gdf = sgpd.from_geopandas(points)
    before = gdf["v"]
    gdf["v"] = gdf["name"]
    with pytest.raises(ValueError, match="different"):
        gdf["z"] = before


def test_series_survives_column_adding_assignments():
    # One captured geometry can supply several derived columns in turn: an
    # assignment that only adds a column leaves rows and existing columns
    # untouched, so an earlier Series still resolves to the values it showed.
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    gdf = sgpd.from_geopandas(points)
    g = gdf.geometry
    v = gdf["v"]
    gdf["buffered"] = g.buffer(1.0)
    gdf["area"] = g.buffer(1.0).area
    gdf["x_plus"] = v
    out = gdf.to_geopandas().sort_values("name")
    assert out["x_plus"].tolist() == points["v"].tolist()
    assert (out["area"] > 3.0).all()
    # ... and a filter by an earlier mask is equally valid.
    mask = gdf["v"] > 1
    gdf["k"] = 1
    assert len(gdf[mask]) == 2


def test_setitem_rejects_bare_expression():
    # A bare expression carries no origin, so one built from another frame would
    # resolve against this frame and silently write this frame's values.
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    left = sgpd.from_geopandas(points)
    right = sgpd.from_geopandas(points)
    with pytest.raises(TypeError, match="bare expression"):
        left["copied"] = right["v"]._expr


def test_series_has_no_unguarded_expr_escape_hatch():
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    assert not hasattr(sgpd.from_geopandas(points)["v"], "expr")


@pytest.mark.parametrize(
    "value",
    [[10, 20, 30], (10, 20, 30), {10, 20}],
    ids=["list", "tuple", "set"],
)
def test_setitem_rejects_sequences(value):
    # Sequences have no __array__, so they used to be broadcast whole into every
    # row rather than rejected.
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    gdf = sgpd.from_geopandas(points)
    with pytest.raises(TypeError, match="isn't supported"):
        gdf["x"] = value


def test_setitem_accepts_numpy_scalar():
    # NumPy scalars do have __array__ but are single values, so they used to be
    # rejected as array-likes.
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    gdf = sgpd.from_geopandas(points)
    gdf["x"] = np.int64(5)
    assert gdf.to_geopandas()["x"].tolist() == [5, 5, 5]


def test_setitem_rejects_numpy_array():
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    gdf = sgpd.from_geopandas(points)
    with pytest.raises(TypeError, match="isn't supported"):
        gdf["x"] = np.array([1, 2, 3])


def test_getitem_rejects_stale_mask():
    # Replacing a column (or filtering) rebinds the frame, so a mask captured
    # beforehand belongs to the previous one. It used to be accepted and
    # quietly resolve against the new frame while the referenced column
    # happened to still exist.
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    gdf = sgpd.from_geopandas(points)
    mask = gdf["v"] > 1
    gdf["v"] = gdf["name"]
    with pytest.raises(ValueError, match="different DataFrame"):
        gdf[mask]
    gdf = sgpd.from_geopandas(points)
    mask = gdf["v"] > 1
    filtered = gdf[gdf["v"] > 0]
    with pytest.raises(ValueError, match="different DataFrame"):
        filtered[mask]


def test_setitem_none_keeps_geometry_column():
    # Assigning None used to turn the column untyped and clear the active geometry;
    # GeoPandas keeps a geometry column with its CRS.
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    gdf = sgpd.from_geopandas(points)
    gdf["geometry"] = None
    assert gdf._geometry_name == "geometry"
    assert "3857" in str(gdf.crs)
    assert gdf.to_geopandas().geometry.isna().all()


def test_setitem_non_geometry_over_geometry_still_clears():
    # The CRS-preserving path must not dress a number up as geometry.
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    gdf = sgpd.from_geopandas(points)
    gdf["geometry"] = 7
    assert gdf._geometry_name is None
    assert gdf.to_geopandas()["geometry"].tolist() == [7, 7, 7]


@pytest.mark.parametrize(
    "value_name",
    ["none", "nan", "pandas_na", "geometry", "literal_geometry", "literal_none"],
)
def test_setitem_geometry_scalars_keep_type_and_crs(value_name):
    # Every supported scalar path has to go through the CRS-preserving branch:
    # GeoPandas treats None, NaN and pd.NA as missing geometry and keeps the typed
    # column and its CRS.
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    values = {
        "none": None,
        "nan": np.nan,
        "pandas_na": pd.NA,
        "geometry": Point(5, 5),
        "literal_geometry": lit(Point(5, 5)),
        "literal_none": lit(None),
    }
    gdf = sgpd.from_geopandas(points)
    gdf["geometry"] = values[value_name]
    assert gdf._geometry_name == "geometry"
    assert "3857" in str(gdf.crs)


def test_setitem_crs_carrying_literal_keeps_its_crs():
    # A literal that carries its own CRS must not be relabeled with the
    # destination column's CRS — that changes what the coordinates mean without
    # transforming them.
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    src = gpd.GeoSeries.from_wkt(["POINT (10 10)"], crs="EPSG:4326")
    gdf = sgpd.from_geopandas(points)  # column is EPSG:3857
    gdf["geometry"] = lit(src)
    assert "4326" in str(gdf.crs)


def test_pandas_na_assigns_as_null_to_ordinary_column():
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    gdf = sgpd.from_geopandas(points)
    gdf["z"] = pd.NA
    assert gdf.to_geopandas()["z"].isna().all()


def test_zero_dimensional_array_normalizes():
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    gdf = sgpd.from_geopandas(points)
    gdf["z"] = np.array(5)  # 0-d: scalar by classification, unwrapped on use
    assert gdf.to_geopandas()["z"].tolist() == [5, 5, 5]


def test_masked_scalar_assigns_as_missing():
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    gdf = sgpd.from_geopandas(points)
    gdf["m"] = np.ma.masked
    assert gdf.to_geopandas()["m"].isna().all()


def test_pyarrow_scalars_broadcast():
    # Arrow scalars implement __len__ but are single values.
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    assert is_scalar(pa.scalar([1, 2]))
    assert is_scalar(pa.scalar({"a": 1}))
    gdf = sgpd.from_geopandas(points)
    gdf["tags"] = pa.scalar([1, 2])
    assert len(gdf.to_geopandas()["tags"]) == 3


def test_geoarrow_scalar_inherits_crs():
    # A GeoArrow WKB scalar has no __geo_interface__, so geometry-ness must come
    # from the resolved schema; it is CRS-less and inherits the column's CRS.
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    w = ga.as_wkb(ga.array(["POINT (5 5)"]))[0]
    gdf = sgpd.from_geopandas(points)
    gdf["geometry"] = w
    assert gdf._geometry_name == "geometry"
    assert "3857" in str(gdf.crs)


def test_explicitly_inactive_geometry_stays_inactive():
    # geometry=None is a choice; a no-op reassignment of an existing geometry
    # column must not silently reactivate it.
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    df = sgpd.default_context().create_data_frame(points)
    gdf = GeoDataFrame(df, geometry=None)
    gdf["geometry"] = gdf["geometry"]
    assert gdf._geometry_name is None


def test_assigned_geometry_column_reads_back_as_geoseries():
    # Only the active geometry name produced a GeoSeries, so the advertised
    # gdf["buffered"] = gdf.geometry.buffer(...) gave back a plain Series
    # with no .area or .buffer(). Geometry-ness comes from the schema.
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    gdf = sgpd.from_geopandas(points)
    gdf["buffered"] = gdf.geometry.buffer(0.5)
    col = gdf["buffered"]
    assert isinstance(col, type(gdf.geometry))
    assert (col.area.to_pandas() > 0).all()


def test_cleared_geometry_is_not_resurrected_by_materialization():
    # With the active geometry replaced by a number, the wrapper records no
    # active geometry — but the materializer heuristically activated any
    # remaining geometry column, so a later to_crs() on the result silently
    # targeted a column this frame never had active.
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    gdf = sgpd.from_geopandas(points)
    gdf["copy"] = gdf.geometry
    gdf["geometry"] = 7
    assert gdf._geometry_name is None
    out = gdf.to_geopandas()
    with pytest.raises(AttributeError):
        out.geometry


def test_series_assignment_inherits_destination_crs():
    # Assigning a CRS-less same-frame geometry column over a column that has
    # a CRS dropped it; GeoPandas keeps the frame CRS. A column carrying its
    # own CRS keeps it instead.
    gdf = GeoDataFrame(
        sgpd.default_context().sql(
            "SELECT ST_SetSRID(ST_Point(0.0, 0.0), 3857) AS geometry, "
            "ST_Point(5.0, 5.0) AS bare, "
            "ST_SetSRID(ST_Point(7.0, 7.0), 4326) AS own, 1 AS v"
        )
    )
    gdf["geometry"] = gdf["bare"]
    assert "3857" in str(gdf.crs)
    gdf["geometry"] = gdf["own"]
    # The engine reports SRID 4326 as OGC:CRS84; the point is that the
    # source's own CRS survives instead of being restamped with 3857.
    assert gdf.crs is not None
    assert "3857" not in str(gdf.crs)


def test_geography_column_replacement_preserves_geography():
    # Replacing a geography column with None or a Shapely scalar rebuilt it
    # as planar geometry; replacements are constructed with the destination's
    # own spatial kind.

    def geog_frame():
        return GeoDataFrame(
            sgpd.default_context().sql(
                "SELECT ST_GeogFromWKT('POINT (0 0)') AS g, 1 AS v"
            ),
            geometry="g",
        )

    gdf = geog_frame()
    gdf["g"] = None
    assert "geography" in str(gdf._df.schema.field("g").type)
    assert gdf._geometry_name == "g"
    gdf = geog_frame()
    gdf["g"] = Point(1, 1)
    assert "geography" in str(gdf._df.schema.field("g").type)
    got = gdf.to_geopandas()["g"]
    assert got.tolist()[0] == Point(1, 1)

    # A non-default CRS survives too: the geography constructors synthesize
    # CRS84, which must not be mistaken for a CRS the value carried itself.
    def geog_4267():
        return GeoDataFrame(
            sgpd.default_context().sql(
                "SELECT ST_SetSRID(ST_GeogFromWKT('POINT (0 0)'), 4267) AS g, 1 AS v"
            ),
            geometry="g",
        )

    for value in (None, Point(1, 1)):
        gdf = geog_4267()
        gdf["g"] = value
        dtype = str(gdf._df.schema.field("g").type)
        assert "geography" in dtype
        assert "4267" in dtype


def test_numpy_scalar_dtypes_are_preserved():
    # .item() promoted np.int8/np.float32 to int64/float64 columns and
    # overflowed np.uint64 past int64, which the engine supports natively.
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    gdf = sgpd.from_geopandas(points)
    gdf["i8"] = np.int8(5)
    gdf["f4"] = np.float32(1.5)
    gdf["u64"] = np.uint64(2**63 + 1)
    gdf["z"] = np.array(np.int32(9))  # 0-d arrays unwrap to the typed scalar
    out = gdf.to_geopandas()
    assert str(out["i8"].dtype) == "int8"
    assert str(out["f4"].dtype) == "float32"
    assert out["u64"].tolist() == [2**63 + 1] * len(out)
    assert str(out["z"].dtype) == "int32"


def test_arrow_wrapped_missing_values_keep_geometry():
    # pa.scalar(None), typed Arrow nulls, and Arrow-wrapped NaN cleared the
    # active geometry and CRS, unlike the equivalent bare None/NaN; a wrapped
    # value means whatever its payload means.
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    for value in (
        pa.scalar(None),
        pa.scalar(None, pa.float64()),
        pa.scalar(float("nan")),
    ):
        gdf = sgpd.from_geopandas(points)
        gdf["geometry"] = value
        assert gdf._geometry_name == "geometry"
        assert "3857" in str(gdf.crs)
        assert gdf.to_geopandas()["geometry"].isna().all()


def test_typed_null_nested_scalars_broadcast():
    # Valid nested Arrow scalars broadcast, but their typed-null forms failed
    # literal construction; they re-enter as one-element typed arrays.
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    for value in (
        pa.scalar(None, pa.list_(pa.int64())),
        pa.scalar(None, pa.map_(pa.string(), pa.int64())),
    ):
        gdf = sgpd.from_geopandas(points)
        gdf["x"] = value
        assert gdf.to_geopandas()["x"].isna().all()


def test_multipart_geometries_classify_as_scalars():
    # A multipart geometry is a single value (Shapely 2 no longer gives it a
    # sequence protocol, and the package requires Shapely 2).
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    value = MultiPoint([(0, 0), (1, 1)])
    assert is_scalar(value)
    gdf = sgpd.from_geopandas(points)
    gdf["mp"] = value
    assert gdf.to_geopandas()["mp"].tolist() == [value] * len(points)


def test_no_active_geometry_survives_any_column_name():
    # The materializer rebuild went through the GeoDataFrame constructor,
    # which auto-activates a geometry column literally named "geometry" — so
    # clearing the active `geom` resurrected the secondary column and a later
    # to_crs() would transform it. The no-active marker is preserved
    # explicitly, whatever the remaining columns are called.
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    gdf = GeoDataFrame(
        sgpd.default_context().sql(
            "SELECT ST_SetSRID(ST_Point(0.0, 0.0), 3857) AS geom, "
            "ST_SetSRID(ST_Point(9.0, 9.0), 4326) AS geometry, 1 AS a"
        ),
        geometry="geom",
    )
    gdf["geom"] = 7
    assert gdf._geometry_name is None
    out = gdf.to_geopandas()
    with pytest.raises(AttributeError):
        out.geometry

    explicit = GeoDataFrame(
        sgpd.default_context().create_data_frame(points), geometry=None
    )
    out = explicit.to_geopandas()
    with pytest.raises(AttributeError):
        out.geometry


def test_numpy_void_scalars_broadcast_as_binary():
    # np.void has no Arrow scalar form, so the typed-scalar conversion broke
    # what previously worked: void values broadcast through .item() as bytes.
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    gdf = sgpd.from_geopandas(points)
    gdf["x"] = np.void(b"abcd")
    assert gdf.to_geopandas()["x"].tolist() == [b"abcd"] * len(points)
    gdf["y"] = np.array(np.void(b"ab"))
    assert gdf.to_geopandas()["y"].tolist() == [b"ab"] * len(points)


def test_typed_null_nested_scalar_keeps_geometry():
    # Normalization rewrites a typed-null nested scalar into its one-element
    # array spelling before missingness is judged; recognizing that spelling
    # only worked through pandas coincidence, and without pandas the null
    # converted the geometry column to a list column. Classification is
    # explicit now: one null element is one missing value.
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    gdf = sgpd.from_geopandas(points)
    gdf["geometry"] = pa.scalar(None, pa.list_(pa.int64()))
    assert gdf._geometry_name == "geometry"
    assert "3857" in str(gdf.crs)
    assert gdf.to_geopandas()["geometry"].isna().all()


def test_crs_less_geography_replacement_stays_crs_less():
    # A geography constructor synthesizes CRS84, so replacing a CRS-less
    # geography with None or a Shapely value silently gained a CRS; the
    # synthesized one is stripped back off when the destination has none.

    def crs_less():
        df = sgpd.default_context().sql(
            "SELECT ST_GeogFromWKT('POINT (0 0)') AS g, 1 AS v"
        )
        df = df.mutate(g=df["g"].funcs.st_setcrs(df._ctx.lit(None)))
        return GeoDataFrame(df, geometry="g")

    base = crs_less()
    assert base._df.schema.field("g").type.crs is None
    for value in (None, Point(1, 1)):
        gdf = crs_less()
        gdf["g"] = value
        dtype = gdf._df.schema.field("g").type
        assert "geography" in str(dtype)
        assert dtype.crs is None
        # The schema alone is not enough: stripping the CRS with a
        # null-propagating expression kept the type right while erasing
        # every value.
        got = gdf.to_geopandas()["g"]
        if value is None:
            assert got.isna().all()
        else:
            assert got.tolist() == [value]


def test_typed_spatial_null_keeps_its_own_crs():
    # An invalid GeoArrow scalar typed EPSG:4267 was routed through the
    # missing path, which synthesized a destination-kind null stamped with
    # the destination CRS — while the equivalent valid scalar kept 4267. A
    # typed spatial null is a geometry value that happens to be null: it
    # keeps its own kind and CRS metadata.
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    crs4267 = gpd.GeoSeries.from_wkt(["POINT (0 0)"], crs="EPSG:4267").crs
    null_scalar = pa.scalar(None, ga.wkb().with_crs(crs4267.to_json()))
    gdf = sgpd.from_geopandas(points)
    gdf["geometry"] = null_scalar
    dtype = gdf._df.schema.field("geometry").type
    assert "4267" in str(dtype)
    assert gdf.to_geopandas()["geometry"].isna().all()
    # A typed *non-spatial* null still means "missing geometry".
    gdf = sgpd.from_geopandas(points)
    gdf["geometry"] = pa.scalar(None, pa.float64())
    assert "3857" in str(gdf._df.schema.field("geometry").type)


def test_spherical_geoarrow_scalars_keep_geography():
    # The scalar literal resolver drops the edge type, so both a null and a
    # valid spherical WKB scalar silently became planar geometry; the
    # one-element-array spelling resolves with the complete extension
    # metadata.
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    crs4267 = gpd.GeoSeries.from_wkt(["POINT (0 0)"], crs="EPSG:4267").crs
    sph = ga.wkb().with_edge_type(ga.EdgeType.SPHERICAL).with_crs(crs4267.to_json())
    for payload in (None, Point(1, 1).wkb):
        gdf = sgpd.from_geopandas(points)
        gdf["geometry"] = pa.scalar(payload, sph)
        dtype = str(gdf._df.schema.field("geometry").type)
        assert "geography" in dtype
        assert "4267" in dtype


def test_non_wkb_geoarrow_nulls_do_not_crash():
    # Null WKT/point GeoArrow scalars raised ValueError from the literal
    # resolver inside the spatial-null detector. They now either resolve
    # through the array spelling or degrade to the destination-kind null —
    # never an error, and always still geometry.
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    for typ in (ga.wkt(), ga.point()):
        gdf = sgpd.from_geopandas(points)
        gdf["geometry"] = pa.scalar(None, typ)
        assert "geometry" in str(gdf._df.schema.field("geometry").type)
        assert gdf.to_geopandas()["geometry"].isna().all()


def test_clearing_active_geometry_does_not_retype_columns():
    # Clearing the marker by reconstructing the frame let the GeoDataFrame
    # constructor coerce an unrelated all-null column named "geometry" to
    # geometry dtype; the marker is cleared in place instead.
    gdf = GeoDataFrame(
        sgpd.default_context().sql(
            "SELECT CAST(NULL AS VARCHAR) AS geometry, "
            "ST_SetSRID(ST_Point(0.0, 0.0), 3857) AS copy"
        ),
        geometry=None,
    )
    out = gdf.to_geopandas()
    assert str(out["geometry"].dtype) != "geometry"
    with pytest.raises(AttributeError):
        out.geometry


def test_structured_numpy_scalars_keep_fields_and_dtypes():
    # A structured np.void flattened to a tuple, silently storing
    # [("count", int16), ("ratio", float32)] as list<float64>; it broadcasts
    # as a typed Arrow struct with field names and dtypes intact.
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    rec = np.array([(3, 1.5)], dtype=[("count", "int16"), ("ratio", "float32")])[0]
    gdf = sgpd.from_geopandas(points)
    gdf["x"] = rec
    dtype = str(gdf._df.schema.field("x").type)
    assert "Int16" in dtype and "Float32" in dtype
    assert gdf.to_geopandas()["x"].tolist() == [{"count": 3, "ratio": 1.5}] * len(
        points
    )


def test_is_missing_handles_null_array_without_pandas(monkeypatch):
    # The one-element-null array classification must not depend on optional
    # pandas; blocking the import proves the branch answers on its own.

    null_array = pa.array([None], type=pa.list_(pa.int64()))
    value_array = pa.array([1], type=pa.int64())
    monkeypatch.setitem(sys.modules, "pandas", None)
    assert _is_missing(null_array)
    assert not _is_missing(value_array)


def test_masked_structured_records():
    # A structured scalar drawn from a MaskedArray failed inside
    # np.ma.is_masked before any conversion ran — even fully unmasked.
    # Unmasked and partially masked records broadcast as typed structs with
    # masked fields null; a fully masked record is missing.
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    dtype = [("count", "int16"), ("ratio", "float32")]

    def record(mask):
        return np.ma.MaskedArray([(3, 1.5)], mask=[mask], dtype=dtype)[0]

    gdf = sgpd.from_geopandas(points)
    gdf["a"] = record((False, False))
    gdf["b"] = record((True, False))
    gdf["c"] = record((True, True))
    out = gdf.to_geopandas()
    assert out["a"].tolist() == [{"count": 3, "ratio": 1.5}] * len(points)
    assert out["b"].tolist() == [{"count": None, "ratio": 1.5}] * len(points)
    assert out["c"].isna().all()


def test_geoarrow_scalars_to_new_columns():
    # The new-column fast path returned lit(raw) before GeoArrow handling
    # began: null WKT/point scalars raised ValueError, and a spherical WKB
    # scalar silently became planar — and could even become the active
    # geometry of a geometry-less frame with the wrong semantics.
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    crs4267 = gpd.GeoSeries.from_wkt(["POINT (0 0)"], crs="EPSG:4267").crs
    sph = ga.wkb().with_edge_type(ga.EdgeType.SPHERICAL).with_crs(crs4267.to_json())

    gdf = sgpd.from_geopandas(points)
    gdf["fresh"] = pa.scalar(Point(1, 1).wkb, sph)
    dtype = str(gdf._df.schema.field("fresh").type)
    assert "geography" in dtype and "4267" in dtype
    for typ in (ga.wkt(), ga.point()):
        gdf = sgpd.from_geopandas(points)
        gdf["fresh"] = pa.scalar(None, typ)
        assert "geometry" in str(gdf._df.schema.field("fresh").type)

    plain = GeoDataFrame(sgpd.default_context().sql("SELECT 1 AS a"))
    plain["g"] = pa.scalar(Point(1, 1).wkb, sph)
    assert "geography" in str(plain._df.schema.field("g").type)
    assert plain._geometry_name == "g"


def test_non_wkb_geoarrow_null_keeps_its_own_crs():
    # The metadata-rebuilt null stamped str(StringCrs(...)) into ST_SetCRS,
    # which is not PROJJSON and failed deserialization; the canonical
    # to_json() form is used instead.
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    crs4267 = gpd.GeoSeries.from_wkt(["POINT (0 0)"], crs="EPSG:4267").crs
    gdf = sgpd.from_geopandas(points)
    gdf["geometry"] = pa.scalar(None, ga.wkt().with_crs(crs4267.to_json()))
    dtype = str(gdf._df.schema.field("geometry").type)
    assert "4267" in dtype
    assert gdf.to_geopandas()["geometry"].isna().all()


def test_zero_dim_structured_masked_containers():
    # A 0-d structured MaskedArray raised an opaque structured-dtype-to-bool
    # TypeError inside the generic mask check, directly and inside a Literal;
    # it unwraps to its record form first.
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    dtype = [("count", "int16"), ("ratio", "float32")]

    def container(mask):
        return np.ma.MaskedArray((3, 1.5), mask=mask, dtype=dtype)

    gdf = sgpd.from_geopandas(points)
    gdf["a"] = container((False, False))
    gdf["b"] = container((True, False))
    gdf["c"] = container((True, True))
    gdf["d"] = lit(container((True, False)))
    out = gdf.to_geopandas()
    assert out["a"].tolist() == [{"count": 3, "ratio": 1.5}] * len(points)
    assert out["b"].tolist() == [{"count": None, "ratio": 1.5}] * len(points)
    assert out["c"].isna().all()
    assert out["d"].tolist() == [{"count": None, "ratio": 1.5}] * len(points)


def test_large_wkb_scalars_normalize_to_binary_storage():
    # ga.large_wkb() scalars carry LargeBinary storage, which SedonaDB's WKB
    # importer rejects; they are rebuilt on Binary storage with the same CRS
    # and edge metadata. Valid and null, direct and Literal, fresh and
    # geometry destinations.
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    crs4267 = gpd.GeoSeries.from_wkt(["POINT (0 0)"], crs="EPSG:4267").crs
    lws = (
        ga.large_wkb().with_edge_type(ga.EdgeType.SPHERICAL).with_crs(crs4267.to_json())
    )
    for payload in (Point(1, 1).wkb, None):
        gdf = sgpd.from_geopandas(points)
        gdf["fresh"] = pa.scalar(payload, lws)
        dtype = str(gdf._df.schema.field("fresh").type)
        assert "geography" in dtype and "4267" in dtype
    gdf = sgpd.from_geopandas(points)
    gdf["geometry"] = lit(pa.scalar(Point(1, 1).wkb, ga.large_wkb()))
    assert "3857" in str(gdf._df.schema.field("geometry").type)
    assert gdf.to_geopandas()["geometry"].tolist() == [Point(1, 1)] * len(points)


def test_masked_records_do_not_recurse():
    # 0-d MaskedRecords' own [()] returns another 0-d MaskedRecords, so the
    # structured-masked unwrap recursed until RecursionError; the base
    # MaskedArray view yields the record form. All three mask states, direct
    # and through Literal.
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    dtype = [("count", "int16"), ("ratio", "float32")]

    def record(mask):
        return np.ma.MaskedArray((3, 1.5), mask=mask, dtype=dtype).view(
            mrecords.MaskedRecords
        )

    gdf = sgpd.from_geopandas(points)
    gdf["a"] = record((False, False))
    gdf["b"] = record((True, False))
    gdf["c"] = record((True, True))
    gdf["d"] = lit(record((True, False)))
    out = gdf.to_geopandas()
    assert out["a"].tolist() == [{"count": 3, "ratio": 1.5}] * len(points)
    assert out["b"].tolist() == [{"count": None, "ratio": 1.5}] * len(points)
    assert out["c"].isna().all()
    assert out["d"].tolist() == [{"count": None, "ratio": 1.5}] * len(points)


@pytest.mark.parametrize(
    "value_name,expected_kind",
    [
        ("ns_datetime", "datetime"),
        ("zero_d_ns_datetime", "datetime"),
        ("day_unit_datetime", "datetime"),
        ("day_unit_timedelta", "timedelta"),
        ("datetime_nat", "datetime_nat"),
        ("timedelta_nat", "timedelta_nat"),
    ],
)
def test_numpy_temporals_materialize_correctly(value_name, expected_kind):
    # Asserted end to end (assignment then collection), not on the normalization
    # helper: .item() flattening, unit rejection, and zeroed durations were all
    # invisible to helper-level equality checks.
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    values = {
        "ns_datetime": np.datetime64("2026-01-01", "ns"),
        "zero_d_ns_datetime": np.array(np.datetime64("2026-01-01", "ns")),
        "day_unit_datetime": np.datetime64("2026-01-01"),
        "day_unit_timedelta": np.timedelta64(2, "D"),
        "datetime_nat": np.datetime64("NaT", "ns"),
        "timedelta_nat": np.timedelta64("NaT", "ns"),
    }
    gdf = sgpd.from_geopandas(points)
    gdf["t"] = values[value_name]
    out = gdf.to_geopandas()["t"]
    if expected_kind == "datetime":
        assert out.tolist() == [pd.Timestamp("2026-01-01")] * 3
    elif expected_kind == "timedelta":
        assert out.tolist() == [pd.Timedelta(days=2)] * 3
    else:
        # NaT stays a typed null of the right family.
        assert out.isna().all()
        expected_dtype = "datetime64" if "datetime" in expected_kind else "timedelta64"
        assert expected_dtype in str(out.dtype)


def test_coarse_unit_temporals_are_exact():
    # Forcing every temporal through nanoseconds silently wrapped values
    # outside the ns range (1677-2262): 2500-01-01 materialized as 1915-06-14.
    # Coarse units convert exactly to seconds instead. Expectations are
    # independent second-resolution constants — deriving them from the
    # materialized dtype would repeat the conversion under test and wrap
    # identically against a broken implementation.
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    cases = [
        (np.datetime64("2500-01-01", "D"), np.datetime64("2500-01-01T00:00:00", "s")),
        (np.timedelta64(200000, "D"), np.timedelta64(200000 * 86400, "s")),
        (np.datetime64("2500", "Y"), np.datetime64("2500-01-01T00:00:00", "s")),
    ]
    for value, expected in cases:
        gdf = sgpd.from_geopandas(points)
        gdf["t"] = value
        got = gdf.to_geopandas()["t"].values[0]
        assert got.dtype == expected.dtype
        assert got == expected


def test_ambiguous_and_subnano_temporal_units_are_rejected():
    # Matches pandas, exception type included: timedelta months/years have no
    # fixed length, and pandas raises ValueError for sub-nanosecond timedeltas
    # even when the value is exactly representable.
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    gdf = sgpd.from_geopandas(points)
    for unit in ("M", "Y", "ps"):
        with pytest.raises(ValueError, match="exactly"):
            gdf["t"] = np.timedelta64(1, unit)
    # 1000 ps is exactly one nanosecond, and both pandas and GeoPandas still
    # reject it — the rejection is per unit, not per value.
    with pytest.raises(ValueError, match="exactly"):
        gdf["t"] = np.timedelta64(1000, "ps")


def test_exact_subnanosecond_datetimes_are_accepted():
    # 10**6 fs and 10**9 as are exactly one nanosecond and GeoPandas accepts
    # them, so rejecting every sub-ns datetime unit up front was too broad.
    # Lossy values are still rejected — GeoPandas silently truncates those to
    # nanoseconds instead, which this layer deliberately does not do.
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    for value in (np.datetime64(10**6, "fs"), np.datetime64(10**9, "as")):
        gdf = sgpd.from_geopandas(points)
        gdf["t"] = value
        assert gdf.to_geopandas()["t"].values[0] == np.datetime64(1, "ns")

    gdf = sgpd.from_geopandas(points)
    with pytest.raises(ValueError, match="precision"):
        gdf["t"] = np.datetime64(1, "fs")


def test_zero_dim_object_array_holding_temporal():
    # A 0-d object array classifies as a broadcastable scalar, but its .item()
    # returns the wrapped numpy temporal, which bypassed temporal
    # normalization and failed assignment for non-Arrow-native units.
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    gdf = sgpd.from_geopandas(points)
    gdf["t"] = np.array(np.datetime64("2500-01-01", "D"), dtype=object)
    got = gdf.to_geopandas()["t"].values[0]
    assert got == np.datetime64("2500-01-01T00:00:00", "s")


def test_pandas_temporal_scalars_keep_nanoseconds():
    # pandas Timestamp/Timedelta scalars resolved to microsecond literals:
    # assignment silently zeroed nanoseconds, and duration arithmetic lost
    # them behind an interval coercion (`t + pd.Timedelta(1)` came back as
    # DateOffset objects with the tick dropped). They route through their
    # numpy form and its lossless unit handling instead.
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    gdf = sgpd.from_geopandas(points)
    gdf["t"] = pd.Timedelta(1)
    got = gdf.to_geopandas()["t"]
    assert got.tolist() == [pd.Timedelta(1)] * len(got)
    gdf["ts"] = pd.Timestamp("2026-01-01 00:00:00.000000001")
    got = gdf.to_geopandas()["ts"]
    assert got.tolist() == [pd.Timestamp("2026-01-01 00:00:00.000000001")] * len(got)


def test_tz_aware_timestamp_scalars_keep_nanoseconds_and_zone():
    # pyarrow resolves a zone-aware Timestamp at microseconds, silently
    # truncating nanoseconds; the scalar is rebuilt at nanosecond ticks with
    # its zone preserved.
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    gdf = sgpd.from_geopandas(points)
    stamp = pd.Timestamp("2026-01-01 00:00:00.000000001", tz="US/Pacific")
    gdf["ts"] = stamp
    got = gdf.to_geopandas()["ts"]
    assert got.dt.tz is not None
    assert got.tolist() == [stamp] * len(got)


def test_nat_assigns_as_datetime_missing():
    # pd.NaT is an instance of neither Timestamp nor Timedelta, so it slipped
    # past temporal normalization entirely: ordinary assignment failed with a
    # backend error while geometry assignment absorbed it as missing. It now
    # assigns the way pandas assigns it — a datetime column of missing values —
    # while geometry columns keep treating it as a missing geometry.
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    gdf = sgpd.from_geopandas(points)
    gdf["x"] = pd.NaT
    got = gdf.to_geopandas()["x"]
    assert got.isna().all()
    assert str(got.dtype).startswith("datetime64")
    gdf = sgpd.from_geopandas(points)
    gdf["geometry"] = pd.NaT
    assert gdf._geometry_name == "geometry"
    assert gdf.to_geopandas()["geometry"].isna().all()
