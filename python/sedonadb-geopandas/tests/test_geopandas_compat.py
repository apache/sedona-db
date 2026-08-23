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
def points():
    """A small frame in a projected CRS, shared across the tests."""
    return gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.GeoSeries.from_wkt(["POINT (0 0)", "POINT (5 5)", "POINT (9 9)"]),
        crs="EPSG:3857",
    )


@pytest.fixture
def parcels():
    """Overlapping polygons per zone, so a dissolve has something to union."""
    return gpd.GeoDataFrame(
        {"zone": ["A", "A", "B"], "v": [1, 2, 3]},
        geometry=gpd.GeoSeries.from_wkt(
            [
                "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
                "POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))",
                "POLYGON ((5 5, 7 5, 7 7, 5 7, 5 5))",
            ]
        ),
        crs="EPSG:3857",
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
    with pytest.raises(TypeError, match="isn't supported"):
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


# -- sjoin -------------------------------------------------------------------


def test_dissolve_matches_geopandas(parcels):
    got = (
        sgpd.from_geopandas(parcels)
        .dissolve(by="zone")
        .to_geopandas()
        .sort_values("zone")
    )
    expected = parcels.dissolve(by="zone")
    # Unioned geometry per group, and non-geometry columns aggregated as "first",
    # both matching GeoPandas.
    assert got.area.round(6).tolist() == expected.area.round(6).tolist()
    assert got["v"].tolist() == expected["v"].tolist()
    # The group key stays a column here rather than becoming the index.
    assert "zone" in got.columns


def test_dissolve_without_by(parcels):
    got = sgpd.from_geopandas(parcels).dissolve().to_geopandas()
    assert len(got) == len(parcels.dissolve()) == 1


def test_dissolve_multiple_keys(parcels):
    got = sgpd.from_geopandas(parcels).dissolve(by=["zone"]).to_geopandas()
    assert sorted(got["zone"]) == ["A", "B"]


def test_dissolve_validation(parcels):
    gdf = sgpd.from_geopandas(parcels)
    with pytest.raises(NotImplementedError, match="aggfunc"):
        gdf.dissolve(by="zone", aggfunc="sum")
    with pytest.raises(KeyError, match="not found"):
        gdf.dissolve(by="nope")
    with pytest.raises(ValueError, match="active geometry"):
        gdf[["zone", "v"]].dissolve(by="zone")


# -- __setitem__ and arithmetic ---------------------------------------------


def test_setitem_from_series(points):
    gdf = sgpd.from_geopandas(points)
    gdf["v2"] = gdf["v"] * 10 - 1
    got = gdf.to_geopandas().sort_values("name")
    assert got["v2"].tolist() == (points["v"] * 10 - 1).tolist()


def test_setitem_scalar_broadcasts(points):
    gdf = sgpd.from_geopandas(points)
    gdf["k"] = 7
    assert gdf.to_geopandas()["k"].tolist() == [7, 7, 7]


def test_setitem_geometry_column(points):
    gdf = sgpd.from_geopandas(points)
    gdf["buffered"] = gdf.geometry.buffer(0.5)
    assert "buffered" in gdf.columns
    # The active geometry column is unchanged by adding another one.
    assert gdf.geometry._name == "geometry"


def test_setitem_replaces_existing_column(points):
    gdf = sgpd.from_geopandas(points)
    gdf["v"] = gdf["v"] + 100
    assert sorted(gdf.to_geopandas()["v"]) == [101, 102, 103]
    assert gdf.columns.count("v") == 1


def test_setitem_makes_geometry_active_when_frame_had_none():
    # A frame that starts without geometry gains an active geometry column when
    # one is assigned.
    from shapely.geometry import Point

    plain = GeoDataFrame(sgpd.default_context().sql("SELECT 1 AS a"))
    assert plain._geometry_name is None

    plain["geometry"] = Point(1, 2)
    assert plain._geometry_name == "geometry"
    assert plain.to_geopandas().geometry.to_wkt().tolist() == ["POINT (1 2)"]


def test_setitem_non_geometry_leaves_frame_without_geometry():
    plain = GeoDataFrame(sgpd.default_context().sql("SELECT 1.0 AS x"))
    plain["doubled"] = plain["x"] * 2
    assert plain._geometry_name is None


def test_setitem_rejects_bad_inputs(points):
    gdf = sgpd.from_geopandas(points)
    with pytest.raises(TypeError, match="must be a string"):
        gdf[0] = 1
    with pytest.raises(TypeError, match="isn't supported"):
        gdf["x"] = points["v"]
    # A Series from a different frame has no row alignment.
    other = sgpd.from_geopandas(points)
    with pytest.raises(ValueError, match="different"):
        gdf["y"] = other["v"]


def test_setitem_stale_series_raises(points):
    # Assignment rebinds the frame, so a Series read beforehand is stale.
    gdf = sgpd.from_geopandas(points)
    before = gdf["v"]
    gdf["k"] = 1
    with pytest.raises(ValueError, match="different"):
        gdf["z"] = before


def test_series_arithmetic(points):
    gdf = sgpd.from_geopandas(points)
    for label, series, expected in [
        ("add", gdf["v"] + 1, (points["v"] + 1).tolist()),
        ("radd", 1 + gdf["v"], (1 + points["v"]).tolist()),
        ("sub", gdf["v"] - 1, (points["v"] - 1).tolist()),
        ("rsub", 10 - gdf["v"], (10 - points["v"]).tolist()),
        ("mul", gdf["v"] * 2, (points["v"] * 2).tolist()),
        ("truediv", gdf["v"] / 2, (points["v"] / 2).tolist()),
        ("neg", -gdf["v"], (-points["v"]).tolist()),
    ]:
        assert sorted(series.to_pandas().tolist()) == sorted(expected), label


def test_series_arithmetic_between_columns(points):
    gdf = sgpd.from_geopandas(points)
    got = (gdf["v"] + gdf["v"]).to_pandas().tolist()
    assert sorted(got) == sorted((points["v"] + points["v"]).tolist())


# -- regressions from review -------------------------------------------------


@pytest.mark.parametrize(
    "wkts",
    [
        ["POINT (0 0)", "POINT (1 1)"],
        ["LINESTRING (0 0, 1 1)", "LINESTRING (1 1, 2 2)"],
        ["POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))", "POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))"],
    ],
    ids=["points", "lines", "polygons"],
)
def test_dissolve_handles_every_geometry_type(wkts):
    # ST_Union_Agg only initializes for polygonal input, so points and linestrings
    # used to dissolve to NULL geometry.
    g = gpd.GeoDataFrame(
        {"zone": ["A", "A"]},
        geometry=gpd.GeoSeries.from_wkt(wkts),
        crs="EPSG:3857",
    )
    got = sgpd.from_geopandas(g).dissolve(by="zone").to_geopandas()
    expected = g.dissolve(by="zone")
    assert got.geometry.iloc[0] is not None
    assert got.geometry.iloc[0].geom_type == expected.geometry.iloc[0].geom_type
    assert got.geometry.iloc[0].equals(expected.geometry.iloc[0])


def test_dissolve_dropna_matches_geopandas():
    g = gpd.GeoDataFrame(
        {"zone": ["A", None], "v": [1, 2]},
        geometry=gpd.GeoSeries.from_wkt(
            [
                "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
                "POLYGON ((2 2, 3 2, 3 3, 2 3, 2 2))",
            ]
        ),
        crs="EPSG:3857",
    )
    gdf = sgpd.from_geopandas(g)
    # GeoPandas drops the missing key by default; dropna=False keeps it.
    assert len(gdf.dissolve(by="zone").to_geopandas()) == len(g.dissolve(by="zone"))
    assert len(gdf.dissolve(by="zone", dropna=False).to_geopandas()) == 2


def test_setitem_rejects_bare_expression(points):
    # A bare expression carries no origin, so one built from another frame would
    # resolve against this frame and silently write this frame's values.
    left = sgpd.from_geopandas(points)
    right = sgpd.from_geopandas(points)
    with pytest.raises(TypeError, match="bare expression"):
        left["copied"] = right["v"]._expr


def test_series_has_no_unguarded_expr_escape_hatch(points):
    assert not hasattr(sgpd.from_geopandas(points)["v"], "expr")


@pytest.mark.parametrize(
    "value",
    [[10, 20, 30], (10, 20, 30), {10, 20}],
    ids=["list", "tuple", "set"],
)
def test_setitem_rejects_sequences(points, value):
    # Sequences have no __array__, so they used to be broadcast whole into every
    # row rather than rejected.
    gdf = sgpd.from_geopandas(points)
    with pytest.raises(TypeError, match="isn't supported"):
        gdf["x"] = value


def test_setitem_accepts_numpy_scalar(points):
    # NumPy scalars do have __array__ but are single values, so they used to be
    # rejected as array-likes.
    import numpy as np

    gdf = sgpd.from_geopandas(points)
    gdf["x"] = np.int64(5)
    assert gdf.to_geopandas()["x"].tolist() == [5, 5, 5]


def test_setitem_rejects_numpy_array(points):
    import numpy as np

    gdf = sgpd.from_geopandas(points)
    with pytest.raises(TypeError, match="isn't supported"):
        gdf["x"] = np.array([1, 2, 3])


def test_operators_reject_bare_expression(points):
    # Same provenance hole as assignment: a bare expression built from another
    # frame resolved against this one and silently contributed this frame's values.
    gdf = sgpd.from_geopandas(points)
    other_raw = sgpd.default_context().create_data_frame(points)
    with pytest.raises(TypeError, match="bare expression"):
        gdf["v"] + other_raw["v"]


def test_operators_still_accept_literals(points):
    from sedonadb.expr import lit

    gdf = sgpd.from_geopandas(points)
    assert sorted((gdf["v"] > lit(1)).to_pandas().tolist()) == [False, True, True]


def test_getitem_rejects_stale_mask(points):
    # Assignment rebinds the frame, so a mask captured beforehand belongs to the
    # previous one. It used to be accepted and quietly resolve against the new
    # frame while the referenced column happened to still exist.
    gdf = sgpd.from_geopandas(points)
    mask = gdf["v"] > 1
    gdf["k"] = 1
    with pytest.raises(ValueError, match="different DataFrame"):
        gdf[mask]


def test_dissolve_drops_nan_group_keys():
    # dropna has to cover IEEE NaN as well as SQL null: a float key read from
    # pandas carries NaN, which grouping otherwise treats as its own group.
    df = sgpd.default_context().sql(
        "SELECT CAST('NaN' AS DOUBLE) k, ST_Point(0.0, 0.0) geometry "
        "UNION ALL SELECT 1.0, ST_Point(1.0, 1.0)"
    )
    assert len(GeoDataFrame(df).dissolve(by="k").to_geopandas()) == 1
    assert len(GeoDataFrame(df).dissolve(by="k", dropna=False).to_geopandas()) == 2


def test_setitem_none_keeps_geometry_column(points):
    # Assigning None used to turn the column untyped and clear the active geometry;
    # GeoPandas keeps a geometry column with its CRS.
    gdf = sgpd.from_geopandas(points)
    gdf["geometry"] = None
    assert gdf._geometry_name == "geometry"
    assert "3857" in str(gdf.crs)
    assert gdf.to_geopandas().geometry.isna().all()


def test_setitem_non_geometry_over_geometry_still_clears(points):
    # The CRS-preserving path must not dress a number up as geometry.
    gdf = sgpd.from_geopandas(points)
    gdf["geometry"] = 7
    assert gdf._geometry_name is None
    assert gdf.to_geopandas()["geometry"].tolist() == [7, 7, 7]


# -- regressions from the third review pass ---------------------------------


def test_dissolve_by_nested_float_column():
    # _is_floating used to match the rendered type string, so `list<item: double>`
    # looked like a float column and dropna called isnan() on it, failing at
    # planning time.
    df = sgpd.default_context().sql(
        "SELECT [1.0, 2.0] AS k, ST_Point(0.0, 0.0) AS geometry"
    )
    got = GeoDataFrame(df).dissolve(by="k").to_geopandas()
    assert len(got) == 1


@pytest.mark.parametrize(
    "value_name",
    ["none", "nan", "pandas_na", "geometry", "literal_geometry", "literal_none"],
)
def test_setitem_geometry_scalars_keep_type_and_crs(points, value_name):
    # Every supported scalar path has to go through the CRS-preserving branch:
    # GeoPandas treats None, NaN and pd.NA as missing geometry and keeps the typed
    # column and its CRS.
    import numpy as np
    import pandas as pd
    from shapely.geometry import Point

    from sedonadb.expr import lit

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


def test_dissolve_all_null_group_is_empty_geometry(points):
    # GeoPandas returns GEOMETRYCOLLECTION EMPTY rather than None, which behaves
    # differently for isna, is_empty, predicates and serialization.
    df = sgpd.default_context().sql(
        "SELECT 'A' AS z, "
        "ST_SetSRID(ST_GeomFromText(CAST(NULL AS VARCHAR)), 3857) AS geometry "
        "UNION ALL SELECT 'A', "
        "ST_SetSRID(ST_GeomFromText(CAST(NULL AS VARCHAR)), 3857)"
    )
    got = GeoDataFrame(df).dissolve(by="z").to_geopandas()
    expected = gpd.GeoDataFrame(
        {"z": ["A", "A"]},
        geometry=gpd.GeoSeries.from_wkt([None, None]),
        crs="EPSG:3857",
    ).dissolve(by="z")
    assert got.geometry.iloc[0].equals(expected.geometry.iloc[0])
    assert got.geometry.iloc[0].is_empty
    assert not got.geometry.isna().any()


def test_operators_accept_numpy_scalars(points):
    # Operators rejected anything with __array__, including NumPy scalars, even
    # though assignment already accepted them.
    import numpy as np

    gdf = sgpd.from_geopandas(points)
    assert sorted((gdf["v"] + np.int64(1)).to_pandas().tolist()) == [2, 3, 4]
    assert sorted((gdf["v"] > np.float64(1)).to_pandas().tolist()) == [
        False,
        True,
        True,
    ]


def test_operators_still_reject_arrays(points):
    import numpy as np

    gdf = sgpd.from_geopandas(points)
    with pytest.raises(TypeError, match="isn't supported"):
        gdf["v"] + np.array([1, 2, 3])


# -- regressions from the fourth review pass --------------------------------


def test_setitem_crs_carrying_literal_keeps_its_crs(points):
    # A literal that carries its own CRS must not be relabeled with the
    # destination column's CRS — that changes what the coordinates mean without
    # transforming them.
    from sedonadb.expr import lit

    src = gpd.GeoSeries.from_wkt(["POINT (10 10)"], crs="EPSG:4326")
    gdf = sgpd.from_geopandas(points)  # column is EPSG:3857
    gdf["geometry"] = lit(src)
    assert "4326" in str(gdf.crs)


def test_division_preserves_decimal_exactness():
    from decimal import Decimal

    import pandas as pd

    df = sgpd.default_context().create_data_frame(
        pd.DataFrame({"d": [Decimal("1.23")]})
    )
    got = (GeoDataFrame(df)["d"] / Decimal("0.1")).to_pandas().tolist()
    # Forced through double this would be 12.299999999999999.
    assert got == [Decimal("12.300000")]


def test_division_still_true_for_integers():
    df = sgpd.default_context().sql("SELECT 1 AS n UNION ALL SELECT 3")
    assert sorted((GeoDataFrame(df)["n"] / 2).to_pandas().tolist()) == [0.5, 1.5]


def test_numpy_array_plus_series_is_rejected_whole(points):
    # Without opting out of ufunc dispatch, NumPy broadcasts element-by-element
    # and returns an object array of lazy Series instead of an error.
    import numpy as np

    gdf = sgpd.from_geopandas(points)
    with pytest.raises(TypeError):
        np.array([10, 20, 30]) + gdf["v"]
    with pytest.raises(TypeError):
        gdf["v"] + np.array([10, 20, 30])


def test_zero_dimensional_array_normalizes(points):
    import numpy as np

    gdf = sgpd.from_geopandas(points)
    gdf["z"] = np.array(5)  # 0-d: scalar by classification, unwrapped on use
    assert gdf.to_geopandas()["z"].tolist() == [5, 5, 5]


def test_pandas_na_assigns_as_null_to_ordinary_column(points):
    import pandas as pd

    gdf = sgpd.from_geopandas(points)
    gdf["z"] = pd.NA
    assert gdf.to_geopandas()["z"].isna().all()


def test_is_floating_unwraps_dictionary_encoding():
    import pyarrow as pa

    from sedonadb_geopandas._frame import _is_floating

    tbl = pa.table(
        {
            "k": pa.DictionaryArray.from_arrays(
                pa.array([0, 1], type=pa.int8()), pa.array([1.0, float("nan")])
            ),
            "x": [1, 2],
        }
    )
    df = sgpd.default_context().create_data_frame(tbl)
    assert _is_floating(df, "k")


# -- regressions from the fifth review pass ---------------------------------


def test_division_unwraps_dictionary_int():
    # A dictionary<int64> column is integer for division purposes; it used to
    # skip the double cast and truncate.
    import pyarrow as pa

    tbl = pa.table(
        {
            "k": pa.DictionaryArray.from_arrays(
                pa.array([0, 1], type=pa.int8()), pa.array([1, 3], type=pa.int64())
            ),
            "x": [1, 2],
        }
    )
    gdf = GeoDataFrame(sgpd.default_context().create_data_frame(tbl))
    assert sorted((gdf["k"] / 2).to_pandas().tolist()) == [0.5, 1.5]


def test_division_by_decimal_stays_decimal():
    # An integer divided by a Decimal must stay in decimal arithmetic; the
    # unconditional double cast used to force a float result.
    from decimal import Decimal

    gdf = GeoDataFrame(sgpd.default_context().sql("SELECT 1 AS n"))
    got = (gdf["n"] / Decimal("0.5")).to_pandas().tolist()
    assert isinstance(got[0], Decimal)


def test_duration_arithmetic_matches_pandas():
    import pandas as pd

    gdf = GeoDataFrame(
        sgpd.default_context().create_data_frame(
            pd.DataFrame({"t": [pd.Timedelta(days=2)]})
        )
    )
    assert (gdf["t"] * 2).to_pandas().tolist() == [pd.Timedelta(days=4)]
    assert (2 * gdf["t"]).to_pandas().tolist() == [pd.Timedelta(days=4)]
    assert (gdf["t"] / 2).to_pandas().tolist() == [pd.Timedelta(days=1)]
    with pytest.raises(TypeError, match="numeric scalars"):
        gdf["t"] * "2"


def test_pyarrow_scalars_broadcast(points):
    # Arrow scalars implement __len__ but are single values.
    import pyarrow as pa

    from sedonadb_geopandas._series import is_scalar

    assert is_scalar(pa.scalar([1, 2]))
    assert is_scalar(pa.scalar({"a": 1}))
    gdf = sgpd.from_geopandas(points)
    gdf["tags"] = pa.scalar([1, 2])
    assert len(gdf.to_geopandas()["tags"]) == 3


def test_geoarrow_scalar_inherits_crs(points):
    # A GeoArrow WKB scalar has no __geo_interface__, so geometry-ness must come
    # from the resolved schema; it is CRS-less and inherits the column's CRS.
    import geoarrow.pyarrow as ga

    w = ga.as_wkb(ga.array(["POINT (5 5)"]))[0]
    gdf = sgpd.from_geopandas(points)
    gdf["geometry"] = w
    assert gdf._geometry_name == "geometry"
    assert "3857" in str(gdf.crs)


def test_explicitly_inactive_geometry_stays_inactive(points):
    # geometry=None is a choice; a no-op reassignment of an existing geometry
    # column must not silently reactivate it.
    df = sgpd.default_context().create_data_frame(points)
    gdf = GeoDataFrame(df, geometry=None)
    gdf["geometry"] = gdf["geometry"]
    assert gdf._geometry_name is None


def test_dissolve_rejects_empty_key_list(points):
    # Matches GeoPandas: an explicit empty iterable is an error, unlike by=None.
    gdf = sgpd.from_geopandas(points)
    with pytest.raises(ValueError, match="No group keys"):
        gdf.dissolve(by=[])
    assert len(gdf.dissolve().to_geopandas()) == 1


def test_dissolve_categorical_is_observed_only():
    # Documented divergence: unused categories do not produce empty groups (the
    # category domain does not survive a relational aggregation).
    import pandas as pd

    cat = gpd.GeoDataFrame(
        {"z": pd.Categorical(["A"], categories=["A", "B"])},
        geometry=gpd.GeoSeries.from_wkt(["POINT (0 0)"]),
        crs="EPSG:3857",
    )
    assert len(cat.dissolve(by="z")) == 2  # GeoPandas observed=False default
    assert len(sgpd.from_geopandas(cat).dissolve(by="z").to_geopandas()) == 1


# -- regressions from the sixth review pass ---------------------------------


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
def test_numpy_temporals_materialize_correctly(points, value_name, expected_kind):
    # Asserted end to end (assignment then collection), not on the normalization
    # helper: .item() flattening, unit rejection, and zeroed durations were all
    # invisible to helper-level equality checks.
    import numpy as np
    import pandas as pd

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


def test_masked_scalar_assigns_as_missing(points):
    import numpy as np

    gdf = sgpd.from_geopandas(points)
    gdf["m"] = np.ma.masked
    assert gdf.to_geopandas()["m"].isna().all()


def test_duration_division_is_exact_for_integral_divisors():
    # Routing an integral divisor through float64 rounded 2**53 + 1 ticks.
    import pandas as pd

    gdf = GeoDataFrame(
        sgpd.default_context().create_data_frame(
            pd.DataFrame({"t": [pd.Timedelta(2**53 + 1)]})
        )
    )
    assert (gdf["t"] / 1).to_pandas().tolist()[0].value == 2**53 + 1


def test_division_by_wrapped_integers_is_true_division():
    # lit(2) and pa.scalar(2) are integer operands just as 2 is.
    import pyarrow as pa

    from sedonadb.expr import lit

    gdf = GeoDataFrame(sgpd.default_context().sql("SELECT 1 AS n UNION ALL SELECT 3"))
    assert sorted((gdf["n"] / lit(2)).to_pandas().tolist()) == [0.5, 1.5]
    assert sorted((gdf["n"] / pa.scalar(2)).to_pandas().tolist()) == [0.5, 1.5]


def test_duration_arithmetic_accepts_wrapped_numbers():
    import pandas as pd
    import pyarrow as pa

    from sedonadb.expr import lit

    gdf = GeoDataFrame(
        sgpd.default_context().create_data_frame(
            pd.DataFrame({"t": [pd.Timedelta(days=2)]})
        )
    )
    assert (gdf["t"] * lit(2)).to_pandas().tolist() == [pd.Timedelta(days=4)]
    assert (gdf["t"] / pa.scalar(2)).to_pandas().tolist() == [pd.Timedelta(days=1)]


def test_duration_non_finite_results_are_nat():
    # Pandas yields NaT for these; the int64 cast-back used to fail on inf/NaN.
    import pandas as pd

    gdf = GeoDataFrame(
        sgpd.default_context().create_data_frame(
            pd.DataFrame({"t": [pd.Timedelta(days=2)]})
        )
    )
    assert (gdf["t"] / 0).to_pandas().isna().all()
    assert (gdf["t"] * float("nan")).to_pandas().isna().all()


# -- regressions from the seventh review pass -------------------------------


def test_coarse_unit_temporals_are_exact(points):
    # Forcing every temporal through nanoseconds silently wrapped values
    # outside the ns range (1677-2262): 2500-01-01 materialized as 1915-06-14.
    # Coarse units convert exactly to seconds instead. Comparisons are done as
    # numpy values because 200000 days exceeds the ns-backed scalar Timedelta.
    import numpy as np

    cases = [
        np.datetime64("2500-01-01", "D"),
        np.timedelta64(200000, "D"),
        np.datetime64("2500", "Y"),
    ]
    for value in cases:
        gdf = sgpd.from_geopandas(points)
        gdf["t"] = value
        got = gdf.to_geopandas()["t"].values[0]
        assert got == value.astype(got.dtype)


def test_ambiguous_and_subnano_temporal_units_are_rejected(points):
    # Matches pandas: timedelta months/years have no fixed length, and
    # sub-nanosecond units cannot be represented.
    import numpy as np

    gdf = sgpd.from_geopandas(points)
    with pytest.raises(TypeError, match="exactly"):
        gdf["t"] = np.timedelta64(1, "M")
    with pytest.raises(TypeError, match="exactly"):
        gdf["t"] = np.timedelta64(1, "ps")


def test_duration_division_by_infinity_is_zero_preserving_nulls():
    # Blanket NaT for non-finite operands regressed /inf, which pandas defines
    # as zero for valid rows while keeping source nulls null.
    import pandas as pd

    gdf = GeoDataFrame(
        sgpd.default_context().create_data_frame(
            pd.DataFrame({"t": [pd.Timedelta(days=2), pd.NaT]})
        )
    )
    got = (gdf["t"] / float("inf")).to_pandas()
    assert got.tolist()[0] == pd.Timedelta(0)
    assert pd.isna(got.tolist()[1])


def test_singleton_container_literals_resolve_as_numbers():
    # SedonaDB accepts one-element containers as single-value literals; the
    # numeric resolver must look through them like any other wrapper.
    import pandas as pd
    import pyarrow as pa

    from sedonadb.expr import lit

    gdf = GeoDataFrame(sgpd.default_context().sql("SELECT 1 AS n UNION ALL SELECT 3"))
    assert sorted((gdf["n"] / lit(pa.array([2]))).to_pandas().tolist()) == [0.5, 1.5]
    assert sorted((gdf["n"] / lit(pd.Series([2]))).to_pandas().tolist()) == [0.5, 1.5]
    tdf = GeoDataFrame(
        sgpd.default_context().create_data_frame(
            pd.DataFrame({"t": [pd.Timedelta(days=2)]})
        )
    )
    assert (tdf["t"] * lit(pd.Series([2]))).to_pandas().tolist() == [
        pd.Timedelta(days=4)
    ]
