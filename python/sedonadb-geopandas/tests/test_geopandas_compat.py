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


# -- column assignment -------------------------------------------------------


def test_setitem_from_series(points):
    gdf = sgpd.from_geopandas(points)
    gdf["v2"] = gdf["v"]
    got = gdf.to_geopandas().sort_values("name")
    assert got["v2"].tolist() == points["v"].tolist()


def test_setitem_replaces_existing_column(points):
    gdf = sgpd.from_geopandas(points)
    gdf["v"] = gdf["name"]
    assert sorted(gdf.to_geopandas()["v"]) == ["A", "B", "C"]
    assert gdf.columns.count("v") == 1


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
    plain["doubled"] = plain["x"]
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


def test_getitem_rejects_stale_mask(points):
    # Assignment rebinds the frame, so a mask captured beforehand belongs to the
    # previous one. It used to be accepted and quietly resolve against the new
    # frame while the referenced column happened to still exist.
    gdf = sgpd.from_geopandas(points)
    mask = gdf["v"] > 1
    gdf["k"] = 1
    with pytest.raises(ValueError, match="different DataFrame"):
        gdf[mask]


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


def test_setitem_crs_carrying_literal_keeps_its_crs(points):
    # A literal that carries its own CRS must not be relabeled with the
    # destination column's CRS — that changes what the coordinates mean without
    # transforming them.
    from sedonadb.expr import lit

    src = gpd.GeoSeries.from_wkt(["POINT (10 10)"], crs="EPSG:4326")
    gdf = sgpd.from_geopandas(points)  # column is EPSG:3857
    gdf["geometry"] = lit(src)
    assert "4326" in str(gdf.crs)


def test_pandas_na_assigns_as_null_to_ordinary_column(points):
    import pandas as pd

    gdf = sgpd.from_geopandas(points)
    gdf["z"] = pd.NA
    assert gdf.to_geopandas()["z"].isna().all()


def test_zero_dimensional_array_normalizes(points):
    import numpy as np

    gdf = sgpd.from_geopandas(points)
    gdf["z"] = np.array(5)  # 0-d: scalar by classification, unwrapped on use
    assert gdf.to_geopandas()["z"].tolist() == [5, 5, 5]


def test_masked_scalar_assigns_as_missing(points):
    import numpy as np

    gdf = sgpd.from_geopandas(points)
    gdf["m"] = np.ma.masked
    assert gdf.to_geopandas()["m"].isna().all()


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


def test_temporal_scalars_are_deferred(points):
    # Representing temporal scalars faithfully needs dedicated unit and
    # timezone handling — a naive literal would silently truncate nanoseconds
    # or reject most NumPy units — which arrives as its own change; until
    # then they are rejected outright rather than stored subtly wrong.
    import numpy as np
    import pandas as pd

    gdf = sgpd.from_geopandas(points)
    for value in (
        np.datetime64("2026-01-01", "ns"),
        np.timedelta64(1, "ns"),
        pd.Timestamp("2026-01-01"),
        pd.Timedelta(1),
    ):
        with pytest.raises(TypeError, match="not supported yet"):
            gdf["t"] = value


# -- regressions from review ------------------------------------------------


def test_assigned_geometry_column_reads_back_as_geoseries(points):
    # Only the active geometry name produced a GeoSeries, so the advertised
    # gdf["buffered"] = gdf.geometry.buffer(...) gave back a plain Series
    # with no .area or .buffer(). Geometry-ness comes from the schema.
    gdf = sgpd.from_geopandas(points)
    gdf["buffered"] = gdf.geometry.buffer(0.5)
    col = gdf["buffered"]
    assert isinstance(col, type(gdf.geometry))
    assert (col.area.to_pandas() > 0).all()


def test_cleared_geometry_is_not_resurrected_by_materialization(points):
    # With the active geometry replaced by a number, the wrapper records no
    # active geometry — but the materializer heuristically activated any
    # remaining geometry column, so a later to_crs() on the result silently
    # targeted a column this frame never had active.
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
    from shapely.geometry import Point

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


def test_numpy_scalar_dtypes_are_preserved(points):
    # .item() promoted np.int8/np.float32 to int64/float64 columns and
    # overflowed np.uint64 past int64, which the engine supports natively.
    import numpy as np

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


def test_arrow_wrapped_missing_values_keep_geometry(points):
    # pa.scalar(None), typed Arrow nulls, and Arrow-wrapped NaN cleared the
    # active geometry and CRS, unlike the equivalent bare None/NaN; a wrapped
    # value means whatever its payload means.
    import pyarrow as pa

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


def test_typed_null_nested_scalars_broadcast(points):
    # Valid nested Arrow scalars broadcast, but their typed-null forms failed
    # literal construction; they re-enter as one-element typed arrays.
    import pyarrow as pa

    for value in (
        pa.scalar(None, pa.list_(pa.int64())),
        pa.scalar(None, pa.map_(pa.string(), pa.int64())),
    ):
        gdf = sgpd.from_geopandas(points)
        gdf["x"] = value
        assert gdf.to_geopandas()["x"].isna().all()


def test_nat_is_rejected_like_other_temporals(points):
    # pd.NaT is an instance of neither Timestamp nor Timedelta, so it slipped
    # past the temporal deferral: ordinary assignment failed with a backend
    # error while geometry assignment absorbed it as missing.
    import pandas as pd

    gdf = sgpd.from_geopandas(points)
    with pytest.raises(TypeError, match="not supported yet"):
        gdf["x"] = pd.NaT
    with pytest.raises(TypeError, match="not supported yet"):
        gdf["geometry"] = pd.NaT


def test_multipart_geometries_classify_as_scalars(points):
    # Shapely 1.x multipart geometries implement __len__, so the generic
    # sequence check rejected them; any BaseGeometry is a single value.
    from shapely.geometry import MultiPoint
    from shapely.geometry.base import BaseGeometry

    from sedonadb_geopandas._series import is_scalar

    value = MultiPoint([(0, 0), (1, 1)])
    assert is_scalar(value)
    import warnings

    legacy = type("LegacyMultipart", (BaseGeometry,), {"__len__": lambda self: 2})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # instantiating BaseGeometry warns
        instance = legacy.__new__(legacy)
    assert is_scalar(instance)
    gdf = sgpd.from_geopandas(points)
    gdf["mp"] = value
    assert gdf.to_geopandas()["mp"].tolist() == [value] * len(points)


def test_no_active_geometry_survives_any_column_name(points):
    # The materializer rebuild went through the GeoDataFrame constructor,
    # which auto-activates a geometry column literally named "geometry" — so
    # clearing the active `geom` resurrected the secondary column and a later
    # to_crs() would transform it. The no-active marker is preserved
    # explicitly, whatever the remaining columns are called.
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


def test_numpy_void_scalars_broadcast_as_binary(points):
    # np.void has no Arrow scalar form, so the typed-scalar conversion broke
    # what previously worked: void values broadcast through .item() as bytes.
    import numpy as np

    gdf = sgpd.from_geopandas(points)
    gdf["x"] = np.void(b"abcd")
    assert gdf.to_geopandas()["x"].tolist() == [b"abcd"] * len(points)
    gdf["y"] = np.array(np.void(b"ab"))
    assert gdf.to_geopandas()["y"].tolist() == [b"ab"] * len(points)


def test_typed_null_nested_scalar_keeps_geometry(points):
    # Normalization rewrites a typed-null nested scalar into its one-element
    # array spelling before missingness is judged; recognizing that spelling
    # only worked through pandas coincidence, and without pandas the null
    # converted the geometry column to a list column. Classification is
    # explicit now: one null element is one missing value.
    import pyarrow as pa

    gdf = sgpd.from_geopandas(points)
    gdf["geometry"] = pa.scalar(None, pa.list_(pa.int64()))
    assert gdf._geometry_name == "geometry"
    assert "3857" in str(gdf.crs)
    assert gdf.to_geopandas()["geometry"].isna().all()


def test_crs_less_geography_replacement_stays_crs_less():
    # A geography constructor synthesizes CRS84, so replacing a CRS-less
    # geography with None or a Shapely value silently gained a CRS; the
    # synthesized one is stripped back off when the destination has none.
    from shapely.geometry import Point

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


def test_typed_spatial_null_keeps_its_own_crs(points):
    # An invalid GeoArrow scalar typed EPSG:4267 was routed through the
    # missing path, which synthesized a destination-kind null stamped with
    # the destination CRS — while the equivalent valid scalar kept 4267. A
    # typed spatial null is a geometry value that happens to be null: it
    # keeps its own kind and CRS metadata.
    import geopandas as gpd
    import geoarrow.pyarrow as ga
    import pyarrow as pa

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


def test_spherical_geoarrow_scalars_keep_geography(points):
    # The scalar literal resolver drops the edge type, so both a null and a
    # valid spherical WKB scalar silently became planar geometry; the
    # one-element-array spelling resolves with the complete extension
    # metadata.
    import geoarrow.pyarrow as ga
    import geopandas as gpd
    import pyarrow as pa
    from shapely.geometry import Point

    crs4267 = gpd.GeoSeries.from_wkt(["POINT (0 0)"], crs="EPSG:4267").crs
    sph = ga.wkb().with_edge_type(ga.EdgeType.SPHERICAL).with_crs(crs4267.to_json())
    for payload in (None, Point(1, 1).wkb):
        gdf = sgpd.from_geopandas(points)
        gdf["geometry"] = pa.scalar(payload, sph)
        dtype = str(gdf._df.schema.field("geometry").type)
        assert "geography" in dtype
        assert "4267" in dtype


def test_non_wkb_geoarrow_nulls_do_not_crash(points):
    # Null WKT/point GeoArrow scalars raised ValueError from the literal
    # resolver inside the spatial-null detector. They now either resolve
    # through the array spelling or degrade to the destination-kind null —
    # never an error, and always still geometry.
    import geoarrow.pyarrow as ga
    import pyarrow as pa

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


def test_structured_numpy_scalars_keep_fields_and_dtypes(points):
    # A structured np.void flattened to a tuple, silently storing
    # [("count", int16), ("ratio", float32)] as list<float64>; it broadcasts
    # as a typed Arrow struct with field names and dtypes intact.
    import numpy as np

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
    import sys

    import pyarrow as pa

    from sedonadb_geopandas._frame import _is_missing

    null_array = pa.array([None], type=pa.list_(pa.int64()))
    value_array = pa.array([1], type=pa.int64())
    monkeypatch.setitem(sys.modules, "pandas", None)
    assert _is_missing(null_array)
    assert not _is_missing(value_array)


def test_masked_structured_records(points):
    # A structured scalar drawn from a MaskedArray failed inside
    # np.ma.is_masked before any conversion ran — even fully unmasked.
    # Unmasked and partially masked records broadcast as typed structs with
    # masked fields null; a fully masked record is missing.
    import numpy as np

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
