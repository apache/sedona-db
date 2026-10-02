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
"""Frame bookkeeping (active geometry, CRS, columns, sorting) and Series
missing-value, membership, and casting methods, compared against GeoPandas."""

import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely

import sedonadb_geopandas as sgpd
from sedonadb_geopandas import GeoDataFrame


def _active(frame):
    """The active geometry column's name, for GeoPandas and wrapper frames."""
    if isinstance(frame, sgpd.GeoDataFrame):
        return frame._geometry_name
    try:
        return frame.geometry.name
    except AttributeError:
        return None


def _same_values(got, expected):
    assert len(got) == len(expected)
    for a, b in zip(got, expected):
        if isinstance(b, shapely.Geometry):
            assert a.geom_type == b.geom_type
            assert (a.is_empty and b.is_empty) or a.equals(b)
        elif pd.isna(b):
            assert pd.isna(a)
        else:
            assert a == b


def test_set_geometry_by_name_matches_geopandas():
    gdf = gpd.GeoDataFrame(
        {"other": gpd.GeoSeries.from_wkt(["POINT (9 9)"], crs=4326)},
        geometry=gpd.GeoSeries.from_wkt(["POINT (0 0)"], crs=3857),
    )
    ours = sgpd.from_geopandas(gdf).set_geometry("other")
    theirs = gdf.set_geometry("other")
    assert _active(ours) == _active(theirs) == "other"
    assert ours.to_geopandas().crs == theirs.crs
    # crs= relabels the new active column, overriding its own CRS.
    ours = sgpd.from_geopandas(gdf).set_geometry("other", crs=3857)
    assert ours.to_geopandas().crs == gdf.set_geometry("other", crs=3857).crs


def test_set_geometry_with_a_geoseries_adds_it_under_its_name():
    gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries.from_wkt(["POINT (0 0)"], crs=3857))
    ours = sgpd.from_geopandas(gdf)
    buffered = ours.geometry.buffer(1.0)
    buffered._name = "buf"
    result = ours.set_geometry(buffered)
    theirs = gdf.set_geometry(gdf.geometry.buffer(1.0).rename("buf"))
    assert _active(result) == _active(theirs) == "buf"
    assert result.columns == list(theirs.columns)


def test_set_geometry_inplace_and_errors():
    gdf = gpd.GeoDataFrame(
        {"v": [1.0], "other": gpd.GeoSeries.from_wkt(["POINT (9 9)"], crs=3857)},
        geometry=gpd.GeoSeries.from_wkt(["POINT (0 0)"], crs=3857),
    )
    ours = sgpd.from_geopandas(gdf)
    assert ours.set_geometry("other", inplace=True) is None
    assert _active(ours) == "other"
    with pytest.raises(ValueError, match="Unknown column"):
        ours.set_geometry("missing")
    with pytest.raises(TypeError, match="geometry"):
        ours.set_geometry("v")


def test_rename_geometry_matches_geopandas():
    gdf = gpd.GeoDataFrame(
        {"v": [1.0]}, geometry=gpd.GeoSeries.from_wkt(["POINT (0 0)"], crs=3857)
    )
    ours = sgpd.from_geopandas(gdf).rename_geometry("geom")
    theirs = gdf.rename_geometry("geom")
    assert _active(ours) == _active(theirs) == "geom"
    assert ours.columns == list(theirs.columns)
    assert ours.to_geopandas().crs == theirs.crs
    with pytest.raises(ValueError, match="already exists"):
        sgpd.from_geopandas(gdf).rename_geometry("v")


def test_rename_geometry_invalidates_earlier_reads():
    # The column an earlier Series referred to no longer exists by that name.
    gdf = gpd.GeoDataFrame(
        {"v": [1.0]}, geometry=gpd.GeoSeries.from_wkt(["POINT (0 0)"], crs=3857)
    )
    ours = sgpd.from_geopandas(gdf)
    before = ours.geometry
    ours.rename_geometry("geom", inplace=True)
    with pytest.raises(ValueError, match="different"):
        ours["copy"] = before


def test_frame_set_crs_matches_geopandas():
    gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries.from_wkt(["POINT (1 2)"], crs=3857))
    with pytest.raises(ValueError, match="allow_override"):
        sgpd.from_geopandas(gdf).set_crs(4326)
    ours = sgpd.from_geopandas(gdf).set_crs(4326, allow_override=True)
    theirs = gdf.set_crs(4326, allow_override=True)
    assert ours.to_geopandas().crs == theirs.crs
    # Relabeled, not transformed.
    assert ours.to_geopandas().geometry.tolist() == [shapely.Point(1, 2)]
    bare = gpd.GeoDataFrame(geometry=gpd.GeoSeries.from_wkt(["POINT (1 2)"]))
    assert (
        sgpd.from_geopandas(bare).set_crs(epsg=32633).to_geopandas().crs == "EPSG:32633"
    )
    target = sgpd.from_geopandas(bare)
    assert target.set_crs(32633, inplace=True) is target
    assert target.to_geopandas().crs == "EPSG:32633"


def test_drop_matches_geopandas():
    gdf = gpd.GeoDataFrame(
        {"v": [1.0], "s": ["a"]},
        geometry=gpd.GeoSeries.from_wkt(["POINT (0 0)"], crs=3857),
    )
    ours = sgpd.from_geopandas(gdf)
    for kwargs in (
        {"columns": "v"},
        {"columns": ["v", "s"]},
        {"labels": "v", "axis": 1},
    ):
        assert ours.drop(**kwargs).columns == list(gdf.drop(**kwargs).columns)
        assert _active(ours.drop(**kwargs)) == "geometry"
    # Dropping the active geometry leaves the frame without one.
    assert _active(ours.drop(columns="geometry")) is None
    with pytest.raises(KeyError):
        ours.drop(columns="nope")
    assert ours.drop(columns="nope", errors="ignore").columns == list(gdf.columns)
    with pytest.raises(NotImplementedError, match="index"):
        ours.drop(labels=[0])


def test_rename_matches_geopandas():
    gdf = gpd.GeoDataFrame(
        {"v": [1.0], "s": ["a"]},
        geometry=gpd.GeoSeries.from_wkt(["POINT (0 0)"], crs=3857),
    )
    ours = sgpd.from_geopandas(gdf)
    for kwargs in (
        {"columns": {"s": "t"}},
        {"columns": str.upper},
        {"mapper": {"v": "w"}, "axis": 1},
    ):
        renamed = ours.rename(**kwargs)
        assert renamed.columns == list(gdf.rename(**kwargs).columns)
    assert _active(ours.rename(columns={"s": "t"})) == "geometry"
    # As in GeoPandas, renaming the active geometry this way deactivates it.
    assert _active(ours.rename(columns={"geometry": "geom"})) is None
    assert _active(gdf.rename(columns={"geometry": "geom"})) is None
    assert ours.rename(columns={"nope": "x"}).columns == list(gdf.columns)
    with pytest.raises(KeyError):
        ours.rename(columns={"nope": "x"}, errors="raise")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"by": "v"},
        {"by": "v", "ascending": False},
        {"by": "v", "na_position": "first"},
        {"by": "v", "ascending": False, "na_position": "first"},
        {"by": ["s", "v"], "ascending": [False, True]},
    ],
    ids=["asc", "desc", "nan-first", "desc-nan-first", "multi"],
)
def test_sort_values_matches_geopandas(kwargs):
    # NaN in a float column sorts as missing, which the engine alone would
    # order above every number.
    gdf = gpd.GeoDataFrame(
        {"v": [3.0, np.nan, 1.0, 2.0], "s": ["b", "a", None, "b"]},
        geometry=gpd.GeoSeries.from_wkt(
            ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)"], crs=3857
        ),
    )
    ours = sgpd.from_geopandas(gdf).sort_values(**kwargs).to_geopandas()
    theirs = gdf.sort_values(**kwargs)
    _same_values(ours["v"].tolist(), theirs["v"].tolist())
    _same_values(ours["s"].tolist(), theirs["s"].tolist())


def test_sort_values_errors():
    gdf = gpd.GeoDataFrame(
        {"v": [1.0]}, geometry=gpd.GeoSeries.from_wkt(["POINT (0 0)"])
    )
    ours = sgpd.from_geopandas(gdf)
    with pytest.raises(KeyError):
        ours.sort_values("nope")
    with pytest.raises(ValueError, match="ascending"):
        ours.sort_values(["v"], ascending=[True, False])
    with pytest.raises(ValueError, match="na_position"):
        ours.sort_values("v", na_position="middle")


def _series_case():
    gdf = gpd.GeoDataFrame(
        {"v": [3.0, np.nan, 1.0], "s": ["a", None, "c"], "n": [1, 2, 3]},
        geometry=gpd.GeoSeries.from_wkt(["POINT (0 0)", "POINT EMPTY", None], crs=3857),
    )
    return gdf, sgpd.from_geopandas(gdf)


@pytest.mark.parametrize("column", ["v", "s", "n", "geometry"])
@pytest.mark.parametrize("method", ["isna", "isnull", "notna", "notnull"])
def test_missing_value_detection_matches_geopandas(method, column):
    # NaN in a float column is missing; an empty geometry is not.
    gdf, ours = _series_case()
    got = getattr(ours[column], method)().to_pandas().tolist()
    with warnings.catch_warnings():
        # GeoPandas' own GeoSeries.notna() warns about a past behavior change.
        warnings.simplefilter("ignore", UserWarning)
        expected = getattr(gdf[column], method)().tolist()
    assert got == expected


@pytest.mark.parametrize("column,value", [("v", 0.0), ("s", "x"), ("n", 9)])
def test_fillna_matches_geopandas(column, value):
    gdf, ours = _series_case()
    _same_values(
        ours[column].fillna(value).to_pandas().tolist(),
        gdf[column].fillna(value).tolist(),
    )


def test_fillna_requires_a_value():
    _, ours = _series_case()
    with pytest.raises(ValueError, match="value"):
        ours["v"].fillna(None)


@pytest.mark.parametrize("value", [None, shapely.Point(5, 5)], ids=["default", "point"])
def test_geoseries_fillna_matches_geopandas(value):
    # The default is an empty geometry collection; empties are not missing.
    gdf, ours = _series_case()
    filled = ours.geometry.fillna(value)
    assert isinstance(filled, sgpd.GeoSeries)
    result = filled.to_geopandas()
    _same_values(result.tolist(), gdf.geometry.fillna(value).tolist())
    assert result.crs == gdf.crs


def test_geoseries_fillna_keeps_geography():
    gdf = GeoDataFrame(
        sgpd.default_context().sql(
            "SELECT ST_GeogFromWKT('POINT (1 1)') AS g "
            "UNION ALL SELECT ST_GeogFromWKT(CAST(NULL AS VARCHAR))"
        ),
        geometry="g",
    )
    filled = gdf.geometry.fillna(shapely.Point(0, 0))
    assert "geography" in str(
        gdf._df.select(filled._expr.alias("x")).schema.field("x").type
    )
    assert sorted(g.wkt for g in filled.to_pandas().tolist()) == [
        "POINT (0 0)",
        "POINT (1 1)",
    ]


@pytest.mark.parametrize(
    "column,values",
    [
        ("v", [3.0, np.nan]),
        ("v", [3.0]),
        ("s", ["a", None]),
        ("s", ["c"]),
        ("n", [2, 3]),
        ("n", np.array([1])),
        ("v", []),
    ],
    ids=["nan-member", "float", "none-member", "string", "int", "ndarray", "empty"],
)
def test_isin_matches_geopandas(column, values):
    # A missing value is a member only when the list contains a missing
    # marker; the engine alone answers null for it.
    gdf, ours = _series_case()
    assert (
        ours[column].isin(values).to_pandas().tolist()
        == gdf[column].isin(values).tolist()
    )


def test_isin_rejects_non_list_like():
    _, ours = _series_case()
    with pytest.raises(TypeError, match="list-like"):
        ours["s"].isin("a")


@pytest.mark.parametrize(
    "column,dtype",
    [
        ("n", "float64"),
        ("n", float),
        ("n", "int32"),
        ("n", str),
        ("n", bool),
        ("v", "float32"),
    ],
)
def test_astype_matches_geopandas(column, dtype):
    gdf, ours = _series_case()
    got = ours[column].astype(dtype).to_pandas()
    expected = gdf[column].astype(dtype)
    _same_values(got.tolist(), expected.tolist())
    if dtype not in (str,):
        assert str(got.dtype) == str(expected.dtype)


def test_astype_to_integer_raises_for_missing_values():
    # The engine casts a null to a null silently; pandas raises.
    gdf, ours = _series_case()
    with pytest.raises(Exception):
        gdf["v"].astype("int64")
    with pytest.raises(Exception, match="non-finite"):
        ours["v"].astype("int64").to_pandas()
    assert ours["v"].fillna(0).astype("int64").to_pandas().tolist() == [3, 0, 1]
    with pytest.raises(TypeError, match="dtype"):
        ours["v"].astype("category")
