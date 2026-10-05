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
"""Exploding multi-part geometries, compared against GeoPandas."""

import geopandas as gpd
import pytest

import sedonadb_geopandas as sgpd

# Single, empty, multi-part, nested, Z, and missing geometries: the cases where
# one-level parts, the engine's part count, and GeoPandas' row dropping differ.
WKT = [
    "POINT (1 1)",
    "POINT EMPTY",
    "LINESTRING EMPTY",
    "POLYGON EMPTY",
    "MULTIPOINT ((0 0), (1 1))",
    "MULTIPOINT EMPTY",
    "MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))",
    "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 0)), ((2 2, 3 2, 3 3, 2 2)))",
    "MULTIPOLYGON EMPTY",
    "GEOMETRYCOLLECTION (MULTIPOINT ((0 0), (1 1)), POINT (2 2))",
    "GEOMETRYCOLLECTION (POINT EMPTY, LINESTRING (0 0, 1 1))",
    "GEOMETRYCOLLECTION EMPTY",
    "MULTIPOINT Z ((0 0 1), (1 1 2))",
    None,
]


def _geometries(series):
    return list(zip(series.to_wkt(), series.has_z))


def test_explode_matches_geopandas():
    # One row per part, one level deep: a nested collection yields its
    # members whole. An empty single geometry is its own one part, while a
    # geometry with no parts, or a missing one, drops its row.
    expected_source = gpd.GeoDataFrame(
        {"id": range(len(WKT)), "label": [f"row{i}" for i in range(len(WKT))]},
        geometry=gpd.GeoSeries.from_wkt(WKT),
        crs=3857,
    )
    ours = sgpd.from_geopandas(expected_source).explode().to_geopandas()
    expected = expected_source.explode()
    assert list(ours.columns) == list(expected.columns)
    assert list(ours["id"]) == list(expected["id"])
    assert list(ours["label"]) == list(expected["label"])
    assert _geometries(ours.geometry) == _geometries(expected.geometry)
    assert ours.crs == expected.crs


def test_explode_moves_the_geometry_column_last():
    source = gpd.GeoDataFrame(
        {"a": [1], "b": [2]},
        geometry=gpd.GeoSeries.from_wkt(["MULTIPOINT ((0 0), (1 1))"]),
    )[["geometry", "a", "b"]].rename_geometry("pt")
    ours = sgpd.from_geopandas(source).explode()
    expected = source.explode()
    assert ours.columns == list(expected.columns) == ["a", "b", "pt"]
    assert ours._geometry_name == expected.geometry.name == "pt"
    assert len(ours.to_geopandas()) == 2


def test_explode_repeats_other_geometry_columns():
    source = gpd.GeoDataFrame(
        {"a": [1]}, geometry=gpd.GeoSeries.from_wkt(["MULTIPOINT ((0 0), (1 1))"])
    )
    source["other"] = gpd.GeoSeries.from_wkt(["POINT (5 5)"], crs=4326)
    ours = sgpd.from_geopandas(source).explode().to_geopandas()
    expected = source.explode()
    assert list(ours.columns) == list(expected.columns)
    assert _geometries(ours["other"]) == _geometries(expected["other"])
    assert ours["other"].crs == expected["other"].crs


def test_explode_keeps_geography():
    # A multi-geometry and a nested collection take different routes to their
    # parts; both keep the geography type and CRS.
    gdf = sgpd.GeoDataFrame(
        sgpd.default_context().sql(
            "SELECT id, ST_GeogFromWKT(w) AS g FROM (VALUES "
            "(1, 'MULTIPOINT ((0 0), (1 1))'), "
            "(2, 'GEOMETRYCOLLECTION (MULTIPOINT ((5 5), (6 6)), POINT (7 7))')"
            ") AS t(id, w)"
        ),
        geometry="g",
    )
    exploded = gdf.explode()
    assert "geography" in str(exploded._df.schema.field("g").type)
    assert exploded.crs == gdf.crs
    result = exploded.to_geopandas().sort_values("id", kind="stable")
    assert [g.wkt for g in result["g"]] == [
        "POINT (0 0)",
        "POINT (1 1)",
        "MULTIPOINT ((5 5), (6 6))",
        "POINT (7 7)",
    ]


def test_explode_of_an_empty_frame():
    source = gpd.GeoDataFrame(
        {"a": [1]}, geometry=gpd.GeoSeries.from_wkt(["MULTIPOINT ((0 0), (1 1))"])
    )
    gdf = sgpd.from_geopandas(source)
    exploded = gdf[gdf["a"] > 5].explode()
    assert exploded.columns == ["a", "geometry"]
    assert len(exploded.to_geopandas()) == 0


def test_explode_with_columns_named_like_its_working_columns():
    source = gpd.GeoDataFrame(
        {"__dump": ["a", "b"], "__member": [1, 2], "__collection": [True, False]},
        geometry=gpd.GeoSeries.from_wkt(
            [
                "MULTIPOINT ((0 0), (1 1))",
                "GEOMETRYCOLLECTION (MULTIPOINT ((0 0), (1 1)), POINT (2 2))",
            ]
        ),
    )
    ours = sgpd.from_geopandas(source).explode().to_geopandas()
    expected = source.explode()
    assert list(ours.columns) == list(expected.columns)
    assert list(ours["__dump"]) == list(expected["__dump"])
    assert _geometries(ours.geometry) == _geometries(expected.geometry)


def test_geoseries_explode_matches_geopandas():
    source = gpd.GeoSeries.from_wkt(WKT, crs=3857, name="geom")
    gdf = sgpd.from_geopandas(gpd.GeoDataFrame(source.to_frame(), geometry="geom"))
    ours = gdf.geometry.explode().to_geopandas()
    expected = source.explode()
    assert ours.name == expected.name == "geom"
    assert _geometries(ours) == _geometries(expected)
    assert ours.crs == expected.crs


def test_geoseries_explode_of_a_derived_series():
    source = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries.from_wkt(
            [
                "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 0)), ((2 2, 3 2, 3 3, 2 2)))",
                "MULTIPOINT ((0 0), (1 1))",
            ]
        )
    )
    ours = sgpd.from_geopandas(source).geometry.boundary.explode().to_geopandas()
    expected = source.geometry.boundary.explode()
    assert _geometries(ours) == _geometries(expected)


def test_geoseries_explode_belongs_to_a_new_frame():
    # The exploded column has more rows than the frame it came from.
    gdf = sgpd.from_geopandas(
        gpd.GeoDataFrame(geometry=gpd.GeoSeries.from_wkt(["MULTIPOINT ((0 0), (1 1))"]))
    )
    with pytest.raises(ValueError):
        gdf["parts"] = gdf.geometry.explode()


def test_explode_validation():
    source = gpd.GeoDataFrame(
        {"v": [1]}, geometry=gpd.GeoSeries.from_wkt(["MULTIPOINT ((0 0), (1 1))"])
    )
    source["other"] = gpd.GeoSeries.from_wkt(["POINT (5 5)"])
    gdf = sgpd.from_geopandas(source)
    # There is no index, so ignore_index changes nothing.
    assert len(gdf.explode(ignore_index=True).to_geopandas()) == 2
    with pytest.raises(NotImplementedError, match="index_parts"):
        gdf.explode(index_parts=True)
    with pytest.raises(NotImplementedError, match="index_parts"):
        gdf.geometry.explode(index_parts=True)
    with pytest.raises(NotImplementedError, match="active geometry"):
        gdf.explode(column="v")
    with pytest.raises(NotImplementedError, match="active geometry"):
        gdf.explode(column="other")
    with pytest.raises(KeyError):
        gdf.explode(column="missing")
    with pytest.raises(AttributeError, match="no active geometry"):
        gdf[["v"]].explode()
