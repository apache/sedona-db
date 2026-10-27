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
"""Building geometries from coordinates, WKT, and WKB, and converting frames
back to WKT and WKB, compared against GeoPandas."""

import math

import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pytest
import shapely

import sedonadb_geopandas as sgpd

XY = {"x": [1.0, -2.5, 0.0], "y": [2.0, 3.5, 0.0], "z": [3.0, 0.0, -1.0]}


def _frame(**columns):
    """A frame of plain columns, without geometry."""
    return sgpd.from_geopandas(pd.DataFrame(columns))


def _geometries(series):
    return series.to_pandas().tolist()


def _listed(values):
    """The values as a list, with None for each missing one: pandas 3's
    string dtype, as of `to_wkb(hex=True)`, marks a missing value as NaN."""
    return [None if pd.isna(value) else value for value in values]


@pytest.mark.parametrize("constructor", ["points_from_xy", "from_xy"])
@pytest.mark.parametrize("with_z", [False, True])
def test_points_from_xy_matches_geopandas(constructor, with_z):
    frame = _frame(**XY)
    build = (
        sgpd.points_from_xy
        if constructor == "points_from_xy"
        else (sgpd.GeoSeries.from_xy)
    )
    z = frame["z"] if with_z else None
    got = build(frame["x"], frame["y"], z, crs="EPSG:4326")
    expected = gpd.GeoSeries.from_xy(
        XY["x"], XY["y"], XY["z"] if with_z else None, crs="EPSG:4326"
    )
    assert isinstance(got, sgpd.GeoSeries)
    assert got.to_geopandas().crs == expected.crs
    for value, reference in zip(_geometries(got), expected):
        assert value.equals_exact(reference, 0) and value.has_z == reference.has_z
    # Lazy, in the frame of its coordinates: it can be assigned there.
    frame["geometry"] = got
    assert frame.set_geometry("geometry").to_geopandas().geometry.tolist() == list(
        expected
    )


def test_points_from_xy_takes_numbers_and_columns_of_one_frame():
    frame = _frame(**XY)
    got = sgpd.points_from_xy(frame["x"], 0)
    assert [(p.x, p.y) for p in _geometries(got)] == [
        (1.0, 0.0),
        (-2.5, 0.0),
        (0.0, 0.0),
    ]
    with pytest.raises(ValueError, match="different DataFrames"):
        sgpd.points_from_xy(frame["x"], _frame(**XY)["y"])
    # Without a column there is no frame to build the points in.
    with pytest.raises(TypeError, match="Series"):
        sgpd.points_from_xy([1.0, 2.0], [3.0, 4.0])
    with pytest.raises(TypeError, match="numeric"):
        strings = _frame(x=XY["x"], s=["a", "b", "c"])
        sgpd.points_from_xy(strings["x"], strings["s"])
    with pytest.raises(NotImplementedError, match="index"):
        sgpd.GeoSeries.from_xy(frame["x"], frame["y"], index=[0, 1, 2])


def test_missing_coordinate_gives_a_missing_point():
    # GeoPandas reads a missing coordinate as NaN, giving POINT (NaN 2); a
    # NaN that is there is kept, as in GeoPandas.
    frame = sgpd.GeoDataFrame(
        sgpd.default_context().sql(
            "SELECT * FROM (VALUES (CAST(NULL AS DOUBLE), 2.0), "
            "(CAST('NaN' AS DOUBLE), 2.0), (1.0, 2.0)) AS t(x, y)"
        ),
        geometry=None,
    )
    first, nan, point = _geometries(sgpd.points_from_xy(frame["x"], frame["y"]))
    assert first is None
    assert math.isnan(nan.x) and nan.y == 2.0
    assert point.equals(shapely.Point(1, 2))
    assert gpd.GeoSeries.from_xy([None], [2.0]).to_wkt().tolist() == ["POINT (NaN 2)"]


WKT = [
    "POINT (1 2)",
    "POINT Z (1 2 3)",
    "LINESTRING M (0 0 1, 1 1 2)",
    "POLYGON ((0 0, 1 0, 1 1, 0 0))",
    "MULTIPOINT ((0 0), (1 1))",
    "GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (0 0, 1 1))",
    "POLYGON EMPTY",
    None,
]


def _same(got, expected):
    assert len(got) == len(expected)
    for value, reference in zip(got, expected):
        if reference is None:
            assert value is None
        else:
            assert value.equals_exact(reference, 0) or (
                value.is_empty and reference.is_empty
            ), (value, reference)
            assert (value.geom_type, value.has_z, value.has_m) == (
                reference.geom_type,
                reference.has_z,
                reference.has_m,
            )


def test_from_wkt_matches_geopandas():
    frame = _frame(wkt=WKT)
    got = sgpd.GeoSeries.from_wkt(frame["wkt"], crs="EPSG:3857")
    expected = gpd.GeoSeries.from_wkt(WKT, crs="EPSG:3857")
    assert got.to_geopandas().crs == expected.crs
    _same(_geometries(got), list(expected))


@pytest.mark.parametrize("encoding", ["bytes", "hex", "lowercase hex"])
def test_from_wkb_matches_geopandas(encoding):
    geometries = gpd.GeoSeries.from_wkt(WKT)
    wkb = _listed(geometries.to_wkb(hex=encoding != "bytes", flavor="iso"))
    if encoding == "lowercase hex":
        wkb = [None if value is None else value.lower() for value in wkb]
    frame = _frame(wkb=wkb)
    got = sgpd.GeoSeries.from_wkb(frame["wkb"], crs="EPSG:3857")
    expected = gpd.GeoSeries.from_wkb(wkb, crs="EPSG:3857")
    assert got.to_geopandas().crs == expected.crs
    _same(_geometries(got), list(expected))


def test_from_wkt_and_wkb_check_their_input():
    frame = _frame(wkt=["POINT (1"], n=[1.0])
    # Invalid text raises once computed (the frame is lazy).
    parsed = sgpd.GeoSeries.from_wkt(frame["wkt"])
    with pytest.raises(Exception, match="WKT"):
        parsed.to_pandas()
    with pytest.raises(NotImplementedError, match="on_invalid"):
        sgpd.GeoSeries.from_wkt(frame["wkt"], on_invalid="ignore")
    with pytest.raises(TypeError, match="string"):
        sgpd.GeoSeries.from_wkt(frame["n"])
    with pytest.raises(TypeError, match="binary"):
        sgpd.GeoSeries.from_wkb(frame["n"])
    with pytest.raises(TypeError, match="Series"):
        sgpd.GeoSeries.from_wkt(["POINT (1 2)"])


def _table_frame(**columns):
    """A frame of Arrow columns as given, without geometry."""
    table = pa.table(columns)
    return sgpd.GeoDataFrame(
        sgpd.default_context().create_data_frame(table), geometry=None
    )


@pytest.mark.parametrize("encode", ["dictionary", "run_end"])
@pytest.mark.parametrize("encoding", ["wkt", "bytes", "hex"])
def test_from_wkt_and_wkb_take_encoded_columns(encode, encoding):
    # Dictionary and run-end encoding change how values are stored, not what
    # they are.
    expected = gpd.GeoSeries.from_wkt(WKT)
    if encoding == "wkt":
        values = pa.array(WKT)
    else:
        wkb = expected.to_wkb(hex=encoding == "hex", flavor="iso")
        values = pa.array(_listed(wkb))
    if encode == "dictionary":
        encoded = values.dictionary_encode()
    else:
        ends = pa.array(range(1, len(values) + 1), pa.int32())
        encoded = pa.RunEndEncodedArray.from_arrays(ends, values)
    frame = _table_frame(data=encoded)
    build = sgpd.GeoSeries.from_wkt if encoding == "wkt" else sgpd.GeoSeries.from_wkb
    _same(_geometries(build(frame["data"])), list(expected))


def test_constructors_take_columns_of_only_missing_values():
    # A column of only None loads with Arrow's null type.
    frame = _table_frame(v=pa.nulls(2), x=pa.array([1.0, 2.0]))
    for got, expected in (
        (sgpd.GeoSeries.from_wkt(frame["v"]), gpd.GeoSeries.from_wkt([None, None])),
        (sgpd.GeoSeries.from_wkb(frame["v"]), gpd.GeoSeries.from_wkb([None, None])),
    ):
        assert _geometries(got) == expected.tolist() == [None, None]
    # A missing coordinate gives a missing point (see above).
    assert _geometries(sgpd.points_from_xy(frame["x"], frame["v"])) == [None, None]
    with_z = sgpd.points_from_xy(frame["x"], frame["x"], frame["v"])
    assert _geometries(with_z) == [None, None]


def _two_geometry_columns():
    source = gpd.GeoDataFrame(
        {
            "a": [1, 2, 3],
            "other": gpd.GeoSeries.from_wkt(
                ["LINESTRING (0 0, 1 1)", None, "POINT EMPTY"], crs="EPSG:3857"
            ),
        },
        geometry=gpd.GeoSeries.from_wkt(
            ["POINT (1 2)", "POLYGON ((0 0, 1 0, 1 1, 0 0))", None], crs="EPSG:3857"
        ),
    )
    return source, sgpd.from_geopandas(source)


def test_to_wkt_converts_every_geometry_column():
    source, gdf = _two_geometry_columns()
    got = gdf.to_wkt()
    assert isinstance(got, sgpd.GeoDataFrame)
    assert got.active_geometry_name is None
    assert got.columns == gdf.columns
    got = got.to_geopandas()
    expected = source.to_wkt()
    assert got["a"].tolist() == expected["a"].tolist()
    for name in ("geometry", "other"):
        parsed = [None if pd.isna(t) else shapely.from_wkt(t) for t in got[name]]
        _same(parsed, list(source[name]))


@pytest.mark.parametrize("hex", [False, True])
def test_to_wkb_converts_every_geometry_column(hex):
    source, gdf = _two_geometry_columns()
    got = gdf.to_wkb(hex=hex)
    assert got.active_geometry_name is None
    got = got.to_geopandas()
    expected = source.to_wkb(hex=hex, flavor="iso")
    for name in ("a", "geometry", "other"):
        assert _listed(got[name]) == _listed(expected[name]), name


def test_active_geometry_name():
    _, gdf = _two_geometry_columns()
    assert gdf.active_geometry_name == "geometry"
    assert gdf.set_geometry("other").active_geometry_name == "other"
    assert gdf.rename_geometry("shape").active_geometry_name == "shape"
    assert _frame(a=[1]).active_geometry_name is None


@pytest.mark.parametrize(
    "convert",
    [lambda f: f.to_wkt(), lambda f: f.to_wkb(), lambda f: f.to_wkb(hex=True)],
    ids=["to_wkt", "to_wkb", "to_wkb hex"],
)
def test_conversions_of_a_frame_without_columns(convert):
    source = gpd.GeoDataFrame()
    got = convert(sgpd.from_geopandas(source))
    assert got.active_geometry_name is None
    assert got.to_geopandas().shape == convert(source).shape == (0, 0)
