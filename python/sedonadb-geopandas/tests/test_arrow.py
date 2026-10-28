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
"""Arrow and Feather interop, compared against GeoPandas."""

import json

import geoarrow.pyarrow as ga
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyproj
import pytest
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal

import sedonadb_geopandas as sgpd


def _source():
    """Attributes, a geometry column with a CRS (with Z, empty and missing
    geometries), and a second geometry column without one, all missing in
    its last row."""
    return gpd.GeoDataFrame(
        {
            "name": ["a", "b", None, "d"],
            "v": [1.5, 2.0, None, 4.0],
            "other": gpd.GeoSeries.from_wkt(
                ["LINESTRING (0 0, 1 1)", "POINT EMPTY", "POINT (5 5)", None]
            ),
        },
        geometry=gpd.GeoSeries.from_wkt(
            [
                "POINT (1 1)",
                "POLYGON ((0 0, 2 0, 2 2, 0 0))",
                "MULTIPOINT Z ((0 0 1), (5 5 2))",
                None,
            ]
        ),
        crs=3857,
    )


def _geography():
    return sgpd.GeoDataFrame(
        sgpd.default_context().sql(
            "SELECT 1 AS id, ST_GeogFromWKT('LINESTRING (0 0, 10 10)') AS g"
        ),
        geometry="g",
    )


def _extension(field):
    """A field's GeoArrow extension name and metadata."""
    return field.type.extension_name, json.loads(field.type.__arrow_ext_serialize__())


def _same_crs(a, b):
    if a is None or b is None:
        return a is b
    return pyproj.CRS.from_user_input(a) == pyproj.CRS.from_user_input(b)


def test_to_arrow_matches_geopandas():
    source = _source()
    ours = sgpd.from_geopandas(source).to_arrow()
    expected = pa.table(source.to_arrow())
    assert isinstance(ours, pa.Table)
    assert ours.column_names == expected.column_names
    for name in ("geometry", "other"):
        ours_name, ours_meta = _extension(ours.schema.field(name))
        name_expected, meta_expected = _extension(expected.schema.field(name))
        assert ours_name == name_expected == "geoarrow.wkb"
        assert _same_crs(ours_meta.get("crs"), meta_expected.get("crs"))
        # The same (ISO) WKB.
        assert ours[name].to_pylist() == expected[name].to_pylist()
    for name in ("name", "v"):
        assert ours[name].to_pylist() == expected[name].to_pylist()
    # GeoPandas reads it back as the frame it came from.
    assert_geodataframe_equal(
        gpd.GeoDataFrame.from_arrow(ours, geometry="geometry"), source
    )


def test_to_arrow_of_a_lazy_result_and_of_geography():
    source = _source()
    gdf = sgpd.from_geopandas(source)
    gdf = gdf[gdf["v"] > 1.5]
    ours = gdf.assign(geometry=gdf.geometry.centroid).to_arrow()
    expected = source[source["v"] > 1.5]
    expected = expected.assign(geometry=expected.geometry.centroid)
    assert_geodataframe_equal(
        gpd.GeoDataFrame.from_arrow(ours, geometry="geometry"),
        expected.reset_index(drop=True),
    )
    # Geography keeps its spherical edges.
    _, meta = _extension(_geography().to_arrow().schema.field("g"))
    assert meta["edges"] == "spherical"


def test_to_arrow_gives_no_view_types(tmp_path):
    # As GeoPandas gives none: SedonaDB reads a Parquet file's strings as
    # string views, which not every Arrow consumer takes.
    path = tmp_path / "file.parquet"
    _source().to_parquet(path)
    table = sgpd.read_parquet(path).to_arrow()
    assert table.schema.field("name").type == pa.string()
    assert table["name"].to_pylist() == ["a", "b", None, "d"]


def test_to_arrow_validation():
    gdf = sgpd.from_geopandas(_source())
    with pytest.raises(NotImplementedError, match="no index"):
        gdf.to_arrow(index=True)
    with pytest.raises(NotImplementedError, match="geometry_encoding='WKB' only"):
        gdf.to_arrow(geometry_encoding="geoarrow")
    with pytest.raises(ValueError, match="Expected geometry encoding 'WKB' or"):
        gdf.to_arrow(geometry_encoding="wkt")
    with pytest.raises(NotImplementedError, match="geometry_encoding='WKB' only"):
        gdf.geometry.to_arrow(geometry_encoding="geoarrow")
    with pytest.raises(ValueError, match="Expected geometry encoding 'WKB' or"):
        gdf.geometry.to_arrow(geometry_encoding="wkt")
    # Either case, and the GeoArrow-only options do nothing, as in GeoPandas.
    table = gdf.to_arrow(index=False, geometry_encoding="wkb", interleaved=False)
    assert table.column_names == gdf.columns


def test_geoseries_to_arrow_matches_geopandas():
    source = _source()
    ours = sgpd.from_geopandas(source).geometry.to_arrow()
    expected = pa.array(source.geometry.to_arrow())
    assert isinstance(ours, pa.Array)
    assert ours.type.extension_name == "geoarrow.wkb"
    assert ours.to_pylist() == expected.to_pylist()
    assert_geoseries_equal(gpd.GeoSeries.from_arrow(ours), source.geometry)


def _from_arrow_inputs():
    source = _source()
    native = source[["v"]].set_geometry(
        gpd.GeoSeries.from_wkt(["POINT (0 0)", "POINT (1 2)", None, "POINT (3 3)"]),
        crs=4326,
    )
    return {
        "WKB": source.to_arrow(),
        "native": native.to_arrow(geometry_encoding="geoarrow"),
    }


@pytest.mark.parametrize("encoding", ["WKB", "native"])
def test_from_arrow_matches_geopandas(encoding):
    table = _from_arrow_inputs()[encoding]
    ours = sgpd.GeoDataFrame.from_arrow(table)
    assert isinstance(ours, sgpd.GeoDataFrame)
    assert_geodataframe_equal(ours.to_geopandas(), gpd.GeoDataFrame.from_arrow(table))


def test_from_arrow_reads_wkt():
    # A GeoArrow encoding GeoPandas does not read.
    wkt = ga.with_crs(ga.as_wkt(["POINT (1 2)", None]), "EPSG:3857")
    ours = sgpd.GeoDataFrame.from_arrow(pa.table({"id": [1, 2], "geometry": wkt}))
    expected = gpd.GeoDataFrame(
        {"id": [1, 2]}, geometry=gpd.GeoSeries.from_wkt(["POINT (1 2)", None]), crs=3857
    )
    assert_geodataframe_equal(ours.to_geopandas(), expected)


def _unregistered(table):
    """`table` with its GeoArrow columns as their storage and field metadata,
    as a table made before the GeoArrow types are registered holds them."""
    fields, columns = [], []
    for field, column in zip(table.schema, table.columns):
        if isinstance(field.type, pa.ExtensionType):
            metadata = {
                "ARROW:extension:name": field.type.extension_name,
                "ARROW:extension:metadata": field.type.__arrow_ext_serialize__(),
            }
            column = pa.chunked_array(
                [chunk.storage for chunk in column.chunks], field.type.storage_type
            )
            field = pa.field(field.name, column.type, metadata=metadata)
        fields.append(field)
        columns.append(column)
    return pa.Table.from_arrays(columns, schema=pa.schema(fields))


@pytest.mark.parametrize("encoding", ["WKB", "native"])
def test_from_arrow_takes_geoarrow_metadata_without_registered_types(encoding):
    table = pa.table(_from_arrow_inputs()[encoding])
    ours = sgpd.GeoDataFrame.from_arrow(_unregistered(table))
    assert_geodataframe_equal(ours.to_geopandas(), gpd.GeoDataFrame.from_arrow(table))


def test_from_arrow_activates_the_first_geometry_column_as_geopandas():
    # Not the one named "geometry", as SedonaDB's own heuristic would.
    source = _source()[["other", "v", "geometry"]].set_geometry("other")
    table = pa.table(source.to_arrow())
    assert sgpd.GeoDataFrame.from_arrow(table).active_geometry_name == "other"
    assert gpd.GeoDataFrame.from_arrow(table).geometry.name == "other"
    chosen = sgpd.GeoDataFrame.from_arrow(table, geometry="geometry")
    assert chosen.active_geometry_name == "geometry"


def test_from_arrow_keeps_spherical_edges():
    # GeoPandas reads them as planar; the engine has geography for them.
    table = pa.table(
        {
            "g": ga.with_edge_type(
                ga.as_geoarrow(["LINESTRING (0 0, 10 10)"]), "spherical"
            )
        }
    )
    gdf = sgpd.GeoDataFrame.from_arrow(table)
    assert gdf.geometry._is_geography()


def test_from_arrow_takes_any_arrow_stream():
    source = _source()
    reader = pa.table(source.to_arrow()).to_reader()
    context = sgpd.default_context()
    for data in (source.to_arrow(), reader):
        ours = sgpd.GeoDataFrame.from_arrow(data, geometry="geometry", context=context)
        assert_geodataframe_equal(ours.to_geopandas(), source)


def test_from_arrow_validation():
    with pytest.raises(ValueError, match="No geometry column found"):
        sgpd.GeoDataFrame.from_arrow(pa.table({"a": [1]}))
    with pytest.raises(NotImplementedError, match="to_pandas_kwargs"):
        sgpd.GeoDataFrame.from_arrow(
            _source().to_arrow(), to_pandas_kwargs={"types_mapper": pd.ArrowDtype}
        )


@pytest.mark.parametrize("encoding", ["WKB", "geoarrow"])
def test_geoseries_from_arrow_matches_geopandas(encoding):
    source = gpd.GeoSeries.from_wkt(["POINT (1 2)", None, "POINT (3 4)"], crs=3857)
    array = source.to_arrow(geometry_encoding=encoding)
    ours = sgpd.GeoSeries.from_arrow(array)
    assert isinstance(ours, sgpd.GeoSeries)
    assert_geoseries_equal(ours.to_geopandas(), gpd.GeoSeries.from_arrow(array))
    named = sgpd.GeoSeries.from_arrow(array, name="pts")
    assert named.to_geopandas().name == "pts"


def test_geoseries_from_arrow_crs_as_geopandas():
    naive = gpd.GeoSeries.from_wkt(["POINT (1 2)"]).to_arrow()
    ours = sgpd.GeoSeries.from_arrow(naive, crs=4326)
    assert ours.to_geopandas().crs == gpd.GeoSeries.from_arrow(naive, crs=4326).crs
    labeled = gpd.GeoSeries.from_wkt(["POINT (1 2)"], crs=3857).to_arrow()
    with pytest.raises(ValueError, match="CRS"):
        gpd.GeoSeries.from_arrow(labeled, crs=4326)
    with pytest.raises(ValueError, match="CRS"):
        sgpd.GeoSeries.from_arrow(labeled, crs=4326)


def test_geoseries_from_arrow_validation():
    with pytest.raises(ValueError, match="No GeoArrow geometry field found"):
        sgpd.GeoSeries.from_arrow(pa.array([1, 2]))
    array = gpd.GeoSeries.from_wkt(["POINT (1 2)"]).to_arrow()
    with pytest.raises(NotImplementedError, match="index"):
        sgpd.GeoSeries.from_arrow(array, index=[0])


def test_arrow_round_trip():
    gdf = sgpd.from_geopandas(_source())
    back = sgpd.GeoDataFrame.from_arrow(gdf.to_arrow(), geometry="geometry")
    assert back.columns == gdf.columns
    assert_geodataframe_equal(back.to_geopandas(), gdf.to_geopandas())
    series = sgpd.GeoSeries.from_arrow(gdf.geometry.to_arrow())
    assert_geoseries_equal(series.to_geopandas(), gdf.geometry.to_geopandas())


def _geo(path):
    """A Feather file's "geo" metadata, but for who wrote it and its version
    (which depends on the version of GeoPandas)."""
    geo = json.loads(feather.read_table(path).schema.metadata[b"geo"])
    geo.pop("creator")
    geo.pop("version")
    return geo


def test_to_feather_matches_geopandas(tmp_path):
    source = _source()
    source.to_feather(tmp_path / "expected.feather")
    sgpd.from_geopandas(source).to_feather(tmp_path / "ours.feather")
    ours, expected = (
        _geo(tmp_path / "ours.feather"),
        _geo(tmp_path / "expected.feather"),
    )
    # The same geometry types, bounds and CRS.
    assert ours["primary_column"] == expected["primary_column"] == "geometry"
    assert ours["columns"].keys() == expected["columns"].keys()
    # The same PROJJSON, without the IDs of the datum ensemble's members.
    assert ours["columns"] == expected["columns"]
    members = ours["columns"]["geometry"]["crs"]["base_crs"]["datum_ensemble"]
    assert all("id" not in member for member in members["members"])
    assert_geodataframe_equal(gpd.read_feather(tmp_path / "ours.feather"), source)
    assert_geodataframe_equal(
        sgpd.read_feather(tmp_path / "expected.feather").to_geopandas(), source
    )


def test_to_feather_metadata_of_columns_without_bounds(tmp_path):
    source = gpd.GeoDataFrame(
        {"a": [1, 2]}, geometry=gpd.GeoSeries.from_wkt(["POINT EMPTY", None])
    )
    source.to_feather(tmp_path / "expected.feather")
    sgpd.from_geopandas(source).to_feather(tmp_path / "ours.feather")
    assert _geo(tmp_path / "ours.feather") == _geo(tmp_path / "expected.feather")


@pytest.mark.parametrize(
    "schema_version", [None, "1.0.0-beta.1", "1.0.0", "1.1.0", "2.0.0"]
)
def test_to_feather_schema_version(tmp_path, schema_version):
    # 1.1.0 by default, as the latest GeoPandas writes.
    gdf = sgpd.from_geopandas(_source())
    gdf.to_feather(tmp_path / "ours.feather", schema_version=schema_version)
    geo = json.loads(
        feather.read_table(tmp_path / "ours.feather").schema.metadata[b"geo"]
    )
    assert geo["version"] == (schema_version or "1.1.0")


def test_to_feather_geometry_types_as_geopandas(tmp_path):
    # Sorted, with " Z" for 3D ones; missing geometries have none.
    source = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries.from_wkt(
            [
                "POLYGON ((0 0, 1 0, 1 1, 0 0))",
                "POINT Z (1 2 3)",
                "MULTILINESTRING ((0 0, 1 1))",
                "GEOMETRYCOLLECTION (POINT (1 1))",
                "LINESTRING Z (0 0 0, 1 1 1)",
                "MULTIPOINT ((0 0))",
                "POINT (0 0)",
                "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 0)))",
                None,
            ]
        ),
        crs=3857,
    )
    source.to_feather(tmp_path / "expected.feather")
    sgpd.from_geopandas(source).to_feather(tmp_path / "ours.feather")
    ours = _geo(tmp_path / "ours.feather")["columns"]["geometry"]
    expected = _geo(tmp_path / "expected.feather")["columns"]["geometry"]
    assert ours["geometry_types"] == expected["geometry_types"]
    assert len(ours["geometry_types"]) == 8


def test_to_feather_compression_as_pyarrow(tmp_path):
    # LZ4 by default; repetitive data shrinks with any codec.
    source = gpd.GeoDataFrame(
        {"s": ["the same text"] * 1000},
        geometry=gpd.points_from_xy([0.0] * 1000, [0.0] * 1000),
        crs=3857,
    )
    gdf = sgpd.from_geopandas(source)
    sizes = {}
    for name, kwargs in (
        ("default", {}),
        ("lz4", {"compression": "lz4"}),
        ("zstd", {"compression": "zstd", "compression_level": 3}),
        ("uncompressed", {"compression": "uncompressed"}),
    ):
        path = tmp_path / f"{name}.feather"
        gdf.to_feather(path, **kwargs)
        sizes[name] = path.stat().st_size
        assert_geodataframe_equal(gpd.read_feather(path), source)
    assert sizes["default"] == sizes["lz4"] < sizes["uncompressed"]
    assert sizes["zstd"] < sizes["uncompressed"]
    # The level is used: varied data compresses further at a higher one.
    rng = np.random.default_rng(0)
    varied = sgpd.from_geopandas(
        gpd.GeoDataFrame(
            {"s": [str(value) for value in rng.integers(0, 50, 20000)]},
            geometry=gpd.points_from_xy(
                rng.integers(0, 50, 20000), rng.integers(0, 50, 20000)
            ),
            crs=3857,
        )
    )
    for level in (1, 19):
        varied.to_feather(
            tmp_path / f"level{level}.feather",
            compression="zstd",
            compression_level=level,
        )
    level1 = (tmp_path / "level1.feather").stat().st_size
    assert (tmp_path / "level19.feather").stat().st_size < level1
    with pytest.raises(ValueError):
        gdf.to_feather(tmp_path / "bad.feather", compression="snappy")


def test_to_feather_chunksize(tmp_path):
    gdf = sgpd.from_geopandas(_source())
    gdf.to_feather(tmp_path / "ours.feather", chunksize=1)
    with pa.ipc.open_file(tmp_path / "ours.feather") as reader:
        assert reader.num_record_batches == 4


def test_to_feather_of_geography(tmp_path):
    # Marked with spherical edges, as SedonaDB's GeoParquet writer does, and
    # without bounds, which planar ones would not be.
    gdf = _geography()
    gdf.to_feather(tmp_path / "ours.feather")
    column = _geo(tmp_path / "ours.feather")["columns"]["g"]
    assert column["edges"] == "spherical"
    assert "bbox" not in column
    assert sgpd.read_feather(tmp_path / "ours.feather").geometry._is_geography()


def test_to_feather_validation(tmp_path):
    gdf = sgpd.from_geopandas(_source())
    path = tmp_path / "ours.feather"
    with pytest.raises(NotImplementedError, match="no index"):
        gdf.to_feather(path, index=True)
    with pytest.raises(NotImplementedError, match="0.4.0"):
        gdf.to_feather(path, schema_version="0.4.0")
    with pytest.raises(ValueError, match="schema_version must be one of"):
        gdf.to_feather(path, schema_version="3.0.0")
    with pytest.raises(NotImplementedError, match="version 2"):
        gdf.to_feather(path, version=1)
    assert not path.exists()


def test_to_feather_expands_the_home_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    sgpd.from_geopandas(_source()).to_feather("~/ours.feather")
    assert (tmp_path / "ours.feather").exists()


def test_read_feather_columns_as_geopandas(tmp_path):
    source = _source()
    path = tmp_path / "expected.feather"
    source.to_feather(path)
    for columns in (["v", "geometry"], ["other", "name"]):
        ours = sgpd.read_feather(path, columns=columns)
        expected = gpd.read_feather(path, columns=columns)
        assert ours.active_geometry_name == expected.geometry.name
        assert_geodataframe_equal(ours.to_geopandas(), expected)


def test_read_feather_crs_as_geopandas(tmp_path):
    # A column without a CRS key is OGC:CRS84, and one with a null CRS has
    # none.
    path = tmp_path / "file.feather"
    table = pa.table(_source().to_arrow())
    geo = {
        "version": "1.0.0",
        "primary_column": "geometry",
        "columns": {
            "geometry": {"encoding": "WKB", "geometry_types": []},
            "other": {"encoding": "WKB", "geometry_types": [], "crs": None},
        },
    }
    plain = pa.schema([field.remove_metadata() for field in table.schema])
    table = pa.Table.from_arrays(
        [
            column.cast(pa.binary()) if field.name in geo["columns"] else column
            for field, column in zip(table.schema, table.columns)
        ],
        schema=plain.with_metadata({"geo": json.dumps(geo)}),
    )
    feather.write_feather(table, path)
    ours = sgpd.read_feather(path).to_geopandas()
    expected = gpd.read_feather(path)
    assert ours.crs == expected.crs == "OGC:CRS84"
    assert ours["other"].crs is None and expected["other"].crs is None
    assert_geodataframe_equal(ours, expected)


def test_read_feather_takes_the_crs_of_its_metadata_as_geopandas(tmp_path):
    # Not that of the extension type the column was written with.
    table = pa.table(_source()[["v", "geometry"]].to_arrow())
    geo = {
        "version": "1.0.0",
        "primary_column": "geometry",
        "columns": {"geometry": {"encoding": "WKB", "crs": "EPSG:4326"}},
    }
    path = tmp_path / "file.feather"
    feather.write_feather(table.replace_schema_metadata({"geo": json.dumps(geo)}), path)
    assert sgpd.read_feather(path).to_geopandas().crs == gpd.read_feather(path).crs
    assert gpd.read_feather(path).crs == "EPSG:4326"


def test_read_feather_promotes_a_non_primary_geometry_as_geopandas(tmp_path):
    source = _source().assign(third=gpd.GeoSeries.from_wkt(["POINT (9 9)"] * 4))
    path = tmp_path / "file.feather"
    source.to_feather(path)
    columns = ["third", "other", "v"]
    with pytest.warns(UserWarning, match="first column read was promoted"):
        expected = gpd.read_feather(path, columns=columns)
    with pytest.warns(UserWarning, match="first column read was promoted"):
        ours = sgpd.read_feather(path, columns=columns)
    assert ours.active_geometry_name == expected.geometry.name == "third"
    assert_geodataframe_equal(ours.to_geopandas(), expected)


def test_read_feather_of_a_native_encoding_as_geopandas(tmp_path):
    source = gpd.GeoDataFrame(
        {"v": [1, 2]}, geometry=gpd.points_from_xy([0, 1], [2, 3]), crs=3857
    )
    table = _unregistered(
        pa.table(source.to_arrow(geometry_encoding="geoarrow", interleaved=False))
    )
    table = table.select(["v", "geometry"])
    plain = pa.schema([field.remove_metadata() for field in table.schema])
    geo = {
        "version": "1.1.0",
        "primary_column": "geometry",
        "columns": {
            "geometry": {"encoding": "point", "crs": source.crs.to_json_dict()}
        },
    }
    path = tmp_path / "file.feather"
    feather.write_feather(
        pa.Table.from_arrays(
            table.columns, schema=plain.with_metadata({"geo": json.dumps(geo)})
        ),
        path,
    )
    assert_geodataframe_equal(
        sgpd.read_feather(path).to_geopandas(), gpd.read_feather(path)
    )
    assert_geodataframe_equal(gpd.read_feather(path), source)


def test_read_feather_validation(tmp_path):
    path = tmp_path / "plain.feather"
    feather.write_feather(pa.table({"a": [1]}), path)
    with pytest.raises(ValueError, match="Missing geo metadata"):
        sgpd.read_feather(path)
    _source().to_feather(tmp_path / "source.feather")
    for read in (gpd.read_feather, sgpd.read_feather):
        with pytest.raises(ValueError, match="No geometry columns are included"):
            read(tmp_path / "source.feather", columns=["v"])
    _source().to_feather(tmp_path / "file.feather")
    with pytest.raises(NotImplementedError, match="to_pandas_kwargs"):
        sgpd.read_feather(tmp_path / "file.feather", to_pandas_kwargs={})
