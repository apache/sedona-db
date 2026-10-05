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
"""GeoParquet and GDAL/OGR I/O, compared against GeoPandas."""

import json

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import pytest

import sedonadb_geopandas as sgpd


def _sample():
    return gpd.GeoDataFrame(
        {"name": ["a", "b", "c", "d"], "v": [1.5, 2.0, None, 4.0]},
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


def _values(series):
    return [None if pd.isna(value) else value for value in series]


def _assert_same_frame(ours, expected):
    assert list(ours.columns) == list(expected.columns)
    assert ours.geometry.name == expected.geometry.name
    assert ours.crs == expected.crs
    for name in expected.columns:
        if name == expected.geometry.name:
            assert list(ours[name].to_wkt()) == list(expected[name].to_wkt())
            assert list(ours[name].has_z) == list(expected[name].has_z)
        else:
            assert _values(ours[name]) == _values(expected[name])


def _codec(path):
    return pq.ParquetFile(path).metadata.row_group(0).column(0).compression


def _geo_metadata(path):
    return json.loads(pq.read_schema(path).metadata[b"geo"])


def test_read_parquet_matches_geopandas(tmp_path):
    path = tmp_path / "sample.parquet"
    _sample().to_parquet(path)
    _assert_same_frame(sgpd.read_parquet(path).to_geopandas(), gpd.read_parquet(path))


def test_read_parquet_columns(tmp_path):
    path = tmp_path / "sample.parquet"
    _sample().to_parquet(path)
    ours = sgpd.read_parquet(path, columns=["v", "geometry", "name"])
    expected = gpd.read_parquet(path, columns=["v", "geometry", "name"])
    assert ours.columns == list(expected.columns) == ["v", "geometry", "name"]
    with pytest.raises(ValueError, match="no geometry column"):
        sgpd.read_parquet(path, columns=["v"])
    with pytest.raises(KeyError, match="missing"):
        sgpd.read_parquet(path, columns=["missing", "geometry"])


def test_read_parquet_of_a_directory(tmp_path):
    sample = _sample()
    (tmp_path / "parts").mkdir()
    sample.iloc[:2].to_parquet(tmp_path / "parts" / "a.parquet")
    sample.iloc[2:].to_parquet(tmp_path / "parts" / "b.parquet")
    ours = sgpd.read_parquet(tmp_path / "parts").to_geopandas()
    assert sorted(ours["name"]) == ["a", "b", "c", "d"]
    assert ours.crs == sample.crs


def test_read_parquet_rejects_unsupported_arguments(tmp_path):
    path = tmp_path / "sample.parquet"
    _sample().to_parquet(path)
    with pytest.raises(NotImplementedError, match="bbox"):
        sgpd.read_parquet(path, bbox=(0, 0, 1, 1))
    with pytest.raises(NotImplementedError, match="storage_options"):
        sgpd.read_parquet(path, storage_options={"anon": True})
    with pytest.raises(NotImplementedError, match="filters"):
        sgpd.read_parquet(path, filters=[("v", ">", 1)])


def test_to_parquet_round_trips_through_geopandas(tmp_path):
    path = tmp_path / "out.parquet"
    sample = _sample()[["geometry", "v", "name"]]
    sgpd.from_geopandas(sample).to_parquet(path)
    _assert_same_frame(gpd.read_parquet(path), sample)
    _assert_same_frame(sgpd.read_parquet(path).to_geopandas(), sample)
    # An existing file is replaced, as in GeoPandas.
    sgpd.from_geopandas(sample.iloc[:1]).to_parquet(path)
    assert len(gpd.read_parquet(path)) == 1


def test_to_parquet_of_a_lazy_result(tmp_path):
    path = tmp_path / "out.parquet"
    gdf = sgpd.from_geopandas(_sample())
    gdf["area"] = gdf.geometry.area
    gdf[gdf["name"] != "a"].to_parquet(path)
    written = gpd.read_parquet(path)
    assert list(written["name"]) == ["b", "c", "d"]
    assert _values(written["area"]) == [2.0, 0.0, None]


@pytest.mark.parametrize(
    "compression", ["snappy", "gzip", "brotli", "zstd", "lz4", None]
)
def test_to_parquet_compression_matches_geopandas(tmp_path, compression):
    ours, expected = tmp_path / "ours.parquet", tmp_path / "expected.parquet"
    sample = _sample()
    sgpd.from_geopandas(sample).to_parquet(ours, compression=compression)
    sample.to_parquet(expected, compression=compression)
    assert _codec(ours) == _codec(expected)
    _assert_same_frame(gpd.read_parquet(ours), gpd.read_parquet(expected))


def test_to_parquet_covering_bbox_filters_like_geopandas(tmp_path):
    ours, expected = tmp_path / "ours.parquet", tmp_path / "expected.parquet"
    sample = _sample()
    sgpd.from_geopandas(sample).to_parquet(ours, write_covering_bbox=True)
    sample.to_parquet(expected, write_covering_bbox=True)
    assert _geo_metadata(ours)["version"] == "1.1.0"
    assert "covering" in _geo_metadata(ours)["columns"]["geometry"]
    for bbox in [(0.5, 0.5, 1.5, 1.5), (4, 4, 6, 6), (10, 10, 11, 11)]:
        got = gpd.read_parquet(ours, bbox=bbox)
        want = gpd.read_parquet(expected, bbox=bbox)
        assert sorted(got["name"]) == sorted(want["name"])


@pytest.mark.parametrize("schema_version", ["1.0.0", "1.1.0"])
def test_to_parquet_schema_version(tmp_path, schema_version):
    path = tmp_path / "out.parquet"
    sgpd.from_geopandas(_sample()).to_parquet(path, schema_version=schema_version)
    assert _geo_metadata(path)["version"] == schema_version


def test_to_parquet_validation(tmp_path):
    gdf = sgpd.from_geopandas(_sample())
    with pytest.raises(NotImplementedError, match="index"):
        gdf.to_parquet(tmp_path / "out.parquet", index=True)
    with pytest.raises(NotImplementedError, match="geometry_encoding"):
        gdf.to_parquet(tmp_path / "out.parquet", geometry_encoding="geoarrow")
    with pytest.raises(NotImplementedError, match="schema_version"):
        gdf.to_parquet(tmp_path / "out.parquet", schema_version="0.4.0")
    with pytest.raises(ValueError, match="covering"):
        gdf.to_parquet(
            tmp_path / "out.parquet", write_covering_bbox=True, schema_version="1.0.0"
        )
    with pytest.raises(ValueError, match="compression"):
        gdf.to_parquet(tmp_path / "out.parquet", compression="lzo")
    with pytest.raises(NotImplementedError, match="row_group_size"):
        gdf.to_parquet(tmp_path / "out.parquet", row_group_size=10)
    with pytest.raises(ValueError, match="extension"):
        gdf.to_parquet(tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_to_parquet_needs_a_crs(tmp_path):
    # GeoPandas writes an unknown CRS as null; SedonaDB's writer refuses.
    gdf = sgpd.from_geopandas(_sample().set_crs(None, allow_override=True))
    with pytest.raises(ValueError, match="set_crs"):
        gdf.to_parquet(tmp_path / "out.parquet")
    assert not (tmp_path / "out.parquet").exists()


@pytest.mark.parametrize(
    "name, kwargs, geometry",
    [
        ("sample.gpkg", {"layer": "things"}, "geom"),
        ("sample.geojson", {}, "wkb_geometry"),
    ],
)
def test_read_file_matches_geopandas(tmp_path, name, kwargs, geometry):
    # The geometry column keeps the source's name, where GeoPandas renames it.
    path = tmp_path / name
    _sample().to_file(path, **kwargs)
    ours = sgpd.read_file(path, **kwargs)
    assert ours._geometry_name == geometry
    _assert_same_frame(
        ours.to_geopandas().rename_geometry("geometry"), gpd.read_file(path, **kwargs)
    )


def test_read_file_columns_and_pyogrio_options(tmp_path):
    path = tmp_path / "sample.gpkg"
    _sample().to_file(path, layer="things")
    ours = sgpd.read_file(path, layer="things", columns=["v"], where="v > 1.5")
    expected = gpd.read_file(path, layer="things", columns=["v"], where="v > 1.5")
    assert ours.columns == ["v", "geom"]
    _assert_same_frame(ours.to_geopandas().rename_geometry("geometry"), expected)
    # Selecting and filtering the result keeps working.
    subset = ours[["v", "geom"]]
    assert len(subset[subset["v"] > 2].to_geopandas()) == 1


def test_read_file_rejects_unsupported_arguments(tmp_path):
    path = tmp_path / "sample.geojson"
    _sample().to_file(path)
    with pytest.raises(NotImplementedError, match="bbox"):
        sgpd.read_file(path, bbox=(0, 0, 1, 1))
    with pytest.raises(NotImplementedError, match="rows"):
        sgpd.read_file(path, rows=2)
    with pytest.raises(NotImplementedError, match="engine"):
        sgpd.read_file(path, engine="fiona")
