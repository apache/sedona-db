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
import math
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyogrio
import pyproj
import pytest
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal

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


def test_read_parquet_from_public_s3_with_storage_options():
    # As in SedonaDB's own reader tests: anonymous access to a public bucket.
    gdf = sgpd.read_parquet(
        "s3://wherobots-examples/data/onboarding_1/nyc_buildings.parquet",
        storage_options={"aws.skip_signature": True, "aws.region": "us-west-2"},
    )
    assert gdf._geometry_name == "geom"
    assert len(gdf.head(3).to_geopandas()) == 3


@pytest.mark.parametrize(
    "url",
    ["s3://bucket/out.parquet", "gs://bucket/out.parquet", "https://host/out.parquet"],
)
def test_to_parquet_refuses_a_remote_url(tmp_path, url):
    with pytest.raises(NotImplementedError, match="1440"):
        sgpd.from_geopandas(_sample()).to_parquet(url)


def test_to_parquet_to_a_file_uri(tmp_path):
    path = tmp_path / "with space.parquet"
    sgpd.from_geopandas(_sample()).to_parquet(path.as_uri())
    assert len(gpd.read_parquet(path)) == 4


def test_read_parquet_rejects_unsupported_arguments(tmp_path):
    path = tmp_path / "sample.parquet"
    _sample().to_parquet(path)
    with pytest.raises(NotImplementedError, match="bbox"):
        sgpd.read_parquet(path, bbox=(0, 0, 1, 1))
    # storage_options are SedonaDB's object-store options, not fsspec's.
    with pytest.raises(Exception, match="anon"):
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


@pytest.mark.parametrize(
    "schema_version, covering", [("1.0.0", False), ("1.1.0", True)]
)
def test_to_parquet_schema_version(tmp_path, schema_version, covering):
    path = tmp_path / "out.parquet"
    sgpd.from_geopandas(_sample()).to_parquet(
        path, schema_version=schema_version, write_covering_bbox=covering
    )
    metadata = _geo_metadata(path)
    assert metadata["version"] == schema_version
    assert ("covering" in metadata["columns"]["geometry"]) == covering


def test_to_parquet_rewrites_a_covering_read_back(tmp_path):
    # A file's covering is left out on read, as in GeoPandas, so writing a
    # covering again computes it from the (here changed) geometry.
    source, out = tmp_path / "source.parquet", tmp_path / "out.parquet"
    _sample().to_parquet(source, write_covering_bbox=True)
    gdf = sgpd.read_parquet(source)
    assert gdf.columns == list(gpd.read_parquet(source).columns)
    assert "bbox" not in gdf.columns
    gdf["geometry"] = gdf.geometry.buffer(1.0)
    gdf.to_parquet(out, write_covering_bbox=True)
    written = gpd.read_parquet(out)
    boxes = pq.read_table(out, columns=["bbox"]).column("bbox").to_pylist()
    for box, bounds in zip(boxes, written.geometry.bounds.itertuples(index=False)):
        if box is None:
            assert pd.isna(bounds.minx)
        else:
            assert (
                box["xmin"],
                box["ymin"],
                box["xmax"],
                box["ymax"],
            ) == pytest.approx(tuple(bounds))


def _with_bbox(tmp_path, name, fields, declared):
    """A GeoParquet file whose `bbox` struct has `fields`, declared as the
    covering or not."""
    source = tmp_path / "source.parquet"
    _sample().to_parquet(source, write_covering_bbox=True)
    table = pq.read_table(source)
    geo = json.loads(table.schema.metadata[b"geo"])
    boxes = pa.StructArray.from_arrays(
        [pa.array([float(i)] * len(table)) for i in range(len(fields))], names=fields
    )
    table = table.set_column(table.schema.get_field_index("bbox"), "bbox", boxes)
    if declared:
        geo["columns"]["geometry"]["covering"] = {
            "bbox": {field: ["bbox", field] for field in fields}
        }
    else:
        geo["columns"]["geometry"].pop("covering")
    metadata = {**table.schema.metadata, b"geo": json.dumps(geo).encode()}
    path = tmp_path / name
    pq.write_table(table.replace_schema_metadata(metadata), path)
    return path


@pytest.mark.parametrize(
    "fields",
    [
        ["xmin", "xmax", "ymin", "ymax"],
        ["xmin", "ymin", "zmin", "xmax", "ymax", "zmax"],
    ],
    ids=["reordered", "with-z"],
)
def test_read_parquet_leaves_out_any_declared_covering(tmp_path, fields):
    # The covering is found by its metadata, whatever its field order or
    # dimensions, so it is left out and the file can be rewritten with one.
    path = _with_bbox(tmp_path, "covering.parquet", fields, declared=True)
    gdf = sgpd.read_parquet(path)
    assert gdf.columns == list(gpd.read_parquet(path).columns)
    gdf.to_parquet(tmp_path / "out.parquet", write_covering_bbox=True)
    # Named explicitly, it is read, as in GeoPandas.
    assert "bbox" in sgpd.read_parquet(path, columns=["v", "geometry", "bbox"]).columns
    # Also for a directory of such files.
    (tmp_path / "parts").mkdir()
    path.rename(tmp_path / "parts" / "a.parquet")
    assert "bbox" not in sgpd.read_parquet(tmp_path / "parts").columns


@pytest.mark.parametrize(
    "spell",
    [
        lambda path: path.as_uri(),
        lambda path: (p for p in [path]),
        lambda path: [str(path)],
    ],
    ids=["file-uri", "generator", "list"],
)
def test_read_parquet_finds_coverings_however_the_path_is_given(tmp_path, spell):
    # A percent-encoded file URI (the space below) and a one-shot iterable
    # both reach the file's metadata, so the covering is left out.
    source = tmp_path / "with covering.parquet"
    _sample().to_parquet(source, write_covering_bbox=True)
    gdf = sgpd.read_parquet(spell(source))
    assert "bbox" not in gdf.columns
    assert len(gdf.to_geopandas()) == 4
    gdf.to_parquet(tmp_path / "out.parquet", write_covering_bbox=True)


def test_to_parquet_never_overwrites_an_undeclared_bbox(tmp_path):
    # A bounding-box struct that the metadata does not declare as a covering
    # is ordinary data (a survey envelope, say): it is read like any column,
    # and writing a covering over it raises rather than replacing it.
    path = _with_bbox(
        tmp_path, "survey.parquet", ["xmin", "ymin", "xmax", "ymax"], declared=False
    )
    gdf = sgpd.read_parquet(path)
    assert gdf.columns == list(gpd.read_parquet(path).columns)
    assert "bbox" in gdf.columns
    with pytest.raises(ValueError, match="bbox"):
        gdf.to_parquet(tmp_path / "out.parquet", write_covering_bbox=True)
    assert not (tmp_path / "out.parquet").exists()


def test_to_parquet_covering_refuses_an_ordinary_bbox_column(tmp_path):
    sample = _sample().assign(bbox="not a box")
    with pytest.raises(ValueError, match="bbox"):
        sample.to_parquet(tmp_path / "theirs.parquet", write_covering_bbox=True)
    with pytest.raises(ValueError, match="bbox"):
        sgpd.from_geopandas(sample).to_parquet(
            tmp_path / "ours.parquet", write_covering_bbox=True
        )
    # Without a covering it is an ordinary column, as in GeoPandas.
    sgpd.from_geopandas(sample).to_parquet(tmp_path / "plain.parquet")
    assert (
        list(gpd.read_parquet(tmp_path / "plain.parquet")["bbox"]) == ["not a box"] * 4
    )


def test_to_parquet_covers_every_geometry_column(tmp_path):
    path = tmp_path / "out.parquet"
    sample = _sample()
    sample["other"] = sample.geometry.centroid
    sgpd.from_geopandas(sample).to_parquet(path, write_covering_bbox=True)
    columns = _geo_metadata(path)["columns"]
    assert columns["geometry"]["covering"]["bbox"]["xmin"] == ["bbox", "xmin"]
    assert columns["other"]["covering"]["bbox"]["xmin"] == ["other_bbox", "xmin"]


def test_to_parquet_validation(tmp_path):
    gdf = sgpd.from_geopandas(_sample())
    with pytest.raises(NotImplementedError, match="index"):
        gdf.to_parquet(tmp_path / "out.parquet", index=True)
    with pytest.raises(NotImplementedError, match="geometry_encoding"):
        gdf.to_parquet(tmp_path / "out.parquet", geometry_encoding="geoarrow")
    with pytest.raises(NotImplementedError, match="schema_version"):
        gdf.to_parquet(tmp_path / "out.parquet", schema_version="0.4.0")
    with pytest.raises(NotImplementedError, match="covering"):
        gdf.to_parquet(
            tmp_path / "out.parquet", write_covering_bbox=True, schema_version="1.0.0"
        )
    with pytest.raises(NotImplementedError, match="write_covering_bbox"):
        gdf.to_parquet(tmp_path / "out.parquet", schema_version="1.1.0")
    with pytest.raises(ValueError, match="compression"):
        gdf.to_parquet(tmp_path / "out.parquet", compression="lzo")
    with pytest.raises(NotImplementedError, match="row_group_size"):
        gdf.to_parquet(tmp_path / "out.parquet", row_group_size=10)
    with pytest.raises(ValueError, match="extension"):
        gdf.to_parquet(tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_to_parquet_writes_null_crs(tmp_path):
    gdf = sgpd.from_geopandas(_sample().set_crs(None, allow_override=True))
    path = tmp_path / "out.parquet"
    gdf.to_parquet(path)

    assert _geo_metadata(path)["columns"]["geometry"]["crs"] is None
    assert sgpd.read_parquet(path).crs is None


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


# Geometries for each layer written by the to_file() tests below, chosen for
# the layer geometry type GeoPandas gives each: one type, single and multi
# types (promoted for GPKG and FlatGeobuf), 3D, several types, 2D with 3D,
# and 2D with an empty 3D geometry (which does not count as 3D).
_LAYERS = {
    "points": ["POINT (0 0)", "POINT (1 2)", None],
    "polygons": [
        "POLYGON ((0 0, 1 0, 1 1, 0 0))",
        "MULTIPOLYGON (((2 2, 3 2, 3 3, 2 2)))",
        "POLYGON EMPTY",
    ],
    "3D lines": ["LINESTRING Z (0 0 1, 1 1 2)", "LINESTRING Z (2 2 0, 3 3 1)"],
    "points and lines": ["POINT (0 0)", "LINESTRING (0 0, 1 1)"],
    "3D points and lines": ["POINT Z (0 0 1)", "LINESTRING Z (0 0 1, 1 1 2)"],
    "2D and 3D points": ["POINT (0 0)", "POINT Z (1 1 1)"],
    "2D and empty 3D points": ["POINT (0 0)", "POINT Z EMPTY"],
}


def _layer(name, crs=3857):
    geometries = _LAYERS[name]
    n = len(geometries)
    return gpd.GeoDataFrame(
        {
            "i": list(range(n)),
            "f": [1.5, None, 4.0][:n],
            "s": ["a", None, "c"][:n],
            "b": [True, False, True][:n],
            # Loaded as Arrow's null type, which GDAL does not take.
            "none": [None] * n,
            # Loaded dictionary-encoded.
            "category": pd.Categorical([2.5, None, 2.5][:n]),
        },
        geometry=gpd.GeoSeries.from_wkt(geometries),
        crs=crs,
    )


def _written(path, **kwargs):
    """A written layer as a reader sees it: its driver, geometry type, CRS and
    fields, and its rows."""
    info = pyogrio.read_info(path, **kwargs)
    fields = dict(zip(info["fields"], info["dtypes"]))
    described = (info["driver"], info["geometry_type"], info["crs"], fields)
    return described, pyogrio.read_dataframe(path, **kwargs)


def _assert_same_layer(ours, expected, **kwargs):
    ours_described, ours_rows = _written(ours, **kwargs)
    described, rows = _written(expected, **kwargs)
    assert ours_described == described
    # The same geometries, not only topologically equal ones (a polygon and
    # a multipolygon of it).
    assert _values(ours_rows.geometry.to_wkt()) == _values(rows.geometry.to_wkt())
    assert_geodataframe_equal(ours_rows, rows)


@pytest.mark.parametrize("layer", sorted(_LAYERS))
@pytest.mark.parametrize("extension", ["gpkg", "geojson", "fgb", "shp"])
def test_to_file_matches_geopandas(tmp_path, extension, layer):
    source = _layer(layer)
    expected = tmp_path / f"expected.{extension}"
    ours = tmp_path / f"ours.{extension}"
    try:
        source.to_file(expected)
    except RuntimeError as error:
        # A layer the driver cannot take (a missing geometry in an indexed
        # FlatGeobuf, 2D with 3D in one, points with lines in a shapefile)
        # fails for the reason it does in GeoPandas: GDAL's, which pyogrio
        # introduces differently when it writes through Arrow.
        reason = str(error).split(": ", 1)[-1]
        with pytest.raises(RuntimeError, match=re.escape(reason)):
            sgpd.from_geopandas(source).to_file(ours)
        return
    sgpd.from_geopandas(source).to_file(ours)
    _assert_same_layer(ours, expected)


@pytest.mark.parametrize(
    "layer, extension, kwargs",
    [
        ("points", "geojson", {"promote_to_multi": True}),
        ("points", "gpkg", {"promote_to_multi": True}),
        ("polygons", "gpkg", {"promote_to_multi": False}),
        ("polygons", "gpkg", {"geometry_type": "Polygon"}),
        ("points", "gpkg", {"geometry_type": "Unknown"}),
        ("3D lines", "gpkg", {"geometry_type": "LineString", "promote_to_multi": True}),
        # Written through Arrow either way.
        ("polygons", "gpkg", {"use_arrow": True}),
    ],
)
def test_to_file_geometry_type_options_match_geopandas(
    tmp_path, layer, extension, kwargs
):
    source = _layer(layer)
    expected = tmp_path / f"expected.{extension}"
    ours = tmp_path / f"ours.{extension}"
    source.to_file(expected, **kwargs)
    sgpd.from_geopandas(source).to_file(ours, **kwargs)
    _assert_same_layer(ours, expected)


@pytest.mark.parametrize("nan_as_null", [None, True, False])
def test_to_file_writes_nan_as_geopandas(tmp_path, nan_as_null):
    # pandas holds NaN as a missing value, so GeoPandas writes it as null
    # unless asked not to; a NaN computed by the engine is written the same.
    source = gpd.GeoDataFrame(
        {"x": [0.0, 1.0]}, geometry=gpd.points_from_xy([0, 1], [0, 1]), crs=4326
    )
    gdf = sgpd.from_geopandas(source)
    kwargs = {} if nan_as_null is None else {"nan_as_null": nan_as_null}
    source.assign(f=source["x"] / source["x"]).to_file(
        tmp_path / "expected.fgb", **kwargs
    )
    gdf.assign(f=gdf["x"] / gdf["x"]).to_file(tmp_path / "ours.fgb", **kwargs)

    def written(name):
        values = pyogrio.read_arrow(tmp_path / name)[1].column("f").to_pylist()
        return ["NaN" if v is not None and math.isnan(v) else v for v in values]

    # FlatGeobuf orders rows by its spatial index.
    assert written("ours.fgb") == written("expected.fgb")
    assert ("NaN" if nan_as_null is False else None) in written("ours.fgb")


@pytest.mark.parametrize("crs", [4326, 3857])
def test_to_file_writes_an_epsg_code_as_geopandas(tmp_path, crs):
    source = _layer("polygons", crs=crs)
    source.to_file(tmp_path / "expected.gpkg")
    sgpd.from_geopandas(source).to_file(tmp_path / "ours.gpkg")
    _assert_same_layer(tmp_path / "ours.gpkg", tmp_path / "expected.gpkg")
    assert pyogrio.read_info(tmp_path / "ours.gpkg")["crs"] == f"EPSG:{crs}"


def test_to_file_writes_another_crs_as_geopandas(tmp_path):
    # As WKT. The frame holds its CRS as PROJJSON, which keeps fewer
    # authority codes for its parts (the ellipsoid's) than the CRS GeoPandas
    # writes from, so the WKT names the same CRS in other words.
    lcc = (
        "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +x_0=0 +y_0=0 "
        "+datum=NAD83 +units=m +no_defs"
    )
    source = _layer("polygons", crs=lcc)
    source.to_file(tmp_path / "expected.gpkg")
    sgpd.from_geopandas(source).to_file(tmp_path / "ours.gpkg")
    ours = pyogrio.read_info(tmp_path / "ours.gpkg")["crs"]
    expected = pyogrio.read_info(tmp_path / "expected.gpkg")["crs"]
    assert ours.startswith("PROJCS[")
    assert pyproj.CRS.from_user_input(ours) == pyproj.CRS.from_user_input(expected)
    assert_geodataframe_equal(
        pyogrio.read_dataframe(tmp_path / "ours.gpkg"),
        pyogrio.read_dataframe(tmp_path / "expected.gpkg"),
    )


def test_to_file_without_a_crs_warns_as_geopandas(tmp_path):
    source = _layer("points", crs=None)
    with pytest.warns(UserWarning, match="'crs' was not provided"):
        source.to_file(tmp_path / "expected.gpkg")
    with pytest.warns(UserWarning, match="'crs' was not provided"):
        sgpd.from_geopandas(source).to_file(tmp_path / "ours.gpkg")
    _assert_same_layer(tmp_path / "ours.gpkg", tmp_path / "expected.gpkg")


def test_to_file_of_a_lazy_result(tmp_path):
    source = _layer("polygons")
    gdf = sgpd.from_geopandas(source)
    gdf = gdf[gdf["i"] > 0]
    gdf.assign(geometry=gdf.geometry.centroid).to_file(tmp_path / "ours.gpkg")
    expected = source[source["i"] > 0].assign(geometry=source.geometry.centroid)
    expected.to_file(tmp_path / "expected.gpkg")
    _assert_same_layer(tmp_path / "ours.gpkg", tmp_path / "expected.gpkg")


@pytest.mark.parametrize(
    "name, kwargs",
    [
        ("layer.json", {}),
        ("layer.GeoJSON", {}),
        ("layer.geojsonl", {}),
        ("folder", {}),
        ("layer.data", {"driver": "GPKG"}),
    ],
)
def test_to_file_detects_the_driver_as_geopandas(tmp_path, name, kwargs):
    # By extension, and a path without one is a shapefile folder.
    source = _layer("points")
    (tmp_path / "expected").mkdir()
    (tmp_path / "ours").mkdir()
    source.to_file(tmp_path / "expected" / name, **kwargs)
    sgpd.from_geopandas(source).to_file(str(tmp_path / "ours" / name), **kwargs)
    _assert_same_layer(tmp_path / "ours" / name, tmp_path / "expected" / name)


def test_to_file_expands_the_home_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    sgpd.from_geopandas(_layer("points")).to_file("~/ours.gpkg")
    assert (tmp_path / "ours.gpkg").exists()


def test_to_file_layers_and_appending_match_geopandas(tmp_path):
    first, second = _layer("points"), _layer("polygons")
    for target, frame in (("expected", lambda f: f), ("ours", sgpd.from_geopandas)):
        path = tmp_path / f"{target}.gpkg"
        frame(first).to_file(path, layer="first")
        frame(second).to_file(path, layer="second")
        frame(first).to_file(path, layer="first", mode="a")
        frame(first).to_file(path, layer="first", append=True)
    pd.testing.assert_frame_equal(
        sgpd.list_layers(tmp_path / "ours.gpkg"),
        gpd.list_layers(tmp_path / "expected.gpkg"),
    )
    for layer in ("first", "second"):
        _assert_same_layer(
            tmp_path / "ours.gpkg", tmp_path / "expected.gpkg", layer=layer
        )
    assert len(pyogrio.read_dataframe(tmp_path / "ours.gpkg", layer="first")) == 9


def test_to_file_metadata_as_geopandas(tmp_path):
    gdf = sgpd.from_geopandas(_layer("points"))
    gdf.to_file(tmp_path / "ours.gpkg", metadata={"title": "points"})
    assert pyogrio.read_info(tmp_path / "ours.gpkg")["layer_metadata"] == {
        "title": "points"
    }
    with pytest.raises(NotImplementedError, match="only supported for the GPKG"):
        gdf.to_file(tmp_path / "ours.geojson", metadata={"title": "points"})


def test_to_file_without_geometry_as_geopandas(tmp_path):
    source = pd.DataFrame({"a": [1, 2], "s": ["x", None]})
    for extension in ("gpkg", "csv"):
        gpd.GeoDataFrame(source).to_file(tmp_path / f"expected.{extension}")
        sgpd.from_geopandas(source).to_file(tmp_path / f"ours.{extension}")
        ours_described, ours_rows = _written(tmp_path / f"ours.{extension}")
        described, rows = _written(tmp_path / f"expected.{extension}")
        assert ours_described == described
        pd.testing.assert_frame_equal(ours_rows, rows)


def test_to_file_warns_of_long_shapefile_names_as_geopandas(tmp_path):
    gdf = sgpd.from_geopandas(_layer("points").rename(columns={"s": "a_long_name"}))
    with pytest.warns(UserWarning, match="longer than 10 characters"):
        gdf.to_file(tmp_path / "ours.shp")


def test_to_file_validation(tmp_path):
    gdf = sgpd.from_geopandas(_layer("points"))
    path = tmp_path / "ours.gpkg"
    with pytest.raises(ValueError, match="'schema' argument is not supported"):
        gdf.to_file(path, schema={"geometry": "Point", "properties": {}})
    with pytest.raises(ValueError, match="Passing 'crs' is not supported"):
        gdf.to_file(path, crs="EPSG:4326")
    with pytest.raises(ValueError, match="'mode' should be one of 'w' or 'a'"):
        gdf.to_file(path, mode="x")
    with pytest.raises(NotImplementedError, match="engine='pyogrio' only"):
        gdf.to_file(path, engine="fiona")
    with pytest.raises(NotImplementedError, match="no index to write"):
        gdf.to_file(path, index=True)
    # As in GeoPandas, a file takes one geometry column.
    two = gdf.assign(other=gdf.geometry.centroid)
    with pytest.raises(ValueError, match="multiple geometry columns"):
        two.to_file(path)
    geography = sgpd.GeoDataFrame(
        sgpd.default_context().sql("SELECT ST_GeogFromWKT('POINT (1 1)') AS g"),
        geometry="g",
    )
    with pytest.raises(NotImplementedError, match="geography"):
        geography.to_file(path)
    # Arguments are checked before anything is written.
    assert not path.exists()
    # There is no index, so not writing one is fine.
    gdf.to_file(path, index=False)
    assert len(pyogrio.read_dataframe(path)) == 3


def test_geoseries_to_file_matches_geopandas(tmp_path):
    # Written as a frame with the series as its column named "geometry".
    source = _layer("polygons")
    series = source.geometry.rename("shape")
    series.to_file(tmp_path / "expected.gpkg")
    gdf = sgpd.from_geopandas(source.rename_geometry("shape"))
    gdf.geometry.to_file(tmp_path / "ours.gpkg")
    _assert_same_layer(tmp_path / "ours.gpkg", tmp_path / "expected.gpkg")


def test_from_file_reads_as_read_file(tmp_path):
    path = tmp_path / "sample.gpkg"
    _sample().to_file(path, layer="things")
    ours = sgpd.GeoDataFrame.from_file(path, layer="things")
    assert isinstance(ours, sgpd.GeoDataFrame)
    _assert_same_frame(
        ours.to_geopandas().rename_geometry("geometry"),
        gpd.GeoDataFrame.from_file(path, layer="things"),
    )
    # The series is named "geometry", as in GeoPandas, though the file names
    # its column "geom".
    series = sgpd.GeoSeries.from_file(path, layer="things")
    assert isinstance(series, sgpd.GeoSeries)
    assert series.to_geopandas().name == "geometry"
    assert_geoseries_equal(
        series.to_geopandas(), gpd.GeoSeries.from_file(path, layer="things")
    )


@pytest.mark.parametrize("extension", ["gpkg", "geojson", "fgb"])
def test_to_file_writes_unsigned_integers_as_geopandas(tmp_path, extension):
    # GDAL writes an unsigned 64-bit integer as a real, which cannot hold
    # 2**53 + 1: it is written as a signed one, as GeoPandas writes it.
    source = gpd.GeoDataFrame(
        {
            "u64": np.array([2**53 + 1, 2**63 - 1], dtype="uint64"),
            "u32": np.array([2**32 - 1, 0], dtype="uint32"),
            "u8": np.array([255, 0], dtype="uint8"),
        },
        geometry=gpd.points_from_xy([0, 1], [0, 1]),
        crs=4326,
    )
    source.to_file(tmp_path / f"expected.{extension}")
    sgpd.from_geopandas(source).to_file(tmp_path / f"ours.{extension}")
    _assert_same_layer(
        tmp_path / f"ours.{extension}", tmp_path / f"expected.{extension}"
    )
    values = pyogrio.read_arrow(tmp_path / f"ours.{extension}")[1]
    assert sorted(values.column("u64").to_pylist()) == [2**53 + 1, 2**63 - 1]


@pytest.mark.parametrize("extension", ["gpkg", "geojson", "fgb"])
def test_to_file_writes_categorical_unsigned_integers_as_geopandas(tmp_path, extension):
    # A categorical column is dictionary-encoded; its unsigned 64-bit values
    # are written as signed ones too.
    values = np.array([2**53 + 1, 2**63 - 1], dtype="uint64")
    source = gpd.GeoDataFrame(
        {"u64": pd.Categorical(values)},
        geometry=gpd.points_from_xy([0, 1], [0, 1]),
        crs=4326,
    )
    source.to_file(tmp_path / f"expected.{extension}")
    sgpd.from_geopandas(source).to_file(tmp_path / f"ours.{extension}")
    _assert_same_layer(
        tmp_path / f"ours.{extension}", tmp_path / f"expected.{extension}"
    )
    written = pyogrio.read_arrow(tmp_path / f"ours.{extension}")[1]
    assert sorted(written.column("u64").to_pylist()) == [2**53 + 1, 2**63 - 1]


@pytest.mark.parametrize("categorical", [False, True])
def test_to_file_refuses_unsigned_integers_too_large_as_geopandas(
    tmp_path, categorical
):
    values = np.array([1, 2**63], dtype="uint64")
    source = gpd.GeoDataFrame(
        {"u64": pd.Categorical(values) if categorical else values},
        geometry=gpd.points_from_xy([0, 1], [0, 1]),
        crs=4326,
    )
    with pytest.raises(OverflowError):
        source.to_file(tmp_path / "expected.gpkg")
    with pytest.raises(OverflowError, match="'u64'"):
        sgpd.from_geopandas(source).to_file(tmp_path / "ours.gpkg")
    assert not (tmp_path / "ours.gpkg").exists()


@pytest.mark.parametrize("extension", ["gpkg", "geojson", "fgb"])
def test_to_file_writes_time_zones_as_geopandas(tmp_path, extension):
    # GDAL keeps UTC and a fixed offset, but would write the time in a named
    # zone as a time without one: it is written in UTC, as GeoPandas writes
    # it, the same instant.
    times = pd.to_datetime(["2020-01-01 12:34:56", "2020-07-01 12:00:00"])
    source = gpd.GeoDataFrame(
        {
            "named": times.tz_localize("America/New_York"),
            "offset": times.tz_localize("-05:00"),
            "utc": times.tz_localize("UTC"),
            "naive": times,
        },
        geometry=gpd.points_from_xy([0, 1], [0, 1]),
        crs=4326,
    )
    source.to_file(tmp_path / f"expected.{extension}")
    sgpd.from_geopandas(source).to_file(tmp_path / f"ours.{extension}")
    _assert_same_layer(
        tmp_path / f"ours.{extension}", tmp_path / f"expected.{extension}"
    )
    written = pyogrio.read_dataframe(tmp_path / f"ours.{extension}")
    assert sorted(written["named"]) == sorted(source["named"])


@pytest.mark.parametrize(
    "geometry, kwargs",
    [
        (True, {}),
        (True, {"promote_to_multi": True}),
        (True, {"geometry_type": "Point", "promote_to_multi": False}),
        (False, {}),
    ],
)
def test_to_file_writes_the_rows_in_the_frame_order(tmp_path, geometry, kwargs):
    # A sorted frame of more rows than the engine computes at once, with a
    # float column (its NaN written as null), is written in its order, as
    # GeoPandas writes a frame's rows, whatever is computed for writing.
    query = (
        "SELECT value AS a, CAST(value AS DOUBLE) AS f, "
        "ST_SetSRID(ST_Point(value, value), 3857) AS geometry FROM range(100000)"
    )
    gdf = sgpd.GeoDataFrame(sgpd.default_context().sql(query))
    gdf = gdf.sort_values("a", ascending=False)
    if not geometry:
        gdf = gdf[["a", "f"]]
    order = list(range(99999, -1, -1))
    assert gdf._df.to_arrow_table()["a"].to_pylist() == order
    gdf.to_file(tmp_path / "ours.gpkg", **kwargs)
    assert pyogrio.read_arrow(tmp_path / "ours.gpkg")[1]["a"].to_pylist() == order


def test_to_file_writes_what_it_inferred_the_geometry_type_of(tmp_path):
    # The frame is computed once, so the layer's geometry type is that of
    # what is written, even for a query whose result changes each time.
    query = (
        "SELECT ST_SetSRID(ST_GeomFromWKT(CASE WHEN random() < 0.5 "
        "THEN 'POINT (0 0)' ELSE 'LINESTRING (0 0, 1 1)' END), 4326) AS geometry"
    )
    for i in range(20):
        gdf = sgpd.GeoDataFrame(sgpd.default_context().sql(query), geometry="geometry")
        path = tmp_path / f"ours{i}.gpkg"
        gdf.to_file(path)
        written = pyogrio.read_dataframe(path).geom_type.tolist()
        assert pyogrio.read_info(path)["geometry_type"] == written[0]
