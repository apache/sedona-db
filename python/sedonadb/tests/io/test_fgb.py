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

import geopandas
import geopandas.testing
import pandas as pd
import pyarrow as pa
import pytest
import shapely


@pytest.mark.parametrize("reader", ["native", "pyogrio"])
@pytest.mark.parametrize("spatial_index", [False, True])
def test_fgb_native_and_gdal_projection(con, tmp_path, reader, spatial_index):
    expected = geopandas.GeoDataFrame(
        {
            "idx": list(range(31)),
            "flag": [i % 2 == 0 for i in range(31)],
            "small": pd.Series(range(31), dtype="int16"),
            "real": [i + 0.125 for i in range(31)],
            "label": [None if i % 3 == 0 else f"value {i}" for i in range(31)],
            "when": pd.date_range("2026-01-01", periods=31, tz="UTC"),
            "geometry": geopandas.GeoSeries.from_xy(
                list(range(31)), list(range(1, 32)), crs="EPSG:3857"
            ),
        }
    )
    path = tmp_path / "points.fgb"
    expected.to_file(path, driver="FlatGeobuf", SPATIAL_INDEX=spatial_index)
    if reader == "native":
        frame = con.read(path, format="fgb_native")
    else:
        frame = con.read_pyogrio(path)
    frame.to_view("fgb_comparison", overwrite=True)
    pd.testing.assert_frame_equal(
        con.sql("SELECT idx FROM fgb_comparison ORDER BY idx").to_pandas(),
        expected[["idx"]],
        check_dtype=False,
    )
    assert con.sql("SELECT count(*) AS n FROM fgb_comparison").to_pandas()["n"][0] == 31
    actual = con.sql("SELECT * FROM fgb_comparison ORDER BY idx").to_pandas()
    if actual.geometry.name != "geometry":
        actual = actual.rename_geometry("geometry")
    geopandas.testing.assert_geodataframe_equal(actual, expected, check_dtype=False)


@pytest.mark.parametrize(
    "fixture",
    [
        "geometry",
        "geometry-z",
        "point-z",
        "polygon-z",
        "geometrycollection-nested",
    ],
)
def test_native_fgb_geoarrow_roundtrip(con, geoarrow_data, tmp_path, fixture):
    path = geoarrow_data / "example" / "files" / f"example_{fixture}_wkb.arrows"
    if not path.exists():
        pytest.skip(f"geoarrow-data fixture not present: {path}")
    table = pa.ipc.open_stream(path).read_all()
    expected = geopandas.GeoDataFrame(
        {"idx": range(table.num_rows)},
        geometry=geopandas.GeoSeries.from_wkb(table["geometry"].to_pylist()),
    )
    target = tmp_path / "roundtrip.fgb"
    # Mixed types and null geometries require the unindexed GDAL writer path.
    expected.to_file(target, driver="FlatGeobuf", SPATIAL_INDEX=False)
    actual = con.read(target, format="fgb_native")
    assert actual.to_arrow_table().column_names == ["idx", "wkb_geometry"]
    actual.to_view("native_roundtrip", overwrite=True)
    restored = con.sql("SELECT * FROM native_roundtrip ORDER BY idx").to_pandas()
    assert restored["idx"].tolist() == expected["idx"].tolist()
    # GDAL normalizes some mixed-type dimensions on write. Compare the actual
    # serialized file's geometry, including nulls, rather than that writer's input.
    # Compare full-precision WKB: the default WKT precision could hide drift.
    written = geopandas.read_file(target).sort_values("idx")
    pd.testing.assert_series_equal(
        pd.Series(shapely.to_wkb(restored.geometry, output_dimension=4)),
        pd.Series(shapely.to_wkb(written.geometry, output_dimension=4)),
        check_names=False,
    )


def test_native_fgb_larger_point_roundtrip(con, tmp_path):
    # Several reader batches and partitions, with every row identified uniquely.
    expected = geopandas.GeoDataFrame(
        {"idx": range(20000)},
        geometry=geopandas.GeoSeries.from_xy(
            range(20000), range(1, 20001), crs="EPSG:3857"
        ),
    )
    target = tmp_path / "points.fgb"
    expected.to_file(target, driver="FlatGeobuf")
    con.read(target, format="fgb_native").to_view("native_large", overwrite=True)
    restored = (
        con.sql("SELECT * FROM native_large ORDER BY idx")
        .to_pandas()
        .rename_geometry("geometry")
    )
    geopandas.testing.assert_geodataframe_equal(restored, expected, check_dtype=False)


def test_native_fgb_directory_and_options(con, tmp_path):
    expected = geopandas.GeoDataFrame(
        {"idx": [0, 1]},
        geometry=geopandas.GeoSeries.from_xy([0, 1], [1, 2], crs="EPSG:3857"),
    )
    expected.iloc[:1].to_file(tmp_path / "first.fgb", driver="FlatGeobuf")
    expected.iloc[1:].to_file(tmp_path / "second.fgb", driver="FlatGeobuf")
    (tmp_path / "unrelated.txt").write_text("not FlatGeobuf")
    frame = con.read(
        tmp_path,
        format="fgb_native",
        options={"geometry_column_name": "shape", "metadata_size_hint": "12"},
    )
    assert frame.to_arrow_table().column_names == ["idx", "shape"]
    frame.to_view("native_directory", overwrite=True)
    restored = (
        con.sql("SELECT * FROM native_directory ORDER BY idx")
        .to_pandas()
        .rename_geometry("geometry")
    )
    geopandas.testing.assert_geodataframe_equal(restored, expected, check_dtype=False)


@pytest.mark.parametrize(
    "fixture",
    [
        "natural-earth/files/natural-earth_countries_geo.parquet",
        "quadrangles/files/quadrangles_100k_geo.parquet",
    ],
)
def test_native_fgb_geoarrow_dataset_roundtrip(con, geoarrow_data, tmp_path, fixture):
    source = geoarrow_data / fixture
    if not source.exists():
        pytest.skip(f"geoarrow-data fixture not present: {source}")
    expected = geopandas.read_parquet(source)
    expected["roundtrip_idx"] = range(len(expected))
    target = tmp_path / "dataset.fgb"
    expected.to_file(target, driver="FlatGeobuf")
    con.read(target, format="fgb_native").to_view("native_dataset", overwrite=True)
    restored = con.sql(
        "SELECT * FROM native_dataset ORDER BY roundtrip_idx"
    ).to_pandas()
    restored = restored.rename_geometry(expected.geometry.name)
    # GDAL can normalize attribute widths; all values, geometries and CRS remain.
    geopandas.testing.assert_geodataframe_equal(
        restored, expected, check_dtype=False, check_like=True
    )
