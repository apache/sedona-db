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

import io
import tempfile
import warnings
from pathlib import Path

import geoarrow.pyarrow as ga
import geopandas
import geopandas.testing
import pandas as pd
import pyarrow as pa
import pytest
import sedonadb
import shapely


def test_read_ogr_projection(con):
    n = 1024
    series = geopandas.GeoSeries.from_xy(
        list(range(n)), list(range(1, n + 1)), crs="EPSG:3857"
    )
    gdf = geopandas.GeoDataFrame({"idx": list(range(n)), "wkb_geometry": series})
    gdf = gdf.set_geometry(gdf["wkb_geometry"])

    with tempfile.TemporaryDirectory() as td:
        temp_fgb_path = f"{td}/temp.fgb"
        gdf.to_file(temp_fgb_path)
        con.read_pyogrio(temp_fgb_path).to_view("test_fgb", overwrite=True)

        # With no projection
        geopandas.testing.assert_geodataframe_equal(
            con.sql("SELECT * FROM test_fgb ORDER BY idx").to_pandas(), gdf
        )

        # With only not geometry selected
        pd.testing.assert_frame_equal(
            con.sql("SELECT idx FROM test_fgb ORDER BY idx").to_pandas(),
            gdf.filter(["idx"]),
        )

        # With reversed columns
        pd.testing.assert_frame_equal(
            con.sql("SELECT wkb_geometry, idx FROM test_fgb ORDER BY idx").to_pandas(),
            gdf.filter(["wkb_geometry", "idx"]),
        )


def test_read_ogr_multi_file(con):
    n = 1024 * 16
    partitions = ["part_{c}" for c in "abcdefghijklmnop"]
    series = geopandas.GeoSeries.from_xy(
        list(range(n)), list(range(1, n + 1)), crs="EPSG:3857"
    )
    gdf = geopandas.GeoDataFrame(
        {
            "idx": list(range(n)),
            "partition": [partitions[i % len(partitions)] for i in range(n)],
            "wkb_geometry": series,
        }
    )
    gdf = gdf.set_geometry(gdf["wkb_geometry"])

    with tempfile.TemporaryDirectory() as td:
        # Create partitioned files by writing Parquet first and translating
        # one file at a time. We need to cast partition in pandas>=3.0 because
        # the default translation of a string column is LargeUtf8 and this is not
        # currently supported by DataFusion partition_by.
        con.create_data_frame(gdf).to_view("tmp_gdf", overwrite=True)
        con.sql(
            """SELECT idx, partition::VARCHAR AS partition, wkb_geometry FROM tmp_gdf"""
        ).to_parquet(td, partition_by="partition")
        for parquet_path in Path(td).rglob("*.parquet"):
            fgb_path = str(parquet_path).replace(".parquet", ".fgb")
            con.read_parquet(parquet_path).to_pandas().to_file(fgb_path)

        # Reading a directory while specifying the extension should work
        con.read_pyogrio(f"{td}", extension="fgb").to_view(
            "gdf_from_dir", overwrite=True
        )
        geopandas.testing.assert_geodataframe_equal(
            con.sql("SELECT * FROM gdf_from_dir ORDER BY idx").to_pandas(),
            gdf.filter(["idx", "wkb_geometry"]),
        )

        # Reading using a glob without specifying the extension should work
        con.read_pyogrio(f"{td}/**/*.fgb").to_view("gdf_from_glob", overwrite=True)
        geopandas.testing.assert_geodataframe_equal(
            con.sql("SELECT * FROM gdf_from_glob ORDER BY idx").to_pandas(),
            gdf.filter(["idx", "wkb_geometry"]),
        )


def test_read_ogr_filter(con):
    n = 1024
    series = geopandas.GeoSeries.from_xy(
        list(range(n)), list(range(1, n + 1)), crs="EPSG:3857"
    )
    gdf = geopandas.GeoDataFrame({"idx": list(range(n)), "wkb_geometry": series})
    gdf = gdf.set_geometry(gdf["wkb_geometry"])

    with tempfile.TemporaryDirectory() as td:
        temp_fgb_path = f"{td}/temp.fgb"
        gdf.to_file(temp_fgb_path)
        con.read_pyogrio(temp_fgb_path).to_view("test_fgb", overwrite=True)

        # With something that should trigger a bounding box filter
        geopandas.testing.assert_geodataframe_equal(
            con.sql(
                """
                SELECT * FROM test_fgb
                WHERE ST_Equals(wkb_geometry, ST_Point(1, 2, 3857))
                """
            ).to_pandas(),
            gdf[gdf.geometry.geom_equals(shapely.Point(1, 2))].reset_index(drop=True),
        )


def test_read_ogr_file_not_found(con):
    with pytest.raises(
        sedonadb._lib.SedonaError, match="Can't infer schema for zero objects"
    ):
        con.read_pyogrio("this/is/not/a/directory")

    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(
            sedonadb._lib.SedonaError, match="Can't infer schema for zero objects"
        ):
            con.read_pyogrio(Path(td) / "file_does_not_exist")


def test_write_ogr(con):
    with tempfile.TemporaryDirectory() as td:
        # Basic write with defaults
        df = con.sql("SELECT ST_Point(0, 1, 3857)")
        expected = geopandas.GeoDataFrame(
            {"geometry": geopandas.GeoSeries.from_wkt(["POINT (0 1)"], crs=3857)}
        )

        df.to_pyogrio(f"{td}/foofy.fgb")
        geopandas.testing.assert_geodataframe_equal(
            geopandas.read_file(f"{td}/foofy.fgb"), expected
        )

        # Ensure Path input works
        df.to_pyogrio(Path(f"{td}/foofy.fgb"))
        geopandas.testing.assert_geodataframe_equal(
            geopandas.read_file(f"{td}/foofy.fgb"), expected
        )

        # Ensure zipped FlatGeoBuf doesn't require specifying the driver
        df.to_pyogrio(Path(f"{td}/foofy.fgb.zip"))
        geopandas.testing.assert_geodataframe_equal(
            geopandas.read_file(f"{td}/foofy.fgb.zip"), expected
        )

        # Ensure inferred CRS that is None works
        # pyogrio warns for the case where a CRS is None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            con.sql("SELECT ST_Point(0, 1)").to_pyogrio(f"{td}/foofy.fgb")
            expected = geopandas.GeoDataFrame(
                {"geometry": geopandas.GeoSeries.from_wkt(["POINT (0 1)"])}
            )
            geopandas.testing.assert_geodataframe_equal(
                geopandas.read_file(f"{td}/foofy.fgb"), expected
            )


def test_write_ogr_buffer(con):
    buf = io.BytesIO()
    df = con.sql("SELECT ST_Point(0, 1, 3857)")
    expected = geopandas.GeoDataFrame(
        {"geometry": geopandas.GeoSeries.from_wkt(["POINT (0 1)"], crs=3857)}
    )

    df.to_pyogrio(buf, driver="FlatGeoBuf")
    geopandas.testing.assert_geodataframe_equal(
        geopandas.read_file(buf.getvalue()), expected
    )

    # Ensure reasonable error if driver is not specified
    with pytest.raises(ValueError, match="driver must be provided"):
        df.to_pyogrio(buf)


def test_write_ogr_no_geometry(con):
    with tempfile.TemporaryDirectory() as td:
        df = con.sql("SELECT 'one' as one")
        expected = pd.DataFrame({"one": ["one"]})

        df.to_pyogrio(f"{td}/foofy.csv")
        pd.testing.assert_frame_equal(pd.read_csv(f"{td}/foofy.csv"), expected)


def test_write_ogr_many_batches(con):
    # Check with a non-trivial number of batches
    con.funcs.table.sd_random_geometry("MultiLineString", 50000, seed=4837).to_view(
        "pyogrio_test"
    )
    df = con.sql(
        """
        SELECT id, ST_SetCrs(geometry, 'EPSG:4326') AS geometry
        FROM pyogrio_test
        ORDER BY id
        """
    )
    expected = df.to_pandas()

    with tempfile.TemporaryDirectory() as td:
        df.to_pyogrio(f"{td}/foofy.gpkg")
        geopandas.testing.assert_geodataframe_equal(
            geopandas.read_file(f"{td}/foofy.gpkg"), expected
        )


def test_write_ogr_from_view_types(con):
    # Check that we can write something with view types (even though it is read back
    # as the simplified type)
    wkb_array = ga.with_crs(ga.as_wkb(["POINT (0 1)", "POINT (1 2)"]), ga.OGC_CRS84)
    wkb_view_array = (
        ga.wkb_view()
        .with_crs(ga.OGC_CRS84)
        .wrap_array(wkb_array.storage.cast(pa.binary_view()))
    )
    tab_simple = pa.table(
        {"string_col": pa.array(["one", "two"], pa.string()), "wkb_geometry": wkb_array}
    )
    tab = pa.table(
        {
            "string_col": pa.array(["one", "two"], pa.string_view()),
            "wkb_geometry": wkb_view_array,
        }
    )

    with tempfile.TemporaryDirectory() as td:
        con.create_data_frame(tab).to_pyogrio(f"{td}/foofy.fgb")
        tab_roundtrip = con.read_pyogrio(f"{td}/foofy.fgb").to_arrow_table()
        assert tab_roundtrip.sort_by("string_col") == tab_simple
