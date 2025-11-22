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

from pathlib import Path
import tempfile

import geopandas
import geopandas.testing
import pandas as pd
import pytest
import shapely
import sedonadb


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
        # one file at a time
        con.create_data_frame(gdf).to_parquet(td, partition_by="partition")
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
                WHERE ST_Equals(wkb_geometry, ST_SetSRID(ST_Point(1, 2), 3857))
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
