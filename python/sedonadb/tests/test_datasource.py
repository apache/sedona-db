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

import tempfile

import geopandas
import geopandas.testing
import pandas as pd


def test_read_ogr(con):
    n = 1024
    series = geopandas.GeoSeries.from_xy(
        list(range(n)), list(range(1, n + 1)), crs="EPSG:3857"
    )
    gdf = geopandas.GeoDataFrame({"idx": list(range(n)), "wkb_geometry": series})
    gdf = gdf.set_geometry(gdf["wkb_geometry"])

    with tempfile.TemporaryDirectory() as td:
        temp_fgb_path = f"{td}/temp.fgb"
        gdf.to_file(temp_fgb_path)
        con.read_ogr(temp_fgb_path).to_view("test_fgb", overwrite=True)

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
