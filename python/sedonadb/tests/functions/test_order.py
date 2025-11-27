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
import pytest
import sedonadb


def test_order_sql(con):
    if "s2geography" not in sedonadb.__features__:
        pytest.skip("Ordering currently requires build with feature s2geography")

    wkt_unsorted = [
        None,
        "POINT EMPTY",
        "POINT (-80 -80)",
        "POINT (80 80)",
        "POINT (-79 -79)",
    ]
    wkt_sorted = [
        "POINT (80 80)",
        "POINT (-80 -80)",
        "POINT (-79 -79)",
        "POINT EMPTY",
        None,
    ]
    gdf_unsorted = geopandas.GeoDataFrame(
        {"geometry": geopandas.GeoSeries.from_wkt(wkt_unsorted)}
    )
    gdf_sorted = geopandas.GeoDataFrame(
        {"geometry": geopandas.GeoSeries.from_wkt(wkt_sorted)}
    )

    con.create_data_frame(gdf_unsorted).to_view("unsorted", overwrite=True)
    hopefully_sorted = con.sql(
        "SELECT * FROM unsorted ORDER BY sd_order(geometry)"
    ).to_pandas()
    geopandas.testing.assert_geodataframe_equal(hopefully_sorted, gdf_sorted)
