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

import pytest
import json
import sedonadb
from sedonadb.testing import SedonaDB, PostGIS

if "s2geography" not in sedonadb.__features__:
    pytest.skip("Python package built without s2geography", allow_module_level=True)


@pytest.mark.parametrize("spatial_join_enabled", [True, False])
@pytest.mark.parametrize(
    "join_type", ["INNER JOIN", "LEFT OUTER JOIN", "RIGHT OUTER JOIN"]
)
@pytest.mark.parametrize(
    "on",
    [
        # PostGIS only supports Intersects and DWithin for geography joins
        "ST_Intersects(sjoin_geog1.geog, sjoin_geog2.geog)",
        "ST_Distance(sjoin_geog1.geog, sjoin_geog2.geog) < 100000",
    ],
)
def test_spatial_join_geography_matches_postgis(spatial_join_enabled, join_type, on):
    with (
        SedonaDB.create_or_skip() as eng_sedonadb,
        PostGIS.create_or_skip() as eng_postgis,
    ):
        eng_sedonadb.con.sql(
            f"SET sedona.spatial_join.enable = {str(spatial_join_enabled).lower()}"
        ).execute()

        # Select two sets of bounding boxes that cross the antimeridian,
        # which would be disjoint on a Euclidean plane. A geography join will produce non-empty results,
        # whereas a geometry join would not.
        east_most_bound = [170, -10, 190, 10]
        west_most_bound = [-190, -10, -170, 10]

        options = json.dumps(
            {
                "geom_type": "Polygon",
                "hole_rate": 0.5,
                "num_parts": [2, 10],
                "num_vertices": [2, 10],
                "bounds": east_most_bound,
                "size": [0.1, 5],
                "seed": 44,
            }
        )
        df_polygon = eng_sedonadb.execute_and_collect(
            f"SELECT id, ST_SetSRID(ST_GeogFromWKB(ST_AsBinary(geometry)), 4326) geog, dist FROM sd_random_geometry('{options}') LIMIT 100"
        )

        options = json.dumps(
            {
                "geom_type": "Point",
                "num_parts": [2, 10],
                "num_vertices": [2, 10],
                "bounds": west_most_bound,
                "size": [0.1, 5],
                "seed": 542,
            }
        )
        df_point = eng_sedonadb.execute_and_collect(
            f"SELECT id, ST_SetSRID(ST_GeogFromWKB(ST_AsBinary(geometry)), 4326) geog, dist FROM sd_random_geometry('{options}') LIMIT 100"
        )

        eng_sedonadb.create_table_arrow("sjoin_geog1", df_polygon)
        eng_sedonadb.create_table_arrow("sjoin_geog2", df_point)
        eng_postgis.create_table_arrow("sjoin_geog1", df_polygon)
        eng_postgis.create_table_arrow("sjoin_geog2", df_point)

        sql = f"""
               SELECT sjoin_geog1.id id0, sjoin_geog2.id id1
               FROM sjoin_geog1 {join_type} sjoin_geog2
               ON {on}
               ORDER BY id0, id1
               """

        # Check that this executes and results in a non-empty result
        sedonadb_results = eng_sedonadb.execute_and_collect(sql).to_pandas()
        assert len(sedonadb_results) > 0

        # Check that a PostGIS join produces the same results
        eng_postgis.assert_query_result(sql, sedonadb_results)
