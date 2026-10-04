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


@pytest.mark.parametrize("geometry_type", ["POINT", "LINESTRING", "POLYGON"])
@pytest.mark.parametrize("dimension", ["", " Z", " M", " ZM"])
@pytest.mark.parametrize(
    "function",
    [
        "ST_Normalize(g)",
        "ST_MakeValid(g)",
        "ST_Simplify(g, 1)",
        "ST_SimplifyPreserveTopology(g, 1)",
    ],
)
def test_geos_empty_simple_geometry_dimensions(con, geometry_type, dimension, function):
    wkt = f"{geometry_type}{dimension} EMPTY"
    result = (
        con.sql(
            f"""
        WITH t AS (
            SELECT ST_GeomFromWKT('{wkt}') AS g, 0 AS row_id
            UNION ALL
            SELECT ST_GeomFromWKT(NULL) AS g, 1 AS row_id
        )
        SELECT ST_AsText({function}) AS wkt FROM t ORDER BY row_id
        """
        )
        .to_arrow_table()
        .to_pylist()
    )
    assert result == [{"wkt": wkt}, {"wkt": None}]


@pytest.mark.parametrize("geometry_type", ["POINT", "LINESTRING", "POLYGON"])
def test_geos_empty_simple_geometry_z_point_result(con, geometry_type):
    result = (
        con.sql(
            f"SELECT ST_AsText(ST_PointOnSurface(ST_GeomFromWKT('{geometry_type} Z EMPTY'))) AS wkt"
        )
        .to_arrow_table()
        .to_pylist()
    )
    assert result == [{"wkt": "POINT Z EMPTY"}]
