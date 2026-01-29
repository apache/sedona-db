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
import shapely
from sedonadb.testing import PostGIS, SedonaDB, geom_or_null


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize("srid", [None, 4326])
@pytest.mark.parametrize(
    "geom",
    [
        # XY dimensions
        "POINT (1 2)",
        "LINESTRING (1 2, 3 4, 5 6)",
        "POLYGON ((0 1, 2 0, 2 3, 0 3, 0 1))",
        "MULTIPOINT ((1 2), (3 4))",
        "MULTILINESTRING ((1 2, 3 4), (5 6, 7 8))",
        "MULTIPOLYGON (((0 1, 2 0, 2 3, 0 3, 0 1)))",
        "GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (3 4, 5 6))",
        # XYZ dimensions
        "POINT Z (1 2 3)",
        "LINESTRING Z (1 2 3, 4 5 6)",
        "POLYGON Z ((0 1 2, 3 0 2, 3 4 2, 0 4 2, 0 1 2))",
        "MULTIPOINT Z ((1 2 3), (4 5 6))",
        "MULTILINESTRING Z ((1 2 3, 4 5 6), (7 8 9, 10 11 12))",
        "MULTIPOLYGON Z (((0 1 2, 3 0 2, 3 4 2, 0 4 2, 0 1 2)))",
        "GEOMETRYCOLLECTION Z (POINT Z (1 2 3))",
        # XYM dimensions
        "POINT M (1 2 3)",
        "LINESTRING M (1 2 3, 4 5 6)",
        "POLYGON M ((0 1 2, 3 0 2, 3 4 2, 0 4 2, 0 1 2))",
        "MULTIPOINT M ((1 2 3), (4 5 6))",
        "MULTILINESTRING M ((1 2 3, 4 5 6), (7 8 9, 10 11 12))",
        "MULTIPOLYGON M (((0 1 2, 3 0 2, 3 4 2, 0 4 2, 0 1 2)))",
        "GEOMETRYCOLLECTION M (POINT M (1 2 3))",
        # XYZM dimensions
        "POINT ZM (1 2 3 4)",
        "LINESTRING ZM (1 2 3 4, 5 6 7 8)",
        "POLYGON ZM ((0 1 2 3, 4 0 2 3, 4 5 2 3, 0 5 2 3, 0 1 2 3))",
        "MULTIPOINT ZM ((1 2 3 4), (5 6 7 8))",
        "MULTILINESTRING ZM ((1 2 3 4, 5 6 7 8), (9 10 11 12, 13 14 15 16))",
        "MULTIPOLYGON ZM (((0 1 2 3, 4 0 2 3, 4 5 2 3, 0 5 2 3, 0 1 2 3)))",
        "GEOMETRYCOLLECTION ZM (POINT ZM (1 2 3 4))",
        # Empty geometries
        "POINT EMPTY",
        "LINESTRING EMPTY",
        "POLYGON EMPTY",
        "MULTIPOINT EMPTY",
        "MULTILINESTRING EMPTY",
        "MULTIPOLYGON EMPTY",
        "GEOMETRYCOLLECTION EMPTY",
        # NULL
        None,
    ],
)
def test_st_asewkb(eng, srid, geom):
    eng = eng.create_or_skip()

    if geom is not None:
        shapely_geom = shapely.from_wkt(geom)
        if srid is not None:
            shapely_geom = shapely.set_srid(shapely_geom, srid)
            write_srid = True
        else:
            write_srid = False

        expected = shapely.to_wkb(
            shapely_geom,
            output_dimension=4,
            byte_order=1,
            flavor="extended",
            include_srid=write_srid,
        )
    else:
        expected = None

    eng.assert_query_result(f"SELECT ST_AsEWKB({geom_or_null(geom, srid)})", expected)
