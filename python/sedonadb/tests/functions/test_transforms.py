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
import pyproj
import pytest
from sedonadb.testing import PostGIS, SedonaDB, geom_or_null, val_or_null


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
def test_st_transform(eng):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        "SELECT ST_Transform(ST_GeomFromText('POINT (1 1)'), 'EPSG:4326', 'EPSG:3857')",
        "POINT (111319.490793274 111325.142866385)",
        wkt_precision=9,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "srid", "expected_srid"),
    [
        ("POINT (1 1)", None, None),
        ("POINT (1 1)", 3857, 3857),
        ("POINT (1 1)", 0, None),
    ],
)
def test_st_setsrid(eng, geom, srid, expected_srid):
    eng = eng.create_or_skip()
    result = eng.execute_and_collect(
        f"SELECT ST_SetSrid({geom_or_null(geom)}, {val_or_null(srid)})"
    )
    df = eng.result_to_pandas(result)
    if expected_srid is None:
        assert df.crs is None
    else:
        assert df.crs == pyproj.CRS(expected_srid)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "srid", "expected_srid"),
    [
        ("POINT (1 1)", 3857, 3857),
        ("POINT (1 1)", 0, 0),
    ],
)
def test_st_srid(eng, geom, srid, expected_srid):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_SRID(ST_SetSrid({geom_or_null(geom)}, {val_or_null(srid)}))",
        expected_srid,
    )


# PostGIS does not have an API ST_SetCrs, ST_Crs
@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geom", "crs", "expected_srid"),
    [
        ("POINT (1 1)", "EPSG:26920", 26920),
        ("POINT (1 1)", pyproj.CRS("EPSG:26920").to_json(), 26920),
    ],
)
def test_st_setcrs_sedonadb(eng, geom, crs, expected_srid):
    eng = eng.create_or_skip()
    result = eng.execute_and_collect(f"SELECT ST_SetCrs({geom_or_null(geom)}, '{crs}')")
    df = eng.result_to_pandas(result)
    assert df.crs.to_epsg() == expected_srid


@pytest.mark.parametrize("eng", [SedonaDB])
def test_st_crs_sedonadb(eng):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        "SELECT ST_CRS(ST_SetCrs(ST_GeomFromText('POINT (1 1)'), 'EPSG:26920'))",
        '"EPSG:26920"',
    )
    eng.assert_query_result(
        "SELECT ST_CRS(ST_SetCrs(ST_GeomFromText('POINT (1 1)'), NULL))",
        None,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "dx", "dy", "expected"),
    [
        # Nulls
        (None, None, None, None),
        (None, 1.0, 2.0, None),
        ("POINT (0 1)", None, 2.0, None),
        ("POINT (0 1)", 1.0, None, None),
        ("POINT (0 1)", 1.0, 2.0, "POINT (1 3)"),  # Positives
        ("POINT (0 1)", -1.0, -2.0, "POINT (-1 -1)"),  # Negatives
        ("POINT (0 1)", 0.0, 0.0, "POINT (0 1)"),  # Zeroes
        ("POINT (0 1)", 1, 2, "POINT (1 3)"),  # Integers
        ("POINT Z (0 1 2)", 1.0, 2.0, "POINT Z (1 3 2)"),  # Z
        ("POINT M (0 1 2)", 1.0, 2.0, "POINT M (1 3 2)"),  # M
        ("POINT ZM (0 1 2 3)", 1.0, 2.0, "POINT ZM (1 3 2 3)"),  # ZM
        # Not points
        ("LINESTRING (0 1, 2 3)", 1.0, 2.0, "LINESTRING (1 3, 3 5)"),
        ("POLYGON ((0 0, 1 0, 0 1, 0 0))", 1.0, 2.0, "POLYGON ((1 2, 2 2, 1 3, 1 2))"),
        ("MULTIPOINT (0 1, 2 3)", 1.0, 2.0, "MULTIPOINT (1 3, 3 5)"),
        ("MULTILINESTRING ((0 1, 2 3))", 1.0, 2.0, "MULTILINESTRING ((1 3, 3 5))"),
        (
            "MULTIPOLYGON (((0 0, 1 0, 0 1, 0 0)))",
            1.0,
            2.0,
            "MULTIPOLYGON (((1 2, 2 2, 1 3, 1 2)))",
        ),
        (
            "GEOMETRYCOLLECTION (POINT (0 1))",
            1.0,
            2.0,
            "GEOMETRYCOLLECTION (POINT (1 3))",
        ),
        # WKT output of geoarrow-c is causing this (both correctly output
        # empties)
        ("POINT EMPTY", 1.0, 2.0, "POINT (nan nan)"),
        ("POINT Z EMPTY", 1.0, 2.0, "POINT Z (nan nan nan)"),
        ("LINESTRING EMPTY", 1.0, 2.0, "LINESTRING EMPTY"),
        ("POLYGON EMPTY", 1.0, 2.0, "POLYGON EMPTY"),
        ("MULTIPOINT EMPTY", 1.0, 2.0, "MULTIPOINT EMPTY"),
        ("MULTILINESTRING EMPTY", 1.0, 2.0, "MULTILINESTRING EMPTY"),
        ("MULTIPOLYGON EMPTY", 1.0, 2.0, "MULTIPOLYGON EMPTY"),
        ("GEOMETRYCOLLECTION EMPTY", 1.0, 2.0, "GEOMETRYCOLLECTION EMPTY"),
    ],
)
def test_st_translate(eng, geom, dx, dy, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Translate({geom_or_null(geom)}, {val_or_null(dx)}, {val_or_null(dy)})",
        expected,
    )
