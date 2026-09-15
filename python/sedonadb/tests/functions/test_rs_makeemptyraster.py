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

"""RS_MakeEmptyRaster: a raster from nothing but a grid definition.

The Rust unit tests pin the exact output (transform, CRS, band layout) per
kernel; these tests cover the SQL surface: literal typing, the geometry
extent form fed by other SQL functions, table-driven grids, and the customer
flow of rasterizing onto the grid.
"""

import numpy as np
import pytest

from sedonadb.raster import Raster

GRID_META = """
SELECT RS_Width(r), RS_Height(r), RS_NumBands(r), RS_BandPixelType(r, 1),
       RS_UpperLeftX(r), RS_UpperLeftY(r), RS_ScaleX(r), RS_ScaleY(r),
       RS_SkewX(r), RS_SkewY(r), RS_SRID(r)
FROM (SELECT {expr} AS r)
"""


def _meta(con, expr):
    return tuple(
        con.sql(GRID_META.format(expr=expr)).to_arrow_table().to_pylist()[0].values()
    )


def _raster(con, expr):
    table = con.sql(f"SELECT {expr} AS r").to_arrow_table()
    return Raster(table["r"], 0)


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        # Sedona Spark's cell-size form: square pixels, no skew, no CRS
        (
            "RS_MakeEmptyRaster(2, 4, 3, 10.0, 20.0, 2.5)",
            (4, 3, 2, "REAL_64BITS", 10.0, 20.0, 2.5, -2.5, 0.0, 0.0, 0),
        ),
        (
            "RS_MakeEmptyRaster(1, 'B', 4, 3, 10, 20, 2)",
            (4, 3, 1, "UNSIGNED_8BITS", 10.0, 20.0, 2.0, -2.0, 0.0, 0.0, 0),
        ),
        # Sedona Spark's affine form
        (
            "RS_MakeEmptyRaster(1, 'I', 5, 4, 100.0, 200.0, 2.0, -3.0, 0.5, 0.25, 3857)",
            (5, 4, 1, "SIGNED_32BITS", 100.0, 200.0, 2.0, -3.0, 0.5, 0.25, 3857),
        ),
        (
            "RS_MakeEmptyRaster(3, 6, 6, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 4326)",
            (6, 6, 3, "REAL_64BITS", 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 4326),
        ),
        # Extent form: ncol, nrow, bbox (+ CRS) — the grid covers the envelope
        (
            "RS_MakeEmptyRaster(1, 'uint8', 400, 400, ST_MakeEnvelope(0, 0, 400, 400, 4326))",
            (400, 400, 1, "UNSIGNED_8BITS", 0.0, 400.0, 1.0, -1.0, 0.0, 0.0, 4326),
        ),
        (
            "RS_MakeEmptyRaster(2, 5, 4, ST_GeomFromText('POLYGON ((0 0, 10 0, 0 20, 0 0))', 'EPSG:3857'))",
            (5, 4, 2, "REAL_64BITS", 0.0, 20.0, 2.0, -5.0, 0.0, 0.0, 3857),
        ),
        (
            "RS_MakeEmptyRaster(1, 2, 2, ST_MakeEnvelope(1, 1, 3, 5))",
            (2, 2, 1, "REAL_64BITS", 1.0, 5.0, 1.0, -2.0, 0.0, 0.0, 0),
        ),
        # No bands at all: a grid template
        (
            "RS_MakeEmptyRaster(0, 4, 3, 0.0, 0.0, 1.0)",
            (4, 3, 0, None, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0),
        ),
    ],
)
def test_grid_metadata(con, expr, expected):
    got = _meta(con, expr)
    if expected[3] is None:
        # RS_BandPixelType errors on a bandless raster; compare the rest
        got_no_band = tuple(
            con.sql(GRID_META.replace("RS_BandPixelType(r, 1),", "").format(expr=expr))
            .to_arrow_table()
            .to_pylist()[0]
            .values()
        )
        assert got_no_band == expected[:3] + expected[4:]
    else:
        assert got == expected


@pytest.mark.parametrize(
    ("type_name", "dtype"),
    [
        ("uint8", np.uint8),
        ("int16", np.int16),
        ("float32", np.float32),
        ("D", np.float64),
        ("US", np.uint16),
        ("SIGNED_32BITS", np.int32),
    ],
)
def test_bands_are_zero_filled(con, type_name, dtype):
    raster = _raster(con, f"RS_MakeEmptyRaster(2, '{type_name}', 7, 3, 0.0, 0.0, 1.0)")
    assert len(raster.bands) == 2
    for band in raster.bands:
        pixels = band.to_numpy()
        assert pixels.shape == (3, 7)
        assert pixels.dtype == dtype
        assert not pixels.any()
        assert band.nodata is None


def test_extent_from_another_raster(con):
    """RS_Envelope carries the raster's CRS as an item CRS; the new grid
    inherits it and covers exactly the envelope."""
    row = (
        con.sql(
            """
        WITH src AS (SELECT RS_Envelope(RS_Example()) AS env),
             g AS (SELECT RS_MakeEmptyRaster(1, 16, 8, env) AS r, env FROM src)
        SELECT RS_Width(r), RS_Height(r), RS_SRID(r),
               RS_UpperLeftX(r) = ST_XMin(env), RS_UpperLeftY(r) = ST_YMax(env),
               RS_UpperLeftX(r) + 16 * RS_ScaleX(r) = ST_XMax(env),
               RS_UpperLeftY(r) + 8 * RS_ScaleY(r) = ST_YMin(env)
        FROM g
        """
        )
        .to_arrow_table()
        .to_pylist()[0]
    )
    assert tuple(row.values()) == (16, 8, 4326, True, True, True, True)


def test_table_driven_grids(con):
    got = (
        con.sql(
            """
        SELECT RS_Width(r), RS_Height(r), RS_ScaleX(r)
        FROM (
          SELECT RS_MakeEmptyRaster(1, w, h, ST_MakeEnvelope(0, 0, 100, 50)) AS r
          FROM (VALUES (4, 2), (10, 5), (NULL, 1)) AS t(w, h)
        )
        """
        )
        .to_arrow_table()
        .to_pylist()
    )
    assert [tuple(r.values()) for r in got] == [
        (4, 2, 25.0),
        (10, 5, 10.0),
        (None, None, None),
    ]


def test_null_arguments_yield_null(con):
    for expr in [
        "RS_MakeEmptyRaster(1, CAST(NULL AS VARCHAR), 2, 2, 0.0, 0.0, 1.0)",
        "RS_MakeEmptyRaster(1, 2, 2, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, CAST(NULL AS INT))",
        "RS_MakeEmptyRaster(1, 2, 2, ST_GeomFromText(NULL))",
    ]:
        assert con.sql(f"SELECT {expr} IS NULL").to_arrow_table().column(0)[0].as_py()


@pytest.mark.parametrize(
    ("expr", "message"),
    [
        ("RS_MakeEmptyRaster(-1, 2, 2, 0.0, 0.0, 1.0)", "num_bands must be >= 0"),
        (
            "RS_MakeEmptyRaster(1, 0, 2, 0.0, 0.0, 1.0)",
            "width and height must be positive",
        ),
        (
            "RS_MakeEmptyRaster(1, 'complex128', 2, 2, 0.0, 0.0, 1.0)",
            "Unsupported pixelType",
        ),
        (
            "RS_MakeEmptyRaster(1, 2, 2, ST_GeomFromText('POINT (1 1)'))",
            "positive width and height",
        ),
        (
            "RS_MakeEmptyRaster(1, 2, 2, ST_GeomFromText('POLYGON EMPTY'))",
            "extent geometry is empty",
        ),
    ],
)
def test_invalid_arguments(con, expr, message):
    with pytest.raises(Exception, match=message):
        con.sql(f"SELECT {expr}").to_arrow_table()


def test_rasterize_onto_grid(con):
    """The motivating flow: define the grid, then burn a geometry onto it."""
    grid = (
        "RS_MakeEmptyRaster(1, 'uint8', 40, 40, ST_MakeEnvelope(0, 0, 400, 400, 4326))"
    )
    poly = "ST_SetSRID(ST_GeomFromText('POLYGON ((0 400, 100 400, 100 300, 0 300, 0 400))'), 4326)"
    raster = _raster(
        con, f"RS_AsRaster({poly}, {grid}, 'uint8', false, 1.0, 0.0, false)"
    )
    pixels = raster.bands[0].to_numpy()
    assert pixels.shape == (40, 40)
    assert tuple(raster.transform) == (0.0, 10.0, 0.0, 400.0, 0.0, -10.0)
    # The polygon covers the top-left 10x10 block of 10-unit pixels
    assert pixels[:10, :10].all()
    assert pixels.sum() == 100
