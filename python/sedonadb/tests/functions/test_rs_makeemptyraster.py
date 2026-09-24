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

import sedonadb

from sedonadb.raster import Raster


def _raster(con, expr):
    table = con.sql(f"SELECT {expr} AS r").to_arrow_table()
    return Raster(table["r"], 0)


@pytest.mark.parametrize(
    ("expr", "size", "transform", "band_types"),
    [
        # Sedona Spark's cell-size form: square pixels, no skew
        (
            "RS_MakeEmptyRaster(2, 4, 3, 10.0, 20.0, 2.5)",
            (4, 3),
            (10.0, 2.5, 0.0, 20.0, 0.0, -2.5),
            ["float64", "float64"],
        ),
        (
            "RS_MakeEmptyRaster(1, 'B', 4, 3, 10, 20, 2)",
            (4, 3),
            (10.0, 2.0, 0.0, 20.0, 0.0, -2.0),
            ["uint8"],
        ),
        # Sedona Spark's affine form: scale and skew given outright
        (
            "RS_MakeEmptyRaster(1, 'I', 5, 4, 100.0, 200.0, 2.0, -3.0, 0.5, 0.25, 3857)",
            (5, 4),
            (100.0, 2.0, 0.5, 200.0, 0.25, -3.0),
            ["int32"],
        ),
        (
            "RS_MakeEmptyRaster(3, 6, 6, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 4326)",
            (6, 6),
            (0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
            ["float64", "float64", "float64"],
        ),
        # Extent form: the grid covers the geometry's envelope
        (
            "RS_MakeEmptyRaster(1, 'uint8', 400, 400, ST_MakeEnvelope(0, 0, 400, 400, 4326))",
            (400, 400),
            (0.0, 1.0, 0.0, 400.0, 0.0, -1.0),
            ["uint8"],
        ),
        (
            "RS_MakeEmptyRaster(2, 5, 4, ST_GeomFromText('POLYGON ((0 0, 10 0, 0 20, 0 0))', 'EPSG:3857'))",
            (5, 4),
            (0.0, 2.0, 0.0, 20.0, 0.0, -5.0),
            ["float64", "float64"],
        ),
        (
            "RS_MakeEmptyRaster(1, 2, 2, ST_MakeEnvelope(1, 1, 3, 5))",
            (2, 2),
            (1.0, 1.0, 0.0, 5.0, 0.0, -2.0),
            ["float64"],
        ),
        # No bands at all: a grid template
        (
            "RS_MakeEmptyRaster(0, 4, 3, 0.0, 0.0, 1.0)",
            (4, 3),
            (0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
            [],
        ),
    ],
)
def test_grid_shape_and_placement(con, expr, size, transform, band_types):
    raster = _raster(con, expr)
    assert (raster.width, raster.height) == size
    # GDAL order: upper-left x, scale x, skew x, upper-left y, skew y, scale y
    assert tuple(raster.transform) == transform
    assert [band.data_type for band in raster.bands] == band_types


@pytest.mark.parametrize(
    ("expr", "srid"),
    [
        ("RS_MakeEmptyRaster(1, 2, 2, 0.0, 0.0, 1.0)", 0),
        ("RS_MakeEmptyRaster(1, 2, 2, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 3857)", 3857),
        ("RS_MakeEmptyRaster(1, 2, 2, ST_MakeEnvelope(0, 0, 4, 4, 4326))", 4326),
        ("RS_MakeEmptyRaster(1, 2, 2, ST_MakeEnvelope(0, 0, 4, 4))", 0),
    ],
)
def test_srid_comes_from_the_placement(con, expr, srid):
    got = con.sql(f"SELECT RS_SRID({expr})").to_arrow_table().column(0)[0].as_py()
    assert got == srid


def test_band_pixel_type_name_round_trips(con):
    """RS_BandPixelType's own spelling is accepted back as a type argument."""
    name = (
        con.sql("SELECT RS_BandPixelType(RS_MakeEmptyRaster(1, 'B', 2, 2, 0, 0, 1), 1)")
        .to_arrow_table()
        .column(0)[0]
        .as_py()
    )
    assert name == "UNSIGNED_8BITS"
    raster = _raster(con, f"RS_MakeEmptyRaster(1, '{name}', 2, 2, 0.0, 0.0, 1.0)")
    assert [band.data_type for band in raster.bands] == ["uint8"]


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


def test_geography_extent(con):
    """A geography extent is accepted, and its envelope follows spherical edges
    via the session's geography bounder rather than a planar coordinate scan."""
    if "s2geography" not in sedonadb.__features__:
        pytest.skip("Geography bounds require a build with feature s2geography")

    wkt = "POLYGON ((0 0, 10 0, 10 20, 0 20, 0 0))"
    raster = _raster(con, f"RS_MakeEmptyRaster(1, 4, 2, ST_GeogFromText('{wkt}'))")
    ulx, scale_x, _, uly, _, scale_y = raster.transform

    # Geodesic edges bow away from the straight lines between the vertices, so
    # the spherical envelope contains the planar one; assert that containment
    # rather than pinning S2's exact bulge.
    assert ulx <= 0.0
    assert uly >= 20.0
    assert ulx + 4 * scale_x >= 10.0
    assert uly + 2 * scale_y <= 0.0


def test_zero_bands_ignores_the_per_band_size_limit(con):
    """A bandless template is only grid metadata, so it allocates no pixel
    buffer and a grid too large for one band is still valid."""
    huge = "100000, 100000, 0.0, 0.0, 1.0"
    assert (
        con.sql(f"SELECT RS_MakeEmptyRaster(0, {huge}) IS NOT NULL")
        .to_arrow_table()
        .column(0)[0]
        .as_py()
    )
    # The same grid with a band is still rejected.
    with pytest.raises(Exception, match="4 GiB per-band limit"):
        con.sql(f"SELECT RS_MakeEmptyRaster(1, {huge})").to_arrow_table()


def test_all_null_extent_column_yields_null_rasters(con):
    """An all-null extent column is typed NULL rather than geometry; each row
    is a null geometry, as a NULL extent scalar already was."""
    got = (
        con.sql(
            """
        SELECT RS_MakeEmptyRaster(1, 2, 2, extent) IS NULL AS is_null
        FROM (VALUES (NULL), (NULL)) AS t(extent)
        """
        )
        .to_arrow_table()
        .column(0)
        .to_pylist()
    )
    assert got == [True, True]


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
