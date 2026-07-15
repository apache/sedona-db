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
import numpy as np

from sedonadb.testing import SedonaDB
from sedonadb.raster import Raster


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("RS_NumBands(RS_Example())", 3),
        ("RS_Width(RS_Example())", 64),
        ("RS_Height(RS_Example())", 32),
        ("RS_BandPixelType(RS_Example(), 1)", "UNSIGNED_8BITS"),
        ("RS_BandNoDataValue(RS_Example(), 1)", 127.0),
        ("RS_ScaleX(RS_Example())", 2.0),
        ("RS_ScaleY(RS_Example())", 2.0),
        ("RS_SkewX(RS_Example())", 1.0),
        ("RS_SkewY(RS_Example())", 1.0),
        ("RS_UpperLeftX(RS_Example())", 43.08),
        ("RS_UpperLeftY(RS_Example())", 79.07),
    ],
)
def test_rs_function(expr, expected):
    eng = SedonaDB()
    eng.assert_query_result(f"SELECT {expr}", expected)


# EPSG:3857 as WKT (carries an embedded EPSG authority) and a bespoke Lambert
# Conformal Conic WKT with no authority code anywhere.
WKT_3857 = (
    'PROJCS["WGS 84 / Pseudo-Mercator",GEOGCS["WGS 84",DATUM["WGS_1984",'
    'SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],'
    'AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
    'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
    'AUTHORITY["EPSG","4326"]],PROJECTION["Mercator_1SP"],'
    'PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],'
    'PARAMETER["false_easting",0],PARAMETER["false_northing",0],'
    'UNIT["metre",1,AUTHORITY["EPSG","9001"]],AUTHORITY["EPSG","3857"]]'
)
WKT_LCC_NO_AUTHORITY = (
    'PROJCS["Custom LCC",GEOGCS["WGS 84",DATUM["WGS_1984",'
    'SPHEROID["WGS 84",6378137,298.257223563]]],'
    'PROJECTION["Lambert_Conformal_Conic_2SP"],'
    'PARAMETER["standard_parallel_1",33],PARAMETER["standard_parallel_2",45],'
    'PARAMETER["latitude_of_origin",39],PARAMETER["central_meridian",-96],'
    'UNIT["metre",1]]'
)


# WKT1/WKT2 CRS strings round-trip through RS_SetCRS/RS_CRS unchanged, whether or
# not they carry an embedded authority.
@pytest.mark.parametrize("wkt", [WKT_3857, WKT_LCC_NO_AUTHORITY])
def test_rs_setcrs_wkt_roundtrips(wkt):
    eng = SedonaDB()
    eng.assert_query_result(f"SELECT RS_CRS(RS_SetCRS(RS_Example(), '{wkt}'))", wkt)


def test_rs_srid_from_wkt():
    """A WKT carrying an EPSG authority resolves to that SRID."""
    eng = SedonaDB()
    eng.assert_query_result(
        f"SELECT RS_SRID(RS_SetCRS(RS_Example(), '{WKT_3857}'))", 3857
    )


def test_rs_srid_from_authorityless_wkt_errors(con):
    """A WKT with no authority code anywhere has no SRID to extract."""
    with pytest.raises(Exception, match="SRID"):
        con.sql(
            f"SELECT RS_SRID(RS_SetCRS(RS_Example(), '{WKT_LCC_NO_AUTHORITY}'))"
        ).to_arrow_table()


def test_rs_ensureloaded(con, sedona_testing):
    path = sedona_testing / "data/raster/sentinel2.tif"
    t = con.sql("SELECT RS_FromPath($1) AS raster", params=(str(path),))
    tab = t.select(raster=t.raster.funcs.rs_ensureloaded()).to_arrow_table()
    r = tab["raster"][0].as_py()
    assert r.height == 512
    assert r.width == 512

    assert len(r.bands) == 1
    b = r.bands[0]
    assert b.shape == (512, 512)
    arr = b.to_numpy()
    assert arr.shape == (512, 512)
    assert arr.dtype == "uint16"
    assert arr[0, 0] == 2324


# Point sampling. RS_Example fills band `b` with the constant value `b`, except
# the top-left pixel which is set to the nodata value (127). (74.58, 110.57) is
# the centroid of pixel (10, 10) (0-based) in the raster's OGC:CRS84 space; the
# point and raster share a CRS so no reprojection happens. A point far outside
# the footprint yields NULL. (The `needs_pixels` -> RS_EnsureLoaded planner path
# is covered against a real OutDb raster by `test_rs_ensureloaded`.)
@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (
            "RS_Value(RS_Example(), ST_SetCRS(ST_Point(74.58, 110.57), 'OGC:CRS84'))",
            1.0,
        ),
        (
            "RS_Value(RS_Example(), ST_SetCRS(ST_Point(74.58, 110.57), 'OGC:CRS84'), 2)",
            2.0,
        ),
        (
            "RS_Value(RS_Example(), ST_SetCRS(ST_Point(74.58, 110.57), 'OGC:CRS84'), 3)",
            3.0,
        ),
        ("RS_Value(RS_Example(), ST_SetCRS(ST_Point(0.0, 0.0), 'OGC:CRS84'))", None),
        # POINT EMPTY has no location to sample -> NULL (not an error).
        (
            "RS_Value(RS_Example(), ST_SetCRS(ST_GeomFromText('POINT EMPTY'), 'OGC:CRS84'))",
            None,
        ),
    ],
)
def test_rs_value_point(expr, expected):
    SedonaDB().assert_query_result(f"SELECT {expr}", expected)


def test_rs_value_matches_rasterio(con):
    """Cross-check RS_Value against rasterio on a random raster.

    Builds an in-memory raster from a random numpy array with a known
    geotransform and no CRS (so neither engine reprojects), then samples a dense
    set of points and asserts RS_Value returns exactly what rasterio reads at the
    same world coordinates. Points cover every pixel center plus four off-center
    positions per pixel (toward the corners, kept inside the pixel to avoid floor
    ambiguity at exact boundaries) and a batch of random interior points.
    """
    import numpy as np
    import pandas as pd

    pytest.importorskip("rasterio")
    from rasterio.io import MemoryFile
    from rasterio.transform import Affine

    from sedonadb.raster import Raster

    rng = np.random.default_rng(42)
    height, width = 7, 5
    data = rng.random((height, width)) * 1000.0

    # GDAL-order geotransform: origin (100, 500), 2-wide pixels, -3 tall
    # (north-up), no skew. Shared verbatim by both engines.
    gdal_transform = (100.0, 2.0, 0.0, 500.0, 0.0, -3.0)
    affine = Affine.from_gdal(*gdal_transform)

    # Sample points in pixel space (col_frac, row_frac).
    pixel_points = []
    for row in range(height):
        for col in range(width):
            for du, dv in [
                (0.5, 0.5),
                (0.25, 0.25),
                (0.75, 0.75),
                (0.25, 0.75),
                (0.75, 0.25),
            ]:
                pixel_points.append((col + du, row + dv))
    n_random = 150
    rand_cols = rng.integers(0, width, n_random)
    rand_rows = rng.integers(0, height, n_random)
    pixel_points.extend(
        zip(
            rand_cols + rng.uniform(0.1, 0.9, n_random),
            rand_rows + rng.uniform(0.1, 0.9, n_random),
        )
    )

    # Map pixel-space positions to world coordinates via the shared affine.
    xs, ys = zip(*(affine * (u, v) for u, v in pixel_points))

    # rasterio reference: a real GDAL read of the same array (no CRS).
    with MemoryFile() as mem:
        with mem.open(
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="float64",
            transform=affine,
        ) as dst:
            dst.write(data, 1)
        with mem.open() as src:
            expected = [vals[0] for vals in src.sample(list(zip(xs, ys)))]

    # sedonadb: sample the same points via RS_Value over a scalar raster.
    raster = Raster.from_numpy(data, transform=gdal_transform)
    pts = con.create_data_frame(pd.DataFrame({"idx": range(len(xs)), "x": xs, "y": ys}))
    view = "test_rs_value_matches_rasterio_pts"
    pts.to_view(view)
    try:
        got = (
            con.sql(
                f"SELECT RS_Value($1, ST_Point(x, y)) AS v FROM {view} ORDER BY idx",
                params=(raster,),
            )
            .to_arrow_table()["v"]
            .to_pylist()
        )
    finally:
        con.drop_view(view)

    assert got == pytest.approx(expected)


def test_rs_setgeoreference_roundtrips_with_getter():
    # RS_GeoReference emits scaleX, skewY, skewX, scaleY, upperLeftX, upperLeftY;
    # RS_SetGeoReference accepts the same six values back (GDAL order).
    eng = SedonaDB()
    eng.assert_query_result(
        "SELECT RS_GeoReference(RS_SetGeoReference(RS_Example(), '2 0 0 -3 100 200'))",
        "2.0000000000\n0.0000000000\n0.0000000000\n-3.0000000000\n100.0000000000\n200.0000000000",
    )


def test_rs_setgeoreference_esri_shifts_to_corner():
    # ESRI upper-left is the pixel center; the stored (GDAL) upper-left is the
    # corner: 101 - 2*0.5 = 100 and 198.5 - (-3)*0.5 = 200.
    eng = SedonaDB()
    eng.assert_query_result(
        "SELECT RS_GeoReference(RS_SetGeoReference(RS_Example(), '2 0 0 -3 101 198.5', 'ESRI'))",
        "2.0000000000\n0.0000000000\n0.0000000000\n-3.0000000000\n100.0000000000\n200.0000000000",
    )


def test_rs_setgeoreference_esri_skewed_roundtrips():
    # The ESRI center shift maps through the full affine (scale and skew
    # halves), so a skewed georeference round-trips exactly through the
    # setter/getter pair in the ESRI convention.
    eng = SedonaDB()
    eng.assert_query_result(
        "SELECT RS_GeoReference(RS_SetGeoReference(RS_Example(), '2 0.5 0.25 -3 100 200', 'ESRI'), 'ESRI')",
        "2.0000000000\n0.5000000000\n0.2500000000\n-3.0000000000\n100.0000000000\n200.0000000000",
    )


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        # Three-arg form targets a specific band; read it back with the getter.
        ("RS_BandNoDataValue(RS_SetBandNoDataValue(RS_Example(), 1, 0), 1)", 0.0),
        ("RS_BandNoDataValue(RS_SetBandNoDataValue(RS_Example(), 2, 255), 2)", 255.0),
        # A null nodata value yields a null raster, so the getter returns null.
        (
            "RS_BandNoDataValue(RS_SetBandNoDataValue(RS_Example(), CAST(NULL AS DOUBLE)), 1)",
            None,
        ),
    ],
)
def test_rs_setbandnodatavalue(expr, expected):
    SedonaDB().assert_query_result(f"SELECT {expr}", expected)


def test_rs_setbandnodatavalue_two_arg_requires_single_band():
    # The 2-arg form is ambiguous on a multiband raster (RS_Example has multiple
    # bands), so it errors rather than silently setting only band 1.
    with pytest.raises(Exception, match="specify which band"):
        SedonaDB().assert_query_result(
            "SELECT RS_SetBandNoDataValue(RS_Example(), 0)", None
        )


def _rs_as_raster_sql(
    pixel_type, all_touched, burn_value, nodata_value, use_geometry_extent
):
    return f"""
        WITH src AS (SELECT RS_FromPath($1) AS raster)
        SELECT RS_AsRaster(
            ST_GeomFromText('POLYGON((2 8, 2 5, 5 5, 5 8, 2 8))', 'EPSG:4326'),
            raster,
            '{pixel_type}',
            {str(all_touched).upper()},
            {burn_value},
            {nodata_value},
            {str(use_geometry_extent).upper()}
        ) AS raster
        FROM src
    """


@pytest.mark.parametrize(
    ("all_touched", "use_geometry_extent"),
    [(False, False), (False, True), (True, False)],
)
def test_rs_as_raster_matches_rasterio(
    con, sedona_testing, all_touched, use_geometry_extent
):
    pytest.importorskip("rasterio")
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    from shapely import wkt

    path = sedona_testing / "data/raster/test4.tiff"
    transform = (0.0, 1.0, 0.0, 10.0, 0.0, -1.0)

    table = con.sql(
        _rs_as_raster_sql("float64", all_touched, 7.0, 0.0, use_geometry_extent),
        params=(str(path),),
    ).to_arrow_table()
    result = Raster(table["raster"], 0)

    got = result.bands[0].to_numpy()
    geom = wkt.loads("POLYGON((2 8, 2 5, 5 5, 5 8, 2 8))")

    if use_geometry_extent:
        expected_shape = (3, 3)
        expected_transform = (2.0, 1.0, 0.0, 8.0, 0.0, -1.0)
    else:
        expected_shape = (10, 10)
        expected_transform = transform

    expected = rasterize(
        [(geom, 7.0)],
        out_shape=expected_shape,
        transform=Affine.from_gdal(*expected_transform),
        all_touched=all_touched,
        fill=0.0,
        dtype="float64",
    )

    assert tuple(result.transform) == expected_transform
    assert result.width == expected_shape[1]
    assert result.height == expected_shape[0]
    np.testing.assert_array_equal(got, expected)


def test_rs_as_raster_all_touched_changes_pixels(con, sedona_testing):
    import numpy as np

    pytest.importorskip("rasterio")
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    from shapely import wkt

    path = sedona_testing / "data/raster/test4.tiff"
    transform = (0.0, 1.0, 0.0, 10.0, 0.0, -1.0)
    geom = "POLYGON((1.9 8.9, 1.9 6.1, 3.1 6.1, 3.1 8.9, 1.9 8.9))"

    false_result = (
        con.sql(
            f"WITH src AS (SELECT RS_FromPath($1) AS raster) SELECT RS_AsRaster(ST_GeomFromText('{geom}', 'EPSG:4326'), raster, 'uint8', FALSE, 1, 0, FALSE) AS raster FROM src",
            params=(str(path),),
        )
        .to_arrow_table()["raster"][0]
        .as_py()
    )
    true_result = (
        con.sql(
            f"WITH src AS (SELECT RS_FromPath($1) AS raster) SELECT RS_AsRaster(ST_GeomFromText('{geom}', 'EPSG:4326'), raster, 'uint8', TRUE, 1, 0, FALSE) AS raster FROM src",
            params=(str(path),),
        )
        .to_arrow_table()["raster"][0]
        .as_py()
    )

    false_pixels = false_result.bands[0].to_numpy()
    true_pixels = true_result.bands[0].to_numpy()

    geom = wkt.loads(geom)
    expected_false = rasterize(
        [(geom, 1)],
        out_shape=(10, 10),
        transform=Affine.from_gdal(*transform),
        all_touched=False,
        fill=0,
        dtype="uint8",
    )
    expected_true = rasterize(
        [(geom, 1)],
        out_shape=(10, 10),
        transform=Affine.from_gdal(*transform),
        all_touched=True,
        fill=0,
        dtype="uint8",
    )

    np.testing.assert_array_equal(false_pixels, expected_false)
    np.testing.assert_array_equal(true_pixels, expected_true)
    assert np.count_nonzero(true_pixels) >= np.count_nonzero(false_pixels)
    assert not np.array_equal(true_pixels, false_pixels)


def test_rs_as_raster_rejects_fractional_integer_nodata(con, sedona_testing):
    path = sedona_testing / "data/raster/test4.tiff"

    with pytest.raises(Exception, match="initial fill value must be an integer"):
        con.sql(
            """
            WITH src AS (SELECT RS_FromPath($1) AS raster)
            SELECT RS_AsRaster(
                ST_GeomFromText('POLYGON((0 2, 0 0, 2 0, 2 2, 0 2))', 'EPSG:4326'),
                raster,
                'uint8',
                FALSE,
                1,
                1.5,
                FALSE
            )
            FROM src
            """,
            params=(str(path),),
        ).to_arrow_table()


def test_rs_as_raster_sets_output_nodata(con, sedona_testing):
    import numpy as np

    path = sedona_testing / "data/raster/test4.tiff"
    tab = con.sql(
        """
        WITH src AS (SELECT RS_FromPath($1) AS raster)
        SELECT
            RS_AsRaster(
                ST_GeomFromText('POLYGON((0 10, 0 9, 1 9, 1 10, 0 10))', 'EPSG:4326'),
                raster,
                'uint8',
                FALSE,
                5,
                9,
                FALSE
            ) AS raster,
            RS_BandNoDataValue(
                RS_AsRaster(
                    ST_GeomFromText('POLYGON((0 10, 0 9, 1 9, 1 10, 0 10))', 'EPSG:4326'),
                    raster,
                    'uint8',
                    FALSE,
                    5,
                    9,
                    FALSE
                ),
                1
            ) AS nodata
        FROM src
        """,
        params=(str(path),),
    ).to_arrow_table()

    assert tab["nodata"][0].as_py() == 9.0
    raster = tab["raster"][0].as_py()
    expected = np.full((10, 10), 9, dtype="uint8")
    expected[0, 0] = 5
    np.testing.assert_array_equal(raster.bands[0].to_numpy(), expected)
