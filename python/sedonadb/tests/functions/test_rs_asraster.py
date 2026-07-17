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

"""RS_AsRaster parity.

The rasterio comparator is `rasterio.features.rasterize` on the same grid,
filling outside the geometry with the subject's policy (SedonaDB
initializes the grid with the nodata value, 0 when none is given). Two
Sedona Spark deviations are on the ledger rather than shrinking the matrix:
Spark burns outside pixels to 0 regardless of nodata (metadata-only
nodata), and its geotools/JAI rasterizer drops some center-inside pixels
along diagonal edges under the centroid rule where GDAL (SedonaDB,
rasterio) burns them. Geometries stay inside the reference raster's extent;
behavior for overhanging geometry envelopes is not compared here.
"""

import pytest

from sedonadb.raster_testing import (
    Deviation,
    SedonaSpark,
    assert_decoded_equal,
    expect_deviations,
    random_raster_data,
    write_geotiff,
)

pytest.importorskip("rasterio")
pytest.importorskip("shapely")

# The band types both dialects can express (Sedona Spark has no int8/64-bit
# integer band types).
DTYPES = ["uint8", "uint16", "int16", "int32", "float32", "float64"]

# GDAL-order geotransform: origin (100, 500), 2-wide by 3-tall north-up
# pixels; with a 7x6 raster the extent is x in [100, 114], y in [482, 500].
GDAL_TRANSFORM = (100.0, 2.0, 0.0, 500.0, 0.0, -3.0)
HEIGHT, WIDTH = 6, 7
GEOM_RECT = (
    "POLYGON ((102.6 495.8, 109.3 495.8, 109.3 485.9, 102.6 485.9, 102.6 495.8))"
)
# Diagonal edges make all_touched change the selection.
GEOM_TRIANGLE = "POLYGON ((101.3 498.6, 112.4 496.9, 104.2 483.7, 101.3 498.6))"

DEVIATIONS = [
    Deviation(
        SedonaSpark,
        "as_raster",
        matches=lambda p: p.get("wkt") == GEOM_TRIANGLE and not p.get("all_touched"),
        reason="geotools/JAI drops some center-inside pixels along diagonal "
        "edges under the centroid rule; GDAL burns every center-inside pixel",
    ),
    Deviation(
        SedonaSpark,
        "as_raster",
        matches=lambda p: p.get("nodata") not in (None, 0.0),
        reason="Sedona Spark burns outside pixels to 0 and records nodata as "
        "band metadata only; SedonaDB initializes the grid with the nodata "
        "value",
    ),
]


@pytest.fixture()
def tiff(tmp_path):
    path = tmp_path / "asraster_reference.tif"
    write_geotiff(
        path,
        random_raster_data("uint8", bands=1, height=HEIGHT, width=WIDTH),
        gdal_transform=GDAL_TRANSFORM,
    )
    return path


@pytest.mark.parametrize("dtype", DTYPES)
def test_rs_asraster_dtypes_match_comparators(
    subject, comparator, request, tiff, dtype
):
    """Burn value 7 into the geometry's grid-snapped envelope for every band
    type both dialects support."""
    expect_deviations(request, comparator, "as_raster", DEVIATIONS)
    kwargs = dict(burn_value=7.0, nodata=0.0, use_geometry_extent=True)
    got = subject.as_raster(GEOM_RECT, tiff, dtype, **kwargs)
    expected = comparator.as_raster(GEOM_RECT, tiff, dtype, **kwargs)
    assert_decoded_equal(got, expected, context=dtype)


@pytest.mark.parametrize(
    ("wkt", "all_touched", "use_geometry_extent", "nodata"),
    [
        (GEOM_RECT, False, True, 0.0),
        (GEOM_RECT, False, False, 0.0),
        (GEOM_RECT, True, True, 0.0),
        # The nodata-9 rows need pixels outside the geometry in the output
        # (that's where the fill policies diverge): the full reference grid,
        # and the triangle's cropped envelope, have them; the rect's cropped
        # envelope is fully covered and does not.
        (GEOM_RECT, False, False, 9.0),
        (GEOM_TRIANGLE, True, True, 9.0),
        (GEOM_TRIANGLE, False, True, 0.0),
        (GEOM_TRIANGLE, True, True, 0.0),
        (GEOM_TRIANGLE, True, False, 0.0),
    ],
    ids=[
        "rect-centroid-cropped",
        "rect-centroid-full",
        "rect-touched-cropped",
        "rect-centroid-full-nodata9",
        "triangle-touched-cropped-nodata9",
        "triangle-centroid-cropped",
        "triangle-touched-cropped",
        "triangle-touched-full",
    ],
)
def test_rs_asraster_options_match_comparators(
    subject, comparator, request, tiff, wkt, all_touched, use_geometry_extent, nodata
):
    """all_touched toggles the selection rule, use_geometry_extent toggles
    between the snapped geometry envelope and the full reference grid, and a
    nonzero nodata exercises the subject's nodata-fill policy. The
    triangle-centroid and nodata-9 rows are on the Sedona Spark deviation
    ledger."""
    expect_deviations(request, comparator, "as_raster", DEVIATIONS)
    kwargs = dict(
        all_touched=all_touched,
        burn_value=7.0,
        nodata=nodata,
        use_geometry_extent=use_geometry_extent,
    )
    got = subject.as_raster(wkt, tiff, "uint8", **kwargs)
    expected = comparator.as_raster(wkt, tiff, "uint8", **kwargs)
    assert_decoded_equal(got, expected, context=(wkt, all_touched, use_geometry_extent))


def test_rs_asraster_without_nodata(subject, comparator, tiff):
    """No nodata argument: every engine burns into zeros and leaves the
    output band without a nodata value."""
    got = subject.as_raster(GEOM_RECT, tiff, "uint8", burn_value=7.0)
    expected = comparator.as_raster(GEOM_RECT, tiff, "uint8", burn_value=7.0)
    assert_decoded_equal(got, expected)
    assert got.nodata == [None]
