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
Spark burns outside pixels to 0 regardless of nodata (metadata-only nodata,
apache/sedona#3112), and its scanline rasterizer mis-places x-intercepts on
non-square pixels so some center-inside pixels along diagonal edges are
dropped under the centroid rule where GDAL (SedonaDB, rasterio) burns them
(apache/sedona#3111). Geometries stay inside the reference raster's extent;
behavior for overhanging geometry envelopes is not compared here.
"""

import random

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
shapely = pytest.importorskip("shapely")

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
        reason="Sedona's scanline rasterizer mis-places x-intercepts on "
        "non-square pixels and drops some center-inside pixels along "
        "diagonal edges; GDAL burns every center-inside pixel "
        "(https://github.com/apache/sedona/issues/3111)",
    ),
    Deviation(
        SedonaSpark,
        "as_raster",
        matches=lambda p: p.get("nodata") not in (None, 0.0),
        reason="Sedona Spark burns outside pixels to 0 and records nodata as "
        "band metadata only; SedonaDB initializes the grid with the nodata "
        "value (https://github.com/apache/sedona/issues/3112)",
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


def _fuzz_cases(count=40, seed=31113):
    """Seeded random polygons over anisotropic north-up and south-up grids.

    The fixed seed makes the corpus deterministic, so a failure reproduces
    from the case id alone. Pixel width and height are drawn independently
    to exercise the non-square aspect ratios where rasterizer arithmetic
    errors hide (square unit grids make pixel-space and world-space slopes
    coincide, see apache/sedona#3111).
    """
    rng = random.Random(seed)
    cases = []
    while len(cases) < count:
        width, height = rng.randint(4, 12), rng.randint(4, 12)
        scale_x = round(rng.uniform(0.3, 5.0), 3)
        scale_y = round(rng.uniform(0.3, 5.0), 3) * rng.choice([-1, 1])
        upper_left_x = round(rng.uniform(-1000, 1000), 2)
        upper_left_y = round(rng.uniform(-1000, 1000), 2)
        xs = sorted([upper_left_x, upper_left_x + width * scale_x])
        ys = sorted([upper_left_y, upper_left_y + height * scale_y])
        margin_x = (xs[1] - xs[0]) * 0.05
        margin_y = (ys[1] - ys[0]) * 0.05

        def random_point():
            return (
                round(rng.uniform(xs[0] + margin_x, xs[1] - margin_x), 3),
                round(rng.uniform(ys[0] + margin_y, ys[1] - margin_y), 3),
            )

        num_points = rng.choice([3, 3, 4, 5])
        for _ in range(50):
            candidate = shapely.Polygon(
                [random_point() for _ in range(num_points)]
            ).buffer(0)
            grid_area = (xs[1] - xs[0]) * (ys[1] - ys[0])
            if candidate.geom_type == "Polygon" and candidate.area > grid_area * 0.02:
                cases.append(
                    (
                        len(cases),
                        width,
                        height,
                        (upper_left_x, scale_x, 0.0, upper_left_y, 0.0, scale_y),
                        candidate.wkt,
                    )
                )
                break
    return cases


def test_rs_asraster_fuzz_matches_comparators(subject, comparator, tmp_path):
    """Centroid-rule burns over the seeded random corpus must match on every
    grid. Only the centroid rule is fuzzed: allTouched boundary selection
    differs between rasterizers by design and is pinned by the deterministic
    cases above."""
    if isinstance(comparator, SedonaSpark):
        pytest.skip(
            "Sedona Spark mis-places scanline intercepts on non-square pixels "
            "(apache/sedona#3111); unskip when the fix ships in a release"
        )
    for case_id, width, height, gdal_transform, wkt in _fuzz_cases():
        path = tmp_path / f"fuzz_{case_id}.tif"
        write_geotiff(
            path,
            random_raster_data("uint8", bands=1, height=height, width=width),
            gdal_transform=gdal_transform,
        )
        kwargs = dict(burn_value=1.0, nodata=0.0, use_geometry_extent=False)
        got = subject.as_raster(wkt, path, "uint8", **kwargs)
        expected = comparator.as_raster(wkt, path, "uint8", **kwargs)
        assert_decoded_equal(
            got, expected, context=f"case {case_id}: {gdal_transform} {wkt}"
        )
