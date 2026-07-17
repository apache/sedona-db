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

"""RS_ZonalStats parity against geometry_mask + numpy reductions.

The rasterio comparator selects pixels with
`rasterio.features.geometry_mask`, drops pixels valued at the band nodata,
and reduces in float64. stddev/variance are the sample (ddof=1) statistics —
that is what Sedona computes. The diagonal-edged zone under the centroid
rule is on the Sedona Spark deviation ledger (its scanline rasterizer
mis-places x-intercepts on non-square pixels and drops some center-inside
pixels there, apache/sedona#3111). Zones that select no pixels are not
compared here.
"""

import pytest

from sedonadb.raster_testing import (
    Deviation,
    SedonaDB,
    SedonaSpark,
    expect_deviations,
    random_raster_data,
    write_geotiff,
)

pytest.importorskip("rasterio")
pytest.importorskip("shapely")

pytestmark = pytest.mark.skipif(
    not SedonaDB.implements("zonal_stats"),
    reason="RS_ZonalStats is not implemented in SedonaDB (the parity subject)",
)

# GDAL-order geotransform: origin (100, 500), 2-wide by 3-tall north-up
# pixels; with a 7x6 raster the extent is x in [100, 114], y in [482, 500].
GDAL_TRANSFORM = (100.0, 2.0, 0.0, 500.0, 0.0, -3.0)
HEIGHT, WIDTH = 6, 7
GEOM_RECT = (
    "POLYGON ((102.6 495.8, 109.3 495.8, 109.3 485.9, 102.6 485.9, 102.6 495.8))"
)
# Diagonal edges make all_touched matter, while staying clear of the corner
# pixels where the fixture plants the dtype extremes (a float64 extreme in
# the zone would push the squared-deviation statistics to infinity).
GEOM_TRIANGLE = "POLYGON ((102.7 497.4, 112.4 496.9, 104.2 483.7, 102.7 497.4))"

STATS = ["count", "sum", "mean", "min", "max", "stddev", "variance", "median"]

DEVIATIONS = [
    Deviation(
        SedonaSpark,
        "zonal_stats",
        matches=lambda p: p.get("wkt") == GEOM_TRIANGLE and not p.get("all_touched"),
        reason="Sedona's scanline rasterizer mis-places x-intercepts on "
        "non-square pixels and drops some center-inside pixels along "
        "diagonal edges; GDAL selects every center-inside pixel "
        "(https://github.com/apache/sedona/issues/3111)",
    ),
]


@pytest.mark.parametrize("stat", STATS)
@pytest.mark.parametrize(
    ("wkt", "all_touched"),
    [
        (GEOM_RECT, False),
        (GEOM_RECT, True),
        (GEOM_TRIANGLE, False),
        (GEOM_TRIANGLE, True),
    ],
    ids=["rect-centroid", "rect-touched", "triangle-centroid", "triangle-touched"],
)
def test_rs_zonalstats_matches_comparators(
    subject, comparator, request, tmp_path, wkt, all_touched, stat
):
    """Every statistic over the float64 fixture, on both selection rules.
    The zone stays clear of the corners so the planted dtype extremes don't
    collapse sums to infinity."""
    expect_deviations(request, comparator, "zonal_stats", DEVIATIONS)
    tiff = tmp_path / "zonal.tif"
    write_geotiff(
        tiff,
        random_raster_data("float64", bands=2, height=HEIGHT, width=WIDTH),
        gdal_transform=GDAL_TRANSFORM,
    )

    got = subject.zonal_stats(tiff, wkt, band=2, stat=stat, all_touched=all_touched)
    expected = comparator.zonal_stats(
        tiff, wkt, band=2, stat=stat, all_touched=all_touched
    )
    # Engines reduce in different orders, so exact float equality is not
    # attainable; 1e-9 passes summation noise and still fails any semantic
    # mismatch (selection, nodata handling, ddof).
    assert got == pytest.approx(expected, rel=1e-9), (wkt, all_touched, stat)


@pytest.mark.parametrize("stat", ["count", "sum"])
def test_rs_zonalstats_excludes_nodata(subject, comparator, tmp_path, stat):
    """A pixel valued at the band nodata inside the zone is excluded from
    the reduction by every engine."""
    tiff = tmp_path / "zonal_nodata.tif"
    write_geotiff(
        tiff,
        random_raster_data(
            "uint8", bands=1, height=HEIGHT, width=WIDTH, plants={(2, 3): 200}
        ),
        gdal_transform=GDAL_TRANSFORM,
        nodata=200.0,
    )

    got = subject.zonal_stats(tiff, GEOM_RECT, stat=stat)
    expected = comparator.zonal_stats(tiff, GEOM_RECT, stat=stat)
    assert got == pytest.approx(expected, rel=1e-9), stat
