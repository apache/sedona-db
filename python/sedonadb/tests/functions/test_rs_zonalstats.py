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

import pyarrow as pa
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


def _invoke_zonalstats(con, tiff, wkt, *, all_stats, options=None):
    """Invoke RS_ZonalStats ('sum') or RS_ZonalStatsAll over a one-row table
    and return the Arrow result.

    Arguments travel as table columns so the kernel runs its real array path
    (literals constant-fold). The argument surface — `(raster, zone, stat,
    options)` for RS_ZonalStats and `(raster, zone, options)` for
    RS_ZonalStatsAll, with `band` inside the JSON `options` — mirrors the
    RS_ZonalStats function tests. These invoke the subject (SedonaDB) directly
    because the harness `zonal_stats` op exposes only a single scalar statistic,
    not the all-stats struct or the band-unspecified path.
    """
    columns = {
        "path": pa.array([str(tiff)], pa.utf8()),
        "wkt": pa.array([wkt], pa.utf8()),
    }
    if not all_stats:
        columns["stat"] = pa.array(["sum"], pa.utf8())
    if options is not None:
        columns["options"] = pa.array([options], pa.utf8())
    df = con.create_data_frame(pa.table(columns))
    raster = df.path.funcs.rs_frompath()
    geom = con.funcs.st_geomfromtext(df.wkt)
    tail = [df.options] if options is not None else []
    if all_stats:
        expr = raster.funcs.rs_zonalstatsall(geom, *tail)
    else:
        expr = raster.funcs.rs_zonalstats(geom, df.stat, *tail)
    return df.select(r=expr).to_arrow_table()


@pytest.mark.parametrize(
    "all_stats", [False, True], ids=["RS_ZonalStats", "RS_ZonalStatsAll"]
)
def test_multiband_raster_requires_band_option(con, tmp_path, all_stats):
    """On a multiband raster with no band chosen, SedonaDB raises rather than
    reducing an arbitrary band.

    Sedona Spark defaults to band 1 here (the documented divergence); asserting
    the raise pins SedonaDB's stricter contract — an ambiguous multiband
    selection is an error, not a silent band-1 pick.

    This is a subject-error case (the parity subject itself raises), so a plain
    `pytest.raises` on the subject is the right shape; it does not go through the
    comparator/deviation ledger. A ledger-integrated "subject_error" Deviation
    kind that also captured the Spark band-1 default declaratively would be a
    possible future enhancement.
    """
    tiff = tmp_path / "multiband.tif"
    write_geotiff(
        tiff,
        random_raster_data("float64", bands=2, height=HEIGHT, width=WIDTH),
        gdal_transform=GDAL_TRANSFORM,
    )
    # GEOM_RECT intersects the raster, so band resolution — not a
    # no-intersection short-circuit — is what fails.
    with pytest.raises(Exception, match="option to choose one"):
        _invoke_zonalstats(con, tiff, GEOM_RECT, all_stats=all_stats)


def test_zonalstatsall_count_field_is_int64(con, tmp_path):
    """RS_ZonalStatsAll returns `count` as an Int64 pixel count.

    Sedona Spark returns a uniform `Double[]` for every statistic, so its count
    is a floating-point value; SedonaDB keeps count as an integer. Pinning the
    Arrow struct field type guards that contract (the rest of the struct is
    Float64).
    """
    tiff = tmp_path / "singleband.tif"
    write_geotiff(
        tiff,
        random_raster_data("float64", bands=1, height=HEIGHT, width=WIDTH),
        gdal_transform=GDAL_TRANSFORM,
    )
    table = _invoke_zonalstats(
        con, tiff, GEOM_RECT, all_stats=True, options='{"band": 1}'
    )
    struct_type = table.schema.field("r").type
    assert struct_type.field("count").type == pa.int64()
