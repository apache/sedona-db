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

"""RS_ZonalStats / RS_ZonalStatsAll parity across their Spark overloads.

The rasterio comparator selects pixels with `rasterio.features.geometry_mask`,
optionally drops pixels valued at the band nodata, and reduces in float64.
stddev/variance are the sample (ddof=1) statistics Sedona computes; mode breaks
ties toward the larger value. The overload ladder is exercised by which optional
arguments each call passes:

- `band` omitted -> the band-less overload (resolves to band 1 on a single-band
  raster; an error on a multiband one).
- a set `band`, then `all_touched`, `exclude_nodata`, `lenient` -> the longer
  overloads.

The diagonal-edged zone under the centroid rule is on the Sedona Spark deviation
ledger (its scanline rasterizer mis-places x-intercepts on non-square pixels and
drops some center-inside pixels there, apache/sedona#3111).
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
# A zone entirely outside the raster extent (x > 114) — selects no pixels.
GEOM_OUTSIDE = "POLYGON ((200 495, 210 495, 210 485, 200 485, 200 495))"

STATS = ["count", "sum", "mean", "min", "max", "stddev", "variance", "median", "mode"]

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
    """Every statistic over the float64 fixture, on both selection rules. The
    zone stays clear of the corners so the planted dtype extremes don't collapse
    sums to infinity. mode on all-distinct pixels is the largest value (its
    tie-break), which both engines agree on."""
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


@pytest.mark.parametrize("stat", ["count", "mean", "max"])
def test_rs_zonalstats_band_default_matches_comparators(
    subject, comparator, tmp_path, stat
):
    """The band-less overload (band omitted) resolves to band 1 on a single-band
    raster. Comparing the subject's 3-argument call against the comparator with
    `band=None` pins that band-1 default."""
    tiff = tmp_path / "zonal_singleband.tif"
    write_geotiff(
        tiff,
        random_raster_data("float64", bands=1, height=HEIGHT, width=WIDTH),
        gdal_transform=GDAL_TRANSFORM,
    )
    got = subject.zonal_stats(tiff, GEOM_RECT, stat=stat)
    expected = comparator.zonal_stats(tiff, GEOM_RECT, stat=stat)
    assert got == pytest.approx(expected, rel=1e-9), stat


@pytest.mark.parametrize("stat", ["count", "sum"])
def test_rs_zonalstats_excludes_nodata(subject, comparator, tmp_path, stat):
    """A pixel valued at the band nodata inside the zone is excluded from the
    reduction by every engine (exclude_nodata defaults to true)."""
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


@pytest.mark.parametrize("stat", ["count", "sum"])
def test_rs_zonalstats_include_nodata_matches_comparators(
    subject, comparator, tmp_path, stat
):
    """The 6-argument overload with exclude_nodata=false keeps the nodata-valued
    pixel in the reduction. Comparing against the comparator with the same flag
    pins that the pixel is counted, not skipped — the opposite of the default."""
    tiff = tmp_path / "zonal_include_nodata.tif"
    write_geotiff(
        tiff,
        random_raster_data(
            "uint8", bands=1, height=HEIGHT, width=WIDTH, plants={(2, 3): 200}
        ),
        gdal_transform=GDAL_TRANSFORM,
        nodata=200.0,
    )
    got = subject.zonal_stats(tiff, GEOM_RECT, band=1, stat=stat, exclude_nodata=False)
    expected = comparator.zonal_stats(
        tiff, GEOM_RECT, band=1, stat=stat, exclude_nodata=False
    )
    assert got == pytest.approx(expected, rel=1e-9), stat


def test_rs_zonalstats_lenient_false_computes(subject, comparator, tmp_path):
    """With lenient=false (the 7-argument overload) an intersecting zone still
    computes normally — lenient only governs the non-intersecting case."""
    tiff = tmp_path / "zonal_lenient.tif"
    write_geotiff(
        tiff,
        random_raster_data("float64", bands=1, height=HEIGHT, width=WIDTH),
        gdal_transform=GDAL_TRANSFORM,
    )
    got = subject.zonal_stats(tiff, GEOM_RECT, band=1, stat="mean", lenient=False)
    expected = comparator.zonal_stats(tiff, GEOM_RECT, band=1, stat="mean")
    assert got == pytest.approx(expected, rel=1e-9)


def test_rs_zonalstats_lenient_false_errors_on_disjoint_zone(con, tmp_path):
    """A zone that misses the raster raises with lenient=false, where the default
    lenient=true returns NULL.

    This is a subject-error case (the parity subject itself raises), so a plain
    `pytest.raises` on the subject is the right shape; it does not go through the
    comparator/deviation ledger. Arguments travel as table columns so the kernel
    runs its real array path."""
    tiff = tmp_path / "zonal_disjoint.tif"
    write_geotiff(
        tiff,
        random_raster_data("float64", bands=1, height=HEIGHT, width=WIDTH),
        gdal_transform=GDAL_TRANSFORM,
    )
    table = pa.table(
        {
            "path": pa.array([str(tiff)], pa.utf8()),
            "wkt": pa.array([GEOM_OUTSIDE], pa.utf8()),
            "band": pa.array([1], pa.int32()),
            "stat": pa.array(["mean"], pa.utf8()),
            "all_touched": pa.array([False], pa.bool_()),
            "exclude_nodata": pa.array([True], pa.bool_()),
            "lenient": pa.array([False], pa.bool_()),
        }
    )
    df = con.create_data_frame(table)
    expr = df.path.funcs.rs_frompath().funcs.rs_zonalstats(
        con.funcs.st_geomfromtext(df.wkt),
        df.band,
        df.stat,
        df.all_touched,
        df.exclude_nodata,
        df.lenient,
    )
    with pytest.raises(Exception, match="does not intersect"):
        df.select(v=expr).to_arrow_table()


@pytest.mark.parametrize(
    "all_stats", [False, True], ids=["RS_ZonalStats", "RS_ZonalStatsAll"]
)
def test_multiband_raster_requires_band_option(con, tmp_path, all_stats):
    """On a multiband raster with no band chosen, SedonaDB raises rather than
    reducing an arbitrary band.

    Sedona Spark defaults to band 1 here (the documented divergence); asserting
    the raise pins SedonaDB's stricter contract — an ambiguous multiband
    selection is an error, not a silent band-1 pick. This is a subject-error case
    (the parity subject itself raises), so a plain `pytest.raises` on the subject
    is the right shape; it does not go through the comparator/deviation ledger.
    The raster travels as a table column so the kernel runs its real array
    path."""
    tiff = tmp_path / "zonal_multiband.tif"
    write_geotiff(
        tiff,
        random_raster_data("float64", bands=2, height=HEIGHT, width=WIDTH),
        gdal_transform=GDAL_TRANSFORM,
    )
    df = con.create_data_frame(
        pa.table(
            {
                "path": pa.array([str(tiff)], pa.utf8()),
                "wkt": pa.array([GEOM_RECT], pa.utf8()),
            }
        )
    )
    raster = df.path.funcs.rs_frompath()
    geom = con.funcs.st_geomfromtext(df.wkt)
    # GEOM_RECT intersects the raster, so band resolution — not a
    # no-intersection short-circuit — is what fails.
    if all_stats:
        expr = raster.funcs.rs_zonalstatsall(geom)
    else:
        expr = raster.funcs.rs_zonalstats(geom, "mean")
    with pytest.raises(Exception, match="choose one"):
        df.select(r=expr).to_arrow_table()


# RS_ZonalStatsAll struct fields, in the order the Rust kernel emits them.
ALL_FIELDS = [
    "count",
    "sum",
    "mean",
    "median",
    "mode",
    "stddev",
    "variance",
    "min",
    "max",
]


@pytest.mark.parametrize("all_touched", [False, True], ids=["centroid", "touched"])
def test_rs_zonalstatsall_matches_comparators(
    subject, comparator, request, tmp_path, all_touched
):
    """RS_ZonalStatsAll returns every statistic as one struct. A repeated value
    is planted inside the zone so `mode` is that value (not merely the largest
    distinct one), exercising the frequency tie-break; the rest match the scalar
    reductions."""
    expect_deviations(request, comparator, "zonal_stats", DEVIATIONS)
    tiff = tmp_path / "zonalall.tif"
    write_geotiff(
        tiff,
        random_raster_data(
            "float64",
            bands=2,
            height=HEIGHT,
            width=WIDTH,
            plants={(2, 2): 50.0, (2, 3): 50.0, (3, 2): 50.0},
        ),
        gdal_transform=GDAL_TRANSFORM,
    )
    got = subject.zonal_stats_all(tiff, GEOM_RECT, band=2, all_touched=all_touched)
    expected = comparator.zonal_stats_all(
        tiff, GEOM_RECT, band=2, all_touched=all_touched
    )
    assert got["count"] == expected["count"]
    assert got["mode"] == pytest.approx(expected["mode"]), "mode tie-break"
    assert got["mode"] == pytest.approx(50.0), "planted repeat is the mode"
    for field in ALL_FIELDS:
        assert got[field] == pytest.approx(expected[field], rel=1e-9), field


def test_rs_zonalstatsall_count_field_is_int64(con, tmp_path):
    """RS_ZonalStatsAll returns `count` as an Int64 pixel count.

    Sedona Spark returns a uniform `Double[]` for every statistic, so its count
    is a floating-point value; SedonaDB keeps count as an integer. Pinning the
    Arrow struct field type guards that contract (the rest of the struct is
    Float64). The raster travels as a table column so the kernel runs its real
    array path."""
    tiff = tmp_path / "zonalall_singleband.tif"
    write_geotiff(
        tiff,
        random_raster_data("float64", bands=1, height=HEIGHT, width=WIDTH),
        gdal_transform=GDAL_TRANSFORM,
    )
    df = con.create_data_frame(
        pa.table(
            {
                "path": pa.array([str(tiff)], pa.utf8()),
                "wkt": pa.array([GEOM_RECT], pa.utf8()),
                "band": pa.array([1], pa.int32()),
            }
        )
    )
    expr = df.path.funcs.rs_frompath().funcs.rs_zonalstatsall(
        con.funcs.st_geomfromtext(df.wkt), df.band
    )
    table = df.select(r=expr).to_arrow_table()
    struct_type = table.schema.field("r").type
    assert struct_type.field("count").type == pa.int64()


def test_rs_zonalstats_sql_smoke(con, tmp_path):
    """One SQL-text invocation keeps the parser path covered (everything else
    routes through the expression API)."""
    tiff = tmp_path / "zonal_smoke.tif"
    write_geotiff(
        tiff,
        random_raster_data("uint8", bands=1, height=HEIGHT, width=WIDTH),
        gdal_transform=GDAL_TRANSFORM,
    )
    tab = con.sql(
        "SELECT RS_ZonalStats(RS_FromPath($1), ST_GeomFromText($2), 1, 'count') AS c",
        params=(str(tiff), GEOM_RECT),
    ).to_arrow_table()
    assert tab["c"][0].as_py() > 0
