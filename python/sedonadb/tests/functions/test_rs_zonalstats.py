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

"""RS_ZonalStats / RS_ZonalStatsAll parity across their Sedona Spark overloads.

The rasterio reference selects pixels with `rasterio.features.geometry_mask`,
optionally drops pixels valued at the band nodata, and reduces in float64.
stddev/variance are the sample (ddof=1) statistics Sedona computes; mode breaks
ties toward the larger value. The overload ladder is exercised by which optional
arguments each call passes: `band` omitted is the band-less overload (resolves to
band 1 on a single-band raster, errors on a multiband one); a set `band`, then
`all_touched`, `exclude_nodata`, `lenient` select the longer overloads.

Two design choices distinguish this module from the first parity modules:

- Known divergences are declared **inline**, as an `xfail`/`skip` mark on the
  parametrized case, not in a separate deviations ledger — so reading one case
  shows both what it asserts and where an engine is expected to disagree.
- SedonaDB and Sedona Spark are both `SedonaDialectEngine`s: `zonal_stats` /
  `zonal_stats_all` build one RS_* argument string (via the shared
  `zonal_stats_expr` / `zonal_stats_all_expr` generators) that both dialects
  execute unchanged. `test_zonal_stats_sql_runs_unchanged_on_both_dialects`
  asserts that compatibility contract directly.
"""

import pyarrow as pa
import pytest

from sedonadb.raster_testing import (
    Rasterio,
    SedonaDB,
    SedonaDialectEngine,
    SedonaSpark,
    ZONAL_ALL_FIELDS,
    random_raster_data,
    write_geotiff,
)

pytest.importorskip("rasterio")
pytest.importorskip("shapely")

# The harness `zonal_stats` methods are pure SQL construction and exist whether
# or not RS_ZonalStats is compiled in, so the module gates on the function being
# registered in this SedonaDB build (see `SedonaDB.registers_function`) rather
# than on the method's presence. Band 1 is named because RS_Example is
# multiband; a CRS error from the probe still means the function is registered.
pytestmark = pytest.mark.skipif(
    not SedonaDB.registers_function(
        "SELECT RS_ZonalStats(RS_Example(), "
        "ST_GeomFromText('POLYGON ((43 79, 60 79, 60 60, 43 60, 43 79))'), 1, 'count')"
    ),
    reason="RS_ZonalStats is not registered in this SedonaDB build (the parity subject)",
)

# GDAL-order geotransform: origin (100, 500), 2-wide by 3-tall north-up pixels;
# with a 7x6 raster the extent is x in [100, 114], y in [482, 500].
GDAL_TRANSFORM = (100.0, 2.0, 0.0, 500.0, 0.0, -3.0)
HEIGHT, WIDTH = 6, 7
GEOM_RECT = (
    "POLYGON ((102.6 495.8, 109.3 495.8, 109.3 485.9, 102.6 485.9, 102.6 495.8))"
)
# Diagonal edges make all_touched matter, while staying clear of the corner
# pixels where the fixture plants the dtype extremes (a float64 extreme in the
# zone would push the squared-deviation statistics to infinity).
GEOM_TRIANGLE = "POLYGON ((102.7 497.4, 112.4 496.9, 104.2 483.7, 102.7 497.4))"
# A zone entirely outside the raster extent (x > 114) — selects no pixels.
GEOM_OUTSIDE = "POLYGON ((200 495, 210 495, 210 485, 200 485, 200 495))"

STATS = ["count", "sum", "mean", "min", "max", "stddev", "variance", "median", "mode"]

# count is an integer pixel count and min/max/mode are pass-through pixel values,
# so once the subject and the comparator select the same pixel set these are
# bit-identical — asserted with exact equality. approx here would hide a real
# selection or tie-break divergence. Only the accumulating stats (sum, mean,
# median, stddev, variance) warrant a tolerance, because engines reduce in
# different orders.
EXACT_STATS = frozenset({"count", "min", "max", "mode"})


def assert_stat_equal(got, expected, stat):
    """Assert one RS_ZonalStats result against a comparator: exact equality for
    the selection/count stats in `EXACT_STATS`, approx (rel=1e-9) for the
    accumulating ones."""
    if stat in EXACT_STATS:
        assert got == expected, stat
    else:
        assert got == pytest.approx(expected, rel=1e-9), stat


def assert_all_stats_equal(got, expected):
    """Assert an RS_ZonalStatsAll struct field-by-field against a comparator,
    keying each field on `EXACT_STATS`: exact for count/min/max/mode, approx
    (rel=1e-9) for the accumulating stats."""
    for field in ZONAL_ALL_FIELDS:
        assert_stat_equal(got[field], expected[field], field)


# Sedona Spark's scanline rasterizer mis-places x-intercepts on non-square pixels
# and drops some center-inside pixels along diagonal edges, where GDAL (which
# both SedonaDB and rasterio burn through) selects every center-inside pixel. It
# is therefore expected to disagree on the diagonal zone under the centroid rule.
SPARK_DIAGONAL_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="Sedona's scanline rasterizer drops center-inside pixels along "
    "diagonal edges under the centroid rule; GDAL selects every "
    "center-inside pixel (https://github.com/apache/sedona/issues/3111)",
)

# (comparator engine, zone, all_touched). Every comparator runs on every zone;
# the one combination Sedona Spark is known to get wrong carries its xfail right
# here, beside the case, rather than in a separate ledger the reader must chase.
COMPARATOR_CASES = [
    pytest.param(Rasterio, GEOM_RECT, False, id="rasterio-rect-centroid"),
    pytest.param(Rasterio, GEOM_RECT, True, id="rasterio-rect-touched"),
    pytest.param(Rasterio, GEOM_TRIANGLE, False, id="rasterio-triangle-centroid"),
    pytest.param(Rasterio, GEOM_TRIANGLE, True, id="rasterio-triangle-touched"),
    pytest.param(SedonaSpark, GEOM_RECT, False, id="spark-rect-centroid"),
    pytest.param(SedonaSpark, GEOM_RECT, True, id="spark-rect-touched"),
    pytest.param(
        SedonaSpark,
        GEOM_TRIANGLE,
        False,
        id="spark-triangle-centroid",
        marks=SPARK_DIAGONAL_XFAIL,
    ),
    pytest.param(SedonaSpark, GEOM_TRIANGLE, True, id="spark-triangle-touched"),
]


@pytest.mark.parametrize("stat", STATS)
@pytest.mark.parametrize(("comparator_cls", "wkt", "all_touched"), COMPARATOR_CASES)
def test_rs_zonalstats_matches_comparators(
    subject, tmp_path, comparator_cls, wkt, all_touched, stat
):
    """Every statistic over the float64 fixture, on both selection rules and both
    comparators. The zone stays clear of the corners so the planted dtype
    extremes don't collapse sums to infinity. mode on all-distinct pixels is the
    largest value (its tie-break), which both engines agree on.

    The comparator is a parametrized case (not the shared `comparator` fixture)
    so the one known Sedona Spark divergence can be marked xfail inline above."""
    comparator = comparator_cls.create_or_skip()
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
    # count/min/max/mode are asserted exactly (same pixel set => bit-identical);
    # the accumulating stats get approx, since engines reduce in different orders
    # (1e-9 passes summation noise and still fails any semantic mismatch —
    # selection, nodata handling, ddof).
    assert_stat_equal(got, expected, stat)


@pytest.mark.parametrize("stat", ["count", "mean", "max"])
def test_rs_zonalstats_band_default_matches_comparators(
    subject, comparator, tmp_path, stat
):
    """The band-less overload (band omitted) resolves to band 1 on a single-band
    raster. Comparing the subject's 3-argument call against the comparator with
    `band=None` pins that band-1 default. No zone here diverges, so the shared
    `comparator` fixture covers both engines."""
    tiff = tmp_path / "zonal_singleband.tif"
    write_geotiff(
        tiff,
        random_raster_data("float64", bands=1, height=HEIGHT, width=WIDTH),
        gdal_transform=GDAL_TRANSFORM,
    )
    got = subject.zonal_stats(tiff, GEOM_RECT, stat=stat)
    expected = comparator.zonal_stats(tiff, GEOM_RECT, stat=stat)
    assert_stat_equal(got, expected, stat)


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
    assert_stat_equal(got, expected, stat)


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
    assert_stat_equal(got, expected, stat)


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
    assert_stat_equal(got, expected, "mean")


def test_rs_zonalstats_lenient_false_errors_on_disjoint_zone(con, tmp_path):
    """A zone that misses the raster raises with lenient=false, where the default
    lenient=true returns NULL.

    This is a subject-error case (the parity subject itself raises), so a plain
    `pytest.raises` on the subject is the right shape. Arguments travel as table
    columns so the kernel runs its real array path."""
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
    is the right shape. The raster travels as a table column so the kernel runs
    its real array path."""
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


@pytest.mark.parametrize("all_touched", [False, True], ids=["centroid", "touched"])
def test_rs_zonalstatsall_matches_comparators(
    subject, comparator, tmp_path, all_touched
):
    """RS_ZonalStatsAll returns every statistic as one struct. A repeated value
    is planted inside the zone so `mode` is that value (not merely the largest
    distinct one), exercising the frequency tie-break; the rest match the scalar
    reductions. GEOM_RECT is used (not the diagonal zone), so the shared
    `comparator` fixture covers both engines without a divergence."""
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
    # count/min/max/mode match bit-for-bit (same pixel set); the accumulating
    # stats get approx. mode is now exact, so the frequency tie-break is pinned
    # exactly against both the comparator and the planted repeat.
    assert_all_stats_equal(got, expected)
    assert got["mode"] == 50.0, "planted repeat is the mode"


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


# The SQL-surface parity contract: one corpus of RS_* argument strings, produced
# by the shared SedonaDialectEngine generators, that must parse and execute identically
# on both dialects. `kind` picks the scalar or struct generator; `kwargs` walks
# the overload ladder. Expected literals pin the exact generated surface so drift
# is caught even on a machine without Spark.
CONTRACT_CALLS = [
    # Band-less overloads (no band argument): on this single-band fixture they
    # resolve to band 1 rather than raising the multiband error.
    (
        "scalar",
        {"stat": "count"},
        f"RS_ZonalStats(rast, ST_GeomFromText('{GEOM_RECT}'), 'count')",
    ),
    (
        "scalar",
        {"band": 1, "stat": "count"},
        f"RS_ZonalStats(rast, ST_GeomFromText('{GEOM_RECT}'), 1, 'count')",
    ),
    (
        "scalar",
        {"band": 1, "stat": "mean", "all_touched": True},
        f"RS_ZonalStats(rast, ST_GeomFromText('{GEOM_RECT}'), 1, 'mean', true)",
    ),
    (
        "scalar",
        {"band": 1, "stat": "sum", "exclude_nodata": False},
        f"RS_ZonalStats(rast, ST_GeomFromText('{GEOM_RECT}'), 1, 'sum', false, false)",
    ),
    (
        "struct",
        {},
        f"RS_ZonalStatsAll(rast, ST_GeomFromText('{GEOM_RECT}'))",
    ),
    (
        "struct",
        {"band": 1},
        f"RS_ZonalStatsAll(rast, ST_GeomFromText('{GEOM_RECT}'), 1)",
    ),
    (
        "struct",
        {"band": 1, "all_touched": True},
        f"RS_ZonalStatsAll(rast, ST_GeomFromText('{GEOM_RECT}'), 1, true)",
    ),
]


@pytest.mark.parametrize(
    ("kind", "kwargs", "expected_sql"),
    CONTRACT_CALLS,
    ids=[call[2] for call in CONTRACT_CALLS],
)
def test_zonal_stats_sql_runs_unchanged_on_both_dialects(
    subject, tmp_path, kind, kwargs, expected_sql
):
    """The compatibility contract asserted directly: the byte-identical RS_* call
    string built by the shared generator both (a) matches the expected surface
    and (b) parses and executes on SedonaDB and Sedona Spark alike, with matching
    results. The SedonaDB arm runs unconditionally (proving the generated string
    parses and executes); the Spark arm skips via `create_or_skip` when the jars
    are absent, so on a jar-less machine this proves only the SedonaDB half."""
    if kind == "scalar":
        expr = SedonaDialectEngine.zonal_stats_expr(GEOM_RECT, **kwargs)
    else:
        expr = SedonaDialectEngine.zonal_stats_all_expr(GEOM_RECT, **kwargs)
    assert expr == expected_sql

    tiff = tmp_path / "contract.tif"
    write_geotiff(
        tiff,
        random_raster_data("float64", bands=1, height=HEIGHT, width=WIDTH),
        gdal_transform=GDAL_TRANSFORM,
    )
    # Both dialects execute the SAME `expr` string; only the I/O around it differs.
    if kind == "scalar":
        db_result = subject._run_scalar(tiff, expr)
        assert db_result is not None
    else:
        db_result = subject._run_struct(tiff, expr)
        assert db_result is not None and db_result["count"] > 0

    spark = SedonaSpark.create_or_skip()
    if kind == "scalar":
        assert_stat_equal(spark._run_scalar(tiff, expr), db_result, kwargs["stat"])
    else:
        assert_all_stats_equal(spark._run_struct(tiff, expr), db_result)


def test_rs_zonalstats_sql_smoke(con, tmp_path):
    """One SQL-text invocation keeps the parser path covered (everything else
    routes through the expression API or the dialect generators)."""
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
