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

"""RS_ZonalStats / RS_ZonalStatsAll cross-checked against a numpy reference.

Both functions mirror Apache Sedona Spark's positional overloads, so the tests
call them positionally: `(raster, roi, stat_type)` /
`(raster, roi, band, stat_type[, all_touched[, exclude_no_data[, lenient]]])`
for RS_ZonalStats and the same ladder without `stat_type` for RS_ZonalStatsAll.

The fixture raster is CRS-less (so nothing reprojects and pixel selection is
bit-comparable). The reference rasterizes the roi with `rasterio.features`
(the same GDAL rasterizer the kernel uses) and reduces the selected pixels with
numpy; exact-selection statistics (count, sum, min, max, median, mode) are
compared exactly and the float-accumulation ones (mean, variance, stddev) with
a tolerance.

rasterio is required to write the fixture GeoTIFF, so the whole module skips
when it is unavailable rather than importing it at module scope.
"""

import math

import numpy as np
import pyarrow as pa
import pytest
import shapely
from shapely.geometry import box

pytest.importorskip("rasterio")

from sedonadb.raster_testing import (
    random_raster_data,
    write_geotiff,
)

# GDAL-order geotransform: origin (100, 500), 2-wide by 3-tall north-up pixels;
# a 6x7 raster then spans x in [100, 114], y in [482, 500].
GDAL_TRANSFORM = (100.0, 2.0, 0.0, 500.0, 0.0, -3.0)
BANDS, HEIGHT, WIDTH = 1, 6, 7
NODATA = -9999

# A rectangle well inside the raster that selects a block of pixels.
GEOM_RECT = (
    "POLYGON ((102.6 495.8, 109.3 495.8, 109.3 485.9, 102.6 485.9, 102.6 495.8))"
)
# Entirely outside the raster extent.
GEOM_DISJOINT = "POLYGON ((900 900, 910 900, 910 890, 900 890, 900 900))"
# Bounding box overlaps the raster, but the geometry itself is disjoint: the
# triangle sits in the far corner of its bounding box, clear of the raster. A
# bounding-box gate would burn no pixels and report count 0; a true-geometry gate
# (matching Sedona Spark's rsIntersects) treats it as a no-intersection case.
GEOM_DISJOINT_BBOX = "POLYGON ((124 490, 124 510, 108 510, 124 490))"
# A thin strip crossing the x = 104 pixel boundary but covering no pixel center
# (centers sit at odd x): selects nothing unless all_touched.
GEOM_SLIVER = "POLYGON ((103.6 499, 104.4 499, 104.4 483, 103.6 483, 103.6 499))"

STATS = ["count", "sum", "mean", "median", "mode", "stddev", "variance", "min", "max"]
EXACT_STATS = {"count", "sum", "min", "max", "median", "mode"}


def fixture_raster(tmp_path):
    """A single-band int32 raster with planted nodata and a repeated value.

    Returns `(path, band)` where `band` is the `(HEIGHT, WIDTH)` numpy array.
    Two interior pixels hold the nodata value and three hold a repeated value
    (66) so the mode is unambiguous and nodata exclusion is observable.
    """
    data = random_raster_data(
        "int32",
        bands=BANDS,
        height=HEIGHT,
        width=WIDTH,
        seed=7,
        plants={(1, 1): NODATA, (2, 2): NODATA, (1, 2): 66, (2, 3): 66, (3, 1): 66},
    )
    path = tmp_path / "zonal.tif"
    write_geotiff(path, data, gdal_transform=GDAL_TRANSFORM, nodata=NODATA)
    return path, data[0]


def float_fixture_raster(tmp_path, *, planted_value, name):
    """A single-band float64 raster with `planted_value` (NaN or inf) at an
    interior pixel that GEOM_RECT selects, so the roi statistics must reckon with
    it. The planted value is not the nodata sentinel, so nodata exclusion (on by
    default) does not drop it.

    Returns `(path, band)` where `band` is the `(HEIGHT, WIDTH)` numpy array.
    """
    data = random_raster_data(
        "float64", bands=BANDS, height=HEIGHT, width=WIDTH, seed=7
    )
    # Pixel centre (105, 492.5) sits inside GEOM_RECT.
    data[0][2, 2] = planted_value
    path = tmp_path / name
    write_geotiff(path, data, gdal_transform=GDAL_TRANSFORM, nodata=NODATA)
    return path, data[0]


def numpy_reference(band, wkt, *, all_touched, exclude_no_data):
    """Reference statistics over the pixels the roi selects, via rasterio+numpy.

    Returns a dict of every statistic, or the sentinel string ``"empty"`` when
    the selection is empty (the caller maps that to count 0 / NULLs).
    """
    import rasterio.features
    from rasterio.transform import Affine

    geom = shapely.from_wkt(wkt)
    mask = rasterio.features.rasterize(
        [(geom, 1)],
        out_shape=band.shape,
        transform=Affine.from_gdal(*GDAL_TRANSFORM),
        all_touched=all_touched,
        fill=0,
        dtype="uint8",
    )
    sel = band[mask == 1].astype(np.float64)
    if exclude_no_data:
        sel = sel[sel != NODATA]
    if sel.size == 0:
        return "empty"

    values, counts = np.unique(sel, return_counts=True)
    mode = float(values[counts == counts.max()].max())  # ties -> largest
    n = sel.size
    return {
        "count": float(n),
        "sum": float(sel.sum()),
        "mean": float(sel.mean()),
        "median": float(np.median(sel)),
        "mode": mode,
        "stddev": float(sel.std(ddof=1)) if n > 1 else 0.0,
        "variance": float(sel.var(ddof=1)) if n > 1 else 0.0,
        "min": float(sel.min()),
        "max": float(sel.max()),
    }


def _assert_stat_equal(stat, got, expected):
    """Compare one statistic against the numpy reference with NaN/inf-aware
    equality: NaN never equals itself, so match it explicitly; inf compares
    exactly; otherwise use exact equality for the integer-exact statistics and a
    tolerance for the float-accumulating ones."""
    if math.isnan(expected):
        assert got is not None and math.isnan(got), f"{stat}: expected NaN, got {got!r}"
    elif math.isinf(expected):
        assert got == expected, f"{stat}: expected {expected}, got {got!r}"
    elif stat in EXACT_STATS:
        assert got == expected, f"{stat}: {got!r} != {expected!r}"
    else:
        assert got == pytest.approx(expected), f"{stat}: {got!r} !~ {expected!r}"


def _one_row(con, path, wkt):
    """A one-row frame with the raster and roi as columns, so the kernel runs
    its real per-row array path rather than constant-folding a scalar."""
    df = con.create_data_frame(
        pa.table(
            {
                "path": pa.array([str(path)], pa.utf8()),
                "wkt": pa.array([wkt], pa.utf8()),
            }
        )
    )
    return df, df.path.funcs.rs_frompath(), con.funcs.st_geomfromtext(df.wkt)


def zonal_stat(con, path, wkt, trailing):
    """RS_ZonalStats over a one-row table.

    `trailing` is the positional argument list after `(raster, roi)` — e.g.
    `["mean"]` (band-less overload) or `[1, "mean", all_touched, exclude_no_data,
    lenient]`. Raster and roi travel as columns; the trailing scalars are
    literals, matching how the SQL reads.
    """
    df, raster, geom = _one_row(con, path, wkt)
    table = df.select(r=raster.funcs.rs_zonalstats(geom, *trailing)).to_arrow_table()
    return table["r"][0].as_py()


def zonal_stats_all(con, path, wkt, trailing):
    """RS_ZonalStatsAll over a one-row table; returns the struct as a dict.

    `trailing` is the positional argument list after `(raster, roi)` — e.g.
    `[]` (band-less overload) or `[1, all_touched, exclude_no_data, lenient]`.
    """
    df, raster, geom = _one_row(con, path, wkt)
    table = df.select(r=raster.funcs.rs_zonalstatsall(geom, *trailing)).to_arrow_table()
    return table["r"][0].as_py()


@pytest.mark.parametrize("stat", STATS)
@pytest.mark.parametrize("all_touched", [False, True])
def test_single_stat_matches_numpy(con, tmp_path, stat, all_touched):
    path, band = fixture_raster(tmp_path)
    expected = numpy_reference(
        band, GEOM_RECT, all_touched=all_touched, exclude_no_data=True
    )
    assert expected != "empty", "GEOM_RECT should select pixels"

    # (raster, roi, band, stat_type, all_touched) — the 5-arg overload.
    got = zonal_stat(con, path, GEOM_RECT, [1, stat, all_touched])
    if stat in EXACT_STATS:
        assert got == expected[stat]
    else:
        assert got == pytest.approx(expected[stat])


def test_all_struct_matches_numpy(con, tmp_path):
    path, band = fixture_raster(tmp_path)
    expected = numpy_reference(band, GEOM_RECT, all_touched=False, exclude_no_data=True)
    # (raster, roi, band) — all_touched defaults to false.
    got = zonal_stats_all(con, path, GEOM_RECT, [1])

    # count is an integer field (Int64); every other field is floating point.
    assert isinstance(got["count"], int)
    assert isinstance(got["sum"], float)
    assert got["count"] == expected["count"]
    for stat in EXACT_STATS - {"count"}:
        assert got[stat] == expected[stat]
    for stat in ("mean", "variance", "stddev"):
        assert got[stat] == pytest.approx(expected[stat])


def test_nan_pixel_poisons_every_statistic_like_numpy(con, tmp_path):
    """A NaN pixel that is not the nodata sentinel poisons every statistic (numpy
    semantics): count stays a real tally, everything else is NaN. Pinned against
    rasterio+numpy over the same masked selection, so the NaN handling is
    validated against a trusted reference rather than asserted on faith."""
    path, band = float_fixture_raster(
        tmp_path, planted_value=float("nan"), name="zonal_nan.tif"
    )
    with np.errstate(all="ignore"):
        expected = numpy_reference(
            band, GEOM_RECT, all_touched=False, exclude_no_data=True
        )
    assert expected != "empty", "GEOM_RECT should select the planted pixel"
    got = zonal_stats_all(con, path, GEOM_RECT, [1])
    for stat in STATS:
        _assert_stat_equal(stat, got[stat], expected[stat])


def test_infinity_pixel_matches_numpy(con, tmp_path):
    """A +inf pixel flows through (not the nodata sentinel, not NaN): sum, mean,
    max and mode go to +inf, min and median stay finite, and variance/stddev
    become NaN (inf - inf). Pinned against rasterio+numpy over the same
    selection."""
    path, band = float_fixture_raster(
        tmp_path, planted_value=float("inf"), name="zonal_inf.tif"
    )
    with np.errstate(all="ignore"):
        expected = numpy_reference(
            band, GEOM_RECT, all_touched=False, exclude_no_data=True
        )
    assert expected != "empty", "GEOM_RECT should select the planted pixel"
    got = zonal_stats_all(con, path, GEOM_RECT, [1])
    for stat in STATS:
        _assert_stat_equal(stat, got[stat], expected[stat])


def test_exclude_no_data_default_and_disabled(con, tmp_path):
    path, band = fixture_raster(tmp_path)
    # Default excludes nodata; disabling it keeps those pixels, raising count.
    excluded = numpy_reference(band, GEOM_RECT, all_touched=False, exclude_no_data=True)
    included = numpy_reference(
        band, GEOM_RECT, all_touched=False, exclude_no_data=False
    )
    assert included["count"] > excluded["count"]

    # Default (4-arg (raster, roi, band, stat_type)) excludes nodata.
    assert zonal_stat(con, path, GEOM_RECT, [1, "count"]) == excluded["count"]
    # exclude_no_data => false keeps it: the 6-arg overload trails (all_touched,
    # exclude_no_data).
    assert (
        zonal_stat(con, path, GEOM_RECT, [1, "count", False, False])
        == included["count"]
    )


def test_sliver_selects_nothing_unless_all_touched(con, tmp_path):
    path, _ = fixture_raster(tmp_path)
    # The roi overlaps the raster but covers no pixel center: count 0, rest NULL.
    assert zonal_stat(con, path, GEOM_SLIVER, [1, "count"]) == 0.0
    assert zonal_stat(con, path, GEOM_SLIVER, [1, "sum"]) is None
    # all_touched (5-arg overload) picks up the pixels it crosses.
    touched = zonal_stat(con, path, GEOM_SLIVER, [1, "count", True])
    assert touched > 0.0


def test_no_intersection_is_null_when_lenient_and_errors_when_strict(con, tmp_path):
    path, _ = fixture_raster(tmp_path)
    # Lenient (default): NULL, including count.
    assert zonal_stat(con, path, GEOM_DISJOINT, [1, "count"]) is None
    assert zonal_stats_all(con, path, GEOM_DISJOINT, [1]) is None
    # Strict (lenient => false): the 7-arg overload trails (all_touched,
    # exclude_no_data, lenient).
    with pytest.raises(Exception, match="does not intersect"):
        zonal_stat(con, path, GEOM_DISJOINT, [1, "count", False, True, False])


def test_bbox_overlapping_but_geometry_disjoint_is_no_intersection(con, tmp_path):
    path, _ = fixture_raster(tmp_path)

    # Premise: the roi's bounding box overlaps the raster extent, but the
    # geometry is disjoint from it (unlike GEOM_DISJOINT, whose bbox misses too).
    ox, px, _, oy, _, py = GDAL_TRANSFORM
    raster_extent = box(ox, oy + py * HEIGHT, ox + px * WIDTH, oy)
    geom = shapely.from_wkt(GEOM_DISJOINT_BBOX)
    assert box(*geom.bounds).intersects(raster_extent), "bbox must overlap the raster"
    assert geom.disjoint(raster_extent), "geometry must be disjoint from the raster"

    # Lenient (default): NULL, not count 0 — a true no-intersection case.
    assert zonal_stat(con, path, GEOM_DISJOINT_BBOX, [1, "count"]) is None
    assert zonal_stats_all(con, path, GEOM_DISJOINT_BBOX, [1]) is None
    # Strict: errors, exactly like the fully-disjoint roi.
    with pytest.raises(Exception, match="does not intersect"):
        zonal_stat(con, path, GEOM_DISJOINT_BBOX, [1, "count", False, True, False])


def test_unknown_statistic_errors(con, tmp_path):
    path, _ = fixture_raster(tmp_path)
    with pytest.raises(Exception, match="unknown statistic"):
        zonal_stat(con, path, GEOM_RECT, [1, "nonsense"])


def test_implicit_band_on_multiband_raster_errors(con, tmp_path):
    # A 2-band raster: the band-less overloads must error rather than default to
    # band 1 (this deliberately diverges from Sedona Spark).
    data = random_raster_data("int32", bands=2, height=HEIGHT, width=WIDTH, seed=3)
    path = tmp_path / "multiband.tif"
    write_geotiff(path, data, gdal_transform=GDAL_TRANSFORM)

    # RS_ZonalStats 3-arg (raster, roi, stat_type): band implicit.
    with pytest.raises(Exception, match="2 bands"):
        zonal_stat(con, path, GEOM_RECT, ["count"])
    # RS_ZonalStatsAll 2-arg (raster, roi): band implicit.
    with pytest.raises(Exception, match="2 bands"):
        zonal_stats_all(con, path, GEOM_RECT, [])
    # Naming the band resolves the ambiguity.
    assert zonal_stat(con, path, GEOM_RECT, [1, "count"]) > 0.0


def test_sql_text_smoke(con, tmp_path):
    """One raw-SQL invocation per function keeps the parser path covered."""
    path, band = fixture_raster(tmp_path)
    expected = numpy_reference(band, GEOM_RECT, all_touched=False, exclude_no_data=True)

    single = con.sql(
        "SELECT RS_ZonalStats(RS_FromPath($1), ST_GeomFromText($2), 1, 'sum') AS r",
        params=(str(path), GEOM_RECT),
    ).to_arrow_table()
    assert single["r"][0].as_py() == expected["sum"]

    everything = con.sql(
        "SELECT RS_ZonalStatsAll(RS_FromPath($1), ST_GeomFromText($2), 1) AS r",
        params=(str(path), GEOM_RECT),
    ).to_arrow_table()
    assert everything["r"][0].as_py()["count"] == expected["count"]
