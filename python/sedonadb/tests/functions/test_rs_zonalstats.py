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

The fixture raster is CRS-less (so nothing reprojects and pixel selection is
bit-comparable). The reference rasterizes the zone with `rasterio.features`
(the same GDAL rasterizer the kernel uses) and reduces the selected pixels with
numpy; exact-selection statistics (count, sum, min, max, median, mode) are
compared exactly and the float-accumulation ones (mean, variance, stddev) with
a tolerance.

rasterio is required to write the fixture GeoTIFF, so the whole module skips
when it is unavailable rather than importing it at module scope.
"""

import json

import numpy as np
import pyarrow as pa
import pytest

pytest.importorskip("rasterio")

from sedonadb.raster_testing import random_raster_data, write_geotiff  # noqa: E402

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


def numpy_reference(band, wkt, *, all_touched, exclude_nodata):
    """Reference statistics over the pixels the zone selects, via rasterio+numpy.

    Returns a dict of every statistic, or the sentinel string ``"empty"`` when
    the selection is empty (the caller maps that to count 0 / NULLs).
    """
    import rasterio.features
    import shapely
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
    if exclude_nodata:
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


def zonal_stat(con, path, wkt, stat, options=None):
    """RS_ZonalStats over a one-row table (values as columns, not literals)."""
    columns = {
        "path": pa.array([str(path)], pa.utf8()),
        "wkt": pa.array([wkt], pa.utf8()),
        "stat": pa.array([stat], pa.utf8()),
    }
    if options is not None:
        columns["options"] = pa.array([options], pa.utf8())
    df = con.create_data_frame(pa.table(columns))
    raster = df.path.funcs.rs_frompath()
    geom = con.funcs.st_geomfromtext(df.wkt)
    args = [geom, df.stat] + ([df.options] if options is not None else [])
    table = df.select(r=raster.funcs.rs_zonalstats(*args)).to_arrow_table()
    return table["r"][0].as_py()


def zonal_stats_all(con, path, wkt, options=None):
    """RS_ZonalStatsAll over a one-row table; returns the struct as a dict."""
    columns = {
        "path": pa.array([str(path)], pa.utf8()),
        "wkt": pa.array([wkt], pa.utf8()),
    }
    if options is not None:
        columns["options"] = pa.array([options], pa.utf8())
    df = con.create_data_frame(pa.table(columns))
    raster = df.path.funcs.rs_frompath()
    geom = con.funcs.st_geomfromtext(df.wkt)
    args = [geom] + ([df.options] if options is not None else [])
    table = df.select(r=raster.funcs.rs_zonalstatsall(*args)).to_arrow_table()
    return table["r"][0].as_py()


@pytest.mark.parametrize("stat", STATS)
@pytest.mark.parametrize("all_touched", [False, True])
def test_single_stat_matches_numpy(con, tmp_path, stat, all_touched):
    path, band = fixture_raster(tmp_path)
    options = json.dumps({"band": 1, "all_touched": all_touched})
    expected = numpy_reference(
        band, GEOM_RECT, all_touched=all_touched, exclude_nodata=True
    )
    assert expected != "empty", "GEOM_RECT should select pixels"

    got = zonal_stat(con, path, GEOM_RECT, stat, options)
    if stat in EXACT_STATS:
        assert got == expected[stat]
    else:
        assert got == pytest.approx(expected[stat])


def test_all_struct_matches_numpy(con, tmp_path):
    path, band = fixture_raster(tmp_path)
    expected = numpy_reference(band, GEOM_RECT, all_touched=False, exclude_nodata=True)
    got = zonal_stats_all(con, path, GEOM_RECT, json.dumps({"band": 1}))

    assert got["count"] == expected["count"]
    for stat in EXACT_STATS - {"count"}:
        assert got[stat] == expected[stat]
    for stat in ("mean", "variance", "stddev"):
        assert got[stat] == pytest.approx(expected[stat])


def test_exclude_nodata_default_and_disabled(con, tmp_path):
    path, band = fixture_raster(tmp_path)
    # Default excludes nodata; disabling it keeps those pixels, raising count.
    excluded = numpy_reference(band, GEOM_RECT, all_touched=False, exclude_nodata=True)
    included = numpy_reference(band, GEOM_RECT, all_touched=False, exclude_nodata=False)
    assert included["count"] > excluded["count"]

    assert (
        zonal_stat(con, path, GEOM_RECT, "count", json.dumps({"band": 1}))
        == excluded["count"]
    )
    assert (
        zonal_stat(
            con,
            path,
            GEOM_RECT,
            "count",
            json.dumps({"band": 1, "exclude_nodata": False}),
        )
        == included["count"]
    )


def test_sliver_selects_nothing_unless_all_touched(con, tmp_path):
    path, _ = fixture_raster(tmp_path)
    # The zone overlaps the raster but covers no pixel center: count 0, rest NULL.
    assert zonal_stat(con, path, GEOM_SLIVER, "count", json.dumps({"band": 1})) == 0.0
    assert zonal_stat(con, path, GEOM_SLIVER, "sum", json.dumps({"band": 1})) is None
    # all_touched picks up the pixels it crosses.
    touched = zonal_stat(
        con, path, GEOM_SLIVER, "count", json.dumps({"band": 1, "all_touched": True})
    )
    assert touched > 0.0


def test_no_intersection_is_null_when_lenient_and_errors_when_strict(con, tmp_path):
    path, _ = fixture_raster(tmp_path)
    # Lenient (default): NULL, including count.
    assert (
        zonal_stat(con, path, GEOM_DISJOINT, "count", json.dumps({"band": 1})) is None
    )
    assert zonal_stats_all(con, path, GEOM_DISJOINT, json.dumps({"band": 1})) is None
    # Strict: errors.
    with pytest.raises(Exception, match="does not intersect"):
        zonal_stat(
            con, path, GEOM_DISJOINT, "count", json.dumps({"band": 1, "lenient": False})
        )


def test_unknown_statistic_errors(con, tmp_path):
    path, _ = fixture_raster(tmp_path)
    with pytest.raises(Exception, match="unknown statistic"):
        zonal_stat(con, path, GEOM_RECT, "nonsense", json.dumps({"band": 1}))


def test_sql_text_smoke(con, tmp_path):
    """One raw-SQL invocation per function keeps the parser path covered."""
    path, band = fixture_raster(tmp_path)
    expected = numpy_reference(band, GEOM_RECT, all_touched=False, exclude_nodata=True)

    single = con.sql(
        "SELECT RS_ZonalStats(RS_FromPath($1), ST_GeomFromText($2), 'sum', '{\"band\": 1}') AS r",
        params=(str(path), GEOM_RECT),
    ).to_arrow_table()
    assert single["r"][0].as_py() == expected["sum"]

    everything = con.sql(
        "SELECT RS_ZonalStatsAll(RS_FromPath($1), ST_GeomFromText($2), '{\"band\": 1}') AS r",
        params=(str(path), GEOM_RECT),
    ).to_arrow_table()
    assert everything["r"][0].as_py()["count"] == expected["count"]
