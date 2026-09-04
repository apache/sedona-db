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
"""SedonaDB vs Sedona Spark parity for RS_ZonalStats.

One shared SQL string per case (the roi travels as `ST_GeomFromWKT`), compared
with the harness-level `sedonadb.testing.compare` and anchored against numpy on
the seeded fixture pixels. Every statistic — including the float-accumulating
mean, variance, and stddev — matched numpy's exact double bit-for-bit on both
engines when probed, so the anchors are exact with no tolerance anywhere:
both engines compute the sample (ddof=1) variance/stddev, average the two
middle values for an even-count median, and break mode ties toward the higher
value.

The roi is a rectangle on pixel boundaries, so which pixels are selected is
unambiguous under the default centre-in rule: x in (102, 110) and
y in (485, 497) selects the 4x4 block rows 1-4 x cols 1-4 of the standard
7x6 grid of 2x3 pixels.
"""

import numpy as np
import pytest

from sedonadb.raster_testing import random_raster_data
from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark

# The 4x4-block roi described in the module docstring.
RECT = "POLYGON((102 485, 110 485, 110 497, 102 497, 102 485))"

# A sliver between pixel centres: covered by parts of pixel (1, 1) but
# containing no centre, so only all_touched selects it.
SLIVER = "POLYGON((102.2 494.9, 103.8 494.9, 103.8 494.1, 102.2 494.1, 102.2 494.9))"

DISJOINT = "POLYGON((300 300, 310 300, 310 310, 300 310, 300 300))"


def _selection(plants=None, band=1):
    """The float64 pixels RECT selects from the standard seeded grid."""
    data = random_raster_data("uint8", bands=2, height=6, width=7, plants=plants)
    return data[band - 1, 1:5, 1:5].astype("float64").ravel()


def _engines(name, tmp_path, **kwargs):
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(name, tmp_path / f"{name}.tif", **kwargs)
    return sedona, spark


STAT_REFERENCES = {
    "count": lambda v: len(v),
    "sum": lambda v: float(v.sum()),
    "mean": lambda v: float(v.mean()),
    "median": lambda v: float(np.median(v)),
    "mode": lambda v: float(v.max()),  # every value unique in the seeded block
    "stddev": lambda v: float(v.std(ddof=1)),
    "variance": lambda v: float(v.var(ddof=1)),
    "min": lambda v: float(v.min()),
    "max": lambda v: float(v.max()),
}


@pytest.mark.parametrize("stat", list(STAT_REFERENCES))
def test_rs_zonalstats_statistics(stat, tmp_path):
    """Each statistic of the 16 selected pixels agrees across engines and with
    numpy exactly (sample variance/stddev; see the module docstring)."""
    sedona, spark = _engines("zs_src", tmp_path)
    sql = (
        f"SELECT RS_ZonalStats(rast, ST_GeomFromWKT('{RECT}'), 1, '{stat}') FROM zs_src"
    )
    compare(sql, sedona, spark, expected=STAT_REFERENCES[stat](_selection()))


@pytest.mark.parametrize(
    "alias,canonical", [("avg", "mean"), ("average", "mean"), ("sd", "stddev")]
)
def test_rs_zonalstats_stat_aliases(alias, canonical, tmp_path):
    """Both engines accept the alias spellings for mean and stddev."""
    sedona, spark = _engines("zs_alias_src", tmp_path)
    sql = f"SELECT RS_ZonalStats(rast, ST_GeomFromWKT('{RECT}'), 1, '{alias}') FROM zs_alias_src"
    compare(sql, sedona, spark, expected=STAT_REFERENCES[canonical](_selection()))


def test_rs_zonalstats_band_2(tmp_path):
    """The band argument addresses band 2's pixels, not band 1's."""
    sedona, spark = _engines("zs_b2_src", tmp_path)
    sql = (
        f"SELECT RS_ZonalStats(rast, ST_GeomFromWKT('{RECT}'), 2, 'sum') FROM zs_b2_src"
    )
    compare(sql, sedona, spark, expected=float(_selection(band=2).sum()))


def test_rs_zonalstats_three_arg_single_band(tmp_path):
    """The band-less 3-argument form resolves unambiguously on a single-band
    raster in both engines."""
    sedona, spark = _engines("zs_3a_src", tmp_path, bands=1)
    sql = f"SELECT RS_ZonalStats(rast, ST_GeomFromWKT('{RECT}'), 'sum') FROM zs_3a_src"
    compare(sql, sedona, spark, expected=float(_selection().sum()))


def test_rs_zonalstats_mode_conventions(tmp_path):
    """Mode returns the most frequent value; on a tie both engines return the
    higher of the tied values."""
    unique_plants = {(1, 1): 150, (2, 2): 150, (3, 3): 150}
    tie_plants = {(1, 1): 40, (2, 2): 40, (3, 3): 90, (4, 4): 90}
    sedona, spark = _engines("zs_mode_src", tmp_path, plants=unique_plants)
    sql = "SELECT RS_ZonalStats(rast, ST_GeomFromWKT('%s'), 1, 'mode') FROM zs_mode_src"
    compare(sql % RECT, sedona, spark, expected=150.0)
    for eng in (sedona, spark):
        eng.create_random_raster_view(
            "zs_tie_src", tmp_path / "zs_tie_src.tif", plants=tie_plants
        )
    sql = f"SELECT RS_ZonalStats(rast, ST_GeomFromWKT('{RECT}'), 1, 'mode') FROM zs_tie_src"
    compare(sql, sedona, spark, expected=90.0)


def test_rs_zonalstats_exclude_nodata(tmp_path):
    """A nodata pixel inside the roi is excluded by default (exclude_no_data
    defaults to true) and included when the flag is passed false. The odd
    15-pixel selection also pins the exact-middle median."""
    plants = {(2, 2): 200.0}
    sedona, spark = _engines("zs_nd_src", tmp_path, nodata=200.0, plants=plants)
    base = (
        "SELECT RS_ZonalStats(rast, ST_GeomFromWKT('%s'), 1, 'count'%s) FROM zs_nd_src"
    )
    compare(base % (RECT, ""), sedona, spark, expected=15)
    compare(base % (RECT, ", false, false"), sedona, spark, expected=16)
    kept = _selection(plants=plants)
    kept = np.sort(kept[kept != 200.0])
    sql = f"SELECT RS_ZonalStats(rast, ST_GeomFromWKT('{RECT}'), 1, 'median') FROM zs_nd_src"
    compare(sql, sedona, spark, expected=float(np.median(kept)))


def test_rs_zonalstats_all_touched(tmp_path):
    """A sliver holding no pixel centre selects nothing under the default
    centre-in rule and exactly one pixel with all_touched."""
    sedona, spark = _engines("zs_at_src", tmp_path)
    base = (
        "SELECT RS_ZonalStats(rast, ST_GeomFromWKT('%s'), 1, 'count'%s) FROM zs_at_src"
    )
    compare(base % (SLIVER, ""), sedona, spark, expected=0)
    compare(base % (SLIVER, ", true"), sedona, spark, expected=1)


def test_rs_zonalstats_disjoint_is_null_when_lenient(tmp_path):
    """A roi that misses the raster yields NULL under the default lenient
    behavior on both engines."""
    sedona, spark = _engines("zs_dj_src", tmp_path)
    sql = f"SELECT RS_ZonalStats(rast, ST_GeomFromWKT('{DISJOINT}'), 1, 'sum') FROM zs_dj_src"
    compare(sql, sedona, spark, expected=[(None,)])


@pytest.mark.parametrize(
    "sql_suffix",
    [
        pytest.param(
            f"ST_GeomFromWKT('{DISJOINT}'), 1, 'sum', false, true, false",
            id="disjoint-strict",
        ),
        pytest.param(f"ST_GeomFromWKT('{RECT}'), 1, 'p95'", id="unknown-stat"),
        pytest.param(f"ST_GeomFromWKT('{RECT}'), 0, 'sum'", id="band-0"),
        pytest.param(f"ST_GeomFromWKT('{RECT}'), 3, 'sum'", id="band-out-of-range"),
    ],
)
def test_rs_zonalstats_rejected(sql_suffix, tmp_path):
    """Both engines refuse a disjoint roi with lenient=false, an unknown
    statistic, and an out-of-range band. Error types and messages differ, so
    parity here is parity on refusal."""
    sedona, spark = _engines("zs_rej_src", tmp_path)
    sql = f"SELECT RS_ZonalStats(rast, {sql_suffix}) FROM zs_rej_src"
    for eng in (sedona, spark):
        with pytest.raises(Exception):
            # result_to_tuples forces collection: Sedona Spark's
            # execute_and_collect is lazy and would not raise on its own.
            eng.result_to_tuples(eng.execute_and_collect(sql))


@pytest.mark.xfail(
    reason="SedonaDB requires the band argument on a multi-band raster; "
    "Sedona Spark's band-less form defaults to band 1"
)
def test_rs_zonalstats_three_arg_multiband(tmp_path):
    """The band-less 3-argument form gets the same answer from both engines on
    a multi-band raster."""
    sedona, spark = _engines("zs_3am_src", tmp_path)
    sql = f"SELECT RS_ZonalStats(rast, ST_GeomFromWKT('{RECT}'), 'sum') FROM zs_3am_src"
    compare(sql, sedona, spark)


@pytest.mark.xfail(
    reason="SedonaDB rejects a roi that carries a CRS when the raster has "
    "none; Sedona Spark computes as if the CRSs matched"
)
def test_rs_zonalstats_srid_roi_on_crsless_raster(tmp_path):
    """A roi with an SRID against a CRS-less raster gets the same treatment
    from both engines."""
    sedona, spark = _engines("zs_srid_src", tmp_path)
    sql = (
        "SELECT RS_ZonalStats(rast, ST_SetSRID(ST_GeomFromWKT("
        f"'{RECT}'), 4326), 1, 'sum') FROM zs_srid_src"
    )
    compare(sql, sedona, spark)
