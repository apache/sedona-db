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
"""SedonaDB vs Sedona Spark parity for RS_Clip.

Both engines share the positional ladder
`(raster, band, geom[, allTouched[, noDataValue[, crop[, lenient]]]])` —
a numeric fourth argument is refused identically — and the arities that
carry a noDataValue agree bit-for-bit: crop keeps the covered subgrid on
its own origin, crop=false keeps the source grid and masks outside the
roi, and a disjoint roi is NULL under the default lenient behavior.
Sedona Spark raises when noDataValue is omitted, where SedonaDB defaults
it — the xfails catalog that.
"""

import numpy as np
import pytest

from sedonadb.raster_testing import DecodedRaster, random_raster_data
from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark

# Selects rows 1-4 x cols 1-4 of the standard grid under the centre-in rule.
RECT = "POLYGON((102 485, 110 485, 110 497, 102 497, 102 485))"
DISJOINT = "POLYGON((300 300, 310 300, 310 310, 300 310, 300 300))"


def _engines(name, tmp_path, **kwargs):
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(name, tmp_path / f"{name}.tif", **kwargs)
    return sedona, spark


def _band(band=1):
    data = random_raster_data("uint8", bands=2, height=6, width=7)
    return data[band - 1]


@pytest.mark.parametrize("band", [1, 2])
def test_rs_clip_crop(band, tmp_path):
    """Cropping to the aligned RECT keeps the covered 4x4 subgrid on its own
    origin, with the given noDataValue as the band nodata."""
    sedona, spark = _engines("clip_src", tmp_path, nodata=200.0)
    sql = (
        f"SELECT RS_Clip(rast, {band}, ST_GeomFromWKT('{RECT}'), false, 99) "
        "FROM clip_src"
    )
    anchor = DecodedRaster(
        _band(band)[np.newaxis, 1:5, 1:5],
        nodata=[99.0],
        bbox=(102.0, 485.0, 110.0, 497.0),
    )
    compare(sql, sedona, spark, expected=anchor)


def test_rs_clip_no_crop(tmp_path):
    """crop=false keeps the source grid and masks everything outside the roi
    with the noDataValue."""
    sedona, spark = _engines("clip_nc_src", tmp_path, nodata=200.0)
    sql = (
        f"SELECT RS_Clip(rast, 1, ST_GeomFromWKT('{RECT}'), false, 99, false) "
        "FROM clip_nc_src"
    )
    pixels = np.full((1, 6, 7), 99, dtype="uint8")
    pixels[0, 1:5, 1:5] = _band()[1:5, 1:5]
    anchor = DecodedRaster(pixels, nodata=[99.0], bbox=(100.0, 482.0, 114.0, 500.0))
    compare(sql, sedona, spark, expected=anchor)


def test_rs_clip_all_touched_sliver(tmp_path):
    """A sliver holding no pixel centre keeps only pixel (1, 1) under
    all_touched; the uncropped output makes the selection visible."""
    sedona, spark = _engines("clip_at_src", tmp_path)
    sliver = (
        "POLYGON((102.2 494.9, 103.8 494.9, 103.8 494.1, 102.2 494.1, 102.2 494.9))"
    )
    sql = (
        f"SELECT RS_Clip(rast, 1, ST_GeomFromWKT('{sliver}'), true, 99, false) "
        "FROM clip_at_src"
    )
    pixels = np.full((1, 6, 7), 99, dtype="uint8")
    pixels[0, 1, 1] = _band()[1, 1]
    anchor = DecodedRaster(pixels, nodata=[99.0], bbox=(100.0, 482.0, 114.0, 500.0))
    compare(sql, sedona, spark, expected=anchor)


def test_rs_clip_disjoint_is_null_when_lenient(tmp_path):
    """A roi that misses the raster yields a NULL raster under the default
    lenient behavior on both engines."""
    sedona, spark = _engines("clip_dj_src", tmp_path)
    sql = (
        f"SELECT RS_Clip(rast, 1, ST_GeomFromWKT('{DISJOINT}'), false, 99) "
        "FROM clip_dj_src"
    )
    compare(sql, sedona, spark)


def test_rs_clip_numeric_fourth_argument_rejected(tmp_path):
    """Both engines refuse a numeric fourth argument — the ladder puts
    allTouched there, not noDataValue. Parity on refusal."""
    sedona, spark = _engines("clip_num_src", tmp_path)
    sql = f"SELECT RS_Clip(rast, 1, ST_GeomFromWKT('{RECT}'), 99) FROM clip_num_src"
    for eng in (sedona, spark):
        with pytest.raises(Exception):
            eng.decode_raster_result(sql)


@pytest.mark.parametrize(
    "args", [pytest.param("", id="3-arg"), pytest.param(", true", id="4-arg")]
)
@pytest.mark.xfail(
    reason="Sedona Spark raises IllegalArgumentException when noDataValue is "
    "omitted; SedonaDB defaults it to the source band's nodata"
)
def test_rs_clip_without_nodata_value(args, tmp_path):
    """The arities that omit noDataValue clip the same way on both engines."""
    sedona, spark = _engines("clip_short_src", tmp_path, nodata=200.0)
    sql = f"SELECT RS_Clip(rast, 1, ST_GeomFromWKT('{RECT}'){args}) FROM clip_short_src"
    compare(sql, sedona, spark)
