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
"""SedonaDB vs Sedona Spark parity for RS_AsRaster.

The 6-argument form
`(geom, raster, pixelType, allTouched, value, noDataValue)` agrees
bit-for-bit: the output grid snaps the geometry's extent to the
reference raster's grid, burned cells hold `value` and the rest the
noDataValue. Sedona Spark raises when noDataValue is omitted where
SedonaDB defaults it, and the engines' default line-rasterization rules
differ — both xfail-cataloged.
"""

import numpy as np
import pytest

from sedonadb.raster_testing import DecodedRaster, write_random_geotiff
from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark

RECT = "POLYGON((102 485, 110 485, 110 497, 102 497, 102 485))"


def _engines(name, tmp_path, **kwargs):
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(name, tmp_path / f"{name}.tif", **kwargs)
    return sedona, spark


@pytest.mark.parametrize(
    "pixel_type,dtype", [("d", "float64"), ("b", "uint8")], ids=["double", "byte"]
)
def test_rs_asraster_rect(pixel_type, dtype, tmp_path):
    """Rasterizing the aligned RECT onto the standard grid burns value 7 into
    the covered 4x4 block on the geometry's own snapped grid."""
    sedona, spark = _engines("ar_src", tmp_path, bands=1)
    sql = (
        f"SELECT RS_AsRaster(ST_GeomFromWKT('{RECT}'), rast, "
        f"'{pixel_type}', false, 7, 99) FROM ar_src"
    )
    anchor = DecodedRaster(
        np.full((1, 4, 4), 7, dtype=dtype),
        nodata=[99.0],
        bbox=(102.0, 485.0, 110.0, 497.0),
    )
    compare(sql, sedona, spark, expected=anchor)


@pytest.mark.parametrize(
    "args",
    [
        pytest.param("", id="3-arg"),
        pytest.param(", true", id="4-arg"),
        pytest.param(", false, 7", id="5-arg"),
    ],
)
@pytest.mark.xfail(
    reason="Sedona Spark raises IllegalArgumentException when noDataValue is "
    "omitted; SedonaDB defaults it"
)
def test_rs_asraster_without_nodata_value(args, tmp_path):
    """The arities that omit noDataValue rasterize the same way on both
    engines."""
    sedona, spark = _engines("ar_short_src", tmp_path, bands=1)
    sql = (
        f"SELECT RS_AsRaster(ST_GeomFromWKT('{RECT}'), rast, 'd'{args}) "
        "FROM ar_short_src"
    )
    compare(sql, sedona, spark)


@pytest.mark.xfail(
    reason="the default line rules differ: SedonaDB (GDAL) burns only "
    "centre/diamond crossings (10 cells here); Sedona Spark burns every "
    "traversed cell (17) — apache/sedona#3322, the divergence the zonal "
    "suite catalogs"
)
def test_rs_asraster_line_default_rule(tmp_path):
    """A segment that never crosses a lattice point burns the same cells on
    both engines under the default rule. The burned count travels out through
    RS_ZonalStats over the full grid."""
    path = tmp_path / "unit.tif"
    write_random_geotiff(
        path,
        "uint8",
        bands=1,
        height=20,
        width=20,
        gdal_transform=(0.0, 1.0, 0.0, 20.0, 0.0, -1.0),
    )
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_raster_view("ar_line_src", path)
    sql = (
        "SELECT RS_ZonalStats(RS_AsRaster(ST_GeomFromWKT("
        "'LINESTRING (1.3 2.7, 8.6 11.4)'), rast, 'd', false, 1, 0), "
        "ST_GeomFromWKT('POLYGON((0 0, 20 0, 20 20, 0 20, 0 0))'), "
        "1, 'sum', false, false) FROM ar_line_src"
    )
    compare(sql, sedona, spark)
