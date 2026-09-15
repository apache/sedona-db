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

"""SedonaDB vs Sedona Spark parity for RS_MakeEmptyRaster.

Only the two Spark-shaped forms (cell size; full affine + SRID) are compared:
the extent-geometry form is a SedonaDB extension with no Spark counterpart.
Anchored expectations are literal zero grids, so a shared no-op on both sides
cannot pass. Same construction and xfail policy as the other raster modules:
the two known divergences (an unknown band type, which Sedona Spark silently
maps to double and SedonaDB rejects; and ``num_bands = 0``, which SedonaDB
allows as a bandless grid template and Sedona Spark rejects) are
xfail-cataloged, the raising engine's error tripping the xfail.
"""

import numpy as np
import pytest

from sedonadb.raster_testing import DecodedRaster
from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark

# Sedona Spark's band type letters and the dtype each decodes to.
SPARK_TYPES = {
    "D": "float64",
    "F": "float32",
    "I": "int32",
    "S": "int16",
    "US": "uint16",
    "B": "uint8",
}


def test_cell_size_form():
    """Square pixels from an upper-left corner: scale_y is the negated cell
    size, no skew, float64 bands with no nodata."""
    sedona, spark = SedonaDB(), SedonaSpark()
    sql = "SELECT RS_MakeEmptyRaster(2, 6, 3, 10.0, 20.0, 2.5)"
    compare(
        sql,
        sedona,
        spark,
        expected=DecodedRaster(
            np.zeros((2, 3, 6), dtype="float64"),
            gdal_transform=(10.0, 2.5, 0.0, 20.0, 0.0, -2.5),
            nodata=[None, None],
        ),
    )


@pytest.mark.parametrize("code", list(SPARK_TYPES))
def test_affine_form(code):
    """Full geotransform with skew and an explicit band type."""
    sedona, spark = SedonaDB(), SedonaSpark()
    sql = f"SELECT RS_MakeEmptyRaster(1, '{code}', 5, 4, 100.0, 200.0, 2.0, -3.0, 0.5, 0.25, 3857)"
    compare(
        sql,
        sedona,
        spark,
        expected=DecodedRaster(
            np.zeros((1, 4, 5), dtype=SPARK_TYPES[code]),
            gdal_transform=(100.0, 2.0, 0.5, 200.0, 0.25, -3.0),
            nodata=[None],
        ),
    )


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT RS_SRID(RS_MakeEmptyRaster(1, 2, 2, 0.0, 0.0, 1.0))", "0"),
        (
            "SELECT RS_SRID(RS_MakeEmptyRaster(1, 2, 2, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 4326))",
            "4326",
        ),
        (
            "SELECT RS_SRID(RS_MakeEmptyRaster(1, 'B', 2, 2, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 32610))",
            "32610",
        ),
        ("SELECT RS_NumBands(RS_MakeEmptyRaster(3, 2, 2, 0.0, 0.0, 1.0))", "3"),
        (
            "SELECT RS_BandPixelType(RS_MakeEmptyRaster(1, 'US', 2, 2, 0.0, 0.0, 1.0), 1)",
            "UNSIGNED_16BITS",
        ),
    ],
)
def test_scalar_metadata(sql, expected):
    """The CRS does not survive the GeoTIFF transport, so SRID (and the
    band count and type) are compared as scalars."""
    sedona, spark = SedonaDB(), SedonaSpark()
    compare(sql, sedona, spark, expected=[(expected,)])


@pytest.mark.xfail(
    reason="Sedona Spark silently defaults an unknown bandDataType to double "
    "(RasterUtils.getDataTypeCode falls through to 5); SedonaDB raises "
    "'Unsupported pixelType'"
)
def test_unknown_band_type():
    sedona, spark = SedonaDB(), SedonaSpark()
    sql = "SELECT RS_BandPixelType(RS_MakeEmptyRaster(1, 'complex128', 2, 2, 0.0, 0.0, 1.0), 1)"
    compare(sql, sedona, spark, expected=[("REAL_64BITS",)])


@pytest.mark.xfail(
    reason="SedonaDB allows num_bands = 0, a bandless grid template; Sedona "
    "Spark's RasterFactory.createBandedRaster raises for zero bands"
)
def test_zero_bands():
    sedona, spark = SedonaDB(), SedonaSpark()
    sql = "SELECT RS_NumBands(RS_MakeEmptyRaster(0, 2, 2, 0.0, 0.0, 1.0))"
    compare(sql, sedona, spark, expected=[("0",)])
