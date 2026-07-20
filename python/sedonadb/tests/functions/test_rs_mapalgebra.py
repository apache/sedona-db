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

"""RS_MapAlgebra cross-checked against numpy.

The reference is numpy: each expression evaluated element-wise over the same
band array must match RS_MapAlgebra pixel-for-pixel. Rasters are built in memory
from numpy arrays (no rasterio needed), functions run through the generated
`rst.map_algebra` accessor over a real query column (not a constant-folded
literal), and one SQL-text invocation covers the parser path. The operation is
deterministic on integer and float inputs, so comparisons are exact.
"""

import numpy as np
import pytest

from sedonadb.raster import Raster

# GDAL-order geotransform: origin (100, 500), 2-wide by 3-tall north-up pixels.
GDAL_TRANSFORM = (100.0, 2.0, 0.0, 500.0, 0.0, -3.0)


def _map_algebra_numpy(con, data, expr, options=None):
    """Run RS_MapAlgebra through the `rst.map_algebra` accessor over a raster
    column and return the single output band as a numpy array."""
    raster = Raster.from_numpy(data, transform=GDAL_TRANSFORM)
    df = con.sql("SELECT $1 AS rast", params=(raster,))
    method = df.rast.rst
    expr_out = (
        method.map_algebra(expr, options=options)
        if options is not None
        else method.map_algebra(expr)
    )
    table = df.select(r=expr_out).to_arrow_table()
    return Raster(table["r"], 0).to_numpy()[0]


def test_map_algebra_scale_and_offset_matches_numpy(con):
    data = (np.arange(7 * 5).reshape(7, 5) * 1.5).astype(np.float64)
    got = _map_algebra_numpy(con, data, "rast0 * 2 + 1", '{"pixel_type": "D"}')
    np.testing.assert_array_equal(got, data * 2 + 1)


def test_map_algebra_defaults_to_input_dtype(con):
    # No options: the output inherits the uint8 input type. Values stay in range.
    data = np.arange(7 * 5, dtype=np.uint8).reshape(7, 5)
    got = _map_algebra_numpy(con, data, "rast0 + 1")
    assert got.dtype == np.uint8
    np.testing.assert_array_equal(got, (data.astype(np.int64) + 1).astype(np.uint8))


def test_map_algebra_math_function_matches_numpy(con):
    # sqrt is correctly rounded (IEEE-754), so the result is bit-exact in f64.
    data = (np.arange(1, 7 * 5 + 1).reshape(7, 5)).astype(np.float64)
    got = _map_algebra_numpy(con, data, "math::sqrt(rast0)", '{"pixel_type": "D"}')
    np.testing.assert_array_equal(got, np.sqrt(data))


def test_map_algebra_pixel_coordinates_matches_numpy(con):
    # `x + y * width` numbers pixels in row-major order.
    height, width = 6, 4
    data = np.zeros((height, width), dtype=np.float64)
    got = _map_algebra_numpy(con, data, "x + y * width", '{"pixel_type": "I"}')
    expected = np.arange(height * width, dtype=np.int32).reshape(height, width)
    assert got.dtype == np.int32
    np.testing.assert_array_equal(got, expected)


def test_map_algebra_lossy_nodata_errors(con):
    # 0.5 is not representable in a uint8 band, so recording it as nodata errors.
    data = np.arange(6, dtype=np.uint8).reshape(2, 3)
    with pytest.raises(Exception, match="nodata"):
        _map_algebra_numpy(con, data, "rast0", '{"nodata": 0.5}')


def test_map_algebra_sql_smoke(con):
    # Parser-path smoke: one SQL-text invocation, materialized to_arrow_table.
    data = (np.arange(6).reshape(2, 3) * 1.0).astype(np.float64)
    raster = Raster.from_numpy(data, transform=GDAL_TRANSFORM)
    table = con.sql(
        "SELECT RS_MapAlgebra($1, 'rast0 * 2 + 1') AS r",
        params=(raster,),
    ).to_arrow_table()
    got = Raster(table["r"], 0).to_numpy()[0]
    np.testing.assert_array_equal(got, data * 2 + 1)
