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
"""SedonaDB vs Sedona Spark parity for RS_Rotation.

One shared SQL string per case, compared with the harness-level
`sedonadb.testing.compare` — same conventions as the rest of the parity
suite (one module per RS_ function, xfail(reason=...) as the catalog of
known divergences).
"""

import math

import pytest

from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark

#
# A rigid rotation by theta of a grid with pixel size (px, py) has the affine
# R(theta) @ diag(px, -py), i.e. the GDAL transform
# (ox, px*cos, py*sin, oy, px*sin, -py*cos). SedonaDB computes the angle as
# atan2(-skewX, scaleX); Sedona Spark as acos(scaleX / hypot(scaleX, skewY)),
# negated when skewY > 0. The formulas coincide exactly when
# |skewX| == |skewY| — axis-aligned grids and rigid rotations of square
# pixels — and separate everywhere else (see the xfails).
_COS30, _SIN30 = math.cos(math.pi / 6), math.sin(math.pi / 6)


def test_rs_rotation_north_up_is_zero(tmp_path):
    """An axis-aligned raster has zero rotation on both engines. The `+ 0.0`
    normalizes the zero's sign, which the engines disagree on
    (test_rs_rotation_zero_sign)."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view("rot_nu", tmp_path / "rot_nu.tif")
    sql = "SELECT RS_Rotation(rast) + 0.0 FROM rot_nu"
    compare(sql, sedona, spark, expected=0.0)


@pytest.mark.parametrize("theta", [math.pi / 6, -math.pi / 6], ids=["ccw", "cw"])
def test_rs_rotation_rigid_square_pixels(theta, tmp_path):
    """A rigid rotation of square 2x2 pixels reads back as -theta from both
    engines. The raw doubles differ by one ulp across runtimes (Rust's atan2
    vs the JVM's acos), so the query rounds to 12 digits — the agreement
    under test is semantic, not bit-level."""
    transform = (
        100.0,
        2 * math.cos(theta),
        2 * math.sin(theta),
        500.0,
        2 * math.sin(theta),
        -2 * math.cos(theta),
    )
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(
            "rot_sq", tmp_path / "rot_sq.tif", gdal_transform=transform
        )
    sql = "SELECT ROUND(RS_Rotation(rast), 12) FROM rot_sq"
    compare(sql, sedona, spark, expected=round(-theta, 12))


@pytest.mark.parametrize(
    "transform",
    [
        pytest.param((100.0, 2.0, 0.0, 500.0, 0.0, -3.0), id="north-up"),
        pytest.param((100.0, 2.0, 0.0, 482.0, 0.0, 3.0), id="south-up"),
    ],
)
@pytest.mark.xfail(
    reason="SedonaDB's atan2(-skewX, scaleX) yields IEEE negative zero for an "
    "axis-aligned raster; Sedona Spark returns positive zero"
)
def test_rs_rotation_zero_sign(transform, tmp_path):
    """The zero rotation of an axis-aligned raster reads back identically,
    including the sign of the zero (it stringifies visibly: '-0' vs '0')."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(
            "rot_zero", tmp_path / "rot_zero.tif", gdal_transform=transform
        )
    sql = "SELECT RS_Rotation(rast) FROM rot_zero"
    compare(sql, sedona, spark)


@pytest.mark.xfail(
    reason="SedonaDB's atan2(-skewX, scaleX) folds the row axis's skew into the "
    "reported angle, so a rigid 30-degree rotation of non-square 2x3 pixels "
    "reads -0.7137 instead of the true -pi/6; Sedona Spark's "
    "acos(scaleX / hypot(scaleX, skewY)) recovers -pi/6 exactly"
)
def test_rs_rotation_rigid_non_square_pixels(tmp_path):
    """A rigid rotation reads back as -theta regardless of pixel shape."""
    transform = (100.0, 2 * _COS30, 3 * _SIN30, 500.0, 2 * _SIN30, -3 * _COS30)
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(
            "rot_nsq", tmp_path / "rot_nsq.tif", gdal_transform=transform
        )
    sql = "SELECT RS_Rotation(rast) FROM rot_nsq"
    compare(sql, sedona, spark)


@pytest.mark.xfail(
    reason="under shear the formulas separate: SedonaDB's atan2(-skewX, scaleX) "
    "gives -0.2915 where Sedona Spark's acos(scaleX / hypot(scaleX, skewY)) "
    "gives -0.3805; they agree only while |skewX| == |skewY|"
)
def test_rs_rotation_shear(tmp_path):
    """A sheared raster's rotation reads back the same from both engines."""
    transform = (100.0, 0.2, 0.06, 500.0, 0.08, -0.3)
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(
            "rot_shear", tmp_path / "rot_shear.tif", gdal_transform=transform
        )
    sql = "SELECT RS_Rotation(rast) FROM rot_shear"
    compare(sql, sedona, spark)
