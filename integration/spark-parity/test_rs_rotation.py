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

Both engines compute the angle of the grid's column-axis direction
`(scaleX, skewY)`: `acos(scaleX / hypot(scaleX, skewY))`, negated when
`skewY > 0`. (SedonaDB previously computed `atan2(-skewX, scaleX)`, which
agreed only while |skewX| == |skewY| and misreported rigid rotations of
non-square pixels; it was aligned to Sedona Spark's formula alongside
these tests.) Axis-aligned grids compare exactly, including the sign of
the zero; rotated grids can differ by one ulp across runtimes (Rust's
acos vs the JVM's), so those queries round to 12 digits — the agreement
under test is semantic, not bit-level.
"""

import math

import pytest

from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark


def _rigid_transform(theta, pixel_x, pixel_y):
    """The GDAL transform of a north-up grid with pixel (pixel_x, pixel_y)
    rigidly rotated by theta: R(theta) @ diag(pixel_x, -pixel_y)."""
    return (
        100.0,
        pixel_x * math.cos(theta),
        pixel_y * math.sin(theta),
        500.0,
        pixel_x * math.sin(theta),
        -pixel_y * math.cos(theta),
    )


@pytest.mark.parametrize(
    "transform",
    [
        pytest.param((100.0, 2.0, 0.0, 500.0, 0.0, -3.0), id="north-up"),
        pytest.param((100.0, 2.0, 0.0, 482.0, 0.0, 3.0), id="south-up"),
    ],
)
def test_rs_rotation_axis_aligned(transform, tmp_path):
    """An axis-aligned raster has exactly zero rotation on both engines —
    positive zero from both, so even the stringified sign agrees."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(
            "rot_axis", tmp_path / "rot_axis.tif", gdal_transform=transform
        )
    compare("SELECT RS_Rotation(rast) FROM rot_axis", sedona, spark, expected=0.0)


@pytest.mark.parametrize(
    "pixel", [(2.0, 2.0), (2.0, 3.0)], ids=["square", "non-square"]
)
@pytest.mark.parametrize("theta", [math.pi / 6, -math.pi / 6], ids=["ccw", "cw"])
def test_rs_rotation_rigid(theta, pixel, tmp_path):
    """A rigid rotation by theta reads back as -theta from both engines,
    regardless of pixel shape."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(
            "rot_rigid",
            tmp_path / "rot_rigid.tif",
            gdal_transform=_rigid_transform(theta, *pixel),
        )
    sql = "SELECT ROUND(RS_Rotation(rast), 12) FROM rot_rigid"
    compare(sql, sedona, spark, expected=round(-theta, 12))


def test_rs_rotation_shear(tmp_path):
    """Under shear the angle follows the column axis (scaleX, skewY): both
    engines report -acos(scaleX / hypot(scaleX, skewY))."""
    transform = (100.0, 0.2, 0.06, 500.0, 0.08, -0.3)
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(
            "rot_shear", tmp_path / "rot_shear.tif", gdal_transform=transform
        )
    sql = "SELECT ROUND(RS_Rotation(rast), 12) FROM rot_shear"
    expected = round(-math.acos(0.2 / math.hypot(0.2, 0.08)), 12)
    compare(sql, sedona, spark, expected=expected)
