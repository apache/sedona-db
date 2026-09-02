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
"""SedonaDB vs Sedona Spark parity for `RS_GeoTransform`.

Each test runs one shared SQL string on both engines and asserts they agree
via the harness-level `compare(sql, sedona, spark, expected=...)`; see the
README for the suite's conventions. The anchors here repeat the
implementation's floating-point operation order (both engines compute e.g.
sqrt(a*a + b*b) then divide before acos), so every expected value is
bit-exact rather than approximate.
"""

import math

from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark


def test_rs_geotransform_north_up(tmp_path):
    """Both engines decompose a north-up geotransform into the same struct.

    The default fixture bbox (100, 482, 114, 500) over a 7x6 grid gives
    scaleX=2, scaleY=-3 with no skew, so the anchor is hand-derivable:
    magnitudes are the pixel sizes, thetaI is acos(1) = 0, and thetaIJ is
    acos(0) = pi/2 negated by its sign test (the i-to-j separation of a
    y-down raster is -90 degrees)."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view("gt_raster", tmp_path / "gt.tif")
    sql = "SELECT RS_GeoTransform(rast) FROM gt_raster"
    anchor = {
        "magnitudeI": 2.0,
        "magnitudeJ": 3.0,
        "thetaI": 0.0,
        "thetaIJ": -math.pi / 2,
        "offsetX": 100.0,
        "offsetY": 500.0,
    }
    compare(sql, sedona, spark, expected=[(anchor,)])


def test_rs_geotransform_skewed(tmp_path):
    """A sheared transform exercises the acos sign tests in the decomposition.

    skewX=5 and skewY=3 are deliberately distinct — equal skews are the
    degenerate regime where the magnitudes coincide and thetaIJ collapses to
    +/-pi/2. The 3-4-5 / 5-12-13 pairs keep the anchor exactly representable:
    magnitudeI = sqrt(4^2 + 3^2) = 5, magnitudeJ = sqrt(12^2 + 5^2) = 13,
    thetaI = -acos(4/5) (negative because skewY > 0), and thetaIJ =
    -acos(-16/65) (products and magnitudes are exact, and the sign test
    acos(-63/65) exceeds pi/2)."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(
            "gt_skew_raster",
            tmp_path / "gt_skew.tif",
            gdal_transform=(100.0, 4.0, 5.0, 500.0, 3.0, -12.0),
        )
    sql = "SELECT RS_GeoTransform(rast) FROM gt_skew_raster"
    anchor = {
        "magnitudeI": 5.0,
        "magnitudeJ": 13.0,
        "thetaI": -math.acos(4.0 / 5.0),
        "thetaIJ": -math.acos(-16.0 / 65.0),
        "offsetX": 100.0,
        "offsetY": 500.0,
    }
    compare(sql, sedona, spark, expected=[(anchor,)])
