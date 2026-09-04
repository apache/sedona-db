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
"""SedonaDB vs Sedona Spark parity for RS_Intersects.

Raster-geometry predicates: the roi travels as `ST_GeomFromWKT` (the
suite's established input spelling) and the boolean result compares raw.
The geometries are placed against the standard grid (x in [100, 114],
y in [482, 500]).
"""

from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark

INSIDE = "POLYGON((105 490, 108 490, 108 493, 105 493, 105 490))"
OVERLAPPING = "POLYGON((90 490, 105 490, 105 493, 90 493, 90 490))"
DISJOINT = "POLYGON((300 300, 310 300, 310 310, 300 310, 300 300))"
COVERING = "POLYGON((90 470, 120 470, 120 510, 90 510, 90 470))"


def test_rs_intersects(tmp_path):
    """True for contained and overlapping geometries, false for disjoint."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view("ix_src", tmp_path / "ix_src.tif")
    for wkt, expected in ((INSIDE, True), (OVERLAPPING, True), (DISJOINT, False)):
        sql = f"SELECT RS_Intersects(rast, ST_GeomFromWKT('{wkt}')) FROM ix_src"
        compare(sql, sedona, spark, expected=expected)
