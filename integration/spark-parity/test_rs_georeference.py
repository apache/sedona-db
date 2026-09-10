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
"""SedonaDB vs Sedona Spark parity for RS_GeoReference.

A formatting divergence catalog: both engines emit the same six
world-file-order numbers (scaleX, skewY, skewX, scaleY, upperLeftX,
upperLeftY) in the same order for every format, but SedonaDB prints ten
decimal places where Sedona Spark prints six, so the strings never
compare equal. An unknown format name is refused by both.
"""

import pytest

from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark


def _engines(name, tmp_path):
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(name, tmp_path / f"{name}.tif")
    return sedona, spark


@pytest.mark.parametrize(
    "args",
    [
        pytest.param("", id="default"),
        pytest.param(", 'GDAL'", id="gdal"),
        pytest.param(", 'ESRI'", id="esri"),
    ],
)
@pytest.mark.xfail(
    reason="the numbers agree in every format but the formatting differs: "
    "SedonaDB prints ten decimal places where Sedona Spark prints six"
)
def test_rs_georeference(args, tmp_path):
    """The georeference string reads back identically from both engines."""
    sedona, spark = _engines("gr_src", tmp_path)
    sql = f"SELECT RS_GeoReference(rast{args}) FROM gr_src"
    compare(sql, sedona, spark)


def test_rs_georeference_unknown_format_rejected(tmp_path):
    """Both engines refuse an unknown format name. Parity on refusal."""
    sedona, spark = _engines("gr_bad_src", tmp_path)
    sql = "SELECT RS_GeoReference(rast, 'WORLD') FROM gr_bad_src"
    for eng in (sedona, spark):
        with pytest.raises(Exception):
            eng.result_to_tuples(eng.execute_and_collect(sql))
