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
"""SedonaDB vs Sedona Spark parity for RS_MetaData.

The struct is compared field by field (`m['field']`, which both dialects
parse): the engines share all twelve field names and every value agrees
exactly on the standard grid. The whole struct is not compared — the
harness stringifies scalar columns, not structs, and the engines' field
types differ (SedonaDB int64/unsigned vs Spark int32) even though the
stringified values match. One divergence: SedonaDB's field lookup is
case-sensitive where Sedona Spark's is not.
"""

import pytest

from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark

FIELD_VALUES = {
    "upperLeftX": 100.0,
    "upperLeftY": 500.0,
    "gridWidth": 7,
    "gridHeight": 6,
    "scaleX": 2.0,
    "scaleY": -3.0,
    "skewX": 0.0,
    "skewY": 0.0,
    "srid": 0,
    "numSampleDimensions": 2,
    "tileWidth": 7,
    "tileHeight": 6,
}


def _engines(name, tmp_path):
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(name, tmp_path / f"{name}.tif")
    return sedona, spark


@pytest.mark.parametrize("field", list(FIELD_VALUES))
def test_rs_metadata_fields(field, tmp_path):
    """Every metadata field of the standard grid agrees across engines and
    with the fixture's own placement."""
    sedona, spark = _engines("md_src", tmp_path)
    sql = f"SELECT m['{field}'] FROM (SELECT RS_MetaData(rast) AS m FROM md_src) q"
    compare(sql, sedona, spark, expected=FIELD_VALUES[field])


def test_rs_metadata_unknown_field_rejected(tmp_path):
    """Both engines refuse a field name the struct does not have. Parity on
    refusal."""
    sedona, spark = _engines("md_bad_src", tmp_path)
    sql = "SELECT m['numBands'] FROM (SELECT RS_MetaData(rast) AS m FROM md_bad_src) q"
    for eng in (sedona, spark):
        with pytest.raises(Exception):
            eng.result_to_tuples(eng.execute_and_collect(sql))


@pytest.mark.xfail(
    reason="field lookup is case-sensitive in SedonaDB ('Field upperleftx not "
    "found in struct') and case-insensitive in Sedona Spark"
)
def test_rs_metadata_field_case(tmp_path):
    """A lowercased field name resolves the same way on both engines."""
    sedona, spark = _engines("md_case_src", tmp_path)
    sql = (
        "SELECT m['upperleftx'] FROM (SELECT RS_MetaData(rast) AS m FROM md_case_src) q"
    )
    compare(sql, sedona, spark)
