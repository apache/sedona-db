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
"""RS_ReplaceBandNoDataValue moves a band's nodata value and rewrites the pixels
that held the old one, so the same pixels read as nodata before and after.

Pixel reads go through RS_SummaryStats's valid-pixel count rather than a grid
coordinate, so the checks don't depend on how RS_Value's grid form counts.
"""

import pytest

from sedonadb.testing import SedonaDB

# RS_Example's band 1 (UInt8) has a nodata value of 127, held by one of its
# 2048 pixels, so 2047 pixels are valid.
BASE = "RS_Example()"


def test_replace_keeps_the_nodata_pixels():
    # Replacing keeps the one nodata pixel as nodata (2047 valid); merely
    # setting the nodata value turns it into ordinary data (2048 valid).
    SedonaDB().assert_query_result(
        f"""SELECT
            RS_SummaryStats(RS_ReplaceBandNoDataValue({BASE}, 1, 200), 'count', 1),
            RS_SummaryStats(RS_SetBandNoDataValue({BASE}, 1, 200), 'count', 1),
            RS_BandNoDataValue(RS_ReplaceBandNoDataValue({BASE}, 1, 200), 1)""",
        [("2047", "2048", "200")],
    )


def test_replace_needs_an_existing_nodata():
    cleared = "RS_SetBandNoDataValue(RS_Example(), 1, CAST(NULL AS DOUBLE))"
    with pytest.raises(Exception, match="no nodata value to replace"):
        SedonaDB().assert_query_result(
            f"SELECT RS_ReplaceBandNoDataValue({cleared}, 1, 200)", None
        )


def test_replace_rejects_a_nodata_the_band_cannot_hold():
    # Band 1 of RS_Example is UInt8.
    with pytest.raises(Exception, match="RS_ReplaceBandNoDataValue"):
        SedonaDB().assert_query_result(
            f"SELECT RS_ReplaceBandNoDataValue({BASE}, 1, 256)", None
        )


def test_replace_with_null_gives_null():
    SedonaDB().assert_query_result(
        f"SELECT RS_ReplaceBandNoDataValue({BASE}, 1, CAST(NULL AS DOUBLE)) IS NULL",
        True,
    )


def test_replace_output_feeds_other_raster_functions(con):
    """The output is already loaded, so the planner must not wrap it in another
    RS_EnsureLoaded when it feeds a pixel-reading function or a second
    RS_ReplaceBandNoDataValue. Reading from a table keeps the raster out of
    constant folding."""
    rasters = con.sql(f"SELECT {BASE} AS r").to_arrow_table()
    con.create_data_frame(rasters).to_view("replace_nodata_nested", overwrite=True)
    sql = """
        SELECT
          RS_BandNoDataValue(RS_ReplaceBandNoDataValue(
              RS_ReplaceBandNoDataValue(r, 1, 200), 1, 100), 1),
          RS_SummaryStats(RS_ReplaceBandNoDataValue(r, 1, 200), 'count', 1)
        FROM replace_nodata_nested
    """
    table = con.sql(sql).to_arrow_table()
    assert [c.to_pylist() for c in table.columns] == [[100.0], [2047.0]]
