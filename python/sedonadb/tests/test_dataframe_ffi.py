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

import pandas as pd
import pytest

import sedonadb
from sedonadb.testing import skip_if_not_exists


# Test cases: (producer_sql, consumer_sql)
FFI_TEST_CASES = [
    # Simple select all
    pytest.param(
        "SELECT * FROM water_point",
        "SELECT * FROM df_producer",
        id="select_all",
    ),
    # Projection - select specific columns
    pytest.param(
        'SELECT "OBJECTID", "FEAT_CODE" FROM water_point',
        'SELECT * FROM df_producer ORDER BY "OBJECTID" LIMIT 10',
        id="projection",
    ),
    # Filter on producer side
    pytest.param(
        'SELECT "OBJECTID", "FEAT_CODE" FROM water_point WHERE "OBJECTID" > 100',
        'SELECT * FROM df_producer ORDER BY "OBJECTID" LIMIT 5',
        id="filter_producer",
    ),
    # Filter on consumer side
    pytest.param(
        'SELECT "OBJECTID", "FEAT_CODE" FROM water_point',
        'SELECT * FROM df_producer WHERE "OBJECTID" < 50 ORDER BY "OBJECTID"',
        id="filter_consumer",
    ),
    # Sort on producer side (descending)
    pytest.param(
        'SELECT "OBJECTID", "FEAT_CODE" FROM water_point ORDER BY "OBJECTID" DESC LIMIT 20',
        "SELECT * FROM df_producer",
        id="sort_producer_desc",
    ),
    # Sort on consumer side
    pytest.param(
        'SELECT "OBJECTID", "FEAT_CODE" FROM water_point',
        'SELECT * FROM df_producer ORDER BY "FEAT_CODE", "OBJECTID" DESC LIMIT 10',
        id="sort_consumer",
    ),
    # Limit on producer
    pytest.param(
        "SELECT * FROM water_point LIMIT 5",
        "SELECT * FROM df_producer",
        id="limit_producer",
    ),
    # Aggregate on consumer
    pytest.param(
        'SELECT "OBJECTID", "FEAT_CODE" FROM water_point',
        'SELECT "FEAT_CODE", COUNT(*) as cnt FROM df_producer GROUP BY "FEAT_CODE" ORDER BY cnt DESC LIMIT 5',
        id="aggregate_consumer",
    ),
    # Aggregate on producer side
    pytest.param(
        'SELECT "FEAT_CODE", COUNT(*) as cnt, MIN("OBJECTID") as min_id, MAX("OBJECTID") as max_id FROM water_point GROUP BY "FEAT_CODE"',
        'SELECT * FROM df_producer ORDER BY cnt DESC, "FEAT_CODE" LIMIT 10',
        id="aggregate_producer",
    ),
    # Spatial distance join on producer side (with pre-filtered data to keep it fast)
    pytest.param(
        """
        WITH subset AS (SELECT * FROM water_point WHERE "OBJECTID" < 1000)
        SELECT a."OBJECTID" as id_a, b."OBJECTID" as id_b, ST_Distance(a.geometry, b.geometry) as dist
        FROM subset a
        JOIN subset b ON ST_DWithin(a.geometry, b.geometry, 100)
        ORDER BY a."OBJECTID", b."OBJECTID"
        """,
        "SELECT * FROM df_producer ORDER BY id_a, id_b LIMIT 10",
        id="spatial_join_producer",
    ),
    # Multiple operations chained
    pytest.param(
        'SELECT "OBJECTID", "FEAT_CODE" FROM water_point WHERE "OBJECTID" BETWEEN 10 AND 100 ORDER BY "OBJECTID"',
        'SELECT "FEAT_CODE", COUNT(*) as cnt FROM df_producer GROUP BY "FEAT_CODE" HAVING COUNT(*) > 1 ORDER BY cnt DESC',
        id="filter_then_aggregate",
    ),
]


@pytest.mark.parametrize("producer_sql,consumer_sql", FFI_TEST_CASES)
def test_ffi_roundtrip(geoarrow_data, producer_sql, consumer_sql):
    # Use a real file with a reasonable number of rows so that parallelism
    # and multiple batches are invoked
    path = geoarrow_data / "ns-water" / "files" / "ns-water_water-point_geo.parquet"
    skip_if_not_exists(path)

    sd_producer = sedonadb.connect()
    sd_consumer = sedonadb.connect()

    sd_producer.read_parquet(path).to_view("water_point")
    df_producer = sd_producer.sql(producer_sql)

    # Run the query on the producer (no FFI)
    df_producer.to_view("df_producer")
    result_no_ffi = sd_producer.sql(consumer_sql).to_pandas()

    # Run the query on the consumer (separated via FFI)
    sd_consumer.create_data_frame(df_producer).to_view("df_producer")
    result_over_ffi = sd_consumer.sql(consumer_sql).to_pandas()

    pd.testing.assert_frame_equal(result_no_ffi, result_over_ffi)
