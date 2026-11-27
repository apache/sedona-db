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

test_that("SQL and Parquet sort geometry and geography using sd_order", {
  for (spatial_type in c("GEOMETRY", "GEOGRAPHY")) {
    query <- sprintf(
      "SELECT id, CAST(ST_GeomFromText(wkt, 4326) AS %s) AS geom
       FROM (VALUES
         (1, 'POINT (30 10)'), (2, 'POINT (-100 40)'),
         (3, NULL), (4, 'POINT (10 -40)'), (5, 'POINT (30 10)')
       ) t(id, wkt)",
      spatial_type
    )
    for (options in c("ASC NULLS LAST", "DESC NULLS FIRST")) {
      expected <- sd_sql(sprintf(
        "SELECT id FROM (%s) ORDER BY sd_order(geom) %s, id",
        query,
        options
      )) |>
        sd_collect()
      actual <- sd_sql(sprintf(
        "SELECT id FROM (%s) ORDER BY geom %s, id",
        query,
        options
      )) |>
        sd_collect()
      expect_identical(actual, expected)
    }

    path <- withr::local_tempfile(fileext = ".parquet")
    sd_write_parquet(sd_sql(query), path, sort_by = c("geom", "id"))
    expected <- sd_sql(sprintf(
      "SELECT id FROM (%s) ORDER BY sd_order(geom) NULLS LAST, id",
      query
    )) |>
      sd_collect()
    actual <- sd_read_parquet(path) |> sd_collect()
    expect_identical(actual$id, expected$id)
  }
})
