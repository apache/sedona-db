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

test_that("longer expands lists lazily and in parallel", {
  df <- sd_sql("SELECT 'a' AS id, [1, 2] AS x, [10] AS y")
  result <- sd_unnest_longer(df, c(x, y))
  expect_s3_class(result, "sedonadb_dataframe")
  expect_identical(result$ctx, df$ctx)
  expect_equal(
    as.data.frame(result),
    data.frame(id = c("a", "a"), x = c(1, 2), y = c(10, NA))
  )
  expect_equal(as.data.frame(sd_unnest_longer(df, starts_with("x")))$x, c(1, 2))
})

test_that("longer drops empty and null lists but retains null elements", {
  df <- sd_sql(
    "SELECT 1 AS id, [1, NULL] AS x
    UNION ALL SELECT 2, arrow_cast([], 'List(Int64)')
    UNION ALL SELECT 3, arrow_cast(NULL, 'List(Int64)')"
  )
  expect_equal(
    as.data.frame(sd_unnest_longer(df, x)),
    data.frame(id = c(1, 1), x = c(1, NA))
  )
})

test_that("longer expands only one level and supports spatial dumps", {
  df <- sd_sql("SELECT [[1, 2], [3]] AS x")
  expect_equal(as.list(as.data.frame(sd_unnest_longer(df, x))$x), list(c(1, 2), 3))
  df <- sd_sql("SELECT ST_Dump(ST_GeomFromText('MULTIPOINT(0 0, 1 1)')) AS parts")
  result <- df |> sd_unnest_longer(parts) |> sd_unnest_wider(parts)
  expect_named(as.data.frame(result), c("path", "geom"))
  expect_equal(nrow(as.data.frame(result)), 2L)
})

test_that("wider expands structs in place and preserves nested fields", {
  df <- sd_sql("SELECT 1 AS id, named_struct('x', 2, 'y', [3, 4]) AS s, 5 AS z")
  result <- sd_unnest_wider(df, s)
  expect_identical(result$ctx, df$ctx)
  expect_named(as.data.frame(result), c("id", "x", "y", "z"))
  expect_equal(as.list(as.data.frame(result)$y), list(c(3, 4)))
  expect_named(
    as.data.frame(sd_unnest_wider(df, s, names_sep = "_")),
    c("id", "s_x", "s_y", "z")
  )
  df <- sd_sql("SELECT named_struct('b', 1) AS a, named_struct('c', 2) AS b")
  expect_equal(as.data.frame(sd_unnest_wider(df, everything())), data.frame(b = 1, c = 2))
})

test_that("unnesting validates selections, types, grouping, and names", {
  df <- sd_sql("SELECT 1 AS x, [2] AS y, named_struct('x', 3) AS s")
  expect_error(sd_unnest_longer(df, x), "must be a list")
  expect_error(sd_unnest_wider(df, y), "must be a struct")
  expect_error(sd_unnest_longer(df, missing_column))
  expect_error(sd_unnest_longer(df, any_of("missing_column")), "at least one")
  expect_error(sd_unnest_longer(df, c(new = y)), "rename")
  expect_error(sd_unnest_longer(sd_group_by(df, x), y), "grouped")
  expect_error(sd_unnest_wider(sd_group_by(df, x), s), "grouped")
  expect_error(sd_unnest_wider(df, s), "unique")
  expect_error(sd_unnest_wider(df, s, names_sep = NA_character_), "names_sep")
  expect_named(as.data.frame(sd_unnest_wider(df, s, names_sep = "_")), c("x", "y", "s_x"))
})

test_that("non-syntactic column and field names work", {
  df <- sd_sql('SELECT [1, 2] AS "a.b", named_struct(\'x y\', 3) AS "s t"')
  expect_equal(as.data.frame(sd_unnest_longer(df, `a.b`))[[1]], c(1, 2))
  expect_named(as.data.frame(sd_unnest_wider(df, `s t`)), c("a.b", "x y"))
})

test_that("wider preserves rows with null structs", {
  df <- sd_sql(
    "SELECT 1 AS id, named_struct('x', 2) AS s
    UNION ALL SELECT 2, NULL"
  )
  expect_equal(
    as.data.frame(sd_arrange(sd_unnest_wider(df, s), id)),
    data.frame(id = c(1, 2), x = c(2, NA))
  )
})

test_that("longer accepts data frames and fixed-size lists", {
  df <- as.data.frame(sd_sql("SELECT 1 AS id, [1, 2] AS x"))
  expect_equal(
    as.data.frame(sd_unnest_longer(df, x)),
    data.frame(id = c(1, 1), x = c(1, 2))
  )
  df <- sd_sql("SELECT arrow_cast([1, 2], 'FixedSizeList(2, Int64)') AS x")
  expect_equal(as.data.frame(sd_unnest_longer(df, x)), data.frame(x = c(1, 2)))
})
