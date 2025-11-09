// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
use datafusion::arrow::util::pretty::pretty_format_batches;
use sedona::context::SedonaContext;
use sedona_testing::data::sedona_testing_dir;

#[tokio::test]
async fn geo_parquet_metrics() {
    // Setup and register test table
    // -----------------------------
    let ctx = SedonaContext::new_local_interactive()
        .await
        .expect("interactive context should initialize");

    let geo_parquet_path = format!(
        "{}/data/parquet/geoparquet-1.1.0.parquet",
        sedona_testing_dir().expect("sedona-testing directory should resolve")
    );
    let create_table_sql =
        format!("CREATE EXTERNAL TABLE test STORED AS PARQUET LOCATION '{geo_parquet_path}'");

    ctx.sql(&create_table_sql)
        .await
        .expect("create table should succeed")
        .collect()
        .await
        .expect("collecting create table result should succeed");

    // Test 1: query with spatial predicate that pruned the entire file
    // ----------------------------------------------------------------
    let prune_query = r#"
        EXPLAIN ANALYZE
        SELECT *
        FROM test
        WHERE ST_Intersects(
            geometry,
            ST_SetSRID(
                ST_GeomFromText('POLYGON((-10 84, -10 88, 10 88, 10 84, -10 84))'),
                4326
            )
        )
    "#;

    let prune_plan = run_and_format(&ctx, prune_query).await;
    assert!(prune_plan.contains("files_ranges_spatial_pruned=1"));
    assert!(prune_plan.contains("files_ranges_spatial_matched=0"));
    assert!(prune_plan.contains("row_groups_spatial_pruned=0"));
    assert!(prune_plan.contains("row_groups_spatial_matched=0"));

    // Test 2: query with spatial filter that can't skip any file or row group
    // -----------------------------------------------------------------------
    let match_query = r#"
        EXPLAIN ANALYZE
        SELECT *
        FROM test
        WHERE ST_Intersects(
            geometry,
            ST_SetSRID(
                ST_GeomFromText(
                    'POLYGON((-180 -18.28799, -180 83.23324, 180 83.23324, 180 -18.28799, -180 -18.28799))'
                ),
                4326
            )
        )
    "#;

    let match_plan = run_and_format(&ctx, match_query).await;
    assert!(match_plan.contains("files_ranges_spatial_pruned=0"));
    assert!(match_plan.contains("files_ranges_spatial_matched=1"));
    assert!(match_plan.contains("row_groups_spatial_pruned=0"));
    assert!(match_plan.contains("row_groups_spatial_matched=1"));
}

async fn run_and_format(ctx: &SedonaContext, sql: &str) -> String {
    let df = ctx
        .sql(sql.trim())
        .await
        .expect("explain analyze query should succeed");
    let batches = df
        .collect()
        .await
        .expect("collecting explain analyze result should succeed");
    format!(
        "{}",
        pretty_format_batches(&batches).expect("formatting plan should succeed")
    )
}
