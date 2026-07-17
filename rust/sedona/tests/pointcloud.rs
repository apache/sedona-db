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
#![cfg(feature = "pointcloud")]

use std::path::PathBuf;

use sedona::context::SedonaContext;

fn pointcloud_data_path(file_name: &str) -> String {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("../sedona-pointcloud/tests/data");
    path.push(file_name);
    path.display().to_string()
}

#[tokio::test]
async fn las_statistics_pruning() {
    // File with two clusters, one at 0.5 one at 1.0.
    let path = pointcloud_data_path("large.las");

    let ctx = SedonaContext::new_local_interactive().await.unwrap();

    // Ensure no faulty chunk pruning.
    ctx.sql("SET las.geometry_encoding = 'plain'")
        .await
        .unwrap();
    ctx.sql("SET las.collect_statistics = 'true'")
        .await
        .unwrap();

    let count = ctx
        .sql(&format!("SELECT * FROM \"{path}\" WHERE x < 0.7"))
        .await
        .unwrap()
        .count()
        .await
        .unwrap();
    assert_eq!(count, 50000);

    let count = ctx
        .sql(&format!("SELECT * FROM \"{path}\" WHERE y < 0.7"))
        .await
        .unwrap()
        .count()
        .await
        .unwrap();
    assert_eq!(count, 50000);

    ctx.sql("SET las.geometry_encoding = 'wkb'").await.unwrap();
    let count = ctx
        .sql(&format!("SELECT * FROM \"{path}\" WHERE ST_Intersects(geometry, ST_GeomFromText('POLYGON ((0 0, 0.7 0, 0.7 0.7, 0 0.7, 0 0))'))"))
        .await
        .unwrap()
        .count()
        .await
        .unwrap();
    assert_eq!(count, 50000);
}

#[tokio::test]
async fn laz_statistics_pruning() {
    // File with two clusters, one at 0.5 one at 1.0.
    let path = pointcloud_data_path("large.laz");

    let ctx = SedonaContext::new_local_interactive().await.unwrap();

    // Ensure no faulty chunk pruning.
    ctx.sql("SET las.geometry_encoding = 'plain'")
        .await
        .unwrap();
    ctx.sql("SET las.collect_statistics = 'true'")
        .await
        .unwrap();

    let count = ctx
        .sql(&format!("SELECT * FROM \"{path}\" WHERE x < 0.7"))
        .await
        .unwrap()
        .count()
        .await
        .unwrap();
    assert_eq!(count, 50000);

    let count = ctx
        .sql(&format!("SELECT * FROM \"{path}\" WHERE y < 0.7"))
        .await
        .unwrap()
        .count()
        .await
        .unwrap();
    assert_eq!(count, 50000);

    ctx.sql("SET las.geometry_encoding = 'wkb'").await.unwrap();
    let count = ctx
        .sql(&format!("SELECT * FROM \"{path}\" WHERE ST_Intersects(geometry, ST_GeomFromText('POLYGON ((0 0, 0.7 0, 0.7 0.7, 0 0.7, 0 0))'))"))
        .await
        .unwrap()
        .count()
        .await
        .unwrap();
    assert_eq!(count, 50000);
}

#[tokio::test]
async fn round_robin_partitioning() {
    use datafusion_common::arrow::compute::{concat_batches, sort_to_indices, take_record_batch};

    fn concat_and_sort(
        batches: &[arrow_array::RecordBatch],
        sort_col: usize,
    ) -> arrow_array::RecordBatch {
        assert!(
            !batches.is_empty(),
            "expected at least one RecordBatch, but the query returned none"
        );
        let schema = batches[0].schema();
        let combined = concat_batches(&schema, batches).unwrap();
        let indices = sort_to_indices(combined.column(sort_col), None, None).unwrap();
        take_record_batch(&combined, &indices).unwrap()
    }

    // File with two clusters, one at 0.5 one at 1.0.
    let path = pointcloud_data_path("large.laz");

    let ctx = SedonaContext::new_local_interactive().await.unwrap();

    let result1 = ctx
        .sql(&format!("SELECT * FROM \"{path}\""))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    ctx.sql("SET las.round_robin_partitioning = 'true'")
        .await
        .unwrap();
    let result2 = ctx
        .sql(&format!("SELECT * FROM \"{path}\""))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    // Compare content independent of batch boundaries and partition ordering.
    // Sort by the geometry column (index 0) to get a deterministic row order.
    // Single-column sort suffices because all rows within each cluster in the
    // test data are identical (see generate.py), so ties are indistinguishable.
    let batch1 = concat_and_sort(&result1, 0);
    let batch2 = concat_and_sort(&result2, 0);
    assert_eq!(batch1, batch2);
}
