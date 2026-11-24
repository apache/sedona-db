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

//! Exercise tile streaming together with Sedona's raster loading rules.
use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use datafusion::catalog::MemTable;
use datafusion::physical_plan::{collect, displayable};
use sedona::context::SedonaContext;
use sedona_schema::{datatypes::SedonaType, raster::BandDataType};
use sedona_testing::raster_spec::RasterSpec;
use std::sync::Arc;

#[tokio::test]
async fn tile_streaming_composes_with_materialization_and_multiple_partitions() {
    let ctx = SedonaContext::new_local_interactive().await.unwrap();
    ctx.sql("SET sedona.raster.max_batch_bytes = 12")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let raster = RasterSpec::d2(5, 3)
        .crs(None)
        .band(BandDataType::UInt8)
        .build();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        SedonaType::Raster.to_storage_field("rast", false).unwrap(),
    ]));
    let partitions = (0..2)
        .map(|id| {
            vec![RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int64Array::from(vec![id])),
                    Arc::new(raster.clone()),
                ],
            )
            .unwrap()]
        })
        .collect();
    ctx.ctx
        .register_table(
            "r",
            Arc::new(MemTable::try_new(schema, partitions).unwrap()),
        )
        .unwrap();

    let sql = "SELECT id, t.x AS x, t.y AS y, RS_Width(t.tile) AS width
        FROM (SELECT id, UNNEST(RS_Tile(rast, 2, 2)) AS t FROM r)
        ORDER BY id, y, x";
    let plan = ctx
        .sql(sql)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let text = displayable(plan.as_ref()).indent(true).to_string();
    assert!(text.contains("RasterTileExec"), "{text}");
    assert!(text.contains("RasterBatchSplitExec"), "{text}");
    let batches = collect(plan, ctx.ctx.task_ctx()).await.unwrap();
    datafusion::assert_batches_eq!(
        [
            "+----+---+---+-------+",
            "| id | x | y | width |",
            "+----+---+---+-------+",
            "| 0  | 0 | 0 | 2     |",
            "| 0  | 1 | 0 | 2     |",
            "| 0  | 2 | 0 | 1     |",
            "| 0  | 0 | 1 | 2     |",
            "| 0  | 1 | 1 | 2     |",
            "| 0  | 2 | 1 | 1     |",
            "| 1  | 0 | 0 | 2     |",
            "| 1  | 1 | 0 | 2     |",
            "| 1  | 2 | 0 | 1     |",
            "| 1  | 0 | 1 | 2     |",
            "| 1  | 1 | 1 | 2     |",
            "| 1  | 2 | 1 | 1     |",
            "+----+---+---+-------+",
        ],
        &batches
    );

    let df = ctx
        .sql("SELECT UNNEST(RS_Tile(rast, 1, 1)) AS t FROM r LIMIT 1")
        .await
        .unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    assert!(displayable(plan.as_ref())
        .indent(true)
        .to_string()
        .contains("RasterTileExec"));
    let batches = collect(plan, ctx.ctx.task_ctx()).await.unwrap();
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
}
