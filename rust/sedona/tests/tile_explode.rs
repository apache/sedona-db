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

//! End-to-end tests for the `RS_TileExplode` streaming generator.
//!
//! `SELECT RS_TileExplode(rast, w, h) FROM t` must emit one row per tile with
//! **top-level** `(x, y, tile)` columns (plus any pass-through sibling columns),
//! which is produced by the tile-explode analyzer rule lifting the marker call
//! into a `TileExplodePlanNode` planned as a streaming `TileExplodeExec`. The
//! tiles are compared against `RS_Tile` — the two functions share one tiling core
//! and must not diverge.

use std::sync::Arc;

use arrow_array::{ArrayRef, Int32Array, RecordBatch, StructArray};
use arrow_schema::{DataType, Field, Schema};
use datafusion::arrow::compute::concat_batches;
use datafusion::catalog::MemTable;
use datafusion::common::cast::{as_int32_array, as_struct_array};
use datafusion::common::ScalarValue;
use datafusion::logical_expr::{ColumnarValue, ScalarUDF};
use sedona::context::SedonaContext;
use sedona_raster::array::RasterStructArray;
use sedona_schema::datatypes::{SedonaType, RASTER};
use sedona_testing::raster_spec::{raster_array, RasterSpec};
use sedona_testing::rasters::assert_raster_arrays_equal;
use sedona_testing::testers::ScalarUdfTester;

/// A 5x3 EPSG-less raster, origin (0, 3), north-up 1x1 pixels, one UInt8 band
/// with values 1..=15 (row-major). Its odd extent makes the last column and last
/// row partial, so a 2x2 tiling exercises edge tiles. Mirrors the tiling-core and
/// exec fixtures.
fn source_5x3() -> RasterSpec {
    RasterSpec::d2(5, 3)
        .crs(None)
        .transform([0.0, 1.0, 0.0, 3.0, 0.0, -1.0])
        .band_values(&[1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
}

/// Register a table `t(id INT, rast RASTER)` holding the given rasters, one per
/// row, with `id` running 0, 1, ... .
fn register_raster_table(ctx: &SedonaContext, rasters: Vec<Option<RasterSpec>>) {
    let ids: Vec<i32> = (0..rasters.len() as i32).collect();
    let id_column = Int32Array::from(ids);
    let raster_column: ArrayRef = Arc::new(raster_array(rasters));

    let raster_field = RASTER.to_storage_field("rast", true).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        raster_field,
    ]));
    let batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(id_column), raster_column]).unwrap();
    let table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
    ctx.ctx.register_table("t", Arc::new(table)).unwrap();
}

/// The error string from planning-and-running `sql`. The tile-explode analyzer
/// rule runs during optimization (triggered by execution), not when the
/// unoptimized `DataFrame` is built, so a placement error surfaces at `collect`
/// time rather than at `sql` time.
async fn tile_explode_error(ctx: &SedonaContext, sql: &str) -> String {
    match ctx.sql(sql).await {
        Err(e) => e.to_string(),
        Ok(df) => df
            .collect()
            .await
            .expect_err("expected a plan-time error")
            .to_string(),
    }
}

/// Collect a query's result batches.
///
/// The tile-explode lift is a schema-changing analyzer rule, so it runs during
/// execution, not when the (unoptimized) `DataFrame` is built — the executed
/// batch schema, not `df.schema()`, is authoritative for the output columns.
async fn collect_rows(ctx: &SedonaContext, sql: &str) -> Vec<RecordBatch> {
    ctx.sql(sql).await.unwrap().collect().await.unwrap()
}

/// Run `sql` and concatenate its (non-empty) result into one batch, using the
/// executed schema.
async fn run_to_batch(ctx: &SedonaContext, sql: &str) -> RecordBatch {
    let batches = collect_rows(ctx, sql).await;
    let schema = batches.first().expect("query produced no batches").schema();
    concat_batches(&schema, &batches).unwrap()
}

/// The `(x, y)` grid positions of a tile-explode result, read from the named
/// columns.
fn grid_positions(batch: &RecordBatch) -> (Vec<i32>, Vec<i32>) {
    let x = as_int32_array(column(batch, "x")).unwrap();
    let y = as_int32_array(column(batch, "y")).unwrap();
    (
        (0..x.len()).map(|i| x.value(i)).collect(),
        (0..y.len()).map(|i| y.value(i)).collect(),
    )
}

fn column<'a>(batch: &'a RecordBatch, name: &str) -> &'a ArrayRef {
    let index = batch.schema().index_of(name).unwrap();
    batch.column(index)
}

/// The `(x, y)` positions and tile rasters `RS_Tile(source_5x3, 2, 2)` produces,
/// extracted from its `List<Struct<x, y, tile>>` scalar result. The reference
/// `RS_TileExplode` output must match these row for row.
fn rs_tile_reference() -> (Vec<i32>, Vec<i32>, StructArray) {
    let udf: ScalarUDF = sedona_raster_gdal::rs_tile_udf().into();
    let tester = ScalarUdfTester::new(
        udf,
        vec![
            RASTER,
            SedonaType::Arrow(DataType::Int32),
            SedonaType::Arrow(DataType::Int32),
        ],
    );
    let result = tester
        .invoke(vec![
            ColumnarValue::Scalar(source_5x3().scalar()),
            ColumnarValue::Scalar(ScalarValue::Int32(Some(2))),
            ColumnarValue::Scalar(ScalarValue::Int32(Some(2))),
        ])
        .unwrap();
    let ColumnarValue::Scalar(ScalarValue::List(list)) = result else {
        panic!("expected a scalar List result, got {result:?}");
    };
    let element = as_struct_array(list.values()).unwrap();
    let x = as_int32_array(element.column(0)).unwrap();
    let y = as_int32_array(element.column(1)).unwrap();
    let tiles = as_struct_array(element.column(2)).unwrap().clone();
    (
        (0..x.len()).map(|i| x.value(i)).collect(),
        (0..y.len()).map(|i| y.value(i)).collect(),
        tiles,
    )
}

#[tokio::test]
async fn tile_explode_lifts_to_top_level_columns() {
    let ctx = SedonaContext::new_local_interactive().await.unwrap();
    register_raster_table(&ctx, vec![Some(source_5x3())]);

    let batch = run_to_batch(&ctx, "SELECT RS_TileExplode(rast, 2, 2) FROM t").await;

    // The single explode column is lifted to top-level (x, y, tile); no raster or
    // id column leaks through.
    assert_eq!(
        batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect::<Vec<_>>(),
        vec!["x".to_string(), "y".to_string(), "tile".to_string()]
    );

    // Six tiles (5x3 tiled 2x2), row-major, matching RS_Tile exactly.
    let (subject_x, subject_y) = grid_positions(&batch);
    let (reference_x, reference_y, reference_tiles) = rs_tile_reference();
    assert_eq!(subject_x.len(), 6, "5x3 tiled 2x2 yields 6 tiles");
    assert_eq!(
        subject_x, reference_x,
        "x grid positions differ from RS_Tile"
    );
    assert_eq!(
        subject_y, reference_y,
        "y grid positions differ from RS_Tile"
    );

    let subject_tiles = as_struct_array(column(&batch, "tile")).unwrap();
    assert_raster_arrays_equal(
        &RasterStructArray::try_new(subject_tiles).unwrap(),
        &RasterStructArray::try_new(&reference_tiles).unwrap(),
    );
}

#[tokio::test]
async fn tile_explode_replicates_sibling_columns() {
    let ctx = SedonaContext::new_local_interactive().await.unwrap();
    register_raster_table(&ctx, vec![Some(source_5x3())]);

    let batch = run_to_batch(&ctx, "SELECT id, RS_TileExplode(rast, 2, 2) FROM t").await;

    // The sibling `id` is carried through and appears before the appended
    // (x, y, tile); the raster argument does not leak into the output.
    assert_eq!(
        batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect::<Vec<_>>(),
        vec![
            "id".to_string(),
            "x".to_string(),
            "y".to_string(),
            "tile".to_string()
        ]
    );

    // The single input row (id = 0) is replicated across all six tile rows.
    let ids = as_int32_array(column(&batch, "id")).unwrap();
    assert_eq!(ids.len(), 6);
    assert!(
        (0..ids.len()).all(|i| ids.value(i) == 0),
        "id should be replicated across every tile row"
    );
}

#[tokio::test]
async fn tile_explode_null_raster_yields_zero_rows() {
    let ctx = SedonaContext::new_local_interactive().await.unwrap();
    register_raster_table(&ctx, vec![None]);

    let batches = collect_rows(&ctx, "SELECT id, RS_TileExplode(rast, 2, 2) FROM t").await;
    assert_eq!(
        batches.iter().map(|b| b.num_rows()).sum::<usize>(),
        0,
        "a null raster row contributes no tiles"
    );
}

#[tokio::test]
async fn tile_explode_in_where_is_rejected() {
    let ctx = SedonaContext::new_local_interactive().await.unwrap();
    register_raster_table(&ctx, vec![Some(source_5x3())]);

    // A generator in a WHERE predicate is an illegal placement: the marker is not
    // in a liftable top-level projection column, so it survives to a plan-time
    // error rather than a kernel panic.
    let err = tile_explode_error(
        &ctx,
        "SELECT id FROM t WHERE RS_TileExplode(rast, 2, 2) IS NOT NULL",
    )
    .await;
    assert!(err.contains("RS_TileExplode"), "unexpected error: {err}");
}

#[tokio::test]
async fn tile_explode_nested_in_expression_is_rejected() {
    let ctx = SedonaContext::new_local_interactive().await.unwrap();
    register_raster_table(&ctx, vec![Some(source_5x3())]);

    // Nesting the generator inside another expression is illegal: here the marker
    // is an argument of an `IS NOT NULL`, not a bare top-level projection column.
    let err =
        tile_explode_error(&ctx, "SELECT RS_TileExplode(rast, 2, 2) IS NOT NULL FROM t").await;
    assert!(err.contains("RS_TileExplode"), "unexpected error: {err}");
}
