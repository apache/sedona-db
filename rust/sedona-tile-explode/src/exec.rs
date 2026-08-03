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

//! [`TileExplodeExec`] — a streaming physical operator that emits one output row
//! per tile of each input raster.
//!
//! It cuts each raster of its single input into a grid of tiles using the shared
//! tiling core (identical tiles, in the same row-major order, as `RS_Tile`) and
//! appends the tile grid position `(x, y)` and the tile raster to the input
//! row's other columns, replicating those sibling columns across every tile row.
//! The child is consumed lazily, one input batch at a time, and output is emitted
//! a batch at a time — a single raster's tiles never all materialize at once.

use std::any::Any;
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::builder::Int32Builder;
use arrow_array::{ArrayRef, RecordBatch, UInt32Array};
use arrow_schema::SchemaRef;
use datafusion_common::cast::as_struct_array;
use datafusion_common::{exec_datafusion_err, plan_err, Result};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_expr::EquivalenceProperties;
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
    RecordBatchStream,
};
use futures::{Stream, StreamExt};

use sedona_query_planner::tile_explode_node::tile_explode_appended_fields;
use sedona_raster::array::RasterStructArray;
use sedona_raster::builder::RasterBuilder;
use sedona_raster::traits::RasterRef;
use sedona_raster_gdal::tiling::{
    append_tile, resolve_band_indices, tile_grid_dims, TileParams, TileWindow,
};

/// Resolved, plan-time-constant tile parameters for a [`TileExplodeExec`].
///
/// In the full pipeline the tile-explode extension planner resolves the
/// `RS_TileExplode` argument expressions into these constants; a caller may also
/// build them directly.
#[derive(Debug, Clone)]
pub struct TileExplodeExecArgs {
    /// Index of the raster column in the input schema.
    pub raster_column: usize,
    /// Tile width in pixels (columns).
    pub tile_width: i64,
    /// Tile height in pixels (rows).
    pub tile_height: i64,
    /// Pad the trailing partial edge tiles to the full tile size with a nodata
    /// fill.
    pub pad_with_nodata: bool,
    /// Explicit nodata fill value for padded pixels. Only meaningful with
    /// `pad_with_nodata = true`.
    pub nodata: Option<f64>,
    /// 1-based band indices to keep in each tile, in order. `None` keeps every
    /// band.
    pub bands: Option<Vec<i64>>,
}

/// Streaming operator that explodes each input raster into one output row per
/// tile, appending `(x, y, tile)` to the replicated input columns.
#[derive(Debug)]
pub struct TileExplodeExec {
    input: Arc<dyn ExecutionPlan>,
    args: TileExplodeExecArgs,
    output_schema: SchemaRef,
    properties: PlanProperties,
}

impl TileExplodeExec {
    /// Build a [`TileExplodeExec`] over `input` with resolved tile parameters.
    ///
    /// The output schema is the input schema followed by the appended
    /// `(x: Int32, y: Int32, tile: raster)` columns.
    pub fn try_new(input: Arc<dyn ExecutionPlan>, args: TileExplodeExecArgs) -> Result<Self> {
        let input_schema = input.schema();
        if args.raster_column >= input_schema.fields().len() {
            return plan_err!(
                "TileExplodeExec: raster column index {} is out of range for a {}-column input",
                args.raster_column,
                input_schema.fields().len()
            );
        }
        // noDataVal is a padding-only knob (mirrors RS_Tile's up-front check),
        // rejected once at plan time rather than per row.
        if args.nodata.is_some() && !args.pad_with_nodata {
            return plan_err!(
                "TileExplodeExec: nodata is only meaningful with pad_with_nodata = true"
            );
        }

        let mut fields: Vec<_> = input_schema.fields().iter().cloned().collect();
        for field in tile_explode_appended_fields()? {
            fields.push(Arc::new(field));
        }
        let output_schema: SchemaRef = Arc::new(arrow_schema::Schema::new_with_metadata(
            fields,
            input_schema.metadata().clone(),
        ));

        // Row-multiplying but order-preserving and never blocking: emit
        // incrementally, bounded by the (bounded) child.
        let properties = PlanProperties::new(
            EquivalenceProperties::new(output_schema.clone()),
            input.output_partitioning().clone(),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );

        Ok(Self {
            input,
            args,
            output_schema,
            properties,
        })
    }

    /// The resolved tile parameters.
    pub fn args(&self) -> &TileExplodeExecArgs {
        &self.args
    }
}

impl DisplayAs for TileExplodeExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => write!(
                f,
                "TileExplodeExec: tile={}x{}, pad_with_nodata={}",
                self.args.tile_width, self.args.tile_height, self.args.pad_with_nodata
            ),
            DisplayFormatType::TreeRender => {
                write!(f, "tile={}x{}", self.args.tile_width, self.args.tile_height)
            }
        }
    }
}

impl ExecutionPlan for TileExplodeExec {
    fn name(&self) -> &str {
        "TileExplodeExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return plan_err!(
                "TileExplodeExec expects exactly 1 child, got {}",
                children.len()
            );
        }
        Ok(Arc::new(Self::try_new(
            children.remove(0),
            self.args.clone(),
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let batch_size = context.session_config().batch_size().max(1);
        let input = self.input.execute(partition, context)?;
        let stream = TileExplodeStream {
            input,
            args: self.args.clone(),
            output_schema: self.output_schema.clone(),
            batch_size,
            current: None,
        };
        Ok(Box::pin(stream))
    }
}

/// The tiling progress for the raster on one input row.
struct RasterCursor {
    num_tile_x: usize,
    total_tiles: usize,
    /// Next tile to emit as a linear, row-major index into the grid.
    next_tile: usize,
    /// 1-based band indices resolved for this raster (resolved once per raster).
    band_indices: Vec<usize>,
    width: usize,
    height: usize,
    tile_w: usize,
    tile_h: usize,
    pad: bool,
}

/// The input batch currently being exploded, plus the cursor into the current
/// raster row.
struct CurrentInput {
    batch: RecordBatch,
    num_rows: usize,
    /// Index of the input row whose raster is being tiled.
    row_idx: usize,
    /// Tiling progress for `row_idx`, or `None` when a new raster row must be
    /// resolved.
    cursor: Option<RasterCursor>,
}

impl CurrentInput {
    fn new(batch: RecordBatch) -> Self {
        let num_rows = batch.num_rows();
        Self {
            batch,
            num_rows,
            row_idx: 0,
            cursor: None,
        }
    }
}

/// The [`Stream`] driving one input partition through the tiling core.
struct TileExplodeStream {
    input: SendableRecordBatchStream,
    args: TileExplodeExecArgs,
    output_schema: SchemaRef,
    batch_size: usize,
    current: Option<CurrentInput>,
}

impl TileExplodeStream {
    fn poll_next_impl(&mut self, cx: &mut Context<'_>) -> Poll<Option<Result<RecordBatch>>> {
        loop {
            if self.current.is_none() {
                match self.input.poll_next_unpin(cx) {
                    Poll::Ready(Some(Ok(batch))) => {
                        self.current = Some(CurrentInput::new(batch));
                    }
                    Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e))),
                    Poll::Ready(None) => return Poll::Ready(None),
                    Poll::Pending => return Poll::Pending,
                }
            }

            let out = match self.build_output_batch() {
                Ok(out) => out,
                Err(e) => return Poll::Ready(Some(Err(e))),
            };

            // A batch is fully consumed once its last raster row has been tiled.
            let exhausted = self
                .current
                .as_ref()
                .map(|c| c.row_idx >= c.num_rows)
                .unwrap_or(true);
            if exhausted {
                self.current = None;
            }

            match out {
                Some(batch) => return Poll::Ready(Some(Ok(batch))),
                // The batch drained without producing rows (all remaining rasters
                // were null/untileable); pull the next input batch.
                None => continue,
            }
        }
    }

    /// Emit up to `batch_size` output rows from the current input batch, resuming
    /// from the cursor. Returns `None` only when the current batch is drained
    /// without producing any row.
    fn build_output_batch(&mut self) -> Result<Option<RecordBatch>> {
        let batch_size = self.batch_size;
        let tile_width = self.args.tile_width;
        let tile_height = self.args.tile_height;
        let params = TileParams {
            bands: self.args.bands.as_deref(),
            pad_with_nodata: self.args.pad_with_nodata,
            nodata: self.args.nodata,
        };

        let current = self
            .current
            .as_mut()
            .ok_or_else(|| exec_datafusion_err!("TileExplodeExec: no current input batch"))?;

        // Re-derive the raster view from the owned input batch each call (cheap
        // downcasts); this keeps the borrow local and avoids a self-referential
        // stream struct.
        let batch = current.batch.clone();
        let raster_struct = as_struct_array(batch.column(self.args.raster_column))?;
        let rasters = RasterStructArray::try_new(raster_struct)
            .map_err(|e| exec_datafusion_err!("TileExplodeExec: invalid raster column: {e}"))?;

        let mut x_builder = Int32Builder::new();
        let mut y_builder = Int32Builder::new();
        let mut rast_builder = RasterBuilder::new(batch_size);
        // Input row index per emitted output row, used to replicate sibling
        // columns via a single `take`.
        let mut take_indices: Vec<u32> = Vec::with_capacity(batch_size);

        while take_indices.len() < batch_size {
            if current.cursor.is_none() {
                // Advance to the next tileable raster row, skipping null/untileable
                // rows (which contribute zero rows).
                loop {
                    if current.row_idx >= current.num_rows {
                        break;
                    }
                    if rasters.is_null(current.row_idx) {
                        current.row_idx += 1;
                        continue;
                    }
                    let raster = rasters.get(current.row_idx).map_err(|e| {
                        exec_datafusion_err!("TileExplodeExec: invalid raster row: {e}")
                    })?;
                    let (width, height) = match (raster.width(), raster.height()) {
                        (Ok(width), Ok(height)) => (width, height),
                        _ => {
                            current.row_idx += 1;
                            continue;
                        }
                    };
                    let (num_tile_x, num_tile_y) =
                        tile_grid_dims(width, height, tile_width, tile_height)?;
                    let band_indices = resolve_band_indices(params.bands, raster.num_bands())?;
                    current.cursor = Some(RasterCursor {
                        num_tile_x,
                        total_tiles: num_tile_x * num_tile_y,
                        next_tile: 0,
                        band_indices,
                        width: width as usize,
                        height: height as usize,
                        tile_w: tile_width as usize,
                        tile_h: tile_height as usize,
                        pad: params.pad_with_nodata,
                    });
                    break;
                }
                if current.cursor.is_none() {
                    // No more tileable rows in this batch.
                    break;
                }
            }

            let raster = rasters
                .get(current.row_idx)
                .map_err(|e| exec_datafusion_err!("TileExplodeExec: invalid raster row: {e}"))?;
            {
                let cursor = current
                    .cursor
                    .as_mut()
                    .ok_or_else(|| exec_datafusion_err!("TileExplodeExec: missing tile cursor"))?;
                while take_indices.len() < batch_size && cursor.next_tile < cursor.total_tiles {
                    let tile_x = cursor.next_tile % cursor.num_tile_x;
                    let tile_y = cursor.next_tile / cursor.num_tile_x;
                    let window = TileWindow::new(
                        tile_x,
                        tile_y,
                        cursor.tile_w,
                        cursor.tile_h,
                        cursor.width,
                        cursor.height,
                        cursor.pad,
                    );
                    append_tile(
                        &raster,
                        &cursor.band_indices,
                        &window,
                        &params,
                        &mut rast_builder,
                    )?;
                    x_builder.append_value(tile_x as i32);
                    y_builder.append_value(tile_y as i32);
                    take_indices.push(current.row_idx as u32);
                    cursor.next_tile += 1;
                }
            }

            // Advance to the next input row once this raster is fully tiled.
            let done = current
                .cursor
                .as_ref()
                .map(|c| c.next_tile >= c.total_tiles)
                .unwrap_or(false);
            if done {
                current.cursor = None;
                current.row_idx += 1;
            }
        }

        if take_indices.is_empty() {
            return Ok(None);
        }

        let indices = UInt32Array::from(take_indices);
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns() + 3);
        for column in batch.columns() {
            columns.push(arrow::compute::take(column.as_ref(), &indices, None)?);
        }
        columns.push(Arc::new(x_builder.finish()));
        columns.push(Arc::new(y_builder.finish()));
        let tiles = rast_builder
            .finish()
            .map_err(|e| exec_datafusion_err!("TileExplodeExec: failed to build tiles: {e}"))?;
        columns.push(Arc::new(tiles));

        let out = RecordBatch::try_new(self.output_schema.clone(), columns)?;
        Ok(Some(out))
    }
}

impl Stream for TileExplodeStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.poll_next_impl(cx)
    }
}

impl RecordBatchStream for TileExplodeStream {
    fn schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::compute::concat_batches;
    use arrow_array::cast::AsArray;
    use arrow_array::StructArray;
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::datasource::memory::MemorySourceConfig;
    use datafusion::physical_plan::collect;
    use datafusion::prelude::{SessionConfig, SessionContext};
    use datafusion_common::cast::{as_int32_array, as_struct_array};
    use datafusion_common::ScalarValue;
    use datafusion_expr::{ColumnarValue, ScalarUDF};
    use sedona_schema::datatypes::{SedonaType, RASTER};
    use sedona_testing::raster_spec::{assert_rasters_equal, raster_array, RasterSpec};
    use sedona_testing::rasters::assert_raster_arrays_equal;
    use sedona_testing::testers::ScalarUdfTester;

    /// A 5x3 EPSG-less raster, origin (0, 3), north-up 1x1 pixels, one UInt8
    /// band with values 1..=15 (row-major). Its odd extent makes the last column
    /// (5 vs tile 2) and last row (3 vs tile 2) partial, exercising edge tiles.
    /// Mirrors the `source_5x3` fixture used by the tiling-core tests.
    fn source_5x3() -> RasterSpec {
        RasterSpec::d2(5, 3)
            .crs(None)
            .transform([0.0, 1.0, 0.0, 3.0, 0.0, -1.0])
            .band_values(&[1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    }

    /// A 2x2, three-band UInt8 raster so band selection is observable.
    fn three_band_2x2() -> RasterSpec {
        RasterSpec::d2(2, 2)
            .crs(None)
            .transform([0.0, 1.0, 0.0, 2.0, 0.0, -1.0])
            .band_values(&[1u8, 2, 3, 4])
            .band_values(&[10u8, 20, 30, 40])
            .band_values(&[100u8, 101, 102, 103])
    }

    /// The raster storage type, used for the single input raster column.
    fn raster_type() -> DataType {
        RASTER
            .to_storage_field("rast", true)
            .unwrap()
            .data_type()
            .clone()
    }

    /// A single-partition, single-batch input plan whose only column ("rast") is
    /// the given rasters.
    fn raster_input(specs: Vec<Option<RasterSpec>>) -> Arc<dyn ExecutionPlan> {
        let column: ArrayRef = Arc::new(raster_array(specs));
        let schema = Arc::new(Schema::new(vec![Field::new("rast", raster_type(), true)]));
        let batch = RecordBatch::try_new(schema.clone(), vec![column]).unwrap();
        let plan: Arc<dyn ExecutionPlan> =
            MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();
        plan
    }

    /// Execute a tile-explode over `input` and collect its output batches. The
    /// session `batch_size` bounds each output batch, so a small value forces the
    /// stream to emit multiple batches.
    async fn tile_explode_output(
        input: Arc<dyn ExecutionPlan>,
        args: TileExplodeExecArgs,
        batch_size: usize,
    ) -> (Vec<RecordBatch>, SchemaRef) {
        let exec = Arc::new(TileExplodeExec::try_new(input, args).unwrap());
        let schema = exec.schema();
        let ctx = SessionContext::new_with_config(SessionConfig::new().with_batch_size(batch_size));
        let batches = collect(exec, ctx.task_ctx()).await.unwrap();
        (batches, schema)
    }

    /// The `(x, y)` positions, tile rasters, and replicated input `rast` column
    /// of a collected tile-explode result. Output columns are `[rast, x, y,
    /// tile]`.
    fn subject_columns(
        batches: &[RecordBatch],
        schema: &SchemaRef,
    ) -> (Vec<i32>, Vec<i32>, StructArray, ArrayRef) {
        let batch = concat_batches(schema, batches).unwrap();
        let xs = as_int32_array(batch.column(1)).unwrap();
        let ys = as_int32_array(batch.column(2)).unwrap();
        let tiles = as_struct_array(batch.column(3)).unwrap().clone();
        (
            (0..xs.len()).map(|i| xs.value(i)).collect(),
            (0..ys.len()).map(|i| ys.value(i)).collect(),
            tiles,
            batch.column(0).clone(),
        )
    }

    /// The reference `(x, y)` positions and tile rasters that `RS_Tile` produces
    /// for the given argument list (one scalar raster row), extracted from its
    /// `List<Struct<x, y, tile>>` result.
    fn rs_tile_reference(
        arg_types: Vec<SedonaType>,
        args: Vec<ColumnarValue>,
    ) -> (Vec<i32>, Vec<i32>, StructArray) {
        let udf: ScalarUDF = sedona_raster_gdal::rs_tile_udf().into();
        let tester = ScalarUdfTester::new(udf, arg_types);
        let result = tester.invoke(args).unwrap();
        let ColumnarValue::Scalar(ScalarValue::List(list)) = result else {
            panic!("expected a scalar List result, got {result:?}");
        };
        let element = as_struct_array(list.values()).unwrap();
        let xs = as_int32_array(element.column(0)).unwrap();
        let ys = as_int32_array(element.column(1)).unwrap();
        let tiles = element.column(2).as_struct().clone();
        (
            (0..xs.len()).map(|i| xs.value(i)).collect(),
            (0..ys.len()).map(|i| ys.value(i)).collect(),
            tiles,
        )
    }

    /// Assert the tile-explode subject emits byte-identical `(x, y, tile)` rows,
    /// in the same order, as the `RS_Tile` reference.
    fn assert_parity(
        subject: &(Vec<i32>, Vec<i32>, StructArray, ArrayRef),
        reference: &(Vec<i32>, Vec<i32>, StructArray),
    ) {
        assert_eq!(subject.0, reference.0, "x grid positions differ");
        assert_eq!(subject.1, reference.1, "y grid positions differ");
        assert_raster_arrays_equal(
            &RasterStructArray::try_new(&subject.2).unwrap(),
            &RasterStructArray::try_new(&reference.2).unwrap(),
        );
    }

    fn all_bands_args(
        tile_width: i64,
        tile_height: i64,
        pad_with_nodata: bool,
        nodata: Option<f64>,
    ) -> TileExplodeExecArgs {
        TileExplodeExecArgs {
            raster_column: 0,
            tile_width,
            tile_height,
            pad_with_nodata,
            nodata,
            bands: None,
        }
    }

    #[tokio::test]
    async fn parity_multi_tile_with_partial_edges() {
        // 5x3 tiled 2x2 (no padding): a 3x2 grid with partial right/bottom edge
        // tiles. The exec must match RS_Tile exactly.
        let (batches, schema) = tile_explode_output(
            raster_input(vec![Some(source_5x3())]),
            all_bands_args(2, 2, false, None),
            8192,
        )
        .await;
        let subject = subject_columns(&batches, &schema);
        let reference = rs_tile_reference(
            vec![
                RASTER,
                SedonaType::Arrow(DataType::Int32),
                SedonaType::Arrow(DataType::Int32),
            ],
            vec![
                ColumnarValue::Scalar(source_5x3().scalar()),
                ColumnarValue::Scalar(ScalarValue::Int32(Some(2))),
                ColumnarValue::Scalar(ScalarValue::Int32(Some(2))),
            ],
        );
        assert_eq!(subject.0.len(), 6, "5x3 tiled 2x2 yields 6 tiles");
        assert_parity(&subject, &reference);
    }

    #[tokio::test]
    async fn parity_padded_edge_tiles() {
        // Same grid, but the short edge tiles are padded to 2x2 with nodata 0.
        let (batches, schema) = tile_explode_output(
            raster_input(vec![Some(source_5x3())]),
            all_bands_args(2, 2, true, Some(0.0)),
            8192,
        )
        .await;
        let subject = subject_columns(&batches, &schema);
        let reference = rs_tile_reference(
            vec![
                RASTER,
                SedonaType::Arrow(DataType::Int32),
                SedonaType::Arrow(DataType::Int32),
                SedonaType::Arrow(DataType::Boolean),
                SedonaType::Arrow(DataType::Float64),
            ],
            vec![
                ColumnarValue::Scalar(source_5x3().scalar()),
                ColumnarValue::Scalar(ScalarValue::Int32(Some(2))),
                ColumnarValue::Scalar(ScalarValue::Int32(Some(2))),
                ColumnarValue::Scalar(ScalarValue::Boolean(Some(true))),
                ColumnarValue::Scalar(ScalarValue::Float64(Some(0.0))),
            ],
        );
        assert_parity(&subject, &reference);
    }

    #[tokio::test]
    async fn parity_band_subset() {
        // Keep only band 2; the exec's `bands` param must select and order bands
        // the same way as RS_Tile's bandIndices overload.
        let band_list = ScalarValue::List(ScalarValue::new_list_nullable(
            &[ScalarValue::Int32(Some(2))],
            &DataType::Int32,
        ));
        let args = TileExplodeExecArgs {
            raster_column: 0,
            tile_width: 2,
            tile_height: 2,
            pad_with_nodata: false,
            nodata: None,
            bands: Some(vec![2]),
        };
        let (batches, schema) =
            tile_explode_output(raster_input(vec![Some(three_band_2x2())]), args, 8192).await;
        let subject = subject_columns(&batches, &schema);
        let reference = rs_tile_reference(
            vec![
                RASTER,
                SedonaType::Arrow(DataType::List(Arc::new(Field::new(
                    "item",
                    DataType::Int32,
                    true,
                )))),
                SedonaType::Arrow(DataType::Int32),
                SedonaType::Arrow(DataType::Int32),
            ],
            vec![
                ColumnarValue::Scalar(three_band_2x2().scalar()),
                ColumnarValue::Scalar(band_list),
                ColumnarValue::Scalar(ScalarValue::Int32(Some(2))),
                ColumnarValue::Scalar(ScalarValue::Int32(Some(2))),
            ],
        );
        assert_parity(&subject, &reference);
    }

    #[tokio::test]
    async fn null_and_only_null_rows_contribute_zero_rows() {
        // A NULL raster row contributes zero tiles; the surviving raster's tiles
        // are unaffected and its sibling column is replicated across them.
        let (batches, schema) = tile_explode_output(
            raster_input(vec![None, Some(source_5x3()), None]),
            all_bands_args(2, 2, false, None),
            8192,
        )
        .await;
        let subject = subject_columns(&batches, &schema);
        assert_eq!(subject.0.len(), 6, "only the one real raster tiles");
        let reference = rs_tile_reference(
            vec![
                RASTER,
                SedonaType::Arrow(DataType::Int32),
                SedonaType::Arrow(DataType::Int32),
            ],
            vec![
                ColumnarValue::Scalar(source_5x3().scalar()),
                ColumnarValue::Scalar(ScalarValue::Int32(Some(2))),
                ColumnarValue::Scalar(ScalarValue::Int32(Some(2))),
            ],
        );
        assert_parity(&subject, &reference);
        // Sibling replication: every output row carries its input row's raster
        // (all six rows come from the single non-null input row).
        let expected_sibling: Vec<Option<RasterSpec>> =
            (0..6).map(|_| Some(source_5x3())).collect();
        assert_rasters_equal(&subject.3, &expected_sibling);

        // A batch of only NULL rasters yields no output rows at all.
        let (batches, _) = tile_explode_output(
            raster_input(vec![None, None]),
            all_bands_args(2, 2, false, None),
            8192,
        )
        .await;
        assert_eq!(
            batches.iter().map(|b| b.num_rows()).sum::<usize>(),
            0,
            "all-null input yields zero output rows"
        );
    }

    #[tokio::test]
    async fn output_spans_multiple_batches() {
        // 5x3 tiled 1x1 = 15 tiles. With batch_size 4 the stream emits 4 batches
        // (4, 4, 4, 3) rather than buffering all tiles into one.
        let (batches, schema) = tile_explode_output(
            raster_input(vec![Some(source_5x3())]),
            all_bands_args(1, 1, false, None),
            4,
        )
        .await;
        assert_eq!(batches.len(), 4, "15 tiles at batch_size 4 span 4 batches");
        assert!(
            batches.iter().all(|b| b.num_rows() <= 4),
            "no batch exceeds the configured batch_size"
        );
        assert_eq!(
            batches.iter().map(|b| b.num_rows()).sum::<usize>(),
            15,
            "all 15 tiles are emitted"
        );
        let subject = subject_columns(&batches, &schema);
        let reference = rs_tile_reference(
            vec![
                RASTER,
                SedonaType::Arrow(DataType::Int32),
                SedonaType::Arrow(DataType::Int32),
            ],
            vec![
                ColumnarValue::Scalar(source_5x3().scalar()),
                ColumnarValue::Scalar(ScalarValue::Int32(Some(1))),
                ColumnarValue::Scalar(ScalarValue::Int32(Some(1))),
            ],
        );
        assert_parity(&subject, &reference);
    }
}
