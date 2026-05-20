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

//! `sd_read_zarr` table function — exposes Zarr-backed raster groups
//! as SQL-queryable tables.
//!
//! ```sql
//! SELECT * FROM sd_read_zarr('file:///path/to/datacube.zarr');
//! SELECT count(*) FROM sd_read_zarr(
//!     'file:///path/to/datacube.zarr',
//!     '{"load_eager": true, "rows_per_batch": 256}'
//! );
//! ```
//!
//! Returns a single-column table `raster: Raster` with one row per chunk
//! position in the Zarr group's chunk grid. All existing `RS_*` UDFs
//! operate on the column unchanged.
//!
//! `load_eager` defaults to `true` so byte-reading kernels work end-to-end
//! without depending on a registered OutDb resolver. When the async
//! `RS_EnsureLoaded` UDF lands, `load_eager = true` will be reinterpreted
//! as the planner auto-injecting that UDF over the scan output instead
//! of fetching at plan time.

use std::any::Any;
use std::sync::Arc;

use arrow_array::{Array, RecordBatch, StructArray};
use arrow_schema::{DataType, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::catalog::{Session, TableFunctionImpl, TableProvider};
use datafusion::common::Result;
use datafusion::datasource::TableType;
use datafusion::execution::context::TaskContext;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::expressions::Column;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PhysicalExpr, PlanProperties,
    SendableRecordBatchStream,
};
use datafusion::prelude::Expr;
use datafusion_common::{plan_err, DataFusionError, ScalarValue};
use sedona_raster_zarr::{group_to_indb_rasters, group_to_outdb_rasters};
use sedona_schema::datatypes::SedonaType;
use serde::{Deserialize, Serialize};

/// Table function `sd_read_zarr(uri[, options_json])`.
///
/// Accepts one or two string arguments:
/// - `uri` (required) — Zarr group URI (e.g. `file:///path/to/foo.zarr`).
/// - `options_json` (optional) — JSON string with any of:
///     - `load_eager`: `true` (default) materializes chunk bytes into
///       the Arrow `data` column eagerly; `false` emits chunk-anchor
///       URIs only and defers byte resolution to the OutDb loader.
///       Long-term, `load_eager = true` will instead trigger the
///       planner to inject an async `RS_EnsureLoaded` over the scan
///       output rather than fetching at plan time.
///     - `rows_per_batch`: chunks per `RecordBatch` (default 1024)
///     - `num_partitions`: scan partitions (default 1; > 1 currently errors)
///     - `arrays`: optional list of array names to read. Default reads
///       every multi-dimensional array (1-D coord variables are
///       auto-skipped); an explicit list reads exactly those arrays.
#[derive(Debug, Default)]
pub struct ZarrReadFunction {}

impl TableFunctionImpl for ZarrReadFunction {
    fn call(&self, exprs: &[Expr]) -> Result<Arc<dyn TableProvider>> {
        if exprs.is_empty() || exprs.len() > 2 {
            return plan_err!(
                "sd_read_zarr() expects 1 or 2 string arguments (uri[, options_json]); got {}",
                exprs.len()
            );
        }

        let uri = literal_utf8(&exprs[0], "sd_read_zarr() uri")?;
        let options = if exprs.len() == 2 {
            Some(literal_utf8(&exprs[1], "sd_read_zarr() options_json")?)
        } else {
            None
        };

        Ok(Arc::new(ZarrChunkProvider::try_new(&uri, options)?))
    }
}

/// Pull a `Utf8` literal out of an `Expr`. UDTF call arguments arrive as
/// `Expr::Literal(ScalarValue, _)`; anything else errors with a planner
/// message naming the parameter.
fn literal_utf8(expr: &Expr, label: &str) -> Result<String> {
    if let Expr::Literal(scalar, _) = expr {
        if let ScalarValue::Utf8(Some(s)) = scalar.cast_to(&DataType::Utf8)? {
            return Ok(s);
        }
    }
    plan_err!("{label} must be a non-null Utf8 literal; got {expr}")
}

/// Materialised view backing `sd_read_zarr`. Holds the raster
/// `StructArray` once for the query lifetime; the executor slices it
/// into `rows_per_batch`-sized `RecordBatch`es.
///
/// The StructArray is built eagerly in `try_new`. When `indb=false`
/// the array is cheap to construct (chunk anchor URIs only); when
/// `indb=true` it pulls every chunk's bytes through the loader at
/// plan time.
#[derive(Debug)]
pub struct ZarrChunkProvider {
    schema: SchemaRef,
    rasters: StructArray,
    rows_per_batch: usize,
}

impl ZarrChunkProvider {
    fn try_new(uri: &str, options_json: Option<String>) -> Result<Self> {
        let opts = parse_options(options_json.as_deref())?;
        let load_eager = opts.load_eager.unwrap_or(true);
        let rows_per_batch = opts.rows_per_batch.unwrap_or(1024).max(1);
        let num_partitions = opts.num_partitions.unwrap_or(1);
        if num_partitions != 1 {
            return plan_err!(
                "sd_read_zarr() supports only num_partitions = 1; got {num_partitions}. \
                 Round-robin partitioning lands with the OutDb resolver work."
            );
        }

        let arrays_filter = opts.arrays.as_deref();
        let rasters = if load_eager {
            group_to_indb_rasters(uri, arrays_filter).map_err(arrow_to_df_err)?
        } else {
            group_to_outdb_rasters(uri, arrays_filter).map_err(arrow_to_df_err)?
        };

        // Single-column schema: `raster: Raster`. `SedonaType::Raster` adds
        // the `sedona.raster` extension-type metadata so downstream RS_*
        // kernels recognise the column without further configuration.
        let raster_field = SedonaType::Raster
            .to_storage_field("raster", true)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        let schema = Arc::new(Schema::new(vec![raster_field]));

        Ok(Self {
            schema,
            rasters,
            rows_per_batch,
        })
    }
}

#[async_trait]
impl TableProvider for ZarrChunkProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::View
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let exec = Arc::new(ZarrChunkExec::new(
            self.schema.clone(),
            self.rasters.clone(),
            self.rows_per_batch,
        ));
        // DataFusion requires the scan to honour the projection it asks
        // for, including the empty projection used by `SELECT count(*)`.
        // Wrap the exec in a `ProjectionExec` so the physical schema
        // matches the requested column subset.
        if let Some(projection) = projection {
            let schema = self.schema();
            let exprs: Vec<(Arc<dyn PhysicalExpr>, String)> = projection
                .iter()
                .map(|index| {
                    let name = schema.field(*index).name();
                    let expr: Arc<dyn PhysicalExpr> = Arc::new(Column::new(name, *index));
                    (expr, name.clone())
                })
                .collect();
            Ok(Arc::new(ProjectionExec::try_new(exprs, exec)?))
        } else {
            Ok(exec)
        }
    }
}

#[derive(Debug)]
struct ZarrChunkExec {
    schema: SchemaRef,
    rasters: StructArray,
    rows_per_batch: usize,
    properties: PlanProperties,
}

impl ZarrChunkExec {
    fn new(schema: SchemaRef, rasters: StructArray, rows_per_batch: usize) -> Self {
        let properties = PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            // A single partition for now. Round-robin across multiple
            // partitions lands with the OutDb resolver work.
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
        Self {
            schema,
            rasters,
            rows_per_batch,
            properties,
        }
    }
}

impl DisplayAs for ZarrChunkExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "ZarrChunkExec: rows={}, rows_per_batch={}",
            self.rasters.len(),
            self.rows_per_batch,
        )
    }
}

impl ExecutionPlan for ZarrChunkExec {
    fn name(&self) -> &str {
        "ZarrChunkExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        Vec::new()
    }

    fn with_new_children(
        self: Arc<Self>,
        _: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return plan_err!("ZarrChunkExec: only partition 0 exists");
        }

        let total = self.rasters.len();
        let rows_per_batch = self.rows_per_batch;
        let schema = self.schema.clone();
        let rasters = self.rasters.clone();

        // Build all batches eagerly into a Vec, then turn into a stream.
        // The StructArray is already materialised in the provider, slicing
        // is O(1), and the only allocation per batch is the RecordBatch
        // wrapper. Lazy streaming becomes interesting once the loader
        // itself is lazy.
        let mut batches = Vec::with_capacity(total.div_ceil(rows_per_batch).max(1));
        let mut offset = 0;
        while offset < total {
            let len = (total - offset).min(rows_per_batch);
            let slice = rasters.slice(offset, len);
            let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(slice)])?;
            batches.push(Ok(batch));
            offset += len;
        }

        let stream = futures::stream::iter(batches);
        let adapter = RecordBatchStreamAdapter::new(schema, stream);
        Ok(Box::pin(adapter))
    }
}

/// Convert an `ArrowError` from the Zarr loader into a `DataFusionError`
/// suitable for planner output.
fn arrow_to_df_err(e: arrow_schema::ArrowError) -> DataFusionError {
    DataFusionError::External(Box::new(e))
}

#[derive(Serialize, Deserialize, Default)]
struct ZarrReadOptions {
    /// `true` (default) materializes chunk bytes into the Arrow `data`
    /// column eagerly; `false` emits chunk-anchor URIs only and defers
    /// byte resolution to the OutDb loader. Long-term, `load_eager =
    /// true` will trigger the planner to inject an async
    /// `RS_EnsureLoaded` over the scan output rather than fetching at
    /// plan time.
    load_eager: Option<bool>,
    rows_per_batch: Option<usize>,
    num_partitions: Option<usize>,
    /// Explicit array-name filter. `None` reads every multi-dimensional
    /// array in the group; `Some` reads exactly the listed arrays (in
    /// the order zarrs's store listing returns them). Unknown names
    /// error so a typo doesn't silently yield an empty result.
    arrays: Option<Vec<String>>,
}

fn parse_options(options_json: Option<&str>) -> Result<ZarrReadOptions> {
    let Some(s) = options_json else {
        return Ok(ZarrReadOptions::default());
    };
    serde_json::from_str(s).map_err(|e| {
        DataFusionError::Plan(format!(
            "sd_read_zarr() options must be valid JSON: {e}\noptions were: {s}"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::prelude::SessionContext;
    use sedona_raster::array::RasterStructArray;
    use sedona_raster::traits::RasterRef;
    use std::sync::Arc;
    use tempfile::TempDir;
    use zarrs::array::data_type;
    use zarrs::array::ArrayBuilder;
    use zarrs::group::GroupBuilder;
    use zarrs_filesystem::FilesystemStore;

    /// Build a tiny 1-array Zarr group on disk and return the temp dir.
    /// 2×2 UInt8 with chunks [1, 2] → chunk grid [2, 1] = 2 chunk rows.
    fn build_fixture() -> TempDir {
        let tmp = TempDir::new().unwrap();
        let store = Arc::new(FilesystemStore::new(tmp.path()).unwrap());
        GroupBuilder::new()
            .build(store.clone(), "/")
            .unwrap()
            .store_metadata()
            .unwrap();
        let array = ArrayBuilder::new(vec![2u64, 2u64], vec![1u64, 2u64], data_type::uint8(), 0u8)
            .dimension_names(Some(["y", "x"]))
            .build(store.clone(), "/temperature")
            .unwrap();
        array.store_metadata().unwrap();
        array.store_chunk(&[0, 0], vec![10u8, 11]).unwrap();
        array.store_chunk(&[1, 0], vec![20u8, 21]).unwrap();
        tmp
    }

    #[tokio::test]
    async fn udtf_returns_one_row_per_chunk_position_with_pixel_bytes() {
        let tmp = build_fixture();
        let uri = format!("file://{}", tmp.path().display());

        let ctx = SessionContext::new();
        ctx.register_udtf("sd_read_zarr", Arc::new(ZarrReadFunction::default()));

        let df = ctx
            .sql(&format!("SELECT raster FROM sd_read_zarr('{uri}')"))
            .await
            .unwrap();
        let batches = df.collect().await.unwrap();

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 2, "expected 2 chunk rows");

        // Pull the raster column out, hand to RasterStructArray, verify
        // chunk 0's bytes round-trip.
        let raster_col = batches[0].column(0);
        let struct_arr = raster_col
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("raster column is StructArray");
        let rasters = RasterStructArray::new(struct_arr);
        let r0 = rasters.get(0).unwrap();
        let band = r0.band(0).unwrap();
        assert_eq!(&*band.contiguous_data().unwrap(), &[10u8, 11]);
    }

    #[tokio::test]
    async fn udtf_respects_rows_per_batch_option() {
        let tmp = build_fixture();
        let uri = format!("file://{}", tmp.path().display());

        let ctx = SessionContext::new();
        ctx.register_udtf("sd_read_zarr", Arc::new(ZarrReadFunction::default()));

        // 2 chunk rows, rows_per_batch=1 → 2 single-row batches.
        let df = ctx
            .sql(&format!(
                r#"SELECT raster FROM sd_read_zarr('{uri}', '{{"rows_per_batch": 1}}')"#,
            ))
            .await
            .unwrap();
        let batches = df.collect().await.unwrap();
        assert_eq!(batches.len(), 2);
        assert!(batches.iter().all(|b| b.num_rows() == 1));
    }

    #[tokio::test]
    async fn udtf_count_works_without_reading_pixel_bytes() {
        let tmp = build_fixture();
        let uri = format!("file://{}", tmp.path().display());

        let ctx = SessionContext::new();
        ctx.register_udtf("sd_read_zarr", Arc::new(ZarrReadFunction::default()));

        // OutDb mode: byte fetching is deferred. SELECT count(*) just
        // walks the chunk grid metadata.
        let df = ctx
            .sql(&format!(
                r#"SELECT count(*) FROM sd_read_zarr('{uri}', '{{"load_eager": false}}')"#,
            ))
            .await
            .unwrap();
        let batches = df.collect().await.unwrap();
        let count_arr = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .unwrap();
        assert_eq!(count_arr.value(0), 2);
    }

    #[tokio::test]
    async fn udtf_rejects_multi_partition_in_phase1() {
        let tmp = build_fixture();
        let uri = format!("file://{}", tmp.path().display());

        let ctx = SessionContext::new();
        ctx.register_udtf("sd_read_zarr", Arc::new(ZarrReadFunction::default()));

        let err = ctx
            .sql(&format!(
                r#"SELECT raster FROM sd_read_zarr('{uri}', '{{"num_partitions": 2}}')"#,
            ))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("num_partitions = 1"), "got: {err}");
    }

    #[tokio::test]
    async fn udtf_arrays_filter_threads_through_sql() {
        // Build a group with one data array + a 1-D coord variable in a
        // different chunk grid. Without auto-skip this errors; with the
        // explicit arrays filter the user gets exactly what they asked
        // for.
        let tmp = TempDir::new().unwrap();
        let store = Arc::new(FilesystemStore::new(tmp.path()).unwrap());
        GroupBuilder::new()
            .build(store.clone(), "/")
            .unwrap()
            .store_metadata()
            .unwrap();
        let temperature =
            ArrayBuilder::new(vec![2u64, 2u64], vec![1u64, 2u64], data_type::uint8(), 0u8)
                .dimension_names(Some(["y", "x"]))
                .build(store.clone(), "/temperature")
                .unwrap();
        temperature.store_metadata().unwrap();
        temperature.store_chunk(&[0, 0], vec![10u8, 11]).unwrap();
        temperature.store_chunk(&[1, 0], vec![20u8, 21]).unwrap();
        let y = ArrayBuilder::new(vec![2u64], vec![2u64], data_type::uint8(), 0u8)
            .dimension_names(Some(["y"]))
            .build(store.clone(), "/y")
            .unwrap();
        y.store_metadata().unwrap();

        let uri = format!("file://{}", tmp.path().display());
        let ctx = SessionContext::new();
        ctx.register_udtf("sd_read_zarr", Arc::new(ZarrReadFunction::default()));

        // Default behaviour: 1-D coord variable auto-skipped, read succeeds.
        let df = ctx
            .sql(&format!("SELECT raster FROM sd_read_zarr('{uri}')"))
            .await
            .unwrap();
        assert_eq!(
            df.collect()
                .await
                .unwrap()
                .iter()
                .map(|b| b.num_rows())
                .sum::<usize>(),
            2
        );

        // Explicit filter to the data array — same result.
        let df = ctx
            .sql(&format!(
                r#"SELECT raster FROM sd_read_zarr('{uri}', '{{"arrays":["temperature"]}}')"#
            ))
            .await
            .unwrap();
        assert_eq!(
            df.collect()
                .await
                .unwrap()
                .iter()
                .map(|b| b.num_rows())
                .sum::<usize>(),
            2
        );

        // Unknown name surfaces as a clear error.
        let err = ctx
            .sql(&format!(
                r#"SELECT raster FROM sd_read_zarr('{uri}', '{{"arrays":["humidity"]}}')"#
            ))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("humidity"), "got: {err}");
    }

    #[tokio::test]
    async fn udtf_rejects_malformed_options_json() {
        let ctx = SessionContext::new();
        ctx.register_udtf("sd_read_zarr", Arc::new(ZarrReadFunction::default()));

        let err = ctx
            .sql(r#"SELECT raster FROM sd_read_zarr('file:///nowhere', '{not json}')"#)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("must be valid JSON"), "got: {err}");
    }
}
