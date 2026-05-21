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
//!     '{"rows_per_batch": 256}'
//! );
//! ```
//!
//! Returns a single-column table `raster: Raster` with one row per chunk
//! position in the Zarr group's chunk grid. All existing `RS_*` UDFs
//! operate on the column unchanged.
//!
//! Every row has empty `data` and a chunk-anchor URI in `outdb_uri`.
//! Pixel-byte materialisation lands with the async `RS_EnsureLoaded`
//! resolver in a follow-up PR; until then every metadata-only query
//! (`count(*)`, `RS_Envelope`, `RS_Width`, …) works against the
//! anchor-only rows.
//!
//! The `sedonadb-zarr` Python package constructs
//! `Arc::new(ZarrReadFunction::default())` and hands it to its session
//! via the plugin capsule. The `sedona` crate itself does not register
//! the UDTF — keeping zarr functionality out of the default bootstrap.

use std::any::Any;
use std::sync::Arc;

use arrow_schema::{DataType, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion_catalog::{Session, TableFunctionImpl, TableProvider};
use datafusion_common::{plan_err, DataFusionError, Result, ScalarValue};
use datafusion_execution::TaskContext;
use datafusion_expr::{Expr, TableType};
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream,
};
use sedona_schema::datatypes::SedonaType;
use serde::{Deserialize, Serialize};

use crate::loader::ZarrChunkReader;

/// Table function `sd_read_zarr(uri[, options_json])`.
///
/// Accepts one or two string arguments:
/// - `uri` (required) — Zarr group URI (e.g. `file:///path/to/foo.zarr`).
/// - `options_json` (optional) — JSON string with any of:
///     - `rows_per_batch`: chunks per `RecordBatch`. Defaults to the
///       session's configured batch size (`SessionConfig::batch_size`,
///       typically 8192) when unset.
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

/// `TableProvider` backing `sd_read_zarr`. Holds the URI + options;
/// the underlying `ZarrChunkReader` is opened fresh per `execute()`
/// call so each scan streams independently.
///
/// Plan-time validation: `try_new` opens the group once and drops the
/// reader, so problems like "URI doesn't resolve", "no arrays",
/// "CRS without transform", and "unknown array name in the `arrays`
/// filter" surface at `ctx.sql(...).await` rather than at collect time.
/// The cost is one extra group-open per scan; for local files this is
/// negligible and for cloud it's two GETs — acceptable until lazy-open
/// becomes a measurable problem.
#[derive(Debug)]
pub struct ZarrChunkProvider {
    schema: SchemaRef,
    uri: String,
    arrays_filter: Option<Vec<String>>,
    /// `None` defers to the session's `SessionConfig::batch_size` at
    /// execute time. Set explicitly via the `rows_per_batch` JSON option.
    rows_per_batch: Option<usize>,
}

impl ZarrChunkProvider {
    fn try_new(uri: &str, options_json: Option<String>) -> Result<Self> {
        let opts = parse_options(options_json.as_deref())?;
        let rows_per_batch = opts.rows_per_batch.map(|n| n.max(1));
        let num_partitions = opts.num_partitions.unwrap_or(1);
        if num_partitions != 1 {
            return plan_err!(
                "sd_read_zarr() supports only num_partitions = 1; got {num_partitions}. \
                 Round-robin partitioning lands with the OutDb resolver work."
            );
        }

        let arrays_filter = opts.arrays;

        // Validate at plan time by opening the reader once and dropping
        // it. Surfaces URI / metadata / arrays-filter errors at
        // ctx.sql() rather than at collect().
        let _ =
            ZarrChunkReader::try_new(uri, arrays_filter.as_deref(), 1).map_err(arrow_to_df_err)?;

        // Single-column schema: `raster: Raster`. `SedonaType::Raster` adds
        // the `sedona.raster` extension-type metadata so downstream RS_*
        // kernels recognise the column without further configuration.
        let raster_field = SedonaType::Raster
            .to_storage_field("raster", true)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        let schema = Arc::new(Schema::new(vec![raster_field]));

        Ok(Self {
            schema,
            uri: uri.to_string(),
            arrays_filter,
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
            self.uri.clone(),
            self.arrays_filter.clone(),
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
    uri: String,
    arrays_filter: Option<Vec<String>>,
    rows_per_batch: Option<usize>,
    properties: PlanProperties,
}

impl ZarrChunkExec {
    fn new(
        schema: SchemaRef,
        uri: String,
        arrays_filter: Option<Vec<String>>,
        rows_per_batch: Option<usize>,
    ) -> Self {
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
            uri,
            arrays_filter,
            rows_per_batch,
            properties,
        }
    }
}

impl DisplayAs for ZarrChunkExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.rows_per_batch {
            Some(n) => write!(f, "ZarrChunkExec: uri={}, rows_per_batch={n}", self.uri),
            None => write!(
                f,
                "ZarrChunkExec: uri={}, rows_per_batch=session_default",
                self.uri
            ),
        }
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
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return plan_err!("ZarrChunkExec: only partition 0 exists");
        }

        // Defer the default to the session's batch size so users can
        // tune via SessionConfig instead of relying on a hard-coded
        // constant baked into this UDTF.
        let batch_size = self
            .rows_per_batch
            .unwrap_or_else(|| context.session_config().batch_size())
            .max(1);

        let reader = ZarrChunkReader::try_new(&self.uri, self.arrays_filter.as_deref(), batch_size)
            .map_err(arrow_to_df_err)?;

        // The reader is a sync Iterator<Item = Result<RecordBatch, ArrowError>>.
        // Wrap it in futures::stream::iter to produce a SendableRecordBatchStream.
        // Each next() walks `batch_size` chunk positions of the chunk grid;
        // there's no I/O until/unless the OutDb resolver materialises bytes.
        let stream = futures::stream::iter(
            reader.map(|r| r.map_err(|e| DataFusionError::External(Box::new(e)))),
        );
        let adapter = RecordBatchStreamAdapter::new(self.schema.clone(), stream);
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
    use arrow_array::{RecordBatch, StructArray};
    use datafusion::prelude::SessionContext;
    use sedona_raster::array::RasterStructArray;
    use sedona_raster::traits::RasterRef;
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
    async fn udtf_returns_one_row_per_chunk_position_with_outdb_anchor() {
        let tmp = build_fixture();
        let uri = format!("file://{}", tmp.path().display());

        let ctx = SessionContext::new();
        ctx.register_udtf(
            "sd_read_zarr",
            std::sync::Arc::new(ZarrReadFunction::default()),
        );

        let df = ctx
            .sql(&format!("SELECT raster FROM sd_read_zarr('{uri}')"))
            .await
            .unwrap();
        let batches = df.collect().await.unwrap();

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 2, "expected 2 chunk rows");

        let raster_col = batches[0].column(0);
        let struct_arr = raster_col
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("raster column is StructArray");
        let rasters = RasterStructArray::new(struct_arr);
        let r0 = rasters.get(0).unwrap();
        let band = r0.band(0).unwrap();
        assert!(!band.is_indb(), "loader emits OutDb rows");
        assert_eq!(band.outdb_format(), Some("zarr"));
        let anchor = band.outdb_uri().expect("outdb_uri set");
        assert!(anchor.contains("#array=temperature"), "got: {anchor}");
        assert!(anchor.contains("&chunk=0,0"), "got: {anchor}");
    }

    #[tokio::test]
    async fn udtf_respects_rows_per_batch_option() {
        let tmp = build_fixture();
        let uri = format!("file://{}", tmp.path().display());

        let ctx = SessionContext::new();
        ctx.register_udtf(
            "sd_read_zarr",
            std::sync::Arc::new(ZarrReadFunction::default()),
        );

        // 2 chunk rows, rows_per_batch=1 → 2 single-row batches.
        let df = ctx
            .sql(&format!(
                r#"SELECT raster FROM sd_read_zarr('{uri}', '{{"rows_per_batch": 1}}')"#,
            ))
            .await
            .unwrap();
        let batches: Vec<RecordBatch> = df.collect().await.unwrap();
        assert_eq!(batches.len(), 2);
        assert!(batches.iter().all(|b| b.num_rows() == 1));
    }

    #[tokio::test]
    async fn udtf_count_works_without_reading_pixel_bytes() {
        let tmp = build_fixture();
        let uri = format!("file://{}", tmp.path().display());

        let ctx = SessionContext::new();
        ctx.register_udtf(
            "sd_read_zarr",
            std::sync::Arc::new(ZarrReadFunction::default()),
        );

        // SELECT count(*) just walks the chunk grid metadata; never opens
        // a chunk file.
        let df = ctx
            .sql(&format!("SELECT count(*) FROM sd_read_zarr('{uri}')"))
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
    async fn udtf_rejects_multi_partition() {
        let tmp = build_fixture();
        let uri = format!("file://{}", tmp.path().display());

        let ctx = SessionContext::new();
        ctx.register_udtf(
            "sd_read_zarr",
            std::sync::Arc::new(ZarrReadFunction::default()),
        );

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
        ctx.register_udtf(
            "sd_read_zarr",
            std::sync::Arc::new(ZarrReadFunction::default()),
        );

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
        ctx.register_udtf(
            "sd_read_zarr",
            std::sync::Arc::new(ZarrReadFunction::default()),
        );

        let err = ctx
            .sql(r#"SELECT raster FROM sd_read_zarr('file:///nowhere', '{not json}')"#)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("must be valid JSON"), "got: {err}");
    }
}
