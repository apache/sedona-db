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
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};

use arrow_array::RecordBatch;
use async_trait::async_trait;
use datafusion::{
    catalog::TableProvider,
    common::{plan_datafusion_err, plan_err},
    error::{DataFusionError, Result},
    execution::{
        context::DataFilePaths, runtime_env::RuntimeEnvBuilder, SendableRecordBatchStream,
        SessionStateBuilder,
    },
    prelude::{DataFrame, SessionConfig, SessionContext},
    sql::parser::{DFParser, Statement},
};
use datafusion_expr::sqlparser::dialect::{dialect_from_str, Dialect};
use parking_lot::Mutex;
use sedona_common::option::add_sedona_option_extension;
use sedona_expr::aggregate_udf::SedonaAccumulatorRef;
use sedona_expr::{
    function_set::FunctionSet,
    projection::wrap_batch_maybe_deprecated,
    scalar_udf::{ArgMatcher, ScalarKernelRef},
};
use sedona_geoparquet::{
    format::GeoParquetFormatFactory,
    provider::{geoparquet_listing_table, GeoParquetReadOptions},
};
use sedona_schema::datatypes::SedonaType;

use crate::{
    catalog::DynamicObjectStoreCatalog,
    object_storage::ensure_object_store_registered,
    projection::{unwrap_df, wrap_df},
    random_geometry_provider::RandomGeometryFunction,
    show::{show_batches, DisplayTableOptions},
};
use crate::{exec::create_plan_from_sql, projection::unwrap_stream};

/// Sedona SessionContext wrapper
///
/// As Sedona extends DataFusion, we also extend its context and include the
/// default geometry-specific functions and datasources (which may vary depending
/// on the feature flags used to build the sedona crate). This provides a common
/// interface for configuring the behaviour of
pub struct SedonaContext {
    pub ctx: SessionContext,
    pub functions: FunctionSet,
}

impl SedonaContext {
    /// Creates a new context with default options
    pub fn new() -> Self {
        // This will panic only if the default build settings are
        // incorrect which we test!
        Self::new_from_context(SessionContext::new()).unwrap()
    }

    /// Creates a new context with default interactive options
    ///
    /// Initializes a context from the current environment and registers access
    /// to the local file system.
    pub async fn new_local_interactive() -> Result<Self> {
        // These three objects enable configuring various elements of the runtime.
        // Eventually we probably want to have a common set of configuration parameters
        // exposed via the CLI/Python as arguments, via ADBC as connection options,
        // and perhaps for all of these initializing them optionally from environment
        // variables.
        let session_config = SessionConfig::from_env()?.with_information_schema(true);
        let session_config = add_sedona_option_extension(session_config);
        let rt_builder = RuntimeEnvBuilder::new();
        let runtime_env = rt_builder.build_arc()?;

        let mut state_builder = SessionStateBuilder::new()
            .with_default_features()
            .with_runtime_env(runtime_env)
            .with_config(session_config);

        // Register the spatial join planner extension
        #[cfg(feature = "spatial-join")]
        {
            state_builder = sedona_spatial_join::register_spatial_join_optimizer(state_builder);
        }

        let mut state = state_builder.build();
        state.register_file_format(Arc::new(GeoParquetFormatFactory::new()), true)?;

        // Enable dynamic file query (i.e., select * from 'filename')
        let ctx = SessionContext::new_with_state(state).enable_url_table();

        // Install dynamic catalog provider that can register required object stores
        ctx.refresh_catalogs().await?;
        ctx.register_catalog_list(Arc::new(DynamicObjectStoreCatalog::new(
            ctx.state().catalog_list().clone(),
            ctx.state_weak_ref(),
        )));

        Self::new_from_context(ctx)
    }

    /// Creates a new context from a previously configured DataFusion context
    pub fn new_from_context(ctx: SessionContext) -> Result<Self> {
        let mut out = Self {
            ctx,
            functions: FunctionSet::new(),
        };

        // Register table functions
        out.ctx.register_udtf(
            "sd_random_geometry",
            Arc::new(RandomGeometryFunction::default()),
        );

        // Always register default function set
        out.register_function_set(sedona_functions::register::default_function_set());

        // Register geos scalar kernels if built with geos support
        #[cfg(feature = "geos")]
        out.register_scalar_kernels(sedona_geos::register::scalar_kernels().into_iter())?;

        // Register geo kernels if built with geo support
        #[cfg(feature = "geo")]
        out.register_scalar_kernels(sedona_geo::register::scalar_kernels().into_iter())?;

        // Register geo aggregate kernels if built with geo support
        #[cfg(feature = "geo")]
        out.register_aggregate_kernels(sedona_geo::register::aggregate_kernels().into_iter())?;

        // Register s2geography scalar kernels if built with s2geography support
        #[cfg(feature = "s2geography")]
        out.register_scalar_kernels(sedona_s2geography::register::scalar_kernels().into_iter())?;

        // Always register proj scalar kernels (although actually calling them will error
        // without this feature unless sedona_proj::register::configure_global_proj_engine()
        // is called).
        out.register_scalar_kernels(sedona_proj::register::scalar_kernels().into_iter())?;

        Ok(out)
    }

    /// Register all functions in a [FunctionSet] with this context
    pub fn register_function_set(&mut self, function_set: FunctionSet) {
        for udf in function_set.scalar_udfs() {
            self.functions.insert_scalar_udf(udf.clone());
            self.ctx.register_udf(udf.clone().into());
        }

        for udf in function_set.aggregate_udfs() {
            self.functions.insert_aggregate_udf(udf.clone());
            self.ctx.register_udaf(udf.clone().into());
        }
    }

    /// Register a collection of kernels with this context
    pub fn register_scalar_kernels<'a>(
        &mut self,
        kernels: impl Iterator<Item = (&'a str, ScalarKernelRef)>,
    ) -> Result<()> {
        for (name, kernel) in kernels {
            let udf = self.functions.add_scalar_udf_kernel(name, kernel)?;
            self.ctx.register_udf(udf.clone().into());
        }

        Ok(())
    }

    pub fn register_aggregate_kernels<'a>(
        &mut self,
        kernels: impl Iterator<Item = (&'a str, SedonaAccumulatorRef)>,
    ) -> Result<()> {
        for (name, kernel) in kernels {
            let udf = self.functions.add_aggregate_udf_kernel(name, kernel)?;
            self.ctx.register_udaf(udf.clone().into());
        }

        Ok(())
    }

    /// Creates a [`DataFrame`] from SQL query text that may contain one or more
    /// statements
    pub async fn multi_sql(&self, sql: &str) -> Result<Vec<DataFrame>> {
        let task_ctx = self.ctx.task_ctx();
        let dialect_str = &task_ctx.session_config().options().sql_parser.dialect;
        let dialect = ThreadSafeDialect::try_new(dialect_str)?;

        let statements = dialect.parse(sql)?;
        let mut results = Vec::with_capacity(statements.len());
        for statement in statements {
            let plan = create_plan_from_sql(self, statement.clone()).await?;
            let df = self.ctx.execute_logical_plan(plan).await?;
            results.push(df);
        }

        Ok(results)
    }

    /// Creates a [`DataFrame`] from SQL query text containing a single statement
    pub async fn sql(&self, sql: &str) -> Result<DataFrame> {
        let results = self.multi_sql(sql).await?;
        if results.len() != 1 {
            return plan_err!("Expected single SQL statement");
        }

        Ok(results[0].clone())
    }

    /// Creates a [`DataFrame`] for reading a Parquet file with Geo type support
    ///
    /// This is the geo-enabled version of [SessionContext::read_parquet].
    pub async fn read_parquet<P: DataFilePaths>(
        &self,
        table_paths: P,
        options: GeoParquetReadOptions<'_>,
    ) -> Result<DataFrame> {
        let urls = table_paths.to_urls()?;
        let provider =
            match geoparquet_listing_table(&self.ctx, urls.clone(), options.clone()).await {
                Ok(provider) => provider,
                Err(e) => {
                    if urls.is_empty() {
                        return Err(e);
                    }

                    ensure_object_store_registered(&mut self.ctx.state(), urls[0].as_str()).await?;
                    geoparquet_listing_table(&self.ctx, urls, options).await?
                }
            };

        self.ctx.read_table(Arc::new(provider))
    }

    /// Registers the [`RecordBatch`] as the specified table name
    pub fn register_batch(
        &self,
        table_name: &str,
        batch: RecordBatch,
    ) -> Result<Option<Arc<dyn TableProvider>>> {
        self.ctx
            .register_batch(table_name, wrap_batch_maybe_deprecated(batch))
    }

    /// Creates a [`DataFrame`] for reading a [`RecordBatch`]
    pub fn read_batch(&self, batch: RecordBatch) -> Result<DataFrame> {
        self.ctx.read_batch(wrap_batch_maybe_deprecated(batch))
    }

    /// Create a [`DataFrame`] for reading a [`Vec[`RecordBatch`]`]
    pub fn read_batches(
        &self,
        batches: impl IntoIterator<Item = RecordBatch>,
    ) -> Result<DataFrame> {
        wrap_df(self.ctx.read_batches(batches)?)
    }
}

impl Default for SedonaContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Sedona-specific [`DataFrame`] actions
///
/// This trait, implemented for [`DataFrame`], extends the DataFrame API to make it
/// ergonomic to work with dataframes that contain geometry columns. Currently these
/// are limited to output functions, as geometry columns currently require special
/// handling when written or exported to an external system.
#[async_trait]
pub trait SedonaDataFrame {
    /// Execute this `DataFrame` and buffer all resulting `RecordBatch`es  into memory.
    ///
    /// Because user-defined data types currently require special handling to work with
    /// DataFusion internals, output with geometry columns must use this function to
    /// be recognized as an Arrow extension type in an external system.
    async fn collect_sedona(self) -> Result<Vec<RecordBatch>>;

    /// Execute this `DataFrame` and stream batches to the consumer
    ///
    /// Because user-defined data types currently require special handling to work with
    /// DataFusion internals, output with geometry columns must use this function to
    /// be recognized as an Arrow extension type in an external system.
    async fn execute_stream_sedona(self) -> Result<SendableRecordBatchStream>;

    /// Return the indices of the columns that are geometry or geography
    fn geometry_column_indices(&self) -> Result<Vec<usize>>;

    /// Return the index of the column that should be considered the primary geometry
    ///
    /// This applies a heuritic to detect the "primary" geometry column for operations
    /// that need this information (e.g., creating a GeoPandas GeoDataFrame). The
    /// heuristic chooses (1) the column named "geometry", (2) the column name
    /// "geography", (3) the column named "geom", (4) the column named "geog",
    /// or (5) the first column with a geometry or geography data type.
    fn primary_geometry_column_index(&self) -> Result<Option<usize>>;

    /// Build a table of the first `limit` results in this DataFrame
    ///
    /// This will limit and execute the query and build a table using [show_batches].
    async fn show_sedona<'a>(
        self,
        ctx: &SedonaContext,
        limit: Option<usize>,
        options: DisplayTableOptions<'a>,
    ) -> Result<String>;
}

#[async_trait]
impl SedonaDataFrame for DataFrame {
    async fn collect_sedona(self) -> Result<Vec<RecordBatch>> {
        let (schema, df) = unwrap_df(self)?;
        let schema_ref = Arc::new(schema.as_arrow().clone());
        let batches = df.collect().await?;

        let unwrapped_batches: Result<Vec<_>> = batches
            .iter()
            .map(|batch| {
                batch
                    .clone()
                    .with_schema(schema_ref.clone())
                    .map_err(|err| {
                        DataFusionError::Internal(format!("batch.with_schema() failed {err}"))
                    })
            })
            .collect();

        unwrapped_batches
    }

    /// Executes this DataFrame and returns a stream over a single partition
    async fn execute_stream_sedona(self) -> Result<SendableRecordBatchStream> {
        Ok(unwrap_stream(self.execute_stream().await?))
    }

    fn geometry_column_indices(&self) -> Result<Vec<usize>> {
        let mut indices = Vec::new();
        let matcher = ArgMatcher::is_geometry_or_geography();
        for (i, field) in self.schema().fields().iter().enumerate() {
            if matcher.match_type(&SedonaType::from_data_type(field.data_type())?) {
                indices.push(i);
            }
        }

        Ok(indices)
    }

    fn primary_geometry_column_index(&self) -> Result<Option<usize>> {
        let indices = self.geometry_column_indices()?;
        if indices.is_empty() {
            return Ok(None);
        }

        let names_map = indices
            .iter()
            .rev()
            .map(|i| (self.schema().field(*i).name().to_lowercase(), *i))
            .collect::<HashMap<_, _>>();

        for special_name in ["geometry", "geography", "geom", "geog"] {
            if let Some(i) = names_map.get(special_name) {
                return Ok(Some(*i));
            }
        }

        Ok(Some(indices[0]))
    }

    async fn show_sedona<'a>(
        self,
        ctx: &SedonaContext,
        limit: Option<usize>,
        options: DisplayTableOptions<'a>,
    ) -> Result<String> {
        let df = self.limit(0, limit)?;
        let schema_without_qualifiers = df.schema().clone().strip_qualifiers();
        let schema = schema_without_qualifiers.as_arrow();
        let batches = df.collect().await?;
        let mut out = Vec::new();
        show_batches(ctx, &mut out, schema, batches, options)?;
        String::from_utf8(out).map_err(|e| DataFusionError::External(Box::new(e)))
    }
}

// Because Dialect/dialect_from_str is not marked as Send, using the async
// function in certain contexts will fail to compile. Here we use a wrapper
// to ensure that that the Dialect can be specified and parsed in any async
// function.
#[derive(Debug)]
struct ThreadSafeDialect {
    inner: Mutex<Box<dyn Dialect>>,
}

unsafe impl Send for ThreadSafeDialect {}

impl ThreadSafeDialect {
    pub fn try_new(dialect_str: &str) -> Result<Self> {
        let dialect = dialect_from_str(dialect_str)
            .ok_or_else(|| plan_datafusion_err!("Unsupported SQL dialect: {dialect_str}"))?;
        Ok(Self {
            inner: dialect.into(),
        })
    }

    pub fn parse(&self, sql: &str) -> Result<VecDeque<Statement>> {
        let dialect = self.inner.lock();
        DFParser::parse_sql_with_dialect(sql, dialect.as_ref())
    }
}

#[cfg(test)]
mod tests {

    use arrow_array::create_array;
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::assert_batches_eq;
    use futures::TryStreamExt;
    use sedona_schema::{
        crs::lnglat,
        datatypes::{Edges, SedonaType, WKB_GEOMETRY},
    };
    use sedona_testing::{create::create_array_storage, data::test_geoparquet};

    use super::*;

    fn test_batch() -> Result<RecordBatch> {
        let schema = Schema::new(vec![
            Field::new("idx", DataType::Int32, true),
            WKB_GEOMETRY.to_storage_field("geometry", true)?,
        ]);
        let idx = create_array!(Int32, [1, 2, 3]);
        let wkb_array = create_array_storage(
            &[
                Some("POINT (1 2)"),
                Some("POINT (3 4)"),
                Some("POINT (5 6)"),
            ],
            &WKB_GEOMETRY,
        );
        Ok(RecordBatch::try_new(Arc::new(schema), vec![idx, wkb_array]).unwrap())
    }

    #[tokio::test]
    async fn basic_sql() -> Result<()> {
        let ctx = SedonaContext::new();

        let batches = ctx
            .sql("SELECT ST_AsText(ST_Point(30, 10)) AS geom")
            .await?
            .collect()
            .await?;
        assert_batches_eq!(
            [
                "+--------------+",
                "| geom         |",
                "+--------------+",
                "| POINT(30 10) |",
                "+--------------+",
            ],
            &batches
        );

        Ok(())
    }

    #[tokio::test]
    async fn register_batch() -> Result<()> {
        let ctx = SedonaContext::new();
        ctx.register_batch("test_batch", test_batch()?)?;
        let batches = ctx
            .sql("SELECT idx, ST_AsText(geometry) AS geometry FROM test_batch")
            .await?
            .collect()
            .await?;
        assert_batches_eq!(
            [
                "+-----+------------+",
                "| idx | geometry   |",
                "+-----+------------+",
                "| 1   | POINT(1 2) |",
                "| 2   | POINT(3 4) |",
                "| 3   | POINT(5 6) |",
                "+-----+------------+",
            ],
            &batches
        );
        Ok(())
    }

    #[tokio::test]
    async fn read_batch() -> Result<()> {
        let ctx = SedonaContext::new();
        let batch_in = test_batch()?;

        let df = ctx.read_batch(batch_in.clone())?;
        let geometry_physical_type: SedonaType = df.schema().field(1).data_type().try_into()?;
        assert_eq!(geometry_physical_type, WKB_GEOMETRY);

        let batches_out = df.collect_sedona().await?;
        assert_eq!(batches_out.len(), 1);
        assert_eq!(batches_out[0], batch_in);

        Ok(())
    }

    #[tokio::test]
    async fn read_batches() -> Result<()> {
        let ctx = SedonaContext::new();
        let batch_in = test_batch()?;

        let df = ctx.read_batches(vec![batch_in.clone(), batch_in.clone()])?;
        let geometry_physical_type: SedonaType = df.schema().field(1).data_type().try_into()?;
        assert_eq!(geometry_physical_type, WKB_GEOMETRY);

        let batches_out = df.collect_sedona().await?;
        assert_eq!(batches_out.len(), 2);
        assert_eq!(batches_out[0], batch_in);
        assert_eq!(batches_out[1], batch_in);

        Ok(())
    }

    #[tokio::test]
    async fn execute_stream() -> Result<()> {
        let ctx = SedonaContext::new();
        let batch_in = test_batch()?;

        let df = ctx.read_batches(vec![batch_in.clone(), batch_in.clone()])?;
        let stream = df.execute_stream_sedona().await?;
        let geometry_physical_type = SedonaType::from_storage_field(stream.schema().field(1))?;
        assert_eq!(geometry_physical_type, WKB_GEOMETRY);

        let batches_out: Vec<_> = stream.try_collect().await?;
        assert_eq!(batches_out.len(), 2);
        assert_eq!(batches_out[0], batch_in);
        assert_eq!(batches_out[1], batch_in);

        Ok(())
    }

    #[tokio::test]
    async fn geometry_columns() {
        let ctx = SedonaContext::new();

        // No geometry column
        let df = ctx.sql("SELECT 1 as one").await.unwrap();
        assert!(df.geometry_column_indices().unwrap().is_empty());
        assert!(df.primary_geometry_column_index().unwrap().is_none());

        // Should list geometry and geography but pick geom as the primary column
        let df = ctx
            .sql("SELECT ST_GeogPoint(0, 1) as geog, ST_Point(0, 1) as geom")
            .await
            .unwrap();
        assert_eq!(df.geometry_column_indices().unwrap(), vec![0, 1]);
        assert_eq!(df.primary_geometry_column_index().unwrap(), Some(1));

        // ...but should still detect a column without a special name
        let df = ctx
            .sql("SELECT ST_Point(0, 1) as name_not_special_cased")
            .await
            .unwrap();
        assert_eq!(df.geometry_column_indices().unwrap(), vec![0]);
        assert_eq!(df.primary_geometry_column_index().unwrap(), Some(0));
    }

    #[tokio::test]
    async fn show() {
        let ctx = SedonaContext::new();
        let tbl = ctx
            .sql("SELECT 1 as one")
            .await
            .unwrap()
            .show_sedona(&ctx, None, DisplayTableOptions::default())
            .await
            .unwrap();

        #[rustfmt::skip]
        assert_eq!(
            tbl.lines().collect::<Vec<_>>(),
            vec![
                "+-----+",
                "| one |",
                "+-----+",
                "|   1 |",
                "+-----+"
            ]
        );
    }

    #[tokio::test]
    async fn geoparquet_format() {
        // Make sure that our context can be set up to identify and read
        // GeoParquet files
        let ctx = SedonaContext::new_local_interactive().await.unwrap();
        let example = test_geoparquet("example", "geometry").unwrap();
        let df = ctx.ctx.table(example).await.unwrap();
        let sedona_types: Result<Vec<_>> = df
            .schema()
            .as_arrow()
            .fields()
            .iter()
            .map(|f| SedonaType::from_data_type(f.data_type()))
            .collect();
        let sedona_types = sedona_types.unwrap();
        assert_eq!(sedona_types.len(), 2);
        assert_eq!(sedona_types[0], SedonaType::Arrow(DataType::Utf8View));
        assert_eq!(
            sedona_types[1],
            SedonaType::WkbView(Edges::Planar, lnglat())
        );
    }
}
