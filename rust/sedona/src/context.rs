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
use std::{collections::VecDeque, sync::Arc};

use crate::exec::create_plan_from_sql;
use crate::{
    catalog::DynamicObjectStoreCatalog,
    random_geometry_provider::RandomGeometryFunction,
    show::{show_batches, DisplayTableOptions},
};
use arrow_array::RecordBatch;
use async_trait::async_trait;
use datafusion::dataframe::DataFrameWriteOptions;
use datafusion::datasource::file_format::format_as_file_type;
use datafusion::{
    common::{plan_datafusion_err, plan_err},
    error::{DataFusionError, Result},
    execution::{context::DataFilePaths, runtime_env::RuntimeEnvBuilder, SessionStateBuilder},
    prelude::{DataFrame, SessionConfig, SessionContext},
    sql::parser::{DFParser, Statement},
};
use datafusion_common::not_impl_err;
use datafusion_expr::dml::InsertOp;
use datafusion_expr::sqlparser::dialect::{dialect_from_str, Dialect};
use datafusion_expr::{LogicalPlanBuilder, SortExpr};
use parking_lot::Mutex;
use sedona_common::option::add_sedona_option_extension;
use sedona_expr::aggregate_udf::SedonaAccumulatorRef;
use sedona_expr::{function_set::FunctionSet, scalar_udf::ScalarKernelRef};
use sedona_geoparquet::options::TableGeoParquetOptions;
use sedona_geoparquet::{
    format::GeoParquetFormatFactory,
    provider::{geoparquet_listing_table, GeoParquetReadOptions},
};

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

        #[cfg(feature = "tg")]
        out.register_scalar_kernels(sedona_tg::register::scalar_kernels().into_iter())?;

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

        // Pre-register object store with our custom options before creating GeoParquetReadOptions
        if !urls.is_empty() {
            use crate::object_storage::ensure_object_store_registered_with_options;
            // Extract the table options from GeoParquetReadOptions for object store registration
            let table_options_map = options.table_options().cloned().unwrap_or_default();

            // TODO: Consider registering object stores per-bucket instead of per-scheme to avoid
            // authentication conflicts. Currently, if a user first accesses a public S3 bucket with
            // aws.skip_signature=true and then tries to access a private bucket, the cached object
            // store will still have skip_signature enabled, preventing authentication to the private
            // bucket. A per-bucket registration approach would solve this by using bucket-specific
            // cache keys like "s3://bucket-name" instead of just "s3://".
            ensure_object_store_registered_with_options(
                &mut self.ctx.state(),
                urls[0].as_str(),
                Some(&table_options_map),
            )
            .await?;
        }

        let provider = geoparquet_listing_table(&self.ctx, urls, options).await?;

        self.ctx.read_table(Arc::new(provider))
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
    /// Build a table of the first `limit` results in this DataFrame
    ///
    /// This will limit and execute the query and build a table using [show_batches].
    async fn show_sedona<'a>(
        self,
        ctx: &SedonaContext,
        limit: Option<usize>,
        options: DisplayTableOptions<'a>,
    ) -> Result<String>;

    async fn write_geoparquet(
        self,
        ctx: &SedonaContext,
        path: &str,
        options: SedonaWriteOptions,
        writer_options: Option<TableGeoParquetOptions>,
    ) -> Result<Vec<RecordBatch>>;
}

#[async_trait]
impl SedonaDataFrame for DataFrame {
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

    async fn write_geoparquet(
        self,
        ctx: &SedonaContext,
        path: &str,
        options: SedonaWriteOptions,
        writer_options: Option<TableGeoParquetOptions>,
    ) -> Result<Vec<RecordBatch>, DataFusionError> {
        if options.insert_op != InsertOp::Append {
            return not_impl_err!(
                "{} is not implemented for DataFrame::write_geoparquet.",
                options.insert_op
            );
        }

        let format = if let Some(parquet_opts) = writer_options {
            Arc::new(GeoParquetFormatFactory::new_with_options(parquet_opts))
        } else {
            Arc::new(GeoParquetFormatFactory::new())
        };

        let file_type = format_as_file_type(format);

        let plan = if options.sort_by.is_empty() {
            self.into_unoptimized_plan()
        } else {
            LogicalPlanBuilder::from(self.into_unoptimized_plan())
                .sort(options.sort_by)?
                .build()?
        };

        let plan = LogicalPlanBuilder::copy_to(
            plan,
            path.into(),
            file_type,
            Default::default(),
            options.partition_by,
        )?
        .build()?;

        DataFrame::new(ctx.ctx.state(), plan).collect().await
    }
}

/// A Sedona-specific copy of [DataFrameWriteOptions]
///
/// This is needed because [DataFrameWriteOptions] has private fields, so we
/// can't use it in our interfaces. This object can be converted to a
/// [DataFrameWriteOptions] using `.into()`.
pub struct SedonaWriteOptions {
    /// Controls how new data should be written to the table, determining whether
    /// to append, overwrite, or replace existing data.
    pub insert_op: InsertOp,
    /// Controls if all partitions should be coalesced into a single output file
    /// Generally will have slower performance when set to true.
    pub single_file_output: bool,
    /// Sets which columns should be used for hive-style partitioned writes by name.
    /// Can be set to empty vec![] for non-partitioned writes.
    pub partition_by: Vec<String>,
    /// Sets which columns should be used for sorting the output by name.
    /// Can be set to empty vec![] for non-sorted writes.
    pub sort_by: Vec<SortExpr>,
}

impl From<SedonaWriteOptions> for DataFrameWriteOptions {
    fn from(value: SedonaWriteOptions) -> Self {
        DataFrameWriteOptions::new()
            .with_insert_operation(value.insert_op)
            .with_single_file_output(value.single_file_output)
            .with_partition_by(value.partition_by)
            .with_sort_by(value.sort_by)
    }
}

impl SedonaWriteOptions {
    /// Create a new SedonaWriteOptions with default values
    pub fn new() -> Self {
        SedonaWriteOptions {
            insert_op: InsertOp::Append,
            single_file_output: false,
            partition_by: vec![],
            sort_by: vec![],
        }
    }

    /// Set the insert operation
    pub fn with_insert_operation(mut self, insert_op: InsertOp) -> Self {
        self.insert_op = insert_op;
        self
    }

    /// Set the single_file_output value to true or false
    pub fn with_single_file_output(mut self, single_file_output: bool) -> Self {
        self.single_file_output = single_file_output;
        self
    }

    /// Sets the partition_by columns for output partitioning
    pub fn with_partition_by(mut self, partition_by: Vec<String>) -> Self {
        self.partition_by = partition_by;
        self
    }

    /// Sets the sort_by columns for output sorting
    pub fn with_sort_by(mut self, sort_by: Vec<SortExpr>) -> Self {
        self.sort_by = sort_by;
        self
    }
}

impl Default for SedonaWriteOptions {
    fn default() -> Self {
        Self::new()
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

    use arrow_schema::DataType;
    use datafusion::assert_batches_eq;
    use sedona_schema::{
        crs::lnglat,
        datatypes::{Edges, SedonaType},
    };
    use sedona_testing::data::test_geoparquet;
    use tempfile::tempdir;

    use super::*;

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
    async fn write_geoparquet() {
        let tmpdir = tempdir().unwrap();
        let tmp_parquet = tmpdir.path().join("tmp.parquet");
        let ctx = SedonaContext::new();
        ctx.sql("SELECT 1 as one")
            .await
            .unwrap()
            .write_parquet(
                &tmp_parquet.to_string_lossy(),
                DataFrameWriteOptions::default(),
                None,
            )
            .await
            .unwrap();
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
            .map(|f| SedonaType::from_storage_field(f))
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
