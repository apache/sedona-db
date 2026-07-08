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
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

use crate::exec::create_plan_from_sql;
use crate::object_storage::ensure_object_store_registered_with_options;
use crate::{
    catalog::DynamicObjectStoreCatalog,
    random_geometry_provider::RandomGeometryFunction,
    show::{show_batches, DisplayTableOptions},
};
use arrow_array::RecordBatch;
use arrow_schema::DataType;
use async_trait::async_trait;
use datafusion::datasource::file_format::format_as_file_type;
use datafusion::{
    common::plan_err,
    error::{DataFusionError, Result},
    execution::{
        context::DataFilePaths,
        runtime_env::{RuntimeEnv, RuntimeEnvBuilder},
        SessionStateBuilder,
    },
    prelude::{DataFrame, SessionConfig, SessionContext},
    sql::parser::{DFParser, Statement},
};
use datafusion::{dataframe::DataFrameWriteOptions, execution::memory_pool::MemoryLimit};
use datafusion_common::not_impl_err;
use datafusion_expr::dml::InsertOp;
use datafusion_expr::sqlparser::dialect::{dialect_from_str, Dialect};
use datafusion_expr::{AggregateUDFImpl, LogicalPlan, LogicalPlanBuilder, ScalarUDFImpl, SortExpr};
use parking_lot::Mutex;
use sedona_common::{sedona_internal_datafusion_err, SedonaOptions};
use sedona_datasource::provider::external_table;
use sedona_datasource::spec::ExternalFormatSpec;
use sedona_expr::scalar_udf::{IntoScalarKernelRefs, SedonaScalarUDF};
use sedona_expr::{
    aggregate_udf::{IntoSedonaAccumulatorRefs, SedonaAggregateUDF},
    function_set::FunctionSet,
};
use sedona_geoparquet::options::TableGeoParquetOptions;
use sedona_geoparquet::{
    format::GeoParquetFormatFactory,
    provider::{geoparquet_listing_table, GeoParquetReadOptions},
};
#[cfg(feature = "pointcloud")]
use sedona_pointcloud::las::{
    format::{Extension, LasFormatFactory},
    options::{GeometryEncoding, LasExtraBytes, LasOptions},
};
use sedona_raster_functions::rs_ensure_loaded::RsEnsureLoaded;
#[cfg(feature = "gpu")]
use sedona_spatial_join_gpu::options::GpuOptions;

use sedona_query_planner::{
    optimizer::{register_ensure_loaded_optimizer, register_spatial_join_logical_optimizer},
    query_planner::SedonaQueryPlanner,
};
use sedona_raster::raster_loader::{AsyncRasterLoader, RasterLoaderConfig, RasterLoaderRegistry};

/// Sedona SessionContext wrapper
///
/// As Sedona extends DataFusion, we also extend its context and include the
/// default geometry-specific functions and datasources (which may vary depending
/// on the feature flags used to build the sedona crate). This provides a common
/// interface for configuring the behaviour of
pub struct SedonaContext {
    pub ctx: SessionContext,
    functions: RwLock<FunctionSet>,
    /// Per-session registry of async raster byte loaders, keyed by
    /// `outdb_format`. Held behind an `Arc<RwLock<…>>` so the registered
    /// `RS_EnsureLoaded` UDF instance and any extension crates' `register(&ctx)`
    /// entry points observe the same map. See
    /// [`SedonaContext::register_raster_loader`].
    raster_loader_registry: Arc<RwLock<RasterLoaderRegistry>>,
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
        let rt_builder = RuntimeEnvBuilder::new();
        let runtime_env = rt_builder.build_arc()?;
        Self::new_local_interactive_with_runtime_env(runtime_env).await
    }

    pub async fn new_local_interactive_with_runtime_env(
        runtime_env: Arc<RuntimeEnv>,
    ) -> Result<Self> {
        // These three objects enable configuring various elements of the runtime.
        // Eventually we probably want to have a common set of configuration parameters
        // exposed via the CLI/Python as arguments, via ADBC as connection options,
        // and perhaps for all of these initializing them optionally from environment
        // variables.
        let session_config = SessionConfig::from_env()?.with_information_schema(true);
        let mut session_config = session_config.with_option_extension(SedonaOptions::default());
        let target_partitions = session_config.target_partitions();

        // Always register the PROJ LazyProjEngine by default (if PROJ is not configured
        // before it is used an error will be raised).
        let opts = session_config
            .options_mut()
            .extensions
            .get_mut::<SedonaOptions>()
            .ok_or_else(|| sedona_internal_datafusion_err!("SedonaOptions not available"))?;
        opts.runtime = opts
            .runtime
            .with_crs_engine(Arc::new(sedona_proj::transform::LazyProjEngine));

        // Set the spilled batch in-memory size threshold to 5% of the per-partition memory limit,
        // with a minimum of 10MB. Batches larger than this threshold will be broken into smaller batches
        // before writing to spill files. This is to avoid overshooting memory limit when reading super
        // large spilled batches.
        const SPILLED_BATCH_THRESHOLD_PERCENT_DIVISOR: usize = 20; // 5% == 1 / 20
        const MIN_SPILLED_BATCH_IN_MEMORY_THRESHOLD_BYTES: usize = 10 * 1024 * 1024; // 10MB
        if let MemoryLimit::Finite(memory_limit) = runtime_env.memory_pool.memory_limit() {
            let per_partition_memory_limit = memory_limit.div_ceil(target_partitions);
            opts.spatial_join.spilled_batch_in_memory_size_threshold = per_partition_memory_limit
                .div_ceil(SPILLED_BATCH_THRESHOLD_PERCENT_DIVISOR)
                .max(MIN_SPILLED_BATCH_IN_MEMORY_THRESHOLD_BYTES);
        }

        // Register the spatial join planner extension
        #[allow(unused_mut)]
        let mut planner = SedonaQueryPlanner::new();
        #[cfg(feature = "spatial-join")]
        {
            use sedona_spatial_join::physical_planner::DefaultSpatialJoinPhysicalPlanner;

            planner = planner.with_spatial_join_physical_planner(Arc::new(
                DefaultSpatialJoinPhysicalPlanner::new(),
            ));

            // Register the geography join after the default planner
            // If a query is not supported, it falls back to the default planner.
            #[cfg(feature = "s2geography")]
            {
                use sedona_spatial_join_geography::physical_planner::GeographySpatialJoinPhysicalPlanner;

                planner = planner.with_spatial_join_physical_planner(Arc::new(
                    GeographySpatialJoinPhysicalPlanner::new(),
                ));
            }

            // Register the GPU join after the default planner
            // If a query is not supported, it falls back to the default planner.
            #[cfg(feature = "gpu")]
            {
                use sedona_spatial_join_gpu::physical_planner::GpuSpatialJoinPhysicalPlanner;

                planner = planner.with_spatial_join_physical_planner(Arc::new(
                    GpuSpatialJoinPhysicalPlanner::new(),
                ));
            }
        }

        // Register the geography bounder for spherical edge types. This enables geography
        // statistics in Parquet files and geography row-group pruning.
        #[cfg(feature = "s2geography")]
        {
            opts.runtime = opts.runtime.with_bounder(
                sedona_geometry::types::Edges::Spherical,
                Arc::new(sedona_s2geography::rect_bounder::WkbGeographyBounder::default()),
            )?;
        }

        // Inject the statically-set accumulator factory, which allows arrow-rs Parquet writer
        // to write Geography statistics based on the geography bounders in our SedonaRuntime.
        let init_result =
            sedona_geoparquet::statistics_accumulator::SedonaGeoStatsAccumulatorFactory::try_init(
                &opts.runtime,
            );
        if let Err(init_err) = init_result {
            if !matches!(init_err, DataFusionError::ParquetError(_)) {
                return Err(init_err);
            }
        }

        #[cfg(feature = "pointcloud")]
        let session_config = session_config.with_option_extension(
            LasOptions::default()
                .with_geometry_encoding(GeometryEncoding::Wkb)
                .with_las_extra_bytes(LasExtraBytes::Typed),
        );

        #[cfg(feature = "gpu")]
        let session_config = session_config.with_option_extension(GpuOptions::default());

        let mut state_builder = SessionStateBuilder::new()
            .with_default_features()
            .with_runtime_env(runtime_env)
            .with_config(session_config);

        state_builder = register_spatial_join_logical_optimizer(state_builder)?;
        state_builder = register_ensure_loaded_optimizer(state_builder)?;
        state_builder = state_builder.with_query_planner(Arc::new(planner));

        let mut state = state_builder.build();

        // Register GeoParquet file format
        state.register_file_format(Arc::new(GeoParquetFormatFactory::new()), true)?;

        #[cfg(feature = "pointcloud")]
        {
            state.register_file_format(Arc::new(LasFormatFactory::new(Extension::Laz)), false)?;
            state.register_file_format(Arc::new(LasFormatFactory::new(Extension::Las)), false)?;
        }

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
            functions: RwLock::new(FunctionSet::new()),
            raster_loader_registry: Arc::new(RwLock::new(RasterLoaderRegistry::new())),
        };

        // Stash a clone of the shared registry handle inside
        // `ConfigOptions` via the `RasterLoaderConfig` extension. The
        // RS_EnsureLoaded async UDF reads from there at dispatch time —
        // `AsyncScalarUDFImpl::invoke_async_with_args` only receives
        // `Arc<ConfigOptions>`, so this is the path that keeps the
        // registry reachable at the UDF's invocation site. Mirrors how
        // `SedonaRuntime` works inside `SedonaOptions`.
        //
        // Writes through `SedonaContext::register_raster_loader` (which
        // mutates the Arc held in `out.raster_loader_registry`) are immediately
        // visible to UDF reads through this config extension because
        // both handles share the same `RwLock`.
        out.ctx
            .state_ref()
            .write()
            .config_mut()
            .options_mut()
            .extensions
            .insert(RasterLoaderConfig::from_handle(Arc::clone(
                &out.raster_loader_registry,
            )));

        // Register the RS_EnsureLoaded async UDF. It pulls the registry
        // out of `args.config_options` at dispatch time, so it doesn't
        // need to close over the Arc itself — the UDF instance is
        // session-agnostic. The logical optimizer rule registered above
        // (`register_ensure_loaded_optimizer`) resolves this UDF from the
        // function registry at rewrite time, so it must be registered
        // before any query is planned.
        {
            use datafusion_expr::async_udf::AsyncScalarUDF;
            let udf = AsyncScalarUDF::new(Arc::new(RsEnsureLoaded::new()));
            out.ctx.register_udf(udf.into_scalar_udf());
        };

        // Register the GDAL raster byte loader. `sedona-raster-gdal` is a
        // mandatory dep on `sedona`, but libgdal itself is dlopen'd
        // lazily by `sedona-gdal` (workspace-default-features = false),
        // so this registration is safe on systems without libgdal —
        // the loader's `load()` call will surface a clean "libgdal not
        // found" error when first invoked, but registration and import
        // succeed regardless.
        out.register_raster_loader(Arc::new(sedona_raster_gdal::GdalLoader::new()));

        // Register table functions
        out.ctx.register_udtf(
            "sd_random_geometry",
            Arc::new(RandomGeometryFunction::default()),
        );

        out.register_function_set(sedona_raster_gdal::register::default_function_set());

        // Always register default function set
        out.register_function_set(sedona_functions::register::default_function_set());

        // Register geos scalar kernels if built with geos support
        #[cfg(feature = "geos")]
        out.register_scalar_kernels(sedona_geos::register::scalar_kernels().into_iter())?;

        // Register geos aggregate kernels if built with geos support
        #[cfg(feature = "geos")]
        out.register_aggregate_kernels(sedona_geos::register::aggregate_kernels().into_iter())?;

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
        out.register_s2geography()?;

        // Always register proj scalar kernels (although actually calling them will error
        // without this feature unless sedona_proj::register::configure_global_proj_engine()
        // is called).
        out.register_scalar_kernels(sedona_proj::register::scalar_kernels().into_iter())?;

        // Always register raster functions
        out.register_function_set(sedona_raster_functions::register::default_function_set());

        Ok(out)
    }

    #[cfg(feature = "s2geography")]
    fn register_s2geography(&mut self) -> Result<()> {
        use sedona_proj::sd_order_lnglat;

        self.register_scalar_kernels(sedona_s2geography::register::scalar_kernels()?.into_iter())?;

        let sd_order_kernel =
            sd_order_lnglat::OrderLngLat::new(sedona_s2geography::utils::s2_cell_id_from_lnglat);
        self.register_scalar_kernels([("sd_order", sd_order_kernel)].into_iter())?;

        self.register_aggregate_kernels(
            sedona_s2geography::register::aggregate_kernels().into_iter(),
        )?;

        Ok(())
    }

    /// Register an async raster byte loader under a `format` key.
    ///
    /// Each loader declares the band `outdb_format` values it handles via
    /// `AsyncRasterLoader::supports_format`. At query time the
    /// `RS_EnsureLoaded` UDF matches each OutDb band's format against the
    /// registered loaders (most-recently-registered first) and dispatches
    /// the byte fetch through the first that accepts it.
    ///
    /// Used by both compiled-in backends (`sedona-raster-gdal`, the catch-all,
    /// registered first at bootstrap) and out-of-tree extensions
    /// (`sedona-raster-zarr::register(&ctx)` from user code after
    /// construction). Loaders registered later win for the formats they
    /// claim, so registration order matters: the catch-all goes first.
    pub fn register_raster_loader(&self, loader: Arc<dyn AsyncRasterLoader>) {
        // Lock poisoning here would mean a previous registrant panicked
        // mid-write — recover-by-ignoring matches how DataFusion handles
        // session-state writes elsewhere.
        if let Ok(mut guard) = self.raster_loader_registry.write() {
            guard.register(loader);
        }
    }

    fn functions(&self) -> Result<RwLockReadGuard<'_, FunctionSet>> {
        self.functions
            .read()
            .map_err(|_| sedona_internal_datafusion_err!("Function registry lock poisoned"))
    }

    fn functions_mut(&self) -> Result<RwLockWriteGuard<'_, FunctionSet>> {
        self.functions
            .write()
            .map_err(|_| sedona_internal_datafusion_err!("Function registry lock poisoned"))
    }

    pub fn scalar_udf(&self, name: &str) -> Result<Option<SedonaScalarUDF>> {
        Ok(self.functions()?.scalar_udf(name).cloned())
    }

    pub fn scalar_udf_names(&self) -> Result<Vec<String>> {
        Ok(self
            .ctx
            .state()
            .scalar_functions()
            .keys()
            .cloned()
            .collect())
    }

    pub fn register_sedona_scalar_udf(&self, udf: SedonaScalarUDF) -> Result<()> {
        let name = udf.name().to_string();
        let mut functions = self.functions_mut()?;
        functions.insert_scalar_udf(udf);
        let udf = functions.scalar_udf(&name).unwrap().clone();
        drop(functions);

        self.ctx.register_udf(udf.into());
        Ok(())
    }

    pub fn register_sedona_aggregate_udf(&self, udf: SedonaAggregateUDF) -> Result<()> {
        let name = udf.name().to_string();
        let mut functions = self.functions_mut()?;
        functions.insert_aggregate_udf(udf);
        let udf = functions.aggregate_udf(&name).unwrap().clone();
        drop(functions);

        self.ctx.register_udaf(udf.into());
        Ok(())
    }

    /// Register all functions in a [FunctionSet] with this context
    pub fn register_function_set(&mut self, function_set: FunctionSet) {
        let mut functions = self
            .functions
            .write()
            .expect("fresh SedonaContext function registry lock should not be poisoned");
        for udf in function_set.scalar_udfs() {
            functions.insert_scalar_udf(udf.clone());
            self.ctx.register_udf(udf.clone().into());
        }

        for udf in function_set.aggregate_udfs() {
            functions.insert_aggregate_udf(udf.clone());
            self.ctx.register_udaf(udf.clone().into());
        }
    }

    /// Register a collection of kernels with this context
    pub fn register_scalar_kernels<'a>(
        &mut self,
        kernels: impl Iterator<Item = (&'a str, impl IntoScalarKernelRefs)>,
    ) -> Result<()> {
        let mut functions = self.functions_mut()?;
        for (name, kernel) in kernels {
            let udf = functions.add_scalar_udf_impl(name, kernel)?;
            self.ctx.register_udf(udf.clone().into());
        }

        Ok(())
    }

    pub fn register_aggregate_kernels<'a>(
        &mut self,
        kernels: impl Iterator<Item = (&'a str, impl IntoSedonaAccumulatorRefs)>,
    ) -> Result<()> {
        let mut functions = self.functions_mut()?;
        for (name, kernel) in kernels {
            let udf = functions.add_aggregate_udf_kernel(name, kernel)?;
            self.ctx.register_udaf(udf.clone().into());
        }

        Ok(())
    }

    /// Creates a [`DataFrame`] from SQL query text that may contain one or more
    /// statements
    pub async fn multi_sql(&self, sql: &str) -> Result<Vec<DataFrame>> {
        let task_ctx = self.ctx.task_ctx();
        let dialect = &task_ctx.session_config().options().sql_parser.dialect;
        let dialect = ThreadSafeDialect::try_new(dialect)?;

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

    /// Creates a [`DataFrame`] for reading a [ExternalFormatSpec]
    ///
    /// The `partitioning` parameter controls hive-style partition discovery:
    /// - `None`: auto-discover partition columns from directory structure
    /// - `Some(vec![])`: disable partition discovery
    /// - `Some(vec![...])`: use explicit partition columns
    pub async fn read_external_format<P: DataFilePaths>(
        &self,
        spec: Arc<dyn ExternalFormatSpec>,
        table_paths: P,
        options: Option<&HashMap<String, String>>,
        check_extension: bool,
        partitioning: Option<Vec<(String, DataType)>>,
    ) -> Result<DataFrame> {
        let urls = table_paths.to_urls()?;

        // Pre-register object store with our custom options before creating GeoParquetReadOptions
        if !urls.is_empty() {
            // Extract the table options from GeoParquetReadOptions for object store registration
            ensure_object_store_registered_with_options(
                &mut self.ctx.state(),
                urls[0].as_str(),
                options,
            )
            .await?;
        }

        let provider = if let Some(options) = options {
            // Strip the filesystem-based options
            let options_without_filesystems = options
                .iter()
                .filter(|(k, _)| !k.starts_with("gcs.") && !k.starts_with("aws."))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect::<HashMap<String, String>>();
            let spec = spec.with_options(&options_without_filesystems)?;
            external_table(spec, &self.ctx, urls, check_extension, partitioning).await?
        } else {
            external_table(spec, &self.ctx, urls, check_extension, partitioning).await?
        };

        self.ctx.read_table(provider)
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
        mut options: DisplayTableOptions<'a>,
    ) -> Result<String> {
        let df = if matches!(
            self.logical_plan(),
            LogicalPlan::Explain(_) | LogicalPlan::DescribeTable(_) | LogicalPlan::Analyze(_)
        ) {
            // Show multi-line output without truncation for plans like `EXPLAIN`
            options.max_row_height = usize::MAX;

            // We don't want to apply an additional .limit() to plans like `Explain`
            // as that will trigger an internal error: Unsupported logical plan: Explain must be root of the plan
            self
        } else {
            // Apply limit if specified
            self.limit(0, limit)?
        };

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
    pub fn try_new(dialect: &datafusion::config::Dialect) -> Result<Self> {
        if let Some(box_dialect) = dialect_from_str(dialect) {
            Ok(Self {
                inner: box_dialect.into(),
            })
        } else {
            plan_err!("Unsupported SQL dialect: {dialect}")
        }
    }

    pub fn parse(&self, sql: &str) -> Result<VecDeque<Statement>> {
        let dialect = self.inner.lock();
        DFParser::parse_sql_with_dialect(sql, dialect.as_ref())
    }
}

#[cfg(test)]
mod tests {

    use arrow_array::{create_array, ArrayRef, RecordBatchIterator, RecordBatchReader};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::assert_batches_eq;
    use sedona_datasource::spec::{Object, OpenReaderArgs};
    use sedona_geometry::types::Edges;
    use sedona_schema::{
        crs::{deserialize_crs, lnglat},
        datatypes::SedonaType,
        schema::SedonaSchema,
    };
    use sedona_testing::data::test_geoparquet;
    use tempfile::tempdir;

    use super::*;

    #[tokio::test]
    async fn outdb_registry_has_gdal_at_bootstrap_and_accepts_runtime_registration() {
        use arrow_buffer::Buffer;
        use sedona_raster::raster_loader::{
            AsyncRasterLoader, RasterLoadRequest, RasterLoadResult,
        };

        let ctx = SedonaContext::new();
        // GDAL is the catch-all backend, registered at bootstrap: it resolves
        // any format, including the unset `None` that RS_FromPath emits.
        // Extension backends like Zarr add themselves later via
        // `register_raster_loader`.
        {
            let reg = ctx.raster_loader_registry.read().unwrap();
            let loader = reg
                .get(None)
                .expect("fresh SedonaContext should register the GDAL backend at bootstrap");
            assert_eq!(loader.name(), "gdal");
        }

        // Mock runtime registration on top of the bootstrap state.
        #[derive(Debug)]
        struct MockLoader;
        #[async_trait]
        impl AsyncRasterLoader for MockLoader {
            fn name(&self) -> &str {
                "mock"
            }
            fn supports_format(&self, format: Option<&str>) -> bool {
                format == Some("mock")
            }
            async fn load(
                &self,
                reqs: &[&RasterLoadRequest],
            ) -> std::result::Result<Vec<RasterLoadResult>, arrow_schema::ArrowError> {
                Ok(reqs
                    .iter()
                    .map(|req| {
                        RasterLoadResult::unresolved(Buffer::from_vec(Vec::<u8>::new()), req)
                    })
                    .collect())
            }
        }
        ctx.register_raster_loader(Arc::new(MockLoader));
        {
            let reg = ctx.raster_loader_registry.read().unwrap();
            // The mock wins for its own format (most-recently-registered)...
            assert_eq!(reg.get(Some("mock")).unwrap().name(), "mock");
            // ...while everything else, including the unset `None`, still
            // falls through to the GDAL catch-all.
            assert_eq!(reg.get(None).unwrap().name(), "gdal");
        }

        // RS_EnsureLoaded is registered as a UDF at session bootstrap.
        let udf = ctx
            .ctx
            .state()
            .scalar_functions()
            .get("rs_ensureloaded")
            .cloned();
        assert!(
            udf.is_some(),
            "RS_EnsureLoaded should be registered by SedonaContext::new()"
        );
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
    async fn nested_expressions_sql() -> Result<()> {
        let ctx = SedonaContext::new();

        // Test get_field on a struct
        let batches = ctx
            .sql("SELECT get_field(named_struct('a', 1, 'b', 2), 'a') AS a")
            .await?
            .collect()
            .await?;
        #[rustfmt::skip]
        assert_batches_eq!(
            [
                "+---+",
                "| a |",
                "+---+",
                "| 1 |",
                "+---+",
            ],
            &batches
        );

        // Test map_extract on a map
        let batches = ctx
            .sql("SELECT map_extract(map {'x': 10, 'y': 20}, 'x') AS x")
            .await?
            .collect()
            .await?;
        #[rustfmt::skip]
        assert_batches_eq!(
            [
                "+------+",
                "| x    |",
                "+------+",
                "| [10] |",
                "+------+",
            ],
            &batches
        );

        // Test map bracket notation shorthand (returns scalar, not array)
        let batches = ctx
            .sql("SELECT map {'x': 10, 'y': 20}['x'] AS x")
            .await?
            .collect()
            .await?;
        #[rustfmt::skip]
        assert_batches_eq!(
            [
                "+----+",
                "| x  |",
                "+----+",
                "| 10 |",
                "+----+",
            ],
            &batches
        );

        // Test struct dot notation and map bracket notation on table columns
        ctx.sql(
            "CREATE TABLE nested_test AS SELECT \
             named_struct('a', 1, 'b', 2) AS s, \
             map {'x': 10, 'y': 20} AS m",
        )
        .await?
        .collect()
        .await?;

        // Struct field access via dot notation on column
        let batches = ctx
            .sql("SELECT s.a FROM nested_test")
            .await?
            .collect()
            .await?;
        #[rustfmt::skip]
        assert_batches_eq!(
            [
                "+------------------+",
                "| nested_test.s[a] |",
                "+------------------+",
                "| 1                |",
                "+------------------+",
            ],
            &batches
        );

        // Map key access via bracket notation on column
        let batches = ctx
            .sql("SELECT m['x'] FROM nested_test")
            .await?
            .collect()
            .await?;
        #[rustfmt::skip]
        assert_batches_eq!(
            [
                "+------------------+",
                "| nested_test.m[x] |",
                "+------------------+",
                "| 10               |",
                "+------------------+",
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
    async fn show_explain() {
        let ctx = SedonaContext::new();
        for limit in [None, Some(10)] {
            let tbl = ctx
                .sql("EXPLAIN SELECT 1 as one")
                .await
                .unwrap()
                .show_sedona(&ctx, limit, DisplayTableOptions::default())
                .await
                .unwrap();

            #[rustfmt::skip]
            assert_eq!(
                tbl.lines().collect::<Vec<_>>(),
                vec![
                    "+---------------+---------------------------------+",
                    "|   plan_type   |               plan              |",
                    "+---------------+---------------------------------+",
                    "| logical_plan  | Projection: Int64(1) AS one     |",
                    "|               |   EmptyRelation: rows=1         |",
                    "| physical_plan | ProjectionExec: expr=[1 as one] |",
                    "|               |   PlaceholderRowExec            |",
                    "|               |                                 |",
                    "+---------------+---------------------------------+",
                ]
            );
        }
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
        let df = ctx.ctx.table(example.clone()).await.unwrap();
        let sedona_types = df
            .schema()
            .sedona_types()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(sedona_types.len(), 2);
        assert_eq!(sedona_types[0], SedonaType::Arrow(DataType::Utf8View));
        assert_eq!(
            sedona_types[1],
            SedonaType::WkbView(Edges::Planar, lnglat())
        );

        // Ensure read_parquet() works
        let df = ctx
            .read_parquet(example.clone(), GeoParquetReadOptions::default())
            .await
            .unwrap();
        let sedona_types = df
            .schema()
            .sedona_types()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(sedona_types.len(), 2);
        assert_eq!(sedona_types[0], SedonaType::Arrow(DataType::Utf8View));
        assert_eq!(
            sedona_types[1],
            SedonaType::WkbView(Edges::Planar, lnglat())
        );
    }

    #[derive(Debug)]
    struct ExampleSpec {}

    #[async_trait]
    impl ExternalFormatSpec for ExampleSpec {
        async fn infer_schema(&self, _location: &Object) -> Result<Schema> {
            Ok(Schema::new(vec![Field::new("x", DataType::Utf8, true)]))
        }

        async fn open_reader(
            &self,
            _args: &OpenReaderArgs,
        ) -> Result<Box<dyn RecordBatchReader + Send>> {
            let batch = RecordBatch::try_from_iter([(
                "x",
                create_array!(Utf8, ["one", "two", "three", "four"]) as ArrayRef,
            )])
            .unwrap();
            let schema = batch.schema();
            Ok(Box::new(RecordBatchIterator::new([Ok(batch)], schema)))
        }

        fn with_options(
            &self,
            options: &HashMap<String, String>,
        ) -> Result<Arc<dyn ExternalFormatSpec>> {
            // Ensure we fail if we see any key/value options to ensure aws/gcs options
            // are stripped.
            if !options.is_empty() {
                return not_impl_err!("key/value options not implemented");
            }

            Ok(Arc::new(Self {}))
        }
    }

    #[tokio::test]
    async fn external_format() {
        let ctx = SedonaContext::new_local_interactive().await.unwrap();
        let spec = Arc::new(ExampleSpec {});
        let file_that_exists = test_geoparquet("example", "geometry").unwrap();

        // Ensure read_external_format() works
        let df = ctx
            .read_external_format(spec.clone(), file_that_exists.clone(), None, false, None)
            .await
            .unwrap();
        let batches = df.collect().await.unwrap();

        assert_batches_eq!(
            [
                "+-------+",
                "| x     |",
                "+-------+",
                "| one   |",
                "| two   |",
                "| three |",
                "| four  |",
                "+-------+",
            ],
            &batches
        );

        // Ensure that key/value options used by aws/gcs are stripped
        let kv_options = HashMap::from([("key".to_string(), "value".to_string())]);
        ctx.read_external_format(
            spec.clone(),
            file_that_exists.clone(),
            Some(&kv_options),
            false,
            None,
        )
        .await
        .expect_err("should error for unsupported key/value options");

        let kv_options = HashMap::from([
            ("gcs.something".to_string(), "value".to_string()),
            ("aws.something".to_string(), "value".to_string()),
        ]);
        ctx.read_external_format(
            spec.clone(),
            file_that_exists.clone(),
            Some(&kv_options),
            false,
            None,
        )
        .await
        .expect("should succeed because aws and gcs options were stripped");
    }

    #[tokio::test]
    async fn format_from_external_table() {
        let ctx = SedonaContext::new_local_interactive().await.unwrap();
        let example = test_geoparquet("example", "geometry").unwrap();
        ctx.sql(&format!(
            r#"
            CREATE EXTERNAL TABLE test
            STORED AS PARQUET
            LOCATION '{example}'
            OPTIONS ('geometry_columns' '{{"geometry": {{"encoding": "WKB", "crs": "EPSG:3857"}}}}');
            "#
        ))
        .await
        .unwrap();

        let df = ctx.ctx.table("test").await.unwrap();

        // Check that the logical plan resulting from a read has the correct schema
        assert_eq!(
            df.schema().clone().strip_qualifiers().field_names(),
            ["wkt", "geometry"]
        );

        let sedona_types = df
            .schema()
            .sedona_types()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(sedona_types.len(), 2);
        assert_eq!(sedona_types[0], SedonaType::Arrow(DataType::Utf8View));
        assert_eq!(
            sedona_types[1],
            SedonaType::WkbView(Edges::Planar, deserialize_crs("EPSG:3857").unwrap())
        );
    }

    #[tokio::test]
    async fn test_auto_configure_spilled_batch_threshold() {
        use crate::context_builder::SedonaContextBuilder;
        use sedona_common::option::SedonaOptions;

        // Specify a memory limit (10GB), spilled batch threshold will also be limited,
        // but no lower than 10MB due to the minimum floor.
        let memory_limit: usize = 10 * 1024 * 1024 * 1024;
        let ctx = SedonaContextBuilder::new()
            .with_memory_limit(memory_limit)
            .build()
            .await
            .unwrap();
        let state = ctx.ctx.state();
        let opts = state
            .config_options()
            .extensions
            .get::<SedonaOptions>()
            .expect("SedonaOptions not found");
        assert!(opts.spatial_join.spilled_batch_in_memory_size_threshold >= 10 * 1024 * 1024);

        // Explicitly disable the memory limit; spilled batch threshold should be unlimited
        // (0 means unlimited)
        let ctx = SedonaContextBuilder::new()
            .without_memory_limit()
            .build()
            .await
            .unwrap();
        let state = ctx.ctx.state();
        let opts = state
            .config_options()
            .extensions
            .get::<SedonaOptions>()
            .expect("SedonaOptions not found");
        assert_eq!(opts.spatial_join.spilled_batch_in_memory_size_threshold, 0);
    }

    /// Test geography literal bounding via the WkbBounder2DFactory → SpatialFilterFactory path.
    ///
    /// This covers antimeridian wraparound, distance expansion (~100km), and NULL handling.
    #[cfg(feature = "s2geography")]
    #[tokio::test]
    async fn test_geography_literal_bounder() {
        use datafusion::physical_expr::expressions::Literal as PhysicalLiteral;
        use sedona_common::option::SedonaOptions;
        use sedona_expr::spatial_filter::SpatialFilterFactory;
        use sedona_geometry::interval::IntervalTrait;
        use sedona_schema::datatypes::{WKB_GEOGRAPHY, WKB_VIEW_GEOGRAPHY};
        use sedona_testing::create::create_scalar;

        let ctx = SedonaContext::new_local_interactive().await.unwrap();

        // Derive the factory from the context's SedonaOptions (which has the geography
        // bounder registered during context creation when s2geography feature is enabled)
        let state = ctx.ctx.state();
        let sedona_options = state
            .config_options()
            .extensions
            .get::<SedonaOptions>()
            .expect("SedonaOptions should be registered");
        let factory = SpatialFilterFactory::default()
            .with_bounder_factory(sedona_options.runtime.bounder_factory().clone());

        for sedona_type in [WKB_GEOGRAPHY, WKB_VIEW_GEOGRAPHY] {
            let storage_field = sedona_type.to_storage_field("", true).unwrap();

            // Test antimeridian wraparound bounds
            let scalar = create_scalar(Some("MULTIPOINT ((-179 42), (179 43))"), &sedona_type);
            let literal =
                PhysicalLiteral::new_with_metadata(scalar, Some(storage_field.metadata().into()));
            let raw_bounds = factory.literal_bounds(&literal, None).unwrap();

            // Should produce a wraparound interval crossing the antimeridian
            assert!(
                raw_bounds.x().is_wraparound(),
                "Expected wraparound x interval for antimeridian-crossing input"
            );
            // Check x bounds are reasonable (near ±179)
            assert!(
                raw_bounds.x().lo() > 178.99999 && raw_bounds.x().lo() <= 179.0,
                "x.lo() {} should be near 179",
                raw_bounds.x().lo()
            );
            assert!(
                raw_bounds.x().hi() >= -179.0 && raw_bounds.x().hi() < -178.99999,
                "x.hi() {} should be near -179",
                raw_bounds.x().hi()
            );
            // Check y bounds
            assert!(
                raw_bounds.y().lo() > 41.0 && raw_bounds.y().lo() <= 42.0,
                "y.lo() {} should be near 42",
                raw_bounds.y().lo()
            );
            assert!(
                raw_bounds.y().hi() >= 43.0 && raw_bounds.y().hi() < 44.0,
                "y.hi() {} should be near 43",
                raw_bounds.y().hi()
            );

            // Test distance expansion (~100km ≈ 0.9 degrees)
            let expanded_bounds = factory.literal_bounds(&literal, Some(100_000.0)).unwrap();
            assert!(
                expanded_bounds.x().is_wraparound(),
                "Expanded bounds should still be wraparound"
            );
            // Expanded bounds should be larger than raw bounds
            assert!(
                raw_bounds.x().lo() - expanded_bounds.x().lo() > 0.5,
                "Expanded x.lo() should be > 0.5 degrees larger"
            );
            assert!(
                expanded_bounds.x().hi() - raw_bounds.x().hi() > 0.5,
                "Expanded x.hi() should be > 0.5 degrees larger"
            );
            assert!(
                raw_bounds.y().lo() - expanded_bounds.y().lo() > 0.5,
                "Expanded y.lo() should be > 0.5 degrees larger"
            );
            assert!(
                expanded_bounds.y().hi() - raw_bounds.y().hi() > 0.5,
                "Expanded y.hi() should be > 0.5 degrees larger"
            );

            // Test NULL literal → empty bounds
            let null_scalar = create_scalar(None, &sedona_type);
            let null_literal = PhysicalLiteral::new_with_metadata(
                null_scalar,
                Some(storage_field.metadata().into()),
            );
            let null_bounds = factory.literal_bounds(&null_literal, None).unwrap();
            assert!(
                null_bounds.is_empty(),
                "NULL literal should produce empty bounds"
            );
        }
    }

    /// Test that LazyProjEngine is properly wired up through the global PROJ engine.
    ///
    /// This covers the three CrsEngine methods: to_projjson, get_transform_crs_to_crs,
    /// and get_transform_pipeline.
    #[cfg(feature = "proj")]
    #[tokio::test]
    async fn test_lazy_proj_engine() {
        use sedona_common::option::SedonaOptions;

        let ctx = SedonaContext::new_local_interactive().await.unwrap();

        // Derive the CrsEngine from the context's SedonaOptions
        let state = ctx.ctx.state();
        let sedona_options = state
            .config_options()
            .extensions
            .get::<SedonaOptions>()
            .expect("SedonaOptions should be registered");
        let engine = sedona_options.runtime.crs_engine();

        // Test to_projjson: convert EPSG:4326 to PROJJSON
        let projjson = engine.to_projjson("EPSG:4326").unwrap();
        assert!(
            projjson.contains("WGS 84") || projjson.contains("\"id\""),
            "PROJJSON should contain WGS 84 or id field: {}",
            projjson
        );

        // Test get_transform_crs_to_crs: WGS84 to Web Mercator
        let transform = engine
            .get_transform_crs_to_crs("EPSG:4326", "EPSG:3857", None, "")
            .unwrap();
        let mut coord = (0.0, 0.0); // (lng, lat) = (0, 0)
        transform.transform_coord(&mut coord).unwrap();
        // (0, 0) in WGS84 should transform to (0, 0) in Web Mercator
        assert!(
            coord.0.abs() < 1.0 && coord.1.abs() < 1.0,
            "Origin should map to near-origin in Web Mercator: {:?}",
            coord
        );

        // Test get_transform_pipeline: a simple pipeline that does nothing
        let pipeline_transform = engine.get_transform_pipeline("+proj=noop", "").unwrap();
        let mut coord2 = (10.0, 20.0);
        pipeline_transform.transform_coord(&mut coord2).unwrap();
        assert!(
            (coord2.0 - 10.0).abs() < 1e-10 && (coord2.1 - 20.0).abs() < 1e-10,
            "noop pipeline should preserve coordinates: {:?}",
            coord2
        );
    }
}
