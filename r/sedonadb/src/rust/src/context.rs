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
use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::RecordBatchReader;
use arrow_schema::{ArrowError, DataType};
use datafusion::catalog::{MemTable, TableProvider};
use datafusion_expr::ScalarUDF;
use datafusion_ffi::udf::FFI_ScalarUDF;
use savvy::{savvy, savvy_err, IntoExtPtrSexp, OwnedStringSexp, Result};

use sedona::{
    context::SedonaContext, context_builder::SedonaContextBuilder,
    record_batch_reader_provider::RecordBatchReaderProvider,
};
use sedona_extension::{runtime::RuntimeHandle, scalar_kernel::with_config_options_for_ffi};
use sedona_geoparquet::provider::GeoParquetReadOptions;

use crate::{
    dataframe::{new_data_frame, InternalDataFrame},
    ffi::{import_array_stream, import_scalar_udf, import_table_provider, FFIScalarUdfR},
    runtime::wait_for_future_captured_r,
};

#[savvy]
pub struct InternalContext {
    pub inner: Arc<SedonaContext>,
    pub runtime: Arc<RuntimeHandle>,
}

#[savvy]
impl InternalContext {
    pub fn new(option_keys: savvy::Sexp, option_values: savvy::Sexp) -> Result<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;

        let keys = savvy::StringSexp::try_from(option_keys)?;
        let values = savvy::StringSexp::try_from(option_values)?;

        let options: HashMap<String, String> = keys
            .iter()
            .zip(values.iter())
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();

        let inner = if options.is_empty() {
            wait_for_future_captured_r(&runtime, SedonaContext::new_local_interactive())??
        } else {
            let builder = SedonaContextBuilder::from_options(&options)?;
            wait_for_future_captured_r(&runtime, builder.build())??
        };

        Ok(Self {
            inner: Arc::new(inner),
            runtime: Arc::new(RuntimeHandle::new(runtime)),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn read_parquet(
        &self,
        paths: savvy::Sexp,
        option_keys: savvy::Sexp,
        option_values: savvy::Sexp,
        geometry_columns: savvy::Sexp,
        validate: bool,
        partitioning: savvy::Sexp,
    ) -> Result<InternalDataFrame> {
        let paths_strsxp = savvy::StringSexp::try_from(paths)?;
        let table_paths = paths_strsxp
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        let option_keys_strsxp = savvy::StringSexp::try_from(option_keys)?;
        let option_values_strsxp = savvy::StringSexp::try_from(option_values)?;
        let options = option_keys_strsxp
            .iter()
            .zip(option_values_strsxp.iter())
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect::<HashMap<_, _>>();

        let mut geo_options = GeoParquetReadOptions::from_table_options(options)
            .map_err(|e| savvy_err!("Invalid table options: {e}"))?;

        if !geometry_columns.is_null() {
            let geometry_columns_strsxp = savvy::StringSexp::try_from(geometry_columns)?;
            let geometry_columns_json = geometry_columns_strsxp
                .iter()
                .next()
                .ok_or_else(|| savvy_err!("`geometry_columns` must have length 1"))?;
            geo_options = geo_options
                .with_geometry_columns_json(geometry_columns_json)
                .map_err(|e| savvy_err!("Invalid geometry_columns JSON: {e}"))?;
        }

        geo_options = geo_options.with_validate(validate);

        if !partitioning.is_null() {
            let partitioning_strsxp = savvy::StringSexp::try_from(partitioning)?;
            geo_options = geo_options.with_table_partition_cols(
                partitioning_strsxp
                    .iter()
                    .map(|name| (name.to_string(), DataType::Utf8View))
                    .collect(),
            );
        }

        let inner_context = self.inner.clone();
        let inner = wait_for_future_captured_r(&self.runtime, async move {
            inner_context.read_parquet(table_paths, geo_options).await
        })??;

        Ok(new_data_frame(inner, self.runtime.clone()))
    }

    pub fn read(
        &self,
        paths: savvy::Sexp,
        option_keys: savvy::Sexp,
        option_values: savvy::Sexp,
        partitioning: savvy::Sexp,
        partitioning_set: bool,
        check_extension: bool,
        format: Option<&str>,
    ) -> Result<InternalDataFrame> {
        let paths = savvy::StringSexp::try_from(paths)?
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>();
        let keys = savvy::StringSexp::try_from(option_keys)?;
        let values = savvy::StringSexp::try_from(option_values)?;
        let options = keys
            .iter()
            .zip(values.iter())
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect::<HashMap<_, _>>();
        let format = format
            .map(|format| format.trim_start_matches('.').to_lowercase())
            .map(|format| {
                self.inner
                    .ctx
                    .state()
                    .get_file_format_factory(&format)
                    .ok_or_else(|| savvy_err!("No format registered for extension '{format}'"))
            })
            .transpose()?;
        let partitioning = if partitioning_set {
            Some(
                savvy::StringSexp::try_from(partitioning)?
                    .iter()
                    .map(|name| (name.to_string(), DataType::Utf8View))
                    .collect(),
            )
        } else {
            None
        };

        let inner_context = self.inner.clone();
        let inner = wait_for_future_captured_r(&self.runtime, async move {
            inner_context
                .read_with_partitioning(paths, &options, format, check_extension, partitioning)
                .await
        })??;

        Ok(new_data_frame(inner, self.runtime.clone()))
    }

    pub fn sql(&self, query: &str) -> Result<InternalDataFrame> {
        let query_string = query.to_string();
        let inner_context = self.inner.clone();
        let inner = wait_for_future_captured_r(&self.runtime, async move {
            inner_context.sql(&query_string).await
        })??;
        Ok(new_data_frame(inner, self.runtime.clone()))
    }

    pub fn view(&self, table_ref: &str) -> Result<InternalDataFrame> {
        let inner_context = self.inner.clone();
        let table_ref_string = table_ref.to_string();
        let inner = wait_for_future_captured_r(&self.runtime, async move {
            inner_context.ctx.table(table_ref_string).await
        })??;
        Ok(new_data_frame(inner, self.runtime.clone()))
    }

    pub fn data_frame_from_array_stream(
        &self,
        stream_xptr: savvy::Sexp,
        collect_now: bool,
    ) -> savvy::Result<InternalDataFrame> {
        let stream_reader = import_array_stream(stream_xptr)?;

        // Some readers are sensitive to being collected on the R thread or not, so
        // provide the option to collect everything immediately.
        let provider: Arc<dyn TableProvider> = if collect_now {
            let schema = stream_reader.schema();
            let batches = stream_reader.collect::<std::result::Result<Vec<_>, ArrowError>>()?;
            Arc::new(MemTable::try_new(schema, vec![batches])?)
        } else {
            Arc::new(RecordBatchReaderProvider::new(Box::new(stream_reader)))
        };

        let inner = self.inner.ctx.read_table(provider)?;
        Ok(new_data_frame(inner, self.runtime.clone()))
    }

    pub fn data_frame_from_table_provider(
        &self,
        provider_xptr: savvy::Sexp,
    ) -> Result<InternalDataFrame> {
        let provider = import_table_provider(provider_xptr)?;
        let inner = self.inner.ctx.read_table(provider)?;
        Ok(new_data_frame(inner, self.runtime.clone()))
    }

    pub fn deregister_table(&self, table_ref: &str) -> savvy::Result<()> {
        self.inner.ctx.deregister_table(table_ref)?;
        Ok(())
    }

    pub fn list_functions(&self) -> savvy::Result<savvy::Sexp> {
        let mut fn_names = Vec::new();
        let state = self.inner.ctx.state();
        fn_names.extend(state.scalar_functions().keys());
        fn_names.extend(state.aggregate_functions().keys());

        let fn_names_sexp = OwnedStringSexp::try_from(fn_names.as_slice())?;
        fn_names_sexp.into()
    }

    pub fn scalar_udf_xptr(&self, name: &str) -> savvy::Result<savvy::Sexp> {
        if let Some(udf) = self.inner.ctx.state().scalar_functions().get(name) {
            let udf = if let Some(sedona_udf) = self.inner.scalar_udf(name)? {
                let config_options = self.inner.ctx.state().config_options().clone();
                Arc::new(ScalarUDF::from(with_config_options_for_ffi(
                    sedona_udf,
                    config_options,
                )?))
            } else {
                udf.clone()
            };
            let ffi_scalar_udf: FFI_ScalarUDF = udf.into();
            let mut ffi_xptr = FFIScalarUdfR(ffi_scalar_udf).into_external_pointer();
            unsafe { savvy_ffi::Rf_protect(ffi_xptr.0) };
            ffi_xptr.set_class(vec!["datafusion_scalar_udf"])?;
            unsafe { savvy_ffi::Rf_unprotect(1) };

            Ok(ffi_xptr)
        } else {
            Err(savvy_err!("Scalar UDF '{name}' was not found"))
        }
    }

    pub fn register_scalar_udf(&self, scalar_udf_xptr: savvy::Sexp) -> savvy::Result<()> {
        let scalar_udf = import_scalar_udf(scalar_udf_xptr)?;
        self.inner.ctx.register_udf(scalar_udf);
        Ok(())
    }
}
