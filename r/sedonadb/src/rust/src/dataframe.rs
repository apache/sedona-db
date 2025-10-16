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
use std::ptr::swap_nonoverlapping;
use std::sync::Arc;

use arrow_array::ffi::FFI_ArrowSchema;
use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use arrow_array::{RecordBatchIterator, RecordBatchReader};
use datafusion::catalog::MemTable;
use datafusion::{logical_expr::SortExpr, prelude::DataFrame};
use datafusion_common::Column;
use datafusion_expr::Expr;
use datafusion_ffi::table_provider::FFI_TableProvider;
use savvy::{savvy, savvy_err, IntoExtPtrSexp, Result};
use sedona::context::{SedonaDataFrame, SedonaWriteOptions};
use sedona::reader::SedonaStreamReader;
use sedona::show::{DisplayMode, DisplayTableOptions};
use sedona_geoparquet::options::{GeoParquetVersion, TableGeoParquetOptions};
use sedona_schema::schema::SedonaSchema;
use tokio::runtime::Runtime;

use crate::context::InternalContext;
use crate::ffi::{import_schema, FFITableProviderR};
use crate::runtime::wait_for_future_captured_r;

#[savvy]
pub struct InternalDataFrame {
    pub inner: DataFrame,
    pub runtime: Arc<Runtime>,
}

pub fn new_data_frame(inner: DataFrame, runtime: Arc<Runtime>) -> InternalDataFrame {
    InternalDataFrame { inner, runtime }
}

#[savvy]
impl InternalDataFrame {
    fn limit(&self, n: f64) -> Result<InternalDataFrame> {
        let inner = self.inner.clone().limit(0, Some(n.floor() as usize))?;
        Ok(InternalDataFrame {
            inner,
            runtime: self.runtime.clone(),
        })
    }

    fn count(&self) -> Result<savvy::Sexp> {
        let inner = self.inner.clone();
        let counted =
            wait_for_future_captured_r(&self.runtime, async move { inner.count().await })??;

        let counted_double = counted as f64;
        savvy::Sexp::try_from(counted_double)
    }

    fn primary_geometry_column_index(&self) -> Result<savvy::Sexp> {
        if let Some(col) = self.inner.schema().primary_geometry_column_index()? {
            Ok(unsafe { savvy::Sexp(savvy_ffi::Rf_ScalarInteger(col.try_into()?)) })
        } else {
            Ok(savvy::NullSexp.into())
        }
    }

    fn to_arrow_schema(&self, out: savvy::Sexp) -> Result<()> {
        let out_void = unsafe { savvy_ffi::R_ExternalPtrAddr(out.0) };
        if out_void.is_null() {
            return Err(savvy_err!("external pointer to null in to_arrow_schema()"));
        }

        let schema_no_qualifiers = self.inner.schema().clone().strip_qualifiers();
        let schema = schema_no_qualifiers.as_arrow();
        let mut ffi_schema = FFI_ArrowSchema::try_from(schema)?;
        let ffi_out = out_void as *mut FFI_ArrowSchema;
        unsafe { swap_nonoverlapping(&mut ffi_schema, ffi_out, 1) };
        Ok(())
    }

    fn to_arrow_stream(&self, out: savvy::Sexp, requested_schema_xptr: savvy::Sexp) -> Result<()> {
        let out_void = unsafe { savvy_ffi::R_ExternalPtrAddr(out.0) };
        if out_void.is_null() {
            return Err(savvy_err!("external pointer to null in to_arrow_stream()"));
        }

        let maybe_requested_schema = if requested_schema_xptr.is_null() {
            None
        } else {
            Some(import_schema(requested_schema_xptr))
        };

        if maybe_requested_schema.is_some() {
            return Err(savvy_err!("Requested schema is not supported"));
        }

        let inner = self.inner.clone();
        let stream =
            wait_for_future_captured_r(
                &self.runtime,
                async move { inner.execute_stream().await },
            )??;

        let reader = SedonaStreamReader::new(self.runtime.clone(), stream);
        let reader: Box<dyn RecordBatchReader + Send> = Box::new(reader);

        let mut ffi_stream = FFI_ArrowArrayStream::new(reader);
        let ffi_out = out_void as *mut FFI_ArrowArrayStream;
        unsafe { swap_nonoverlapping(&mut ffi_stream, ffi_out, 1) };

        Ok(())
    }

    fn to_provider(&self) -> Result<savvy::Sexp> {
        let provider = self.inner.clone().into_view();
        // Literal true is because the TableProvider that wraps this DataFrame
        // can support filters being pushed down.
        let ffi_provider =
            FFI_TableProvider::new(provider, true, Some(self.runtime.handle().clone()));

        let mut ffi_xptr = FFITableProviderR(ffi_provider).into_external_pointer();
        unsafe { savvy_ffi::Rf_protect(ffi_xptr.0) };
        ffi_xptr.set_class(vec!["datafusion_table_provider"])?;
        unsafe { savvy_ffi::Rf_unprotect(1) };

        Ok(ffi_xptr)
    }

    fn compute(&self, ctx: &InternalContext) -> Result<InternalDataFrame> {
        let schema = self.inner.schema();
        let batches =
            wait_for_future_captured_r(&self.runtime, self.inner.clone().collect_partitioned())??;
        let provider = Arc::new(MemTable::try_new(schema.clone().into(), batches)?);
        let inner = ctx.inner.ctx.read_table(provider)?;
        Ok(new_data_frame(inner, self.runtime.clone()))
    }

    fn collect(&self, out: savvy::Sexp) -> Result<savvy::Sexp> {
        let out_void = unsafe { savvy_ffi::R_ExternalPtrAddr(out.0) };
        if out_void.is_null() {
            return Err(savvy_err!("external pointer to null in collect()"));
        }

        let inner = self.inner.clone();
        let batches =
            wait_for_future_captured_r(&self.runtime, async move { inner.collect().await })??;

        let size: usize = batches.iter().map(|batch| batch.num_rows()).sum();

        let reader: Box<dyn RecordBatchReader + Send> = if batches.is_empty() {
            let schema_no_qualifiers = self.inner.schema().clone().strip_qualifiers();
            let schema = schema_no_qualifiers.as_arrow();
            Box::new(RecordBatchIterator::new(
                vec![].into_iter(),
                Arc::new(schema.clone()),
            ))
        } else {
            let schema = batches[0].schema();
            Box::new(RecordBatchIterator::new(
                batches.into_iter().map(Ok),
                schema,
            ))
        };

        let mut ffi_stream = FFI_ArrowArrayStream::new(reader);
        let ffi_out = out_void as *mut FFI_ArrowArrayStream;
        unsafe { swap_nonoverlapping(&mut ffi_stream, ffi_out, 1) };

        savvy::Sexp::try_from(size as f64)
    }

    fn to_view(&self, ctx: &InternalContext, table_ref: &str, overwrite: bool) -> Result<()> {
        let provider = self.inner.clone().into_view();
        if overwrite && ctx.inner.ctx.table_exist(table_ref)? {
            ctx.deregister_table(table_ref)?;
        }

        ctx.inner.ctx.register_table(table_ref, provider)?;
        Ok(())
    }

    fn show(
        &self,
        ctx: &InternalContext,
        width_chars: i32,
        ascii: bool,
        limit: Option<f64>,
    ) -> Result<savvy::Sexp> {
        let mut options = DisplayTableOptions::new();
        options.table_width = width_chars.try_into().unwrap_or(u16::MAX);
        options.arrow_options = options.arrow_options.with_types_info(true);
        if !ascii {
            options.display_mode = DisplayMode::Utf8;
        }

        let inner = self.inner.clone();
        let inner_context = ctx.inner.clone();
        let limit_usize = limit.map(|value| {
            if value > i32::MAX as f64 {
                i32::MAX as usize
            } else {
                value as usize
            }
        });

        let out_string = wait_for_future_captured_r(&self.runtime, async move {
            inner
                .show_sedona(&inner_context, limit_usize, options)
                .await
        })??;

        savvy::Sexp::try_from(out_string)
    }

    #[allow(clippy::too_many_arguments)]
    fn to_parquet(
        &self,
        ctx: &InternalContext,
        path: &str,
        partition_by: savvy::Sexp,
        sort_by: savvy::Sexp,
        single_file_output: bool,
        overwrite_bbox_columns: bool,
        geoparquet_version: Option<&str>,
    ) -> savvy::Result<()> {
        let partition_by_strsxp = savvy::StringSexp::try_from(partition_by)?;
        let partition_by_vec = partition_by_strsxp
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        let sort_by_strsxp = savvy::StringSexp::try_from(sort_by)?;
        let sort_by_vec = sort_by_strsxp
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        let sort_by_expr = sort_by_vec
            .iter()
            .map(|name| {
                let column = Expr::Column(Column::new_unqualified(name));
                SortExpr::new(column, true, false)
            })
            .collect::<Vec<_>>();

        let options = SedonaWriteOptions::new()
            .with_partition_by(partition_by_vec)
            .with_sort_by(sort_by_expr)
            .with_single_file_output(single_file_output);

        let mut writer_options = TableGeoParquetOptions::new();
        writer_options.overwrite_bbox_columns = overwrite_bbox_columns;
        if let Some(geoparquet_version) = geoparquet_version {
            writer_options.geoparquet_version = geoparquet_version
                .parse()
                .map_err(|e| savvy::Error::new(format!("Invalid geoparquet_version: {e}")))?;
        } else {
            writer_options.geoparquet_version = GeoParquetVersion::Omitted;
        }

        let inner = self.inner.clone();
        let inner_context = ctx.inner.clone();
        let path_owned = path.to_string();

        wait_for_future_captured_r(&self.runtime, async move {
            inner
                .write_geoparquet(&inner_context, &path_owned, options, Some(writer_options))
                .await
        })??;

        Ok(())
    }
}
