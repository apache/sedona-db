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
use datafusion::prelude::DataFrame;
use savvy::{savvy, savvy_err, Result};
use sedona::context::SedonaDataFrame;
use sedona::reader::SedonaStreamReader;
use sedona::show::{DisplayMode, DisplayTableOptions};
use tokio::runtime::Runtime;

use crate::context::InternalContext;
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
        if let Some(col) = self.inner.primary_geometry_column_index()? {
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

    fn to_arrow_stream(&self, out: savvy::Sexp) -> Result<()> {
        let out_void = unsafe { savvy_ffi::R_ExternalPtrAddr(out.0) };
        if out_void.is_null() {
            return Err(savvy_err!("external pointer to null in to_arrow_stream()"));
        }

        let inner = self.inner.clone();
        let stream = wait_for_future_captured_r(&self.runtime, async move {
            inner.execute_stream_sedona().await
        })??;

        let reader = SedonaStreamReader::new(self.runtime.clone(), stream);
        let reader: Box<dyn RecordBatchReader + Send> = Box::new(reader);

        let mut ffi_stream = FFI_ArrowArrayStream::new(reader);
        let ffi_out = out_void as *mut FFI_ArrowArrayStream;
        unsafe { swap_nonoverlapping(&mut ffi_stream, ffi_out, 1) };

        Ok(())
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
            wait_for_future_captured_r(
                &self.runtime,
                async move { inner.collect_sedona().await },
            )??;

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
}
