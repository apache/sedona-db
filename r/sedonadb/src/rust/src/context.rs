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
use std::sync::Arc;

use arrow_array::{
    ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream},
    RecordBatchReader,
};
use arrow_schema::ArrowError;
use datafusion::catalog::{MemTable, TableProvider};
use savvy::{savvy, savvy_err, Result};

use sedona::{context::SedonaContext, record_batch_reader_provider::RecordBatchReaderProvider};
use sedona_geoparquet::provider::GeoParquetReadOptions;
use tokio::runtime::Runtime;

use crate::{
    dataframe::{new_data_frame, InternalDataFrame},
    runtime::wait_for_future_captured_r,
};

#[savvy]
pub struct InternalContext {
    pub inner: Arc<SedonaContext>,
    pub runtime: Arc<Runtime>,
}

#[savvy]
impl InternalContext {
    pub fn new() -> Result<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;

        let inner = wait_for_future_captured_r(&runtime, SedonaContext::new_local_interactive())??;

        Ok(Self {
            inner: Arc::new(inner),
            runtime: Arc::new(runtime),
        })
    }

    pub fn read_parquet(&self, paths: savvy::Sexp) -> Result<InternalDataFrame> {
        let paths_strsxp = savvy::StringSexp::try_from(paths)?;
        let table_paths = paths_strsxp
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        let inner_context = self.inner.clone();
        let inner = wait_for_future_captured_r(&self.runtime, async move {
            inner_context
                .read_parquet(table_paths, GeoParquetReadOptions::default())
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
        let ffi_stream =
            unsafe { savvy_ffi::R_ExternalPtrAddr(stream_xptr.0) as *mut FFI_ArrowArrayStream };
        if ffi_stream.is_null() {
            return Err(savvy_err!("external pointer to null in to_arrow_schema()"));
        }

        let stream = unsafe { FFI_ArrowArrayStream::from_raw(ffi_stream as _) };
        let stream_reader = ArrowArrayStreamReader::try_new(stream)?;

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

    pub fn deregister_table(&self, table_ref: &str) -> savvy::Result<()> {
        self.inner.ctx.deregister_table(table_ref)?;
        Ok(())
    }
}
