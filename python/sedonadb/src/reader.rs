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
use futures::TryStreamExt;
use std::sync::Arc;

use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::{ArrowError, SchemaRef};
use datafusion::execution::SendableRecordBatchStream;
use tokio::runtime::Runtime;

use crate::runtime::wait_for_future_from_rust;

/// Utility to convert a [SendableRecordBatchStream] into a [RecordBatchReader]
///
/// This is like the SedonaStreamReader except it checks for Python signals such
/// as cancellation.
pub struct PySedonaStreamReader {
    runtime: Arc<Runtime>,
    stream: SendableRecordBatchStream,
}

impl PySedonaStreamReader {
    pub fn new(runtime: Arc<Runtime>, stream: SendableRecordBatchStream) -> Self {
        Self { runtime, stream }
    }
}

impl Iterator for PySedonaStreamReader {
    type Item = std::result::Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match wait_for_future_from_rust(&self.runtime, self.stream.try_next()) {
                Ok(Ok(maybe_batch)) => match maybe_batch {
                    Some(batch) => {
                        if batch.num_rows() == 0 {
                            continue;
                        }

                        return Some(Ok(batch));
                    }
                    None => return None,
                },
                Ok(Err(df_err)) => return Some(Err(ArrowError::ExternalError(Box::new(df_err)))),
                Err(py_err) => return Some(Err(ArrowError::ExternalError(Box::new(py_err)))),
            }
        }
    }
}

impl RecordBatchReader for PySedonaStreamReader {
    fn schema(&self) -> SchemaRef {
        self.stream.schema()
    }
}
