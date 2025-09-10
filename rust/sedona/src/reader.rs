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

/// Utility to convert a [SendableRecordBatchStream] into a [RecordBatchReader]
///
/// This is needed for clients like ADBC and Python where this is the format
/// that is required for export.
pub struct SedonaStreamReader {
    runtime: Arc<Runtime>,
    stream: SendableRecordBatchStream,
}

impl SedonaStreamReader {
    pub fn new(runtime: Arc<Runtime>, stream: SendableRecordBatchStream) -> Self {
        Self { runtime, stream }
    }
}

impl Iterator for SedonaStreamReader {
    type Item = std::result::Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.runtime.block_on(self.stream.try_next()) {
            Ok(maybe_batch) => maybe_batch.map(Ok),
            Err(err) => Some(Err(ArrowError::ExternalError(Box::new(err)))),
        }
    }
}

impl RecordBatchReader for SedonaStreamReader {
    fn schema(&self) -> SchemaRef {
        self.stream.schema()
    }
}

#[cfg(test)]
mod test {

    use arrow_schema::{DataType, Field, Schema};

    use crate::context::SedonaContext;

    use super::*;

    #[test]
    fn reader() {
        let runtime = Arc::new(tokio::runtime::Runtime::new().unwrap());
        let ctx = SedonaContext::new();
        let df = runtime.block_on(ctx.sql("SELECT 1 as one")).unwrap();
        let expected_batches = runtime.block_on(df.clone().collect()).unwrap();
        assert_eq!(expected_batches.len(), 1);

        let stream = runtime.block_on(df.execute_stream()).unwrap();
        let mut reader = SedonaStreamReader::new(runtime, stream);

        let expected_schema = Arc::new(Schema::new([
            Field::new("one", DataType::Int64, false).into()
        ]));
        assert_eq!(reader.schema(), expected_schema);

        assert_eq!(reader.next().unwrap().unwrap(), expected_batches[0]);
        assert!(reader.next().is_none());
    }
}
