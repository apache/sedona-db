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

use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::{ArrowError, SchemaRef};
use datafusion_common::Result;

pub struct ProjectedRecordBatchReader {
    inner: Box<dyn RecordBatchReader + Send>,
    projection: Vec<usize>,
    schema: SchemaRef,
}

impl ProjectedRecordBatchReader {
    pub fn from_projection(
        inner: Box<dyn RecordBatchReader + Send>,
        projection: Vec<usize>,
    ) -> Result<Self> {
        let schema = inner.schema().project(&projection)?;
        Ok(Self {
            inner,
            projection,
            schema: Arc::new(schema),
        })
    }

    pub fn from_output_names(
        inner: Box<dyn RecordBatchReader + Send>,
        projection: &[&str],
    ) -> Result<Self> {
        let input_indices = projection
            .iter()
            .map(|col| inner.schema().index_of(col))
            .collect::<Result<Vec<usize>, ArrowError>>()?;
        Self::from_projection(inner, input_indices)
    }
}

impl RecordBatchReader for ProjectedRecordBatchReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

impl Iterator for ProjectedRecordBatchReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(next) = self.inner.next() {
            match next {
                Ok(batch) => Some(batch.project(&self.projection)),
                Err(err) => Some(Err(err)),
            }
        } else {
            None
        }
    }
}
