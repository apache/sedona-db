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

/// [RecordBatchReader] wrapper that applies a projection
///
/// This utility can be used to implement a reader that conforms to the
/// DataFusion requirement that datasources apply the specified projection
/// when producing output.
pub struct ProjectedRecordBatchReader {
    inner: Box<dyn RecordBatchReader + Send>,
    projection: Vec<usize>,
    schema: SchemaRef,
}

impl ProjectedRecordBatchReader {
    /// Create a new wrapper from the indices into the input desired in the output
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

    /// Create a new wrapper from the column names from the input desired in the output
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

#[cfg(test)]
mod test {

    use arrow_array::{create_array, ArrayRef, RecordBatchIterator};
    use datafusion::assert_batches_eq;

    use super::*;

    #[test]
    fn projected_record_batch_reader() {
        let batch = RecordBatch::try_from_iter([
            (
                "x",
                create_array!(Utf8, ["one", "two", "three", "four"]) as ArrayRef,
            ),
            (
                "y",
                create_array!(Utf8, ["five", "six", "seven", "eight"]) as ArrayRef,
            ),
        ])
        .unwrap();

        let schema = batch.schema();

        // From indices
        let reader = RecordBatchIterator::new([Ok(batch.clone())], schema.clone());
        let projected =
            ProjectedRecordBatchReader::from_projection(Box::new(reader), vec![1, 0]).unwrap();
        let projected_batches = projected.collect::<Result<Vec<_>, ArrowError>>().unwrap();
        assert_batches_eq!(
            [
                "+-------+-------+",
                "| y     | x     |",
                "+-------+-------+",
                "| five  | one   |",
                "| six   | two   |",
                "| seven | three |",
                "| eight | four  |",
                "+-------+-------+",
            ],
            &projected_batches
        );

        // From output names
        let reader = RecordBatchIterator::new([Ok(batch.clone())], schema.clone());
        let projected =
            ProjectedRecordBatchReader::from_output_names(Box::new(reader), &["y", "x"]).unwrap();
        let projected_batches = projected.collect::<Result<Vec<_>, ArrowError>>().unwrap();
        assert_batches_eq!(
            [
                "+-------+-------+",
                "| y     | x     |",
                "+-------+-------+",
                "| five  | one   |",
                "| six   | two   |",
                "| seven | three |",
                "| eight | four  |",
                "+-------+-------+",
            ],
            &projected_batches
        );
    }
}
