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
use std::{any::Any, fmt::Debug, sync::Arc};

use arrow_array::RecordBatchReader;
use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion::execution::context::TaskContext;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{Partitioning, SendableRecordBatchStream};
use datafusion::{
    catalog::{Session, TableProvider},
    common::Result,
    datasource::TableType,
    physical_expr::EquivalenceProperties,
    physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties},
    prelude::Expr,
};
use datafusion_common::DataFusionError;
use parking_lot::Mutex;
use sedona_common::sedona_internal_err;

/// A [TableProvider] wrapping a [RecordBatchReader]
///
/// This provider wraps a once-scannable [RecordBatchReader]. If scanned
/// more than once, this provider will error. This reader wraps its input
/// such that extension types are preserved in DataFusion internals (i.e.,
/// it is intended for scanning external tables as SedonaDB).
pub struct RecordBatchReaderProvider {
    reader: Mutex<Option<Box<dyn RecordBatchReader + Send>>>,
    schema: SchemaRef,
}

unsafe impl Sync for RecordBatchReaderProvider {}

impl RecordBatchReaderProvider {
    pub fn new(reader: Box<dyn RecordBatchReader + Send>) -> Self {
        let schema = reader.schema();
        Self {
            reader: Mutex::new(Some(reader)),
            schema,
        }
    }
}

impl Debug for RecordBatchReaderProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecordBatchReaderProvider")
            .field("reader", &"<RecordBatchReader>".to_string())
            .field("schema", &self.schema)
            .finish()
    }
}

#[async_trait]
impl TableProvider for RecordBatchReaderProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::View
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let mut reader_guard = self.reader.lock();
        if let Some(reader) = reader_guard.take() {
            let projection = projection.cloned();
            Ok(Arc::new(RecordBatchReaderExec::try_new(
                reader, limit, projection,
            )?))
        } else {
            sedona_internal_err!("Can't scan RecordBatchReader provider more than once")
        }
    }
}

/// An iterator that limits the number of rows from a RecordBatchReader
struct RowLimitedIterator {
    reader: Option<Box<dyn RecordBatchReader + Send>>,
    limit: usize,
    rows_consumed: usize,
}

impl RowLimitedIterator {
    fn new(reader: Box<dyn RecordBatchReader + Send>, limit: usize) -> Self {
        Self {
            reader: Some(reader),
            limit,
            rows_consumed: 0,
        }
    }
}

impl Iterator for RowLimitedIterator {
    type Item = Result<arrow_array::RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        // Check if we have already consumed enough rows
        if self.rows_consumed >= self.limit {
            self.reader = None;
            return None;
        }

        let reader = self.reader.as_mut()?;
        match reader.next() {
            Some(Ok(batch)) => {
                let batch_rows = batch.num_rows();

                if self.rows_consumed + batch_rows <= self.limit {
                    // Batch fits within limit, consume it entirely
                    self.rows_consumed += batch_rows;
                    Some(Ok(batch))
                } else {
                    // Batch would exceed limit, need to truncate it
                    let rows_to_take = self.limit - self.rows_consumed;
                    self.rows_consumed = self.limit;
                    self.reader = None;
                    Some(Ok(batch.slice(0, rows_to_take)))
                }
            }
            Some(Err(e)) => {
                self.reader = None;
                Some(Err(DataFusionError::from(e)))
            }
            None => {
                self.reader = None;
                None
            }
        }
    }
}

struct RecordBatchReaderExec {
    reader: Mutex<Option<Box<dyn RecordBatchReader + Send>>>,
    schema: SchemaRef,
    properties: PlanProperties,
    limit: Option<usize>,
    projection: Option<Vec<usize>>,
}

impl RecordBatchReaderExec {
    fn try_new(
        reader: Box<dyn RecordBatchReader + Send>,
        limit: Option<usize>,
        projection: Option<Vec<usize>>,
    ) -> Result<Self> {
        let full_schema = reader.schema();
        let schema: SchemaRef = if let Some(indices) = projection.as_ref() {
            SchemaRef::new(
                full_schema
                    .project(indices)
                    .map_err(DataFusionError::from)?,
            )
        } else {
            full_schema.clone()
        };
        let properties = PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );

        Ok(Self {
            reader: Mutex::new(Some(reader)),
            schema,
            properties,
            limit,
            projection,
        })
    }
}

impl Debug for RecordBatchReaderExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecordBatchReaderExec")
            .field("reader", &"<RecordBatchReader>".to_string())
            .field("schema", &self.schema)
            .field("properties", &self.properties)
            .field("limit", &self.limit)
            .field("projection", &self.projection)
            .finish()
    }
}

impl DisplayAs for RecordBatchReaderExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "RecordBatchReaderExec")
    }
}

impl ExecutionPlan for RecordBatchReaderExec {
    fn name(&self) -> &str {
        "RecordBatchReaderExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        Vec::new()
    }

    fn with_new_children(
        self: Arc<Self>,
        _: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let mut reader_guard = self.reader.lock();

        let reader = if let Some(reader) = reader_guard.take() {
            reader
        } else {
            return sedona_internal_err!("Can't scan RecordBatchReader provider more than once");
        };

        match self.limit {
            Some(limit) => {
                // Create a row-limited iterator that properly handles row counting
                let projection = self.projection.clone();
                let iter = RowLimitedIterator::new(reader, limit).map(move |res| match res {
                    Ok(batch) => {
                        if let Some(indices) = projection.as_ref() {
                            batch.project(indices).map_err(|e| e.into())
                        } else {
                            Ok(batch)
                        }
                    }
                    Err(e) => Err(e),
                });
                let stream = Box::pin(futures::stream::iter(iter));
                let record_batch_stream =
                    RecordBatchStreamAdapter::new(self.schema.clone(), stream);
                Ok(Box::pin(record_batch_stream))
            }
            None => {
                // No limit, just convert the reader directly to a stream
                let projection = self.projection.clone();
                let iter = reader.map(move |item| match item {
                    Ok(batch) => {
                        if let Some(indices) = projection.as_ref() {
                            batch.project(indices).map_err(|e| e.into())
                        } else {
                            Ok(batch)
                        }
                    }
                    Err(e) => Err(e.into()),
                });
                let stream = Box::pin(futures::stream::iter(iter));
                let record_batch_stream =
                    RecordBatchStreamAdapter::new(self.schema.clone(), stream);
                Ok(Box::pin(record_batch_stream))
            }
        }
    }
}

#[cfg(test)]
mod test {

    use arrow_array::{RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::prelude::{col, DataFrame, SessionContext};
    use rstest::rstest;
    use sedona_schema::datatypes::WKB_GEOMETRY;
    use sedona_testing::create::create_array_storage;

    use super::*;

    fn create_test_batch(size: usize, start_id: i32) -> RecordBatch {
        let schema = Schema::new(vec![Field::new("id", DataType::Int32, false)]);
        let ids: Vec<i32> = (start_id..start_id + size as i32).collect();
        RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(arrow_array::Int32Array::from(ids))],
        )
        .unwrap()
    }

    fn create_test_reader(batch_sizes: Vec<usize>) -> Box<dyn RecordBatchReader + Send> {
        let mut start_id = 0i32;
        let batches: Vec<RecordBatch> = batch_sizes
            .into_iter()
            .map(|size| {
                let batch = create_test_batch(size, start_id);
                start_id += size as i32;
                batch
            })
            .collect();
        let schema = batches[0].schema();
        Box::new(RecordBatchIterator::new(
            batches.into_iter().map(Ok),
            schema,
        ))
    }

    #[tokio::test]
    async fn provider() {
        let ctx = SessionContext::new();

        let schema: SchemaRef = Schema::new(vec![
            Field::new("not_geometry", DataType::Int32, true),
            WKB_GEOMETRY.to_storage_field("geometry", true).unwrap(),
        ])
        .into();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                arrow_array::create_array!(Int32, [1, 2]),
                create_array_storage(&[Some("POINT (0 1)"), Some("POINT (2 3)")], &WKB_GEOMETRY),
            ],
        )
        .unwrap();

        // Create the provider
        let reader =
            RecordBatchIterator::new(vec![batch.clone()].into_iter().map(Ok), schema.clone());
        let provider = RecordBatchReaderProvider::new(Box::new(reader));

        // Ensure we get the expected output
        let df = ctx.read_table(Arc::new(provider)).unwrap();
        assert_eq!(Arc::new(df.schema().as_arrow().clone()), schema);
        let results = df.collect().await.unwrap();
        assert_eq!(results, vec![batch])
    }

    #[rstest]
    #[case(vec![10, 20, 30], None, 60)] // No limit
    #[case(vec![10, 20, 30], Some(5), 5)] // Limit within first batch
    #[case(vec![10, 20, 30], Some(10), 10)] // Limit exactly at first batch boundary
    #[case(vec![10, 20, 30], Some(15), 15)] // Limit within second batch
    #[case(vec![10, 20, 30], Some(30), 30)] // Limit at second batch boundary
    #[case(vec![10, 20, 30], Some(45), 45)] // Limit within third batch
    #[case(vec![10, 20, 30], Some(60), 60)] // Limit at total rows
    #[case(vec![10, 20, 30], Some(100), 60)] // Limit exceeds total rows
    #[case(vec![0, 5, 0, 3], Some(6), 6)] // Empty batches mixed in, limit within data
    #[case(vec![0, 5, 0, 3], Some(8), 8)] // Empty batches mixed in, limit equals total
    #[case(vec![0, 5, 0, 3], None, 8)] // Empty batches mixed in, no limit
    #[tokio::test]
    async fn test_scan_with_row_limit(
        #[case] batch_sizes: Vec<usize>,
        #[case] limit: Option<usize>,
        #[case] expected_rows: usize,
    ) {
        let ctx = SessionContext::new();

        // Verify that the RecordBatchReaderExec node in the execution plan should contain the correct limit
        let physical_plan = read_test_table_with_limit(&ctx, batch_sizes.clone(), limit)
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        let reader_exec = find_record_batch_reader_exec(physical_plan.as_ref())
            .expect("The plan should contain RecordBatchReaderExec");
        assert_eq!(reader_exec.limit, limit);

        let df = read_test_table_with_limit(&ctx, batch_sizes, limit).unwrap();
        let results = df.collect().await.unwrap();
        let total_rows: usize = results.iter().map(|batch| batch.num_rows()).sum();
        assert_eq!(total_rows, expected_rows);

        // Verify row values are correct (sequential IDs starting from 0)
        if expected_rows > 0 {
            let mut expected_id = 0i32;
            for batch in results.iter() {
                let id_array = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<arrow_array::Int32Array>()
                    .unwrap();
                for i in 0..id_array.len() {
                    assert_eq!(id_array.value(i), expected_id);
                    expected_id += 1;
                }
            }
        }
    }

    #[tokio::test]
    async fn test_projection_pushdown() {
        let ctx = SessionContext::new();

        // Create a two-column batch
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(arrow_array::Int32Array::from(vec![1, 2, 3])),
                Arc::new(arrow_array::Int32Array::from(vec![10, 20, 30])),
            ],
        )
        .unwrap();

        // Wrap in a RecordBatchReaderProvider
        let reader =
            RecordBatchIterator::new(vec![batch.clone()].into_iter().map(Ok), Arc::new(schema));
        let provider = Arc::new(RecordBatchReaderProvider::new(Box::new(reader)));

        // Read table then select only column b (this should push projection into scan)
        let df = ctx.read_table(provider).unwrap();
        let df_b = df.select(vec![col("b")]).unwrap();
        let results = df_b.collect().await.unwrap();
        assert_eq!(results.len(), 1);
        let out_batch = &results[0];
        assert_eq!(out_batch.num_columns(), 1);
        assert_eq!(out_batch.schema().field(0).name(), "b");
        let values = out_batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .unwrap();
        assert_eq!(values.values(), &[10, 20, 30]);
    }

    fn read_test_table_with_limit(
        ctx: &SessionContext,
        batch_sizes: Vec<usize>,
        limit: Option<usize>,
    ) -> Result<DataFrame> {
        let reader = create_test_reader(batch_sizes);
        let provider = Arc::new(RecordBatchReaderProvider::new(reader));
        let df = ctx.read_table(provider)?;
        if let Some(limit) = limit {
            df.limit(0, Some(limit))
        } else {
            Ok(df)
        }
    }

    // Navigate through the plan structure to find our RecordBatchReaderExec
    fn find_record_batch_reader_exec(plan: &dyn ExecutionPlan) -> Option<&RecordBatchReaderExec> {
        if let Some(reader_exec) = plan.as_any().downcast_ref::<RecordBatchReaderExec>() {
            return Some(reader_exec);
        }

        // Recursively search children
        for child in plan.children() {
            if let Some(reader_exec) = find_record_batch_reader_exec(child.as_ref()) {
                return Some(reader_exec);
            }
        }
        None
    }
}
