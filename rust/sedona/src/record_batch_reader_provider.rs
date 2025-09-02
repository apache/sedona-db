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
use std::sync::RwLock;
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
use sedona_common::sedona_internal_err;
use sedona_expr::projection::wrap_batch_maybe_deprecated;
use sedona_schema::projection::wrap_schema_maybe_deprecated;

/// A [TableProvider] wrapping a [RecordBatchReader]
///
/// This provider wraps a once-scannable [RecordBatchReader]. If scanned
/// more than once, this provider will error. This reader wraps its input
/// such that extension types are preserved in DataFusion internals (i.e.,
/// it is intended for scanning external tables as SedonaDB).
pub struct RecordBatchReaderProvider {
    reader: RwLock<Option<Box<dyn RecordBatchReader + Send>>>,
    schema: SchemaRef,
}

unsafe impl Sync for RecordBatchReaderProvider {}

impl RecordBatchReaderProvider {
    pub fn new(reader: Box<dyn RecordBatchReader + Send>) -> Self {
        let schema = wrap_schema_maybe_deprecated(&reader.schema());
        Self {
            reader: RwLock::new(Some(reader)),
            schema: Arc::new(schema),
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
        _projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let mut writable_reader = self.reader.try_write().map_err(|_| {
            DataFusionError::Internal("Failed to acquire lock on RecordBatchReader".to_string())
        })?;
        if let Some(reader) = writable_reader.take() {
            Ok(Arc::new(RecordBatchReaderExec::new(reader, limit)))
        } else {
            sedona_internal_err!("Can't scan RecordBatchReader provider more than once")
        }
    }
}

struct RecordBatchReaderExec {
    reader: RwLock<Option<Box<dyn RecordBatchReader + Send>>>,
    schema: SchemaRef,
    properties: PlanProperties,
    limit: Option<usize>,
}

unsafe impl Sync for RecordBatchReaderExec {}

impl RecordBatchReaderExec {
    fn new(reader: Box<dyn RecordBatchReader + Send>, limit: Option<usize>) -> Self {
        let schema = reader.schema();
        let properties = PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );

        Self {
            reader: RwLock::new(Some(reader)),
            schema,
            properties,
            limit,
        }
    }
}

impl Debug for RecordBatchReaderExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecordBatchReaderExec")
            .field("reader", &"<RecordBatchReader>".to_string())
            .field("schema", &self.schema)
            .field("properties", &self.properties)
            .field("limit", &self.limit)
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
        let mut writable_reader = self.reader.try_write().map_err(|_| {
            DataFusionError::Internal("Failed to acquire lock on RecordBatchReader".to_string())
        })?;

        let reader = if let Some(reader) = writable_reader.take() {
            reader
        } else {
            return sedona_internal_err!("Can't scan RecordBatchReader provider more than once");
        };

        let limit = self.limit;

        // Create a stream from the RecordBatchReader iterator
        let iter = reader
            .map(|item| match item {
                Ok(batch) => Ok(wrap_batch_maybe_deprecated(batch)),
                Err(e) => Err(DataFusionError::from(e)),
            })
            .take(limit.unwrap_or(usize::MAX));

        let stream = Box::pin(futures::stream::iter(iter));
        let record_batch_stream = RecordBatchStreamAdapter::new(self.schema.clone(), stream);
        Ok(Box::pin(record_batch_stream))
    }
}

#[cfg(test)]
mod test {

    use arrow_array::{RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::prelude::SessionContext;
    use sedona_schema::datatypes::WKB_GEOMETRY;
    use sedona_testing::create::create_array_storage;

    use super::*;

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

        // Ensure we get wrapped output
        let df = ctx.read_table(Arc::new(provider)).unwrap();
        assert_eq!(
            df.schema().as_arrow(),
            &wrap_schema_maybe_deprecated(&schema)
        );
        let results = df.collect().await.unwrap();
        assert_eq!(results, vec![wrap_batch_maybe_deprecated(batch)])
    }
}
