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

use arrow::datatypes::{DataType, Field, Schema};
use arrow_array::RecordBatch;
use datafusion::execution::context::TaskContext;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, PlanProperties, RecordBatchStream, SendableRecordBatchStream,
};
use datafusion_common::Result as DFResult;
use futures::{Stream, StreamExt};
use sedona_spatial_join_gpu::{
    GeometryColumnInfo, GpuSpatialJoinConfig, GpuSpatialJoinExec, GpuSpatialPredicate,
    SpatialPredicate,
};
use std::any::Any;
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

/// Mock execution plan for testing
struct MockExec {
    schema: Arc<Schema>,
}

impl MockExec {
    fn new() -> Self {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("geometry", DataType::Binary, false),
        ]));
        Self { schema }
    }
}

impl fmt::Debug for MockExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MockExec")
    }
}

impl DisplayAs for MockExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "MockExec")
    }
}

impl ExecutionPlan for MockExec {
    fn name(&self) -> &str {
        "MockExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }

    fn properties(&self) -> &PlanProperties {
        unimplemented!("properties not needed for test")
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        Ok(Box::pin(MockStream {
            schema: self.schema.clone(),
        }))
    }
}

struct MockStream {
    schema: Arc<Schema>,
}

impl Stream for MockStream {
    type Item = DFResult<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Ready(None)
    }
}

impl RecordBatchStream for MockStream {
    fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }
}

#[tokio::test]
async fn test_gpu_join_exec_creation() {
    // Create simple mock execution plans as children
    let left_plan = Arc::new(MockExec::new()) as Arc<dyn ExecutionPlan>;
    let right_plan = Arc::new(MockExec::new()) as Arc<dyn ExecutionPlan>;

    // Create GPU spatial join configuration
    let config = GpuSpatialJoinConfig {
        join_type: datafusion::logical_expr::JoinType::Inner,
        left_geom_column: GeometryColumnInfo {
            name: "geometry".to_string(),
            index: 1,
        },
        right_geom_column: GeometryColumnInfo {
            name: "geometry".to_string(),
            index: 1,
        },
        predicate: GpuSpatialPredicate::Relation(SpatialPredicate::Intersects),
        device_id: 0,
        batch_size: 8192,
        additional_filters: None,
        max_memory: None,
        fallback_to_cpu: true,
    };

    // Create GPU spatial join exec
    let gpu_join = GpuSpatialJoinExec::new(left_plan, right_plan, config);
    assert!(gpu_join.is_ok(), "Failed to create GpuSpatialJoinExec");

    let gpu_join = gpu_join.unwrap();
    assert_eq!(gpu_join.children().len(), 2);
}

#[tokio::test]
async fn test_gpu_join_exec_display() {
    let left_plan = Arc::new(MockExec::new()) as Arc<dyn ExecutionPlan>;
    let right_plan = Arc::new(MockExec::new()) as Arc<dyn ExecutionPlan>;

    let config = GpuSpatialJoinConfig {
        join_type: datafusion::logical_expr::JoinType::Inner,
        left_geom_column: GeometryColumnInfo {
            name: "geometry".to_string(),
            index: 1,
        },
        right_geom_column: GeometryColumnInfo {
            name: "geometry".to_string(),
            index: 1,
        },
        predicate: GpuSpatialPredicate::Relation(SpatialPredicate::Intersects),
        device_id: 0,
        batch_size: 8192,
        additional_filters: None,
        max_memory: None,
        fallback_to_cpu: true,
    };

    let gpu_join = Arc::new(GpuSpatialJoinExec::new(left_plan, right_plan, config).unwrap());
    let display_str = format!("{:?}", gpu_join);

    assert!(display_str.contains("GpuSpatialJoinExec"));
    assert!(display_str.contains("Inner"));
}

#[tokio::test]
async fn test_gpu_join_execution_with_fallback() {
    // This test should handle GPU not being available and fallback to CPU error
    let left_plan = Arc::new(MockExec::new()) as Arc<dyn ExecutionPlan>;
    let right_plan = Arc::new(MockExec::new()) as Arc<dyn ExecutionPlan>;

    let config = GpuSpatialJoinConfig {
        join_type: datafusion::logical_expr::JoinType::Inner,
        left_geom_column: GeometryColumnInfo {
            name: "geometry".to_string(),
            index: 1,
        },
        right_geom_column: GeometryColumnInfo {
            name: "geometry".to_string(),
            index: 1,
        },
        predicate: GpuSpatialPredicate::Relation(SpatialPredicate::Intersects),
        device_id: 0,
        batch_size: 8192,
        additional_filters: None,
        max_memory: None,
        fallback_to_cpu: true,
    };

    let gpu_join = Arc::new(GpuSpatialJoinExec::new(left_plan, right_plan, config).unwrap());

    // Try to execute
    let task_context = Arc::new(TaskContext::default());
    let stream_result = gpu_join.execute(0, task_context);

    // Execution should succeed (creating the stream)
    assert!(stream_result.is_ok(), "Failed to create execution stream");

    // Now try to read from the stream
    // If GPU is not available, it should either:
    // 1. Return an error indicating fallback is needed
    // 2. Return empty results
    let mut stream = stream_result.unwrap();
    let mut batch_count = 0;
    let mut had_error = false;

    while let Some(result) = stream.next().await {
        match result {
            Ok(batch) => {
                batch_count += 1;
                // Verify schema is correct (combined left + right)
                assert_eq!(batch.schema().fields().len(), 4); // 2 from left + 2 from right
            }
            Err(e) => {
                // Expected if GPU is not available - should mention fallback
                had_error = true;
                let error_msg = e.to_string();
                assert!(
                    error_msg.contains("GPU") || error_msg.contains("fallback"),
                    "Unexpected error message: {}",
                    error_msg
                );
                break;
            }
        }
    }

    // Either we got results (GPU available) or an error (GPU not available with fallback message)
    assert!(
        batch_count > 0 || had_error,
        "Expected either results or a fallback error"
    );
}

#[tokio::test]
async fn test_gpu_join_with_empty_input() {
    // Test with empty batches (MockExec returns empty stream)
    let left_plan = Arc::new(MockExec::new()) as Arc<dyn ExecutionPlan>;
    let right_plan = Arc::new(MockExec::new()) as Arc<dyn ExecutionPlan>;

    let config = GpuSpatialJoinConfig {
        join_type: datafusion::logical_expr::JoinType::Inner,
        left_geom_column: GeometryColumnInfo {
            name: "geometry".to_string(),
            index: 1,
        },
        right_geom_column: GeometryColumnInfo {
            name: "geometry".to_string(),
            index: 1,
        },
        predicate: GpuSpatialPredicate::Relation(SpatialPredicate::Intersects),
        device_id: 0,
        batch_size: 8192,
        additional_filters: None,
        max_memory: None,
        fallback_to_cpu: true,
    };

    let gpu_join = Arc::new(GpuSpatialJoinExec::new(left_plan, right_plan, config).unwrap());

    let task_context = Arc::new(TaskContext::default());
    let stream_result = gpu_join.execute(0, task_context);
    assert!(stream_result.is_ok());

    let mut stream = stream_result.unwrap();
    let mut total_rows = 0;

    while let Some(result) = stream.next().await {
        if let Ok(batch) = result {
            total_rows += batch.num_rows();
        } else {
            // Error is acceptable if GPU is not available
            break;
        }
    }

    // Should have 0 rows (empty input produces empty output)
    assert_eq!(total_rows, 0);
}
