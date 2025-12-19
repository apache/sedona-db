use arrow::datatypes::{DataType, Field, Schema};
use arrow_array::RecordBatch;
// Add these imports to create data for the test
use arrow::array::{BinaryArray, Int32Array};
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
    properties: PlanProperties,
    batches: Vec<RecordBatch>, // Added to hold test data
}

impl MockExec {
    // Modified to accept batches
    fn new(batches: Vec<RecordBatch>) -> Self {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("geometry", DataType::Binary, false),
        ]));
        let eq_props = datafusion::physical_expr::EquivalenceProperties::new(schema.clone());
        let partitioning = datafusion::physical_plan::Partitioning::UnknownPartitioning(1);
        let properties = datafusion::physical_plan::PlanProperties::new(
            eq_props,
            partitioning,
            datafusion::physical_plan::execution_plan::EmissionType::Final,
            datafusion::physical_plan::execution_plan::Boundedness::Bounded,
        );
        Self {
            schema,
            properties,
            batches,
        }
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
        &self.properties
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
            batches: self.batches.clone().into_iter(), // Pass iterator of batches
        }))
    }
}

struct MockStream {
    schema: Arc<Schema>,
    batches: std::vec::IntoIter<RecordBatch>, // Added iterator
}

impl Stream for MockStream {
    type Item = DFResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Return next batch from the iterator
        Poll::Ready(self.batches.next().map(Ok))
    }
}

impl RecordBatchStream for MockStream {
    fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }
}

#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_gpu_join_exec_creation() {
    // Create simple mock execution plans as children
    let left_plan = Arc::new(MockExec::new(vec![])) as Arc<dyn ExecutionPlan>; // Empty input
    let right_plan = Arc::new(MockExec::new(vec![])) as Arc<dyn ExecutionPlan>;

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

#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_gpu_join_exec_display() {
    let left_plan = Arc::new(MockExec::new(vec![])) as Arc<dyn ExecutionPlan>;
    let right_plan = Arc::new(MockExec::new(vec![])) as Arc<dyn ExecutionPlan>;

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

#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_gpu_join_execution_with_fallback() {
    // This test should handle GPU not being available and fallback to CPU error
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("geometry", DataType::Binary, false),
    ]));

    // Create a dummy batch with 1 row
    let id_col = Arc::new(Int32Array::from(vec![1]));
    let geom_col = Arc::new(BinaryArray::from(vec![&b"POINT(0 0)"[..]]));
    let batch = RecordBatch::try_new(schema.clone(), vec![id_col, geom_col]).unwrap();

    // Use MockExec with data
    let left_plan = Arc::new(MockExec::new(vec![batch.clone()])) as Arc<dyn ExecutionPlan>;
    let right_plan = Arc::new(MockExec::new(vec![batch])) as Arc<dyn ExecutionPlan>;

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

#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_gpu_join_with_empty_input() {
    // Keep this test using empty input to verify behavior on empty streams if needed
    let left_plan = Arc::new(MockExec::new(vec![])) as Arc<dyn ExecutionPlan>;
    let right_plan = Arc::new(MockExec::new(vec![])) as Arc<dyn ExecutionPlan>;

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
