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
use arrow_array::{Int32Array, RecordBatch};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use datafusion::execution::context::TaskContext;
use datafusion::physical_plan::ExecutionPlan;
use futures::StreamExt;
use sedona_schema::crs::lnglat;
use sedona_schema::datatypes::{Edges, SedonaType, WKB_GEOMETRY};
use sedona_spatial_join_gpu::{
    GeometryColumnInfo, GpuSpatialJoinConfig, GpuSpatialJoinExec, GpuSpatialPredicate,
    SpatialPredicate,
};
use sedona_testing::create::create_array_storage;
use std::sync::Arc;
use tokio::runtime::Runtime;

// Helper execution plan that returns a single pre-loaded batch
struct SingleBatchExec {
    schema: Arc<Schema>,
    batch: RecordBatch,
    props: datafusion::physical_plan::PlanProperties,
}

impl SingleBatchExec {
    fn new(batch: RecordBatch) -> Self {
        let schema = batch.schema();
        let eq_props = datafusion::physical_expr::EquivalenceProperties::new(schema.clone());
        let partitioning = datafusion::physical_plan::Partitioning::UnknownPartitioning(1);
        let props = datafusion::physical_plan::PlanProperties::new(
            eq_props,
            partitioning,
            datafusion::physical_plan::execution_plan::EmissionType::Final,
            datafusion::physical_plan::execution_plan::Boundedness::Bounded,
        );
        Self {
            schema,
            batch,
            props,
        }
    }
}

impl std::fmt::Debug for SingleBatchExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SingleBatchExec")
    }
}

impl datafusion::physical_plan::DisplayAs for SingleBatchExec {
    fn fmt_as(
        &self,
        _t: datafusion::physical_plan::DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        write!(f, "SingleBatchExec")
    }
}

impl datafusion::physical_plan::ExecutionPlan for SingleBatchExec {
    fn name(&self) -> &str {
        "SingleBatchExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }

    fn properties(&self) -> &datafusion::physical_plan::PlanProperties {
        &self.props
    }

    fn children(&self) -> Vec<&Arc<dyn datafusion::physical_plan::ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn datafusion::physical_plan::ExecutionPlan>>,
    ) -> datafusion_common::Result<Arc<dyn datafusion::physical_plan::ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<datafusion::execution::context::TaskContext>,
    ) -> datafusion_common::Result<datafusion::physical_plan::SendableRecordBatchStream> {
        use datafusion::physical_plan::{RecordBatchStream, SendableRecordBatchStream};
        use futures::Stream;
        use std::pin::Pin;
        use std::task::{Context, Poll};

        struct OnceBatchStream {
            schema: Arc<Schema>,
            batch: Option<RecordBatch>,
        }

        impl Stream for OnceBatchStream {
            type Item = datafusion_common::Result<RecordBatch>;

            fn poll_next(
                mut self: Pin<&mut Self>,
                _cx: &mut Context<'_>,
            ) -> Poll<Option<Self::Item>> {
                Poll::Ready(self.batch.take().map(Ok))
            }
        }

        impl RecordBatchStream for OnceBatchStream {
            fn schema(&self) -> Arc<Schema> {
                self.schema.clone()
            }
        }

        Ok(Box::pin(OnceBatchStream {
            schema: self.schema.clone(),
            batch: Some(self.batch.clone()),
        }) as SendableRecordBatchStream)
    }
}

/// Generate random points within a bounding box
fn generate_random_points(count: usize) -> Vec<String> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| {
            let x: f64 = rng.gen_range(-180.0..180.0);
            let y: f64 = rng.gen_range(-90.0..90.0);
            format!("POINT ({} {})", x, y)
        })
        .collect()
}

/// Generate random polygons (squares) within a bounding box
fn generate_random_polygons(count: usize, size: f64) -> Vec<String> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| {
            let x: f64 = rng.gen_range(-180.0..180.0);
            let y: f64 = rng.gen_range(-90.0..90.0);
            format!(
                "POLYGON (({} {}, {} {}, {} {}, {} {}, {} {}))",
                x,
                y,
                x + size,
                y,
                x + size,
                y + size,
                x,
                y + size,
                x,
                y
            )
        })
        .collect()
}

/// Pre-created benchmark data
struct BenchmarkData {
    // For GPU benchmark
    polygon_batch: RecordBatch,
    point_batch: RecordBatch,
    // For CPU benchmark (need to keep WKT strings)
    polygon_wkts: Vec<String>,
    point_wkts: Vec<String>,
}

/// Prepare all data structures before benchmarking
fn prepare_benchmark_data(polygons: &[String], points: &[String]) -> BenchmarkData {
    // Convert WKT to Option<&str>
    let polygon_opts: Vec<Option<&str>> = polygons.iter().map(|s| Some(s.as_str())).collect();
    let point_opts: Vec<Option<&str>> = points.iter().map(|s| Some(s.as_str())).collect();

    // Create Arrow arrays from WKT (WKT -> WKB conversion happens here, NOT in benchmark)
    let polygon_array = create_array_storage(&polygon_opts, &WKB_GEOMETRY);
    let point_array = create_array_storage(&point_opts, &WKB_GEOMETRY);

    // Create RecordBatches
    let polygon_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("geometry", DataType::Binary, false),
    ]));

    let point_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("geometry", DataType::Binary, false),
    ]));

    let polygon_ids = Int32Array::from((0..polygons.len() as i32).collect::<Vec<_>>());
    let point_ids = Int32Array::from((0..points.len() as i32).collect::<Vec<_>>());

    let polygon_batch = RecordBatch::try_new(
        polygon_schema.clone(),
        vec![Arc::new(polygon_ids), polygon_array],
    )
    .unwrap();

    let point_batch =
        RecordBatch::try_new(point_schema.clone(), vec![Arc::new(point_ids), point_array]).unwrap();

    BenchmarkData {
        polygon_batch,
        point_batch,
        polygon_wkts: polygons.to_vec(),
        point_wkts: points.to_vec(),
    }
}

/// Benchmark GPU spatial join (timing only the join execution, not data preparation)
fn bench_gpu_spatial_join(rt: &Runtime, data: &BenchmarkData) -> usize {
    rt.block_on(async {
        // Create execution plans (lightweight - just wraps the pre-created batches)
        let left_plan =
            Arc::new(SingleBatchExec::new(data.polygon_batch.clone())) as Arc<dyn ExecutionPlan>;
        let right_plan =
            Arc::new(SingleBatchExec::new(data.point_batch.clone())) as Arc<dyn ExecutionPlan>;

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
            fallback_to_cpu: false,
        };

        let gpu_join = Arc::new(GpuSpatialJoinExec::new(left_plan, right_plan, config).unwrap());
        let task_context = Arc::new(TaskContext::default());
        let mut stream = gpu_join.execute(0, task_context).unwrap();

        // Collect results
        let mut total_rows = 0;
        while let Some(result) = stream.next().await {
            let batch = result.expect("GPU join failed");
            total_rows += batch.num_rows();
        }

        total_rows
    })
}

/// Benchmark CPU GEOS spatial join (timing only the join, using pre-created tester)
fn bench_cpu_spatial_join(
    data: &BenchmarkData,
    tester: &sedona_testing::testers::ScalarUdfTester,
) -> usize {
    let mut result_count = 0;

    // Nested loop join using GEOS (on WKT strings, same as GPU input)
    for poly in data.polygon_wkts.iter() {
        for point in data.point_wkts.iter() {
            let result = tester
                .invoke_scalar_scalar(poly.as_str(), point.as_str())
                .unwrap();

            if result == true.into() {
                result_count += 1;
            }
        }
    }

    result_count
}

fn benchmark_spatial_join(c: &mut Criterion) {
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_geos::register::scalar_kernels;
    use sedona_testing::testers::ScalarUdfTester;

    let rt = Runtime::new().unwrap();

    // Pre-create CPU tester (NOT timed)
    let kernels = scalar_kernels();
    let st_intersects = kernels
        .into_iter()
        .find(|(name, _)| *name == "st_intersects")
        .map(|(_, kernel_ref)| kernel_ref)
        .unwrap();

    let sedona_type = SedonaType::Wkb(Edges::Planar, lnglat());
    let udf = SedonaScalarUDF::from_kernel("st_intersects", st_intersects);
    let cpu_tester =
        ScalarUdfTester::new(udf.into(), vec![sedona_type.clone(), sedona_type.clone()]);

    let mut group = c.benchmark_group("spatial_join");
    // Reduce sample count to 10 for faster benchmarking
    group.sample_size(10);

    // Test different data sizes
    let test_sizes = vec![
        (100, 1000),   // 100 polygons, 1000 points
        (500, 5000),   // 500 polygons, 5000 points
        (1000, 10000), // 1000 polygons, 10000 points
    ];

    for (num_polygons, num_points) in test_sizes {
        let polygons = generate_random_polygons(num_polygons, 1.0);
        let points = generate_random_points(num_points);

        // Pre-create all data structures (NOT timed)
        let data = prepare_benchmark_data(&polygons, &points);

        // Benchmark GPU (only join execution is timed)
        group.bench_with_input(
            BenchmarkId::new("GPU", format!("{}x{}", num_polygons, num_points)),
            &data,
            |b, data| {
                b.iter(|| bench_gpu_spatial_join(&rt, data));
            },
        );

        // Benchmark CPU (only for smaller datasets, only join execution is timed)
        if num_polygons <= 500 {
            group.bench_with_input(
                BenchmarkId::new("CPU", format!("{}x{}", num_polygons, num_points)),
                &data,
                |b, data| {
                    b.iter(|| bench_cpu_spatial_join(data, &cpu_tester));
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, benchmark_spatial_join);
criterion_main!(benches);
