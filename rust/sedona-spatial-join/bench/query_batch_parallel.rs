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

//! Benchmarks for query_batch probe-row parallelism.
//!
//! The uniform workload keeps candidate counts low and spread across the indexed extent.
//! The skewed workload gives many probe rows a moderate candidate count, which is the case
//! `parallel_probe_chunk_size` is intended to help when per-row refinement does not trigger.

use std::hint::black_box;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use datafusion_common::Result;
use datafusion_expr::JoinType;
use datafusion_physical_expr::expressions::Column;
use futures::Stream;
use sedona_expr::statistics::GeoStatistics;
use sedona_schema::datatypes::WKB_GEOMETRY;
use sedona_spatial_join::evaluated_batch::evaluated_batch_stream::{
    EvaluatedBatchStream, SendableEvaluatedBatchStream,
};
use sedona_spatial_join::evaluated_batch::EvaluatedBatch;
use sedona_spatial_join::index::default_spatial_index_builder::DefaultSpatialIndexBuilder;
use sedona_spatial_join::index::spatial_index::SpatialIndexRef;
use sedona_spatial_join::index::spatial_index_builder::{
    SpatialIndexBuilder, SpatialJoinBuildMetrics,
};
use sedona_spatial_join::operand_evaluator::EvaluatedGeometryArray;
use sedona_spatial_join::spatial_predicate::{RelationPredicate, SpatialRelationType};
use sedona_spatial_join::{SpatialJoinOptions, SpatialPredicate};
use sedona_testing::create::create_array;

const GRID_SIZE: usize = 100;
const PROBE_ROWS: usize = 8_192;
const BUILD_HALF_SIZE: f64 = 0.10;
const PROBE_PADDING: f64 = 0.20;
const UNIFORM_BLOCK: usize = 2;
const HOTSPOT_BLOCK: usize = 10;
const PROBE_CHUNK_SIZES: [usize; 5] = [0, 64, 256, 1024, 4096];

struct SingleBatchStream {
    batch: Option<EvaluatedBatch>,
    schema: SchemaRef,
}

impl SingleBatchStream {
    fn new(batch: EvaluatedBatch, schema: SchemaRef) -> Self {
        Self {
            batch: Some(batch),
            schema,
        }
    }
}

impl Stream for SingleBatchStream {
    type Item = Result<EvaluatedBatch>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Ready(self.batch.take().map(Ok))
    }
}

impl EvaluatedBatchStream for SingleBatchStream {
    fn is_external(&self) -> bool {
        false
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

#[derive(Clone, Copy)]
enum WorkloadKind {
    Uniform,
    Skewed,
}

impl WorkloadKind {
    fn name(self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::Skewed => "skewed",
        }
    }

    fn probe_block_size(self) -> usize {
        match self {
            Self::Uniform => UNIFORM_BLOCK,
            Self::Skewed => HOTSPOT_BLOCK,
        }
    }
}

struct QueryBatchWorkload {
    kind: WorkloadKind,
    probe_batch: Arc<EvaluatedBatch>,
    expected_count: usize,
}

fn spatial_predicate() -> SpatialPredicate {
    SpatialPredicate::Relation(RelationPredicate::new(
        Arc::new(Column::new("probe", 0)),
        Arc::new(Column::new("build", 0)),
        SpatialRelationType::Intersects,
    ))
}

fn geometry_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new(
        "geom",
        DataType::Binary,
        true,
    )]))
}

fn square_wkt(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> String {
    format!(
        "POLYGON (({min_x} {min_y}, {max_x} {min_y}, {max_x} {max_y}, {min_x} {max_y}, {min_x} {min_y}))"
    )
}

fn block_probe_wkt(x_start: usize, y_start: usize, block_size: usize) -> String {
    let min_x = x_start as f64 - PROBE_PADDING;
    let min_y = y_start as f64 - PROBE_PADDING;
    let max_x = (x_start + block_size - 1) as f64 + PROBE_PADDING;
    let max_y = (y_start + block_size - 1) as f64 + PROBE_PADDING;
    square_wkt(min_x, min_y, max_x, max_y)
}

fn build_side_wkts() -> Vec<String> {
    let mut wkts = Vec::with_capacity(GRID_SIZE * GRID_SIZE);
    for x in 0..GRID_SIZE {
        for y in 0..GRID_SIZE {
            let x = x as f64;
            let y = y as f64;
            wkts.push(square_wkt(
                x - BUILD_HALF_SIZE,
                y - BUILD_HALF_SIZE,
                x + BUILD_HALF_SIZE,
                y + BUILD_HALF_SIZE,
            ));
        }
    }
    wkts
}

fn probe_wkts(kind: WorkloadKind) -> Vec<String> {
    let block_size = kind.probe_block_size();
    let max_start = GRID_SIZE - block_size;
    let mut wkts = Vec::with_capacity(PROBE_ROWS);

    for i in 0..PROBE_ROWS {
        let (x_start, y_start) = match kind {
            WorkloadKind::Uniform => ((i * 17) % max_start, (i * 37) % max_start),
            WorkloadKind::Skewed => {
                let hotspot_x = (GRID_SIZE - block_size) / 2;
                let hotspot_y = (GRID_SIZE - block_size) / 2;
                (hotspot_x + (i % 4), hotspot_y + ((i / 4) % 4))
            }
        };
        wkts.push(block_probe_wkt(x_start, y_start, block_size));
    }

    wkts
}

fn make_evaluated_batch(wkts: &[String]) -> EvaluatedBatch {
    let schema = geometry_schema();
    let wkt_refs = wkts
        .iter()
        .map(|wkt| Some(wkt.as_str()))
        .collect::<Vec<_>>();
    let geom_array = create_array(&wkt_refs, &WKB_GEOMETRY);
    let batch = RecordBatch::try_new(schema, vec![Arc::new(geom_array.clone())])
        .expect("failed to create benchmark record batch");
    let geom_array = EvaluatedGeometryArray::try_new(geom_array, &WKB_GEOMETRY)
        .expect("failed to evaluate benchmark geometry array");
    EvaluatedBatch { batch, geom_array }
}

async fn build_index(options: SpatialJoinOptions) -> SpatialIndexRef {
    let schema = geometry_schema();
    let build_batch = make_evaluated_batch(&build_side_wkts());
    let mut builder = DefaultSpatialIndexBuilder::new(
        Arc::clone(&schema),
        spatial_predicate(),
        options,
        JoinType::Inner,
        1,
        SpatialJoinBuildMetrics::default(),
    )
    .expect("failed to create spatial index builder");

    let stream: SendableEvaluatedBatchStream =
        Box::pin(SingleBatchStream::new(build_batch, schema));
    builder
        .add_stream(stream, GeoStatistics::empty())
        .await
        .expect("failed to add benchmark build batch");
    builder.finish().expect("failed to finish benchmark index")
}

fn make_workload(kind: WorkloadKind) -> QueryBatchWorkload {
    let probe_batch = Arc::new(make_evaluated_batch(&probe_wkts(kind)));
    let expected_count = PROBE_ROWS * kind.probe_block_size() * kind.probe_block_size();
    QueryBatchWorkload {
        kind,
        probe_batch,
        expected_count,
    }
}

fn chunk_label(chunk_size: usize) -> String {
    if chunk_size == 0 {
        "off".to_string()
    } else {
        chunk_size.to_string()
    }
}

fn query_batch_parallel(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(
            std::thread::available_parallelism()
                .map(|parallelism| parallelism.get())
                .unwrap_or(8),
        )
        .build()
        .expect("failed to create benchmark runtime");

    for kind in [WorkloadKind::Uniform, WorkloadKind::Skewed] {
        let workload = make_workload(kind);
        let mut group = c.benchmark_group(format!("query_batch_parallel/{}", workload.kind.name()));
        group.throughput(Throughput::Elements(PROBE_ROWS as u64));

        for chunk_size in PROBE_CHUNK_SIZES {
            let options = SpatialJoinOptions {
                parallel_probe_chunk_size: chunk_size,
                ..Default::default()
            };
            let index = runtime.block_on(build_index(options));
            let probe_batch = Arc::clone(&workload.probe_batch);
            let expected_count = workload.expected_count;

            group.bench_with_input(
                BenchmarkId::from_parameter(chunk_label(chunk_size)),
                &chunk_size,
                |b, _| {
                    b.iter_batched(
                        || {
                            (
                                Vec::with_capacity(expected_count),
                                Vec::with_capacity(expected_count),
                            )
                        },
                        |(mut build_batch_positions, mut probe_indices)| {
                            let (metrics, next_idx) = runtime
                                .block_on(index.query_batch(
                                    &probe_batch,
                                    0..PROBE_ROWS,
                                    usize::MAX,
                                    &mut build_batch_positions,
                                    &mut probe_indices,
                                ))
                                .expect("query_batch failed in benchmark");

                            assert_eq!(next_idx, PROBE_ROWS);
                            assert_eq!(metrics.count, expected_count);
                            assert_eq!(build_batch_positions.len(), expected_count);
                            assert_eq!(probe_indices.len(), expected_count);
                            black_box(metrics.candidate_count);
                            black_box(build_batch_positions);
                            black_box(probe_indices);
                        },
                        BatchSize::SmallInput,
                    )
                },
            );
        }

        group.finish();
    }
}

criterion_group!(query_batch_parallel_group, query_batch_parallel);
criterion_main!(query_batch_parallel_group);
