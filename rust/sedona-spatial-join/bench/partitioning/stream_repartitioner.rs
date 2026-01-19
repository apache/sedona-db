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

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};
use std::time::Duration;

use arrow_array::{
    ArrayRef, BinaryArray, Date32Array, Int64Array, RecordBatch, StringArray,
    TimestampMicrosecondArray,
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use datafusion::config::SpillCompression;
use datafusion_common::Result;
use datafusion_execution::runtime_env::RuntimeEnv;
use datafusion_physical_plan::metrics::{ExecutionPlanMetricsSet, SpillMetrics};
use futures::executor::block_on;
use rand::{rngs::StdRng, Rng, SeedableRng};
use sedona_geometry::{bounding_box::BoundingBox, interval::IntervalTrait};
use sedona_schema::datatypes::WKB_GEOMETRY;
use sedona_spatial_join::evaluated_batch::{
    evaluated_batch_stream::{in_mem::InMemoryEvaluatedBatchStream, SendableEvaluatedBatchStream},
    EvaluatedBatch,
};
use sedona_spatial_join::operand_evaluator::EvaluatedGeometryArray;
use sedona_spatial_join::partitioning::PartitionedSide;
use sedona_spatial_join::partitioning::{
    kdb::KDBPartitioner, stream_repartitioner::repartition_evaluated_batches, SpatialPartition,
    SpatialPartitioner,
};

const RNG_SEED: u64 = 0x5ED0_4A5;
const NUM_BATCHES: usize = 50;
const ROWS_PER_BATCH: usize = 8192;
const SAMPLE_FOR_PARTITIONER: usize = 1_000;
const MAX_ITEMS_PER_NODE: usize = 128;
const MAX_LEVELS: usize = 4;
const REPARTITIONER_BUFFER_BYTES: usize = 8 * 1024 * 1024;

fn bench_stream_partitioner(c: &mut Criterion) {
    let extent = Arc::new(default_extent());
    let partitioner = build_partitioner(extent.as_ref());
    let schema = Arc::new(build_schema());
    let runtime_env = Arc::new(RuntimeEnv::default());
    let metrics_set = ExecutionPlanMetricsSet::new();
    let spill_metrics = SpillMetrics::new(&metrics_set, 0);
    let seed_counter = Arc::new(AtomicU64::new(RNG_SEED));

    let mut group = c.benchmark_group("stream_partitioner_repartition");
    group.throughput(Throughput::Elements((NUM_BATCHES * ROWS_PER_BATCH) as u64));

    group.bench_function("kdb_repartition", |b| {
        let seed_counter = Arc::clone(&seed_counter);
        let schema = Arc::clone(&schema);
        let runtime_env = Arc::clone(&runtime_env);
        let partitioner = Arc::clone(&partitioner);
        let spill_metrics = spill_metrics.clone();
        let extent = Arc::clone(&extent);

        b.iter_batched(
            move || {
                let seed = seed_counter.fetch_add(1, Ordering::Relaxed);
                generate_stream(seed, schema.clone(), extent.as_ref())
            },
            move |stream| {
                block_on(async {
                    repartition_evaluated_batches(
                        runtime_env.clone(),
                        stream,
                        partitioner.clone(),
                        PartitionedSide::BuildSide,
                        SpillCompression::Uncompressed,
                        spill_metrics.clone(),
                        REPARTITIONER_BUFFER_BYTES,
                        ROWS_PER_BATCH,
                        None,
                    )
                    .await
                    .expect("repartition should succeed in benchmark");
                });
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn generate_stream(
    seed: u64,
    schema: Arc<Schema>,
    extent: &BoundingBox,
) -> SendableEvaluatedBatchStream {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut batches = Vec::with_capacity(NUM_BATCHES);
    for _ in 0..NUM_BATCHES {
        batches.push(random_evaluated_batch(
            schema.clone(),
            ROWS_PER_BATCH,
            extent,
            &mut rng,
        ));
    }
    in_memory_stream(schema, batches)
}

fn random_evaluated_batch(
    schema: Arc<Schema>,
    rows: usize,
    extent: &BoundingBox,
    rng: &mut StdRng,
) -> EvaluatedBatch {
    let batch = random_record_batch(schema, rows, rng);
    let geom_array = random_geometry_array(rows, extent, rng);
    let geom_array = EvaluatedGeometryArray::try_new(geom_array, &WKB_GEOMETRY)
        .expect("geometry array allocation should succeed");
    EvaluatedBatch { batch, geom_array }
}

fn random_record_batch(schema: Arc<Schema>, rows: usize, rng: &mut StdRng) -> RecordBatch {
    let ids = Int64Array::from_iter_values((0..rows).map(|_| rng.gen_range(0..1_000_000) as i64));
    let words = StringArray::from_iter_values((0..rows).map(|_| random_string(rng)));
    let dates = Date32Array::from_iter_values((0..rows).map(|_| rng.gen_range(18_000..20_000)));
    let timestamps = TimestampMicrosecondArray::from_iter_values(
        (0..rows).map(|_| rng.gen_range(1_600_000_000_000_000i64..1_700_000_000_000_000)),
    );

    let columns: Vec<ArrayRef> = vec![
        Arc::new(ids),
        Arc::new(words),
        Arc::new(dates),
        Arc::new(timestamps),
    ];

    RecordBatch::try_new(schema, columns).expect("record batch assembly should succeed")
}

fn random_geometry_array(rows: usize, extent: &BoundingBox, rng: &mut StdRng) -> ArrayRef {
    let wkbs: Vec<Vec<u8>> = (0..rows)
        .map(|_| {
            let x = rng.gen_range(extent.x().lo()..=extent.x().hi());
            let y = rng.gen_range(extent.y().lo()..=extent.y().hi());
            point_wkb(x, y)
        })
        .collect();

    let binary = BinaryArray::from_iter_values(wkbs.iter().map(|wkb| wkb.as_slice()));
    Arc::new(binary)
}

fn random_string(rng: &mut StdRng) -> String {
    const CHARSET: &[u8] = b"abcdefghijklmnopqrstuvwxyz";
    let mut buf = [0u8; 8];
    for slot in &mut buf {
        let idx = rng.gen_range(0..CHARSET.len());
        *slot = CHARSET[idx];
    }
    String::from_utf8_lossy(&buf).to_string()
}

fn build_schema() -> Schema {
    Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("category", DataType::Utf8, true),
        Field::new("event_date", DataType::Date32, true),
        Field::new(
            "event_ts",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ),
    ])
}

fn build_partitioner(extent: &BoundingBox) -> Arc<dyn SpatialPartitioner + Send + Sync> {
    let mut rng = StdRng::seed_from_u64(RNG_SEED ^ 0xFF_FFFF);
    let samples = (0..SAMPLE_FOR_PARTITIONER)
        .map(|_| random_bbox(extent, &mut rng))
        .collect::<Vec<_>>();

    let partitioner = KDBPartitioner::build(
        samples.into_iter(),
        MAX_ITEMS_PER_NODE,
        MAX_LEVELS,
        extent.clone(),
    )
    .expect("kdb builder should succeed");

    Arc::new(LockedPartitioner::new(partitioner))
}

fn random_bbox(extent: &BoundingBox, rng: &mut StdRng) -> BoundingBox {
    let span_x = (extent.x().hi() - extent.x().lo()) / 20.0;
    let span_y = (extent.y().hi() - extent.y().lo()) / 20.0;
    let width = rng.gen_range(10.0..=span_x).max(1.0);
    let height = rng.gen_range(10.0..=span_y).max(1.0);
    let min_x = rng.gen_range(extent.x().lo()..=extent.x().hi() - width);
    let min_y = rng.gen_range(extent.y().lo()..=extent.y().hi() - height);
    BoundingBox::xy((min_x, min_x + width), (min_y, min_y + height))
}

fn default_extent() -> BoundingBox {
    BoundingBox::xy((0.0, 10_000.0), (0.0, 10_000.0))
}

fn point_wkb(x: f64, y: f64) -> Vec<u8> {
    let mut buf = vec![1u8, 1, 0, 0, 0];
    buf.extend_from_slice(&x.to_le_bytes());
    buf.extend_from_slice(&y.to_le_bytes());
    buf
}

criterion_group! {
    name = stream_partitioner;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(4))
        .warm_up_time(Duration::from_secs(2));
    targets = bench_stream_partitioner
}
criterion_main!(stream_partitioner);

fn in_memory_stream(
    schema: Arc<Schema>,
    batches: Vec<EvaluatedBatch>,
) -> SendableEvaluatedBatchStream {
    Box::pin(InMemoryEvaluatedBatchStream::new(schema, batches))
}

/// Wraps [`KDBPartitioner`] in a mutex so it can satisfy `Send + Sync` for benchmarking.
struct LockedPartitioner {
    inner: Mutex<KDBPartitioner>,
}

impl LockedPartitioner {
    fn new(partitioner: KDBPartitioner) -> Self {
        Self {
            inner: Mutex::new(partitioner),
        }
    }
}

impl SpatialPartitioner for LockedPartitioner {
    fn num_regular_partitions(&self) -> usize {
        self.inner.lock().expect("mutex poisoned").num_partitions()
    }

    fn partition(&self, bbox: &BoundingBox) -> Result<SpatialPartition> {
        self.inner.lock().expect("mutex poisoned").partition(bbox)
    }

    fn partition_no_multi(&self, bbox: &BoundingBox) -> Result<SpatialPartition> {
        self.inner
            .lock()
            .expect("mutex poisoned")
            .partition_no_multi(bbox)
    }
}
