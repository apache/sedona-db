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

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{rngs::StdRng, Rng, SeedableRng};
use sedona_geometry::{bounding_box::BoundingBox, interval::IntervalTrait};
use sedona_spatial_join::partitioning::{kdb::KDBPartitioner, SpatialPartitioner};

const SAMPLE_COUNT: usize = 20_000;
const QUERY_BATCH_SIZE: usize = 1_024;
const MAX_ITEMS_PER_NODE: usize = 64;
const DEPTHS: [usize; 5] = [2, 4, 6, 8, 10];
const RNG_SEED: u64 = 0x005E_D04A; // stable dataset across runs

fn bench_partition_queries(c: &mut Criterion) {
    let extent = default_extent();
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    let samples = synthetic_bboxes(SAMPLE_COUNT, &extent, &mut rng, span_for_extent(&extent));
    let queries = synthetic_bboxes(
        QUERY_BATCH_SIZE,
        &extent,
        &mut rng,
        span_for_extent(&extent) / 2.0,
    );

    let partitioners = build_partitioners(&samples, &extent);

    let mut group = c.benchmark_group("kdb_partition_queries");
    group.throughput(Throughput::Elements(QUERY_BATCH_SIZE as u64));

    for (depth, partitioner) in &partitioners {
        for (method, no_multi) in [("partition", false), ("partition_no_multi", true)] {
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("depth-{depth}_{method}")),
                partitioner,
                |b, part: &KDBPartitioner| {
                    b.iter(|| {
                        for query in &queries {
                            let partition = if no_multi {
                                part.partition_no_multi(black_box(query))
                            } else {
                                part.partition(black_box(query))
                            };
                            let partition =
                                partition.expect("query should intersect at least one partition");
                            black_box(partition);
                        }
                    });
                },
            );
        }
    }

    group.finish();
}

fn build_partitioners(
    samples: &[BoundingBox],
    extent: &BoundingBox,
) -> Vec<(usize, KDBPartitioner)> {
    DEPTHS
        .iter()
        .map(|&depth| {
            let partitioner = KDBPartitioner::build(
                samples.iter().cloned(),
                MAX_ITEMS_PER_NODE,
                depth,
                extent.clone(),
            )
            .expect("failed to build KDB partitioner for benchmark");
            (depth, partitioner)
        })
        .collect()
}

fn synthetic_bboxes(
    count: usize,
    extent: &BoundingBox,
    rng: &mut StdRng,
    max_span: f64,
) -> Vec<BoundingBox> {
    let mut boxes = Vec::with_capacity(count);
    for _ in 0..count {
        boxes.push(random_bbox(extent, rng, max_span));
    }
    boxes
}

fn random_bbox(extent: &BoundingBox, rng: &mut StdRng, max_span: f64) -> BoundingBox {
    let (min_x, max_x) = (extent.x().lo(), extent.x().hi());
    let (min_y, max_y) = (extent.y().lo(), extent.y().hi());

    let span_x = rng.gen_range(0.01..max_span).min(max_x - min_x);
    let span_y = rng.gen_range(0.01..max_span).min(max_y - min_y);

    let start_x = rng.gen_range(min_x..=max_x - span_x);
    let start_y = rng.gen_range(min_y..=max_y - span_y);

    BoundingBox::xy((start_x, start_x + span_x), (start_y, start_y + span_y))
}

fn span_for_extent(extent: &BoundingBox) -> f64 {
    let width = extent.x().hi() - extent.x().lo();
    let height = extent.y().hi() - extent.y().lo();
    (width.min(height) / 32.0).max(0.01)
}

fn default_extent() -> BoundingBox {
    BoundingBox::xy((0.0, 10_000.0), (0.0, 10_000.0))
}

criterion_group!(kdb_partition, bench_partition_queries);
criterion_main!(kdb_partition);
