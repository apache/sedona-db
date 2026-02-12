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

//! Head-to-head benchmark of FlatPartitioner vs RTreePartitioner across
//! varying partition counts to find the optimal switch point.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use sedona_spatial_join::partitioning::{
    flat::FlatPartitioner, rtree::RTreePartitioner, SpatialPartitioner,
};
use sedona_spatial_join::utils::internal_benchmark_util::{
    default_extent, grid_partitions, sample_queries, QUERY_BATCH_SIZE,
};

/// Grid dimensions to benchmark. Each produces dim*dim partitions.
/// 4x4=16, 5x5=25, 6x6=36, 8x8=64, 10x10=100, 16x16=256, 20x20=400
const GRID_DIMS: [usize; 7] = [4, 5, 6, 8, 10, 16, 20];

fn bench_flat_vs_rtree(c: &mut Criterion) {
    let extent = default_extent();

    let mut group = c.benchmark_group("flat_vs_rtree");
    group.throughput(Throughput::Elements(QUERY_BATCH_SIZE as u64));

    for &dim in &GRID_DIMS {
        let num_partitions = dim * dim;
        let partitions = grid_partitions(&extent, dim);
        let queries = sample_queries(&extent, QUERY_BATCH_SIZE);

        let flat =
            FlatPartitioner::try_new(partitions.clone()).expect("failed to build FlatPartitioner");
        let rtree =
            RTreePartitioner::try_new(partitions).expect("failed to build RTreePartitioner");

        group.bench_with_input(
            BenchmarkId::new("flat", num_partitions),
            &flat,
            |b, partitioner| {
                b.iter(|| {
                    for query in &queries {
                        let result = partitioner
                            .partition(black_box(query))
                            .expect("partition failed");
                        black_box(result);
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("rtree", num_partitions),
            &rtree,
            |b, partitioner| {
                b.iter(|| {
                    for query in &queries {
                        let result = partitioner
                            .partition(black_box(query))
                            .expect("partition failed");
                        black_box(result);
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(flat_vs_rtree, bench_flat_vs_rtree);
criterion_main!(flat_vs_rtree);
