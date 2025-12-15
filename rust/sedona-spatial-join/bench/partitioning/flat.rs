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

mod common;

use std::hint::black_box;

use common::{default_extent, grid_partitions, sample_queries, GRID_DIM, QUERY_BATCH_SIZE};
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use sedona_spatial_join::partitioning::{flat::FlatPartitioner, SpatialPartitioner};

fn bench_flat_partition_queries(c: &mut Criterion) {
    let extent = default_extent();
    let partitions = grid_partitions(&extent, GRID_DIM);
    let queries = sample_queries(&extent, QUERY_BATCH_SIZE);
    let partitioner = FlatPartitioner::try_new(partitions)
        .expect("failed to build Flat partitioner for benchmark");

    let mut group = c.benchmark_group("flat_partition_queries");
    group.throughput(Throughput::Elements(QUERY_BATCH_SIZE as u64));

    for (label, no_multi) in [("partition", false), ("partition_no_multi", true)] {
        group.bench_function(format!("linear_scan_{label}"), |b| {
            b.iter(|| {
                for query in &queries {
                    let partition = if no_multi {
                        partitioner.partition_no_multi(black_box(query))
                    } else {
                        partitioner.partition(black_box(query))
                    };
                    let partition = partition.expect("query classification failed");
                    black_box(partition);
                }
            });
        });
    }

    group.finish();
}

criterion_group!(flat_partition, bench_flat_partition_queries);
criterion_main!(flat_partition);
