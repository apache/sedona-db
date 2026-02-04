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

use common::{default_extent, sample_queries, QUERY_BATCH_SIZE};
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use sedona_spatial_join::partitioning::{broadcast::BroadcastPartitioner, SpatialPartitioner};

fn bench_broadcast_partition_queries(c: &mut Criterion) {
    let extent = default_extent();
    let queries = sample_queries(&extent, QUERY_BATCH_SIZE);
    let partitioner = BroadcastPartitioner::new(100);

    let mut group = c.benchmark_group("broadcast_partition_queries");
    group.throughput(Throughput::Elements(QUERY_BATCH_SIZE as u64));

    group.bench_function("broadcast_partition", |b| {
        b.iter(|| {
            for query in &queries {
                let partition = partitioner.partition(black_box(query));
                let partition = partition.expect("query classification failed");
                black_box(partition);
            }
        });
    });
}

criterion_group!(benches, bench_broadcast_partition_queries);
criterion_main!(benches);
