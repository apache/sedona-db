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
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use sedona_geometry::bounding_box::BoundingBox;
use sedona_spatial_join::partitioning::{rtree::RTreePartitioner, SpatialPartitioner};
const NODE_SIZES: [u16; 5] = [4, 8, 16, 32, 64]; // smaller node size => deeper tree

fn bench_rtree_partition_queries(c: &mut Criterion) {
    let extent = default_extent();
    let partitions = grid_partitions(&extent, GRID_DIM);
    let queries = sample_queries(&extent, QUERY_BATCH_SIZE);
    let partitioners = build_partitioners(&partitions);

    let mut group = c.benchmark_group("rtree_partition_queries");
    group.throughput(Throughput::Elements(QUERY_BATCH_SIZE as u64));

    for (node_size, depth, partitioner) in &partitioners {
        for (method, no_multi) in [("partition", false), ("partition_no_multi", true)] {
            group.bench_with_input(
                BenchmarkId::from_parameter(format!(
                    "node_size-{node_size}_depth-{depth}_{method}"
                )),
                partitioner,
                |b, part: &RTreePartitioner| {
                    b.iter(|| {
                        for query in &queries {
                            let partition = if no_multi {
                                part.partition_no_multi(black_box(query))
                            } else {
                                part.partition(black_box(query))
                            };
                            let partition = partition.expect("query classification failed");
                            black_box(partition);
                        }
                    });
                },
            );
        }
    }

    group.finish();
}

fn build_partitioners(boundaries: &[BoundingBox]) -> Vec<(u16, usize, RTreePartitioner)> {
    NODE_SIZES
        .iter()
        .map(|&node_size| {
            let partitioner =
                RTreePartitioner::try_new_with_node_size(boundaries.to_vec(), node_size)
                    .expect("failed to build RTree partitioner for benchmark");
            let depth = partitioner.depth();
            (node_size, depth, partitioner)
        })
        .collect()
}

criterion_group!(rtree_partition, bench_rtree_partition_queries);
criterion_main!(rtree_partition);
