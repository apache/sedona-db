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

use std::sync::atomic::{AtomicUsize, Ordering};

use datafusion_common::Result;
use sedona_geometry::bounding_box::BoundingBox;

use crate::partitioning::{SpatialPartition, SpatialPartitioner};

/// A partitioner that assigns partitions in a round-robin fashion.
///
/// This partitioner is used for KNN join, where the build side is partitioned
/// into `num_partitions` partitions, and the probe side is assigned to the
/// `Multi` partition (i.e., broadcast to all partitions).
pub struct RoundRobinPartitioner {
    num_partitions: usize,
    counter: AtomicUsize,
}

impl RoundRobinPartitioner {
    pub fn new(num_partitions: usize) -> Self {
        Self {
            num_partitions,
            counter: AtomicUsize::new(0),
        }
    }
}

impl SpatialPartitioner for RoundRobinPartitioner {
    fn num_regular_partitions(&self) -> usize {
        self.num_partitions
    }

    fn partition(&self, bbox: &BoundingBox) -> Result<SpatialPartition> {
        self.partition_no_multi(bbox)
    }

    fn partition_no_multi(&self, _bbox: &BoundingBox) -> Result<SpatialPartition> {
        let idx = self.counter.fetch_add(1, Ordering::Relaxed);
        Ok(SpatialPartition::Regular(
            (idx % self.num_partitions) as u32,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_robin_partitioner() {
        let num_partitions = 4;
        let partitioner = RoundRobinPartitioner::new(num_partitions);
        assert_eq!(partitioner.num_regular_partitions(), num_partitions);

        let bbox = BoundingBox::xy((0.0, 10.0), (0.0, 10.0));

        for i in 0..10 {
            assert_eq!(
                partitioner.partition_no_multi(&bbox).unwrap(),
                SpatialPartition::Regular((i % num_partitions) as u32)
            );
        }
    }
}
