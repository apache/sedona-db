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

use datafusion_common::Result;
use sedona_common::sedona_internal_err;
use sedona_geometry::bounding_box::BoundingBox;

use crate::partitioning::{SpatialPartition, SpatialPartitioner};

/// A partitioner that assigns everything to the Multi partition.
///
/// This partitioner is useful when we want to broadcast the data to all partitions.
/// Currently it is used for KNN join where regular spatial partitioning is hard because
/// it is hard to know in advance how far away a given number of neighbours will be to assign it.
#[derive(Clone)]
pub struct BroadcastPartitioner {
    num_partitions: usize,
}

impl BroadcastPartitioner {
    pub fn new(num_partitions: usize) -> Self {
        Self { num_partitions }
    }
}

impl SpatialPartitioner for BroadcastPartitioner {
    fn num_regular_partitions(&self) -> usize {
        self.num_partitions
    }

    fn partition(&self, _bbox: &BoundingBox) -> Result<SpatialPartition> {
        Ok(SpatialPartition::Multi)
    }

    fn partition_no_multi(&self, _bbox: &BoundingBox) -> Result<SpatialPartition> {
        sedona_internal_err!("BroadcastPartitioner does not support partition_no_multi")
    }

    fn box_clone(&self) -> Box<dyn SpatialPartitioner> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_partitioner() {
        let num_partitions = 4;
        let partitioner = BroadcastPartitioner::new(num_partitions);
        assert_eq!(partitioner.num_regular_partitions(), num_partitions);

        let bbox = BoundingBox::xy((0.0, 10.0), (0.0, 10.0));

        // Test partition
        assert_eq!(
            partitioner.partition(&bbox).unwrap(),
            SpatialPartition::Multi
        );

        // Test partition_no_multi
        assert!(partitioner.partition_no_multi(&bbox).is_err());
    }
}
