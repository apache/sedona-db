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

//! Flat (linear scan) spatial partitioner.
//!
//! This module provides a minimal partitioner that shares the same
//! intersection semantics as [`crate::partitioning::rtree::RTreePartitioner`]
//! but avoids the RTree indexing overhead. It stores partition boundaries
//! in a flat array and performs a linear scan to classify each query
//! bounding box. [`FlatPartitioner`] will definitely be more efficient
//! than [`crate::partitioning::rtree::RTreePartitioner`] when the number of
//! partitions is less than 16, which is the size of R-tree's leaf nodes.
//!
//! The partitioner follows the standard spatial partition semantics:
//! - Returns [`SpatialPartition::Regular`] when exactly one boundary
//!   intersects the query bbox.
//! - Returns [`SpatialPartition::Multi`] when multiple boundaries
//!   intersect the query bbox.
//! - Returns [`SpatialPartition::None`] when no boundary intersects
//!   the query bbox.

use datafusion_common::Result;
use sedona_geometry::bounding_box::BoundingBox;
use sedona_geometry::interval::IntervalTrait;

use crate::partitioning::{SpatialPartition, SpatialPartitioner};

/// Spatial partitioner that linearly scans partition boundaries.
pub struct FlatPartitioner {
    boundaries: Vec<BoundingBox>,
}

impl FlatPartitioner {
    /// Create a new flat partitioner from explicit partition boundaries.
    pub fn try_new(boundaries: Vec<BoundingBox>) -> Result<Self> {
        Ok(Self { boundaries })
    }
}

impl SpatialPartitioner for FlatPartitioner {
    fn num_regular_partitions(&self) -> usize {
        self.boundaries.len()
    }

    fn partition(&self, bbox: &BoundingBox) -> Result<SpatialPartition> {
        let mut first_match = None;
        for (idx, boundary) in self.boundaries.iter().enumerate() {
            if boundary.intersects(bbox) {
                if first_match.is_some() {
                    return Ok(SpatialPartition::Multi);
                }
                first_match = Some(idx as u32);
            }
        }

        Ok(match first_match {
            Some(id) => SpatialPartition::Regular(id),
            None => SpatialPartition::None,
        })
    }

    fn partition_no_multi(&self, bbox: &BoundingBox) -> Result<SpatialPartition> {
        let mut best_partition = None;
        let mut best_area = -1.0;

        for (idx, boundary) in self.boundaries.iter().enumerate() {
            if boundary.intersects(bbox) {
                let area = {
                    if let Ok(intersection) = boundary.intersection(bbox) {
                        if !intersection.x().is_wraparound() {
                            intersection.x().width() * intersection.y().width()
                        } else {
                            // Intersection has a wraparound X interval. Use a fallback
                            // area value of 0. This makes the partitioner prefer other partitions.
                            0.0
                        }
                    } else {
                        // Intersection cannot be represented as a single bbox. Use a fallback
                        // area value of 0. This makes the partitioner prefer other partitions.
                        0.0
                    }
                };
                if area > best_area {
                    best_area = area;
                    best_partition = Some(idx as u32);
                }
            }
        }

        Ok(match best_partition {
            Some(id) => SpatialPartition::Regular(id),
            None => SpatialPartition::None,
        })
    }
}

#[cfg(test)]
mod tests {
    use sedona_geometry::interval::{Interval, IntervalTrait};

    use super::*;

    fn sample_partitions() -> Vec<BoundingBox> {
        vec![
            BoundingBox::xy((0.0, 50.0), (0.0, 50.0)),
            BoundingBox::xy((50.0, 100.0), (0.0, 50.0)),
            BoundingBox::xy((0.0, 50.0), (50.0, 100.0)),
            BoundingBox::xy((50.0, 100.0), (50.0, 100.0)),
        ]
    }

    #[test]
    fn test_flat_partitioner_creation() {
        let partitioner = FlatPartitioner::try_new(sample_partitions()).unwrap();
        assert_eq!(partitioner.num_regular_partitions(), 4);
    }

    #[test]
    fn test_flat_partitioner_regular() {
        let partitioner = FlatPartitioner::try_new(sample_partitions()).unwrap();
        let bbox = BoundingBox::xy((10.0, 20.0), (10.0, 20.0));
        assert_eq!(
            partitioner.partition(&bbox).unwrap(),
            SpatialPartition::Regular(0)
        );
    }

    #[test]
    fn test_flat_partitioner_multi() {
        let partitioner = FlatPartitioner::try_new(sample_partitions()).unwrap();
        let bbox = BoundingBox::xy((45.0, 55.0), (10.0, 20.0));
        assert_eq!(
            partitioner.partition(&bbox).unwrap(),
            SpatialPartition::Multi
        );
    }

    #[test]
    fn test_flat_partitioner_no_dup_prefers_largest_overlap() {
        let partitioner = FlatPartitioner::try_new(sample_partitions()).unwrap();
        let bbox = BoundingBox::xy((45.0, 80.0), (10.0, 20.0));
        match partitioner.partition_no_multi(&bbox).unwrap() {
            SpatialPartition::Regular(id) => assert_eq!(id, 1),
            _ => panic!("expected Regular partition"),
        }
    }

    #[test]
    fn test_flat_partitioner_none() {
        let partitioner = FlatPartitioner::try_new(sample_partitions()).unwrap();
        let bbox = BoundingBox::xy((200.0, 250.0), (200.0, 250.0));
        assert_eq!(
            partitioner.partition(&bbox).unwrap(),
            SpatialPartition::None
        );
    }

    #[test]
    fn test_flat_partitioner_empty_bbox_build() {
        let boundaries = vec![
            BoundingBox::xy((0.0, 50.0), (0.0, 50.0)),
            BoundingBox::xy(Interval::empty(), Interval::empty()),
            BoundingBox::xy((50.0, 100.0), (0.0, 50.0)),
        ];
        let partitioner = FlatPartitioner::try_new(boundaries).unwrap();
        assert_eq!(partitioner.num_regular_partitions(), 3);

        // Verify that the third partition (index 2) is correctly mapped
        let query = BoundingBox::xy((60.0, 70.0), (10.0, 20.0));
        assert_eq!(
            partitioner.partition(&query).unwrap(),
            SpatialPartition::Regular(2)
        );
    }

    #[test]
    fn test_flat_partitioner_empty_bbox_query() {
        let boundaries = vec![BoundingBox::xy((0.0, 50.0), (0.0, 50.0))];
        let partitioner = FlatPartitioner::try_new(boundaries).unwrap();

        let bbox = BoundingBox::xy(Interval::empty(), Interval::empty());
        assert_eq!(
            partitioner.partition(&bbox).unwrap(),
            SpatialPartition::None
        );
        assert_eq!(
            partitioner.partition_no_multi(&bbox).unwrap(),
            SpatialPartition::None
        );
    }
}
