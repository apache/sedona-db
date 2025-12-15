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

//! RTree-based spatial partitioning implementation.
//!
//! This module provides an RTree-based implementation for spatial partitioning,
//! designed for out-of-core spatial joins. Unlike KDB-tree partitioning which
//! builds the partition structure from sample data, RTree partitioning takes
//! a pre-defined set of partition boundaries (rectangles) and uses an RTree
//! index to efficiently determine which partition a given bounding box belongs to.
//!
//! # Partitioning Strategy
//!
//! The RTree partitioner follows these rules:
//!
//! 1. **Single Partition Assignment**: Each bounding box is assigned to exactly one partition.
//! 2. **Intersection-Based Assignment**: The partitioner finds which partition boundaries
//!    intersect with the input bounding box and produces a [`SpatialPartition::Regular`] result.
//! 3. **Multi-partition Handling**: When a bbox intersects multiple partitions, it's assigned
//!    to [`SpatialPartition::Multi`].
//! 4. **None-partition Handling**: If a bbox doesn't intersect any partition boundary, it's assigned
//!    to [`SpatialPartition::None`].

use datafusion_common::Result;
use geo::Rect;
use geo_index::rtree::{sort::HilbertSort, RTree, RTreeBuilder, RTreeIndex};
use sedona_geometry::bounding_box::BoundingBox;

use crate::partitioning::util::{
    bbox_to_f32_rect, bbox_to_geo_rect, make_rect, rect_intersection_area,
};
use crate::partitioning::{SpatialPartition, SpatialPartitioner};

/// RTree-based spatial partitioner that uses pre-defined partition boundaries.
///
/// This partitioner constructs an RTree index over a set of partition boundaries
/// (rectangles) and uses it to efficiently determine which partition a given
/// bounding box belongs to based on spatial intersection.
pub struct RTreePartitioner {
    /// The RTree index storing partition boundaries as f32 rectangles
    rtree: RTree<f32>,
    /// Flat representation of partition boundaries for overlap calculations
    boundaries: Vec<Rect<f32>>,
    /// Number of partitions (excluding None and Multi)
    num_partitions: usize,
    /// Map from RTree index to original partition index
    partition_map: Vec<usize>,
}

impl RTreePartitioner {
    /// Create a new RTree partitioner from a collection of partition boundaries.
    ///
    /// # Arguments
    /// * `boundaries` - A vector of bounding boxes representing partition boundaries.
    ///   Each bounding box defines the spatial extent of one partition. The partition ID
    ///   is the index in this vector.
    ///
    /// # Errors
    /// Returns an error if any boundary has wraparound coordinates (not supported)
    ///
    /// # Example
    /// ```rust
    /// use sedona_geometry::bounding_box::BoundingBox;
    /// use sedona_spatial_join::partitioning::rtree::RTreePartitioner;
    ///
    /// let boundaries = vec![
    ///     BoundingBox::xy((0.0, 50.0), (0.0, 50.0)),
    ///     BoundingBox::xy((50.0, 100.0), (0.0, 50.0)),
    /// ];
    /// let partitioner = RTreePartitioner::try_new(boundaries).unwrap();
    /// ```
    pub fn try_new(boundaries: Vec<BoundingBox>) -> Result<Self> {
        Self::build(boundaries, None)
    }

    /// Create a new RTree partitioner with a custom node size.
    pub fn try_new_with_node_size(boundaries: Vec<BoundingBox>, node_size: u16) -> Result<Self> {
        Self::build(boundaries, Some(node_size))
    }

    fn build(boundaries: Vec<BoundingBox>, node_size: Option<u16>) -> Result<Self> {
        let num_partitions = boundaries.len();

        // Filter valid boundaries and keep track of original indices
        let mut valid_boundaries = Vec::with_capacity(num_partitions);
        let mut partition_map = Vec::with_capacity(num_partitions);

        for (i, bbox) in boundaries.iter().enumerate() {
            if let Some(rect) = bbox_to_f32_rect(bbox)? {
                valid_boundaries.push(rect);
                partition_map.push(i);
            }
        }

        let num_valid = valid_boundaries.len();

        // Build RTree index with partition boundaries
        let mut rtree_builder = match node_size {
            Some(size) => RTreeBuilder::<f32>::new_with_node_size(num_valid as u32, size),
            None => RTreeBuilder::<f32>::new(num_valid as u32),
        };

        let mut rects = Vec::with_capacity(num_valid);
        for (min_x, min_y, max_x, max_y) in valid_boundaries {
            rtree_builder.add(min_x, min_y, max_x, max_y);
            rects.push(make_rect(min_x, min_y, max_x, max_y));
        }

        let rtree = rtree_builder.finish::<HilbertSort>();

        Ok(RTreePartitioner {
            rtree,
            boundaries: rects,
            num_partitions,
            partition_map,
        })
    }

    /// Return the number of levels in the underlying RTree.
    pub fn depth(&self) -> usize {
        self.rtree.num_levels()
    }
}

impl SpatialPartitioner for RTreePartitioner {
    fn num_regular_partitions(&self) -> usize {
        self.num_partitions
    }

    fn partition(&self, bbox: &BoundingBox) -> Result<SpatialPartition> {
        // Convert bbox to f32 for RTree query with proper bounds handling
        let (min_x, min_y, max_x, max_y) = match bbox_to_f32_rect(bbox)? {
            Some(rect) => rect,
            None => return Ok(SpatialPartition::None),
        };

        // Query RTree for intersecting partitions
        let intersecting_partitions = self.rtree.search(min_x, min_y, max_x, max_y);

        // Handle different cases based on number of intersecting partitions
        match intersecting_partitions.len() {
            0 => {
                // No intersection with any partition -> None
                Ok(SpatialPartition::None)
            }
            1 => {
                // Single intersection -> Regular partition
                let rtree_index = intersecting_partitions[0];
                Ok(SpatialPartition::Regular(
                    self.partition_map[rtree_index as usize] as u32,
                ))
            }
            _ => {
                // Multiple intersections -> Always return Multi
                Ok(SpatialPartition::Multi)
            }
        }
    }

    fn partition_no_multi(&self, bbox: &BoundingBox) -> Result<SpatialPartition> {
        let rect = match bbox_to_geo_rect(bbox)? {
            Some(rect) => rect,
            None => return Ok(SpatialPartition::None),
        };
        let min = rect.min();
        let max = rect.max();
        let intersecting_partitions = self.rtree.search(min.x, min.y, max.x, max.y);

        if intersecting_partitions.is_empty() {
            return Ok(SpatialPartition::None);
        }

        let mut best_partition = None;
        let mut best_area = -1.0_f32;
        for &partition_id in &intersecting_partitions {
            let boundary = &self.boundaries[partition_id as usize];
            let area = rect_intersection_area(boundary, &rect);
            if area > best_area {
                best_area = area;
                best_partition = Some(partition_id);
            }
        }

        Ok(match best_partition {
            Some(id) => SpatialPartition::Regular(self.partition_map[id as usize] as u32),
            None => SpatialPartition::None,
        })
    }
}

#[cfg(test)]
mod tests {
    use sedona_geometry::interval::{Interval, IntervalTrait};

    use super::*;

    #[test]
    fn test_rtree_partitioner_creation() {
        let boundaries = vec![
            BoundingBox::xy((0.0, 50.0), (0.0, 50.0)),
            BoundingBox::xy((50.0, 100.0), (0.0, 50.0)),
        ];
        let partitioner = RTreePartitioner::try_new(boundaries).unwrap();
        assert_eq!(partitioner.num_regular_partitions(), 2);
    }

    #[test]
    fn test_rtree_partitioner_empty_boundaries() {
        let boundaries: Vec<BoundingBox> = vec![];
        let partitioner = RTreePartitioner::try_new(boundaries).unwrap();
        assert_eq!(partitioner.num_regular_partitions(), 0);
    }

    #[test]
    fn test_partition_with_empty_boundaries_returns_none() {
        let boundaries: Vec<BoundingBox> = vec![];
        let partitioner = RTreePartitioner::try_new(boundaries).unwrap();

        let bbox = BoundingBox::xy((0.0, 1.0), (0.0, 1.0));
        let result = partitioner.partition(&bbox).unwrap();
        assert_eq!(result, SpatialPartition::None);
    }

    #[test]
    fn test_rtree_partitioner_wraparound_boundary() {
        // Create a boundary with wraparound (crossing anti-meridian)
        let boundaries = vec![
            BoundingBox::xy((170.0, -170.0), (0.0, 50.0)), // This represents wraparound
        ];
        let result = RTreePartitioner::try_new(boundaries);
        assert!(result.is_err());
    }

    #[test]
    fn test_partition_single_intersection() {
        let boundaries = vec![
            BoundingBox::xy((0.0, 50.0), (0.0, 50.0)),   // Partition 0
            BoundingBox::xy((50.0, 100.0), (0.0, 50.0)), // Partition 1
            BoundingBox::xy((0.0, 50.0), (50.0, 100.0)), // Partition 2
            BoundingBox::xy((50.0, 100.0), (50.0, 100.0)), // Partition 3
        ];
        let partitioner = RTreePartitioner::try_new(boundaries).unwrap();

        // Test bbox that falls entirely in partition 0
        let bbox = BoundingBox::xy((10.0, 20.0), (10.0, 20.0));
        let result = partitioner.partition(&bbox).unwrap();
        assert_eq!(result, SpatialPartition::Regular(0));

        // Test bbox that falls entirely in partition 3
        let bbox = BoundingBox::xy((60.0, 80.0), (60.0, 80.0));
        let result = partitioner.partition(&bbox).unwrap();
        assert_eq!(result, SpatialPartition::Regular(3));
    }

    #[test]
    fn test_partition_multiple_intersections() {
        let boundaries = vec![
            BoundingBox::xy((0.0, 50.0), (0.0, 50.0)),   // Partition 0
            BoundingBox::xy((50.0, 100.0), (0.0, 50.0)), // Partition 1
        ];
        let partitioner = RTreePartitioner::try_new(boundaries).unwrap();

        // Test bbox that spans both partitions - should return Multi
        let bbox = BoundingBox::xy((30.0, 60.0), (10.0, 20.0));
        let result = partitioner.partition(&bbox).unwrap();
        assert_eq!(result, SpatialPartition::Multi);

        assert!(matches!(
            partitioner.partition_no_multi(&bbox).unwrap(),
            SpatialPartition::Regular(_)
        ));
    }

    #[test]
    fn test_partition_none() {
        let boundaries = vec![
            BoundingBox::xy((0.0, 50.0), (0.0, 50.0)),   // Partition 0
            BoundingBox::xy((50.0, 100.0), (0.0, 50.0)), // Partition 1
        ];
        let partitioner = RTreePartitioner::try_new(boundaries).unwrap();

        // Test bbox that doesn't intersect any partition
        let bbox = BoundingBox::xy((200.0, 300.0), (200.0, 300.0));
        let result = partitioner.partition(&bbox).unwrap();
        assert_eq!(result, SpatialPartition::None);
    }

    #[test]
    fn test_partition_wraparound_input() {
        let boundaries = vec![BoundingBox::xy((0.0, 50.0), (0.0, 50.0))];
        let partitioner = RTreePartitioner::try_new(boundaries).unwrap();

        // Test with wraparound input bbox (should be rejected)
        let bbox = BoundingBox::xy((170.0, -170.0), (10.0, 20.0));
        let result = partitioner.partition(&bbox);
        assert!(result.is_err());
    }

    #[test]
    fn test_num_partitions() {
        let boundaries = vec![
            BoundingBox::xy((0.0, 50.0), (0.0, 50.0)),
            BoundingBox::xy((50.0, 100.0), (0.0, 50.0)),
            BoundingBox::xy((0.0, 50.0), (50.0, 100.0)),
        ];
        let partitioner = RTreePartitioner::try_new(boundaries).unwrap();
        assert_eq!(partitioner.num_regular_partitions(), 3);
    }

    #[test]
    fn test_partition_boundary_edge() {
        let boundaries = vec![
            BoundingBox::xy((0.0, 50.0), (0.0, 50.0)),
            BoundingBox::xy((50.0, 100.0), (0.0, 50.0)),
        ];
        let partitioner = RTreePartitioner::try_new(boundaries).unwrap();

        // Test bbox exactly on the boundary between two partitions
        let bbox = BoundingBox::xy((45.0, 55.0), (10.0, 20.0));
        let result = partitioner.partition(&bbox).unwrap();
        // Should be Multi since it spans multiple partitions
        assert_eq!(result, SpatialPartition::Multi);
    }

    #[test]
    fn test_rtree_partitioner_empty_bbox_build() {
        let boundaries = vec![
            BoundingBox::xy((0.0, 50.0), (0.0, 50.0)),
            BoundingBox::xy(Interval::empty(), Interval::empty()),
            BoundingBox::xy((50.0, 100.0), (0.0, 50.0)),
        ];
        let partitioner = RTreePartitioner::try_new(boundaries).unwrap();
        assert_eq!(partitioner.num_regular_partitions(), 3);

        // Verify that the third partition (index 2) is correctly mapped
        let query = BoundingBox::xy((60.0, 70.0), (10.0, 20.0));
        assert_eq!(
            partitioner.partition(&query).unwrap(),
            SpatialPartition::Regular(2)
        );
    }

    #[test]
    fn test_rtree_partitioner_empty_bbox_query() {
        let boundaries = vec![BoundingBox::xy((0.0, 50.0), (0.0, 50.0))];
        let partitioner = RTreePartitioner::try_new(boundaries).unwrap();

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
