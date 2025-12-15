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

//! KDB-tree spatial partitioning implementation.
//!
//! This module provides a K-D-B tree (K-Dimensional B-tree) implementation for spatial
//! partitioning, which is essential for out-of-core spatial joins.
//!
//! # Partitioning Strategy
//!
//! - **Partition Assignment**:
//!   - `partition()`: Returns `SpatialPartition::Regular(id)` if the bbox intersects exactly one partition,
//!     `SpatialPartition::Multi` if it intersects multiple, or `SpatialPartition::None` if it intersects none.
//!   - `partition_no_multi()`: Assigns the bbox to the partition with the largest intersection area
//!     (Maximum Overlap Strategy), or `SpatialPartition::None` if no intersection.
//!
//! # Algorithm
//!
//! The KDB tree partitions space by recursively splitting it along alternating axes:
//!
//! 1. Start with the full spatial extent
//! 2. When a node exceeds `max_items_per_node`, split it:
//!    - Choose the longer dimension (x or y)
//!    - Sort items by their minimum coordinate on that dimension
//!    - Split at the median item's coordinate
//! 3. Continue until reaching `max_levels` depth or items fit in nodes
//! 4. Assign sequential IDs to leaf nodes

use std::sync::Arc;

use crate::partitioning::{
    util::{bbox_to_geo_rect, rect_contains_point, rect_intersection_area, rects_intersect},
    SpatialPartition, SpatialPartitioner,
};
use datafusion_common::Result;
use geo::{Coord, Rect};
use sedona_common::sedona_internal_err;
use sedona_geometry::bounding_box::BoundingBox;

/// K-D-B tree spatial partitioner implementation.
///
/// See https://en.wikipedia.org/wiki/K-D-B-tree
///
/// The KDB tree is a hierarchical spatial data structure that recursively
/// partitions space using axis-aligned splits. It adapts to the spatial
/// distribution of data, making it effective for spatial partitioning.
///
/// # Example
///
/// ```
/// use sedona_geometry::bounding_box::BoundingBox;
/// use sedona_spatial_join::partitioning::kdb::KDBPartitioner;
/// use sedona_spatial_join::partitioning::{SpatialPartitioner, SpatialPartition};
///
/// // Create sample bounding boxes
/// let bboxes = (0..20).map(|i| {
///     let x = (i % 10) as f64 * 10.0;
///     let y = (i / 10) as f64 * 10.0;
///     BoundingBox::xy((x, x + 5.0), (y, y + 5.0))
/// });
///
/// // Build the KDB partitioner
/// let extent = BoundingBox::xy((0.0, 100.0), (0.0, 100.0));
/// let partitioner = KDBPartitioner::build(bboxes, 5, 3, extent)
///     .expect("failed to build KDB partitioner");
///
/// // Query which partition a bounding box belongs to
/// let test_bbox = BoundingBox::xy((1.0, 4.0), (1.0, 4.0));
/// match partitioner.partition(&test_bbox).unwrap() {
///     SpatialPartition::Regular(id) => println!("Partition ID: {}", id),
///     _ => unreachable!(),
/// }
/// ```
pub(crate) struct KDBTree {
    max_items_per_node: usize,
    max_levels: usize,
    extent: Rect<f32>,
    level: usize,
    items: Vec<Rect<f32>>,
    children: Option<Box<[KDBTree; 2]>>,
    leaf_id: u32,
}

#[cfg(test)]
pub(crate) struct KDBResult {
    extent: Rect<f32>,
    leaf_id: u32,
}

#[cfg(test)]
impl KDBResult {
    fn new(kdb: &KDBTree) -> Self {
        Self {
            extent: kdb.extent,
            leaf_id: kdb.leaf_id,
        }
    }
}

impl KDBTree {
    /// Create a new KDB tree with the given parameters.
    ///
    /// # Arguments
    /// * `max_items_per_node` - Maximum number of items before splitting a node
    /// * `max_levels` - Maximum depth of the tree
    /// * `extent` - The spatial extent covered by this tree
    pub fn try_new(
        max_items_per_node: usize,
        max_levels: usize,
        extent: BoundingBox,
    ) -> Result<Self> {
        if max_items_per_node == 0 {
            return sedona_internal_err!("max_items_per_node must be greater than 0");
        }
        let Some(extent_rect) = bbox_to_geo_rect(&extent)? else {
            return sedona_internal_err!("KDBTree extent cannot be empty");
        };
        Ok(Self::new_with_level(
            max_items_per_node,
            max_levels,
            0,
            extent_rect,
        ))
    }

    fn new_with_level(
        max_items_per_node: usize,
        max_levels: usize,
        level: usize,
        extent: Rect<f32>,
    ) -> Self {
        KDBTree {
            max_items_per_node,
            max_levels,
            extent,
            level,
            items: Vec::new(),
            children: None,
            leaf_id: 0,
        }
    }

    /// Insert a bounding box into the tree.
    pub fn insert(&mut self, bbox: BoundingBox) -> Result<()> {
        if let Some(rect) = bbox_to_geo_rect(&bbox)? {
            if rect_contains_point(&self.extent, &rect.min()) {
                self.insert_rect(rect);
            }
        }
        Ok(())
    }

    fn insert_rect(&mut self, rect: Rect<f32>) {
        if self.items.len() < self.max_items_per_node || self.level >= self.max_levels {
            self.items.push(rect);
        } else {
            if self.children.is_none() {
                // Split over longer side
                let split_x = self.extent.width() > self.extent.height();
                let mut ok = self.split(split_x);
                if !ok {
                    // Try splitting by the other side
                    ok = self.split(!split_x);
                }

                if !ok {
                    // This could happen if all envelopes are the same.
                    self.items.push(rect);
                    return;
                }
            }

            // Insert into appropriate child
            if let Some(ref mut children) = self.children {
                let min_point = rect.min();
                for child in children.iter_mut() {
                    if rect_contains_point(child.extent(), &min_point) {
                        child.insert_rect(rect);
                        break;
                    }
                }
            }
        }
    }

    /// Check if this node is a leaf node.
    pub fn is_leaf(&self) -> bool {
        self.children.is_none()
    }

    /// Get the leaf ID (only valid for leaf nodes).
    pub fn leaf_id(&self) -> u32 {
        assert!(self.is_leaf(), "leaf_id() called on non-leaf node");
        self.leaf_id
    }

    /// Get the spatial extent of this node.
    pub fn extent(&self) -> &Rect<f32> {
        &self.extent
    }

    /// Assign leaf IDs to all leaf nodes in the tree (breadth-first traversal).
    pub fn assign_leaf_ids(&mut self) {
        let mut next_id = 0;
        self.assign_leaf_ids_recursive(&mut next_id);
    }

    fn assign_leaf_ids_recursive(&mut self, next_id: &mut u32) {
        if self.is_leaf() {
            self.leaf_id = *next_id;
            *next_id += 1;
        } else if let Some(ref mut children) = self.children {
            for child in children.iter_mut() {
                child.assign_leaf_ids_recursive(next_id);
            }
        }
    }

    /// Find all leaf nodes that intersect with the given bounding box.
    #[cfg(test)]
    pub fn find_leaf_nodes(&self, rect: &Rect<f32>, matches: &mut Vec<KDBResult>) {
        self.visit_intersecting_leaf_nodes(rect, &mut |kdb| {
            matches.push(KDBResult::new(kdb));
        });
    }

    pub fn visit_intersecting_leaf_nodes<'a>(
        &'a self,
        rect: &Rect<f32>,
        f: &mut impl FnMut(&'a KDBTree),
    ) {
        if !rects_intersect(&self.extent, rect) {
            return;
        }

        if self.is_leaf() {
            f(self)
        } else if let Some(ref children) = self.children {
            for child in children.iter() {
                child.visit_intersecting_leaf_nodes(rect, f);
            }
        }
    }

    pub fn visit_leaf_nodes<'a>(&'a self, f: &mut impl FnMut(&'a KDBTree)) {
        if self.is_leaf() {
            f(self)
        } else if let Some(ref children) = self.children {
            for child in children.iter() {
                child.visit_leaf_nodes(f);
            }
        }
    }

    #[cfg(test)]
    pub fn collect_leaf_nodes(&self) -> Vec<&KDBTree> {
        let mut leaf_nodes = Vec::new();
        self.visit_leaf_nodes(&mut |kdb| {
            leaf_nodes.push(kdb);
        });
        leaf_nodes
    }

    pub fn num_leaf_nodes(&self) -> usize {
        let mut num = 0;
        self.visit_leaf_nodes(&mut |_| {
            num += 1;
        });
        num
    }

    /// Drop all stored items to save memory after tree construction.
    pub fn drop_elements(&mut self) {
        self.items.clear();
        if let Some(ref mut children) = self.children {
            for child in children.iter_mut() {
                child.drop_elements();
            }
        }
    }

    /// Split this node along the specified axis.
    /// Returns true if the split was successful, false otherwise.
    /// The split would fail when too many objects are crowded at the edge of the extent.
    /// If splitting on one axis fails, we'll try the other axis. If both fail, we won't split.
    /// Please refer to [Self::insert_rect] for details.
    fn split(&mut self, split_x: bool) -> bool {
        // Sort items by the appropriate coordinate
        if split_x {
            self.items.sort_by(|a, b| {
                let a_min = a.min();
                let b_min = b.min();
                a_min
                    .x
                    .partial_cmp(&b_min.x)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        a_min
                            .y
                            .partial_cmp(&b_min.y)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
            });
        } else {
            self.items.sort_by(|a, b| {
                let a_min = a.min();
                let b_min = b.min();
                a_min
                    .y
                    .partial_cmp(&b_min.y)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        a_min
                            .x
                            .partial_cmp(&b_min.x)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
            });
        }

        // Find the split coordinate from the middle item
        let middle_idx = self.items.len() / 2;
        let middle_min = self.items[middle_idx].min();

        let split_coord = if split_x { middle_min.x } else { middle_min.y };

        let extent_min = self.extent.min();
        let extent_max = self.extent.max();

        // Check if split coordinate is valid
        if split_x {
            if split_coord <= extent_min.x || split_coord >= extent_max.x {
                // Too many objects are crowded at the edge of the extent. Can't split.
                return false;
            }
        } else if split_coord <= extent_min.y || split_coord >= extent_max.y {
            // Too many objects are crowded at the edge of the extent. Can't split.
            return false;
        }

        // Create child extents
        let (left_extent, right_extent) = if split_x {
            let left = Rect::new(
                extent_min,
                Coord {
                    x: split_coord,
                    y: extent_max.y,
                },
            );
            let right = Rect::new(
                Coord {
                    x: split_coord,
                    y: extent_min.y,
                },
                extent_max,
            );
            (left, right)
        } else {
            let left = Rect::new(
                extent_min,
                Coord {
                    x: extent_max.x,
                    y: split_coord,
                },
            );
            let right = Rect::new(
                Coord {
                    x: extent_min.x,
                    y: split_coord,
                },
                extent_max,
            );
            (left, right)
        };

        // Create children
        let mut left_child = KDBTree::new_with_level(
            self.max_items_per_node,
            self.max_levels,
            self.level + 1,
            left_extent,
        );
        let mut right_child = KDBTree::new_with_level(
            self.max_items_per_node,
            self.max_levels,
            self.level + 1,
            right_extent,
        );

        // Distribute items to children
        if split_x {
            for item in self.items.drain(..) {
                if item.min().x <= split_coord {
                    left_child.insert_rect(item);
                } else {
                    right_child.insert_rect(item);
                }
            }
        } else {
            for item in self.items.drain(..) {
                if item.min().y <= split_coord {
                    left_child.insert_rect(item);
                } else {
                    right_child.insert_rect(item);
                }
            }
        }

        self.children = Some(Box::new([left_child, right_child]));
        true
    }
}

/// KDB tree partitioner that implements the SpatialPartitioner trait.
///
/// # Example Usage
///
/// ```rust
/// use sedona_geometry::bounding_box::BoundingBox;
/// use sedona_spatial_join::partitioning::kdb::KDBPartitioner;
/// use sedona_spatial_join::partitioning::SpatialPartitioner;
///
/// // Sample bounding boxes from your data
/// let bboxes = vec![
///     BoundingBox::xy((0.0, 10.0), (0.0, 10.0)),
///     BoundingBox::xy((10.0, 20.0), (0.0, 10.0)),
///     BoundingBox::xy((20.0, 30.0), (0.0, 10.0)),
/// ];
///
/// // Build partitioner
/// let extent = BoundingBox::xy((0.0, 100.0), (0.0, 100.0));
/// let partitioner = KDBPartitioner::build(
///     bboxes.into_iter(),
///     10,    // max_items_per_node
///     4,     // max_levels
///     extent
/// ).expect("failed to build KDB partitioner");
///
/// // Query partition for a new bbox
/// let query_bbox = BoundingBox::xy((5.0, 15.0), (5.0, 15.0));
/// let partition = partitioner.partition(&query_bbox).unwrap();
/// ```
pub struct KDBPartitioner {
    tree: Arc<KDBTree>,
}

impl KDBPartitioner {
    /// Create a new KDB partitioner from a KDB tree.
    ///
    /// The tree should already be built and have leaf IDs assigned.
    pub(crate) fn new(tree: Arc<KDBTree>) -> Self {
        KDBPartitioner { tree }
    }

    /// Build a KDB tree from a collection of bounding boxes.
    ///
    /// # Arguments
    /// * `bboxes` - Iterator of bounding boxes to partition
    /// * `max_items_per_node` - Maximum items per node before splitting
    /// * `max_levels` - Maximum tree depth
    /// * `extent` - The spatial extent to partition
    pub fn build(
        bboxes: impl Iterator<Item = BoundingBox>,
        max_items_per_node: usize,
        max_levels: usize,
        extent: BoundingBox,
    ) -> Result<Self> {
        let mut tree = KDBTree::try_new(max_items_per_node, max_levels, extent)?;

        for bbox in bboxes {
            tree.insert(bbox)?;
        }

        tree.assign_leaf_ids();
        tree.drop_elements();

        Ok(Self::new(Arc::new(tree)))
    }

    /// Get the number of leaf partitions in the tree.
    pub fn num_partitions(&self) -> usize {
        self.tree.num_leaf_nodes()
    }
}

impl SpatialPartitioner for KDBPartitioner {
    fn num_regular_partitions(&self) -> usize {
        self.num_partitions()
    }

    fn partition(&self, bbox: &BoundingBox) -> Result<SpatialPartition> {
        let rect = match bbox_to_geo_rect(bbox)? {
            Some(rect) => rect,
            None => return Ok(SpatialPartition::None),
        };
        let mut sp: Option<SpatialPartition> = None;

        self.tree.visit_intersecting_leaf_nodes(&rect, &mut |kdb| {
            if sp.is_none() {
                sp = Some(SpatialPartition::Regular(kdb.leaf_id()));
            } else {
                sp = Some(SpatialPartition::Multi)
            }
        });

        Ok(sp.unwrap_or(SpatialPartition::None))
    }

    fn partition_no_multi(&self, bbox: &BoundingBox) -> Result<SpatialPartition> {
        let rect = match bbox_to_geo_rect(bbox)? {
            Some(rect) => rect,
            None => return Ok(SpatialPartition::None),
        };

        let mut best_leaf_id: Option<u32> = None;
        let mut max_area = -1.0_f32;

        self.tree.visit_intersecting_leaf_nodes(&rect, &mut |kdb| {
            let area = rect_intersection_area(&rect, kdb.extent());
            if area > max_area {
                best_leaf_id = Some(kdb.leaf_id());
                max_area = area;
            }
        });

        match best_leaf_id {
            Some(leaf_id) => Ok(SpatialPartition::Regular(leaf_id)),
            None => Ok(SpatialPartition::None),
        }
    }
}

#[cfg(test)]
mod tests {
    use sedona_geometry::interval::{Interval, IntervalTrait};

    use super::*;

    #[test]
    fn test_kdb_insert_and_leaf_assignment() {
        let extent = BoundingBox::xy((0.0, 100.0), (0.0, 100.0));
        let mut tree = KDBTree::try_new(5, 3, extent).unwrap();

        for i in 0..20 {
            let x = (i % 10) as f64 * 10.0;
            let y = (i / 10) as f64 * 10.0;
            tree.insert(BoundingBox::xy((x, x + 5.0), (y, y + 5.0)))
                .unwrap();
        }

        tree.assign_leaf_ids();

        let leaves = tree.collect_leaf_nodes();
        assert!(leaves.len() > 1, "Tree should have split");

        let mut leaf_ids: Vec<_> = leaves.iter().map(|l| l.leaf_id()).collect();
        leaf_ids.sort();
        for (i, &id) in leaf_ids.iter().enumerate() {
            assert_eq!(id, i as u32);
        }
    }

    #[test]
    fn test_kdb_partitioner() {
        let extent = BoundingBox::xy((0.0, 100.0), (0.0, 100.0));

        let bboxes = (0..20).map(|i| {
            if i % 5 != 0 {
                let x = (i % 10) as f64 * 10.0;
                let y = (i / 10) as f64 * 10.0;
                BoundingBox::xy((x, x + 5.0), (y, y + 5.0))
            } else {
                // Mix in some empty bboxes
                BoundingBox::xy(Interval::empty(), Interval::empty())
            }
        });

        let partitioner = KDBPartitioner::build(bboxes, 5, 3, extent).unwrap();

        let test_bbox = BoundingBox::xy((1.0, 4.0), (1.0, 4.0));
        let result = partitioner.partition(&test_bbox).unwrap();
        assert!(
            matches!(result, SpatialPartition::Regular(_)),
            "Expected Regular partition"
        );
        if let SpatialPartition::Regular(id) = result {
            assert!(id < partitioner.num_partitions() as u32);
        }
    }

    #[test]
    fn test_kdb_find_overlapping_partitions() {
        let extent = BoundingBox::xy((0.0, 100.0), (0.0, 100.0));

        let bboxes = (0..20).map(|i| {
            let x = (i % 10) as f64 * 10.0;
            let y = (i / 10) as f64 * 10.0;
            BoundingBox::xy((x, x + 5.0), (y, y + 5.0))
        });

        let partitioner = KDBPartitioner::build(bboxes, 5, 3, extent).unwrap();

        let large_bbox = BoundingBox::xy((0.0, 100.0), (0.0, 100.0));
        assert_eq!(
            partitioner.partition(&large_bbox).unwrap(),
            SpatialPartition::Multi
        );
        assert!(
            matches!(
                partitioner.partition_no_multi(&large_bbox).unwrap(),
                SpatialPartition::Regular(_)
            ),
            "Expected Regular partition"
        );
    }

    #[test]
    fn test_find_leaf_nodes_matches_visit() {
        let extent = BoundingBox::xy((0.0, 100.0), (0.0, 100.0));
        let mut tree = KDBTree::try_new(5, 3, extent).unwrap();

        for i in 0..20 {
            let x = (i % 10) as f64 * 10.0;
            let y = (i / 10) as f64 * 10.0;
            tree.insert(BoundingBox::xy((x, x + 5.0), (y, y + 5.0)))
                .unwrap();
        }

        tree.assign_leaf_ids();

        let query_bbox = BoundingBox::xy((0.0, 60.0), (0.0, 60.0));
        let query_rect = bbox_to_geo_rect(&query_bbox).unwrap().unwrap();

        let mut via_find = Vec::new();
        tree.find_leaf_nodes(&query_rect, &mut via_find);
        assert!(!via_find.is_empty());

        let mut via_visit = Vec::new();
        tree.visit_intersecting_leaf_nodes(&query_rect, &mut |kdb| {
            via_visit.push(kdb.leaf_id());
        });

        let mut find_ids: Vec<_> = via_find.iter().map(|res| res.leaf_id).collect();
        find_ids.sort_unstable();
        let mut visit_ids = via_visit.clone();
        visit_ids.sort_unstable();
        assert_eq!(find_ids, visit_ids);

        for res in via_find.iter() {
            assert!(rects_intersect(&res.extent, &query_rect));
        }
    }

    #[test]
    fn test_find_leaf_nodes_no_match() {
        let extent = BoundingBox::xy((0.0, 100.0), (0.0, 100.0));
        let mut tree = KDBTree::try_new(5, 3, extent).unwrap();

        for i in 0..20 {
            let x = (i % 10) as f64 * 10.0;
            let y = (i / 10) as f64 * 10.0;
            tree.insert(BoundingBox::xy((x, x + 5.0), (y, y + 5.0)))
                .unwrap();
        }

        tree.assign_leaf_ids();

        let query_bbox = BoundingBox::xy((200.0, 210.0), (200.0, 210.0));
        let query_rect = bbox_to_geo_rect(&query_bbox).unwrap().unwrap();

        let mut matches = Vec::new();
        tree.find_leaf_nodes(&query_rect, &mut matches);

        assert!(matches.is_empty());
    }

    #[test]
    fn test_max_overlap_selection() {
        let extent = BoundingBox::xy((0.0, 100.0), (0.0, 100.0));
        let mut bboxes = Vec::new();
        for i in 0..10 {
            bboxes.push(BoundingBox::xy(
                (10.0, 15.0),
                (i as f64 * 10.0, i as f64 * 10.0 + 5.0),
            ));
        }
        for i in 0..10 {
            bboxes.push(BoundingBox::xy(
                (60.0, 65.0),
                (i as f64 * 10.0, i as f64 * 10.0 + 5.0),
            ));
        }

        let partitioner = KDBPartitioner::build(bboxes.into_iter(), 5, 2, extent).unwrap();
        let test_bbox = BoundingBox::xy((30.0, 60.0), (25.0, 35.0));
        assert!(
            matches!(
                partitioner.partition(&test_bbox).unwrap(),
                SpatialPartition::Regular(_)
            ),
            "Expected Regular partition"
        );
    }

    #[test]
    fn test_build_with_samples_outside_boundary() {
        let extent = BoundingBox::xy((0.0, 100.0), (0.0, 100.0));
        let bboxes: Vec<_> = (-100..200)
            .map(|k| {
                let k_f64 = k as f64;
                BoundingBox::xy((k_f64, k_f64), (k_f64, k_f64))
            })
            .collect();

        let partitioner = KDBPartitioner::build(bboxes.into_iter(), 4, 100, extent).unwrap();
        assert!(partitioner.num_partitions() >= 4);
    }

    #[test]
    fn test_partition_assignment_consistency() {
        let extent = BoundingBox::xy((0.0, 100.0), (0.0, 100.0));
        let bboxes: Vec<_> = (0..100)
            .map(|i| {
                let x = (i % 10) as f64 * 10.0;
                let y = (i / 10) as f64 * 10.0;
                BoundingBox::xy((x, x + 2.0), (y, y + 2.0))
            })
            .collect();

        let partitioner = KDBPartitioner::build(bboxes.into_iter(), 10, 3, extent).unwrap();
        let query_bbox = BoundingBox::xy((25.0, 27.0), (35.0, 37.0));
        let first_result = partitioner.partition(&query_bbox).unwrap();

        for _ in 0..10 {
            let result = partitioner.partition(&query_bbox).unwrap();
            assert!(
                matches!(
                    (first_result, result),
                    (SpatialPartition::Regular(_), SpatialPartition::Regular(_))
                ),
                "Expected Regular partitions"
            );
            if let (SpatialPartition::Regular(id1), SpatialPartition::Regular(id2)) =
                (first_result, result)
            {
                assert_eq!(id1, id2, "Partition assignment should be consistent");
            }
        }
    }

    #[test]
    fn test_empty_tree() {
        let extent = BoundingBox::xy((0.0, 100.0), (0.0, 100.0));
        let tree = KDBTree::try_new(5, 3, extent).unwrap();
        assert_eq!(tree.items.len(), 0);
        assert!(tree.is_leaf());
    }

    #[test]
    fn test_split_behavior() {
        let extent = BoundingBox::xy((0.0, 100.0), (0.0, 100.0));
        let mut tree = KDBTree::try_new(3, 5, extent).unwrap();

        tree.insert(BoundingBox::xy((10.0, 20.0), (10.0, 20.0)))
            .unwrap();
        tree.insert(BoundingBox::xy((30.0, 40.0), (30.0, 40.0)))
            .unwrap();
        tree.insert(BoundingBox::xy((50.0, 60.0), (50.0, 60.0)))
            .unwrap();
        assert!(tree.is_leaf(), "Should still be a leaf with 3 items");

        tree.insert(BoundingBox::xy((70.0, 80.0), (70.0, 80.0)))
            .unwrap();
        assert!(!tree.is_leaf(), "Should have split after 4th item");
    }

    #[test]
    fn test_bbox_at_partition_boundary() {
        let extent = BoundingBox::xy((0.0, 100.0), (0.0, 100.0));
        let mut bboxes = Vec::new();
        for i in 0..20 {
            let x = if i < 10 { 25.0 } else { 75.0 };
            let y = (i % 10) as f64 * 10.0;
            bboxes.push(BoundingBox::xy((x, x + 5.0), (y, y + 5.0)));
        }

        let partitioner = KDBPartitioner::build(bboxes.into_iter(), 5, 2, extent).unwrap();
        let boundary_bbox = BoundingBox::xy((49.0, 51.0), (45.0, 55.0));
        assert!(
            matches!(
                partitioner.partition(&boundary_bbox).unwrap(),
                SpatialPartition::Regular(_)
            ),
            "Expected Regular partition"
        );
    }

    #[test]
    fn test_point_like_bboxes() {
        let extent = BoundingBox::xy((0.0, 100.0), (0.0, 100.0));
        let bboxes: Vec<_> = (0..50)
            .map(|i| {
                let x = (i % 10) as f64 * 10.0 + 5.0;
                let y = (i / 10) as f64 * 10.0 + 5.0;
                BoundingBox::xy((x, x), (y, y))
            })
            .collect();

        let partitioner = KDBPartitioner::build(bboxes.into_iter(), 10, 3, extent).unwrap();
        let query_bbox = BoundingBox::xy((25.0, 25.0), (35.0, 35.0));
        assert!(matches!(
            partitioner.partition(&query_bbox).unwrap(),
            SpatialPartition::Regular(_)
        ));
    }

    #[test]
    fn test_large_number_of_partitions() {
        let extent = BoundingBox::xy((0.0, 1000.0), (0.0, 1000.0));
        let bboxes: Vec<_> = (0..1000)
            .map(|i| {
                let x = (i % 100) as f64 * 10.0;
                let y = (i / 100) as f64 * 10.0;
                BoundingBox::xy((x, x + 5.0), (y, y + 5.0))
            })
            .collect();

        let partitioner = KDBPartitioner::build(bboxes.into_iter(), 20, 5, extent).unwrap();
        let test_cases = vec![
            BoundingBox::xy((0.0, 5.0), (0.0, 5.0)),
            BoundingBox::xy((500.0, 505.0), (500.0, 505.0)),
            BoundingBox::xy((995.0, 999.0), (995.0, 999.0)),
        ];

        for bbox in test_cases {
            assert!(matches!(
                partitioner.partition(&bbox).unwrap(),
                SpatialPartition::Regular(_)
            ));
        }
    }

    #[test]
    fn test_out_of_bounds() {
        let extent = BoundingBox::xy((0.0, 100.0), (0.0, 100.0));
        let bboxes = vec![BoundingBox::xy((0.0, 10.0), (0.0, 10.0))];

        let partitioner = KDBPartitioner::build(bboxes.into_iter(), 10, 4, extent).unwrap();

        // Query partition for an out of bounds bbox
        let query_bbox = BoundingBox::xy((200.0, 210.0), (200.0, 210.0));
        let partition = partitioner.partition(&query_bbox).unwrap();

        assert_eq!(partition, SpatialPartition::None);
    }

    #[test]
    fn test_wraparound_samples_rejected() {
        let extent = BoundingBox::xy((0.0, 360.0), (-90.0, 90.0));
        let wrap_bbox = BoundingBox::xy((170.0, -170.0), (-10.0, 10.0));
        let result = KDBPartitioner::build(vec![wrap_bbox].into_iter(), 4, 2, extent);
        assert!(result.is_err());
    }

    #[test]
    fn test_wraparound_queries_rejected() {
        let extent = BoundingBox::xy((0.0, 360.0), (-90.0, 90.0));
        let samples = vec![BoundingBox::xy((0.0, 10.0), (-5.0, 5.0))];
        let partitioner = KDBPartitioner::build(samples.into_iter(), 2, 2, extent).unwrap();
        let wrap_query = BoundingBox::xy((170.0, -170.0), (-5.0, 5.0));
        assert!(partitioner.partition(&wrap_query).is_err());
    }

    #[test]
    fn test_kdb_partitioner_empty_bbox_query() {
        let extent = BoundingBox::xy((0.0, 100.0), (0.0, 100.0));
        let bboxes = vec![BoundingBox::xy((0.0, 10.0), (0.0, 10.0))];
        let partitioner = KDBPartitioner::build(bboxes.into_iter(), 10, 4, extent).unwrap();

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
