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
use sedona_geometry::bounding_box::BoundingBox;

pub mod flat;
pub mod kdb;
pub mod rtree;
pub(crate) mod util;

/// Spatial partitioning is different from traditional data partitioning such as hash partitioning.
/// There is no perfect spatial partitioner that can partition spatial objects with extents (linestrings,
/// polygons, etc.) into disjoint partitions without overlaps. Therefore, a spatial partitioner
/// usually defines a set of spatial partitions (e.g., grid cells), and assigns each spatial object
/// to one or more partitions based on its spatial extent. The spatial partitioner for our
/// out-of-core spatial join follows a similar, but a bit different approach:
///
/// 1. It defines a fixed number of regular spatial partitions (e.g., grid cells), just like
///    traditional spatial partitioners.
/// 2. It defines a `None` partition for spatial objects that does not intersect with any of the
///    partitioning grids.
/// 3. It defines a `Multi` partition for spatial objects that intersect with multiple regular partitions.
///
/// The partitioning result can be one of the following:
/// - Assigned to one of the regular partitions (if it intersects with exactly one regular partition).
/// - Assigned to the `None` partition (if it does not intersect with any regular partition).
/// - Assigned to the `Multi` partition (if it intersects with multiple regular partitions).
///
/// This spatial partitioning scheme assigns one and only one partition to each spatial object,
/// which simplifies the partitioning logic for out-of-core spatial join. The partitioner will be designed
/// to produce only `Regular` partitions for indexed objects, and may produce `None` and `Multi` partitions
/// for probe objects, depending on their spatial extents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpatialPartition {
    Regular(u32),
    None,
    Multi,
}

/// Partitioning larger-than-memory indexed side to support out-of-core spatial join.
pub trait SpatialPartitioner: Send + Sync {
    /// Get the total number of spatial partitions, excluding the None partition and Multi partition.
    fn num_regular_partitions(&self) -> usize;

    /// Given a bounding box, return the partition it is assigned to.
    fn partition(&self, bbox: &BoundingBox) -> Result<SpatialPartition>;

    /// Given a bounding box, return the partition it is assigned to. This function never returns
    /// Multi partition. If `bbox` intersects with multiple partitions, only one of them will be
    /// selected as regular partition.
    fn partition_no_multi(&self, bbox: &BoundingBox) -> Result<SpatialPartition>;
}

/// Indicates for which side of the spatial join the partitioning is being performed.
/// Different methods should be used for the build side and probe side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PartitionedSide {
    /// Invoke [`SpatialPartitioner::partition_no_multi`] for partitioning
    BuildSide,
    /// Invoke [`SpatialPartitioner::partition`] for partitioning
    ProbeSide,
}
