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

use std::cmp::max;

use datafusion_common::{DataFusionError, Result};

use super::BuildPartition;

/// The memory accounting summary of a build side partition. This is collected
/// during the build side collection phase and used to estimate the memory usage for
/// running spatial join.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PartitionMemorySummary {
    /// Number of rows in the partition.
    pub num_rows: usize,
    /// The total memory reserved when collecting this build side partition.
    pub reserved_memory: usize,
    /// The estimated memory usage for building the spatial index for all the data in
    /// this build side partition.
    pub estimated_index_memory_usage: usize,
}

impl From<&BuildPartition> for PartitionMemorySummary {
    fn from(partition: &BuildPartition) -> Self {
        Self {
            num_rows: partition.num_rows,
            reserved_memory: partition.reservation.size(),
            estimated_index_memory_usage: partition.estimated_spatial_index_memory_usage,
        }
    }
}

/// A detailed plan for memory usage during spatial join execution. The spatial join
/// could be spatial-partitioned if the reserved memory is not sufficient to hold the
/// entire spatial index.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct MemoryPlan {
    /// The total number of rows in the build side.
    pub num_rows: usize,
    /// The total memory reserved for the build side.
    pub reserved_memory: usize,
    /// The estimated memory usage for building the spatial index for the entire build side.
    /// It could be larger than [`Self::reserved_memory`], and in that case we need to
    /// partition the build side using spatial partitioning.
    pub estimated_index_memory_usage: usize,
    /// The memory budget for holding the spatial index. If the spatial join is partitioned,
    /// this is the memory budget for holding the spatial index of a single partition.
    pub memory_for_spatial_index: usize,
    /// The memory budget for intermittent usage, such as buffering data during repartitioning.
    pub memory_for_intermittent_usage: usize,
    /// The number of spatial partitions to split the build side into.
    pub num_partitions: usize,
}

/// Compute the memory plan for running spatial join based on the memory summaries of
/// build side partitions.
pub(crate) fn compute_memory_plan<I>(partition_summaries: I) -> Result<MemoryPlan>
where
    I: IntoIterator<Item = PartitionMemorySummary>,
{
    let mut num_rows = 0;
    let mut reserved_memory = 0;
    let mut estimated_index_memory_usage = 0;

    for summary in partition_summaries {
        num_rows += summary.num_rows;
        reserved_memory += summary.reserved_memory;
        estimated_index_memory_usage += summary.estimated_index_memory_usage;
    }

    if reserved_memory == 0 && num_rows > 0 {
        return Err(DataFusionError::ResourcesExhausted(
            "Insufficient memory for spatial join".to_string(),
        ));
    }

    // Use 80% of reserved memory for holding the spatial index. The other 20% are reserved for
    // intermittent usage like repartitioning buffers.
    let memory_for_spatial_index =
        calculate_memory_for_spatial_index(reserved_memory, estimated_index_memory_usage);
    let memory_for_intermittent_usage = reserved_memory - memory_for_spatial_index;

    let num_partitions = if num_rows > 0 {
        max(
            1,
            estimated_index_memory_usage.div_ceil(memory_for_spatial_index),
        )
    } else {
        1
    };

    Ok(MemoryPlan {
        num_rows,
        reserved_memory,
        estimated_index_memory_usage,
        memory_for_spatial_index,
        memory_for_intermittent_usage,
        num_partitions,
    })
}

fn calculate_memory_for_spatial_index(
    reserved_memory: usize,
    estimated_index_memory_usage: usize,
) -> usize {
    if reserved_memory >= estimated_index_memory_usage {
        // Reserved memory is sufficient to hold the entire spatial index. Make sure that
        // the memory for spatial index is enough for holding the entire index. The rest
        // can be used for intermittent usage.
        estimated_index_memory_usage
    } else {
        // Reserved memory is not sufficient to hold the entire spatial index, We need to
        // partition the dataset using spatial partitioning. Use 80% of reserved memory
        // for holding the partitioned spatial index. The rest is used for intermittent usage.
        let reserved_portion = reserved_memory.saturating_mul(80) / 100;
        if reserved_portion == 0 {
            reserved_memory
        } else {
            reserved_portion
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary(
        num_rows: usize,
        reserved_memory: usize,
        estimated_usage: usize,
    ) -> PartitionMemorySummary {
        PartitionMemorySummary {
            num_rows,
            reserved_memory,
            estimated_index_memory_usage: estimated_usage,
        }
    }

    #[test]
    fn memory_plan_errors_when_no_memory_but_rows_exist() {
        let err = compute_memory_plan(vec![summary(10, 0, 512)]).unwrap_err();
        assert!(matches!(
            err,
            DataFusionError::ResourcesExhausted(msg) if msg.contains("Insufficient memory")
        ));
    }

    #[test]
    fn memory_plan_partitions_large_jobs() {
        let plan =
            compute_memory_plan(vec![summary(100, 2_000, 1_500), summary(150, 1_000, 3_500)])
                .expect("plan should succeed");

        assert_eq!(plan.num_rows, 250);
        assert_eq!(plan.reserved_memory, 3_000);
        assert_eq!(plan.memory_for_spatial_index, 2_400);
        assert_eq!(plan.memory_for_intermittent_usage, 600);
        assert_eq!(plan.num_partitions, 3);
    }

    #[test]
    fn memory_plan_handles_zero_rows() {
        let plan = compute_memory_plan(vec![summary(0, 0, 0)]).expect("plan should succeed");
        assert_eq!(plan.num_partitions, 1);
        assert_eq!(plan.memory_for_spatial_index, 0);
        assert_eq!(plan.memory_for_intermittent_usage, 0);
    }

    #[test]
    fn memory_plan_uses_entire_reservation_when_fraction_rounds_down() {
        let plan = compute_memory_plan(vec![summary(10, 1, 1)]).expect("plan should succeed");
        assert_eq!(plan.memory_for_spatial_index, 1);
        assert_eq!(plan.memory_for_intermittent_usage, 0);
    }
}
