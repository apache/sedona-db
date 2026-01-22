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

use crate::partitioning::SpatialPartition;

#[derive(Clone, Copy, Debug)]
/// Maintains the slot mapping for all `SpatialPartition` variants, reserving
/// contiguous indices for regular partitions plus dedicated None/Multi slots.
pub struct PartitionSlots {
    num_regular: usize,
}

impl PartitionSlots {
    /// Create a slot manager for `num_regular` `SpatialPartition::Regular` entries.
    /// Two additional slots are implicitly reserved: one for `None` and one for `Multi`.
    pub fn new(num_regular: usize) -> Self {
        Self { num_regular }
    }

    /// Return the total slot count (`Regular + None + Multi`).
    pub fn total_slots(&self) -> usize {
        self.num_regular + 2
    }

    /// Convert a `SpatialPartition` into its backing slot index.
    pub fn slot(&self, partition: SpatialPartition) -> Option<usize> {
        match partition {
            SpatialPartition::Regular(id) => {
                let id = id as usize;
                if id < self.num_regular {
                    Some(id)
                } else {
                    None
                }
            }
            SpatialPartition::None => Some(self.none_slot()),
            SpatialPartition::Multi => Some(self.multi_slot()),
        }
    }

    /// Convert a slot index back into the corresponding `SpatialPartition` variant.
    pub fn partition(&self, slot: usize) -> SpatialPartition {
        if slot < self.num_regular {
            SpatialPartition::Regular(slot as u32)
        } else if slot == self.none_slot() {
            SpatialPartition::None
        } else if slot == self.multi_slot() {
            SpatialPartition::Multi
        } else {
            panic!(
                "invalid partition slot {slot} for {} regular partitions",
                self.num_regular
            );
        }
    }

    /// Number of regular partitions
    pub fn num_regular_partitions(&self) -> usize {
        self.num_regular
    }

    /// Slot dedicated to `SpatialPartition::None`.
    pub fn none_slot(&self) -> usize {
        self.num_regular
    }

    /// Slot dedicated to `SpatialPartition::Multi`.
    pub fn multi_slot(&self) -> usize {
        self.num_regular + 1
    }
}
