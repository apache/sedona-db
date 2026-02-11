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

use datafusion::execution::memory_pool::{
    MemoryConsumer, MemoryLimit, MemoryPool, MemoryReservation,
};
use datafusion_common::{resources_datafusion_err, DataFusionError, Result};
use parking_lot::Mutex;

pub const DEFAULT_UNSPILLABLE_RESERVE_RATIO: f64 = 0.2;

/// A [`MemoryPool`] implementation similar to DataFusion's [`datafusion::execution::memory_pool::FairSpillPool`],
/// but with the following changes:
///
/// It implements a reservation mechanism for unspillable memory consumers. This addresses an issue
/// where spillable consumers could potentially exhaust all available memory, preventing unspillable
/// operations from acquiring necessary resources. This behavior is tracked in DataFusion issue
/// https://github.com/apache/datafusion/issues/17334. In the context of Sedona, a typical example
/// is a [`sedona_spatial_join::exec::SpatialJoinExec`] operator with an auto inserted
/// [`datafusion::physical_plan::repartition::RepartitionExec`] for the probe side. The Merge
/// consumer of [`datafusion::physical_plan::repartition::RepartitionExec`] is unspillable, while
/// the [`sedona_spatial_join::exec::SpatialJoinExec`] is spillable.
/// [`sedona_spatial_join::exec::SpatialJoinExec`] could consume all memory, resulting a reservation
/// failure of [`datafusion::physical_plan::repartition::RepartitionExec`].
///
/// By reserving a configurable fraction of the total memory pool specifically for unspillable
/// allocations (defined by `unspillable_reserve_ratio`), this pool ensures that critical
/// non-spillable operations can proceed even under heavy memory pressure from spillable operators.
#[derive(Debug)]
pub struct SedonaFairSpillPool {
    /// The total memory limit
    pool_size: usize,
    /// The fraction of memory reserved for unspillable consumers (0.0 - 1.0)
    unspillable_reserve_ratio: f64,

    state: Mutex<FairSpillPoolState>,
}

#[derive(Debug)]
struct FairSpillPoolState {
    /// The number of consumers that can spill
    num_spill: usize,

    /// The total amount of memory reserved that can be spilled
    spillable: usize,

    /// The total amount of memory reserved by consumers that cannot spill
    unspillable: usize,
}

impl SedonaFairSpillPool {
    /// Allocate up to `limit` bytes
    pub fn new(pool_size: usize, unspillable_reserve_ratio: f64) -> Self {
        Self {
            pool_size,
            unspillable_reserve_ratio,
            state: Mutex::new(FairSpillPoolState {
                num_spill: 0,
                spillable: 0,
                unspillable: 0,
            }),
        }
    }
}

impl MemoryPool for SedonaFairSpillPool {
    fn register(&self, consumer: &MemoryConsumer) {
        if consumer.can_spill() {
            self.state.lock().num_spill += 1;
        }
    }

    fn unregister(&self, consumer: &MemoryConsumer) {
        if consumer.can_spill() {
            let mut state = self.state.lock();
            state.num_spill = state.num_spill.checked_sub(1).unwrap();
        }
    }

    fn grow(&self, reservation: &MemoryReservation, additional: usize) {
        let mut state = self.state.lock();
        match reservation.consumer().can_spill() {
            true => state.spillable += additional,
            false => state.unspillable += additional,
        }
    }

    fn shrink(&self, reservation: &MemoryReservation, shrink: usize) {
        let mut state = self.state.lock();
        match reservation.consumer().can_spill() {
            true => state.spillable -= shrink,
            false => state.unspillable -= shrink,
        }
    }

    fn try_grow(&self, reservation: &MemoryReservation, additional: usize) -> Result<()> {
        let mut state = self.state.lock();

        // Calculate the amount of memory reserved for unspillable consumers
        let reserved_for_unspillable =
            (self.pool_size as f64 * self.unspillable_reserve_ratio) as usize;

        // The effective unspillable usage is the max of actual usage and the reserved amount
        let effective_unspillable = state.unspillable.max(reserved_for_unspillable);

        // The total amount of memory available to spilling consumers
        let spill_available = self.pool_size.saturating_sub(effective_unspillable);

        match reservation.consumer().can_spill() {
            true => {
                // No spiller may use more than their fraction of the memory available
                let available = spill_available
                    .checked_div(state.num_spill)
                    .unwrap_or(spill_available);

                if reservation.size() + additional > available {
                    return Err(insufficient_capacity_err(
                        reservation,
                        additional,
                        available,
                        effective_unspillable,
                        spill_available,
                    ));
                }
                state.spillable += additional;
            }
            false => {
                let available = self
                    .pool_size
                    .saturating_sub(state.unspillable + state.spillable);

                if available < additional {
                    return Err(insufficient_capacity_err(
                        reservation,
                        additional,
                        available,
                        effective_unspillable,
                        spill_available,
                    ));
                }
                state.unspillable += additional;
            }
        }
        Ok(())
    }

    fn reserved(&self) -> usize {
        let state = self.state.lock();
        state.spillable + state.unspillable
    }

    fn memory_limit(&self) -> MemoryLimit {
        MemoryLimit::Finite(self.pool_size)
    }
}

fn insufficient_capacity_err(
    reservation: &MemoryReservation,
    additional: usize,
    available: usize,
    unspillable: usize,
    spill_available: usize,
) -> DataFusionError {
    resources_datafusion_err!(
        "Failed to allocate additional {} bytes for {} with {} bytes already allocated - maximum available is {} bytes. \
        Current unspillable memory usage: {} bytes, spillable memory available: {} bytes",
        additional,
        reservation.consumer().name(),
        reservation.size(),
        available,
        unspillable,
        spill_available
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_sedona_fair_spill_pool_reserve() {
        // Pool size 100, 20% reserved for unspillable (20 bytes)
        let pool: Arc<dyn MemoryPool> = Arc::new(SedonaFairSpillPool::new(100, 0.2));

        let spillable_consumer = MemoryConsumer::new("spillable").with_can_spill(true);
        let mut spillable = spillable_consumer.register(&pool);

        let unspillable_consumer = MemoryConsumer::new("unspillable").with_can_spill(false);
        let mut unspillable = unspillable_consumer.register(&pool);

        // Case 1: Spillable cannot eat into reserved memory
        // Available for spillable = 100 - 20 = 80
        spillable.try_grow(80).unwrap();
        assert_eq!(pool.reserved(), 80);

        // Try to grow by 1, should fail because 80 is the limit
        assert!(spillable.try_grow(1).is_err());

        // Case 2: Unspillable can use the reserved memory
        unspillable.try_grow(10).unwrap();
        assert_eq!(pool.reserved(), 90); // 80 (spillable) + 10 (unspillable)

        // Spillable still cannot grow
        assert!(spillable.try_grow(1).is_err());

        // Case 3: Unspillable can grow beyond reserved if space available
        // But currently pool is 90/100.
        // Unspillable wants 15 more (total 25).
        // Available total = 100. Used = 90. Free = 10.
        // Unspillable try_grow(15) -> needs 105 total? No, 90+15 = 105 > 100.
        // Let's shrink spillable first to test "Unspillable eating into Spillable's share"

        spillable.shrink(20); // Spillable = 60. Total = 70.
                              // Unspillable = 10. Reserved = 20.
                              // Effective unspillable = 20. Spill available = 80.
                              // Spillable usage = 60.

        // Unspillable grows by 20. Total unspillable = 30.
        // 30 > 20 (reserved).
        // Total usage = 60 + 30 = 90 <= 100. Should succeed.
        unspillable.try_grow(20).unwrap();
        assert_eq!(pool.reserved(), 90);

        // Now unspillable = 30.
        // Effective unspillable = 30.
        // Spill available = 100 - 30 = 70.
        // Spillable usage = 60.
        // Spillable tries to grow by 11 (60+11=71 > 70). Should fail.
        assert!(spillable.try_grow(11).is_err());
        // Spillable tries to grow by 10 (60+10=70). Should succeed.
        spillable.try_grow(10).unwrap();
        assert_eq!(pool.reserved(), 100);
    }

    #[test]
    fn test_fairness_among_spillers() {
        // Pool size 100, 0% reserved.
        let pool: Arc<dyn MemoryPool> = Arc::new(SedonaFairSpillPool::new(100, 0.0));

        let c1 = MemoryConsumer::new("c1").with_can_spill(true);
        let mut r1 = c1.register(&pool);

        let c2 = MemoryConsumer::new("c2").with_can_spill(true);
        let mut r2 = c2.register(&pool);

        // With 2 spillers, each gets 50.
        r1.try_grow(50).unwrap();
        assert!(r1.try_grow(1).is_err());

        r2.try_grow(50).unwrap();
        assert!(r2.try_grow(1).is_err());

        // If one shrinks, other can't grow immediately if we strictly enforce N-way split?
        // DataFusion FairSpillPool:
        // let available = spill_available.checked_div(state.num_spill).unwrap_or(spill_available);
        // Yes, it strictly enforces split.

        r1.shrink(50);
        // r1 = 0, r2 = 50.
        // r2 tries to grow. Available per spiller = 50. r2 has 50.
        // So r2 cannot grow even if r1 is empty. This is how FairSpillPool works.
        assert!(r2.try_grow(1).is_err());
    }
}
