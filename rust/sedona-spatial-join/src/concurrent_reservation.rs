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
use parking_lot::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use datafusion_execution::memory_pool::MemoryReservation;

/// A wrapper around `MemoryReservation` that reserves memory ahead of time to reduce the contention
/// caused by concurrent access to the reservation. It will reserve more memory than requested to
/// skip growing the reservation for the next few reservation growth requests, until the actually
/// reserved size is smaller than the requested size.
pub(crate) struct ConcurrentReservation {
    reservation: Mutex<MemoryReservation>,
    /// The size of reservation. This should be equal to `reservation.size()`. This is used to
    /// minimize contention and avoid growing the underlying reservation in the fast path.
    reserved_size: AtomicUsize,
    prealloc_size: usize,
}

impl ConcurrentReservation {
    pub fn try_new(prealloc_size: usize, reservation: MemoryReservation) -> Result<Self> {
        let actual_size = reservation.size();

        Ok(Self {
            reservation: Mutex::new(reservation),
            reserved_size: AtomicUsize::new(actual_size),
            prealloc_size,
        })
    }

    /// Resize the reservation to the given size. If the new size is smaller or equal to the current
    /// reserved size, do nothing. Otherwise grow the reservation to be `prealloc_size` larger than
    /// the new size. This is for reducing the frequency of growing the underlying reservation.
    pub fn resize(&self, new_size: usize) -> Result<()> {
        // Fast path: the reserved size is already large enough, no need to lock and grow the reservation
        if new_size <= self.reserved_size.load(Ordering::Relaxed) {
            return Ok(());
        }

        // Slow path: lock the mutex for possible reservation growth
        let mut reservation = self.reservation.lock();
        let current_size = reservation.size();

        // Double-check under the lock in case another thread already grew it
        if new_size <= current_size {
            return Ok(());
        }

        // Grow the reservation to the target size
        let growth_needed = new_size + self.prealloc_size - current_size;
        reservation.try_grow(growth_needed)?;

        // Update our atomic to reflect the new size
        let final_size = reservation.size();
        self.reserved_size.store(final_size, Ordering::Relaxed);

        Ok(())
    }

    #[cfg(test)]
    pub fn actual_size(&self) -> usize {
        self.reservation.lock().size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_execution::memory_pool::{MemoryConsumer, MemoryPool};
    use std::sync::Arc;
    use std::thread;

    fn create_test_reservation(size: usize) -> MemoryReservation {
        let pool: Arc<dyn MemoryPool> =
            Arc::new(datafusion_execution::memory_pool::UnboundedMemoryPool::default());
        let consumer = MemoryConsumer::new("test");
        let mut reservation = consumer.register(&pool);
        if size > 0 {
            reservation.grow(size);
        }
        reservation
    }

    #[test]
    fn test_basic_functionality() {
        let reservation = create_test_reservation(0);
        let concurrent = ConcurrentReservation::try_new(100, reservation).unwrap();

        assert_eq!(concurrent.reserved_size.load(Ordering::Relaxed), 0);

        // Resize to smaller should be no-op
        concurrent.resize(0).unwrap();
        assert_eq!(concurrent.reserved_size.load(Ordering::Relaxed), 0);

        // Resize to larger should grow
        let initial_size = concurrent.reserved_size.load(Ordering::Relaxed);
        concurrent.resize(200).unwrap();
        let final_size = concurrent.reserved_size.load(Ordering::Relaxed);
        assert!(final_size >= 200);
        assert!(final_size > initial_size);
        assert!(concurrent.actual_size() >= 200);
    }

    #[test]
    fn test_with_existing_reservation() {
        let reservation = create_test_reservation(50);
        let concurrent = ConcurrentReservation::try_new(100, reservation).unwrap();

        // Initial state should have already reserved size accounted
        assert_eq!(concurrent.reserved_size.load(Ordering::Relaxed), 50);

        // Smaller size won't grow the reservation
        concurrent.resize(10).unwrap();
        assert_eq!(concurrent.reserved_size.load(Ordering::Relaxed), 50);
        assert_eq!(concurrent.actual_size(), 50);

        // Won't grow the reservation
        concurrent.resize(50).unwrap();
        assert_eq!(concurrent.reserved_size.load(Ordering::Relaxed), 50);
        assert_eq!(concurrent.actual_size(), 50);

        // Resize to larger should grow
        concurrent.resize(100).unwrap();
        assert_eq!(concurrent.reserved_size.load(Ordering::Relaxed), 200);
        assert_eq!(concurrent.actual_size(), 200);
    }

    #[test]
    fn test_concurrent_access() {
        let reservation = create_test_reservation(0);
        let concurrent = Arc::new(ConcurrentReservation::try_new(1000, reservation).unwrap());

        let mut handles = vec![];

        // Spawn multiple threads that try to resize concurrently
        for i in 0..10 {
            let concurrent_clone: Arc<ConcurrentReservation> = Arc::clone(&concurrent);
            let handle = thread::spawn(move || {
                // Each thread tries to resize to a different size
                let target_size = (i + 1) * 500;
                concurrent_clone.resize(target_size).unwrap();

                // Verify the invariant: actual size should be >= requested size
                let actual = concurrent_clone.reserved_size.load(Ordering::Relaxed);
                assert!(actual >= target_size);
                let actual = concurrent_clone.actual_size();
                assert!(actual >= target_size);
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // Final size should be at least as large as the largest request
        assert!(concurrent.reserved_size.load(Ordering::Relaxed) >= 5000);
        assert!(concurrent.actual_size() >= 5000);
    }
}
