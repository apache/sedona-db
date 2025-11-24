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
use datafusion_common::{utils::proxy::VecAllocExt, Result};
use std::sync::atomic::{AtomicPtr, Ordering};

/// A thread-safe array of values with lock-free concurrent access.
///
/// This array allows multiple threads to concurrently:
/// - Read existing values
/// - Set new values if slots are empty
/// - Create and cache values on-demand
///
/// Uses atomic pointers for lock-free operations with proper memory ordering.
/// Null pointers indicate empty slots.
pub(crate) struct InitOnceArray<T: Send + Sync> {
    ptrs: Vec<AtomicPtr<T>>,
}

impl<T: Send + Sync> InitOnceArray<T> {
    /// Create a new array with the specified size.
    /// All slots start as empty (null pointers).
    pub fn new(size: usize) -> Self {
        let ptrs = (0..size)
            .map(|_| AtomicPtr::new(std::ptr::null_mut()))
            .collect();
        Self { ptrs }
    }

    /// Atomically set a value at the given index if the slot is currently empty.
    ///
    /// Returns tuple containing:
    /// - reference to the value that is now at that index:
    ///   - If the slot was empty, returns the newly set value
    ///   - If the slot was occupied, returns the existing value
    /// - boolean indicating whether the slot was empty
    ///
    /// # Panics
    /// Panics if index is out of bounds (internal API, bounds checking is caller's responsibility).
    pub fn set_if_null(&self, index: usize, value: Box<T>) -> (&T, bool) {
        let new_ptr = Box::into_raw(value);
        let result = self.ptrs[index].compare_exchange_weak(
            std::ptr::null_mut(),
            new_ptr,
            Ordering::AcqRel,
            Ordering::Acquire,
        );

        match result {
            Ok(_) => (unsafe { &*new_ptr }, true), // Successfully set - return reference to the new value
            Err(old_ptr) => {
                // Drop the new value since it's not set. The slot at index is already occupied.
                let _ = unsafe { Box::from_raw(new_ptr) };
                (unsafe { &*old_ptr }, false)
            }
        }
    }

    /// Get the value at the given index, if present.
    ///
    /// Returns None if the slot is empty or index is out of bounds.
    ///
    /// # Panics
    /// Panics if index is out of bounds (internal API, bounds checking is caller's responsibility).
    pub fn get(&self, index: usize) -> Option<&T> {
        let ptr = self.ptrs[index].load(Ordering::Acquire);
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { &*ptr })
        }
    }

    /// Get the value at the given index, or create and set it if not present.
    ///
    /// This is the primary method for lazy initialization of values.
    /// Multiple threads can safely call this method concurrently.
    ///
    /// Returns tuple containing:
    /// - reference to the value that is now at that index:
    ///   - If the slot was empty, returns the newly set value
    ///   - If the slot was occupied, returns the existing value
    /// - boolean indicating whether the returned object is newly created
    ///
    /// # Panics
    /// Panics if index is out of bounds (internal API, bounds checking is caller's responsibility).
    pub fn get_or_create(&self, index: usize, f: impl FnOnce() -> Result<T>) -> Result<(&T, bool)> {
        if let Some(value) = self.get(index) {
            return Ok((value, false));
        }

        let new_value = f()?;
        Ok(self.set_if_null(index, Box::new(new_value)))
    }

    /// Get the allocated size of the array. This does not include the size of the objects
    /// pointed by the elements in the array.
    pub fn allocated_size(&self) -> usize {
        self.ptrs.allocated_size()
    }

    /// Get the size of the array.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.ptrs.len()
    }

    /// Check if the array is empty.
    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.ptrs.is_empty()
    }
}

impl<T: Send + Sync> Drop for InitOnceArray<T> {
    fn drop(&mut self) {
        for ptr in &self.ptrs {
            let ptr = ptr.load(Ordering::Acquire);
            if !ptr.is_null() {
                unsafe {
                    let _ = Box::from_raw(ptr);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    // Helper test struct for testing
    #[derive(Debug, Clone, PartialEq)]
    struct TestData {
        id: usize,
        value: String,
    }

    impl TestData {
        fn new(id: usize, value: &str) -> Self {
            Self {
                id,
                value: value.to_string(),
            }
        }
    }

    #[test]
    fn test_init_once_array_new() {
        let array = InitOnceArray::<String>::new(10);
        assert_eq!(array.len(), 10);
        assert!(!array.is_empty());

        // All slots should be empty initially
        for i in 0..10 {
            assert!(array.get(i).is_none());
        }
    }

    #[test]
    fn test_init_once_array_empty() {
        let array = InitOnceArray::<String>::new(0);
        assert_eq!(array.len(), 0);
        assert!(array.is_empty());
    }

    #[test]
    fn test_set_if_null_empty_slot() {
        let array = InitOnceArray::<String>::new(5);
        let test_value = "test_value".to_string();
        let boxed_value = Box::new(test_value.clone());

        let (result, was_empty) = array.set_if_null(2, boxed_value);

        // Should return reference to the set value
        assert_eq!(result, &test_value);
        assert!(was_empty);

        // Slot should now be occupied
        assert!(array.get(2).is_some());
        assert_eq!(array.get(2).unwrap(), &test_value);
    }

    #[test]
    fn test_set_if_null_occupied_slot() {
        let array = InitOnceArray::<String>::new(5);
        let first_value = "first_value".to_string();
        let second_value = "second_value".to_string();

        let boxed_first = Box::new(first_value.clone());
        let boxed_second = Box::new(second_value.clone());

        // Set first value
        let (result1, was_empty1) = array.set_if_null(2, boxed_first);

        // Try to set second value in same slot
        let (result2, was_empty2) = array.set_if_null(2, boxed_second);

        // Both should return the same value (the first one)
        assert_eq!(result1, &first_value);
        assert!(was_empty1);

        assert_eq!(result2, &first_value);
        assert!(!was_empty2);

        // Slot should contain the first value
        assert_eq!(array.get(2).unwrap(), &first_value);
    }

    #[test]
    fn test_get_or_create_success() {
        let array = InitOnceArray::<TestData>::new(5);
        let test_data = TestData::new(42, "test");

        // First call should create the value
        let result1 = array.get_or_create(2, || Ok(test_data.clone()));

        // Second call should return the same value
        let result2 =
            array.get_or_create(2, || panic!("Should not be called - value already exists"));

        assert!(result1.is_ok());
        assert!(result2.is_ok());

        let (ref1, is_newly_created1) = result1.unwrap();
        let (ref2, is_newly_created2) = result2.unwrap();

        assert_eq!(ref1, &test_data);
        assert!(is_newly_created1);
        assert_eq!(ref2, &test_data);
        assert!(!is_newly_created2);

        // Should be the same reference
        let ptr1 = ref1 as *const TestData;
        let ptr2 = ref2 as *const TestData;
        assert_eq!(ptr1, ptr2);
    }

    #[test]
    fn test_get_or_create_error() {
        let array = InitOnceArray::<String>::new(5);
        let result = array.get_or_create(2, || {
            Err(datafusion_common::DataFusionError::Execution(
                "Failed to create value".to_string(),
            ))
        });
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Failed to create value"));
    }

    #[test]
    fn test_concurrent_set_if_null() {
        let array = Arc::new(InitOnceArray::<String>::new(100));
        let mut handles = vec![];

        // Spawn multiple threads trying to set the same slot
        for i in 0..10 {
            let array_clone = Arc::clone(&array);
            let handle = thread::spawn(move || {
                let value = format!("value_{i}");
                let boxed_value = Box::new(value);
                let (result, _) = array_clone.set_if_null(50, boxed_value);
                result as *const String as usize // Return pointer address as usize
            });
            handles.push(handle);
        }

        // Collect all results
        let mut results = vec![];
        for handle in handles {
            results.push(handle.join().unwrap());
        }

        // All threads should return the same value reference
        let first_ptr = results[0];
        for result in &results[1..] {
            assert_eq!(*result, first_ptr);
        }

        // Only one value should be in the array
        assert!(array.get(50).is_some());
    }

    #[test]
    fn test_concurrent_get_or_create() {
        let array = Arc::new(InitOnceArray::<TestData>::new(100));
        let mut handles = vec![];

        // Spawn multiple threads trying to get_or_create the same slot
        for i in 0..10 {
            let array_clone = Arc::clone(&array);
            let handle = thread::spawn(move || {
                let result = array_clone.get_or_create(75, || {
                    Ok(TestData::new(i, &format!("concurrent_value_{i}")))
                });
                let (ref1, is_newly_created1) = result.unwrap();
                (ref1 as *const TestData as usize, is_newly_created1) // Return pointer address as usize
            });
            handles.push(handle);
        }

        // Collect all results
        let mut results = vec![];
        for handle in handles {
            results.push(handle.join().unwrap());
        }

        // All threads should return the same value reference
        let (first_ptr, _) = results[0];
        for result in &results[1..] {
            assert_eq!(result.0, first_ptr);
        }

        // Only one of the threads should have created the value
        let created_count = results.iter().filter(|v| v.1).count();
        assert_eq!(created_count, 1);

        // Verify the value is actually there
        assert!(array.get(75).is_some());
    }

    #[test]
    fn test_concurrent_different_indices() {
        let array = Arc::new(InitOnceArray::<String>::new(100));
        let mut handles = vec![];

        // Spawn threads targeting different indices
        for i in 0..10 {
            let array_clone = Arc::clone(&array);
            let handle = thread::spawn(move || {
                array_clone
                    .get_or_create(i, || Ok(format!("value_{i}")))
                    .unwrap_or_else(|_| panic!("Failed to create value for index {i}"));
                // Just return () since we don't need the reference
            });
            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }

        // All indices should be filled
        for i in 0..10 {
            assert!(array.get(i).is_some());
            assert_eq!(array.get(i).unwrap(), &format!("value_{i}"));
        }

        // Indices beyond 10 should be empty
        for i in 10..20 {
            assert!(array.get(i).is_none());
        }
    }

    #[test]
    fn test_drop_cleans_up_memory() {
        // This test ensures that Drop implementation properly cleans up
        {
            let array = InitOnceArray::<String>::new(5);
            for i in 0..5 {
                let value = format!("value_{i}");
                let boxed_value = Box::new(value);
                array.set_if_null(i, boxed_value);
            }
            // array goes out of scope here and should clean up all values
        }
        // If we reach here without crashing, the drop worked correctly
    }

    #[test]
    fn test_send_sync_traits() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<InitOnceArray<String>>();
        assert_sync::<InitOnceArray<String>>();
    }

    #[test]
    fn test_with_different_types() {
        // Test with integers
        let int_array = InitOnceArray::<i32>::new(3);
        let result = int_array.get_or_create(1, || Ok(42));
        assert!(result.is_ok());
        assert_eq!(result.unwrap().0, &42);

        // Test with vectors
        let vec_array = InitOnceArray::<Vec<i32>>::new(3);
        let test_vec = vec![1, 2, 3];
        let result = vec_array.get_or_create(0, || Ok(test_vec.clone()));
        assert!(result.is_ok());
        assert_eq!(result.unwrap().0, &test_vec);

        // Test with custom struct
        let struct_array = InitOnceArray::<TestData>::new(3);
        let test_data = TestData::new(123, "custom");
        let result = struct_array.get_or_create(2, || Ok(test_data.clone()));
        assert!(result.is_ok());
        assert_eq!(result.unwrap().0, &test_data);
    }

    #[test]
    fn test_error_doesnt_set_value() {
        let array = InitOnceArray::<String>::new(5);

        // Try to create with an error
        let result = array.get_or_create(2, || {
            Err(datafusion_common::DataFusionError::Execution(
                "Creation failed".to_string(),
            ))
        });
        assert!(result.is_err());

        // Slot should still be empty
        assert!(array.get(2).is_none());

        // Subsequent successful creation should work
        let result2 = array.get_or_create(2, || Ok("success".to_string()));
        assert!(result2.is_ok());
        assert_eq!(result2.unwrap().0, "success");
    }

    #[test]
    fn test_concurrent_mixed_operations() {
        let array = Arc::new(InitOnceArray::<String>::new(50));
        let mut handles = vec![];

        // Spawn threads doing different operations
        for i in 0..20 {
            let array_clone = Arc::clone(&array);
            let handle = thread::spawn(move || {
                match i % 3 {
                    0 => {
                        // set_if_null
                        let value = format!("set_value_{i}");
                        let boxed_value = Box::new(value);
                        array_clone.set_if_null(i, boxed_value);
                    }
                    1 => {
                        // get_or_create
                        let _ = array_clone.get_or_create(i, || Ok(format!("create_value_{i}")));
                    }
                    2 => {
                        // get
                        let _ = array_clone.get(i);
                    }
                    _ => unreachable!(),
                }
            });
            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }

        // Verify some values were set
        let mut set_count = 0;
        for i in 0..20 {
            if array.get(i).is_some() {
                set_count += 1;
            }
        }
        assert!(set_count > 0);
    }
}
