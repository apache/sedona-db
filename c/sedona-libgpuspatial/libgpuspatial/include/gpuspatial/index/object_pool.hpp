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
#pragma once

#include <memory>
#include <mutex>
#include <vector>

namespace gpuspatial {
// Forward declaration of ObjectPool to be used in the custom deleter.
template <typename T>
class ObjectPool;

// A helper struct to allow std::make_shared to access the private constructor.
// It inherits from ObjectPool and is defined outside of it.
template <typename T>
struct PoolEnabler : public ObjectPool<T> {
  PoolEnabler(size_t size) : ObjectPool<T>(size) {}
};

// A custom deleter for std::shared_ptr.
// When the shared_ptr's reference count goes to zero, this deleter
// will be invoked, returning the object to the pool instead of deleting it.
template <typename T>
class PoolDeleter {
 public:
  // Constructor takes a weak_ptr to the pool to avoid circular references.
  PoolDeleter(std::weak_ptr<ObjectPool<T>> pool) : pool_(pool) {}

  // The function call operator is what std::shared_ptr invokes.
  void operator()(T* ptr) const {
    // Attempt to lock the weak_ptr to get a shared_ptr to the pool.
    if (auto pool_sp = pool_.lock()) {
      // If the pool still exists, return the object to it.
      pool_sp->release(ptr);
    } else {
      // If the pool no longer exists, we must delete the pointer to avoid a memory leak.
      delete ptr;
    }
  }

 private:
  std::weak_ptr<ObjectPool<T>> pool_;
};

/**
 * @brief A thread-safe object pool for reusable objects.
 *
 * @tparam T The type of object to pool.
 */
template <typename T>
class ObjectPool : public std::enable_shared_from_this<ObjectPool<T>> {
  friend struct PoolEnabler<T>;

  // Constructor is private to force object creation through the static 'create' method.
  // This ensures the ObjectPool is always managed by a std::shared_ptr.
  ObjectPool(size_t initial_size = 0) {
    for (size_t i = 0; i < initial_size; ++i) {
      pool_.push_back(new T());
    }
  }

 public:
  /**
   * @brief Factory method to create an instance of the ObjectPool.
   * Guarantees that the pool is managed by a std::shared_ptr, which is required
   * for the custom deleter mechanism to work correctly.
   *
   * @param initial_size The number of objects to pre-allocate.
   * @return A std::shared_ptr to the new ObjectPool instance.
   */
  static std::shared_ptr<ObjectPool<T>> create(size_t initial_size = 0) {
    return std::make_shared<PoolEnabler<T>>(initial_size);
  }

  /**
   * @brief Destructor. Cleans up any remaining objects in the pool.
   */
  ~ObjectPool() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (T* item : pool_) {
      delete item;
    }
    pool_.clear();
  }

  // Disable copy constructor and assignment operator
  ObjectPool(const ObjectPool&) = delete;
  ObjectPool& operator=(const ObjectPool&) = delete;

  /**
   * @brief Acquires an object from the pool.
   *
   * If the pool is empty, a new object is created. The returned shared_ptr
   * has a custom deleter that will return the object to the pool when it's
   * no longer referenced.
   *
   * @return A std::shared_ptr to an object of type T.
   */
  std::shared_ptr<T> take() {
    std::lock_guard<std::mutex> lock(mutex_);
    T* resource_ptr = nullptr;
    if (!pool_.empty()) {
      // Take an existing object from the pool
      resource_ptr = pool_.back();
      pool_.pop_back();
    } else {
      // Pool is empty, create a new object
      resource_ptr = new T();
    }

    // Create a custom deleter that knows how to return the object to this pool.
    // this->shared_from_this() is now safe because creation is forced through the
    // 'create' method.
    PoolDeleter<T> deleter(this->shared_from_this());

    // Return a shared_ptr with the custom deleter.
    return std::shared_ptr<T>(resource_ptr, deleter);
  }

  /**
   * @brief Returns an object to the pool.
   *
   * This method is intended to be called by the PoolDeleter, not directly by clients.
   *
   * @param object The raw pointer to the object to return to the pool.
   */
  void release(T* object) {
    std::lock_guard<std::mutex> lock(mutex_);
    pool_.push_back(object);
  }

  /**
   * @brief Gets the current number of available objects in the pool.
   * @return The size of the pool.
   */
  size_t size() {
    std::lock_guard<std::mutex> lock(mutex_);
    return pool_.size();
  }

 private:
  std::vector<T*> pool_;
  std::mutex mutex_;
};

}  // namespace gpuspatial
