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
#include "gpuspatial/utils/exception.h"

#include <cuda_runtime.h>  // For CUDA memory management functions

#include <algorithm>  // For std::copy
#include <cstring>    // For memcpy
#include <stdexcept>  // For std::out_of_range
#include <utility>    // For std::move

namespace gpuspatial {

template <typename T>
class PinnedVector {
  // Enforce at compile time that this class is only used with types
  // that can be safely copied with memcpy.
  static_assert(std::is_trivially_copyable<T>::value,
                "PinnedVector requires a trivially-copyable type.");

  T* data_;          // Pointer to the page-locked (pinned) host array
  size_t size_;      // Number of elements currently in the vector
  size_t capacity_;  // Total storage capacity of the vector

  // Private helper to reallocate memory when capacity is exceeded.
  void reallocate(size_t new_capacity) {
    T* new_data = nullptr;
    CUDA_CHECK(cudaMallocHost((void**)&new_data, new_capacity * sizeof(T)));
    if (data_) {
      memcpy(new_data, data_, size_ * sizeof(T));
      CUDA_CHECK(cudaFreeHost(data_));
    }
    data_ = new_data;
    capacity_ = new_capacity;
  }

 public:
  // Default constructor
  PinnedVector() : data_(nullptr), size_(0), capacity_(0) {}

  // Destructor
  ~PinnedVector() { cudaFreeHost(data_); }

  // Constructor with initial size (value-initialized)
  explicit PinnedVector(size_t size) : size_(size), capacity_(size) {
    CUDA_CHECK(cudaMallocHost((void**)&data_, capacity_ * sizeof(T)));
    // For trivially-copyable types, this often means zero-initialization,
    // but it's safer to do it explicitly if needed.
    memset(data_, 0, capacity_ * sizeof(T));
  }

  // Constructor with initial size and value
  PinnedVector(size_t size, const T& value) : size_(size), capacity_(size) {
    CUDA_CHECK(cudaMallocHost((void**)&data_, capacity_ * sizeof(T)));
    for (size_t i = 0; i < size_; ++i) {
      data_[i] = value;
    }
  }

  // Copy constructor
  PinnedVector(const PinnedVector& other)
      : size_(other.size_), capacity_(other.capacity_) {
    CUDA_CHECK(cudaMallocHost((void**)&data_, capacity_ * sizeof(T)));
    memcpy(data_, other.data_, size_ * sizeof(T));
  }

  // Move constructor
  PinnedVector(PinnedVector&& other) noexcept
      : data_(other.data_), size_(other.size_), capacity_(other.capacity_) {
    // Leave the moved-from object in a valid, empty state
    other.data_ = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
  }

  // Copy assignment operator
  PinnedVector& operator=(const PinnedVector& other) {
    if (this == &other) {
      return *this;
    }
    if (capacity_ < other.size_) {
      reallocate(other.capacity_);
    }
    size_ = other.size_;
    memcpy(data_, other.data_, size_ * sizeof(T));
    return *this;
  }

  // Move assignment operator
  PinnedVector& operator=(PinnedVector&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    // Free existing resources
    cudaFreeHost(data_);
    // Steal resources from the other object
    data_ = other.data_;
    size_ = other.size_;
    capacity_ = other.capacity_;
    // Leave the moved-from object in a valid, empty state
    other.data_ = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
    return *this;
  }

  // --- Iterator methods ---
  T* begin() { return data_; }
  const T* begin() const { return data_; }
  T* end() { return data_ + size_; }
  const T* end() const { return data_ + size_; }

  // --- Raw data access ---
  T* data() { return data_; }
  const T* data() const { return data_; }

  // --- Member functions ---
  void reserve(size_t new_capacity) {
    if (new_capacity > capacity_) {
      reallocate(new_capacity);
    }
  }

  // --- Member functions ---
  void push_back(const T& value) {
    if (size_ >= capacity_) {
      size_t new_capacity = (capacity_ == 0) ? 1 : capacity_ * 2;
      reallocate(new_capacity);
    }
    data_[size_] = value;
    size_++;
  }

  // push_back overload for rvalues
  void push_back(T&& value) {
    if (size_ >= capacity_) {
      size_t new_capacity = (capacity_ == 0) ? 1 : capacity_ * 2;
      reallocate(new_capacity);
    }
    data_[size_] = std::move(value);
    size_++;
  }

  void pop_back() {
    if (size_ > 0) {
      size_--;
    }
  }

  void resize(size_t new_size) {
    if (new_size > capacity_) {
      reallocate(new_size);
    }
    size_ = new_size;
  }

  T& at(size_t index) {
    if (index >= size_) {
      throw std::out_of_range("Vector index out of range");
    }
    return data_[index];
  }

  const T& at(size_t index) const {
    if (index >= size_) {
      throw std::out_of_range("Vector index out of range");
    }
    return data_[index];
  }

  T& operator[](size_t index) { return data_[index]; }

  const T& operator[](size_t index) const { return data_[index]; }

  size_t size() const { return size_; }

  size_t capacity() const { return capacity_; }

  bool empty() const { return size_ == 0; }

  void clear() { size_ = 0; }
};

}  // namespace gpuspatial
