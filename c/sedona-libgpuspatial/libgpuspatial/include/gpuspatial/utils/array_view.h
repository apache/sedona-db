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
#include "gpuspatial/utils/cuda_utils.h"

#include <thrust/swap.h>
namespace gpuspatial {
template <typename T>
class ArrayView {
 public:
  ArrayView() = default;

  template <typename VectorType>
  explicit ArrayView(VectorType& vec) : data_(vec.data()), size_(vec.size()) {}

  template <typename VectorType>
  explicit ArrayView(const VectorType& vec)
      : data_(const_cast<T*>(vec.data())), size_(vec.size()) {}

  DEV_HOST ArrayView(T* data, size_t size) : data_(data), size_(size) {}

  DEV_HOST_INLINE T* data() { return data_; }

  DEV_HOST_INLINE const T* data() const { return data_; }

  DEV_HOST_INLINE size_t size() const { return size_; }

  DEV_HOST_INLINE bool empty() const { return size_ == 0; }

  DEV_HOST_INLINE T& operator[](size_t i) {
#ifndef NDEBUG
#if defined(__CUDA_ARCH__)
    if (i >= size_) {
      printf("thread: %u i: %llu size: %llu\n", TID_1D, i, size_);
    }
#endif
#endif
    assert(i < size_);
    return data_[i];
  }

  DEV_HOST_INLINE const T& operator[](size_t i) const {
#ifndef NDEBUG
#if defined(__CUDA_ARCH__)
    if (i >= size_) {
      printf("thread: %u i: %llu size: %llu\n", TID_1D, i, size_);
    }
#endif
#endif
    assert(i < size_);
    return data_[i];
  }

  DEV_HOST_INLINE void Swap(ArrayView<T>& rhs) {
    thrust::swap(data_, rhs.data_);
    thrust::swap(size_, rhs.size_);
  }

  DEV_HOST_INLINE T* begin() { return data_; }

  DEV_HOST_INLINE T* end() { return data_ + size_; }

  DEV_HOST_INLINE const T* begin() const { return data_; }

  DEV_HOST_INLINE const T* end() const { return data_ + size_; }

 private:
  T* data_{};
  size_t size_{};
};
}  // namespace gpuspatial
