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

#include "gpuspatial/utils/array_view.h"
#include "gpuspatial/utils/cuda_utils.h"

#include <cooperative_groups.h>

namespace gpuspatial {
template <typename T, typename SIZE_T = uint32_t>
class QueueView {
 public:
  using value_type = T;

  QueueView() = default;

  DEV_HOST explicit QueueView(const ArrayView<T>& data, SIZE_T* last_pos)
      : data_(data), last_pos_(last_pos) {}

  DEV_INLINE SIZE_T Append(const T& item) {
    auto allocation = atomicAdd(last_pos_, 1);
#if defined(__CUDA_ARCH__)
    if (allocation >= data_.size()) {
      printf("Queue overflow, TID %u, allocation %u, capacity %lu\n", TID_1D, allocation,
             data_.size());
      __trap();
    }
#endif
    assert(allocation < data_.size());
    data_[allocation] = item;
    return allocation;
  }

  DEV_INLINE SIZE_T AppendWarp(const T& item) {
    auto g = cooperative_groups::coalesced_threads();
    SIZE_T warp_res;

    if (g.thread_rank() == 0) {
      warp_res = atomicAdd(last_pos_, g.size());
    }
    auto begin = g.shfl(warp_res, 0) + g.thread_rank();
    assert(begin < data_.size());
    data_[begin] = item;
    return begin;
  }

  DEV_INLINE void Clear() const { *last_pos_ = 0; }

  DEV_INLINE T& operator[](SIZE_T i) { return data_[i]; }

  DEV_INLINE const T& operator[](SIZE_T i) const { return data_[i]; }

  DEV_INLINE SIZE_T size() const { return *last_pos_; }

  DEV_INLINE void Swap(QueueView& rhs) {
    data_.Swap(rhs.data_);
    thrust::swap(last_pos_, rhs.last_pos_);
  }

  DEV_INLINE T* data() { return data_.data(); }

  DEV_INLINE const T* data() const { return data_.data(); }

 private:
  ArrayView<T> data_;
  SIZE_T* last_pos_{};
};
}  // namespace gpuspatial
