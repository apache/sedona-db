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
#include "rmm/mr/device/cuda_async_memory_resource.hpp"
#include "rmm/mr/device/device_memory_resource.hpp"
#include "rmm/mr/device/pool_memory_resource.hpp"
#include "rmm/mr/device/tracking_resource_adaptor.hpp"

#include <memory>
namespace gpuspatial {
/** @brief An optional singleton memory manager to use asynchronous memory allocation and
 * memory pool with RAPIDS's RMM memory resources.
 * Once the memory manager is initialized, all GPU memory allocations will use the RMM's
 * memory allocator. The user should call Shutdown() to cleanly release RMM resources
 * before program exit.
 */
class MemoryManager {
 public:
  static MemoryManager& instance();

  MemoryManager(const MemoryManager&) = delete;
  MemoryManager& operator=(const MemoryManager&) = delete;

  /**
   * @brief Initializes the memory resources.
   * @param use_pool Whether to use RMM pool allocator
   * @param init_pool_precent Initial pool size as percent of total GPU memory
   */
  void Init(bool use_pool, int init_pool_precent = 50);

  /**
   * @brief Estimates free memory available in bytes
   * * If using a pool: Returns (Total GPU Mem - Tracked Bytes) * 0.95 safety factor.
   * If direct: Returns actual CUDA free memory.
   */
  size_t get_available_device_memory() const;

  /**
   * @brief Estimates free host memory available in bytes
   */
  static size_t get_available_host_memory();
  /**
   * @brief Cleanly resets RMM resources. Automatically called on destruction.
   */
  void Shutdown();

 private:
  MemoryManager() = default;
  ~MemoryManager();

  // --- Type Aliases ---
  using CudaMR = rmm::mr::cuda_async_memory_resource;
  using PoolMR = rmm::mr::pool_memory_resource<CudaMR>;

  // We have two possible tracker types depending on configuration
  using PoolTracker = rmm::mr::tracking_resource_adaptor<PoolMR>;
  using CudaTracker = rmm::mr::tracking_resource_adaptor<CudaMR>;

  // --- State ---
  bool is_initialized_ = false;
  bool use_pool_ = false;

  std::unique_ptr<CudaMR> cuda_mr_;
  std::unique_ptr<PoolMR> pool_mr_;
  std::unique_ptr<rmm::mr::device_memory_resource> active_resource_;

  void* raw_tracker_ptr_ = nullptr;
};
}  // namespace gpuspatial
