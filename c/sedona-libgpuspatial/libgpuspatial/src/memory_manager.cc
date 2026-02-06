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

#include "gpuspatial/mem/memory_manager.hpp"
#include "gpuspatial/utils/logger.hpp"

#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#else  // POSIX (BSD, Solaris, etc.)
#include <unistd.h>
#endif
namespace gpuspatial {
namespace detail {
inline long long get_free_physical_memory() {
#if defined(_WIN32)
  // --- Windows ---
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  if (GlobalMemoryStatusEx(&status)) {
    return (long long)status.ullAvailPhys;
  }
  return 0;

#elif defined(__linux__)
  // --- Linux (sysinfo) ---
  struct sysinfo info;
  if (sysinfo(&info) == 0) {
    return (long long)info.freeram * (long long)info.mem_unit;
  }
  return 0;

#else
  // --- Generic POSIX ---
  // _SC_AVPHYS_PAGES: The number of physical memory pages not currently in use.
  long pages = sysconf(_SC_AVPHYS_PAGES);
  long page_size = sysconf(_SC_PAGESIZE);

  if (pages > 0 && page_size > 0) {
    return (long long)pages * (long long)page_size;
  }
  return 0;
#endif
}
}  // namespace detail

MemoryManager& MemoryManager::instance() {
  static MemoryManager instance;
  return instance;
}

MemoryManager::~MemoryManager() { Shutdown(); }

void MemoryManager::Shutdown() {
  if (is_initialized_) {
    rmm::mr::set_current_device_resource(nullptr);
    active_resource_.reset();
    pool_mr_.reset();
    cuda_mr_.reset();
    raw_tracker_ptr_ = nullptr;
    is_initialized_ = false;
  }
}

void MemoryManager::Init(bool use_pool, int init_pool_precent) {
  if (is_initialized_) {
    GPUSPATIAL_LOG_WARN(
        "MemoryManager is already initialized. Skipping re-initialization.");
    return;
  }

  cuda_mr_ = std::make_unique<CudaMR>();
  use_pool_ = use_pool;

  if (use_pool_) {
    auto safe_precent = std::max(0, std::min(init_pool_precent, 100));
    auto pool_bytes = rmm::percent_of_free_device_memory(safe_precent);

    GPUSPATIAL_LOG_INFO("Creating RMM pool memory resource with size %zu MB",
                        pool_bytes / 1024 / 1024);

    pool_mr_ = std::make_unique<PoolMR>(cuda_mr_.get(), pool_bytes);
    active_resource_ = std::make_unique<PoolTracker>(pool_mr_.get());
  } else {
    active_resource_ = std::make_unique<CudaTracker>(cuda_mr_.get());
  }

  raw_tracker_ptr_ = active_resource_.get();

  rmm::mr::set_current_device_resource(active_resource_.get());
  is_initialized_ = true;
}

size_t MemoryManager::get_available_device_memory() const {
  auto avail_bytes = rmm::available_device_memory().first;
  if (!is_initialized_ || !use_pool_) {
    return avail_bytes;
  }

  // --- POOL STRATEGY ---
  auto* tracker = static_cast<PoolTracker*>(raw_tracker_ptr_);
  size_t used = tracker->get_allocated_bytes();

  // Safety Buffer: 5% of TOTAL capacity (not just pool capacity)
  size_t safe_limit = static_cast<size_t>(avail_bytes * 0.95);

  return (used < safe_limit) ? (safe_limit - used) : 0;
}

size_t MemoryManager::get_available_host_memory() {
  return detail::get_free_physical_memory();
}
}  // namespace gpuspatial
