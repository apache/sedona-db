
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
#include "gpuspatial/loader/parallel_wkb_loader.hpp"
#include "gpuspatial/refine/rt_spatial_refiner.cuh"
#include "gpuspatial/relate/relate_engine.cuh"
#include "gpuspatial/utils/logger.hpp"

#include "rt/shaders/shader_id.hpp"

#include "rmm/cuda_stream_pool.hpp"
#include "rmm/exec_policy.hpp"


#include <thrust/gather.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <future>
#include <locale>
#include <numeric>
#include <vector>

#define OPTIX_MAX_RAYS (1lu << 30)

namespace gpuspatial {

namespace detail {
template <typename INDEX_IT>
void ReorderIndices(rmm::cuda_stream_view stream, INDEX_IT index_begin,
                    INDEX_IT index_end,
                    rmm::device_uvector<uint32_t>& sorted_uniq_indices,
                    rmm::device_uvector<uint32_t>& reordered_indices) {
  auto sorted_begin = sorted_uniq_indices.begin();
  auto sorted_end = sorted_uniq_indices.end();
  thrust::transform(rmm::exec_policy_nosync(stream), index_begin, index_end,
                    reordered_indices.begin(), [=] __device__(uint32_t val) {
                      auto it =
                          thrust::lower_bound(thrust::seq, sorted_begin, sorted_end, val);
                      return thrust::distance(sorted_begin, it);
                    });
}

template <typename LoaderT, typename DeviceGeomT>
struct PipelineSlot {
  rmm::cuda_stream_view stream;
  std::unique_ptr<LoaderT> loader;
  std::future<DeviceGeomT> prep_future;

  RTSpatialRefiner::IndicesMap indices_map;

  // These will be moved out after every batch
  rmm::device_uvector<uint32_t> d_batch_build_indices;
  rmm::device_uvector<uint32_t> d_batch_probe_indices;

  PipelineSlot(rmm::cuda_stream_view s, const std::shared_ptr<ThreadPool>& tp,
               typename LoaderT::Config config)
      : stream(s), d_batch_build_indices(0, s), d_batch_probe_indices(0, s) {
    loader = std::make_unique<LoaderT>(tp);
    loader->Init(config);
  }
};
}  // namespace detail

RTSpatialRefiner::RTSpatialRefiner(const RTSpatialRefinerConfig& config)
    : config_(config) {
  thread_pool_ = std::make_shared<ThreadPool>(config_.parsing_threads);
  stream_pool_ = std::make_unique<rmm::cuda_stream_pool>(config_.concurrency);
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, config_.stack_size_bytes));
  wkb_loader_ = std::make_unique<loader_t>(thread_pool_);

  ParallelWkbLoader<point_t, index_t>::Config loader_config;

  loader_config.memory_quota = config_.wkb_parser_memory_quota;

  wkb_loader_->Init(loader_config);
}

void RTSpatialRefiner::Clear() {
  auto stream = rmm::cuda_stream_default;
  wkb_loader_->Clear(stream);
  build_geometries_.Clear(stream);
}

void RTSpatialRefiner::PushBuild(const ArrowSchema* build_schema,
                                 const ArrowArray* build_array) {
  auto stream = rmm::cuda_stream_default;

  wkb_loader_->Parse(stream, build_schema, build_array, 0, build_array->length);
}

void RTSpatialRefiner::FinishBuilding() {
  auto stream = rmm::cuda_stream_default;
  build_geometries_ = std::move(wkb_loader_->Finish(stream));
}

uint32_t RTSpatialRefiner::Refine(const ArrowSchema* probe_schema,
                                  const ArrowArray* probe_array, Predicate predicate,
                                  uint32_t* build_indices, uint32_t* probe_indices,
                                  uint32_t len) {
  if (len == 0) {
    return 0;
  }

  if (config_.pipeline_batches > 1) {
    return RefinePipelined(probe_schema, probe_array, predicate, build_indices,
                           probe_indices, len);
  }

  SpatialRefinerContext ctx;
  ctx.cuda_stream = stream_pool_->get_stream();

  IndicesMap probe_indices_map;
  rmm::device_uvector<uint32_t> d_probe_indices(len, ctx.cuda_stream);

  CUDA_CHECK(cudaMemcpyAsync(d_probe_indices.data(), probe_indices,
                             sizeof(uint32_t) * len, cudaMemcpyHostToDevice,
                             ctx.cuda_stream));

  buildIndicesMap(ctx.cuda_stream, d_probe_indices.begin(), d_probe_indices.end(),
                  probe_indices_map);

  loader_t loader(thread_pool_);
  loader_t::Config loader_config;
  loader_config.memory_quota = config_.wkb_parser_memory_quota / config_.concurrency;

  loader.Init(loader_config);
  loader.Parse(ctx.cuda_stream, probe_schema, probe_array,
               probe_indices_map.h_uniq_indices.begin(),
               probe_indices_map.h_uniq_indices.end());
  auto probe_geoms = std::move(loader.Finish(ctx.cuda_stream));

  GPUSPATIAL_LOG_INFO(
      "RTSpatialRefiner %p (Free %zu MB), Loaded Geometries, ProbeArray %ld, Loaded %u, Type %s",
      this, rmm::available_device_memory().first / 1024 / 1024, probe_array->length,
      probe_geoms.num_features(),
      GeometryTypeToString(probe_geoms.get_geometry_type()).c_str());

  RelateEngine<point_t, index_t> relate_engine(&build_geometries_,
                                               config_.rt_engine.get());
  RelateEngine<point_t, index_t>::Config re_config;

  re_config.memory_quota = config_.relate_engine_memory_quota / config_.concurrency;
  re_config.bvh_fast_build = config_.prefer_fast_build;
  re_config.bvh_compact = config_.compact;

  relate_engine.set_config(re_config);

  rmm::device_uvector<uint32_t> d_build_indices(len, ctx.cuda_stream);
  CUDA_CHECK(cudaMemcpyAsync(d_build_indices.data(), build_indices,
                             sizeof(uint32_t) * len, cudaMemcpyHostToDevice,
                             ctx.cuda_stream));

  GPUSPATIAL_LOG_INFO(
      "RTSpatialRefiner %p (Free %zu MB), Evaluating %u Geometry Pairs with Predicate %s",
      this, rmm::available_device_memory().first / 1024 / 1024, len,
      PredicateToString(predicate));

  ctx.timer.start(ctx.cuda_stream);
  relate_engine.Evaluate(ctx.cuda_stream, probe_geoms, predicate, d_build_indices,
                         probe_indices_map.d_reordered_indices);
  float refine_ms = ctx.timer.stop(ctx.cuda_stream);
  auto new_size = d_build_indices.size();

  GPUSPATIAL_LOG_INFO("RTSpatialRefiner %p (Free %zu MB), Refine time %f, new size %zu",
                      this, rmm::available_device_memory().first / 1024 / 1024, refine_ms,
                      new_size);

  d_probe_indices.resize(new_size, ctx.cuda_stream);

  thrust::gather(rmm::exec_policy_nosync(ctx.cuda_stream),
                 probe_indices_map.d_reordered_indices.begin(),
                 probe_indices_map.d_reordered_indices.end(),
                 probe_indices_map.d_uniq_indices.begin(), d_probe_indices.begin());

  if (config_.sort_probe_indices) {
    thrust::sort_by_key(rmm::exec_policy_nosync(ctx.cuda_stream), d_probe_indices.begin(),
                        d_probe_indices.end(), d_build_indices.begin());
  }

  CUDA_CHECK(cudaMemcpyAsync(build_indices, d_build_indices.data(),
                             sizeof(uint32_t) * new_size, cudaMemcpyDeviceToHost,
                             ctx.cuda_stream));

  CUDA_CHECK(cudaMemcpyAsync(probe_indices, d_probe_indices.data(),
                             sizeof(uint32_t) * new_size, cudaMemcpyDeviceToHost,
                             ctx.cuda_stream));
  ctx.cuda_stream.synchronize();
  return new_size;
}

uint32_t RTSpatialRefiner::RefinePipelined(const ArrowSchema* probe_schema,
                                           const ArrowArray* probe_array,
                                           Predicate predicate, uint32_t* build_indices,
                                           uint32_t* probe_indices, uint32_t len) {
  if (len == 0) return 0;
  auto main_stream = stream_pool_->get_stream();

  rmm::device_uvector<uint32_t> d_build_indices(len, main_stream);
  rmm::device_uvector<uint32_t> d_probe_indices(len, main_stream);

  CUDA_CHECK(cudaMemcpyAsync(d_build_indices.data(), build_indices,
                             sizeof(uint32_t) * len, cudaMemcpyHostToDevice,
                             main_stream));
  CUDA_CHECK(cudaMemcpyAsync(d_probe_indices.data(), probe_indices,
                             sizeof(uint32_t) * len, cudaMemcpyHostToDevice,
                             main_stream));

  thrust::sort_by_key(rmm::exec_policy_nosync(main_stream), d_probe_indices.begin(),
                      d_probe_indices.end(), d_build_indices.begin());

  rmm::device_uvector<uint32_t> d_final_build_indices(len, main_stream);
  rmm::device_uvector<uint32_t> d_final_probe_indices(len, main_stream);

  uint32_t tail_offset = 0;

  // Capture device ID for thread safety
  int device_id;
  CUDA_CHECK(cudaGetDevice(&device_id));

  // Pipeline Config
  const int NUM_SLOTS = 2;
  int n_batches = config_.pipeline_batches;
  size_t batch_size = (len + n_batches - 1) / n_batches;

  GPUSPATIAL_LOG_INFO(
      "RTSpatialRefiner %p, pipeline refinement, total len %u, batches %d, batch size %zu",
      this, len, n_batches, batch_size);

  // Resource allocation for slots
  using loader_t = ParallelWkbLoader<point_t, index_t>;
  loader_t::Config loader_config;
  loader_config.memory_quota =
      config_.wkb_parser_memory_quota / config_.concurrency / NUM_SLOTS;

  rmm::cuda_stream_pool local_pool(NUM_SLOTS);
  std::vector<std::unique_ptr<detail::PipelineSlot<loader_t, dev_geometries_t>>> slots;

  for (int i = 0; i < NUM_SLOTS; ++i) {
    slots.push_back(std::make_unique<detail::PipelineSlot<loader_t, dev_geometries_t>>(
        local_pool.get_stream(), thread_pool_, loader_config));
  }

  // Engine Setup (Shared across slots)
  RelateEngine<point_t, index_t> relate_engine(&build_geometries_,
                                               config_.rt_engine.get());
  RelateEngine<point_t, index_t>::Config re_config;
  re_config.memory_quota =
      config_.relate_engine_memory_quota / config_.concurrency / NUM_SLOTS;
  re_config.bvh_fast_build = config_.prefer_fast_build;
  re_config.bvh_compact = config_.compact;
  relate_engine.set_config(re_config);

  // --- BACKGROUND TASK (CPU Phase) ---
  // This lambda handles: buildIndicesMap + WKB Parsing
  auto prepare_batch_task = [&](detail::PipelineSlot<loader_t, dev_geometries_t>* slot,
                                size_t offset, size_t count) {
    // 1. Critical: Set context for this thread
    CUDA_CHECK(cudaSetDevice(device_id));

    // 2. Wait for GPU to finish previous work on this slot
    slot->stream.synchronize();

    // 3. Prepare Indices (CPU + H2D)
    const uint32_t* batch_probe_ptr = d_probe_indices.data() + offset;
    buildIndicesMap(slot->stream, batch_probe_ptr, batch_probe_ptr + count,
                    slot->indices_map);

    // 4. Parse WKB (CPU Heavy)
    slot->loader->Clear(slot->stream);
    slot->loader->Parse(slot->stream, probe_schema, probe_array,
                        slot->indices_map.h_uniq_indices.begin(),
                        slot->indices_map.h_uniq_indices.end());

    // Return future geometries (H2D copy happens on Finish)
    return slot->loader->Finish(slot->stream);
  };

  // --- PIPELINE PRIMING ---
  // Start processing Batch 0 immediately in background
  size_t first_batch_len = std::min(batch_size, (size_t)len);
  slots[0]->prep_future = std::async(std::launch::async, prepare_batch_task,
                                     slots[0].get(), 0, first_batch_len);

  main_stream.synchronize();  // Ensure allocation is done before main loop

  // --- MAIN PIPELINE LOOP ---
  for (size_t offset = 0; offset < len; offset += batch_size) {
    int curr_idx = (offset / batch_size) % NUM_SLOTS;
    int next_idx = (curr_idx + 1) % NUM_SLOTS;
    auto& curr_slot = slots[curr_idx];
    auto& next_slot = slots[next_idx];
    size_t current_batch_len = std::min(batch_size, len - offset);

    // 1. WAIT & RETRIEVE: Get Geometries from Background Task
    // This will block only if CPU work for this batch is slower than GPU work for
    // previous batch
    dev_geometries_t probe_geoms;
    if (curr_slot->prep_future.valid()) {
      probe_geoms = std::move(curr_slot->prep_future.get());
    }

    // 2. KICKOFF NEXT: Start CPU work for Batch (N+1)
    size_t next_offset = offset + batch_size;
    if (next_offset < len) {
      size_t next_len = std::min(batch_size, len - next_offset);
      next_slot->prep_future = std::async(std::launch::async, prepare_batch_task,
                                          next_slot.get(), next_offset, next_len);
    }

    // 3. GPU EXECUTION PHASE
    const uint32_t* batch_build_ptr = d_build_indices.data() + offset;

    // Copy build indices for this batch
    curr_slot->d_batch_build_indices.resize(current_batch_len, curr_slot->stream);
    CUDA_CHECK(cudaMemcpyAsync(curr_slot->d_batch_build_indices.data(), batch_build_ptr,
                               sizeof(uint32_t) * current_batch_len,
                               cudaMemcpyHostToDevice, curr_slot->stream));

    // Relate/Refine
    // Note: Evaluate filters d_batch_build_indices in-place
    relate_engine.Evaluate(curr_slot->stream, probe_geoms, predicate,
                           curr_slot->d_batch_build_indices,
                           curr_slot->indices_map.d_reordered_indices);

    // 4. GATHER & APPEND RESULTS
    // We need the size to know how much to gather
    size_t new_size = curr_slot->d_batch_build_indices.size();

    if (new_size > 0) {
      // Gather original probe indices
      curr_slot->d_batch_probe_indices.resize(new_size, curr_slot->stream);
      thrust::gather(rmm::exec_policy_nosync(curr_slot->stream),
                     curr_slot->indices_map.d_reordered_indices.begin(),
                     curr_slot->indices_map.d_reordered_indices.end(),
                     curr_slot->indices_map.d_uniq_indices.begin(),
                     curr_slot->d_batch_probe_indices.begin());

      // Append to Final Buffers (Device-to-Device Copy)
      CUDA_CHECK(cudaMemcpyAsync(d_final_build_indices.data() + tail_offset,
                                 curr_slot->d_batch_build_indices.data(),
                                 sizeof(uint32_t) * new_size, cudaMemcpyDeviceToDevice,
                                 curr_slot->stream));

      CUDA_CHECK(cudaMemcpyAsync(d_final_probe_indices.data() + tail_offset,
                                 curr_slot->d_batch_probe_indices.data(),
                                 sizeof(uint32_t) * new_size, cudaMemcpyDeviceToDevice,
                                 curr_slot->stream));

      tail_offset += new_size;
    }
  }

  // --- FINALIZATION ---

  // Wait for all streams to finish writing to final buffers
  for (auto& slot : slots) {
    slot->stream.synchronize();
  }

  // Shrink probe vector to actual size for sorting
  d_final_probe_indices.resize(tail_offset, main_stream);
  d_final_build_indices.resize(tail_offset, main_stream);

  if (config_.sort_probe_indices) {
    thrust::sort_by_key(rmm::exec_policy_nosync(main_stream),
                        d_final_probe_indices.begin(),
                        d_final_probe_indices.end(),  // Sort only valid range
                        d_final_build_indices.begin());
  }

  // Final Copy to Host
  CUDA_CHECK(cudaMemcpyAsync(build_indices, d_final_build_indices.data(),
                             sizeof(uint32_t) * tail_offset, cudaMemcpyDeviceToHost,
                             main_stream));

  CUDA_CHECK(cudaMemcpyAsync(probe_indices, d_final_probe_indices.data(),
                             sizeof(uint32_t) * tail_offset, cudaMemcpyDeviceToHost,
                             main_stream));

  main_stream.synchronize();
  return tail_offset;
}

uint32_t RTSpatialRefiner::Refine(const ArrowSchema* build_schema,
                                  const ArrowArray* build_array,
                                  const ArrowSchema* probe_schema,
                                  const ArrowArray* probe_array, Predicate predicate,
                                  uint32_t* build_indices, uint32_t* probe_indices,
                                  uint32_t len) {
  if (len == 0) {
    return 0;
  }

  auto cuda_stream = stream_pool_->get_stream();
  SpatialRefinerContext ctx;

  ctx.cuda_stream = cuda_stream;

  IndicesMap build_indices_map, probe_indices_map;
  rmm::device_uvector<uint32_t> d_indices(len, cuda_stream);

  CUDA_CHECK(cudaMemcpyAsync(d_indices.data(), build_indices, sizeof(uint32_t) * len,
                             cudaMemcpyHostToDevice, cuda_stream));
  buildIndicesMap(cuda_stream, d_indices.begin(), d_indices.end(), build_indices_map);

  CUDA_CHECK(cudaMemcpyAsync(d_indices.data(), probe_indices, sizeof(uint32_t) * len,
                             cudaMemcpyHostToDevice, cuda_stream));
  buildIndicesMap(cuda_stream, d_indices.begin(), d_indices.end(), probe_indices_map);
  d_indices.resize(0, cuda_stream);
  d_indices.shrink_to_fit(cuda_stream);

  loader_t loader(thread_pool_);
  loader_t::Config loader_config;
  loader_config.memory_quota = config_.wkb_parser_memory_quota / config_.concurrency;
  loader.Init(loader_config);
  loader.Parse(ctx.cuda_stream, build_schema, build_array,
               build_indices_map.h_uniq_indices.begin(),
               build_indices_map.h_uniq_indices.end());
  auto geoms1 = std::move(loader.Finish(ctx.cuda_stream));

  loader.Clear(ctx.cuda_stream);
  loader.Parse(ctx.cuda_stream, probe_schema, probe_array,
               probe_indices_map.h_uniq_indices.begin(),
               probe_indices_map.h_uniq_indices.end());
  auto geoms2 = std::move(loader.Finish(ctx.cuda_stream));

  GPUSPATIAL_LOG_INFO(
      "RTSpatialRefiner %p (Free %zu MB), Loaded Geometries, build_array %ld, Loaded %u, Type %s, probe_array %ld, Loaded %u, Type %s",
      this, rmm::available_device_memory().first / 1024 / 1024, build_array->length,
      geoms1.num_features(), GeometryTypeToString(geoms1.get_geometry_type()).c_str(),
      probe_array->length, geoms2.num_features(),
      GeometryTypeToString(geoms2.get_geometry_type()).c_str());

  RelateEngine<point_t, index_t> relate_engine(&geoms1, config_.rt_engine.get());
  RelateEngine<point_t, index_t>::Config re_config;

  re_config.memory_quota = config_.relate_engine_memory_quota / config_.concurrency;
  re_config.bvh_fast_build = config_.prefer_fast_build;
  re_config.bvh_compact = config_.compact;

  relate_engine.set_config(re_config);

  GPUSPATIAL_LOG_INFO(
      "RTSpatialRefiner %p (Free %zu MB), Evaluating %u Geometry Pairs with Predicate %s",
      this, rmm::available_device_memory().first / 1024 / 1024, len,
      PredicateToString(predicate));

  ctx.timer.start(ctx.cuda_stream);

  relate_engine.Evaluate(ctx.cuda_stream, geoms2, predicate,
                         build_indices_map.d_reordered_indices,
                         probe_indices_map.d_reordered_indices);
  float refine_ms = ctx.timer.stop(ctx.cuda_stream);

  auto new_size = build_indices_map.d_reordered_indices.size();
  GPUSPATIAL_LOG_INFO("RTSpatialRefiner %p (Free %zu MB), Refine time %f, new size %zu",
                      this, rmm::available_device_memory().first / 1024 / 1024, refine_ms,
                      new_size);
  rmm::device_uvector<uint32_t> d_build_indices(new_size, ctx.cuda_stream);
  rmm::device_uvector<uint32_t> d_probe_indices(new_size, ctx.cuda_stream);

  thrust::gather(rmm::exec_policy_nosync(ctx.cuda_stream),
                 build_indices_map.d_reordered_indices.begin(),
                 build_indices_map.d_reordered_indices.end(),
                 build_indices_map.d_uniq_indices.begin(), d_build_indices.begin());

  thrust::gather(rmm::exec_policy_nosync(ctx.cuda_stream),
                 probe_indices_map.d_reordered_indices.begin(),
                 probe_indices_map.d_reordered_indices.end(),
                 probe_indices_map.d_uniq_indices.begin(), d_probe_indices.begin());

  if (config_.sort_probe_indices) {
    thrust::sort_by_key(rmm::exec_policy_nosync(ctx.cuda_stream), d_probe_indices.begin(),
                        d_probe_indices.end(), d_build_indices.begin());
  }

  CUDA_CHECK(cudaMemcpyAsync(build_indices, d_build_indices.data(),
                             sizeof(uint32_t) * new_size, cudaMemcpyDeviceToHost,
                             ctx.cuda_stream));

  CUDA_CHECK(cudaMemcpyAsync(probe_indices, d_probe_indices.data(),
                             sizeof(uint32_t) * new_size, cudaMemcpyDeviceToHost,
                             ctx.cuda_stream));
  ctx.cuda_stream.synchronize();
  return new_size;
}

template <typename INDEX_IT>
void RTSpatialRefiner::buildIndicesMap(rmm::cuda_stream_view stream, INDEX_IT index_begin,
                                       INDEX_IT index_end,
                                       IndicesMap& indices_map) const {
  auto len = thrust::distance(index_begin, index_end);
  auto& d_uniq_indices = indices_map.d_uniq_indices;
  auto& h_uniq_indices = indices_map.h_uniq_indices;

  d_uniq_indices.resize(len, stream);
  CUDA_CHECK(cudaMemcpyAsync(d_uniq_indices.data(), index_begin, sizeof(uint32_t) * len,
                             cudaMemcpyDeviceToDevice, stream));

  thrust::sort(rmm::exec_policy_nosync(stream), d_uniq_indices.begin(),
               d_uniq_indices.end());
  auto uniq_end = thrust::unique(rmm::exec_policy_nosync(stream), d_uniq_indices.begin(),
                                 d_uniq_indices.end());
  auto uniq_size = thrust::distance(d_uniq_indices.begin(), uniq_end);

  d_uniq_indices.resize(uniq_size, stream);
  h_uniq_indices.resize(uniq_size);

  CUDA_CHECK(cudaMemcpyAsync(h_uniq_indices.data(), d_uniq_indices.data(),
                             sizeof(uint32_t) * uniq_size, cudaMemcpyDeviceToHost,
                             stream));

  auto& d_reordered_indices = indices_map.d_reordered_indices;

  d_reordered_indices.resize(len, stream);
  detail::ReorderIndices(stream, index_begin, index_end, d_uniq_indices,
                         d_reordered_indices);
}

std::unique_ptr<SpatialRefiner> CreateRTSpatialRefiner(
    const RTSpatialRefinerConfig& config) {
  auto refiner = std::make_unique<RTSpatialRefiner>(config);
  GPUSPATIAL_LOG_INFO(
      "Create RTSpatialRefiner %p, fast_build = %d, compact = %d, "
      "parsing_threads = %u, concurrency = %u, pipeline_batches = %u, "
      "wkb_parser_memory_quota = %.2f, relate_engine_memory_quota = %.2f",
      refiner.get(), config.prefer_fast_build, config.compact, config.parsing_threads,
      config.concurrency, config.pipeline_batches, config.wkb_parser_memory_quota,
      config.relate_engine_memory_quota);
  return std::move(refiner);
}

}  // namespace gpuspatial
