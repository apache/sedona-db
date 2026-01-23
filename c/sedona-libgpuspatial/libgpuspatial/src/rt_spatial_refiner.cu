
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
#include "gpuspatial/loader/parallel_wkb_loader.h"
#include "gpuspatial/refine/rt_spatial_refiner.cuh"
#include "gpuspatial/relate/relate_engine.cuh"
#include "gpuspatial/utils/logger.hpp"
#include "gpuspatial/utils/stopwatch.h"

#include "rt/shaders/shader_id.hpp"

#include <thrust/gather.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <locale>

#include "rmm/exec_policy.hpp"

#define OPTIX_MAX_RAYS (1lu << 30)

namespace gpuspatial {

namespace detail {

void ReorderIndices(rmm::cuda_stream_view stream, rmm::device_uvector<uint32_t>& indices,
                    rmm::device_uvector<uint32_t>& sorted_uniq_indices,
                    rmm::device_uvector<uint32_t>& reordered_indices) {
  auto sorted_begin = sorted_uniq_indices.begin();
  auto sorted_end = sorted_uniq_indices.end();
  thrust::transform(rmm::exec_policy_nosync(stream), indices.begin(), indices.end(),
                    reordered_indices.begin(), [=] __device__(uint32_t val) {
                      auto it =
                          thrust::lower_bound(thrust::seq, sorted_begin, sorted_end, val);
                      return thrust::distance(sorted_begin, it);
                    });
}
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
  SpatialRefinerContext ctx;
  ctx.cuda_stream = stream_pool_->get_stream();

  IndicesMap probe_indices_map;
  buildIndicesMap(&ctx, probe_indices, len, probe_indices_map);

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
  re_config.bvh_fast_compact = config_.compact;

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

  rmm::device_uvector<uint32_t> d_probe_indices(new_size, ctx.cuda_stream);

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

uint32_t RTSpatialRefiner::Refine(const ArrowSchema* build_schema,
                                  const ArrowArray* build_array,
                                  const ArrowSchema* probe_schema,
                                  const ArrowArray* probe_array, Predicate predicate,
                                  uint32_t* build_indices, uint32_t* probe_indices,
                                  uint32_t len) {
  if (len == 0) {
    return 0;
  }
  SpatialRefinerContext ctx;
  ctx.cuda_stream = stream_pool_->get_stream();

  IndicesMap build_indices_map, probe_indices_map;
  buildIndicesMap(&ctx, build_indices, len, build_indices_map);
  buildIndicesMap(&ctx, probe_indices, len, probe_indices_map);

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
  re_config.bvh_fast_compact = config_.compact;

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

void RTSpatialRefiner::buildIndicesMap(SpatialRefinerContext* ctx,
                                       const uint32_t* indices, size_t len,
                                       IndicesMap& indices_map) const {
  auto stream = ctx->cuda_stream;

  rmm::device_uvector<uint32_t> d_indices(len, stream);

  CUDA_CHECK(cudaMemcpyAsync(d_indices.data(), indices, sizeof(uint32_t) * len,
                             cudaMemcpyHostToDevice, stream));

  auto& d_uniq_indices = indices_map.d_uniq_indices;
  auto& h_uniq_indices = indices_map.h_uniq_indices;

  d_uniq_indices.resize(len, stream);
  CUDA_CHECK(cudaMemcpyAsync(d_uniq_indices.data(), d_indices.data(),
                             sizeof(uint32_t) * len, cudaMemcpyDeviceToDevice, stream));

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
  detail::ReorderIndices(stream, d_indices, d_uniq_indices, d_reordered_indices);
}

std::unique_ptr<SpatialRefiner> CreateRTSpatialRefiner(
    const RTSpatialRefinerConfig& config) {
  return std::make_unique<RTSpatialRefiner>(config);
}

}  // namespace gpuspatial
