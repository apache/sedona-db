
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
#include "gpuspatial/index/detail/launch_parameters.h"
#include "gpuspatial/index/relate_engine.cuh"
#include "gpuspatial/index/spatial_joiner.cuh"
#include "gpuspatial/loader/parallel_wkb_loader.h"
#include "gpuspatial/utils/logger.hpp"
#include "gpuspatial/utils/stopwatch.h"

#include "rt/shaders/shader_id.hpp"

#include "rmm/exec_policy.hpp"

#define OPTIX_MAX_RAYS (1lu << 30)
namespace gpuspatial {

namespace detail {

template <int N_DIM>
static rmm::device_uvector<OptixAabb> ComputeAABBs(
    rmm::cuda_stream_view stream, const ArrayView<Box<Point<float, N_DIM>>>& mbrs) {
  rmm::device_uvector<OptixAabb> aabbs(mbrs.size(), stream);

  thrust::transform(rmm::exec_policy_nosync(stream), mbrs.begin(), mbrs.end(),
                    aabbs.begin(), [] __device__(const Box<Point<float, N_DIM>>& mbr) {
                      OptixAabb aabb{0, 0, 0, 0, 0, 0};
                      auto min_corner = mbr.get_min();
                      auto max_corner = mbr.get_max();
                      for (int dim = 0; dim < N_DIM; dim++) {
                        (&aabb.minX)[dim] = min_corner[dim];
                        (&aabb.maxX)[dim] = max_corner[dim];
                      }
                      return aabb;
                    });
  return std::move(aabbs);
}

}  // namespace detail

void SpatialJoiner::Init(const Config* config) {
  config_ = *dynamic_cast<const SpatialJoinerConfig*>(config);
  GPUSPATIAL_LOG_INFO("SpatialJoiner %p (Free %zu MB), Initialize, Concurrency %u", this,
                      rmm::available_device_memory().first / 1024 / 1024,
                      config_.concurrency);
  details::RTConfig rt_config = details::get_default_rt_config(config_.ptx_root);
  rt_engine_.Init(rt_config);

  loader_t::Config loader_config;

  thread_pool_ = std::make_shared<ThreadPool>(config_.parsing_threads);
  build_loader_ = std::make_unique<loader_t>(thread_pool_);
  build_loader_->Init(loader_config);
  stream_pool_ = std::make_unique<rmm::cuda_stream_pool>(config_.concurrency);
  ctx_pool_ = ObjectPool<SpatialJoinerContext>::create(config_.concurrency);
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, config_.stack_size_bytes));
  Clear();
}

void SpatialJoiner::Clear() {
  GPUSPATIAL_LOG_INFO("SpatialJoiner %p (Free %zu MB), Clear", this,
                      rmm::available_device_memory().first / 1024 / 1024);
  bvh_buffer_ = nullptr;
  geometry_grouper_.Clear();
  auto stream = rmm::cuda_stream_default;
  build_loader_->Clear(stream);
  build_geometries_.Clear(stream);
  stream.synchronize();
}

void SpatialJoiner::PushBuild(const ArrowSchema* schema, const ArrowArray* array,
                              int64_t offset, int64_t length) {
  GPUSPATIAL_LOG_INFO("SpatialJoiner %p (Free %zu MB), PushBuild, offset %ld, length %ld",
                      this, rmm::available_device_memory().first / 1024 / 1024, offset,
                      length);
  build_loader_->Parse(rmm::cuda_stream_default, array, offset, length);
}

void SpatialJoiner::FinishBuilding() {
  auto stream = rmm::cuda_stream_default;

  build_geometries_ = std::move(build_loader_->Finish(stream));

  GPUSPATIAL_LOG_INFO(
      "SpatialJoiner %p (Free %zu MB), FinishBuilding, n_features: %ld, type %s", this,
      rmm::available_device_memory().first / 1024 / 1024,
      build_geometries_.num_features(),
      GeometryTypeToString(build_geometries_.get_geometry_type()));

  if (build_geometries_.get_geometry_type() == GeometryType::kPoint) {
    geometry_grouper_.Group(stream, build_geometries_, config_.n_points_per_aabb);
    handle_ = buildBVH(stream, geometry_grouper_.get_aabbs(), bvh_buffer_);
  } else {
    auto aabbs = detail::ComputeAABBs(stream, build_geometries_.get_mbrs());
    handle_ = buildBVH(stream, ArrayView<OptixAabb>(aabbs), bvh_buffer_);
  }

  relate_engine_ = RelateEngine(&build_geometries_, &rt_engine_);
  RelateEngine<point_t, index_t>::Config re_config;

  re_config.memory_quota = config_.relate_engine_memory_quota;
  re_config.bvh_fast_build = config_.prefer_fast_build;
  re_config.bvh_fast_compact = config_.compact;

  relate_engine_.set_config(re_config);
}

void SpatialJoiner::PushStream(Context* base_ctx, const ArrowSchema* schema,
                               const ArrowArray* array, int64_t offset, int64_t length,
                               Predicate predicate, std::vector<uint32_t>* build_indices,
                               std::vector<uint32_t>* stream_indices,
                               int32_t array_index_offset) {
  auto* ctx = (SpatialJoinerContext*)base_ctx;
  ctx->cuda_stream = stream_pool_->get_stream();

#ifdef GPUSPATIAL_PROFILING
  Stopwatch sw;
  sw.start();
#endif
  ctx->array_index_offset = array_index_offset;

  if (ctx->stream_loader == nullptr) {
    ctx->stream_loader = std::make_unique<loader_t>(thread_pool_);
    loader_t::Config loader_config;

    ctx->stream_loader->Init(loader_config);
  }
  ctx->stream_loader->Parse(ctx->cuda_stream, array, offset, length);
  ctx->stream_geometries = std::move(ctx->stream_loader->Finish(ctx->cuda_stream));

  auto build_type = build_geometries_.get_geometry_type();
  auto stream_type = ctx->stream_geometries.get_geometry_type();

  GPUSPATIAL_LOG_INFO(
      "SpatialJoiner %p, PushStream, build features %zu, type %s, stream features %zu, type %s",
      this, build_geometries_.num_features(),
      GeometryTypeToString(build_geometries_.get_geometry_type()),
      ctx->stream_geometries.num_features(),
      GeometryTypeToString(ctx->stream_geometries.get_geometry_type()));

#ifdef GPUSPATIAL_PROFILING
  sw.stop();
  ctx->parse_ms += sw.ms();
#endif

  if (build_type == GeometryType::kPoint) {
    if (stream_type == GeometryType::kPoint) {
      handleBuildPointStreamPoint(ctx, predicate, build_indices, stream_indices);
    } else {
      handleBuildPointStreamBox(ctx, predicate, build_indices, stream_indices);
    }
  } else {
    if (stream_type == GeometryType::kPoint) {
      handleBuildBoxStreamPoint(ctx, predicate, build_indices, stream_indices);
    } else {
      handleBuildBoxStreamBox(ctx, predicate, build_indices, stream_indices);
    }
  }
#ifdef GPUSPATIAL_PROFILING
  printf("parse %lf, alloc %lf, filter %lf, refine %lf, copy_res %lf ms\n", ctx->parse_ms,
         ctx->alloc_ms, ctx->filter_ms, ctx->refine_ms, ctx->copy_res_ms);
#endif
}

void SpatialJoiner::handleBuildPointStreamPoint(SpatialJoinerContext* ctx,
                                                Predicate predicate,
                                                std::vector<uint32_t>* build_indices,
                                                std::vector<uint32_t>* stream_indices) {
  allocateResultBuffer(ctx);

  ctx->shader_id = GetPointQueryShaderId<point_t>();
  assert(ctx->stream_geometries.get_geometry_type() == GeometryType::kPoint);

  using launch_params_t = detail::LaunchParamsPointQuery<point_t>;
  ctx->launch_params_buffer =
      std::make_unique<rmm::device_buffer>(sizeof(launch_params_t), ctx->cuda_stream);
  ctx->h_launch_params_buffer.resize(sizeof(launch_params_t));
  auto& launch_params = *(launch_params_t*)ctx->h_launch_params_buffer.data();

  launch_params.grouped = true;
  launch_params.prefix_sum = geometry_grouper_.get_prefix_sum();
  launch_params.reordered_indices = geometry_grouper_.get_reordered_indices();
  launch_params.mbrs1 = ArrayView<box_t>();  // no MBRs for point
  launch_params.points2 = ctx->stream_geometries.get_points();
  launch_params.handle = handle_;
  launch_params.ids = ctx->results.DeviceObject();
  CUDA_CHECK(cudaMemcpyAsync(ctx->launch_params_buffer->data(), &launch_params,
                             sizeof(launch_params_t), cudaMemcpyHostToDevice,
                             ctx->cuda_stream));

  uint32_t dim_x = std::min(OPTIX_MAX_RAYS, ctx->stream_geometries.num_features());

  filter(ctx, dim_x);
  refine(ctx, predicate, build_indices, stream_indices);
}

void SpatialJoiner::handleBuildBoxStreamPoint(SpatialJoinerContext* ctx,
                                              Predicate predicate,
                                              std::vector<uint32_t>* build_indices,
                                              std::vector<uint32_t>* stream_indices) {
  allocateResultBuffer(ctx);

  ctx->shader_id = GetPointQueryShaderId<point_t>();
  assert(ctx->stream_geometries.get_geometry_type() == GeometryType::kPoint);

  using launch_params_t = detail::LaunchParamsPointQuery<point_t>;
  ctx->launch_params_buffer =
      std::make_unique<rmm::device_buffer>(sizeof(launch_params_t), ctx->cuda_stream);
  ctx->h_launch_params_buffer.resize(sizeof(launch_params_t));
  auto& launch_params = *(launch_params_t*)ctx->h_launch_params_buffer.data();

  launch_params.grouped = false;
  launch_params.mbrs1 = build_geometries_.get_mbrs();
  launch_params.points2 = ctx->stream_geometries.get_points();
  launch_params.handle = handle_;
  launch_params.ids = ctx->results.DeviceObject();
  CUDA_CHECK(cudaMemcpyAsync(ctx->launch_params_buffer->data(), &launch_params,
                             sizeof(launch_params_t), cudaMemcpyHostToDevice,
                             ctx->cuda_stream));

  uint32_t dim_x = std::min(OPTIX_MAX_RAYS, ctx->stream_geometries.num_features());

  filter(ctx, dim_x);
  refine(ctx, predicate, build_indices, stream_indices);
}

void SpatialJoiner::handleBuildPointStreamBox(SpatialJoinerContext* ctx,
                                              Predicate predicate,
                                              std::vector<uint32_t>* build_indices,
                                              std::vector<uint32_t>* stream_indices) {
  allocateResultBuffer(ctx);

  ctx->shader_id = GetPointQueryShaderId<point_t>();
  assert(build_geometries_.get_geometry_type() == GeometryType::kPoint);

  using launch_params_t = detail::LaunchParamsPointQuery<point_t>;
  ctx->launch_params_buffer =
      std::make_unique<rmm::device_buffer>(sizeof(launch_params_t), ctx->cuda_stream);
  ctx->h_launch_params_buffer.resize(sizeof(launch_params_t));
  auto& launch_params = *(launch_params_t*)ctx->h_launch_params_buffer.data();

  auto aabbs = detail::ComputeAABBs(ctx->cuda_stream, ctx->stream_geometries.get_mbrs());
  auto handle = buildBVH(ctx->cuda_stream, ArrayView<OptixAabb>(aabbs), ctx->bvh_buffer);

  // mbrs1 are from stream; points2 are from build
  launch_params.grouped = false;
  launch_params.mbrs1 = ctx->stream_geometries.get_mbrs();
  launch_params.points2 = build_geometries_.get_points();
  launch_params.handle = handle;
  launch_params.ids = ctx->results.DeviceObject();
  CUDA_CHECK(cudaMemcpyAsync(ctx->launch_params_buffer->data(), &launch_params,
                             sizeof(launch_params_t), cudaMemcpyHostToDevice,
                             ctx->cuda_stream));

  uint32_t dim_x = std::min(OPTIX_MAX_RAYS, build_geometries_.num_features());
  // IMPORTANT: In this case, the BVH is built from stream geometries and points2 are
  // build geometries, so the result pairs are (stream_id, build_id) instead of (build_id,
  // stream_id). We need to swap the output buffers to correct this.
  filter(ctx, dim_x, true);
  refine(ctx, predicate, build_indices, stream_indices);
}

void SpatialJoiner::handleBuildBoxStreamBox(SpatialJoinerContext* ctx,
                                            Predicate predicate,
                                            std::vector<uint32_t>* build_indices,
                                            std::vector<uint32_t>* stream_indices) {
  allocateResultBuffer(ctx);

  // forward cast: cast rays from stream geometries with the BVH of build geometries
  {
    auto dim_x = std::min(OPTIX_MAX_RAYS, ctx->stream_geometries.num_features());

    prepareLaunchParamsBoxQuery(ctx, true);
    filter(ctx, dim_x);
    refine(ctx, predicate, build_indices, stream_indices);
    ctx->results.Clear(ctx->cuda_stream);  // results have been copied, reuse space
  }
  // need allocate again as the previous results buffer has been shrinked to fit
  allocateResultBuffer(ctx);
  // backward cast: cast rays from the build geometries with the BVH of stream geometries
  {
    auto dim_x = std::min(OPTIX_MAX_RAYS, build_geometries_.num_features());
    auto v_mbrs = ctx->stream_geometries.get_mbrs();
    rmm::device_uvector<OptixAabb> aabbs(v_mbrs.size(), ctx->cuda_stream);

    thrust::transform(rmm::exec_policy_nosync(ctx->cuda_stream), v_mbrs.begin(),
                      v_mbrs.end(), aabbs.begin(),
                      [] __device__(const box_t& mbr) { return mbr.ToOptixAabb(); });

    // Build a BVH over the MBRs of the stream geometries
    ctx->handle =
        buildBVH(ctx->cuda_stream, ArrayView<OptixAabb>(aabbs.data(), aabbs.size()),
                 ctx->bvh_buffer);
    prepareLaunchParamsBoxQuery(ctx, false);
    filter(ctx, dim_x);
    refine(ctx, predicate, build_indices, stream_indices);
  }
}

OptixTraversableHandle SpatialJoiner::buildBVH(
    const rmm::cuda_stream_view& stream, const ArrayView<OptixAabb>& aabbs,
    std::unique_ptr<rmm::device_buffer>& buffer) {
  auto buffer_size_bytes = rt_engine_.EstimateMemoryUsageForAABB(
      aabbs.size(), config_.prefer_fast_build, config_.compact);

  if (buffer == nullptr || buffer->size() < buffer_size_bytes) {
    buffer = std::make_unique<rmm::device_buffer>(buffer_size_bytes, stream);
  }

  return rt_engine_.BuildAccelCustom(stream, aabbs, *buffer, config_.prefer_fast_build,
                                     config_.compact);
}

void SpatialJoiner::allocateResultBuffer(SpatialJoinerContext* ctx) {
#ifdef GPUSPATIAL_PROFILING
  ctx->timer.start(ctx->cuda_stream);
#endif
  int64_t avail_bytes = rmm::available_device_memory().first;
  auto stream_type = ctx->stream_geometries.get_geometry_type();
  if (stream_type != GeometryType::kPoint) {
    // need to reserve space for the BVH of stream
    auto n_aabbs = ctx->stream_geometries.get_mbrs().size();

    avail_bytes -= rt_engine_.EstimateMemoryUsageForAABB(
        n_aabbs, config_.prefer_fast_build, config_.compact);
  }

  if (avail_bytes <= 0) {
    throw std::runtime_error(
        "Not enough memory to allocate result space for spatial index");
  }

  uint64_t reserve_bytes = ceil(avail_bytes * config_.result_buffer_memory_reserve_ratio);
  reserve_bytes = reserve_bytes / config_.concurrency + 1;
  // two index_t for each result pair (build index, stream index) and another index_t for
  // the temp storage
  uint32_t n_items = reserve_bytes / (2 * sizeof(index_t) + sizeof(index_t));

  GPUSPATIAL_LOG_INFO(
      "SpatialJoiner %p, Allocate result buffer quota %zu MB, queue size %u", this,
      reserve_bytes / 1024 / 1024, n_items);

  ctx->results.Init(ctx->cuda_stream, n_items);
  ctx->results.Clear(ctx->cuda_stream);
#ifdef GPUSPATIAL_PROFILING
  ctx->alloc_ms += ctx->timer.stop(ctx->cuda_stream);
#endif
}

void SpatialJoiner::prepareLaunchParamsBoxQuery(SpatialJoinerContext* ctx, bool foward) {
  using launch_params_t = detail::LaunchParamsBoxQuery<point_t>;
  ctx->launch_params_buffer =
      std::make_unique<rmm::device_buffer>(sizeof(launch_params_t), ctx->cuda_stream);
  ctx->h_launch_params_buffer.resize(sizeof(launch_params_t));
  auto& launch_params = *(launch_params_t*)ctx->h_launch_params_buffer.data();

  assert(ctx->stream_geometries.get_geometry_type() != GeometryType::kPoint);

  launch_params.mbrs1 = build_geometries_.get_mbrs();
  launch_params.mbrs2 = ctx->stream_geometries.get_mbrs();
  if (foward) {
    launch_params.handle = handle_;
    ctx->shader_id = GetBoxQueryForwardShaderId<point_t>();
  } else {
    launch_params.handle = ctx->handle;
    ctx->shader_id = GetBoxQueryBackwardShaderId<point_t>();
  }

  launch_params.ids = ctx->results.DeviceObject();
  CUDA_CHECK(cudaMemcpyAsync(ctx->launch_params_buffer->data(), &launch_params,
                             sizeof(launch_params_t), cudaMemcpyHostToDevice,
                             ctx->cuda_stream));
}

void SpatialJoiner::filter(SpatialJoinerContext* ctx, uint32_t dim_x, bool swap_id) {
#ifdef GPUSPATIAL_PROFILING
  ctx->timer.start(ctx->cuda_stream);
#endif
  Stopwatch sw;
  sw.start();
  if (dim_x > 0) {
    rt_engine_.Render(ctx->cuda_stream, ctx->shader_id, dim3{dim_x, 1, 1},
                      ArrayView<char>((char*)ctx->launch_params_buffer->data(),
                                      ctx->launch_params_buffer->size()));
  }
  auto result_size = ctx->results.size(ctx->cuda_stream);
  sw.stop();
  GPUSPATIAL_LOG_INFO(
      "SpatialJoiner %p, Filter stage, Launched %u rays, Found %u candidates, time %lf ms",
      this, dim_x, result_size, sw.ms());
  if (swap_id && result_size > 0) {
    // swap the pair (build_id, stream_id) to (stream_id, build_id)
    thrust::for_each(rmm::exec_policy_nosync(ctx->cuda_stream), ctx->results.data(),
                     ctx->results.data() + result_size,
                     [] __device__(thrust::pair<uint32_t, uint32_t> & pair) {
                       thrust::swap(pair.first, pair.second);
                     });
  }
  ctx->results.shrink_to_fit(ctx->cuda_stream);

#ifdef GPUSPATIAL_PROFILING
  ctx->filter_ms += ctx->timer.stop(ctx->cuda_stream);
#endif
}

void SpatialJoiner::refine(SpatialJoinerContext* ctx, Predicate predicate,
                           std::vector<uint32_t>* build_indices,
                           std::vector<uint32_t>* stream_indices) {
#ifdef GPUSPATIAL_PROFILING
  ctx->timer.start(ctx->cuda_stream);
#endif
  relate_engine_.Evaluate(ctx->cuda_stream, ctx->stream_geometries, predicate,
                          ctx->results);
#ifdef GPUSPATIAL_PROFILING
  ctx->refine_ms += ctx->timer.stop(ctx->cuda_stream);
#endif
  auto n_results = ctx->results.size(ctx->cuda_stream);

#ifdef GPUSPATIAL_PROFILING
  ctx->timer.start(ctx->cuda_stream);
#endif
  rmm::device_uvector<uint32_t> tmp_result_buffer(n_results, ctx->cuda_stream);

  thrust::transform(
      rmm::exec_policy_nosync(ctx->cuda_stream), ctx->results.data(),
      ctx->results.data() + n_results, tmp_result_buffer.begin(),
      [] __device__(const thrust::pair<index_t, index_t>& pair) -> uint32_t {
        return pair.first;
      });
  auto prev_size = build_indices->size();
  build_indices->resize(build_indices->size() + n_results);

  CUDA_CHECK(cudaMemcpyAsync(build_indices->data() + prev_size, tmp_result_buffer.data(),
                             sizeof(uint32_t) * n_results, cudaMemcpyDeviceToHost,
                             ctx->cuda_stream));

  auto array_index_offset = ctx->array_index_offset;

  thrust::transform(
      rmm::exec_policy_nosync(ctx->cuda_stream), ctx->results.data(),
      ctx->results.data() + n_results, tmp_result_buffer.begin(),
      [=] __device__(const thrust::pair<index_t, index_t>& pair) -> uint32_t {
        return pair.second + array_index_offset;
      });

  stream_indices->resize(stream_indices->size() + n_results);

  CUDA_CHECK(cudaMemcpyAsync(stream_indices->data() + prev_size, tmp_result_buffer.data(),
                             sizeof(uint32_t) * n_results, cudaMemcpyDeviceToHost,
                             ctx->cuda_stream));
#ifdef GPUSPATIAL_PROFILING
  ctx->copy_res_ms += ctx->timer.stop(ctx->cuda_stream);
#endif
  ctx->cuda_stream.synchronize();
}

std::unique_ptr<StreamingJoiner> CreateSpatialJoiner() {
  return std::make_unique<SpatialJoiner>();
}

void InitSpatialJoiner(StreamingJoiner* index, const char* ptx_root,
                       uint32_t concurrency) {
  SpatialJoiner::SpatialJoinerConfig config;
  config.ptx_root = ptx_root;
  config.concurrency = concurrency;
  index->Init(&config);
}

}  // namespace gpuspatial
