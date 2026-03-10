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

#include "geography_glue.h"

#include <cerrno>
#include <cstdint>

#include "absl/base/config.h"

#include <openssl/opensslv.h>
#include <s2geography.h>
#include <s2geography/sedona_udf/sedona_extension.h>

#include <s2/s2earth.h>

using namespace s2geography;

const char* SedonaGeographyGlueOpenSSLVersion(void) {
  static std::string version = std::string() + std::to_string(OPENSSL_VERSION_MAJOR) +
                               "." + std::to_string(OPENSSL_VERSION_MINOR) + "." +
                               std::to_string(OPENSSL_VERSION_PATCH);
  return version.c_str();
}

const char* SedonaGeographyGlueS2GeometryVersion(void) {
  static std::string version = std::string() + std::to_string(S2_VERSION_MAJOR) + "." +
                               std::to_string(S2_VERSION_MINOR) + "." +
                               std::to_string(S2_VERSION_PATCH);
  return version.c_str();
}

const char* SedonaGeographyGlueAbseilVersion(void) {
#if defined(ABSL_LTS_RELEASE_VERSION)
  static std::string version = std::string() + std::to_string(ABSL_LTS_RELEASE_VERSION) +
                               "." + std::to_string(ABSL_LTS_RELEASE_PATCH_LEVEL);
  return version.c_str();
#else
  return "<live at head>";
#endif
}

double SedonaGeographyGlueTestLinkage(void) {
  PointGeography geog1(S2LatLng::FromDegrees(45, -64).ToPoint());
  PointGeography geog2(S2LatLng::FromDegrees(4, -97).ToPoint());

  ShapeIndexGeography index1(geog1);
  ShapeIndexGeography index2(geog2);

  return S2Earth::RadiusMeters() * s2_distance(index1, index2);
}

uint64_t SedonaGeographyGlueLngLatToCellId(double lng, double lat) {
  if (std::isnan(lng) || std::isnan(lat)) {
    return S2CellId::Sentinel().id();
  } else {
    return S2CellId(S2LatLng::FromDegrees(lat, lng).Normalized().ToPoint()).id();
  }
}

size_t SedonaGeographyGlueNumKernels(void) { return 18; }

int SedonaGeographyGlueInitKernels(void* kernels_array, size_t kerenels_size_bytes) {
  if (kerenels_size_bytes !=
      (sizeof(SedonaCScalarKernel) * SedonaGeographyGlueNumKernels())) {
    return EINVAL;
  }

  auto* kernel_ptr = reinterpret_cast<struct SedonaCScalarKernel*>(kernels_array);

  s2geography::sedona_udf::AreaKernel(kernel_ptr++);
  s2geography::sedona_udf::CentroidKernel(kernel_ptr++);
  s2geography::sedona_udf::ClosestPointKernel(kernel_ptr++);
  s2geography::sedona_udf::ContainsKernel(kernel_ptr++);
  s2geography::sedona_udf::ConvexHullKernel(kernel_ptr++);
  s2geography::sedona_udf::DifferenceKernel(kernel_ptr++);
  s2geography::sedona_udf::DistanceKernel(kernel_ptr++);
  s2geography::sedona_udf::EqualsKernel(kernel_ptr++);
  s2geography::sedona_udf::IntersectionKernel(kernel_ptr++);
  s2geography::sedona_udf::IntersectsKernel(kernel_ptr++);
  s2geography::sedona_udf::LengthKernel(kernel_ptr++);
  s2geography::sedona_udf::LineInterpolatePointKernel(kernel_ptr++);
  s2geography::sedona_udf::LineLocatePointKernel(kernel_ptr++);
  s2geography::sedona_udf::MaxDistanceKernel(kernel_ptr++);
  s2geography::sedona_udf::PerimeterKernel(kernel_ptr++);
  s2geography::sedona_udf::ShortestLineKernel(kernel_ptr++);
  s2geography::sedona_udf::SymDifferenceKernel(kernel_ptr++);
  s2geography::sedona_udf::UnionKernel(kernel_ptr++);

  return 0;
}
