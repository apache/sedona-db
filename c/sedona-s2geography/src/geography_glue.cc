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
#include <cmath>
#include <cstdint>
#include <string>

#include "absl/base/config.h"

#include <openssl/opensslv.h>
#include <s2geography/accessors-geog.h>
#include <s2geography/accessors.h>
#include <s2geography/build.h>
#include <s2geography/coverings.h>
#include <s2geography/distance.h>
#include <s2geography/linear-referencing.h>
#include <s2geography/predicates.h>
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

uint64_t SedonaGeographyGlueLngLatToCellId(double lng, double lat) {
  if (std::isnan(lng) || std::isnan(lat)) {
    return S2CellId::Sentinel().id();
  } else {
    return S2CellId(S2LatLng::FromDegrees(lat, lng).Normalized().ToPoint()).id();
  }
}

size_t SedonaGeographyGlueNumKernels(void) { return 27; }

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
  s2geography::sedona_udf::ReducePrecisionKernel(kernel_ptr++);
  s2geography::sedona_udf::SimplifyKernel(kernel_ptr++);
  s2geography::sedona_udf::BufferKernel(kernel_ptr++);
  s2geography::sedona_udf::BufferQuadSegsKernel(kernel_ptr++);
  s2geography::sedona_udf::BufferParamsKernel(kernel_ptr++);
  s2geography::sedona_udf::DistanceWithinKernel(kernel_ptr++);
  s2geography::sedona_udf::CellIdFromPointKernel(kernel_ptr++);
  s2geography::sedona_udf::CoveringCellIdsKernel(kernel_ptr++);
  s2geography::sedona_udf::LongestLineKernel(kernel_ptr++);

  return 0;
}

/// Opaque wrapper around a GeoArrowGeography for C API
struct SedonaGeography {
  s2geography::GeoArrowGeography geog;
};

int SedonaGeographyCreateFromWkbNonOwning(struct SedonaGeography** geog,
                                          const uint8_t* buf, size_t len) {
  // // Parse WKB into a geometry view
  // struct GeoArrowWKBReader reader;
  // GeoArrowErrorCode code = GeoArrowWKBReaderInit(&reader);
  // if (code != GEOARROW_OK) {
  //   return EINVAL;
  // }

  // struct GeoArrowBufferView src{buf, static_cast<int64_t>(len)};
  // struct GeoArrowGeometryView view;
  // code = GeoArrowWKBReaderRead(&reader, src, &view, nullptr);
  // if (code != GEOARROW_OK) {
  //   GeoArrowWKBReaderReset(&reader);
  //   return EINVAL;
  // }

  // // Create SedonaGeography and initialize from view
  // // The GeoArrowGeography references the original WKB buffer (non-owning)
  // auto* result = new SedonaGeography();
  // result->geog.Init(view);

  // GeoArrowWKBReaderReset(&reader);

  // *geog = result;
  return 0;
}

int SedonaGeographyGetBounds(const struct SedonaGeography* geog, double* xmin,
                             double* ymin, double* xmax, double* ymax) {
  s2geography::LatLngRectBounder bounder;
  bounder.Clear();
  bounder.Update(geog->geog);
  S2LatLngRect bounds = bounder.Finish();

  if (bounds.is_empty()) {
    *xmin = *ymin = *xmax = *ymax = std::nan("");
    return 0;
  }

  *xmin = bounds.lng_lo().degrees();
  *ymin = bounds.lat_lo().degrees();
  *xmax = bounds.lng_hi().degrees();
  *ymax = bounds.lat_hi().degrees();
  return 0;
}

void SedonaGeographyDestroy(struct SedonaGeography* geog) { delete geog; }
