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

#include <cstdint>

#include "absl/base/config.h"

#include <geoarrow/geoarrow.h>
#include <nanoarrow/nanoarrow.h>
#include <openssl/opensslv.h>
#include <s2geography.h>
#include <s2geography/arrow_udf/arrow_udf.h>

#include <s2/s2earth.h>

using namespace s2geography;

const char* SedonaGeographyGlueNanoarrowVersion(void) { return ArrowNanoarrowVersion(); }

const char* SedonaGeographyGlueGeoArrowVersion(void) { return GeoArrowVersion(); }

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

struct UdfExporter {
  static void Export(std::unique_ptr<s2geography::arrow_udf::ArrowUDF> udf,
                     struct SedonaGeographyArrowUdf* out) {
    out->private_data = udf.release();
    out->init = &CInit;
    out->execute = &CExecute;
    out->get_last_error = &CLastError;
    out->release = &CRelease;
  }

  static int CInit(struct SedonaGeographyArrowUdf* self, struct ArrowSchema* arg_schema,
                   const char* options, struct ArrowSchema* out) {
    auto udf = reinterpret_cast<s2geography::arrow_udf::ArrowUDF*>(self->private_data);
    return udf->Init(arg_schema, options, out);
  }
  static int CExecute(struct SedonaGeographyArrowUdf* self, struct ArrowArray** args,
                      int64_t n_args, struct ArrowArray* out) {
    auto udf = reinterpret_cast<s2geography::arrow_udf::ArrowUDF*>(self->private_data);
    return udf->Execute(args, n_args, out);
  }
  static const char* CLastError(struct SedonaGeographyArrowUdf* self) {
    auto udf = reinterpret_cast<s2geography::arrow_udf::ArrowUDF*>(self->private_data);
    return udf->GetLastError();
  }

  static void CRelease(struct SedonaGeographyArrowUdf* self) {
    auto udf = reinterpret_cast<s2geography::arrow_udf::ArrowUDF*>(self->private_data);
    delete udf;
    self->private_data = nullptr;
  }
};

#define INIT_UDF_IMPL(name)                                                \
  void SedonaGeographyInitUdf##name(struct SedonaGeographyArrowUdf* out) { \
    return UdfExporter::Export(s2geography::arrow_udf::name(), out);       \
  }

INIT_UDF_IMPL(Area);
INIT_UDF_IMPL(Centroid);
INIT_UDF_IMPL(ClosestPoint);
INIT_UDF_IMPL(Contains);
INIT_UDF_IMPL(ConvexHull);
INIT_UDF_IMPL(Difference);
INIT_UDF_IMPL(Distance);
INIT_UDF_IMPL(Equals);
INIT_UDF_IMPL(Intersection);
INIT_UDF_IMPL(Intersects);
INIT_UDF_IMPL(Length);
INIT_UDF_IMPL(LineInterpolatePoint);
INIT_UDF_IMPL(LineLocatePoint);
INIT_UDF_IMPL(MaxDistance);
INIT_UDF_IMPL(Perimeter);
INIT_UDF_IMPL(ShortestLine);
INIT_UDF_IMPL(SymDifference);
INIT_UDF_IMPL(Union);
