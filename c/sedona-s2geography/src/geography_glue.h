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

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// \file geography_glue.h
///
/// This file exposes C functions and/or data structures used to call
/// s2geography from Rust. Ideally most logic is implemented in
/// s2geography; however, this header is the one that is parsed by
/// bindgen and wrapped in the Rust bindings.
///
/// These functions are internal. See s2geography.rs for the
/// user-facing documentation of these functions.

const char* SedonaGeographyGlueNanoarrowVersion(void);

const char* SedonaGeographyGlueGeoArrowVersion(void);

const char* SedonaGeographyGlueOpenSSLVersion(void);

const char* SedonaGeographyGlueS2GeometryVersion(void);

const char* SedonaGeographyGlueAbseilVersion(void);

double SedonaGeographyGlueTestLinkage(void);

uint64_t SedonaGeographyGlueLngLatToCellId(double lng, double lat);

size_t SedonaGeographyGlueNumKernels(void);

int SedonaGeographyGlueInitKernels(void* kernels_array, size_t kernels_size_bytes);

#ifdef __cplusplus
}
#endif
