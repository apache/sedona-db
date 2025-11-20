# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# This function finds geoarrow and sets any additional necessary environment variables.
function(find_and_configure_geoarrow)
  if(NOT BUILD_SHARED_LIBS)
    set(_exclude_from_all EXCLUDE_FROM_ALL FALSE)
  else()
    set(_exclude_from_all EXCLUDE_FROM_ALL TRUE)
  endif()

  # Currently we need to always build geoarrow so we don't pickup a previous installed version
  set(CPM_DOWNLOAD_geoarrow ON)
  rapids_cpm_find(geoarrow
                  geoarrow-c-python-0.3.1
                  GLOBAL_TARGETS
                  geoarrow
                  CPM_ARGS
                  GIT_REPOSITORY
                  https://github.com/geoarrow/geoarrow-c.git
                  GIT_TAG
                  eae46da505d9a5a8c156fc6bbb80798f2cb4a3d0
                  GIT_SHALLOW
                  FALSE
                  OPTIONS
                  "BUILD_SHARED_LIBS OFF"
                  ${_exclude_from_all})
  set_target_properties(geoarrow PROPERTIES POSITION_INDEPENDENT_CODE ON)
  rapids_export_find_package_root(BUILD
                                  geoarrow
                                  "${geoarrow_BINARY_DIR}"
                                  EXPORT_SET
                                  gpuspatial-exports)
endfunction()

find_and_configure_geoarrow()
