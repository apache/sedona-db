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

set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)
set(VCPKG_BUILD_TYPE release)
set(VCPKG_CMAKE_SYSTEM_NAME MinGW)

set(VCPKG_ENV_PASSTHROUGH
    PATH
    SEDONADB_R_CC
    SEDONADB_R_CXX
    SEDONADB_R_CFLAGS
    SEDONADB_R_CXXFLAGS
    SEDONADB_R_LDFLAGS)

set(VCPKG_C_FLAGS_RELEASE "$ENV{SEDONADB_R_CFLAGS}")
set(VCPKG_CXX_FLAGS_RELEASE "$ENV{SEDONADB_R_CXXFLAGS} -D_USE_MATH_DEFINES")
set(VCPKG_LINKER_FLAGS_RELEASE "$ENV{SEDONADB_R_LDFLAGS}")
set(VCPKG_CHAINLOAD_TOOLCHAIN_FILE "${CMAKE_CURRENT_LIST_DIR}/rtools-mingw.cmake")
