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
cmake_minimum_required(VERSION 3.14)
project(sedonadb_c_geoarrow_c)

# Note that this is not currently used by build.rs, it just exists to help
# IDEs understand the C sources
file(GLOB SOURCES src/**/*.c src/**/*.cc)
add_library(geoarrow_c ${SOURCES})
target_include_directories(geoarrow_c PRIVATE src)
target_compile_definitions(geoarrow_c PUBLIC GEOARROW_NAMESPACE=SedonaDB
                                             NANOARROW_NAMESPACE=SedonaDB)
