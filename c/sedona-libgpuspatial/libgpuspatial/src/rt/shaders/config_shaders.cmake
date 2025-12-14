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
include(cmake/nvcuda_compile_module.cmake)

function(CONFIG_SHADERS SHADER_PTX_FILES)
  set(SHADER_POINT_TYPES "SHADER_POINT_FLOAT_2D;SHADER_POINT_DOUBLE_2D")

  set(SHADERS_DEPS "${PROJECT_SOURCE_DIR}/include/gpuspatial/geom"
                   "${PROJECT_SOURCE_DIR}/include/gpuspatial/index/detail")

  set(OUTPUT_DIR "${PROJECT_BINARY_DIR}/shaders_ptx")
  set(OPTIX_MODULE_EXTENSION ".ptx")
  set(OPTIX_PROGRAM_TARGET "--ptx")

  set(ALL_GENERATED_FILES "")

  foreach(POINT_TYPE IN LISTS SHADER_POINT_TYPES)
    nvcuda_compile_module(SOURCES
                          "${PROJECT_SOURCE_DIR}/src/rt/shaders/point_query.cu"
                          DEPENDENCIES
                          ${SHADERS_DEPS}
                          TARGET_PATH
                          "${OUTPUT_DIR}"
                          PREFIX
                          "${POINT_TYPE}_"
                          EXTENSION
                          "${OPTIX_MODULE_EXTENSION}"
                          GENERATED_FILES
                          PROGRAM_MODULES
                          NVCC_OPTIONS
                          "${OPTIX_PROGRAM_TARGET}"
                          "--gpu-architecture=compute_75"
                          "--relocatable-device-code=true"
                          "--expt-relaxed-constexpr"
                          "-Wno-deprecated-gpu-targets"
                          "-std=c++17"
                          "-I${optix_SOURCE_DIR}/include"
                          "-I${PROJECT_SOURCE_DIR}/include"
                          "-D${POINT_TYPE}")
    list(APPEND ALL_GENERATED_FILES ${PROGRAM_MODULES})

    nvcuda_compile_module(SOURCES
                          "${PROJECT_SOURCE_DIR}/src/rt/shaders/box_query_forward.cu"
                          DEPENDENCIES
                          ${SHADERS_DEPS}
                          TARGET_PATH
                          "${OUTPUT_DIR}"
                          PREFIX
                          "${POINT_TYPE}_"
                          EXTENSION
                          "${OPTIX_MODULE_EXTENSION}"
                          GENERATED_FILES
                          PROGRAM_MODULES
                          NVCC_OPTIONS
                          "${OPTIX_PROGRAM_TARGET}"
                          "--gpu-architecture=compute_75"
                          "--relocatable-device-code=true"
                          "--expt-relaxed-constexpr"
                          "-Wno-deprecated-gpu-targets"
                          "-std=c++17"
                          "-I${optix_SOURCE_DIR}/include"
                          "-I${PROJECT_SOURCE_DIR}/include"
                          "-D${POINT_TYPE}")
    list(APPEND ALL_GENERATED_FILES ${PROGRAM_MODULES})

    nvcuda_compile_module(SOURCES
                          "${PROJECT_SOURCE_DIR}/src/rt/shaders/box_query_backward.cu"
                          DEPENDENCIES
                          ${SHADERS_DEPS}
                          TARGET_PATH
                          "${OUTPUT_DIR}"
                          PREFIX
                          "${POINT_TYPE}_"
                          EXTENSION
                          "${OPTIX_MODULE_EXTENSION}"
                          GENERATED_FILES
                          PROGRAM_MODULES
                          NVCC_OPTIONS
                          "${OPTIX_PROGRAM_TARGET}"
                          "--gpu-architecture=compute_75"
                          "--relocatable-device-code=true"
                          "--expt-relaxed-constexpr"
                          "-Wno-deprecated-gpu-targets"
                          "-std=c++17"
                          "-I${optix_SOURCE_DIR}/include"
                          "-I${PROJECT_SOURCE_DIR}/include"
                          "-D${POINT_TYPE}")
    list(APPEND ALL_GENERATED_FILES ${PROGRAM_MODULES})

    nvcuda_compile_module(SOURCES
                          "${PROJECT_SOURCE_DIR}/src/rt/shaders/polygon_point_query.cu"
                          DEPENDENCIES
                          ${SHADERS_DEPS}
                          TARGET_PATH
                          "${OUTPUT_DIR}"
                          PREFIX
                          "${POINT_TYPE}_"
                          EXTENSION
                          "${OPTIX_MODULE_EXTENSION}"
                          GENERATED_FILES
                          PROGRAM_MODULES
                          NVCC_OPTIONS
                          "${OPTIX_PROGRAM_TARGET}"
                          "--gpu-architecture=compute_75"
                          "--relocatable-device-code=true"
                          "--expt-relaxed-constexpr"
                          "-Wno-deprecated-gpu-targets"
                          "-std=c++17"
                          "-I${optix_SOURCE_DIR}/include"
                          "-I${PROJECT_SOURCE_DIR}/include"
                          "-D${POINT_TYPE}")
    list(APPEND ALL_GENERATED_FILES ${PROGRAM_MODULES})

    nvcuda_compile_module(SOURCES
                          "${PROJECT_SOURCE_DIR}/src/rt/shaders/multipolygon_point_query.cu"
                          DEPENDENCIES
                          ${SHADERS_DEPS}
                          TARGET_PATH
                          "${OUTPUT_DIR}"
                          PREFIX
                          "${POINT_TYPE}_"
                          EXTENSION
                          "${OPTIX_MODULE_EXTENSION}"
                          GENERATED_FILES
                          PROGRAM_MODULES
                          NVCC_OPTIONS
                          "${OPTIX_PROGRAM_TARGET}"
                          "--gpu-architecture=compute_75"
                          "--relocatable-device-code=true"
                          "--expt-relaxed-constexpr"
                          "-Wno-deprecated-gpu-targets"
                          "-std=c++17"
                          "-I${optix_SOURCE_DIR}/include"
                          "-I${PROJECT_SOURCE_DIR}/include"
                          "-D${POINT_TYPE}")
    list(APPEND ALL_GENERATED_FILES ${PROGRAM_MODULES})
  endforeach()
  set(${SHADER_PTX_FILES}
      ${ALL_GENERATED_FILES}
      PARENT_SCOPE)
endfunction()
