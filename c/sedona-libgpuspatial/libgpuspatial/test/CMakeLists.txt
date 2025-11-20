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
if(GPUSPATIAL_BUILD_TESTS)
  add_library(geoarrow_geos geoarrow_geos/geoarrow_geos.c)
  target_link_libraries(geoarrow_geos
                        PUBLIC GEOS::geos_c
                        PRIVATE geoarrow)
endif()

if(GPUSPATIAL_BUILD_TESTS)
  enable_testing()

  add_executable(gpuspatial_testing_test gpuspatial_testing_test.cc)
  target_link_libraries(gpuspatial_testing_test
                        geoarrow
                        GTest::gtest_main
                        GTest::gmock_main
                        gpuspatial)

  add_executable(array_stream_test array_stream_test.cc array_stream.cc)
  target_link_libraries(array_stream_test
                        GTest::gtest_main
                        GTest::gmock_main
                        geoarrow
                        nanoarrow::nanoarrow
                        nanoarrow::nanoarrow_ipc)

  add_executable(loader_test array_stream.cc main.cc loader_test.cu)
  target_link_libraries(loader_test
                        cuda
                        GTest::gtest_main
                        GTest::gmock_main
                        gpuspatial
                        GEOS::geos
                        GEOS::geos_c
                        nanoarrow::nanoarrow_ipc)
  target_include_directories(loader_test PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_compile_options(loader_test
                         PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda
                                 --expt-relaxed-constexpr>)

  add_executable(joiner_test array_stream.cc main.cc joiner_test.cu)
  target_link_libraries(joiner_test
                        cuda
                        GTest::gtest_main
                        GTest::gmock_main
                        gpuspatial
                        GEOS::geos
                        GEOS::geos_c
                        geoarrow_geos
                        Arrow::arrow_static
                        Parquet::parquet_static
                        nanoarrow::nanoarrow_ipc)
  target_include_directories(joiner_test PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_compile_options(joiner_test
                         PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda
                                 --expt-relaxed-constexpr>)

  add_executable(relate_test main.cc array_stream.cc related_test.cu)
  target_link_libraries(relate_test
                        PRIVATE cuda
                                GTest::gtest_main
                                GTest::gmock_main
                                gpuspatial
                                GEOS::geos
                                nanoarrow::nanoarrow
                                nanoarrow::nanoarrow_ipc)
  target_compile_options(relate_test
                         PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda
                                 --expt-relaxed-constexpr>)

  add_executable(c_wrapper_test main.cc c_wrapper_test.cc array_stream.cc)
  target_link_libraries(c_wrapper_test PRIVATE GTest::gtest_main GTest::gmock_main
                                               gpuspatial_c nanoarrow::nanoarrow_ipc)

  include(GoogleTest)

  gtest_discover_tests(gpuspatial_testing_test)
  gtest_discover_tests(array_stream_test)
  gtest_discover_tests(loader_test)
  gtest_discover_tests(joiner_test)
  gtest_discover_tests(relate_test)
endif()
