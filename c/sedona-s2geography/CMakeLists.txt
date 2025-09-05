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
project(sedonadb_c_geography)

include(FetchContent)

# Required for S2
set(CMAKE_CXX_STANDARD 17)

# These are our two external dependencies that we don't build on the fly.
# These can be installed with homebrew or vcpkg (see README).
find_package(OpenSSL REQUIRED)
find_package(absl REQUIRED)

# Use FetchContent to build nanoarrow and geoarrow, being careful to namespace
# symbols so they don't conflict with other nanoarrows that might be statically
# linked into the same executable (even though we're careful to align versions
# within this workspace, we don't have control of anything outside that stack)
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

set(NANOARROW_NAMESPACE SedonaGeography)
fetchcontent_declare(nanoarrow
                     URL https://github.com/apache/arrow-nanoarrow/archive/8c4e869cc8e4920737a513bc3012780050016bc5.zip
                     URL_HASH SHA256=1c5c44ab3b67199c873cc8d0c130c27ee94d6c7d91f9a4e269bd26f5646faf6b
)

set(GEOARROW_NAMESPACE SedonaGeography)
fetchcontent_declare(geoarrow
                     URL https://github.com/geoarrow/geoarrow-c/archive/9a4ceeebb6ce4272450df5ff4a56c22cb3111cef.zip
                     URL_HASH SHA256=479066b3c89dcb86a7b2af04ef8e57ea50254f151f4d6e46fb5e241fce86feb1
)

fetchcontent_makeavailable(nanoarrow)
fetchcontent_makeavailable(geoarrow)

# S2's CMake is pretty awful so we just build the library ourselves. We pin the
# version of s2geometry for predictability (although it is also available on
# vcpkg and homebrew).
add_library(s2
            s2geometry/src/s2/encoded_s2cell_id_vector.cc
            s2geometry/src/s2/encoded_s2point_vector.cc
            s2geometry/src/s2/encoded_s2shape_index.cc
            s2geometry/src/s2/encoded_string_vector.cc
            s2geometry/src/s2/id_set_lexicon.cc
            s2geometry/src/s2/mutable_s2shape_index.cc
            s2geometry/src/s2/r2rect.cc
            s2geometry/src/s2/s1angle.cc
            s2geometry/src/s2/s1chord_angle.cc
            s2geometry/src/s2/s1interval.cc
            s2geometry/src/s2/s2boolean_operation.cc
            s2geometry/src/s2/s2buffer_operation.cc
            s2geometry/src/s2/s2builder.cc
            s2geometry/src/s2/s2builder_graph.cc
            s2geometry/src/s2/s2builderutil_closed_set_normalizer.cc
            s2geometry/src/s2/s2builderutil_find_polygon_degeneracies.cc
            s2geometry/src/s2/s2builderutil_get_snapped_winding_delta.cc
            s2geometry/src/s2/s2builderutil_lax_polygon_layer.cc
            s2geometry/src/s2/s2builderutil_lax_polyline_layer.cc
            s2geometry/src/s2/s2builderutil_s2point_vector_layer.cc
            s2geometry/src/s2/s2builderutil_s2polygon_layer.cc
            s2geometry/src/s2/s2builderutil_s2polyline_layer.cc
            s2geometry/src/s2/s2builderutil_s2polyline_vector_layer.cc
            s2geometry/src/s2/s2builderutil_snap_functions.cc
            s2geometry/src/s2/s2cap.cc
            s2geometry/src/s2/s2cell.cc
            s2geometry/src/s2/s2cell_id.cc
            s2geometry/src/s2/s2cell_index.cc
            s2geometry/src/s2/s2cell_union.cc
            s2geometry/src/s2/s2centroids.cc
            s2geometry/src/s2/s2closest_cell_query.cc
            s2geometry/src/s2/s2closest_edge_query.cc
            s2geometry/src/s2/s2closest_point_query.cc
            s2geometry/src/s2/s2contains_vertex_query.cc
            s2geometry/src/s2/s2convex_hull_query.cc
            s2geometry/src/s2/s2coords.cc
            s2geometry/src/s2/s2crossing_edge_query.cc
            s2geometry/src/s2/s2debug.cc
            s2geometry/src/s2/s2earth.cc
            s2geometry/src/s2/s2edge_clipping.cc
            s2geometry/src/s2/s2edge_crosser.cc
            s2geometry/src/s2/s2edge_crossings.cc
            s2geometry/src/s2/s2edge_distances.cc
            s2geometry/src/s2/s2edge_tessellator.cc
            s2geometry/src/s2/s2error.cc
            s2geometry/src/s2/s2furthest_edge_query.cc
            s2geometry/src/s2/s2hausdorff_distance_query.cc
            s2geometry/src/s2/s2latlng.cc
            s2geometry/src/s2/s2latlng_rect.cc
            s2geometry/src/s2/s2latlng_rect_bounder.cc
            s2geometry/src/s2/s2lax_loop_shape.cc
            s2geometry/src/s2/s2lax_polygon_shape.cc
            s2geometry/src/s2/s2lax_polyline_shape.cc
            s2geometry/src/s2/s2loop.cc
            s2geometry/src/s2/s2loop_measures.cc
            s2geometry/src/s2/s2measures.cc
            s2geometry/src/s2/s2memory_tracker.cc
            s2geometry/src/s2/s2metrics.cc
            s2geometry/src/s2/s2max_distance_targets.cc
            s2geometry/src/s2/s2min_distance_targets.cc
            s2geometry/src/s2/s2padded_cell.cc
            s2geometry/src/s2/s2point_compression.cc
            s2geometry/src/s2/s2point_region.cc
            s2geometry/src/s2/s2pointutil.cc
            s2geometry/src/s2/s2polygon.cc
            s2geometry/src/s2/s2polyline.cc
            s2geometry/src/s2/s2polyline_alignment.cc
            s2geometry/src/s2/s2polyline_measures.cc
            s2geometry/src/s2/s2polyline_simplifier.cc
            s2geometry/src/s2/s2predicates.cc
            s2geometry/src/s2/s2projections.cc
            s2geometry/src/s2/s2r2rect.cc
            s2geometry/src/s2/s2region.cc
            s2geometry/src/s2/s2region_term_indexer.cc
            s2geometry/src/s2/s2region_coverer.cc
            s2geometry/src/s2/s2region_intersection.cc
            s2geometry/src/s2/s2region_union.cc
            s2geometry/src/s2/s2shape_index.cc
            s2geometry/src/s2/s2shape_index_buffered_region.cc
            s2geometry/src/s2/s2shape_index_measures.cc
            s2geometry/src/s2/s2shape_measures.cc
            s2geometry/src/s2/s2shape_nesting_query.cc
            s2geometry/src/s2/s2shapeutil_build_polygon_boundaries.cc
            s2geometry/src/s2/s2shapeutil_coding.cc
            s2geometry/src/s2/s2shapeutil_contains_brute_force.cc
            s2geometry/src/s2/s2shapeutil_conversion.cc
            s2geometry/src/s2/s2shapeutil_edge_iterator.cc
            s2geometry/src/s2/s2shapeutil_get_reference_point.cc
            s2geometry/src/s2/s2shapeutil_visit_crossing_edge_pairs.cc
            s2geometry/src/s2/s2text_format.cc
            s2geometry/src/s2/s2wedge_relations.cc
            s2geometry/src/s2/s2winding_operation.cc
            s2geometry/src/s2/util/bits/bit-interleave.cc
            s2geometry/src/s2/util/coding/coder.cc
            s2geometry/src/s2/util/coding/varint.cc
            s2geometry/src/s2/util/math/exactfloat/exactfloat.cc
            s2geometry/src/s2/util/math/mathutil.cc
            s2geometry/src/s2/util/units/length-units.cc)

if(WIN32 AND NOT MSVC)
  # mingw (e.g., R for Windows) needs extra libraries
  target_compile_definitions(s2 PUBLIC _USE_MATH_DEFINES)
  set(S2_EXTRA_OPENSSL_LIBS
      crypt32
      z
      ws2_32
      gdi32
      crypt32)
elseif(MSVC)
  target_compile_definitions(s2 PUBLIC NOMINMAX _USE_MATH_DEFINES)
  target_compile_options(s2 PUBLIC /J)
else()
  target_compile_options(s2 PRIVATE -Wno-attributes -Wno-deprecated-declarations
                                    -Wno-pedantic)
endif()

set(ABSL_LIBRARIES
    absl::base
    absl::btree
    absl::check
    absl::config
    absl::core_headers
    absl::dynamic_annotations
    absl::endian
    absl::fixed_array
    absl::flags
    absl::flat_hash_map
    absl::flat_hash_set
    absl::hash
    absl::inlined_vector
    absl::int128
    absl::log
    absl::log_severity
    absl::memory
    absl::span
    absl::status
    absl::str_format
    absl::strings
    absl::type_traits
    absl::utility)

target_include_directories(s2
                           PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/s2geometry/src>
)
target_link_libraries(s2 ${OPENSSL_LIBRARIES} ${ABSL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

install(TARGETS s2
        EXPORT "s2geography-targets"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")

# Build s2geography
add_library(s2::s2 ALIAS s2)
set(S2_VERSION_MAJOR 0)
set(S2_VERSION_MINOR 11)
set(S2_VERSION_PATCH 1)
set(BUILD_SHARED_LIBS OFF)
add_subdirectory(s2geography)

# Add our glue library
add_library(geography_glue src/geography_glue.cc)
target_link_libraries(geography_glue
                      s2geography
                      s2
                      OpenSSL::SSL
                      OpenSSL::Crypto
                      ${S2_EXTRA_OPENSSL_LIBS}
                      geoarrow::geoarrow
                      nanoarrow::nanoarrow)

install(TARGETS geography_glue nanoarrow_static
        EXPORT "geoarrow"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")

# Write a file to the binary directory containing the resolved locations
# of OpenSSL and the libraries we need to link
file(TOUCH "${CMAKE_BINARY_DIR}/openssl_libraries.txt")
foreach(lib ${OPENSSL_LIBRARIES})
  file(APPEND "${CMAKE_BINARY_DIR}/openssl_libraries.txt" "${lib}\n")
endforeach()
install(FILES "${CMAKE_BINARY_DIR}/openssl_libraries.txt"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}")

# The Abseil libraries are totally insane to link properly once we leave
# the CMake universe, which we are about to do. The essence of the problem
# is that Abseil ships as many small libraries in the name of modularity; however,
# there are many interconnected dependencies among the components and these
# change by Abseil version and by platform. This solution writes a file
# that lists all the libraries linked in exactly the way that would be used
# to link an target that we can parse from the Rust build script. Another
# solution would be to bundle all of these static libraries and write a
# .a file (but this might not work if the absl libraries weren't static,
# as they aren't on Homebrew and linux distributions).

if(NOT WIN32)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(LINK_CXX_STANDARD_LIB "-lc++")
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(LINK_CXX_STANDARD_LIB "-lstdc++")
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    # set MSVC-specific flags if we need them
    set(LINK_CXX_STANDARD_LIB "")
  else()
    set(LINK_CXX_STANDARD_LIB "")
  endif()

  set(CMAKE_ECHO_STANDARD_LIBRARIES ${CMAKE_CXX_STANDARD_LIBRARIES})
  set(CMAKE_ECHO_FLAGS ${CMAKE_CXX_FLAGS})
  set(CMAKE_ECHO_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS})
  set(CMAKE_ECHO_IMPLICIT_LINK_DIRECTORIES ${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES})

  # Select the appropriate command based on the platform
  if(WIN32)
    set(CMAKE_ECHO_LINK_EXECUTABLE
        "cmake -E echo \"<FLAGS> <LINK_FLAGS> <LINK_LIBRARIES> ${LINK_CXX_STANDARD_LIB}\" > <TARGET>"
    )
  else()
    set(CMAKE_ECHO_LINK_EXECUTABLE
        "sh -c \"echo <FLAGS> <LINK_FLAGS> <LINK_LIBRARIES> ${LINK_CXX_STANDARD_LIB} > <TARGET>\""
    )
  endif()

  add_executable(linker_flags "CMakeLists.txt")
  target_link_libraries(linker_flags
                        OpenSSL::SSL
                        OpenSSL::Crypto
                        ${S2_EXTRA_OPENSSL_LIBS}
                        ${ABSL_LIBRARIES})

  set_target_properties(linker_flags PROPERTIES LINKER_LANGUAGE ECHO SUFFIX ".txt")

  install(TARGETS linker_flags DESTINATION "${CMAKE_INSTALL_LIBDIR}")

else()

  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/linker_flags.cc"
       "
int main(int argc, const char** args) {
  return 0;
}")
  add_executable(linker_flags "${CMAKE_CURRENT_BINARY_DIR}/linker_flags.cc")
  target_link_libraries(linker_flags
                        OpenSSL::SSL
                        OpenSSL::Crypto
                        ${S2_EXTRA_OPENSSL_LIBS}
                        ${ABSL_LIBRARIES})

  add_custom_command(TARGET linker_flags
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy
                             "${CMAKE_CURRENT_BINARY_DIR}/linker_flags.dir/$<CONFIG>/linker_flags.tlog/link.command.1.tlog"
                             "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/linker_flags.txt"
                     COMMENT "Copying linker command file for configuration $<CONFIG>")

  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/linker_flags.txt"
          DESTINATION "${CMAKE_INSTALL_LIBDIR}")
endif()
