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

# We always want static libraries for the Rust crate
set(BUILD_SHARED_LIBS
    OFF
    CACHE BOOL "" FORCE)

# Try to find s2 via find_package first (e.g., from vcpkg which has Windows patches)
find_package(s2 QUIET)

if(NOT s2_FOUND)
  # Build s2geometry using its own CMake (v0.12.0 has proper add_subdirectory support).
  set(BUILD_TESTS
      OFF
      CACHE BOOL "" FORCE)
  set(BUILD_EXAMPLES
      OFF
      CACHE BOOL "" FORCE)
  add_subdirectory(s2geometry)
  set(S2_BUILT_FROM_SUBDIRECTORY TRUE)
else()
  message(STATUS "Found s2 via find_package")
  set(S2_BUILT_FROM_SUBDIRECTORY FALSE)
endif()

if(WIN32 AND NOT MSVC)
  # mingw (e.g., R for Windows) needs extra libraries
  set(S2_EXTRA_OPENSSL_LIBS
      crypt32
      z
      ws2_32
      gdi32
      crypt32)
endif()

# Abseil libraries needed by the linker_flags discovery target below.
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
    absl::utility
    absl::vlog_is_on)

# Build s2geography (needs the s2::s2 alias and version info)
if(S2_BUILT_FROM_SUBDIRECTORY)
  add_library(s2::s2 ALIAS s2)
endif()
set(S2_VERSION_MAJOR 0)
set(S2_VERSION_MINOR 13)
set(S2_VERSION_PATCH 1)
add_subdirectory(s2geography)

# s2geography's install export set references s2, so we need to include it.
if(S2_BUILT_FROM_SUBDIRECTORY)
  install(TARGETS s2
          EXPORT "s2geography-targets"
          RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
          ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
          LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
endif()

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

  set(CMAKE_ECHO_LINK_EXECUTABLE
      "sh -c \"echo <FLAGS> <LINK_FLAGS> <LINK_LIBRARIES> ${LINK_CXX_STANDARD_LIB} > <TARGET>\""
  )

  add_executable(linker_flags "CMakeLists.txt")
  target_link_libraries(linker_flags
                        s2::s2
                        OpenSSL::SSL
                        OpenSSL::Crypto
                        ${S2_EXTRA_OPENSSL_LIBS}
                        ${ABSL_LIBRARIES})

  set_target_properties(linker_flags PROPERTIES LINKER_LANGUAGE ECHO SUFFIX ".txt")

  install(TARGETS linker_flags DESTINATION "${CMAKE_INSTALL_LIBDIR}")

else()
  # On Windows, MSBuild will write this file for us, but we have to look in a very specific place
  # to find it. This is possibly brittle but makes it possible to build this on Windows at all.
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/linker_flags.cc"
       "
int main(int argc, const char** args) {
  return 0;
}")
  add_executable(linker_flags "${CMAKE_CURRENT_BINARY_DIR}/linker_flags.cc")
  target_link_libraries(linker_flags
                        s2::s2
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
