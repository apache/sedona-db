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
#include <filesystem>  // Requires C++17
#include <iostream>
#include <string>
#include "gtest/gtest.h"

namespace TestUtils {
// Global variable to store the executable's directory.
// Alternatively, use a singleton or pass it through test fixtures.
std::filesystem::path g_executable_dir;

std::string GetTestDataPath(const std::string& relative_path_to_dir) {
  const char* test_dir = std::getenv("GPUSPATIAL_TEST_DIR");
  if (test_dir == nullptr || std::string_view(test_dir) == "") {
    throw std::runtime_error("Environment variable GPUSPATIAL_TEST_DIR is not set");
  }
  std::filesystem::path dir(test_dir);
  return dir / relative_path_to_dir;
}

std::string GetTestShaderPath() {
  std::filesystem::path dir(g_executable_dir);
  return dir / "../shaders_ptx";
}

// Call this from main()
void Initialize(const char* argv0) {
  if (argv0 == nullptr) {
    // This should ideally not happen if called from main
    g_executable_dir = std::filesystem::current_path();  // Fallback, less reliable
    std::cerr
        << "Warning: argv[0] was null. Using current_path() as executable directory."
        << std::endl;
    return;
  }
  // Get the absolute path to the executable.
  // std::filesystem::absolute can correctly interpret argv[0] whether it's
  // a full path, relative path, or just the executable name (if in PATH).
  std::filesystem::path exe_path =
      std::filesystem::absolute(std::filesystem::path(argv0));
  g_executable_dir = exe_path.parent_path();
  std::cout << "Test executable directory initialized to: " << g_executable_dir
            << std::endl;
}

}  // namespace TestUtils

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  TestUtils::Initialize(argv[0]);  // Initialize our utility
  return RUN_ALL_TESTS();
}
