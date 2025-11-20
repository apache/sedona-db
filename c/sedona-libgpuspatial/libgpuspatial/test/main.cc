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

// Helper function to get the full path to a test data file
std::string GetTestDataPath(const std::string& relative_path_to_file) {
  if (g_executable_dir.empty()) {
    // Fallback or error if g_executable_dir was not initialized.
    // This indicates an issue with main() or test setup.
    throw std::runtime_error(
        "Executable directory not set. Ensure TestUtils::Initialize is called from main().");
  }
  std::filesystem::path full_path = g_executable_dir / relative_path_to_file;
  return full_path.string();
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
