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
#pragma once
#include <chrono>
namespace gpuspatial {
class Stopwatch {
 private:
  std::chrono::high_resolution_clock::time_point t1, t2;

 public:
  explicit Stopwatch(bool run = false) {
    if (run) {
      start();
    }
  }

  void start() { t2 = t1 = std::chrono::high_resolution_clock::now(); }
  void stop() { t2 = std::chrono::high_resolution_clock::now(); }

  double ms() const {
    return std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() /
           1000.0;
  }
};
}  // namespace gpuspatial
