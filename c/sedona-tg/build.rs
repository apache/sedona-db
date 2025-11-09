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

fn main() {
    println!("cargo:rerun-if-changed=src/tg/tg.c");

    cc::Build::new()
        .file("src/tg/tg.c")
        // MSVC needs some extra flags to support tg's use of atomics
        .flag_if_supported("/std:c11")
        .flag_if_supported("/experimental:c11atomics")
        .compile("tg");
}
