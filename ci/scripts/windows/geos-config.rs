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
    // The first line is a -Lpath/to/ that contains geos_c.dll
    if let Ok(lib_dir) = std::env::var("GEOS_LIB_DIR") {
        println!("-L{}", lib_dir);
    } else {
        eprintln!("GEOS_LIB_DIR environment variable not set");
        std::process::exit(1);
    }

    // The second line is the GEOS version
    if let Ok(version) = std::env::var("GEOS_VERSION") {
        println!("{}", version);
    } else {
        eprintln!("GEOS_VERSION environment variable not set");
        std::process::exit(1);
    }

    std::process::exit(0);
}
