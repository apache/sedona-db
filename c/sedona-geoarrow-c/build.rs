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
    println!("cargo:rerun-if-changed=src/geoarrow/double_parse_fast_float.cc");
    println!("cargo:rerun-if-changed=src/geoarrow/geoarrow.c");
    println!("cargo:rerun-if-changed=src/nanoarrow/nanoarrow.c");

    cc::Build::new()
        .file("src/geoarrow/ryu/d2s.c")
        .file("src/geoarrow/geoarrow.c")
        .file("src/nanoarrow/nanoarrow.c")
        .include("src/")
        .flag("-DGEOARROW_NAMESPACE=SedonaDB")
        .flag("-DNANOARROW_NAMESPACE=SedonaDB")
        .compile("geoarrow");

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("src/geoarrow/double_parse_fast_float.cc")
        .include("src/")
        .flag("-DGEOARROW_NAMESPACE=SedonaDB")
        .flag("-DNANOARROW_NAMESPACE=SedonaDB")
        .compile("geoarrow_fast_float");
}
