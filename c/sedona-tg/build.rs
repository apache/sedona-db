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
#[cfg(feature = "bindgen")]
use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=src/tg/tg.c");
    generate_bindings();
}

#[cfg(not(feature = "bindgen"))]
fn generate_bindings() {
    // Do nothing
}

#[cfg(feature = "bindgen")]
fn generate_bindings() {
    cc::Build::new()
        .file("src/tg/tg.c")
        // MSVC needs some extra flags to support tg's use of atomics
        .flag_if_supported("/std:c11")
        .flag_if_supported("/experimental:c11atomics")
        .compile("tg");

    let bindings = bindgen::Builder::default()
        .header("src/tg/tg.h")
        .generate_comments(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let bindings_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");
    bindings
        .write_to_file(&bindings_path)
        .expect("Couldn't write bindings!");

    // If SEDONA_TG_BINDINGS_OUTPUT_PATH is set, copy the output binding.
    if let Ok(dst) = env::var("SEDONA_TG_BINDINGS_OUTPUT_PATH") {
        std::fs::copy(bindings_path, dst).expect("Failed to copy bindings");
    }
}
