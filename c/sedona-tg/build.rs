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
use std::{
    env,
    path::{Path, PathBuf},
};

fn configure_bindings_path(prebuilt_bindings_path: &Path) -> (PathBuf, bool) {
    // Honor explicit output path to regenerate bindings where caller expects them.
    if let Ok(output_path) = env::var("SEDONA_TG_BINDINGS_OUTPUT_PATH") {
        let output_path =
            std::path::absolute(PathBuf::from(output_path)).expect("Failed to get absolute path");
        // Guard against missing parent directories when writing the new bindings file.
        if let Some(output_dir) = output_path.parent() {
            std::fs::create_dir_all(output_dir).expect("Failed to create parent dirs");
        }
        return (output_path, true);
    }

    // Reuse prebuilt bindings and skip bindgen when an exact match exists.
    if prebuilt_bindings_path.exists() {
        return (prebuilt_bindings_path.to_path_buf(), false);
    }

    let output_dir = env::var("OUT_DIR").unwrap();
    let output_path = std::path::absolute(PathBuf::from(format!("{output_dir}/bindings.rs")))
        .expect("Failed to get absolute path");

    (output_path, true)
}

fn main() {
    println!("cargo:rerun-if-changed=src/tg/tg.c");
    cc::Build::new()
        .file("src/tg/tg.c")
        // MSVC needs some extra flags to support tg's use of atomics
        .flag_if_supported("/std:c11")
        .flag_if_supported("/experimental:c11atomics")
        .compile("tg");

    println!("cargo:rerun-if-env-changed=SEDONA_TG_BINDINGS_OUTPUT_PATH");

    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap();
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let prebuilt_bindings_path =
        std::path::absolute(PathBuf::from(format!("src/bindings/{os}-{arch}.rs")))
            .expect("Failed to get absolute path");

    let (bindings_path, generate_bindings) = configure_bindings_path(&prebuilt_bindings_path);

    println!(
        "cargo::rustc-env=BINDINGS_PATH={}",
        bindings_path.to_string_lossy()
    );

    if !generate_bindings {
        return;
    }

    let bindings = bindgen::Builder::default()
        .header("src/tg/tg.h")
        .generate_comments(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file(&bindings_path)
        .expect("Couldn't write bindings!");
}
