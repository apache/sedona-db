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
use std::{env, path::PathBuf};

// Since relative path differs between build.rs and a file under `src/`, use the
// absolute path.
fn get_absolute_path(path: String) -> PathBuf {
    std::path::absolute(PathBuf::from(path)).expect("Failed to get absolute path")
}

fn configure_bindings_path(prebuilt_bindings_path: String) -> (PathBuf, bool) {
    // If SEDONA_PROJ_BINDINGS_OUTPUT_PATH is set, honor the explicit output
    // path to regenerate bindings.
    if let Ok(output_path) = env::var("SEDONA_PROJ_BINDINGS_OUTPUT_PATH") {
        let output_path = get_absolute_path(output_path);
        if let Some(output_dir) = output_path.parent() {
            std::fs::create_dir_all(output_dir).expect("Failed to create parent dirs");
        }
        return (output_path, true);
    }

    // If a prebuilt bindings exists, use it and skip bindgen.
    let prebuilt_bindings_path = get_absolute_path(prebuilt_bindings_path);
    if prebuilt_bindings_path.exists() {
        return (prebuilt_bindings_path, false);
    }

    let output_dir = env::var("OUT_DIR").unwrap();
    let output_path = get_absolute_path(format!("{output_dir}/bindings.rs"));

    (output_path, true)
}

fn main() {
    println!("cargo:rerun-if-changed=src/proj_dyn.c");
    cc::Build::new().file("src/proj_dyn.c").compile("proj_dyn");

    println!("cargo:rerun-if-env-changed=SEDONA_PROJ_BINDINGS_OUTPUT_PATH");

    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap();
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let prebuilt_bindings_path = format!("src/bindings/{os}-{arch}.rs");

    let (bindings_path, generate_bindings) = configure_bindings_path(prebuilt_bindings_path);

    println!(
        "cargo::rustc-env=BINDINGS_PATH={}",
        bindings_path.to_string_lossy()
    );

    if !generate_bindings {
        return;
    }

    let bindings = bindgen::Builder::default()
        .header("src/proj_dyn.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    bindings
        .write_to_file(&bindings_path)
        .expect("Couldn't write bindings!");
}
