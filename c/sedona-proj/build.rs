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

fn main() {
    println!("cargo:rerun-if-changed=src/proj_dyn.c");
    cc::Build::new().file("src/proj_dyn.c").compile("proj_dyn");

    println!("cargo:rerun-if-env-changed=SEDONA_PROJ_BINDINGS_OUTPUT_PATH");

    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap();
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let prebuilt_bindings_path =
        std::path::absolute(PathBuf::from(format!("src/bindings/{os}-{arch}.rs")))
            .expect("Failed to get absolute path");

    let bindings_path = match (
        env::var("SEDONA_PROJ_BINDINGS_OUTPUT_PATH"),
        PathBuf::from(&prebuilt_bindings_path).exists(),
    ) {
        // case 1) If SEDONA_PROJ_BINDINGS_OUTPUT_PATH is set, generate new bindings to the path and use it.
        (Ok(output_path), _) => {
            let output_path = std::path::absolute(PathBuf::from(output_path))
                .expect("Failed to get absolute path");
            if let Some(output_dir) = output_path.parent() {
                std::fs::create_dir_all(output_dir).expect("Failed to create parent dirs");
            }
            println!(
                "cargo::rustc-env=BINDINGS_PATH={}",
                output_path.to_string_lossy()
            );
            output_path
        }
        // case 2) If SEDONA_PROJ_BINDINGS_OUTPUT_PATH is not set and the prebuilt bindings exists, use it without running bindgen
        (Err(_), true) => {
            println!(
                "cargo::rustc-env=BINDINGS_PATH={}",
                prebuilt_bindings_path.to_string_lossy()
            );
            return;
        }
        // case 3) If SEDONA_PROJ_BINDINGS_OUTPUT_PATH is not set and the prebuilt bindings doesn't exists, generate new bindings to the default path.
        (Err(_), false) => {
            let output_dir = env::var("OUT_DIR").unwrap();
            let output_path =
                std::path::absolute(PathBuf::from(format!("{output_dir}/bindings.rs")))
                    .expect("Failed to get absolute path");
            println!(
                "cargo::rustc-env=BINDINGS_PATH={}",
                output_path.to_string_lossy()
            );
            PathBuf::from(output_path)
        }
    };

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
