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

use regex::Regex;

fn main() {
    // Run cmake. The cmake crate translates the Rust environment into a plausible
    // cmake one, although it's not perfect. Early observations are that
    // MACOS_DEPLOYMENT_TARGET isn't passed through properly but this can probably
    // be solved by setting the environment variable in CI before building a portable
    // target like a Python package.
    let dst = cmake::Config::new(".").build();

    // Link the libraries that are easy to enumerate by hand and whose location
    // we control in CMakeLists.txt.
    println!(
        "cargo:rustc-link-search=native={}",
        dst.join("lib").display()
    );
    println!("cargo:rustc-link-lib=static=geography_glue");
    println!("cargo:rustc-link-lib=static=s2geography");
    println!("cargo:rustc-link-lib=static=s2");
    println!("cargo:rustc-link-lib=static=geoarrow");
    println!("cargo:rustc-link-lib=static=nanoarrow_static");

    // Parse the output we wrote from CMake that is the linker flags
    // that CMake thinks we need for Abseil and OpenSSL.
    parse_cmake_linker_flags(&dst);

    // Generate bindings from the header
    println!("cargo::rerun-if-changed=src/geography_glue.h");
    let bindings = bindgen::Builder::default()
        .header("src/geography_glue.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn parse_cmake_linker_flags(binary_dir: &Path) {
    // e.g., libabsl_base.a
    let re_lib = Regex::new("^(lib|)([^.]+).*?(lib|a|dylib|so|dll)$").unwrap();
    // e.g., -L/path/to/lib (CMake doesn't usually output this, preferrig instead
    // to pass the full path to the library)
    let re_linker_dir = Regex::new("^-L(.*)").unwrap();
    // e.g., -lstdc++
    let re_linker_lib = Regex::new("^-l(.*)").unwrap();
    let path = binary_dir.join("lib").join("linker_flags.txt");
    let values = std::fs::read_to_string(path).expect("Read linker_flags.txt");

    // Print out the whole thing for debugging failures
    println!("Parsing CMake linker flags: {values}");

    let mut last_lib_dir = "".to_string();

    // Split flags on whitespace. This probably won't work if library paths
    // contain spaces.
    for item in values.split_whitespace() {
        if let Some(dir_match) = re_linker_dir.captures(item) {
            let (_, [dir]) = dir_match.extract();
            println!("cargo:rustc-link-search=native={dir}");
            continue;
        } else if let Some(lib_match) = re_linker_lib.captures(item) {
            let (_, [lib]) = lib_match.extract();
            println!("cargo:rustc-link-lib={lib}");
            continue;
        }

        // Try to interpret as a path to a library. CMake loves to do this.
        let mut path = PathBuf::from(item);

        // If it's a relative path, it's relative to the binary directory
        if path.is_relative() {
            path = binary_dir.join("build").join(path);
        }

        match (path.parent(), path.file_name()) {
            (Some(dir), Some(basename)) => {
                if let Some(basename_match) = re_lib.captures(&basename.to_string_lossy()) {
                    let (_, [_, lib, suffix]) = basename_match.extract();
                    let dir = dir.to_string_lossy();
                    if dir != last_lib_dir {
                        println!("cargo:rustc-link-search=native={dir}");
                        last_lib_dir = dir.to_string();
                    }

                    match suffix {
                        "a" | "lib" => println!("cargo:rustc-link-lib=static={lib}"),
                        _ => println!("cargo:rustc-link-lib=dylib={lib}"),
                    }
                }
            }
            _ => {
                continue;
            }
        }
    }
}
