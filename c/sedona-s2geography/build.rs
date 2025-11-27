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
    collections::HashSet,
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
    let mut lib_dirs = [
        "geography_glue",
        "s2geography",
        "s2",
        "geoarrow",
        "nanoarrow_static",
    ]
    .map(|lib| find_lib_dir(&dst, lib))
    .into_iter()
    .collect::<HashSet<_>>()
    .into_iter()
    .collect::<Vec<_>>();

    lib_dirs.sort();
    for lib_dir in lib_dirs {
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
    }

    println!("cargo:rustc-link-lib=static=geography_glue");
    println!("cargo:rustc-link-lib=static=s2geography");
    println!("cargo:rustc-link-lib=static=s2");
    println!("cargo:rustc-link-lib=static=geoarrow");
    println!("cargo:rustc-link-lib=static=nanoarrow_static");

    // Parse the output we wrote from CMake that is the linker flags
    // that CMake thinks we need for Abseil and OpenSSL.
    parse_cmake_linker_flags(&dst);
}

fn parse_cmake_linker_flags(binary_dir: &Path) {
    // e.g., libabsl_base.a
    let re_lib = Regex::new("^(lib|)([^.]+).*?(LIB|lib|a|dylib|so|dll)$").unwrap();
    // e.g., -L/path/to/lib (CMake doesn't usually output this, preferrig instead
    // to pass the full path to the library)
    let re_linker_dir = Regex::new("^-L(.*)").unwrap();
    // e.g., -lstdc++
    let re_linker_lib = Regex::new("^-l(.*)").unwrap();

    let path = find_cmake_linker_flags(binary_dir);
    let linker_flags_string = read_file_maybe_utf16(&path);

    // Print out the whole thing for debugging failures
    println!("Parsing CMake linker flags: {linker_flags_string}");

    let mut last_lib_dir = "".to_string();

    // Split flags on whitespace. This probably won't work if library paths
    // contain spaces.
    for item in linker_flags_string.split_whitespace() {
        if item.is_empty() {
            continue;
        }

        if let Some(dir_match) = re_linker_dir.captures(item) {
            let (_, [dir]) = dir_match.extract();
            println!("cargo:rustc-link-search=native={dir}");
            continue;
        } else if let Some(lib_match) = re_linker_lib.captures(item) {
            let (_, [lib]) = lib_match.extract();
            println!("cargo:rustc-link-lib={lib}");
            continue;
        }

        // Try to interpret as a path to a library. CMake loves to do this. It might be quoted (Windows)
        let mut path = if item.starts_with('"') && item.ends_with('"') {
            PathBuf::from(item[1..(item.len() - 1)].to_string())
        } else {
            PathBuf::from(item)
        };

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
                        "lib" | "LIB" => println!("cargo:rustc-link-lib={lib}"),
                        "a" => println!("cargo:rustc-link-lib=static={lib}"),
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

fn find_cmake_linker_flags(binary_dir: &Path) -> PathBuf {
    // Usually lib but could be lib64 (e.g., the Linux used for wheel builds)
    let possible_lib_dirs = ["lib", "lib64", "build/Release"];
    for possible_lib in possible_lib_dirs {
        let path = binary_dir.join(possible_lib).join("linker_flags.txt");
        if path.exists() {
            return path;
        }
    }

    panic!(
        "Can't find linker_flags.txt output at {}",
        binary_dir.to_string_lossy()
    )
}

fn find_lib_dir(binary_dir: &Path, lib_file: &str) -> PathBuf {
    // Usually lib but could be lib64 (e.g., the Linux used for wheel builds)
    let possible_lib_dirs = ["lib", "lib64", "build/Release"];
    for possible_lib in possible_lib_dirs {
        let path = binary_dir.join(possible_lib);
        let static_lib_posix = path.join(format!("lib{lib_file}.a"));
        let static_lib_windows = path.join(format!("{lib_file}.lib"));
        if static_lib_posix.exists() || static_lib_windows.exists() {
            return path;
        }
    }

    panic!(
        "Can't find library dir for static library '{lib_file}' output at {}",
        binary_dir.to_string_lossy()
    )
}

// Linker flags scraped from MSBuild are UTF-16 with a byte order mark; linker flags scraped otherwise
// are system encoding (likely UTF-8 or compatible).
fn read_file_maybe_utf16(path: &PathBuf) -> String {
    let linker_flags_bytes = std::fs::read(path).expect("Read linker_flags.txt");

    // Check if the first two bytes are UTF-16 BOM (0xFF 0xFE or 0xFE 0xFF)
    if linker_flags_bytes.len() >= 2
        && ((linker_flags_bytes[0] == 0xFF && linker_flags_bytes[1] == 0xFE)
            || (linker_flags_bytes[0] == 0xFE && linker_flags_bytes[1] == 0xFF))
    {
        // Determine endianness from BOM
        let is_le = linker_flags_bytes[0] == 0xFF;

        // Skip the BOM and convert the rest
        let u16_bytes = &linker_flags_bytes[2..];
        let u16_vec: Vec<u16> = u16_bytes
            .chunks_exact(2)
            .map(|chunk| {
                if is_le {
                    u16::from_le_bytes([chunk[0], chunk[1]])
                } else {
                    u16::from_be_bytes([chunk[0], chunk[1]])
                }
            })
            .collect();

        String::from_utf16_lossy(&u16_vec).to_string()
    } else {
        String::from_utf8_lossy(&linker_flags_bytes).to_string()
    }
}
