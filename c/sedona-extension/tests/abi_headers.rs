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

// Exercise the public header with the C and C++ compilers on Unix CI.
#![cfg(unix)]

use std::io::Write;
use std::process::{Command, Stdio};

use sedona_extension::extension::{
    FFI_ArrowAsyncDeviceStreamHandler, FFI_ArrowAsyncProducer, FFI_ArrowAsyncTask,
    FFI_ArrowDeviceArray,
};

#[test]
fn arrow_headers_interoperate_in_either_include_order() {
    for (compiler, language, standard) in [("cc", "c", "c11"), ("c++", "c++", "c++11")] {
        for arrow_first in [false, true] {
            let headers = if arrow_first {
                "#include \"arrow_abi.h\"\n#include \"sedona_extension.h\"\n"
            } else {
                "#include \"sedona_extension.h\"\n#include \"arrow_abi.h\"\n"
            };
            let assertion = if language == "c" {
                "_Static_assert"
            } else {
                "static_assert"
            };
            let mut source = headers.to_string();
            for (device, value) in [
                ("CPU", 1),
                ("CUDA", 2),
                ("CUDA_HOST", 3),
                ("OPENCL", 4),
                ("VULKAN", 7),
                ("METAL", 8),
                ("VPI", 9),
                ("ROCM", 10),
                ("ROCM_HOST", 11),
                ("EXT_DEV", 12),
                ("CUDA_MANAGED", 13),
                ("ONEAPI", 14),
                ("WEBGPU", 15),
                ("HEXAGON", 16),
            ] {
                source.push_str(&format!(
                    "{assertion}(ARROW_DEVICE_{device} == {value}, \"device constant\");\n"
                ));
            }
            // Whichever header supplied the declarations must match Rust's ABI.
            for (name, size) in [
                ("ArrowDeviceArray", size_of::<FFI_ArrowDeviceArray>()),
                ("ArrowAsyncTask", size_of::<FFI_ArrowAsyncTask>()),
                ("ArrowAsyncProducer", size_of::<FFI_ArrowAsyncProducer>()),
                (
                    "ArrowAsyncDeviceStreamHandler",
                    size_of::<FFI_ArrowAsyncDeviceStreamHandler>(),
                ),
            ] {
                source.push_str(&format!(
                    "{assertion}(sizeof(struct {name}) == {size}, \"Rust ABI size\");\n"
                ));
            }
            let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
            let mut child = Command::new(compiler)
                .args([
                    "-Werror",
                    "-fsyntax-only",
                    "-x",
                    language,
                    &format!("-std={standard}"),
                    "-",
                ])
                .arg("-I")
                .arg(root.join("src"))
                .arg("-I")
                .arg(root.join("tests/fixtures"))
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .expect("C/C++ compiler must be available");
            child
                .stdin
                .take()
                .unwrap()
                .write_all(source.as_bytes())
                .unwrap();
            let output = child.wait_with_output().unwrap();
            assert!(
                output.status.success(),
                "{compiler}, arrow_first={arrow_first}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
}
