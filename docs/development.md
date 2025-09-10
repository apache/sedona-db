<!---
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

# Development

## Rust

SedonaDB is written and Rust and is a standard `cargo` workspace. You can
install a recent version of the Rust compiler and cargo from
[rustup.rs](https://rustup.rs/) and run tests using `cargo test`. A local
development version of the CLI can be run with `cargo run --bin sedona-cli`.

Some tests require submodules that contain test data or pinned versions of
external dependencies. These submodules can be initialized with:

```shell
git submodule init
git submodule update --recursive
```

Additionally, some of the data required in the tests can be downloaded by running the following script.

```bash
python submodules/download-assets.py
```

Some crates wrap external native libraries and require system dependencies
to build. At this time the only crate that requires this is the sedona-s2geography
crate, which requires [CMake](https://cmake.org),
[Abseil](https://github.com/abseil/abseil-cpp) and OpenSSL. These can be installed
on MacOS with [Homebrew](https://brew.sh):

```shell
brew install abseil openssl cmake geos
```

On Linux and Windows, it is recommended to use [vcpkg](https://github.com/microsoft/vcpkg)
to provide external dependencies. This can be done by setting the `CMAKE_TOOLCHAIN_FILE`
environment variable:

```shell
export CMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
```

When using VSCode, it may be necessary to set this environment variable in settings.json
such that it can be found by rust-analyzer when running build/run tasks:

```json
{
    "rust-analyzer.runnables.extraEnv": {
        "CMAKE_TOOLCHAIN_FILE": "/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake"
    },
    "rust-analyzer.cargo.extraEnv": {
        "CMAKE_TOOLCHAIN_FILE": "/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake"
    }
}
```

## Python

Python bindings to SedonaDB are built with the [Maturin](https://www.maturin.rs) build
backend. Installing a development version of the main Python bindings the first time
can be done with:

```shell
cd python/sedonadb
pip install -e ".[test]"
```

If editing Rust code in either SedonaDB or the Python bindings, you can recompile the
native component with:

```shell
maturin develop
```

## Debugging

Debugging Rust code is most easily done by writing or finding a test that triggers
the desired behavior and running it using the *Debug* selection in
[VSCode](https://code.visualstudio.com/) with the
[rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer)
extension. Rust code can also debugged using the CLI by finding the `main()` function in
sedona-cli and choosing the *Debug* run option.

Installation of Python bindings with `maturin develop` ensures a debug-friendly build for
debugging Rust, Python, or C/C++ code. Python code can be debugged using breakpoints in
any IDE that supports debugging an editable Python package installation (e.g., VSCode);
Rust, C, or C++ code can be debugged using the
[CodeLLDB](https://marketplace.visualstudio.com/items?itemName=vadimcn.vscode-lldb)
*Attach to Process...* command from the command palette in VSCode.

## Low-level benchmarking

Low-level Rust benchmarks use [criterion](https://github.com/bheisler/criterion.rs).
In general, there is at least one benchmark for every implementation of a function
(some functions have more than one implementation provided by different libraries),
and a few other benchmarks for low-level iteration where work was done to optimize
specific cases.

Briefly, benchmarks for a specific crate can be run with `cargo bench`:

```shell
cd rust/sedona-geo
cargo bench
```

Benchmarks for a specific function can be run with a filter. These can be run
from the workspace or a specific crate (although the output is usually easier
to read for a specific crate).

```shell
cargo bench -- st_area
```

By default, criterion saves the last run and will report the difference between the
current benchmark and the last time it was run (although there are options to
save and load various baselines). A report containing the last run for any
benchmark that was ever run can be opened with:

```shell
# MacOS
open target/criterion/report/index.html
# Ubuntu
xdg-open target/criterion/report/index.html
```

All previous saved benchmark runs can be cleared with:

```shell
rm -rf target/criterion
```

## Documentation

* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.
