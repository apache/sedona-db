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

# Contributors Guide

This guide details how to set up your development environment as a SedonaDB Contributor.

## Fork and clone the repository

Your first step is to create a personal copy of the repository and connect it to the main project.

1. Fork the repository

      * Navigate to the official [SedonaDB GitHub repository](https://github.com/apache/sedona-db).
      * Click the **Fork** button in the top-right corner. This creates a complete copy of the project in your own GitHub account.

1. Clone your fork

      * Next, clone your newly created fork to your local machine. This command downloads the repository into a new folder named `sedona-db`.
      * Replace `YourUsername` with your actual GitHub username.

        ```shell
        git clone https://github.com/YourUsername/sedona-db.git
        cd sedona-db
        ```

1. Configure the remotes

      * Your local repository needs to know where the original project is so you can pull in updates. You'll add a remote link, traditionally named **`upstream`**, to the main SedonaDB repository.
      * Your fork is automatically configured as the **`origin`** remote.

        ```shell
        # Add the main repository as the "upstream" remote
        git remote add upstream https://github.com/apache/sedona-db.git
        ```

1. Verify the configuration

      * Run the following command to verify that you have two remotes configured correctly: `origin` (your fork) and `upstream` (the main repository).

        ```shell
        git remote -v
        ```

      * The output should look like this:

        ```shell
        origin    https://github.com/YourUsername/sedona-db.git (fetch)
        origin    https://github.com/YourUsername/sedona-db.git (push)
        upstream  https://github.com/apache/sedona-db.git (fetch)
        upstream  https://github.com/apache/sedona-db.git (push)
        ```

## Rust

SedonaDB is written in Rust and is a standard `cargo` workspace.

You can install a recent version of the Rust compiler and cargo from
[rustup.rs](https://rustup.rs/) and run tests using `cargo test`.

A local development version of the CLI can be run with `cargo run --bin sedona-cli`.

### Test data setup

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

### System dependencies

Some crates wrap external native libraries and require system dependencies
to build.

!!!note "`sedona-s2geography`"
    At this time, the only crate that requires this is the `sedona-s2geography`
    crate, which requires [CMake](https://cmake.org),
    [Abseil](https://github.com/abseil/abseil-cpp) and OpenSSL.

#### macOS

These can be installed on macOS with [Homebrew](https://brew.sh):

```shell
brew install abseil openssl cmake geos
```

#### Linux and Windows

On Linux and Windows, it is recommended to use [vcpkg](https://github.com/microsoft/vcpkg)
to provide external dependencies. This can be done by setting the `CMAKE_TOOLCHAIN_FILE`
environment variable:

```shell
export CMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
```

#### Visual Studio Code (VSCode) Configuration

When using VSCode, it may be necessary to set this environment variable in `settings.json`
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
backend.

To install a development version of the main Python bindings for the first time, run the following commands:

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

### Rust

Debugging Rust code is most easily done by writing or finding a test that triggers
the desired behavior and running it using the *Debug* selection in
[VSCode](https://code.visualstudio.com/) with the
[rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer)
extension. Rust code can also be debugged using the CLI by finding the `main()` function in
`sedona-cli` and choosing the *Debug* run option.

### Python, C, and C++

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

### Running benchmarks

Benchmarks for a specific crate can be run with `cargo bench`:

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

### Managing results

By default, criterion saves the last run and will report the difference between the
current benchmark and the last time it was run (although there are options to
save and load various baselines).

A report of the latest results for all benchmarks can be opened with the following command:

=== "macOS"
    ```shell
    open target/criterion/report/index.html
    ```
=== "Ubuntu"
    ```shell
    xdg-open target/criterion/report/index.html
    ```

All previous saved benchmark runs can be cleared with:

```shell
rm -rf target/criterion
```

## Documentation

To contribute to the SedonaDB documentation:

1. Clone the repository and create a fork.
1. Install the Documentation dependencies:
    ```sh
    pip install -r docs/requirements.txt
    ```
1. Make your changes to the documentation files.
1. Preview your changes locally using these commands:
    * `mkdocs serve` - Start the live-reloading docs server.
    * `mkdocs build` - Build the documentation site.
    * `mkdocs -h` - Print help message and exit.
1. Push your changes and open a pull request.
