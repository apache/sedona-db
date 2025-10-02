# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# R builds a package by copying only the sources under the directory of the R
# package, which means it cannot refer to the Rust files above the directory.
# So, this R script copies the necessary Rust crates under the R package dir.
# Note that, this is not a standard mechanism of R, but is only invoked by
# pkgbuild (cf. https://github.com/r-lib/pkgbuild/pull/157)

# Tweak Cargo.toml
cargo_toml <- "src/rust/Cargo.toml"
lines <- readLines(cargo_toml)
writeLines(
  gsub("../../../../", "../", lines, fixed = TRUE),
  cargo_toml
)

file.copy(
  c(
    "../../rust",
    "../../c",
    "../../Cargo.toml",
    "../../Cargo.lock"
  ),
  "src/",
  recursive = TRUE
)

# Tweak workspace Cargo.toml
top_cargo_toml <- "src/Cargo.toml"
lines <- readLines(top_cargo_toml)
# change the path to the R package's Rust code
lines <- gsub("r/sedonadb/src/rust", "rust", lines, fixed = TRUE)
# remove unnecessary workspace members
lines <- gsub('"python/sedonadb",', "", lines, fixed = TRUE)
lines <- gsub('"sedona-cli",', "", lines, fixed = TRUE)
writeLines(lines, top_cargo_toml)
