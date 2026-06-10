#!/usr/bin/env bash
#
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

# Build Linux wheels for sedonadb-zarr.
#
# sedonadb-zarr is pure Rust + pyo3 with no vcpkg-managed native dependencies
# and no sedonadb-only cargo features, so this is just cibuildwheel over the
# Rust build. The manylinux image needs cmake (builds c-blosc via blosc-src
# and aws-lc-sys) plus clang/perl (back ring and aws-lc-sys).
#
# This build profile is specific to sedonadb-zarr. Other extensions may have
# different native build requirements and should get their own script rather
# than reusing this one.
#
# Builds one arch at a time; musllinux must be skipped via CIBW_BUILD/CIBW_SKIP.
#
# Local usage:
# CIBW_BUILD=cp313-manylinux_x86_64 ./wheels-build-zarr-linux.sh x86_64

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
SEDONADB_DIR="$(cd "${SOURCE_DIR}/../.." && pwd)"

ARCH="$1"

export CIBW_BEFORE_ALL="yum install -y clang perl cmake"

pushd "${SEDONADB_DIR}"
python -m cibuildwheel --platform linux --archs ${ARCH} --output-dir python/sedonadb-zarr/dist python/sedonadb-zarr
