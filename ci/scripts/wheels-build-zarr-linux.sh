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

# Build Linux wheels for sedonadb-zarr: pure Rust + pyo3, no vcpkg. cmake
# builds c-blosc + aws-lc-sys; clang/perl back ring/aws-lc-sys. One arch per
# invocation; skip musllinux via CIBW_BUILD/CIBW_SKIP.
#
# Local usage:
# CIBW_BUILD=cp313-manylinux_x86_64 ./wheels-build-zarr-linux.sh x86_64

set -e
set -o pipefail

if [ ${VERBOSE:-0} -gt 0 ]; then
  set -x
fi

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
SEDONADB_DIR="$(cd "${SOURCE_DIR}/../.." && pwd)"

ARCH="$1"

export CIBW_BEFORE_ALL="yum install -y clang perl cmake"

pushd "${SEDONADB_DIR}"
python -m cibuildwheel --platform linux --archs "${ARCH}" --output-dir python/sedonadb-zarr/dist python/sedonadb-zarr
