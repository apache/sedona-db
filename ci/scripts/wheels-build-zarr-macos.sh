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

# Build macOS wheels for sedonadb-zarr: pure Rust + pyo3, no vcpkg. cmake
# (preinstalled) builds c-blosc + aws-lc-sys; default delocate repair
# suffices. Set SEDONADB_MACOS_ARCH=x86_64 to cross-build for Intel.
#
# Local usage:
# CIBW_BUILD='cp313*' ./wheels-build-zarr-macos.sh

set -e
set -o pipefail

if [ ${VERBOSE:-0} -gt 0 ]; then
  set -x
fi

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
SEDONADB_DIR="$(cd "${SOURCE_DIR}/../.." && pwd)"

# Default to arm64 but allow override to x86_64
SEDONADB_MACOS_ARCH=${SEDONADB_MACOS_ARCH:-arm64}

# Match the deployment target the main package uses so the wheel platform tag
# lines up across both packages.
export CIBW_ENVIRONMENT_MACOS="$CIBW_ENVIRONMENT_MACOS _PYTHON_HOST_PLATFORM=macosx-12.0-${SEDONADB_MACOS_ARCH} MACOSX_DEPLOYMENT_TARGET=12.0"

pushd "${SEDONADB_DIR}"
python -m cibuildwheel --output-dir python/sedonadb-zarr/dist python/sedonadb-zarr
