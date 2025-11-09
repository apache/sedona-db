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

# Local usage (e.g., just check that this script works)
# Requires VCPKG_ROOT to be set.
# CIBW_BUILD='cp313*' ./wheels-build-macos.sh sedonadb

set -e
set -o pipefail

if [ ${VERBOSE:-0} -gt 0 ]; then
  set -x
fi

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
SEDONADB_DIR="$(cd "${SOURCE_DIR}/../.." && pwd)"

# Ensure we have our VCPKG_ROOT set and bootstrapped
export VCPKG_DEFAULT_TRIPLET=arm64-osx-dynamic-release
source ./wheels-bootstrap-vcpkg.sh

# Set environment variables for vcpkg dependencies on MacOS. This is required because
# vcpkg sets its libraries to use @rpath so delocate-wheel cannot locate them without
# setting DYLD_LIBRARY_PATH. For security reasons, MacOS aggressively strips the value
# of DYLD_LIBRARY_PATH.
# https://github.com/pypa/cibuildwheel/issues/816
export CIBW_REPAIR_WHEEL_COMMAND_MACOS="DYLD_LIBRARY_PATH=$VCPKG_INSTALL_NAME_DIR delocate-listdeps {wheel} && DYLD_LIBRARY_PATH=$VCPKG_INSTALL_NAME_DIR delocate-wheel --require-archs {delocate_archs} -w {dest_dir} {wheel}"

# Pass on environment variables specifically for the build
export CIBW_ENVIRONMENT_MACOS="$CIBW_ENVIRONMENT_MACOS _PYTHON_HOST_PLATFORM=macosx-12.0-arm64 MACOSX_DEPLOYMENT_TARGET=12.0 CMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} MATURIN_PEP517_ARGS='--features s2geography,pyo3/extension-module'"

pushd "${SEDONADB_DIR}"
python -m cibuildwheel --output-dir python/$1/dist python/$1
