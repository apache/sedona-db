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
# CIBW_BUILD=cp313-manylinux_aarch64 ./wheels-build-linux.sh aarch64 sedonadb
#
# This script builds for linux but can only build one arch at a time at the moment.
# musllinux isn't supported yet (must be skipped by CIBW_BUILD or CIBW_SKIP).

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
SEDONADB_DIR="$(cd "${SOURCE_DIR}/../.." && pwd)"

ARCH="$1"

if [ "${ARCH}" = "aarch64" ]; then
    VCPKG_DEFAULT_TRIPLET=arm64-linux-dynamic-release
elif [ "${ARCH}" = "x86_64" ]; then
    VCPKG_DEFAULT_TRIPLET=x86-linux-dynamic-release
else
    echo "Unknown arch: $ARCH"
    exit 1
fi

# manylinux is AlmaLinux/Fedora-based, musllinux is Alpine-based
# If we want musllinux support there will be some workshopping required (vcpkg
# needs some newer components than are provided by the default musllinux image)
BEFORE_ALL_MANYLINUX="yum install -y curl zip unzip tar clang perl"

# This approach downloads and builds native dependencies with vcpkg once for every image.
# Compared to the Rust build time, the native dependency build time is not too bad. We could
# periodically build base images with our dependencies (and perhaps a rust cache), which would
# add quite a bit of complexity but could save time if we build wheels for linux frequently.
# The native and Rust builds are cached on each image such that compile work is effectively
# cached between Python versions (just not between invocations of this script).
export CIBW_ENVIRONMENT_LINUX="VCPKG_ROOT=/vcpkg VCPKG_REF=$VCPKG_REF VCPKG_DEFAULT_TRIPLET=$VCPKG_DEFAULT_TRIPLET CMAKE_TOOLCHAIN_FILE=/vcpkg/scripts/buildsystems/vcpkg.cmake PKG_CONFIG_PATH=/vcpkg/installed/$VCPKG_DEFAULT_TRIPLET/lib/pkgconfig LD_LIBRARY_PATH=/vcpkg/installed/$VCPKG_DEFAULT_TRIPLET/lib MATURIN_PEP517_ARGS='--features s2geography,pyo3/extension-module'"
export CIBW_BEFORE_ALL="$BEFORE_ALL_MANYLINUX && git clone https://github.com/microsoft/vcpkg.git /vcpkg && bash {package}/../../ci/scripts/wheels-bootstrap-vcpkg.sh"

pushd "${SEDONADB_DIR}"
python -m cibuildwheel --platform linux --archs ${ARCH} --output-dir python/$2/dist python/$2
