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
#!/usr/bin/env bash
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
SEDONADB_DIR="$(cd "${SOURCE_DIR}/../.." && pwd)"

if [ -z "$VCPKG_ROOT" ] || [ -z "$VCPKG_DEFAULT_TRIPLET" ]; then
    echo "wheels-bootstrap-vcpkg.sh requires environment variable VCPKG_ROOT to be set"
    echo "wheels-bootstrap-vcpkg.sh requires environment variable VCPKG_DEFAULT_TRIPLET to be set"
    exit 1
else
    export VCPKG_INSTALL_NAME_DIR="${VCPKG_ROOT}/installed/${VCPKG_DEFAULT_TRIPLET}/lib"

    # Ensure that geos-config from vcpkg is picked up before system geos-config
    # (e.g., from Homebrew)
    export PATH="${VCPKG_ROOT}/installed/${VCPKG_DEFAULT_TRIPLET}/tools/geos/bin:${PATH}"

    pushd ${VCPKG_ROOT}

    # If we have an explicitly requested reference, ensure it is checked out
    if [ ! -z "${VCPKG_REF}" ]; then
        git checkout ${VCPKG_REF}
    fi

    ./bootstrap-vcpkg.sh
    ./vcpkg install --overlay-triplets="${SEDONADB_DIR}/ci/scripts/custom-triplets" geos abseil openssl
    popd

    export CMAKE_TOOLCHAIN_FILE="${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
    export PKG_CONFIG_PATH=${VCPKG_ROOT}/installed/${VCPKG_DEFAULT_TRIPLET}/lib/pkgconfig:$PKG_CONFIG_PATH
fi
