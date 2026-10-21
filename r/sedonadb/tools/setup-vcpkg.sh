#!/bin/sh

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

# Check out and bootstrap the vcpkg revision used to build the R package. Keep
# stdout reserved for the checkout path so configure can capture it.

set -eu

VCPKG_REF="${SEDONADB_VCPKG_REF:-580d480f750618f8affeb77abbb956e6eeaee0ce}"
VCPKG_DIR="${1:-$(pwd)/tools/vcpkg}"

case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) VCPKG_PROGRAM="${VCPKG_DIR}/vcpkg.exe" ;;
  *) VCPKG_PROGRAM="${VCPKG_DIR}/vcpkg" ;;
esac

if [ -x "${VCPKG_PROGRAM}" ] && \
    [ -f "${VCPKG_DIR}/scripts/buildsystems/vcpkg.cmake" ]; then
  if [ -d "${VCPKG_DIR}/.git" ]; then
    VCPKG_HEAD="$(git -C "${VCPKG_DIR}" rev-parse HEAD 2>/dev/null || true)"
    if [ "${VCPKG_HEAD}" = "${VCPKG_REF}" ]; then
      printf '%s\n' "${VCPKG_DIR}"
      exit 0
    fi
  else
    printf '%s\n' "${VCPKG_DIR}"
    exit 0
  fi
fi

if ! command -v git >/dev/null 2>&1; then
  echo "Automatic vcpkg setup requires git." >&2
  exit 1
fi

if [ ! -d "${VCPKG_DIR}/.git" ]; then
  if [ -e "${VCPKG_DIR}" ] && [ -n "$(ls -A "${VCPKG_DIR}" 2>/dev/null)" ]; then
    echo "Cannot set up vcpkg in non-empty directory: ${VCPKG_DIR}" >&2
    exit 1
  fi

  mkdir -p "${VCPKG_DIR}"
  git -C "${VCPKG_DIR}" init >&2
  git -C "${VCPKG_DIR}" remote add origin https://github.com/microsoft/vcpkg.git
fi

echo "** Checking out vcpkg ${VCPKG_REF}" >&2
if ! git -C "${VCPKG_DIR}" config remote.origin.url >/dev/null 2>&1; then
  git -C "${VCPKG_DIR}" remote add origin https://github.com/microsoft/vcpkg.git
fi
git -C "${VCPKG_DIR}" fetch --depth 1 origin "${VCPKG_REF}" >&2
git -C "${VCPKG_DIR}" -c advice.detachedHead=false checkout --detach FETCH_HEAD >&2

if [ ! -f "${VCPKG_DIR}/scripts/buildsystems/vcpkg.cmake" ]; then
  echo "The vcpkg checkout is incomplete: ${VCPKG_DIR}" >&2
  exit 1
fi

echo "** Bootstrapping vcpkg" >&2
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*)
    if ! command -v cmd.exe >/dev/null 2>&1; then
      echo "Automatic vcpkg setup on Windows requires cmd.exe." >&2
      exit 1
    fi
    VCPKG_BOOTSTRAP="$(cygpath -w "${VCPKG_DIR}/bootstrap-vcpkg.bat")"
    cmd.exe //c "${VCPKG_BOOTSTRAP}" -disableMetrics >&2
    ;;
  *)
    sh "${VCPKG_DIR}/bootstrap-vcpkg.sh" -disableMetrics >&2
    ;;
esac

if [ ! -x "${VCPKG_PROGRAM}" ]; then
  echo "vcpkg bootstrap did not create ${VCPKG_PROGRAM}." >&2
  exit 1
fi

printf '%s\n' "${VCPKG_DIR}"
