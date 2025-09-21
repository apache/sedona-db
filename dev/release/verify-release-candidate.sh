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

set -e
set -o pipefail

if [ ${VERBOSE:-0} -gt 0 ]; then
  set -x
fi

# Check that required dependencies are installed
check_dependencies() {
  local missing_deps=0

  local required_deps=("curl" "git" "gpg" "cc" "cargo" "cmake")
  for dep in "${required_deps[@]}"; do
    if ! command -v $dep &> /dev/null; then
      echo "Error: $dep is not installed or not in PATH"
      missing_deps=1
    fi
  done

  # pkg-config doesn't work with the above check
  if ! pkg-config --version &> /dev/null; then
    echo "Error: pkg-config is not installed or not in PATH"
    missing_deps=1
  fi

  # Check for either shasum or sha256sum/sha512sum
  local has_sha_tools=0
  if command -v shasum &> /dev/null; then
    has_sha_tools=1
  elif command -v sha256sum &> /dev/null && command -v sha512sum &> /dev/null; then
    has_sha_tools=1
  else
    echo "Error: Neither shasum nor sha256sum/sha512sum are installed or in PATH"
    missing_deps=1
  fi

  if [ $missing_deps -ne 0 ]; then
    echo "Please install missing dependencies and try again"
    exit 1
  fi
}


# Check that required native dependencies are installed. For the purposes of this
# script we require geos, proj, absl_base, and openssl via pkg-config, even though
# technically absl_base and openssl are resolved via CMake for sedona-s2geography.
check_pkg_config_dependencies() {
  local missing_deps=0
  local required_deps=(geos proj openssl absl_base)
  for dep in "${required_deps[@]}"; do
    if ! pkg-config --modversion $dep &> /dev/null; then
      echo "Error: $dep is not installed or not in PKG_CONFIG_PATH"
      missing_deps=1
    fi
  done

  local absl_version=$(pkg-config --modversion absl_base)
  # Check if Abseil version is sufficient (need at least version 20230802)
  if [ "$absl_version" -lt "20230802" ]; then
    echo "Error: Abseil version $absl_version is too old, need at least 20230802"
    echo "On Linux, this typically requires verification within a conda environment"
    echo "or by installing Abseil via vcpkg and ensuring PKG_CONFIG_PATH and"
    echo "CMAKE_TOOLCHAIN_FILE are updated appropriately."
    missing_deps=1
  fi

  if [ $missing_deps -ne 0 ]; then
    echo "Please install or update missing dependencies and try again"
    echo "Using Homebrew: brew install geos proj openssl abseil"
    echo "Using conda: conda install geos proj openssl libabseil"
    exit 1
  fi
}

case $# in
  0) VERSION="HEAD"
     SOURCE_KIND="local"
     ;;
  1) VERSION="TARBALL"
     SOURCE_KIND="local_tarball"
     LOCAL_TARBALL="$(realpath $1)"
     ;;
  2) VERSION="$1"
     RC_NUMBER="$2"
     SOURCE_KIND="tarball"
     ;;
  *) echo "Usage:"
     echo "  Verify release candidate:"
     echo "    $0 X.Y.Z RC_NUMBER"
     echo ""
     echo "  Run the source verification tasks on this sedona-db checkout:"
     echo "    $0"
     exit 1
     ;;
esac

# Check dependencies early
check_dependencies
check_pkg_config_dependencies

# Note that these point to the current verify-release-candidate.sh directories
# which is different from the SEDONADB_SOURCE_DIR set in ensure_source_directory()
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
SEDONADB_DIR="$(cd "${SOURCE_DIR}/../.." && pwd)"

show_header() {
  if [ -z "$GITHUB_ACTIONS" ]; then
    echo ""
    printf '=%.0s' $(seq ${#1}); printf '\n'
    echo "${1}"
    printf '=%.0s' $(seq ${#1}); printf '\n'
  else
    echo "::group::${1}"; printf '\n'
  fi
}

show_info() {
  echo "└ ${1}"
}

SEDONADB_DIST_URL='https://dist.apache.org/repos/dist/dev/sedona'

download_dist_file() {
  curl \
    --silent \
    --show-error \
    --fail \
    --location \
    --remote-name $SEDONADB_DIST_URL/$1
}

download_rc_file() {
  download_dist_file apache-sedona-db-${VERSION}-rc${RC_NUMBER}/$1
}

import_gpg_keys() {
  if [ "${GPGKEYS_ALREADY_IMPORTED:-0}" -gt 0 ]; then
    return 0
  fi
  download_dist_file KEYS

  if [ "${SEDONADB_ACCEPT_IMPORT_GPG_KEYS_ERROR:-0}" -gt 0 ]; then
    gpg --import KEYS || true
  else
    gpg --import KEYS
  fi

  GPGKEYS_ALREADY_IMPORTED=1
}

if type shasum >/dev/null 2>&1; then
  sha512_verify="shasum -a 512 -c"
else
  sha512_verify="sha512sum -c"
fi

fetch_archive() {
  import_gpg_keys

  local dist_name=$1
  download_rc_file ${dist_name}.tar.gz
  download_rc_file ${dist_name}.tar.gz.asc
  download_rc_file ${dist_name}.tar.gz.sha512
  gpg --verify ${dist_name}.tar.gz.asc ${dist_name}.tar.gz
  ${sha512_verify} ${dist_name}.tar.gz.sha512
}

verify_dir_artifact_signatures() {
  import_gpg_keys

  # verify the signature and the checksums of each artifact
  find $1 -name '*.asc' | while read sigfile; do
    artifact=${sigfile/.asc/}
    gpg --verify $sigfile $artifact || exit 1

    # go into the directory because the checksum files contain only the
    # basename of the artifact
    pushd $(dirname $artifact)
    base_artifact=$(basename $artifact)
    if [ -f $base_artifact.sha512 ]; then
      ${sha512_verify} $base_artifact.sha512 || exit 1
    fi
    popd
  done
}

setup_tempdir() {
  cleanup() {
    if [ "${TEST_SUCCESS}" = "yes" ]; then
      rm -fr "${SEDONADB_TMPDIR}"
    else
      echo "Failed to verify release candidate. See ${SEDONADB_TMPDIR} for details."
    fi
  }

  show_header "Creating temporary directory"

  if [ -z "${SEDONADB_TMPDIR}" ]; then
    # clean up automatically if SEDONADB_TMPDIR is not defined
    SEDONADB_TMPDIR=$(mktemp -d -t "sedonadb-${VERSION}.XXXXXX")
    trap cleanup EXIT
  else
    # don't clean up automatically
    mkdir -p "${SEDONADB_TMPDIR}"
  fi

  echo "Working in sandbox ${SEDONADB_TMPDIR}"
}

test_rust() {
  show_header "Build and test Rust libraries"

  pushd "${SEDONADB_SOURCE_DIR}"
  cargo test --workspace --exclude sedona-s2geography
  popd
}

activate_or_create_venv() {
  if [ ! -z "${SEDONADB_PYTHON_VENV}" ]; then
    show_info "Activating virtual environment at ${SEDONADB_PYTHON_VENV}"
    source "${SEDONADB_PYTHON_VENV}/bin/activate"
  else
    # Try python3 first, then try regular python (e.g., already in a venv)
    if [ -z "${PYTHON_BIN}" ] && python3 --version >/dev/null; then
      PYTHON_BIN=python3
    elif [ -z "${PYTHON_BIN}" ]; then
      PYTHON_BIN=python
    fi

    show_info "Creating temporary virtual environment using ${PYTHON_BIN}..."
    "${PYTHON_BIN}" -m venv "${SEDONADB_TMPDIR}/venv"
    source "${SEDONADB_TMPDIR}/venv/bin/activate"
    python -m pip install --upgrade pip
  fi
}

test_python() {
  show_header "Build and test Python package"
  activate_or_create_venv

  pushd "${SEDONADB_SOURCE_DIR}/python"

  show_info "Installing Python package"
  rm -rf "${SEDONADB_TMPDIR}/python"
  pip install "sedonadb/[test]" -v

  show_info "Testing Python package"
  python -m pytest -vv

  popd
}

ensure_source_directory() {
  show_header "Ensuring source directory"

  dist_name="apache-sedona-db-${VERSION}"

  if [ "${SOURCE_KIND}" = "local" ]; then
    # Local repository
    if [ -z "$SEDONADB_SOURCE_DIR" ]; then
      export SEDONADB_SOURCE_DIR="${SEDONADB_DIR}"
    fi
    echo "Verifying local sedona-db checkout at ${SEDONADB_SOURCE_DIR}"
  elif [ "${SOURCE_KIND}" = "local_tarball" ]; then
    # Local tarball
    pushd $SEDONADB_TMPDIR
    tar xf "$LOCAL_TARBALL"
    dist_name=$(ls)
    export SEDONADB_SOURCE_DIR="${SEDONADB_TMPDIR}/${dist_name}"

    # Ensure submodules are where tests expect them to be
    pushd "$SEDONADB_SOURCE_DIR/submodules"
    git clone https://github.com/apache/sedona-testing.git
    git clone https://github.com/geoarrow/geoarrow-data.git
    popd

    popd

    echo "Verifying local tarball at $LOCAL_TARBALL"
  else
    # Release tarball
    echo "Verifying official SedonaDB release candidate ${VERSION}-rc${RC_NUMBER}"
    export SEDONADB_SOURCE_DIR="${SEDONADB_TMPDIR}/${dist_name}"
    if [ ! -d "${SEDONADB_SOURCE_DIR}" ]; then
      pushd $SEDONADB_TMPDIR
      fetch_archive ${dist_name}
      tar xf ${dist_name}.tar.gz

      # Ensure submodules are where tests expect them to be
      pushd "${SEDONADB_SOURCE_DIR}/submodules"
      git clone https://github.com/apache/sedona-testing.git
      git clone https://github.com/geoarrow/geoarrow-data.git
      popd

      popd
    fi
  fi
}

test_source_distribution() {
  pushd $SEDONADB_SOURCE_DIR

  if [ ${TEST_RUST} -gt 0 ]; then
    test_rust
  fi

  if [ ${TEST_PYTHON} -gt 0 ]; then
    test_python
  fi

  popd
}

# By default test all functionalities.
# To deactivate one test, deactivate the test and all of its dependents
# To explicitly select one test, set TEST_DEFAULT=0 TEST_X=1
: ${TEST_DEFAULT:=1}

: ${TEST_SOURCE:=${TEST_DEFAULT}}
: ${TEST_RUST:=${TEST_SOURCE}}
: ${TEST_PYTHON:=${TEST_SOURCE}}

TEST_SUCCESS=no

setup_tempdir
ensure_source_directory
test_source_distribution

TEST_SUCCESS=yes

echo "Release candidate ${VERSION}-RC${RC_NUMBER} looks good!"
exit 0
