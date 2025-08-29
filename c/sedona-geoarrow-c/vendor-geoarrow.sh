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

main() {
  local -r repo_url="https://github.com/geoarrow/geoarrow-c"
  # Check releases page: https://github.com/geoarrow/geoarrow-c/releases/
  local -r commit_sha=9a4ceeebb6ce4272450df5ff4a56c22cb3111cef

  echo "Fetching $commit_sha from $repo_url"
  SCRATCH=$(mktemp -d)
  trap 'rm -rf "$SCRATCH"' EXIT

  local -r tarball="$SCRATCH/geoarrow.tar.gz"
  wget -O "$tarball" "$repo_url/archive/$commit_sha.tar.gz"
  tar --strip-components 1 -C "$SCRATCH" -xf "$tarball"

  # Remove previous bundle
  rm -rf src/geoarrow

  # Build the bundle
  local -r destination="$(pwd)/src/geoarrow"
  pushd "${SCRATCH}"
  mkdir build && cd build
  cmake .. \
    -DGEOARROW_BUNDLE=ON -DGEOARROW_USE_RYU=ON -DGEOARROW_USE_FAST_FLOAT=ON
  cmake --install . --prefix "${destination}"
  popd

  curl -L https://raw.githubusercontent.com/geoarrow/geoarrow-c/${commit_sha}/src/geoarrow/fast_float.h \
    -o src/geoarrow/fast_float.h

  curl -L https://raw.githubusercontent.com/geoarrow/geoarrow-c/${commit_sha}/src/geoarrow/double_parse_fast_float.cc \
    -o src/geoarrow/double_parse_fast_float.cc

  mkdir src/geoarrow/ryu
  for f in common.h d2fixed_full_table.h d2s_full_table.h d2s_intrinsics.h digit_table.h ryu.h d2s.c; do
    curl -L https://raw.githubusercontent.com/geoarrow/geoarrow-c/${commit_sha}/src/geoarrow/ryu/${f} \
      -o src/geoarrow/ryu/${f}
  done

  sed -i.bak 's/geoarrow_type.h/geoarrow.h/' src/geoarrow/double_parse_fast_float.cc
  sed -i.bak 's/geoarrow_type.h/geoarrow.h/' src/geoarrow/ryu/common.h
  rm src/geoarrow/*.bak
  rm src/geoarrow/ryu/*.bak
}

main
