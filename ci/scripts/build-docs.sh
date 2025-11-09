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

if [ ${VERBOSE:-0} -gt 0 ]; then
  set -x
fi

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
SEDONADB_DIR="$(cd "${SOURCE_DIR}/../.." && pwd)"

# Avoid a deprecation warning when building the docs
export JUPYTER_PLATFORM_DIRS=1

# Convert all Jupyter notebooks in docs/ directory to markdown
for notebook in $(find "${SEDONADB_DIR}/docs" -name "*.ipynb"); do
  echo "Rendering ${notebook}"
  jupyter nbconvert --to markdown "${notebook}"
done

pushd "${SEDONADB_DIR}"
if mkdocs build --strict ; then
  echo "Success!"
  exit 0
else
  echo "Documentation build failed"
  exit 1
fi
