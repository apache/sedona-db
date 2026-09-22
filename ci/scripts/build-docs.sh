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

# Clean + build SQL function documentation
pushd "${SEDONADB_DIR}/docs/reference/sql"

# Remove built markdown files (they confuse quarto)
find . -name "*.md" -delete

# Render the Quarto project
if quarto render ; then
  echo "Function reference Quarto project rendered successfully"
else
  echo "Function reference Quarto project build failed"
  exit 1
fi

if grep -e "-- Example failed to render:" *.md; then
  echo "Example rendering failed"
  exit 1
fi

popd

pushd "${SEDONADB_DIR}"

# Install the R package from source
R CMD INSTALL r/sedonadb --preclean

# Build R documentation inside the MkDocs source directory so both the MkDocs
# build and the mike deployment include it. pkgdown paths are relative to the
# package root (r/sedonadb).
rm -rf docs/r
if ! Rscript -e 'pkgdown::build_site("r/sedonadb", override = list(destination = "../../docs/r"), new_process = FALSE, install = FALSE)'; then
  echo "R documentation build failed"
  exit 1
fi

if ! mkdocs build --strict ; then
  echo "Documentation build failed"
  exit 1
fi

echo "Success!"
