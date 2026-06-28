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

if (nzchar(Sys.getenv("R_SEDONADB_FNS_SKIP_BOOTSTRAP"))) {
  message("Skipping sedonafns bootstrap because R_SEDONADB_FNS_SKIP_BOOTSTRAP is set")
} else {
  # On r-universe, bootstrap.R runs before dependencies are installed.
  # Install the packages we need for source generation.
  required_packages <- c("yaml", "here", "glue", "rlang", "roxygen2")
  missing_packages <- required_packages[
    !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
  ]
  if (length(missing_packages) > 0 && nzchar(Sys.getenv("MY_UNIVERSE"))) {
    message(
      "Running on r-universe, installing bootstrap dependencies: ",
      paste(missing_packages, collapse = ", ")
    )
    install.packages(missing_packages)
  }

  source("tools/update-sd-funcs.R")
  update_sd_funcs()

  local({
    # Let the configure script know that we don't want to bootstrap or we will
    # go in circles
    on.exit(Sys.setenv(R_SEDONADB_FNS_SKIP_BOOTSTRAP = ""))
    Sys.setenv(R_SEDONADB_FNS_SKIP_BOOTSTRAP = "1")
    roxygen2::roxygenise(".")
  })
}
