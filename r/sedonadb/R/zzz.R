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

.onLoad <- function(...) {
  # Load geoarrow to manage conversion of arrow results to/from spatial objects
  requireNamespace("geoarrow", quietly = TRUE)

  # Call at least one function for R CMD check
  geoarrow::as_geoarrow_array("POINT (0 1")

  # Inject what we need to reduce the Rust code to a simple Rf_eval()
  ns <- asNamespace("sedonadb")
  call <- call("check_interrupts")
  init_r_runtime_interrupts(call, ns)
}

# The function we call from Rust to check for interrupts. R checks for
# interrupts automatically when evaluating regular R code and signals
# an interrupt condition,
check_interrupts <- function() {
  tryCatch({
    FALSE
  }, interrupt = function(...) TRUE, error = function(...) TRUE)
}
