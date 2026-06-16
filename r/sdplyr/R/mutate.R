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

#' @exportS3Method dplyr::transmute
transmute.sedonadb_dataframe <- function(.data, ...) {
  exprs <- rlang::enquos(...)
  sedonadb::sd_transmute(.data, !!!exprs)
}

#' @exportS3Method dplyr::mutate
mutate.sedonadb_dataframe <- function(
  .data,
  ...,
  .by = sdplyr_unsupported(),
  .keep = sdplyr_unsupported(),
  .before = sdplyr_unsupported(),
  .after = sdplyr_unsupported()
) {
  assert_unsupported(.by, .keep, .before, .after)

  # Capture user expressions (may have duplicate names)
  exprs <- rlang::enquos(...)

  # Generate identity expressions for all original columns
  cols <- colnames(.data)
  syms <- rlang::syms(cols)
  names(syms) <- cols

  # Combine: original columns first (for output order), then user expressions

  # sd_transmute handles sequential evaluation and deduplication
  all_exprs <- c(syms, exprs)
  sedonadb::sd_transmute(.data, !!!all_exprs)
}
