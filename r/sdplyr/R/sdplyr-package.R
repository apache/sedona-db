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

#' @keywords internal
"_PACKAGE"

## usethis namespace: start
## usethis namespace: end
NULL

sdplyr_unsupported <- function() {
  structure(list(), class = "sdplyr_unsupported")
}

assert_unsupported <- function(...) {
  args <- tibble::lst(...)
  args_is_unsupported <- vapply(args, inherits, logical(1), "sdplyr_unsupported")
  arg_names <- names(args)

  if (!all(args_is_unsupported)) {
    bad_args <- arg_names[!args_is_unsupported] # nolint: object_usage_linter
    cli::cli_abort(
      c(
        "{cli::qty(bad_args)} Argument{?s} {.arg {bad_args}} {?is/are} not supported
        by sdplyr."
      ),
      call = parent.frame()
    )
  }
}
