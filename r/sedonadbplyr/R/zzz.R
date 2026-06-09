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
  # Ensure we load the sedona R namespaces
  requireNamespace("sedonadb", quietly = TRUE)
  requireNamespace("sedonafns", quietly = TRUE)

  # Lazy register dplyr methods
  vctrs::s3_register("dplyr::collect", "sedonadb_dataframe")
  vctrs::s3_register("dplyr::compute", "sedonadb_dataframe")
  vctrs::s3_register("dplyr::select", "sedonadb_dataframe")
  vctrs::s3_register("dplyr::arrange", "sedonadb_dataframe")
  vctrs::s3_register("dplyr::left_join", "sedonadb_dataframe")
  vctrs::s3_register("dplyr::right_join", "sedonadb_dataframe")
  vctrs::s3_register("dplyr::inner_join", "sedonadb_dataframe")
  vctrs::s3_register("dplyr::full_join", "sedonadb_dataframe")
  vctrs::s3_register("dplyr::semi_join", "sedonadb_dataframe")
  vctrs::s3_register("dplyr::anti_join", "sedonadb_dataframe")
  vctrs::s3_register("dplyr::cross_join", "sedonadb_dataframe")
}

.onAttach <- function(libname, pkgname) {
  pkgs <- c("sedonadb", "sedonafns", "dplyr")

  # Attach packages silently
  suppressPackageStartupMessages({
    for (pkg in pkgs) {
      library(pkg, character.only = TRUE)
    }
  })

  # Get versions
  versions <- vapply(
    pkgs,
    function(pkg) {
      as.character(utils::packageVersion(pkg))
    },
    character(1)
  )

  # Format package info
  pkg_info <- paste0(
    cli::col_green(cli::symbol$tick),
    " ",
    cli::col_blue(format(pkgs, width = max(nchar(pkgs)))),
    " ",
    cli::col_grey(versions)
  )

  # Build message
  header <- cli::rule(
    left = cli::style_bold("Attaching sedonadbplyr packages"),
    right = utils::packageVersion(pkgname)
  )

  msg <- paste(c(header, pkg_info), collapse = "\n")
  packageStartupMessage(msg)
}
