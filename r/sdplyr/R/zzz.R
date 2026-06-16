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
  # Ensure we use sedonafns:: and dplyr:: at least once for Imports
  dplyr::select(data.frame())
  try(sedonafns::sd_point(), silent = TRUE)
}

.onAttach <- function(libname, pkgname) {
  pkgs <- c("sedonadb", "sedonafns", "dplyr")

  # Attach packages silently
  suppressPackageStartupMessages({
    for (pkg in pkgs) {
      sdplyr_attach(pkg)
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
    left = cli::style_bold("Attaching sdplyr packages"),
    right = utils::packageVersion(pkgname)
  )

  msg <- paste(c(header, pkg_info), collapse = "\n")
  packageStartupMessage(msg)
}

# from tidyverse::same_library
sdplyr_attach <- function(pkg) {
  loc <- if (pkg %in% loadedNamespaces()) dirname(getNamespaceInfo(pkg, "path"))
  library(pkg, lib.loc = loc, character.only = TRUE, warn.conflicts = FALSE)
}
