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

# Generated from st_length.qmd 0b98911e38c996bb83e9d3a8736dab59

#' Returns the length of a geometry
#'
#' Returns the length of geom. This function only supports LineString,
#' MultiLineString, and GeometryCollections containing linear geometries.
#' Use ST_Perimeter for polygons.
#'
#' @seealso [SedonaDB SQL documentation for ST_Length()](https://sedona.apache.org/sedonadb/latest/reference/sql/st_length/)
#'
#' @param geom (geometry): Input geometry
#' @param ... For S3 generic compatibility. Must be empty.
#'
#' @returns (double)
#' @export
#'
sd_length <- function(geom, ...) {
  UseMethod("sd_length")
}

#' @export
sd_length.default <- function(geom, ...) {
  call_sd_function_default()
}

sd_length_translation <- function(.ctx, geom) {
  sedonadb::sd_expr_any_function(
    "st_length",
    list(geom),
    factory = .ctx$factory
  )
}
