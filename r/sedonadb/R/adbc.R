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

#' SedonaDB ADBC Driver
#'
#' @returns An [adbcdrivermanager::adbc_driver()] of class
#'   'sedonadb_driver_sedonadb'
#' @export
#'
#' @examples
#' library(adbcdrivermanager)
#'
#' con <- sedonadb_adbc() |>
#'   adbc_database_init() |>
#'   adbc_connection_init()
#' con |>
#'   read_adbc("SELECT ST_Point(0, 1) as geometry") |>
#'   as.data.frame()
#'
sedonadb_adbc <- function() {
    init_func <- structure(sedonadb_adbc_init_func(), class = "adbc_driver_init_func")
    adbcdrivermanager::adbc_driver(init_func, subclass = "sedonadb_driver_sedonadb")
}
