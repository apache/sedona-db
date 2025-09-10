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

#' @export
as_sedonadb_dataframe.sf <- function(x, ..., schema = NULL) {
  stream <- nanoarrow::as_nanoarrow_array_stream(
    x,
    schema = schema,
    geometry_schema = geoarrow::geoarrow_wkb()
  )
  ctx <- ctx()
  df <- ctx$data_frame_from_array_stream(stream, collect_now = TRUE)

  # Verify schema is handled
  as_sedonadb_dataframe(new_sedonadb_dataframe(ctx, df), schema = schema)
}

# dynamically registered in zzz.R
st_as_sf.sedonadb_dataframe <- function(x, ...) {
  stream <- nanoarrow::nanoarrow_allocate_array_stream()
  size <- x$df$collect(stream)
  sf::st_as_sf(stream)
}
