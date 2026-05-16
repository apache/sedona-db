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

call_sd_function <- function() {
  # Peek into the call stack to get the parent call for the error message
  # that says this isn't called in a SedonaDB expression context
  stop("Can't use <function> outside a SedonaDB context")
}

.onLoad <- function(...) {
  # List all functions in the namespace that end with _translation and
  # register them under sedonafns::st_name and sedonafns::sd_name
  # this should be done in an on-package-load hook like in sedonadb:::s3_register
  # so that sedonadb is not strictly necessary
}
