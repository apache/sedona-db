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

call_sd_function_default <- function() {
  # Get the parent call (the .default method's caller, e.g., sd_length())
  parent_call <- sys.call(-2)
  fn_name <- as.character(parent_call[[1]])

  stop(
    "Can't use `", fn_name, "()` outside a SedonaDB context",
    call. = FALSE
  )
}

.onLoad <- function(...) {
  register <- function(...) {
    if (!isNamespaceLoaded("sedonadb")) {
      return()
    }

    ns <- asNamespace("sedonafns")
    all_names <- ls(envir = ns, all.names = TRUE)
    translation_names <- grep("_translation$", all_names, value = TRUE)

    for (trans_name in translation_names) {
      # sd_length_translation to sd_length
      fn_name <- sub("_translation$", "", trans_name)
      qualified_name <- paste0("sedonafns::", fn_name)

      # Also register st_ variant if this is sd_ function
      st_name <- sub("^sd_", "st_", fn_name)
      st_qualified <- paste0("sedonafns::", st_name)

      trans_fn <- get(trans_name, envir = ns)
      register_fn <- asNamespace("sedonadb")[["sd_register_translation"]]
      register_fn(qualified_name, trans_fn)

      if (st_name != fn_name) {
        register_fn(st_qualified, trans_fn)
      }
    }
  }

  # Register when sedonadb loads (if not already loaded)
  setHook(packageEvent("sedonadb", "onLoad"), register)

  # If sedonadb is already loaded, register now
  if (isNamespaceLoaded("sedonadb")) {
    register()
  }
}
