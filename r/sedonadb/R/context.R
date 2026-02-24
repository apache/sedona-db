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

#' Create a SedonaDB context
#'
#' Runtime options configure the execution environment. Use
#' `global = TRUE` to configure the global context or use the
#' returned object as a scoped context. A scoped context is
#' recommended for programmatic usage as it prevents named
#' views from interfering with each other.
#'
#' @param global Use TRUE to set options on the global context.
#' @param memory_limit Maximum memory for query execution, as a
#'   human-readable string (e.g., `"4gb"`, `"512m"`) or `NULL` for
#'   unbounded (the default).
#' @param temp_dir Directory for temporary/spill files, or `NULL` to
#'   use the DataFusion default.
#' @param memory_pool_type Memory pool type: `"greedy"` (default) or
#'   `"fair"`. Only takes effect when `memory_limit` is set.
#' @param unspillable_reserve_ratio Fraction of memory (0--1) reserved for
#'   unspillable consumers. Only applies when `memory_pool_type` is
#'   `"fair"`. Defaults to 0.2 when not explicitly set.
#' @param ... Reserved for future options
#'
#' @returns The constructed context, invisibly.
#' @export
#'
#' @examples
#' sd_connect(memory_limit = "100mb", memory_pool_type = "fair")
#'
#'
sd_connect <- function(
  ...,
  global = FALSE,
  memory_limit = NULL,
  temp_dir = NULL,
  memory_pool_type = NULL,
  unspillable_reserve_ratio = NULL
) {
  check_dots_empty(..., label = "sd_connect()")
  options <- list()

  if (!is.null(memory_limit)) {
    stopifnot(is.character(memory_limit), length(memory_limit) == 1L)
    options[["memory_limit"]] <- memory_limit
  }

  if (!is.null(temp_dir)) {
    stopifnot(is.character(temp_dir), length(temp_dir) == 1L)
    options[["temp_dir"]] <- temp_dir
  }

  if (!is.null(memory_pool_type)) {
    memory_pool_type <- match.arg(memory_pool_type, c("greedy", "fair"))
    options[["memory_pool_type"]] <- memory_pool_type
  }

  if (!is.null(unspillable_reserve_ratio)) {
    stopifnot(
      is.numeric(unspillable_reserve_ratio),
      length(unspillable_reserve_ratio) == 1L,
      unspillable_reserve_ratio >= 0,
      unspillable_reserve_ratio <= 1
    )
    options[["unspillable_reserve_ratio"]] <- as.character(
      unspillable_reserve_ratio
    )
  }

  # Don't replace the global context
  if (global && length(options) > 0 && !is.null(global_ctx$ctx)) {
    warning(
      "Cannot change runtime options after the context has been initialized. ",
      "Set global options with sd_connect() before executing your first query ",
      "or use a scoped context by passing global = FALSE"
    )

    return(invisible(global_ctx$ctx))
  } else if (global && !is.null(global_ctx$ctx)) {
    return(invisible(global_ctx$ctx))
  }

  keys <- as.character(names(options))
  values <- as.character(options)
  ctx <- InternalContext$new(keys, values)

  if (global) {
    global_ctx$ctx <- ctx
  }

  invisible(ctx)
}

#' Create a DataFrame from one or more Parquet files
#'
#' The query will only be executed when requested.
#'
#' @param ctx A SedonaDB context.
#' @param path One or more paths or URIs to Parquet files
#'
#' @returns A sedonadb_dataframe
#' @export
#'
#' @examples
#' path <- system.file("files/natural-earth_cities_geo.parquet", package = "sedonadb")
#' sd_read_parquet(path) |> head(5) |> sd_preview()
#'
sd_read_parquet <- function(path) {
  sd_ctx_read_parquet(ctx(), path)
}

#' @rdname sd_read_parquet
#' @export
sd_ctx_read_parquet <- function(ctx, path) {
  check_ctx(ctx)
  df <- ctx$read_parquet(path)
  new_sedonadb_dataframe(ctx, df)
}

#' Create a DataFrame from SQL
#'
#' The query will only be executed when requested.
#'
#' @param ctx A SedonaDB context.
#' @param sql A SQL string to execute
#' @param params A list of parameters to fill placeholders in the query.
#'
#' @returns A sedonadb_dataframe
#' @export
#'
#' @examples
#' sd_sql("SELECT ST_Point(0, 1) as geom")
#' sd_sql("SELECT ST_Point($1, $2) as geom", params = list(1, 2))
#' sd_sql("SELECT ST_Point($x, $y) as geom", params = list(x = 1, y = 2))
#'
sd_sql <- function(sql, ..., params = NULL) {
  sd_ctx_sql(ctx(), sql, ..., params = params)
}

#' @rdname sd_sql
#' @export
sd_ctx_sql <- function(ctx, sql, ..., params = NULL) {
  check_ctx(ctx)
  df <- ctx$sql(sql)
  check_dots_empty(..., label = "sd_sql()")

  if (!is.null(params)) {
    params <- lapply(params, as_sedonadb_literal)
    df <- df$with_params(params)
  }

  new_sedonadb_dataframe(ctx, df)
}

#' Create or Drop a named view
#'
#' Remove a view created with [sd_to_view()] from the context.
#'
#' @param ctx A SedonaDB context.
#' @param table_ref The name of the view reference
#' @returns The context, invisibly
#' @export
#'
#' @examples
#' sd_sql("SELECT 1 as one") |> sd_to_view("foofy")
#' sd_view("foofy")
#' sd_drop_view("foofy")
#' try(sd_view("foofy"))
#'
sd_drop_view <- function(table_ref) {
  sd_ctx_drop_view(ctx(), table_ref)
}

#' @rdname sd_drop_view
#' @export
sd_ctx_drop_view <- function(ctx, table_ref) {
  check_ctx(ctx)
  ctx$deregister_table(table_ref)
  invisible(ctx)
}

#' @rdname sd_drop_view
#' @export
sd_view <- function(table_ref) {
  sd_ctx_view(ctx(), table_ref)
}

#' @rdname sd_drop_view
#' @export
sd_ctx_view <- function(ctx, table_ref) {
  check_ctx(ctx)
  df <- ctx$view(table_ref)
  new_sedonadb_dataframe(ctx, df)
}

# nolint start: line_length_linter
#' Register a user-defined function
#'
#' Several types of user-defined functions can be registered into a session
#' context. Currently, the only implemented variety is an external pointer
#' to a Rust `FFI_ScalarUDF`, an example of which is available from the
#' [DataFusion Python documentation](https://github.com/apache/datafusion-python/blob/6f3b1cab75cfaa0cdf914f9b6fa023cb9afccd7d/examples/datafusion-ffi-example/src/scalar_udf.rs).
#'
#' @param ctx A SedonaDB context.
#' @param udf An object of class 'datafusion_scalar_udf'
#'
#' @returns NULL, invisibly
#' @export
#'
sd_register_udf <- function(udf) {
  sd_ctx_register_udf(ctx(), udf)
}

#' @rdname sd_register_udf
#' @export
sd_ctx_register_udf <- function(ctx, udf) {
  check_ctx(ctx)
  ctx$register_scalar_udf(udf)
}
# nolint end

# We mostly use one context but support isolated contexts via sd_ctx_*
# functions, which more cleanly scope the registration of views and
# UDFs.
ctx <- function() {
  if (is.null(global_ctx$ctx)) {
    global_ctx$ctx <- sd_connect()
  }

  global_ctx$ctx
}

check_ctx <- function(ctx) {
  if (!inherits(ctx, "sedonadb::InternalContext")) {
    stop(
      sprintf(
        "Expected ctx to be a SedonaDB context but got object of class '%s'",
        class(ctx)[1]
      )
    )
  }

  invisible(ctx)
}

global_ctx <- new.env(parent = emptyenv())
global_ctx$ctx <- NULL


#' Configure PROJ
#'
#' Performs a runtime configuration of PROJ, which can be used in place of
#' a build-time linked version of PROJ or to add in support if PROJ was
#' not linked at build time.
#'
#' @param preset One of:
#'   - `"homebrew"`: Look for PROJ installed by Homebrew. This is the easiest
#'     option on MacOS.
#'   - `"system"`: Look for PROJ in the platform library load path (e.g.,
#'     after installing system proj on Linux).
#'   - `"auto"`: Try all presets in the order listed above, issuing a warning
#'     if none can be configured.
#' @param shared_library An absolute or relative path to a shared library
#'   valid for the platform.
#' @param database_path A path to proj.db
#' @param search_path A path to the data files required by PROJ for some
#'   transforms.
#'
#' @returns NULL, invisibly
#' @export
#'
#' @examples
#' sd_configure_proj("auto")
#'
sd_configure_proj <- function(
  preset = NULL,
  shared_library = NULL,
  database_path = NULL,
  search_path = NULL
) {
  if (!is.null(preset)) {
    switch(
      preset,
      homebrew = {
        configure_proj_prefix(Sys.getenv("HOMEBREW_PREFIX", "/opt/homebrew"))
        return(invisible(NULL))
      },
      system = {
        configure_proj_system()
        return(invisible(NULL))
      },
      auto = {
        presets <- c("homebrew", "system")
        errors <- c()
        for (preset in presets) {
          maybe_err <- try(sd_configure_proj(preset), silent = TRUE)
          if (!inherits(maybe_err, "try-error")) {
            return(invisible(NULL))
          } else {
            errors <- c(errors, sprintf("%s: %s", preset, maybe_err))
          }
        }

        packageStartupMessage(
          sprintf(
            "Failed to configure PROJ (tried %s):\n%s",
            paste0("'", presets, "'", collapse = ", "),
            paste0(errors, collapse = "\n")
          )
        )

        return(invisible(NULL))
      },
      stop(sprintf("Unknown preset: '%s'", preset))
    )
  }

  # We could check a shared library with dyn.load(), but this may error for
  # valid system PROJ that isn't an absolute filename.

  if (!is.null(database_path)) {
    if (!file.exists(database_path)) {
      stop(sprintf("Invalid database path: '%s' does not exist", database_path))
    }
  }

  if (!is.null(search_path)) {
    if (!dir.exists(search_path)) {
      stop(sprintf("Invalid search path: '%s' does not exist", search_path))
    }
  }

  configure_proj_shared(
    shared_library_path = shared_library,
    database_path = database_path,
    search_path = search_path
  )
}

configure_proj_system <- function() {
  sd_configure_proj(shared_library = proj_dll_name())
}

configure_proj_prefix <- function(prefix) {
  if (!dir.exists(prefix)) {
    stop(sprintf("Can't configure PROJ from prefix '%s': does not exist", prefix))
  }

  sd_configure_proj(
    shared_library = file.path(prefix, "lib", proj_dll_name()),
    database_path = file.path(prefix, "share", "proj", "proj.db"),
    search_path = file.path(prefix, "share", "proj")
  )
}

proj_dll_name <- function() {
  switch(
    tolower(Sys.info()[["sysname"]]),
    windows = "proj.dll",
    darwin = "libproj.dylib",
    linux = "libproj.so",
    stop(sprintf(
      "Can't determine system PROJ shared library name for OS: %s",
      Sys.info()[["sysname"]]
    ))
  )
}
