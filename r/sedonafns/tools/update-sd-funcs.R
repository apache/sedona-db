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

library(yaml)
library(here)
library(glue)

# Configuration - paths relative to sedonafns package root
docs_dir <- here::here("..", "..", "docs", "reference", "sql")
output_dir <- here::here("R")
docs_base_url <- "https://sedona.apache.org/sedonadb/latest/reference/sql"

# Type to parameter name mapping
type_to_param <- list(
  geometry = "geom",
  geography = "geom",
  raster = "rast",
  float64 = "x",
  double = "x",

  integer = "n",
  int64 = "n",
  string = "s",
  boolean = "b"
)

# Apache license header
lincense_header <- "# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# \"License\"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# \"AS IS\" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License."

#' Extract YAML frontmatter from a .qmd file
#'
#' @param file_path Path to the .qmd file
#' @returns Named list with frontmatter fields
extract_frontmatter <- function(file_path) {
  lines <- readLines(file_path, warn = FALSE)

  # Find YAML delimiters

  start_idx <- which(lines == "---")[1]
  end_idx <- which(lines == "---")[2]

  if (is.na(start_idx) || is.na(end_idx)) {
    stop("Could not find YAML frontmatter in ", file_path)
  }

  yaml_text <- paste(lines[(start_idx + 1):(end_idx - 1)], collapse = "\n")

  # Remove license comment from YAML
  yaml_text <- gsub("#[^\n]*\n", "", yaml_text)

  yaml::yaml.load(yaml_text)
}

#' Extract description section from .qmd file body
#'
#' @param file_path Path to the .qmd file
#' @returns Character string with description text
extract_description_section <- function(file_path) {
  lines <- readLines(file_path, warn = FALSE)

  # Find end of frontmatter
  delimiters <- which(lines == "---")
  if (length(delimiters) < 2) {
    return(NULL)
  }

  body_start <- delimiters[2] + 1
  body_lines <- lines[body_start:length(lines)]

  # Find ## Description section
  desc_start <- which(grepl("^## Description", body_lines))[1]
  if (is.na(desc_start)) {
    return(NULL)
  }

  # Find next section or end
  remaining <- body_lines[(desc_start + 1):length(body_lines)]
  next_section <- which(grepl("^## ", remaining))[1]

  if (is.na(next_section)) {
    desc_lines <- remaining
  } else {
    desc_lines <- remaining[1:(next_section - 1)]
  }

  # Clean up and join
  desc_text <- trimws(paste(desc_lines, collapse = " "))
  desc_text <- gsub("\\s+", " ", desc_text)
  trimws(desc_text)
}

#' Generate parameter name from type
#'
#' @param arg_type The type string (e.g., "geometry", "float64")
#' @param index Numeric index for disambiguation (1=a, 2=b, etc.)
#' @param needs_suffix Whether this type appears multiple times
#' @returns Character parameter name
type_to_param_name <- function(arg_type, index = 1, needs_suffix = FALSE) {
  base_name <- type_to_param[[arg_type]]
  if (is.null(base_name)) {
    base_name <- "arg"
  }

  if (needs_suffix) {
    suffix <- letters[index] # 1=a, 2=b, 3=c, ...
    paste0(base_name, "_", suffix)
  } else {
    base_name
  }
}

#' Parse kernel arguments and generate parameter info
#'
#' @param kernels List of kernel definitions from frontmatter
#' @param fn_name Function name for error messages
#' @returns List with params (for roxygen) and args (for function signature)
parse_kernel_params <- function(kernels, fn_name = "unknown") {
  if (length(kernels) == 0) {
    return(list(params = list(), args = character(), returns = "unknown"))
  }

  # Helper to extract arg info from a single kernel

  extract_kernel_arg_info <- function(kernel_args) {
    lapply(kernel_args, function(arg) {
      if (is.character(arg)) {
        list(type = arg, name = NULL, desc = NULL)
      } else if (is.list(arg)) {
        list(
          type = arg$type %||% "unknown",
          name = arg$name,
          desc = arg$description
        )
      } else {
        list(type = "unknown", name = NULL, desc = NULL)
      }
    })
  }

  # Helper to generate arg names for a kernel's args
  generate_arg_names <- function(arg_info_list) {
    types <- vapply(arg_info_list, function(x) x$type, character(1))
    type_totals <- table(types)
    type_counts <- list()
    arg_names <- character()

    for (info in arg_info_list) {
      arg_type <- info$type
      arg_name <- info$name

      if (is.null(arg_name)) {
        type_counts[[arg_type]] <- (type_counts[[arg_type]] %||% 0) + 1
        needs_suffix <- type_totals[[arg_type]] > 1
        arg_name <- type_to_param_name(arg_type, type_counts[[arg_type]], needs_suffix)
      }

      arg_names <- c(arg_names, arg_name)
    }

    arg_names
  }

  # Process all kernels to get their argument names
  all_kernel_args <- lapply(kernels, function(k) {
    info <- extract_kernel_arg_info(k$args)
    generate_arg_names(info)
  })

  # Find the kernel with the most arguments
  kernel_lengths <- vapply(all_kernel_args, length, integer(1))
  max_args <- max(kernel_lengths)

  # Validate that argument names align at each position
  for (pos in seq_len(max_args)) {
    names_at_pos <- character()
    for (i in seq_along(all_kernel_args)) {
      if (pos <= length(all_kernel_args[[i]])) {
        names_at_pos <- c(names_at_pos, all_kernel_args[[i]][pos])
      }
    }
    unique_names <- unique(names_at_pos)
    if (length(unique_names) > 1) {
      stop(
        "Cannot generate R function for ",
        fn_name,
        ": ",
        "argument names at position ",
        pos,
        " do not match across kernels: ",
        paste(unique_names, collapse = ", "),
        call. = FALSE
      )
    }
  }

  # Use the kernel with the most arguments as the reference
  ref_idx <- which.max(kernel_lengths)
  kernel <- kernels[[ref_idx]]
  args <- kernel$args
  returns <- kernels[[1]]$returns %||% "unknown"

  # Extract full info from reference kernel
  arg_info <- extract_kernel_arg_info(args)
  arg_names <- all_kernel_args[[ref_idx]]

  # Build params with descriptions
  params <- list()
  for (i in seq_along(arg_info)) {
    info <- arg_info[[i]]
    arg_name <- arg_names[i]
    arg_desc <- info$desc

    if (is.null(arg_desc)) {
      arg_desc <- paste0("(", info$type, "): Input ", info$type)
    } else {
      arg_desc <- paste0("(", info$type, "): ", trimws(arg_desc))
    }

    params[[arg_name]] <- arg_desc
  }

  list(params = params, args = arg_names, returns = returns)
}

#' Wrap text to specified width with roxygen prefix
#'
#' @param text Text to wrap
#' @param width Maximum line width
#' @param prefix Prefix for each line
#' @returns Character vector of wrapped lines
wrap_roxygen <- function(text, width = 80, prefix = "#' ") {
  if (is.null(text) || text == "") {
    return(character())
  }

  words <- strsplit(text, "\\s+")[[1]]
  lines <- character()
  current_line <- ""

  for (word in words) {
    test_line <- if (current_line == "") word else paste(current_line, word)
    if (nchar(paste0(prefix, test_line)) > width && current_line != "") {
      lines <- c(lines, paste0(prefix, current_line))
      current_line <- word
    } else {
      current_line <- test_line
    }
  }

  if (current_line != "") {
    lines <- c(lines, paste0(prefix, current_line))
  }

  lines
}

#' Generate roxygen documentation block
#'
#' @param title Title/description from frontmatter
#' @param description Extended description from body
#' @param fn_name Original function name (e.g., "st_length")
#' @param kernel_info Parsed kernel info with params and returns
#' @returns Character string with roxygen block
generate_roxygen <- function(title, description, fn_name, kernel_info) {
  # Title lines
  title_block <- paste(wrap_roxygen(title), collapse = "\n")

  # Extended description (if different from title)
  if (!is.null(description) && description != "" && description != title) {
    desc_lines <- paste(wrap_roxygen(description), collapse = "\n")
    desc_block <- paste0("#'\n", desc_lines, "\n")
  } else {
    desc_block <- ""
  }

  # @seealso
  doc_url <- glue("{docs_base_url}/{fn_name}/")
  seealso_block <- glue(
    "#' @seealso\n#' [SedonaDB SQL documentation for {toupper(fn_name)}]({doc_url})"
  )

  # @param entries
  param_lines <- vapply(
    names(kernel_info$params),
    function(name) {
      glue("#' @param {name} {kernel_info$params[[name]]}")
    },
    character(1)
  )
  param_block <- paste(param_lines, collapse = "\n")

  # @returns
  returns_block <- glue("#' @returns ({kernel_info$returns})")

  # Assemble with paste to ensure proper newlines
  paste0(
    title_block,
    "\n",
    desc_block,
    "#'\n",
    seealso_block,
    "\n",
    "#'\n",
    param_block,
    "\n",
    "#'\n",
    returns_block,
    "\n",
    "#' @export\n",
    "#'\n"
  )
}

#' Generate the function definition
#'
#' @param sd_name Function name (e.g., "sd_length")
#' @param args_str Comma-separated argument string
#' @returns Character string with function definition
generate_function <- function(sd_name, args_str) {
  glue(
    "
{sd_name} <- function({args_str}) {{
  call_sd_function_default()
}}
"
  )
}

#' Generate the translation function
#'
#' @param sd_name Function name (e.g., "sd_length")
#' @param fn_name Original SQL function name (e.g., "st_length")
#' @param args Character vector of argument names
#' @returns Character string with translation function
generate_translation <- function(sd_name, fn_name, args) {
  if (length(args) > 0 && any(nzchar(args))) {
    args_with_defaults <- paste0(args, " = sd_missing_arg()")
    trans_args <- paste(c(".ctx", args_with_defaults), collapse = ", ")
  } else {
    trans_args <- ".ctx"
  }
  list_args <- if (length(args) > 0 && any(nzchar(args))) {
    paste0("list(", paste(args, collapse = ", "), ")")
  } else {
    "list()"
  }

  glue(
    '
{sd_name}_translation <- function({trans_args}) {{
  call_sd_translation_default(
    .ctx,
    "{fn_name}",
    {list_args}
  )
}}
'
  )
}

#' Generate R function file content from parsed .qmd data
#'
#' @param fn_name Function name (e.g., "st_length")
#' @param frontmatter Parsed YAML frontmatter
#' @param description Description text from body
#' @param file_hash MD5 hash of source .qmd file
#' @returns Character string with complete R file content
generate_r_file <- function(fn_name, frontmatter, description, file_hash) {
  sd_name <- sub("^st_", "sd_", fn_name)
  kernel_info <- parse_kernel_params(frontmatter$kernels, fn_name)
  title <- frontmatter$description %||% frontmatter$title
  if (length(kernel_info$args) > 0 && any(nzchar(kernel_info$args))) {
    args_with_defaults <- paste0(kernel_info$args, " = sd_missing_arg()")
    args_str <- paste(args_with_defaults, collapse = ", ")
  } else {
    args_str <- ""
  }

  # Generate pieces
  # nolint start: object_usage_linter
  roxygen <- generate_roxygen(title, description, fn_name, kernel_info)
  fn_def <- generate_function(sd_name, args_str)
  translation <- generate_translation(sd_name, fn_name, kernel_info$args)
  # nolint end

  # Assemble full file
  glue(
    "
{lincense_header}

# Generated from {fn_name}.qmd {file_hash}

{roxygen}{fn_def}

{translation}",
    .trim = FALSE
  )
}

#' Generate an R file from a .qmd documentation file
#'
#' @param qmd_path Path to the .qmd file
#' @param force Force regeneration even if hash matches
#' @returns List with status ("generated", "skipped", "failed") and error message if failed
generate_from_qmd <- function(qmd_path, force = FALSE) {
  fn_name <- tools::file_path_sans_ext(basename(qmd_path))
  sd_name <- sub("^st_", "sd_", fn_name)
  output_path <- file.path(output_dir, paste0(sd_name, ".R"))

  # Compute hash
  file_hash <- rlang::hash_file(qmd_path)

  # Check if regeneration needed
  if (!force && file.exists(output_path)) {
    existing <- readLines(output_path, n = 20, warn = FALSE)
    hash_line <- grep("^# Generated from", existing, value = TRUE)[1]
    if (!is.na(hash_line) && grepl(file_hash, hash_line)) {
      message("Skipping ", fn_name, " (unchanged)")
      return(list(status = "skipped", fn_name = fn_name))
    }
  }

  # Parse and generate with error handling
  result <- tryCatch(
    {
      message("Generating ", sd_name, ".R from ", fn_name, ".qmd")

      frontmatter <- extract_frontmatter(qmd_path)
      description <- extract_description_section(qmd_path)

      content <- generate_r_file(fn_name, frontmatter, description, file_hash)

      writeLines(content, output_path)
      list(status = "generated", fn_name = fn_name)
    },
    error = function(e) {
      list(status = "failed", fn_name = fn_name, error = conditionMessage(e))
    }
  )

  result
}

#' Process specified .qmd files or all st_*.qmd files
#'
#' @param files Character vector of function names (without .qmd) or NULL for all
#' @param force Force regeneration
#' @returns Invisible NULL
update_sd_funcs <- function(files = NULL, force = TRUE) {
  if (is.null(files)) {
    qmd_files <- list.files(docs_dir, pattern = "^(st|rs)_.*\\.qmd$", full.names = TRUE)
  } else {
    qmd_files <- file.path(docs_dir, paste0(files, ".qmd"))
    missing <- !file.exists(qmd_files)
    if (any(missing)) {
      stop("Missing .qmd files: ", paste(files[missing], collapse = ", "))
    }
  }

  generated <- 0
  failed <- list()

  for (qmd_path in qmd_files) {
    result <- generate_from_qmd(qmd_path, force = force)
    if (result$status == "generated") {
      generated <- generated + 1
    } else if (result$status == "failed") {
      failed <- c(failed, list(result))
    }
  }

  message("Generated ", generated, " files")

  if (length(failed) > 0) {
    message("\nFailed to generate ", length(failed), " files:")
    for (f in failed) {
      message("  - ", f$fn_name, ": ", f$error)
    }
  }

  invisible(NULL)
}
