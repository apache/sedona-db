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
  geography = "geog",
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
#' @returns List with params (for roxygen) and args (for function signature)
parse_kernel_params <- function(kernels) {
  if (length(kernels) == 0) {
    return(list(params = list(), args = character(), returns = "unknown"))
  }

  # Use first kernel as reference (most common case)
  kernel <- kernels[[1]]
  args <- kernel$args
  returns <- kernel$returns %||% "unknown"

  # First pass: extract types and count occurrences
  arg_info <- lapply(args, function(arg) {
    if (is.character(arg)) {
      list(type = arg, name = NULL, desc = NULL)
    } else if (is.list(arg)) {
      list(type = arg$type %||% "unknown", name = arg$name, desc = arg$description)
    } else {
      list(type = "unknown", name = NULL, desc = NULL)
    }
  })

  # Count types that need disambiguation
  types <- vapply(arg_info, function(x) x$type, character(1))
  type_totals <- table(types)

  # Second pass: generate names
  params <- list()
  arg_names <- character()
  type_counts <- list()

  for (info in arg_info) {
    arg_type <- info$type
    arg_name <- info$name
    arg_desc <- info$desc

    # Generate name if not provided
    if (is.null(arg_name)) {
      type_counts[[arg_type]] <- (type_counts[[arg_type]] %||% 0) + 1
      needs_suffix <- type_totals[[arg_type]] > 1
      arg_name <- type_to_param_name(arg_type, type_counts[[arg_type]], needs_suffix)
    }

    # Generate description
    if (is.null(arg_desc)) {
      arg_desc <- paste0("(", arg_type, "): Input ", arg_type)
    } else {
      arg_desc <- paste0("(", arg_type, "): ", trimws(arg_desc))
    }

    params[[arg_name]] <- arg_desc
    arg_names <- c(arg_names, arg_name)
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
  seealso_block <- glue("#' @seealso\n#' [{toupper(fn_name)}]({doc_url})")

  # @param entries
  param_lines <- vapply(
    names(kernel_info$params),
    function(name) {
      glue("#' @param {name} {kernel_info$params[[name]]}")
    },
    character(1)
  )
  param_block <- paste(
    c(param_lines, "#' @param ... For S3 generic compatibility. Must be empty."),
    collapse = "\n"
  )

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

#' Generate the S3 generic function definition
#'
#' @param sd_name Function name (e.g., "sd_length")
#' @param args_str Comma-separated argument string
#' @returns Character string with function definition
generate_s3_generic <- function(sd_name, args_str) {
  glue(
    '
{sd_name} <- function({args_str}) {{
    UseMethod("{sd_name}")
}}
'
  )
}

#' Generate the default method
#'
#' @param sd_name Function name (e.g., "sd_length")
#' @param args_str Comma-separated argument string
#' @returns Character string with default method
generate_default_method <- function(sd_name, args_str) {
  glue(
    "
#\' @export
{sd_name}.default <- function({args_str}) {{
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
  trans_args <- paste(c(".ctx", args), collapse = ", ")
  list_args <- if (length(args) > 0) {
    paste0("list(", paste(args, collapse = ", "), ")")
  } else {
    "list()"
  }

  glue(
    '
{sd_name}_translation <- function({trans_args}) {{
    sedonadb::sd_expr_any_function(
        "{fn_name}",
        {list_args},
        factory = .ctx$factory
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
  kernel_info <- parse_kernel_params(frontmatter$kernels)
  title <- frontmatter$description %||% frontmatter$title
  args_str <- paste(c(kernel_info$args, "..."), collapse = ", ")

  # Generate pieces
  # nolint start: object_usage_linter
  roxygen <- generate_roxygen(title, description, fn_name, kernel_info)
  generic <- generate_s3_generic(sd_name, args_str)
  default <- generate_default_method(sd_name, args_str)
  translation <- generate_translation(sd_name, fn_name, kernel_info$args)
  # nolint end

  # Assemble full file
  glue(
    "
{lincense_header}

# Generated from {fn_name}.qmd {file_hash}

{roxygen}{generic}
{default}
{translation}
",
    .trim = FALSE
  )
}

#' Generate an R file from a .qmd documentation file
#'
#' @param qmd_path Path to the .qmd file
#' @param force Force regeneration even if hash matches
#' @returns TRUE if file was generated, FALSE if skipped
generate_from_qmd <- function(qmd_path, force = FALSE) {
  fn_name <- tools::file_path_sans_ext(basename(qmd_path))
  sd_name <- sub("^st_", "sd_", fn_name)
  output_path <- file.path(output_dir, paste0(sd_name, ".R"))

  # Compute hash
  file_hash <- digest::digest(file = qmd_path, algo = "md5")

  # Check if regeneration needed
  if (!force && file.exists(output_path)) {
    existing <- readLines(output_path, n = 20, warn = FALSE)
    hash_line <- grep("^# Generated from", existing, value = TRUE)[1]
    if (!is.na(hash_line) && grepl(file_hash, hash_line)) {
      message("Skipping ", fn_name, " (unchanged)")
      return(FALSE)
    }
  }

  # Parse and generate
  message("Generating ", sd_name, ".R from ", fn_name, ".qmd")

  frontmatter <- extract_frontmatter(qmd_path)
  description <- extract_description_section(qmd_path)

  content <- generate_r_file(fn_name, frontmatter, description, file_hash)

  writeLines(content, output_path)
  TRUE
}

#' Process specified .qmd files or all st_*.qmd files
#'
#' @param files Character vector of function names (without .qmd) or NULL for all
#' @param force Force regeneration
#' @returns Invisible NULL
update_sd_funcs <- function(files = NULL, force = FALSE) {
  if (is.null(files)) {
    qmd_files <- list.files(docs_dir, pattern = "^st_.*\\.qmd$", full.names = TRUE)
  } else {
    qmd_files <- file.path(docs_dir, paste0(files, ".qmd"))
    missing <- !file.exists(qmd_files)
    if (any(missing)) {
      stop("Missing .qmd files: ", paste(files[missing], collapse = ", "))
    }
  }

  generated <- 0
  for (qmd_path in qmd_files) {
    if (generate_from_qmd(qmd_path, force = force)) {
      generated <- generated + 1
    }
  }

  message("Generated ", generated, " files")

  invisible(NULL)
}
