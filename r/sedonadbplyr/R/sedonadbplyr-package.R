#' @keywords internal
"_PACKAGE"

## usethis namespace: start
## usethis namespace: end
NULL

sdplyr_unsupported <- function() {
  structure(list(), class = "sdplyr_unsupported")
}

assert_unsupported <- function(...) {
  args <- tibble::lst(...)
  args_is_unsupported <- vapply(args, inherits, logical(1), "sdplyr_unsupported")
  arg_names <- names(args)

  if (!all(args_is_unsupported)) {
    bad_args <- arg_names[!args_is_unsupported] # nolint: object_usage_linter
    cli::cli_abort(
      c(
        "{cli::qty(bad_args)} Argument{?s} {.arg {bad_args}} {?is/are} not supported
        by sedonadbplyr."
      ),
      call = parent.frame()
    )
  }
}
