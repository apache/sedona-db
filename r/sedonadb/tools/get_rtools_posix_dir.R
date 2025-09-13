rtools_home <-
  if (identical(R.version$major, "4")) {
    minor <- substr(R.version$minor, 0, 1)
    if (identical(minor, "4")) {
      Sys.getenv("RTOOLS44_HOME")
    } else if (identical(minor, "5")) {
      Sys.getenv("RTOOLS45_HOME")
    }
  }

if (is.null(rtools_home) || identical(rtools_home, "")) {
  stop("Failed to detect Rtools home!")
}

cat(file.path(rtools_home, "x86_64-w64-mingw32.static.posix"))
