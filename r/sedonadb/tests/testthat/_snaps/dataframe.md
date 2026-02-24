# dataframe can be printed

    Code
      print(df)
    Output
      <sedonab_dataframe: NA x 1>
      +------------+
      |     pt     |
      |  geometry  |
      +------------+
      | POINT(0 1) |
      +------------+
      Preview of up to 6 row(s)

---

    Code
      print(grouped)
    Output
      <grouped sedonab_dataframe: NA x 1 | [`x`]>
      +------------+
      |     pt     |
      |  geometry  |
      +------------+
      | POINT(0 1) |
      +------------+
      Preview of up to 6 row(s)

# sd_write_parquet validates geoparquet_version parameter

    This feature is not implemented: GeoParquetVersion V2_0 is not yet supported

# sd_write_parquet() errors for inappropriately sized options

    All option values must be length 1

---

    All option values must be named

