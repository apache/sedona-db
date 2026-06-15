# summarise() errors on unsupported .groups values

    sdplyr::summarise() only supports .groups = "drop"

---

    Can't supply both `.by` and `.groups`.

# summarise() messages for implicit .groups

    Code
      summarise(df, total = sum(x))
    Message
      In sdplyr, groups are dropped by default on summarise
      i Use .groups = "drop" to suppress this message
    Output
      <sedonab_dataframe: NA x 1>
      +---------+
      |  total  |
      | float64 |
      +---------+
      |     6.0 |
      +---------+
      Preview of up to 6 row(s)

