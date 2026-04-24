# sd_join_by() prints nicely

    Code
      print(jb1)
    Output
      <sedonadb_join_by>
        x$id == y$id

---

    Code
      print(jb2)
    Output
      <sedonadb_join_by>
        x$id == y$id
        x$value > y$threshold

# qualified column references produce correct expressions

    Code
      x_id
    Output
      <SedonaDBExpr>
      x.id

---

    Code
      y_id
    Output
      <SedonaDBExpr>
      y.id

# sd_eval_join_conditions() evaluates equality conditions

    Code
      conditions[[1]]
    Output
      <SedonaDBExpr>
      x.id = y.id

# sd_eval_join_conditions() evaluates inequality conditions

    Code
      conditions[[1]]
    Output
      <SedonaDBExpr>
      x.value > y.threshold

# sd_eval_join_conditions() evaluates multiple conditions

    Code
      conditions[[1]]
    Output
      <SedonaDBExpr>
      x.id = y.id

---

    Code
      conditions[[2]]
    Output
      <SedonaDBExpr>
      x.date >= y.start_date

