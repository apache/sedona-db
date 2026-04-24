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

# sd_join_select_default() prints nicely

    Code
      print(spec)
    Output
      <sedonadb_join_select_default>
        suffix: c(".x", ".y")

---

    Code
      print(spec2)
    Output
      <sedonadb_join_select_default>
        suffix: c("_l", "_r")

# sd_join_select() prints nicely

    Code
      print(sel1)
    Output
      <sedonadb_join_select>
        x$id
        y$value

---

    Code
      print(sel2)
    Output
      <sedonadb_join_select>
        out = x$id
        val = y$value

# sd_eval_join_select_exprs() evaluates column references

    Code
      sd_eval_join_select_exprs(sd_join_select(x$id, y$value), ctx)
    Output
      $id
      <SedonaDBExpr>
      x.id
      
      $value
      <SedonaDBExpr>
      y.value
      

# sd_eval_join_select_exprs() handles renaming

    Code
      sd_eval_join_select_exprs(sd_join_select(my_id = x$id, my_val = y$value,
      x_name = x$name), ctx)
    Output
      $my_id
      <SedonaDBExpr>
      x.id
      
      $my_val
      <SedonaDBExpr>
      y.value
      
      $x_name
      <SedonaDBExpr>
      x.name
      

# sd_build_default_select() removes y-side equijoin keys

    Code
      print(result)
    Output
      $id
      <SedonaDBExpr>
      x.id
      
      $x_val
      <SedonaDBExpr>
      x.x_val
      
      $y_val
      <SedonaDBExpr>
      y.y_val
      

