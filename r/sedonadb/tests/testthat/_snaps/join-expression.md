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

# sd_eval_join_conditions() evaluates symbols as equality conditions

    Code
      print(sd_eval_join_conditions(sd_join_by(id), ctx))
    Output
      [[1]]
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

# sd_build_join_conditions() creates natural join when by is NULL

    Code
      print(sd_build_join_conditions(ctx))
    Message
      Joining with `by = sd_join_by(id)`
    Output
      [[1]]
      <SedonaDBExpr>
      x.id = y.id
      

# sd_build_join_conditions() creates equijoin conditions for names

    Code
      sd_build_join_conditions(ctx, c("id", x_val = "y_val"))
    Output
      [[1]]
      <SedonaDBExpr>
      x.id = y.id
      
      [[2]]
      <SedonaDBExpr>
      x.x_val = y.y_val
      

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
      sd_eval_join_select_exprs(sd_join_select(x$id, y$value, name), ctx)
    Output
      $id
      <SedonaDBExpr>
      x.id
      
      $value
      <SedonaDBExpr>
      y.value
      
      $name
      <SedonaDBExpr>
      x.name
      

# sd_eval_join_select_exprs() handles renaming

    Code
      sd_eval_join_select_exprs(sd_join_select(my_id = x$id, my_val = y$value,
      x_name = name), ctx)
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
      

# sd_build_default_select() handles equijoin keys

    Code
      sd_build_default_select(ctx, conditions, join_type = "inner")
    Output
      $id_x
      <SedonaDBExpr>
      x.id_x
      
      $x_key
      <SedonaDBExpr>
      x.x_key
      
      $id_y
      <SedonaDBExpr>
      y.id_y
      

---

    Code
      sd_build_default_select(ctx, conditions, join_type = "left")
    Output
      $id_x
      <SedonaDBExpr>
      x.id_x
      
      $x_key
      <SedonaDBExpr>
      x.x_key
      
      $id_y
      <SedonaDBExpr>
      y.id_y
      

---

    Code
      sd_build_default_select(ctx, conditions, join_type = "right")
    Output
      $id_x
      <SedonaDBExpr>
      x.id_x
      
      $x_key
      <SedonaDBExpr>
      y.y_key
      
      $id_y
      <SedonaDBExpr>
      y.id_y
      

---

    Code
      sd_build_default_select(ctx, conditions, join_type = "full")
    Output
      $id_x
      <SedonaDBExpr>
      x.id_x
      
      $x_key
      <SedonaDBExpr>
      coalesce(x.x_key, y.y_key)
      
      $id_y
      <SedonaDBExpr>
      y.id_y
      

