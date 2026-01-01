# expressions can be printed

    Code
      print(as_sedonadb_literal("foofy"))
    Output
      <SedonaDBExpr>
      Utf8("foofy")

# literal expressions can be translated

    Code
      sd_eval_expr(quote(1L))
    Output
      <SedonaDBExpr>
      Int32(1)

# column expressions can be translated

    Code
      sd_eval_expr(quote(col0), expr_ctx)
    Output
      <SedonaDBExpr>
      col0

---

    Code
      sd_eval_expr(quote(.data$col0), expr_ctx)
    Output
      <SedonaDBExpr>
      col0

---

    Code
      sd_eval_expr(quote(.data[[col_zero]]), expr_ctx)
    Output
      <SedonaDBExpr>
      col0

# function calls with a translation become function calls

    Code
      sd_eval_expr(quote(abs(-1L)))
    Output
      <SedonaDBExpr>
      abs(Int32(-1))

# function calls without a translation are evaluated in R

    Code
      sd_eval_expr(quote(function_without_a_translation(1L)))
    Output
      <SedonaDBExpr>
      Int32(2)

