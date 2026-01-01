# expressions can be printed

    Code
      print(as_sedonadb_literal("foofy"))
    Output
      <SedonaDBExpr>
      Literal(Utf8("foofy"), None)

# literal expressions can be translated

    Code
      sd_eval_expr(quote(1L))
    Output
      <SedonaDBExpr>
      Literal(Int32(1), None)

# column expressions can be translated

    Code
      sd_eval_expr(quote(col0), expr_ctx)
    Output
      <SedonaDBExpr>
      Column(Column { relation: None, name: "col0" })

---

    Code
      sd_eval_expr(quote(.data$col0), expr_ctx)
    Output
      <SedonaDBExpr>
      Column(Column { relation: None, name: "col0" })

---

    Code
      sd_eval_expr(quote(.data[[col_zero]]), expr_ctx)
    Output
      <SedonaDBExpr>
      Column(Column { relation: None, name: "col0" })

# function calls containing no SedonaDB expressions can be translated

    Code
      sd_eval_expr(quote(abs(-1L)))
    Output
      <SedonaDBExpr>
      Literal(Int32(1), None)

# function calls containing SedonaDB expressions can be translated

    Code
      sd_eval_expr(quote(abs(col0)), expr_ctx)
    Output
      <SedonaDBExpr>
      ScalarFunction(ScalarFunction { func: ScalarUDF { inner: AbsFunc { signature: Signature { type_signature: Numeric(1), volatility: Immutable } } }, args: [Column(Column { relation: None, name: "col0" })] })

