# basic expression types can be constructed

    Code
      sd_expr_column("foofy")
    Output
      <SedonaDBExpr>
      foofy

---

    Code
      sd_expr_literal(1L)
    Output
      <SedonaDBExpr>
      Int32(1)

---

    Code
      sd_expr_scalar_function("abs", list(1L))
    Output
      <SedonaDBExpr>
      abs(Int32(1))

---

    Code
      sd_expr_cast(1L, nanoarrow::na_int64())
    Output
      <SedonaDBExpr>
      CAST(Int32(1) AS Int64)

---

    Code
      sd_expr_alias(1L, "foofy")
    Output
      <SedonaDBExpr>
      Int32(1) AS foofy

---

    Code
      sd_expr_binary("+", 1L, 2L)
    Output
      <SedonaDBExpr>
      Int32(1) + Int32(2)

---

    Code
      sd_expr_negative(1L)
    Output
      <SedonaDBExpr>
      (- Int32(1))

---

    Code
      sd_expr_aggregate_function("sum", list(1L))
    Output
      <SedonaDBExpr>
      sum(Int32(1)) RESPECT NULLS

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
      abs((- Int32(1)))

---

    Code
      sd_eval_expr(quote(base::abs(-1L)))
    Output
      <SedonaDBExpr>
      abs((- Int32(1)))

# function calls explicitly referencing DataFusion functions work

    Code
      sd_eval_expr(quote(.fns$abs(-1L)))
    Output
      <SedonaDBExpr>
      abs((- Int32(1)))

---

    Error evaluating translated expression `.fns$absolutely_not(-1L)`
    Caused by error:
    ! Scalar UDF 'absolutely_not' not found

# function calls referencing SedonaDB SQL functions work

    Code
      sd_eval_expr(quote(st_point(0, 1)))
    Output
      <SedonaDBExpr>
      st_point(Float64(0), Float64(1))

---

    Code
      sd_eval_expr(quote(st_envelope_agg(st_point(0, 1))))
    Output
      <SedonaDBExpr>
      st_envelope_agg(st_point(Float64(0), Float64(1))) RESPECT NULLS

---

    Code
      sd_eval_expr(quote(st_envelope_agg(st_point(0, 1), na.rm = TRUE)))
    Output
      <SedonaDBExpr>
      st_envelope_agg(st_point(Float64(0), Float64(1))) IGNORE NULLS

---

    Error evaluating translated expression `st_point(1, y = 2)`
    Caused by error:
    ! Arguments in `...` must be passed by position, not name.
    x Problematic argument:
    * y = 2

# function calls without a translation are evaluated in R

    Code
      sd_eval_expr(quote(function_without_a_translation(1L)))
    Output
      <SedonaDBExpr>
      Int32(2)

# function calls that map to binary expressions are translated

    Code
      sd_eval_expr(quote(+2))
    Output
      <SedonaDBExpr>
      (- (- Float64(2)))

---

    Code
      sd_eval_expr(quote(1 + 2))
    Output
      <SedonaDBExpr>
      Float64(1) + Float64(2)

---

    Code
      sd_eval_expr(quote(-2))
    Output
      <SedonaDBExpr>
      (- Float64(2))

---

    Code
      sd_eval_expr(quote(1 - 2))
    Output
      <SedonaDBExpr>
      Float64(1) - Float64(2)

---

    Code
      sd_eval_expr(quote(1 > 2))
    Output
      <SedonaDBExpr>
      Float64(1) > Float64(2)

# errors that occur during evaluation have reasonable context

    Code
      sd_eval_expr(quote(stop("this will error")))
    Condition
      Error in `sd_eval_expr()`:
      ! Error evaluating translated expression `stop("this will error")`
      Caused by error:
      ! this will error

