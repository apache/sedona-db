// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Logical optimizer rule that wraps raster arguments of `needs_bytes`
//! UDFs with `RS_EnsureLoaded`, so OutDb byte materialisation happens
//! explicitly in the logical plan instead of as a hidden side effect
//! inside the kernel.
//!
//! After this rule, calls like `RS_Value(raster, x, y)` (where
//! `RS_Value` is annotated with the `needs_pixels` metadata flag) become
//! `RS_Value(RS_EnsureLoaded(raster), x, y)`. DataFusion's
//! `CommonSubexprEliminate` pass deduplicates identical
//! `RS_EnsureLoaded(col)` calls across multiple `needs_bytes` UDFs
//! sharing the same raster column — provided `RS_EnsureLoaded`'s
//! signature is `Volatility::Stable` (not `Volatile`).
//!
//! This is a logical optimizer rule (not an analyzer rule) so it can
//! look `RS_EnsureLoaded` up from the [`FunctionRegistry`] rather than
//! capturing an `Arc` at construction time. Because optimizer rules run
//! to a fixpoint, the rewrite is idempotent: an argument already wrapped
//! in `RS_EnsureLoaded` is left alone (see [`is_ensure_loaded_call`]).

use std::sync::Arc;

use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_common::{DFSchema, Result};
use datafusion_expr::expr::ScalarFunction;
use datafusion_expr::expr_schema::ExprSchemable;
use datafusion_expr::{Expr, LogicalPlan, ScalarUDF};
use datafusion_optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};
use sedona_common::sedona_internal_err;
use sedona_expr::scalar_udf::SedonaScalarUDF;
use sedona_schema::datatypes::SedonaType;

/// Logical optimizer rule wrapping raster arguments of `needs_bytes`
/// UDFs with `RS_EnsureLoaded`. Stateless — the `RS_EnsureLoaded` UDF
/// is resolved from the session's [`FunctionRegistry`] at rewrite time.
#[derive(Default, Debug)]
pub struct EnsureLoadedOptimizerRule;

impl OptimizerRule for EnsureLoadedOptimizerRule {
    fn name(&self) -> &str {
        "sedona.ensure_loaded"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        // Bottom-up so a nested `RS_X(RS_Y(rast))` is rewritten
        // inside-out: the inner call's raster arg is wrapped first, then
        // the outer call sees the (now-wrapped, still raster-typed) arg
        // and the idempotency guard keeps it from double-wrapping.
        Some(ApplyOrder::BottomUp)
    }

    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        // Resolve RS_EnsureLoaded from the registry. A context that never
        // registered it (no raster support) has nothing to rewrite.
        let Some(registry) = config.function_registry() else {
            return Ok(Transformed::no(plan));
        };
        let Ok(ensure_loaded_udf) = registry.udf("rs_ensureloaded") else {
            return Ok(Transformed::no(plan));
        };

        // Expressions on a node are evaluated against the combined schema
        // of its inputs: single-input nodes (Projection, Filter, …) use
        // their one input; a Join's `filter` references left ⋈ right, so
        // a predicate like `RS_Intersects(a.rast, b.rast)` resolves only
        // against the merged schema. Leaf nodes carry no wrappable
        // expressions.
        let inputs = plan.inputs();
        if inputs.is_empty() {
            return Ok(Transformed::no(plan));
        }
        let Some(schema) = merged_input_schema(&inputs) else {
            // Schemas couldn't be merged (e.g. ambiguous duplicate
            // qualifiers in a self-join). Skip this node rather than
            // failing the query — a missed wrap surfaces later as a clear
            // "raster bytes not loaded" error, not a wrong result.
            return Ok(Transformed::no(plan));
        };
        drop(inputs);

        plan.map_expressions(|e| {
            e.transform_up(|expr| rewrite_expr_node(expr, &schema, &ensure_loaded_udf))
        })
    }
}

/// Merge the schemas of all inputs into one. Returns `None` if the merge
/// fails (DataFusion's [`DFSchema::join`] errors on ambiguous duplicate
/// qualified fields).
fn merged_input_schema(inputs: &[&LogicalPlan]) -> Option<Arc<DFSchema>> {
    let mut merged = inputs[0].schema().as_ref().clone();
    for input in &inputs[1..] {
        merged = merged.join(input.schema()).ok()?;
    }
    Some(Arc::new(merged))
}

/// Single-step rewrite: if `expr` is a `needs_bytes` UDF call, wrap each
/// raster-typed arg with `RS_EnsureLoaded`. Two guards keep it correct:
/// it never wraps `RS_EnsureLoaded` itself (recursion), and it never
/// re-wraps an arg already wrapped in `RS_EnsureLoaded` (idempotency,
/// required because optimizer rules run to a fixpoint).
fn rewrite_expr_node(
    expr: Expr,
    schema: &Arc<DFSchema>,
    ensure_loaded_udf: &Arc<ScalarUDF>,
) -> Result<Transformed<Expr>> {
    let Expr::ScalarFunction(ref func_call) = expr else {
        return Ok(Transformed::no(expr));
    };

    // Recursion guard.
    if func_call.func.name() == "rs_ensureloaded" {
        return Ok(Transformed::no(expr));
    }

    // Only annotated SedonaScalarUDFs participate. DataFusion built-ins
    // and unannotated UDFs pass through unchanged.
    let needs_bytes = func_call
        .func
        .inner()
        .as_any()
        .downcast_ref::<SedonaScalarUDF>()
        .map(|u| u.needs_bytes())
        .unwrap_or(false);
    if !needs_bytes {
        return Ok(Transformed::no(expr));
    }

    // Structurally impossible: we matched `expr` as `Expr::ScalarFunction`
    // a few lines up. Surface it as an internal error rather than a panic
    // so a future refactor that breaks the invariant fails the query
    // cleanly instead of crashing a worker.
    let Expr::ScalarFunction(ScalarFunction { func, args }) = expr else {
        return sedona_internal_err!(
            "rewrite_expr_node: expected ScalarFunction after match, got a different Expr variant"
        );
    };
    let mut changed = false;
    let new_args: Vec<Expr> = args
        .into_iter()
        .map(|arg| {
            // Idempotency guard: a fixpoint re-run sees the wrapped arg
            // (still raster-typed after RS_EnsureLoaded's identity
            // `return_field`); don't wrap it again.
            if is_ensure_loaded_call(&arg) {
                return arg;
            }
            if expr_is_raster(&arg, schema) {
                changed = true;
                Expr::ScalarFunction(ScalarFunction {
                    func: Arc::clone(ensure_loaded_udf),
                    args: vec![arg],
                })
            } else {
                arg
            }
        })
        .collect();

    let rewritten = Expr::ScalarFunction(ScalarFunction {
        func,
        args: new_args,
    });
    if changed {
        Ok(Transformed::yes(rewritten))
    } else {
        Ok(Transformed::no(rewritten))
    }
}

/// True if `expr` is a call to `RS_EnsureLoaded`.
fn is_ensure_loaded_call(expr: &Expr) -> bool {
    matches!(expr, Expr::ScalarFunction(sf) if sf.func.name() == "rs_ensureloaded")
}

/// True if `expr` evaluates to a `SedonaType::Raster` under the given
/// schema. Uses `to_field` (not `get_type`) so the Field's extension
/// metadata is available — `SedonaType::Raster` is identified by an
/// `"sedona.raster"` extension type, not by raw `DataType::Struct`.
fn expr_is_raster(expr: &Expr, schema: &Arc<DFSchema>) -> bool {
    let Ok((_, field)) = expr.to_field(schema.as_ref()) else {
        return false;
    };
    matches!(
        SedonaType::from_storage_field(&field),
        Ok(SedonaType::Raster)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_schema::{DataType, Field, Schema};
    use datafusion_common::tree_node::TreeNodeRecursion;
    use datafusion_expr::{col, ScalarUDF, Volatility};
    use sedona_expr::scalar_udf::{
        ScalarKernelRef, SedonaScalarUDF, SimpleSedonaScalarKernel, NEEDS_PIXELS_METADATA_KEY,
    };
    use sedona_schema::matchers::ArgMatcher;

    /// A stand-in `rs_ensureloaded` UDF. The rule keys off the name and
    /// the `needs_bytes` marker, never the real async impl (which lives
    /// in the `sedona` crate and can't be referenced here), so a plain
    /// SedonaScalarUDF carrying the canonical name is sufficient.
    fn fake_ensure_loaded_udf() -> Arc<ScalarUDF> {
        let kernel: ScalarKernelRef = SimpleSedonaScalarKernel::new_ref(
            ArgMatcher::new(vec![ArgMatcher::is_raster()], SedonaType::Raster),
            Arc::new(|_, _| unreachable!("stub kernel; rewrite never invokes it")),
        );
        let udf = SedonaScalarUDF::new("rs_ensureloaded", vec![kernel], Volatility::Immutable);
        Arc::new(ScalarUDF::new_from_impl(udf))
    }

    /// A `needs_bytes` UDF accepting a raster, returning Int32.
    fn needs_bytes_udf(name: &str) -> Arc<ScalarUDF> {
        let kernel: ScalarKernelRef = SimpleSedonaScalarKernel::new_ref(
            ArgMatcher::new(
                vec![ArgMatcher::is_raster()],
                SedonaType::Arrow(DataType::Int32),
            ),
            Arc::new(|_, _| unreachable!("stub kernel; not invoked at plan time")),
        );
        let udf = SedonaScalarUDF::new(name, vec![kernel], Volatility::Immutable)
            .with_metadata(NEEDS_PIXELS_METADATA_KEY, "true");
        Arc::new(ScalarUDF::new_from_impl(udf))
    }

    /// Same shape but without the `needs_bytes` annotation.
    fn metadata_only_udf(name: &str) -> Arc<ScalarUDF> {
        let kernel: ScalarKernelRef = SimpleSedonaScalarKernel::new_ref(
            ArgMatcher::new(
                vec![ArgMatcher::is_raster()],
                SedonaType::Arrow(DataType::Int32),
            ),
            Arc::new(|_, _| unreachable!("stub kernel; not invoked at plan time")),
        );
        let udf = SedonaScalarUDF::new(name, vec![kernel], Volatility::Immutable);
        Arc::new(ScalarUDF::new_from_impl(udf))
    }

    fn raster_schema_named(name: &str) -> Arc<DFSchema> {
        let field = SedonaType::Raster.to_storage_field(name, true).unwrap();
        let arrow_schema = Arc::new(Schema::new(vec![field]));
        Arc::new(DFSchema::try_from(arrow_schema.as_ref().clone()).unwrap())
    }

    fn int_schema(name: &str) -> Arc<DFSchema> {
        let field = Field::new(name, DataType::Int64, true);
        let arrow_schema = Arc::new(Schema::new(vec![field]));
        Arc::new(DFSchema::try_from(arrow_schema.as_ref().clone()).unwrap())
    }

    fn count_ensure_loaded(expr: &Expr) -> usize {
        let mut n = 0;
        expr.apply(|e| {
            if let Expr::ScalarFunction(sf) = e {
                if sf.func.name() == "rs_ensureloaded" {
                    n += 1;
                }
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .unwrap();
        n
    }

    fn rewrite(expr: Expr, schema: &Arc<DFSchema>, udf: &Arc<ScalarUDF>) -> Expr {
        rewrite_expr_node(expr, schema, udf).unwrap().data
    }

    #[test]
    fn wraps_raster_arg_of_needs_bytes_udf() {
        let schema = raster_schema_named("rast");
        let udf = fake_ensure_loaded_udf();
        let call = Expr::ScalarFunction(ScalarFunction {
            func: needs_bytes_udf("rs_mock"),
            args: vec![col("rast")],
        });
        let out = rewrite(call, &schema, &udf);
        let Expr::ScalarFunction(ScalarFunction { args, .. }) = &out else {
            panic!("expected ScalarFunction, got {out:?}");
        };
        assert!(
            is_ensure_loaded_call(&args[0]),
            "raster arg should be wrapped"
        );
    }

    #[test]
    fn leaves_non_raster_args_alone() {
        let schema = int_schema("n");
        let udf = fake_ensure_loaded_udf();
        let call = Expr::ScalarFunction(ScalarFunction {
            func: needs_bytes_udf("rs_mock"),
            args: vec![col("n")],
        });
        let out = rewrite(call, &schema, &udf);
        assert_eq!(count_ensure_loaded(&out), 0);
    }

    #[test]
    fn leaves_metadata_only_udfs_alone() {
        let schema = raster_schema_named("rast");
        let udf = fake_ensure_loaded_udf();
        let call = Expr::ScalarFunction(ScalarFunction {
            func: metadata_only_udf("rs_meta"),
            args: vec![col("rast")],
        });
        let out = rewrite(call, &schema, &udf);
        assert_eq!(count_ensure_loaded(&out), 0);
    }

    #[test]
    fn recursion_guard_does_not_wrap_rs_ensure_loaded_itself() {
        let schema = raster_schema_named("rast");
        let udf = fake_ensure_loaded_udf();
        let call = Expr::ScalarFunction(ScalarFunction {
            func: Arc::clone(&udf),
            args: vec![col("rast")],
        });
        let out = rewrite(call, &schema, &udf);
        // Still exactly one — its raster arg is not itself wrapped.
        assert_eq!(count_ensure_loaded(&out), 1);
    }

    #[test]
    fn idempotency_guard_does_not_rewrap_already_wrapped_arg() {
        // Models the fixpoint re-run: the input already has
        // rs_mock(rs_ensureloaded(rast)). A second pass must NOT produce
        // rs_mock(rs_ensureloaded(rs_ensureloaded(rast))).
        let schema = raster_schema_named("rast");
        let udf = fake_ensure_loaded_udf();
        let already_wrapped = Expr::ScalarFunction(ScalarFunction {
            func: Arc::clone(&udf),
            args: vec![col("rast")],
        });
        let call = Expr::ScalarFunction(ScalarFunction {
            func: needs_bytes_udf("rs_mock"),
            args: vec![already_wrapped],
        });
        let out = rewrite(call, &schema, &udf);
        assert_eq!(
            count_ensure_loaded(&out),
            1,
            "already-wrapped arg must not be wrapped again: {out:?}"
        );
    }

    #[test]
    fn registers_immediately_before_cse() {
        use crate::optimizer::register_ensure_loaded_optimizer;
        use datafusion::execution::session_state::SessionStateBuilder;

        let builder = SessionStateBuilder::new().with_default_features();
        let mut builder = register_ensure_loaded_optimizer(builder).unwrap();

        let rules = &builder.optimizer().as_ref().unwrap().rules;
        let ours = rules
            .iter()
            .position(|r| r.name() == "sedona.ensure_loaded")
            .expect("rule registered");
        let cse = rules
            .iter()
            .position(|r| r.name() == "common_sub_expression_eliminate")
            .expect("CSE present in default optimizer");
        assert_eq!(
            ours + 1,
            cse,
            "ensure_loaded must sit immediately before CSE so wraps dedupe in the same pass"
        );
    }

    #[test]
    fn merged_schema_resolves_raster_across_a_join() {
        // Two single-raster inputs (left `a`, right `b`); the merged
        // schema must see both so a join filter referencing either side's
        // raster resolves and gets wrapped.
        let left = LogicalPlan::EmptyRelation(datafusion_expr::EmptyRelation {
            produce_one_row: false,
            schema: raster_schema_named("a"),
        });
        let right = LogicalPlan::EmptyRelation(datafusion_expr::EmptyRelation {
            produce_one_row: false,
            schema: raster_schema_named("b"),
        });
        let inputs = [&left, &right];
        let merged = merged_input_schema(&inputs).expect("schemas merge");

        let udf = fake_ensure_loaded_udf();
        // rs_mock(b) — the right side's raster, only resolvable via the
        // merged schema.
        let call = Expr::ScalarFunction(ScalarFunction {
            func: needs_bytes_udf("rs_mock"),
            args: vec![col("b")],
        });
        let out = rewrite(call, &merged, &udf);
        assert_eq!(
            count_ensure_loaded(&out),
            1,
            "raster arg from the right join input should be wrapped: {out:?}"
        );
    }
}
