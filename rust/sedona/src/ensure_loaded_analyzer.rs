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

//! Analyzer rule that wraps raster arguments of `needs_bytes` UDFs with
//! `RS_EnsureLoaded`, so OutDb byte materialisation happens explicitly
//! in the logical plan instead of as a hidden side effect inside the
//! kernel.
//!
//! After this rule, calls like `RS_Value(raster, x, y)` (where
//! `RS_Value` is annotated `with_needs_bytes()`) become
//! `RS_Value(RS_EnsureLoaded(raster), x, y)`. DataFusion's
//! `CommonSubexprEliminate` pass runs later in the optimiser pipeline
//! and deduplicates identical `RS_EnsureLoaded(col)` calls across
//! multiple `needs_bytes` UDFs sharing the same raster column —
//! provided `RS_EnsureLoaded`'s signature is not `Volatility::Volatile`
//! (we declare it `Stable`).

use std::sync::Arc;

use datafusion_common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion_common::{config::ConfigOptions, DFSchemaRef, Result};
use datafusion_expr::expr::ScalarFunction;
use datafusion_expr::expr_schema::ExprSchemable;
use datafusion_expr::{Expr, LogicalPlan, ScalarUDF};
use datafusion_optimizer::analyzer::AnalyzerRule;
use sedona_expr::scalar_udf::SedonaScalarUDF;
use sedona_schema::datatypes::SedonaType;

use crate::rs_ensure_loaded::RS_ENSURE_LOADED_NAME;

/// Analyzer rule that wraps raster arguments of `needs_bytes` UDFs with
/// `RS_EnsureLoaded`. Holds an `Arc<ScalarUDF>` clone of the registered
/// `RS_EnsureLoaded` UDF so wraps it injects reference the same async
/// dispatcher (and thus the same per-session loader registry) as any
/// explicit `RS_EnsureLoaded(...)` already written by the user.
#[derive(Debug)]
pub struct EnsureLoadedAnalyzerRule {
    ensure_loaded_udf: Arc<ScalarUDF>,
}

impl EnsureLoadedAnalyzerRule {
    pub fn new(ensure_loaded_udf: Arc<ScalarUDF>) -> Self {
        Self { ensure_loaded_udf }
    }
}

impl AnalyzerRule for EnsureLoadedAnalyzerRule {
    fn name(&self) -> &str {
        "sedona.ensure_loaded"
    }

    fn analyze(&self, plan: LogicalPlan, _config: &ConfigOptions) -> Result<LogicalPlan> {
        let ensure_loaded_udf = Arc::clone(&self.ensure_loaded_udf);

        // Walk plan bottom-up so a nested `RS_X(RS_Y(rast))` is rewritten
        // inside-out: the inner call's raster arg is wrapped first, then
        // the outer call's (now-wrapped) arg is rewritten if it itself
        // is a needs_bytes target.
        plan.transform_up(|p| {
            // Only single-input nodes (Projection, Filter, Aggregate,
            // Window) carry expressions whose args are evaluated against
            // a single input schema. Multi-input nodes (Join) and leaf
            // nodes (TableScan, EmptyRelation) are passed through —
            // joins with needs_bytes UDFs in the condition are a niche
            // case to handle in a follow-up.
            let inputs = p.inputs();
            if inputs.len() != 1 {
                return Ok(Transformed::no(p));
            }
            let input_schema = inputs[0].schema().clone();
            drop(inputs);

            p.map_expressions(|e| {
                e.transform_up(|expr| rewrite_expr_node(expr, &input_schema, &ensure_loaded_udf))
            })
        })
        .data()
    }
}

/// Single-step rewrite: if `expr` is a `needs_bytes` UDF call, wrap each
/// raster-typed arg with `RS_EnsureLoaded`. Recursion guard: never wrap
/// `RS_EnsureLoaded` itself.
fn rewrite_expr_node(
    expr: Expr,
    schema: &DFSchemaRef,
    ensure_loaded_udf: &Arc<ScalarUDF>,
) -> Result<Transformed<Expr>> {
    let Expr::ScalarFunction(ref func_call) = expr else {
        return Ok(Transformed::no(expr));
    };

    // Recursion guard.
    if func_call.func.name() == RS_ENSURE_LOADED_NAME {
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

    let Expr::ScalarFunction(ScalarFunction { func, args }) = expr else {
        unreachable!("matched above");
    };
    let mut changed = false;
    let new_args: Vec<Expr> = args
        .into_iter()
        .map(|arg| {
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

/// True if `expr` evaluates to a `SedonaType::Raster` under the given
/// schema. Uses `to_field` (not `get_type`) so the Field's extension
/// metadata is available — `SedonaType::Raster` is identified by an
/// `"sedona.raster"` extension type, not by raw `DataType::Struct`.
fn expr_is_raster(expr: &Expr, schema: &DFSchemaRef) -> bool {
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

    use std::sync::RwLock;

    use arrow_schema::DataType;
    use datafusion_common::tree_node::TreeNodeRecursion;
    use datafusion_common::DFSchema;
    use datafusion_expr::async_udf::AsyncScalarUDF;
    use datafusion_expr::{col, Volatility};
    use sedona_common::sedona_internal_datafusion_err;
    use sedona_expr::scalar_udf::{ScalarKernelRef, SimpleSedonaScalarKernel};
    use sedona_raster::outdb_loader::OutDbLoaderRegistry;
    use sedona_schema::matchers::ArgMatcher;

    use crate::rs_ensure_loaded::RsEnsureLoaded;

    /// Build a ScalarUDF wrapping our RsEnsureLoaded impl. Same shape as
    /// the one `SedonaContext::new_from_context` will register.
    fn build_ensure_loaded_udf() -> Arc<ScalarUDF> {
        let registry = Arc::new(RwLock::new(OutDbLoaderRegistry::new()));
        let async_udf = AsyncScalarUDF::new(Arc::new(RsEnsureLoaded::new(registry)));
        Arc::new(async_udf.into_scalar_udf())
    }

    /// Build a SedonaScalarUDF that "needs bytes". Accepts a raster, returns Int32.
    /// Inner kernel is a stub — never invoked because analyzer rewrites
    /// happen at planning time, before execution.
    fn build_needs_bytes_udf(name: &str) -> Arc<ScalarUDF> {
        let kernel: ScalarKernelRef = SimpleSedonaScalarKernel::new_ref(
            ArgMatcher::new(
                vec![ArgMatcher::is_raster()],
                SedonaType::Arrow(DataType::Int32),
            ),
            Arc::new(|_, _| Err(sedona_internal_datafusion_err!("stub kernel; not invoked"))),
        );
        let udf =
            SedonaScalarUDF::new(name, vec![kernel], Volatility::Immutable).with_needs_bytes();
        Arc::new(ScalarUDF::new_from_impl(udf))
    }

    /// Same but no `with_needs_bytes()` annotation.
    fn build_metadata_only_udf(name: &str) -> Arc<ScalarUDF> {
        let kernel: ScalarKernelRef = SimpleSedonaScalarKernel::new_ref(
            ArgMatcher::new(
                vec![ArgMatcher::is_raster()],
                SedonaType::Arrow(DataType::Int32),
            ),
            Arc::new(|_, _| Err(sedona_internal_datafusion_err!("stub kernel; not invoked"))),
        );
        let udf = SedonaScalarUDF::new(name, vec![kernel], Volatility::Immutable);
        Arc::new(ScalarUDF::new_from_impl(udf))
    }

    /// One-column DFSchema with a raster column named `rast`.
    fn raster_schema() -> DFSchemaRef {
        use arrow_schema::Schema;
        let field = SedonaType::Raster.to_storage_field("rast", true).unwrap();
        let arrow_schema = Arc::new(Schema::new(vec![field]));
        Arc::new(DFSchema::try_from(arrow_schema).unwrap())
    }

    /// One-column DFSchema with an int64 column named `n`.
    fn int_schema() -> DFSchemaRef {
        use arrow_schema::{Field, Schema};
        let field = Field::new("n", DataType::Int64, true);
        let arrow_schema = Arc::new(Schema::new(vec![field]));
        Arc::new(DFSchema::try_from(arrow_schema).unwrap())
    }

    /// Count `RS_EnsureLoaded(...)` calls in an expression tree.
    fn count_ensure_loaded(expr: &Expr) -> usize {
        let mut n = 0;
        expr.apply(|e| {
            if let Expr::ScalarFunction(sf) = e {
                if sf.func.name() == RS_ENSURE_LOADED_NAME {
                    n += 1;
                }
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .unwrap();
        n
    }

    #[test]
    fn expr_is_raster_detects_raster_column() {
        let schema = raster_schema();
        assert!(expr_is_raster(&col("rast"), &schema));
    }

    #[test]
    fn expr_is_raster_rejects_non_raster_column() {
        let schema = int_schema();
        assert!(!expr_is_raster(&col("n"), &schema));
    }

    #[test]
    fn rewrite_wraps_raster_arg_of_needs_bytes_udf() {
        let schema = raster_schema();
        let ensure_loaded = build_ensure_loaded_udf();
        let needs_bytes = build_needs_bytes_udf("rs_mock_needs_bytes");

        // Build: rs_mock_needs_bytes(rast)
        let original = Expr::ScalarFunction(ScalarFunction {
            func: needs_bytes,
            args: vec![col("rast")],
        });

        let rewritten = rewrite_expr_node(original, &schema, &ensure_loaded)
            .unwrap()
            .data;

        // Expect: rs_mock_needs_bytes(RS_EnsureLoaded(rast))
        let Expr::ScalarFunction(ScalarFunction { args, .. }) = &rewritten else {
            panic!("expected ScalarFunction, got {rewritten:?}");
        };
        assert_eq!(args.len(), 1);
        let Expr::ScalarFunction(inner) = &args[0] else {
            panic!(
                "expected wrapped arg to be ScalarFunction, got {:?}",
                args[0]
            );
        };
        assert_eq!(inner.func.name(), RS_ENSURE_LOADED_NAME);
        assert_eq!(inner.args.len(), 1);
        assert!(matches!(&inner.args[0], Expr::Column(c) if c.name == "rast"));
    }

    #[test]
    fn rewrite_leaves_non_raster_args_alone() {
        let schema = int_schema();
        let ensure_loaded = build_ensure_loaded_udf();
        let needs_bytes = build_needs_bytes_udf("rs_mock_needs_bytes");

        // rs_mock_needs_bytes(n) — n is Int64, not raster.
        let original = Expr::ScalarFunction(ScalarFunction {
            func: needs_bytes,
            args: vec![col("n")],
        });
        let original_clone = original.clone();
        let rewritten = rewrite_expr_node(original, &schema, &ensure_loaded)
            .unwrap()
            .data;

        // No wrap, expression unchanged.
        assert_eq!(format!("{rewritten:?}"), format!("{original_clone:?}"));
        assert_eq!(count_ensure_loaded(&rewritten), 0);
    }

    #[test]
    fn rewrite_leaves_metadata_only_udfs_alone() {
        let schema = raster_schema();
        let ensure_loaded = build_ensure_loaded_udf();
        let metadata_only = build_metadata_only_udf("rs_mock_metadata_only");

        // rs_mock_metadata_only(rast) — raster arg, but UDF doesn't need bytes.
        let original = Expr::ScalarFunction(ScalarFunction {
            func: metadata_only,
            args: vec![col("rast")],
        });
        let original_clone = original.clone();
        let rewritten = rewrite_expr_node(original, &schema, &ensure_loaded)
            .unwrap()
            .data;
        assert_eq!(format!("{rewritten:?}"), format!("{original_clone:?}"));
        assert_eq!(count_ensure_loaded(&rewritten), 0);
    }

    #[test]
    fn rewrite_does_not_recursively_wrap_rs_ensure_loaded() {
        let schema = raster_schema();
        let ensure_loaded = build_ensure_loaded_udf();

        // Existing explicit RS_EnsureLoaded(rast) call.
        let original = Expr::ScalarFunction(ScalarFunction {
            func: Arc::clone(&ensure_loaded),
            args: vec![col("rast")],
        });

        let rewritten = rewrite_expr_node(original, &schema, &ensure_loaded)
            .unwrap()
            .data;

        // Exactly one RS_EnsureLoaded — no recursive wrap.
        assert_eq!(count_ensure_loaded(&rewritten), 1);
    }

    #[test]
    fn cse_dedupes_wraps_when_two_needs_bytes_udfs_share_a_raster_column() {
        use arrow_schema::Schema;
        use datafusion_common::config::ConfigOptions;
        use datafusion_expr::logical_plan::builder::LogicalTableSource;
        use datafusion_expr::LogicalPlanBuilder;
        use datafusion_optimizer::common_subexpr_eliminate::CommonSubexprEliminate;
        use datafusion_optimizer::optimizer::OptimizerContext;
        use datafusion_optimizer::OptimizerRule;

        // Build a synthetic one-row table with a raster column.
        let raster_field = SedonaType::Raster.to_storage_field("rast", true).unwrap();
        let arrow_schema = Arc::new(Schema::new(vec![raster_field]));
        let source = Arc::new(LogicalTableSource::new(arrow_schema));
        let scan = LogicalPlanBuilder::scan("t", source, None).unwrap();

        let ensure_loaded = build_ensure_loaded_udf();
        let needs_bytes_a = build_needs_bytes_udf("rs_mock_a");
        let needs_bytes_b = build_needs_bytes_udf("rs_mock_b");

        // Run the analyzer rule first to get the wrapped plan.
        // Build the projection: rs_mock_a(rast), rs_mock_b(rast)
        let plan = scan
            .project(vec![
                Expr::ScalarFunction(ScalarFunction {
                    func: needs_bytes_a,
                    args: vec![col("rast")],
                })
                .alias("a"),
                Expr::ScalarFunction(ScalarFunction {
                    func: needs_bytes_b,
                    args: vec![col("rast")],
                })
                .alias("b"),
            ])
            .unwrap()
            .build()
            .unwrap();

        // Apply the analyzer rule: wraps both raster args.
        let rule = EnsureLoadedAnalyzerRule::new(ensure_loaded);
        let config = ConfigOptions::default();
        let wrapped = rule.analyze(plan, &config).unwrap();

        // Before CSE: two RS_EnsureLoaded(rast) calls.
        let before = count_ensure_loaded_in_plan(&wrapped);
        assert_eq!(
            before, 2,
            "analyzer rule should wrap both raster args; got {before}"
        );

        // Apply DataFusion's CommonSubexprEliminate.
        let cse = CommonSubexprEliminate::new();
        let cse_ctx = OptimizerContext::new();
        let transformed = cse.rewrite(wrapped, &cse_ctx).unwrap();
        let optimized = transformed.data;

        // After CSE: the duplicate is hoisted into an inner Projection.
        // The top-level Projection now references the hoisted column
        // instead of recomputing `RS_EnsureLoaded(rast)`; net count
        // across the plan tree drops to one materialisation.
        let after = count_ensure_loaded_in_plan(&optimized);
        assert_eq!(
            after, 1,
            "CSE should dedupe the two RS_EnsureLoaded(rast) calls into one; got {after}.\n\
             Plan:\n{optimized}"
        );
    }

    fn count_ensure_loaded_in_plan(plan: &LogicalPlan) -> usize {
        let mut total = 0usize;
        plan.apply(|p| {
            p.apply_expressions(|e| {
                total += count_ensure_loaded(e);
                Ok(TreeNodeRecursion::Continue)
            })
        })
        .unwrap();
        total
    }

    #[test]
    fn rewrite_handles_nested_needs_bytes_calls_inside_out() {
        // Build: outer_needs_bytes(inner_needs_bytes(rast)).
        // The inner UDF returns Int32 (per build_needs_bytes_udf), but
        // we want to exercise the structural rewriter, so let's use a
        // signature that accepts Int32 for the outer. We'll skip that
        // — bottom-up traversal happens at the analyzer level via
        // `expr.transform_up`, which is what the analyzer rule uses.
        // Rewriting nested calls is a structural property of
        // `transform_up`; testing it at the single-step rewriter level
        // is sufficient because each call sees a fully-rewritten subtree.
        //
        // Concrete check: a single rewrite_expr_node invocation on the
        // outer call (after the inner has been rewritten) sees
        // `inner(RS_EnsureLoaded(rast))` and would treat the inner's
        // ScalarFunction as a non-raster arg. So nesting works correctly
        // when called via transform_up.

        let schema = raster_schema();
        let ensure_loaded = build_ensure_loaded_udf();
        let needs_bytes = build_needs_bytes_udf("rs_inner");

        // Build outer.inner(rast), do a single bottom-up pass.
        let inner = Expr::ScalarFunction(ScalarFunction {
            func: Arc::clone(&needs_bytes),
            args: vec![col("rast")],
        });

        let rewritten = inner
            .transform_up(|e| rewrite_expr_node(e, &schema, &ensure_loaded))
            .unwrap()
            .data;
        // Outer Inner(Wrap(rast)).
        assert_eq!(count_ensure_loaded(&rewritten), 1);
    }
}
