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
//! in `RS_EnsureLoaded` is left alone (see [`already_loaded`]), including
//! when the wrap has been hoisted into a projection below by DataFusion's
//! `CommonSubexprEliminate` and only a column reference remains (see
//! [`loaded_columns`]).

use std::collections::HashSet;
use std::sync::Arc;

use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_common::{Column, DFSchema, Result};
use datafusion_expr::expr::{Alias, ScalarFunction};
use datafusion_expr::expr_schema::ExprSchemable;
use datafusion_expr::{Expr, LogicalPlan, ScalarUDF};
use datafusion_optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};
use sedona_common::sedona_internal_err;
use sedona_expr::scalar_udf::SedonaScalarUDF;
use sedona_schema::datatypes::SedonaType;

use crate::restore_metadata::RESTORE_METADATA_NAME;

/// `SedonaScalarUDF` metadata key marking a UDF whose kernels read raster
/// pixel bytes. Duplicated from `sedona_raster_functions` (the owner),
/// which this crate can't depend on — keep the literal in sync with
/// `sedona_raster_functions::rs_ensure_loaded::NEEDS_PIXELS_METADATA_KEY`.
const NEEDS_PIXELS_METADATA_KEY: &str = "needs_pixels";

/// `SedonaScalarUDF` metadata key marking a UDF whose returned raster is
/// already fully materialised in-database. When a `needs_pixels` argument is
/// itself such a call, the rule skips wrapping it: its result is already
/// loaded, so an extra `RS_EnsureLoaded` would be redundant — and, being
/// async, would nest inside the argument's own async wrap, which DataFusion
/// cannot currently hoist (apache/datafusion#20031). Duplicated from
/// `sedona_raster_functions` — keep in sync with
/// `sedona_raster_functions::rs_ensure_loaded::RETURNS_BYTES_METADATA_KEY`.
const RETURNS_BYTES_METADATA_KEY: &str = "returns_bytes";

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

        // Type-check argument expressions against the merged schema of the
        // node's INPUTS, not the node's own (output) schema. For a
        // Projection the output schema holds the projected results
        // (`rs_value(rast, …)`), not the input `rast` column the argument
        // references, so `plan.schema()` would fail to recognise the raster
        // arg and silently skip wrapping. Single-input nodes (Projection,
        // Filter, …) use their one input; a Join's `filter` references
        // left ⋈ right, so the merged schema resolves either side. Leaf
        // nodes carry no wrappable expressions.
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
        // Columns the inputs already deliver as loaded rasters, so a bare
        // column reference to a hoisted `rs_ensureloaded(..)` isn't wrapped
        // again (see `loaded_columns`).
        let loaded: HashSet<Column> = inputs
            .iter()
            .flat_map(|input| loaded_columns(input))
            .collect();
        drop(inputs);

        plan.map_expressions(|e| {
            e.transform_up(|expr| rewrite_expr_node(expr, &schema, &ensure_loaded_udf, &loaded))
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
/// required because optimizer rules run to a fixpoint). `loaded` names the
/// input columns that already carry loaded bytes (see [`loaded_columns`]).
fn rewrite_expr_node(
    expr: Expr,
    schema: &Arc<DFSchema>,
    ensure_loaded_udf: &Arc<ScalarUDF>,
    loaded: &HashSet<Column>,
) -> Result<Transformed<Expr>> {
    let Expr::ScalarFunction(ref func_call) = expr else {
        return Ok(Transformed::no(expr));
    };

    // Recursion guard: don't wrap rs_ensureloaded itself.
    let name = func_call.func.name();
    if name == "rs_ensureloaded" {
        return Ok(Transformed::no(expr));
    }

    // Only annotated SedonaScalarUDFs participate. DataFusion built-ins
    // and unannotated UDFs pass through unchanged.
    let needs_bytes = func_call
        .func
        .inner()
        .downcast_ref::<SedonaScalarUDF>()
        .map(|u| {
            u.metadata()
                .get(NEEDS_PIXELS_METADATA_KEY)
                .map(String::as_str)
                == Some("true")
        })
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
            // Don't wrap an argument that already yields loaded bytes —
            // whether it's an injected loader (idempotency across fixpoint
            // passes) or a function that returns materialised bytes (avoids a
            // redundant nested async wrap; apache/datafusion#20031).
            if already_loaded(&arg, loaded) {
                return arg;
            }
            if expr_is_raster(&arg, schema) {
                changed = true;
                wrap_for_loading(arg, ensure_loaded_udf)
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

/// Wrap a raster argument so its byte materialisation is explicit in the
/// plan: `rs_ensureloaded(arg)`, aliased back to the argument's name.
///
/// The alias matters: an optimizer rule must not change the plan's output
/// schema, but rewriting `f(rast)` to `f(rs_ensureloaded(rast))` would change
/// the *derived name* of the enclosing expression — and when that expression
/// is a projection's output (e.g. `SELECT RS_DimToBand(rast, …)`), the column
/// is renamed from `rs_dimtoband(rast, …)` to `rs_dimtoband(rs_ensureloaded(rast), …)`.
/// DataFusion's `optimize_projections` invariant check then fails the query.
/// Aliasing the wrap back to the argument's original name keeps the enclosing
/// name — and the output schema — stable. (`WrapAsyncUdfRule` uses the same
/// trick to preserve its wrapper's name.)
fn wrap_for_loading(arg: Expr, ensure_loaded_udf: &Arc<ScalarUDF>) -> Expr {
    let original_name = arg.schema_name().to_string();
    let wrapped = Expr::ScalarFunction(ScalarFunction {
        func: Arc::clone(ensure_loaded_udf),
        args: vec![arg],
    });
    Expr::Alias(Alias::new(wrapped, None::<&str>, original_name))
}

/// True if `expr` already yields loaded (in-database) raster bytes, so the
/// rule must not wrap it in `RS_EnsureLoaded`. Looks through aliases and the
/// `sd_restore_metadata(...)` wrapper that [`WrapAsyncUdfRule`] stamps onto
/// async calls between optimizer passes. Three reasons an argument is already
/// loaded, unified here:
///
/// - it is (or wraps) an injected `rs_ensureloaded` call — the idempotency
///   guard that stops a fixpoint re-run from stacking loaders into a tower of
///   unresolved async calls. `RS_EnsureLoaded` is matched by name because it
///   is a plain async UDF, not a `SedonaScalarUDF` that could carry metadata.
/// - it is a call to a function tagged [`RETURNS_BYTES_METADATA_KEY`] (e.g.
///   `RS_DimToBand`), whose output is already materialised. Wrapping it would
///   inject a redundant async `rs_ensureloaded` that nests inside the
///   argument's own async wrap and can't be hoisted (apache/datafusion#20031).
/// - it is a reference to one of the `loaded` columns: a column the input
///   plan computes from an expression of the two shapes above (see
///   [`loaded_columns`]). The shape itself is out of reach once DataFusion
///   hoists the loader into a projection below, so the column is all there
///   is to go on.
fn already_loaded(expr: &Expr, loaded: &HashSet<Column>) -> bool {
    match expr {
        Expr::Column(column) => loaded.contains(column),
        Expr::ScalarFunction(sf) if sf.func.name() == "rs_ensureloaded" => true,
        Expr::ScalarFunction(sf) if sf.func.name() == RESTORE_METADATA_NAME => sf
            .args
            .first()
            .is_some_and(|arg| already_loaded(arg, loaded)),
        Expr::ScalarFunction(sf) => sf
            .func
            .inner()
            .downcast_ref::<SedonaScalarUDF>()
            .is_some_and(|u| {
                u.metadata()
                    .get(RETURNS_BYTES_METADATA_KEY)
                    .map(String::as_str)
                    == Some("true")
            }),
        Expr::Alias(alias) => already_loaded(&alias.expr, loaded),
        _ => false,
    }
}

/// The output columns of `plan` that already carry loaded raster bytes.
///
/// [`already_loaded`] recognises a loaded argument by its shape, and two
/// DataFusion rules replace that shape with a plain column reference between
/// optimizer passes:
///
/// - `CommonSubexprEliminate`: when several `needs_pixels` calls share a raster
///   argument, the shared `sd_restore_metadata(rs_ensureloaded(rast))` is
///   hoisted into a projection below and each call is left with a bare
///   `__common_expr_N` column.
/// - `OptimizeProjections`: a user-written `RS_EnsureLoaded(rast) AS rast` in a
///   subquery is a projection whose output column the outer call references.
///
/// In both cases the column is raster-typed and, by shape, not loaded, so
/// without this the next pass wraps it again. The extra loader is hoisted or
/// merged the same way, and every pass stacks one more:
/// `rs_ensureloaded(sd_restore_metadata(rs_ensureloaded(..)))`. DataFusion's
/// physical planner can't hoist nested async calls (apache/datafusion#20031):
/// only the innermost lands in `AsyncFuncExec`, the outer one is invoked
/// synchronously, and the query fails with "async functions should not be
/// called directly".
///
/// So resolve columns through the plan that produces them: a projection's
/// output column is loaded when the expression it assigns is loaded (with the
/// projection's own column references resolved the same way). Nodes that pass
/// their input's columns through unchanged (`SubqueryAlias`, `Filter`, `Sort`,
/// `Limit`) forward the input's set, re-qualified for an alias. Anything else
/// contributes nothing, which is the conservative answer (wrap).
fn loaded_columns(plan: &LogicalPlan) -> HashSet<Column> {
    match plan {
        LogicalPlan::Projection(projection) => {
            let inner = loaded_columns(&projection.input);
            projection
                .expr
                .iter()
                .zip(projection.schema.iter())
                .filter(|(expr, _)| already_loaded(expr, &inner))
                .map(|(_, (qualifier, field))| Column::new(qualifier.cloned(), field.name()))
                .collect()
        }
        LogicalPlan::SubqueryAlias(alias) => loaded_columns(&alias.input)
            .into_iter()
            .map(|column| Column::new(Some(alias.alias.clone()), column.name))
            .collect(),
        LogicalPlan::Filter(filter) => loaded_columns(&filter.input),
        LogicalPlan::Sort(sort) => loaded_columns(&sort.input),
        LogicalPlan::Limit(limit) => loaded_columns(&limit.input),
        _ => HashSet::new(),
    }
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

    use std::collections::HashMap;
    use std::hash::{Hash, Hasher};

    use arrow_schema::{DataType, Field, FieldRef, Schema};
    use async_trait::async_trait;
    use datafusion::execution::session_state::SessionStateBuilder;
    use datafusion_common::tree_node::TreeNodeRecursion;
    use datafusion_expr::async_udf::{AsyncScalarUDF, AsyncScalarUDFImpl};
    use datafusion_expr::registry::FunctionRegistry;
    use datafusion_expr::{
        ColumnarValue, EmptyRelation, LogicalPlanBuilder, ReturnFieldArgs, ScalarFunctionArgs,
        ScalarUDF, ScalarUDFImpl, Signature, Volatility, col,
    };
    use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarUDF, SimpleSedonaScalarKernel};
    use sedona_schema::matchers::ArgMatcher;

    use crate::restore_metadata::restore_metadata_udf;

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

    /// A raster-returning UDF that both reads pixels (`needs_bytes`) and
    /// promises loaded output (`returns_bytes`) — like RS_DimToBand / RS_Slice.
    fn returns_bytes_udf(name: &str) -> Arc<ScalarUDF> {
        let kernel: ScalarKernelRef = SimpleSedonaScalarKernel::new_ref(
            ArgMatcher::new(vec![ArgMatcher::is_raster()], SedonaType::Raster),
            Arc::new(|_, _| unreachable!("stub kernel; not invoked at plan time")),
        );
        let udf = SedonaScalarUDF::new(name, vec![kernel], Volatility::Immutable)
            .with_metadata(NEEDS_PIXELS_METADATA_KEY, "true")
            .with_metadata(RETURNS_BYTES_METADATA_KEY, "true");
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
            if let Expr::ScalarFunction(sf) = e
                && sf.func.name() == "rs_ensureloaded"
            {
                n += 1;
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .unwrap();
        n
    }

    fn rewrite(expr: Expr, schema: &Arc<DFSchema>, udf: &Arc<ScalarUDF>) -> Expr {
        rewrite_expr_node(expr, schema, udf, &HashSet::new())
            .unwrap()
            .data
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
            already_loaded(&args[0], &HashSet::new()),
            "raster arg should be wrapped"
        );
    }

    #[test]
    fn does_not_wrap_arg_that_returns_loaded_bytes() {
        // Models RS_BandToDim(RS_DimToBand(rast)): the inner call already
        // returns loaded bytes, so the outer needs_bytes call must NOT wrap it.
        // Otherwise a redundant async rs_ensureloaded nests inside the inner
        // call's own async wrap and DataFusion can't hoist it.
        let schema = raster_schema_named("rast");
        let udf = fake_ensure_loaded_udf();
        let inner = Expr::ScalarFunction(ScalarFunction {
            func: returns_bytes_udf("rs_inner"),
            args: vec![col("rast")],
        });
        let outer = Expr::ScalarFunction(ScalarFunction {
            func: needs_bytes_udf("rs_outer"),
            args: vec![inner],
        });
        let out = rewrite(outer, &schema, &udf);
        assert_eq!(
            count_ensure_loaded(&out),
            0,
            "an argument that already returns loaded bytes must not be wrapped: {out:?}"
        );
    }

    #[test]
    fn wrapping_preserves_enclosing_expression_name() {
        // An optimizer rule must not change the plan's output schema. Wrapping
        // the raster arg must leave the enclosing call's derived name unchanged,
        // or DataFusion's optimize_projections invariant check fails for an
        // unaliased projection such as `SELECT rs_mock(rast)`.
        let schema = raster_schema_named("rast");
        let udf = fake_ensure_loaded_udf();
        let call = Expr::ScalarFunction(ScalarFunction {
            func: needs_bytes_udf("rs_mock"),
            args: vec![col("rast")],
        });
        let original_name = call.schema_name().to_string();

        let out = rewrite(call, &schema, &udf);

        assert_eq!(
            out.schema_name().to_string(),
            original_name,
            "wrapping the raster arg must not rename the enclosing expression: {out:?}"
        );
        // The arg is still recognised as wrapped, so the fixpoint stays idempotent
        // through the name-preserving alias.
        let Expr::ScalarFunction(ScalarFunction { args, .. }) = &out else {
            panic!("expected ScalarFunction, got {out:?}");
        };
        assert!(
            already_loaded(&args[0], &HashSet::new()),
            "wrapped arg should still be detected"
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
        // Models the fixpoint re-run: the input already has the wrapped form
        // rs_mock(rs_ensureloaded(rast)). A second pass must not wrap it again.
        let schema = raster_schema_named("rast");
        let udf = fake_ensure_loaded_udf();
        let already_wrapped = wrap_for_loading(col("rast"), &udf);
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

        // Same scenario but the wrapped expr is aliased:
        // rs_mock(rs_ensureloaded(rast) AS loaded) should also not rewrap.
        let already_wrapped_aliased = wrap_for_loading(col("rast"), &udf).alias("loaded");
        let call_aliased = Expr::ScalarFunction(ScalarFunction {
            func: needs_bytes_udf("rs_mock"),
            args: vec![already_wrapped_aliased],
        });
        let out_aliased = rewrite(call_aliased, &schema, &udf);
        assert_eq!(
            count_ensure_loaded(&out_aliased),
            1,
            "aliased already-wrapped arg must not be wrapped again: {out_aliased:?}"
        );
    }

    #[test]
    fn idempotent_through_restore_metadata_wrapper() {
        // Models the cross-rule fixpoint: after the first pass wraps the arg,
        // WrapAsyncUdfRule re-stamps it as `sd_restore_metadata(rs_ensureloaded(rast))`
        // (aliased back to the original name). A later pass must recognise this
        // as already-loaded and not inject another wrapper — otherwise the
        // async calls tower up and only the innermost is ever extracted, so the
        // rest are invoked synchronously and the query fails at runtime.
        let schema = raster_schema_named("rast");
        let udf = fake_ensure_loaded_udf();

        let restore = restore_metadata_udf(HashMap::new());
        let wrapped = Expr::ScalarFunction(ScalarFunction {
            func: restore,
            args: vec![wrap_for_loading(col("rast"), &udf)],
        })
        .alias("rs_ensureloaded(rast)");

        let call = Expr::ScalarFunction(ScalarFunction {
            func: needs_bytes_udf("rs_mock"),
            args: vec![wrapped],
        });
        let out = rewrite(call, &schema, &udf);
        assert_eq!(
            count_ensure_loaded(&out),
            1,
            "arg already wrapped as sd_restore_metadata(rs_ensureloaded(..)) must not be re-wrapped: {out:?}"
        );
    }

    #[test]
    fn registers_before_cse_with_wrap_async_between() {
        use crate::optimizer::register_ensure_loaded_optimizer;

        let builder = SessionStateBuilder::new().with_default_features();
        let mut builder = register_ensure_loaded_optimizer(builder).unwrap();

        let rules = &builder.optimizer().as_ref().unwrap().rules;
        let ensure_loaded = rules
            .iter()
            .position(|r| r.name() == "sedona.ensure_loaded")
            .expect("ensure_loaded rule registered");
        let wrap_async = rules
            .iter()
            .position(|r| r.name() == "sedona.wrap_async_udf")
            .expect("wrap_async_udf rule registered");
        let cse = rules
            .iter()
            .position(|r| r.name() == "common_sub_expression_eliminate")
            .expect("CSE present in default optimizer");

        // Order: ensure_loaded -> wrap_async_udf -> CSE
        assert_eq!(
            ensure_loaded + 1,
            wrap_async,
            "wrap_async_udf must follow ensure_loaded"
        );
        assert_eq!(
            wrap_async + 1,
            cse,
            "CSE must follow wrap_async_udf so metadata wrappers dedupe in the same pass"
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

    #[test]
    fn rule_wraps_raster_arg_through_a_projection() {
        // Drives the real `OptimizerRule::rewrite()` (not the
        // `rewrite_expr_node` helper) on a Projection — `SELECT rs_mock(rast)`.
        // The projection's OUTPUT schema holds the result column, not the
        // input `rast`, so the rule must type-check against the INPUT schema
        // to recognise and wrap the raster arg. A regression guard against
        // switching to `plan.schema()`, which would silently skip wrapping
        // here (the common single-projection case).
        // SessionState doubles as the OptimizerConfig and carries the
        // function registry the rule resolves `rs_ensureloaded` from.
        let mut state = SessionStateBuilder::new().with_default_features().build();
        state.register_udf(fake_ensure_loaded_udf()).unwrap();

        let scan = LogicalPlan::EmptyRelation(EmptyRelation {
            produce_one_row: false,
            schema: raster_schema_named("rast"),
        });
        let proj = Expr::ScalarFunction(ScalarFunction {
            func: needs_bytes_udf("rs_mock"),
            args: vec![col("rast")],
        });
        let plan = LogicalPlanBuilder::from(scan)
            .project(vec![proj])
            .unwrap()
            .build()
            .unwrap();

        let out = EnsureLoadedOptimizerRule.rewrite(plan, &state).unwrap();

        let wrapped: usize = out.data.expressions().iter().map(count_ensure_loaded).sum();
        assert_eq!(
            wrapped, 1,
            "projection's raster arg should be wrapped via the input schema: {:?}",
            out.data
        );

        // Confirm the wrapped arg is rs_ensureloaded.
        let Expr::ScalarFunction(ScalarFunction { args, .. }) = &out.data.expressions()[0] else {
            panic!("expected the projected expr to be a ScalarFunction");
        };
        assert!(
            already_loaded(&args[0], &HashSet::new()),
            "wrapped arg should be rs_ensureloaded: {:?}",
            args[0]
        );
    }

    /// Verify that running `transform_up` multiple times doesn't grow nesting.
    /// This tests the full transform pattern used by the optimizer rule.
    #[test]
    fn idempotent_with_transform_up() {
        let schema = raster_schema_named("rast");
        let udf = fake_ensure_loaded_udf();

        // Initial expression: rs_mock(rast)
        let call = Expr::ScalarFunction(ScalarFunction {
            func: needs_bytes_udf("rs_mock"),
            args: vec![col("rast")],
        });

        // First pass via transform_up (matching the optimizer's pattern).
        let first_pass = call
            .transform_up(|e| rewrite_expr_node(e, &schema, &udf, &HashSet::new()))
            .unwrap();
        assert!(first_pass.transformed, "first pass should wrap");
        assert_eq!(
            count_ensure_loaded(&first_pass.data),
            1,
            "should have exactly one wrapper after first pass"
        );

        // Second pass: should be a no-op.
        let second_pass = first_pass
            .data
            .transform_up(|e| rewrite_expr_node(e, &schema, &udf, &HashSet::new()))
            .unwrap();
        assert_eq!(
            count_ensure_loaded(&second_pass.data),
            1,
            "second pass should not add more wrappers"
        );
    }

    /// An `AsyncScalarUDF` stand-in for the real `rs_ensureloaded`, so the
    /// whole optimizer runs on it as in a `SedonaContext`: this rule wraps,
    /// `WrapAsyncUdfRule` stamps `sd_restore_metadata`, CSE hoists, and
    /// `OptimizeProjections` merges. Returns its argument's (raster) field.
    #[derive(Debug)]
    struct FakeAsyncEnsureLoaded {
        signature: Signature,
    }

    impl PartialEq for FakeAsyncEnsureLoaded {
        fn eq(&self, _other: &Self) -> bool {
            true
        }
    }
    impl Eq for FakeAsyncEnsureLoaded {}
    impl Hash for FakeAsyncEnsureLoaded {
        fn hash<H: Hasher>(&self, state: &mut H) {
            "rs_ensureloaded".hash(state);
        }
    }

    impl ScalarUDFImpl for FakeAsyncEnsureLoaded {
        fn name(&self) -> &str {
            "rs_ensureloaded"
        }

        fn signature(&self) -> &Signature {
            &self.signature
        }

        fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
            Ok(SedonaType::Raster
                .to_storage_field("", true)?
                .data_type()
                .clone())
        }

        fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
            Ok(Arc::clone(&args.arg_fields[0]))
        }

        fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
            unreachable!("stub; never executed")
        }
    }

    #[async_trait]
    impl AsyncScalarUDFImpl for FakeAsyncEnsureLoaded {
        async fn invoke_async_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
            unreachable!("stub; never executed")
        }
    }

    fn fake_async_ensure_loaded_udf() -> ScalarUDF {
        AsyncScalarUDF::new(Arc::new(FakeAsyncEnsureLoaded {
            signature: Signature::any(1, Volatility::Stable),
        }))
        .into_scalar_udf()
    }

    /// `(total rs_ensureloaded calls in the plan, how many of them have another
    /// rs_ensureloaded somewhere in their arguments)`.
    fn loader_stats(plan: &LogicalPlan) -> (usize, usize) {
        let (mut total, mut nested) = (0, 0);
        plan.apply(|node| {
            node.apply_expressions(|expr| {
                expr.apply(|e| {
                    if let Expr::ScalarFunction(sf) = e
                        && sf.func.name() == "rs_ensureloaded"
                    {
                        total += 1;
                        if sf.args.iter().any(|arg| count_ensure_loaded(arg) > 0) {
                            nested += 1;
                        }
                    }
                    Ok(TreeNodeRecursion::Continue)
                })
            })
        })
        .unwrap();
        (total, nested)
    }

    fn state_with_fake_ensure_loaded() -> datafusion::execution::session_state::SessionState {
        let mut state = SessionStateBuilder::new().with_default_features().build();
        state.register_udf(fake_ensure_loaded_udf()).unwrap();
        state
    }

    fn raster_scan(name: &str) -> LogicalPlan {
        LogicalPlan::EmptyRelation(EmptyRelation {
            produce_one_row: false,
            schema: raster_schema_named(name),
        })
    }

    #[test]
    fn does_not_rewrap_column_hoisted_by_cse() {
        // Models the optimizer's second pass over
        //   SELECT rs_mock_a(rast), rs_mock_b(rast) FROM t
        // after the first pass wrapped both raster args and DataFusion's
        // CommonSubexprEliminate hoisted the shared loader into a projection
        // below, leaving each call a bare `__common_expr_1` column:
        //   Projection: rs_mock_a(__common_expr_1 AS rast), rs_mock_b(__common_expr_1 AS rast)
        //     Projection: sd_restore_metadata(rs_ensureloaded(rast)) AS __common_expr_1
        // The column is raster-typed; only its producer says it is loaded.
        let state = state_with_fake_ensure_loaded();
        let udf = fake_ensure_loaded_udf();

        let raster_metadata = SedonaType::Raster
            .to_storage_field("rast", true)
            .unwrap()
            .metadata()
            .clone();
        let hoisted = Expr::ScalarFunction(ScalarFunction {
            func: restore_metadata_udf(raster_metadata),
            args: vec![wrap_for_loading(col("rast"), &udf)],
        })
        .alias("__common_expr_1");
        let consumer = |name: &str| {
            Expr::ScalarFunction(ScalarFunction {
                func: needs_bytes_udf(name),
                args: vec![col("__common_expr_1").alias("rast")],
            })
        };
        let plan = LogicalPlanBuilder::from(raster_scan("rast"))
            .project(vec![hoisted])
            .unwrap()
            .project(vec![consumer("rs_mock_a"), consumer("rs_mock_b")])
            .unwrap()
            .build()
            .unwrap();

        let out = EnsureLoadedOptimizerRule.rewrite(plan, &state).unwrap();
        assert!(
            !out.transformed,
            "a column CSE hoisted the loader into must count as loaded, got: {}",
            out.data.display_indent()
        );
        assert_eq!(loader_stats(&out.data), (1, 0));
    }

    #[test]
    fn does_not_rewrap_column_from_user_ensure_loaded_projection() {
        // A user-written loader in a subquery, the outer call referencing its
        // output column through the subquery alias:
        //   SELECT rs_mock(sub.rast) FROM (SELECT rs_ensureloaded(rast) AS rast FROM t) sub
        // Wrapping `sub.rast` again would nest the loaders once
        // OptimizeProjections merges the single-use projection into its parent.
        let state = state_with_fake_ensure_loaded();
        let udf = fake_ensure_loaded_udf();

        let user_loaded = Expr::ScalarFunction(ScalarFunction {
            func: Arc::clone(&udf),
            args: vec![col("rast")],
        })
        .alias("rast");
        let plan = LogicalPlanBuilder::from(raster_scan("rast"))
            .project(vec![user_loaded])
            .unwrap()
            .alias("sub")
            .unwrap()
            .project(vec![Expr::ScalarFunction(ScalarFunction {
                func: needs_bytes_udf("rs_mock"),
                args: vec![col("sub.rast")],
            })])
            .unwrap()
            .build()
            .unwrap();

        let out = EnsureLoadedOptimizerRule.rewrite(plan, &state).unwrap();
        assert!(
            !out.transformed,
            "a column produced by rs_ensureloaded must count as loaded, got: {}",
            out.data.display_indent()
        );
        assert_eq!(loader_stats(&out.data), (1, 0));
    }

    #[test]
    fn loaded_columns_flow_through_schema_preserving_nodes() {
        let udf = fake_ensure_loaded_udf();
        let loaded = Expr::ScalarFunction(ScalarFunction {
            func: Arc::clone(&udf),
            args: vec![col("rast")],
        })
        .alias("loaded");
        let plan = LogicalPlanBuilder::from(raster_scan("rast"))
            .project(vec![loaded, col("rast")])
            .unwrap()
            .filter(col("rast").is_not_null())
            .unwrap()
            .sort(vec![col("rast").sort(true, true)])
            .unwrap()
            .limit(0, Some(10))
            .unwrap()
            .alias("sub")
            .unwrap()
            .build()
            .unwrap();

        let columns = loaded_columns(&plan);
        assert!(
            columns.contains(&Column::new(Some("sub"), "loaded")),
            "{columns:?}"
        );
        assert!(
            !columns.contains(&Column::new(Some("sub"), "rast")),
            "{columns:?}"
        );
        assert_eq!(columns.len(), 1);
    }

    /// The reported shape, through the whole optimizer to its fixpoint: two
    /// `needs_pixels` calls sharing a raster column over a scan. Each pass used
    /// to add a loader around the column CSE had hoisted the previous one
    /// into, ending with `rs_ensureloaded(sd_restore_metadata(rs_ensureloaded(..)))`
    /// that the physical planner can't hoist (apache/datafusion#20031), so the
    /// outer call ran synchronously and the query failed with "async functions
    /// should not be called directly".
    #[tokio::test]
    async fn optimizer_fixpoint_keeps_one_loader_for_calls_sharing_a_raster() {
        use crate::optimizer::register_ensure_loaded_optimizer;
        use datafusion::catalog::MemTable;
        use datafusion::physical_plan::displayable;
        use datafusion::prelude::SessionContext;

        let builder =
            register_ensure_loaded_optimizer(SessionStateBuilder::new().with_default_features())
                .unwrap();
        let ctx = SessionContext::new_with_state(builder.build());
        ctx.register_udf(fake_async_ensure_loaded_udf());
        ctx.register_udf(ScalarUDF::clone(&needs_bytes_udf("rs_mock_a")));
        ctx.register_udf(ScalarUDF::clone(&needs_bytes_udf("rs_mock_b")));
        let schema = Arc::new(Schema::new(vec![
            SedonaType::Raster.to_storage_field("rast", true).unwrap(),
        ]));
        let table = MemTable::try_new(schema, vec![vec![]]).unwrap();
        ctx.register_table("t", Arc::new(table)).unwrap();

        let df = ctx
            .sql("SELECT rs_mock_a(rast) AS a, rs_mock_b(rast) AS b FROM t")
            .await
            .unwrap();

        let optimized = df.clone().into_optimized_plan().unwrap();
        assert_eq!(
            loader_stats(&optimized),
            (1, 0),
            "expected one shared, un-nested loader, got: {}",
            optimized.display_indent()
        );

        // The one loader is the AsyncFuncExec's; nothing calls it synchronously.
        let physical = df.create_physical_plan().await.unwrap();
        let physical_str = displayable(physical.as_ref()).indent(true).to_string();
        assert!(physical_str.contains("AsyncFuncExec"), "{physical_str}");
        assert_eq!(
            physical_str.matches("rs_ensureloaded").count(),
            1,
            "{physical_str}"
        );
    }
}
