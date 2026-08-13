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

//! Logical optimizer rule that normalises raster arguments of raster UDFs by
//! making their byte-access preconditions explicit in the logical plan instead
//! of as hidden side effects inside the kernel. Two preconditions are handled:
//!
//! - a UDF tagged `needs_pixels` has its raster arguments wrapped with
//!   `RS_EnsureLoaded`, materialising any OutDb bytes;
//! - a UDF tagged `needs_contiguous` has them wrapped with
//!   `RS_EnsureContiguous`, repacking any strided in-database band into a
//!   packed row-major layout the kernel can borrow zero-copy.
//!
//! A UDF that needs both (e.g. an export function that loads bytes and then
//! hands them to GDAL) yields the nested form
//! `RS_X(RS_EnsureContiguous(RS_EnsureLoaded(raster)), …)` — load innermost,
//! then repack the loaded result. So after this rule `RS_Value(raster, x, y)`
//! (tagged `needs_pixels`) becomes `RS_Value(RS_EnsureLoaded(raster), x, y)`,
//! and `RS_AsGeoTiff(raster)` (tagged both) becomes
//! `RS_AsGeoTiff(RS_EnsureContiguous(RS_EnsureLoaded(raster)))`. DataFusion's
//! `CommonSubexprEliminate` pass deduplicates identical wrapper calls across
//! multiple UDFs sharing the same raster column — provided both wrappers'
//! signatures are `Volatility::Stable` (not `Volatile`).
//!
//! This is a logical optimizer rule (not an analyzer rule) so it can look the
//! wrapper UDFs up from the [`FunctionRegistry`] rather than capturing an `Arc`
//! at construction time. Because optimizer rules run to a fixpoint, the rewrite
//! is idempotent: an argument already wrapped is left alone (see
//! [`already_loaded`] and [`already_contiguous`]).

use std::sync::Arc;

use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_common::{DFSchema, Result};
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

/// `SedonaScalarUDF` metadata key marking a UDF whose kernels require their
/// raster's band bytes laid out contiguously (they call `as_contiguous`,
/// directly or through the GDAL bridge). Raster arguments of such a UDF are
/// wrapped with `RS_EnsureContiguous`. Duplicated from `sedona_raster_functions`
/// (the owner), which this crate can't depend on — keep the literal in sync with
/// `sedona_raster_functions::rs_ensure_contiguous::NEEDS_CONTIGUOUS_METADATA_KEY`.
const NEEDS_CONTIGUOUS_METADATA_KEY: &str = "needs_contiguous";

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
        // Resolve both wrapper UDFs from the registry. A context that never
        // registered them (no raster support) has nothing to rewrite; they are
        // registered together, so if either is missing there is no wrapping to
        // do and we bail cleanly rather than half-normalise.
        let Some(registry) = config.function_registry() else {
            return Ok(Transformed::no(plan));
        };
        let Ok(ensure_loaded_udf) = registry.udf("rs_ensureloaded") else {
            return Ok(Transformed::no(plan));
        };
        let Ok(ensure_contiguous_udf) = registry.udf("rs_ensurecontiguous") else {
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
        drop(inputs);

        plan.map_expressions(|e| {
            e.transform_up(|expr| {
                rewrite_expr_node(expr, &schema, &ensure_loaded_udf, &ensure_contiguous_udf)
            })
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

/// Single-step rewrite: if `expr` is a UDF call tagged `needs_pixels` and/or
/// `needs_contiguous`, wrap each raster-typed arg with the corresponding
/// wrapper(s). Two guards keep it correct: it never wraps either wrapper UDF
/// itself (recursion), and it never re-adds a wrapper an arg already carries
/// (idempotency, required because optimizer rules run to a fixpoint).
fn rewrite_expr_node(
    expr: Expr,
    schema: &Arc<DFSchema>,
    ensure_loaded_udf: &Arc<ScalarUDF>,
    ensure_contiguous_udf: &Arc<ScalarUDF>,
) -> Result<Transformed<Expr>> {
    let Expr::ScalarFunction(ref func_call) = expr else {
        return Ok(Transformed::no(expr));
    };

    // Recursion guard: don't wrap either wrapper UDF itself.
    let name = func_call.func.name();
    if name == "rs_ensureloaded" || name == "rs_ensurecontiguous" {
        return Ok(Transformed::no(expr));
    }

    // Only annotated SedonaScalarUDFs participate. DataFusion built-ins and
    // unannotated UDFs pass through unchanged. Each flag read is its own
    // statement so the metadata borrow of `expr` (via `func_call`) is a
    // temporary that ends before `expr` is moved out below.
    let has_flag = |key: &str| {
        func_call
            .func
            .inner()
            .as_any()
            .downcast_ref::<SedonaScalarUDF>()
            .is_some_and(|u| u.metadata().get(key).map(String::as_str) == Some("true"))
    };
    let needs_pixels = has_flag(NEEDS_PIXELS_METADATA_KEY);
    let needs_contiguous = has_flag(NEEDS_CONTIGUOUS_METADATA_KEY);
    if !needs_pixels && !needs_contiguous {
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
            if !expr_is_raster(&arg, schema) {
                return arg;
            }
            // Add each wrapper only if the flag is set AND the arg doesn't
            // already carry it. `already_loaded`/`already_contiguous` look
            // through the other wrapper too, so a fixpoint re-run over a
            // fully-wrapped arg is a no-op.
            let add_load = needs_pixels && !already_loaded(&arg);
            let add_contiguous = needs_contiguous && !already_contiguous(&arg);
            if !add_load && !add_contiguous {
                return arg;
            }
            changed = true;
            wrap_raster_arg(
                arg,
                add_load,
                add_contiguous,
                ensure_loaded_udf,
                ensure_contiguous_udf,
            )
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

/// Wrap a raster argument so its byte-access preconditions are explicit in the
/// plan, then alias the result back to the argument's original name. With both
/// wrappers requested the nesting is
/// `rs_ensurecontiguous(rs_ensureloaded(arg))` — load innermost so the repack
/// operates on the materialised bytes.
///
/// The alias matters: an optimizer rule must not change the plan's output
/// schema, but rewriting `f(rast)` to `f(rs_ensureloaded(rast))` would change
/// the *derived name* of the enclosing expression — and when that expression
/// is a projection's output (e.g. `SELECT RS_DimToBand(rast, …)`), the column
/// is renamed from `rs_dimtoband(rast, …)` to `rs_dimtoband(rs_ensureloaded(rast), …)`.
/// DataFusion's `optimize_projections` invariant check then fails the query.
/// Aliasing back to the argument's original name keeps the enclosing name — and
/// the output schema — stable. (`WrapAsyncUdfRule` uses the same trick to
/// preserve its wrapper's name.)
///
/// `add_load` / `add_contiguous` are precomputed by the caller (already
/// accounting for the idempotency guards), so at least one is `true` here.
fn wrap_raster_arg(
    arg: Expr,
    add_load: bool,
    add_contiguous: bool,
    ensure_loaded_udf: &Arc<ScalarUDF>,
    ensure_contiguous_udf: &Arc<ScalarUDF>,
) -> Expr {
    let original_name = arg.schema_name().to_string();
    let mut wrapped = arg;
    if add_load {
        wrapped = Expr::ScalarFunction(ScalarFunction {
            func: Arc::clone(ensure_loaded_udf),
            args: vec![wrapped],
        });
    }
    if add_contiguous {
        wrapped = Expr::ScalarFunction(ScalarFunction {
            func: Arc::clone(ensure_contiguous_udf),
            args: vec![wrapped],
        });
    }
    Expr::Alias(Alias::new(wrapped, None::<&str>, original_name))
}

/// True if `expr` already yields loaded (in-database) raster bytes, so the
/// rule must not wrap it in `RS_EnsureLoaded`. Looks through aliases and the
/// `sd_restore_metadata(...)` wrapper that [`WrapAsyncUdfRule`] stamps onto
/// async calls between optimizer passes. Two reasons an argument is already
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
fn already_loaded(expr: &Expr) -> bool {
    match expr {
        Expr::ScalarFunction(sf) if sf.func.name() == "rs_ensureloaded" => true,
        // `rs_ensurecontiguous` only repacks already-in-database bytes — it
        // never un-loads — so `rs_ensurecontiguous(X)` is loaded iff `X` is.
        // This makes a fixpoint re-run over a fully-wrapped both-flags arg
        // (`rs_ensurecontiguous(rs_ensureloaded(rast))`) recognise it as
        // already loaded and skip re-adding the loader.
        Expr::ScalarFunction(sf) if sf.func.name() == "rs_ensurecontiguous" => {
            sf.args.first().is_some_and(already_loaded)
        }
        Expr::ScalarFunction(sf) if sf.func.name() == RESTORE_METADATA_NAME => {
            sf.args.first().is_some_and(already_loaded)
        }
        Expr::ScalarFunction(sf) => sf
            .func
            .inner()
            .as_any()
            .downcast_ref::<SedonaScalarUDF>()
            .is_some_and(|u| {
                u.metadata()
                    .get(RETURNS_BYTES_METADATA_KEY)
                    .map(String::as_str)
                    == Some("true")
            }),
        Expr::Alias(alias) => already_loaded(&alias.expr),
        _ => false,
    }
}

/// True if `expr` already yields contiguous raster bytes, so the rule must not
/// wrap it in `RS_EnsureContiguous`. An argument is already contiguous when it
/// is (or, through an alias, wraps) an injected `rs_ensurecontiguous` call —
/// the idempotency guard that stops a fixpoint re-run from stacking repackers.
/// `RS_EnsureContiguous` is matched by name because it is a plain UDF, not a
/// `SedonaScalarUDF` that could carry metadata.
///
/// Unlike [`already_loaded`], there is no `sd_restore_metadata` arm:
/// `RS_EnsureContiguous` is synchronous, so `WrapAsyncUdfRule` never stamps the
/// async-only restore wrapper onto it.
fn already_contiguous(expr: &Expr) -> bool {
    match expr {
        Expr::ScalarFunction(sf) if sf.func.name() == "rs_ensurecontiguous" => true,
        Expr::Alias(alias) => already_contiguous(&alias.expr),
        _ => false,
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

    use arrow_schema::{DataType, Field, Schema};
    use datafusion_common::tree_node::TreeNodeRecursion;
    use datafusion_expr::{col, ScalarUDF, Volatility};
    use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarUDF, SimpleSedonaScalarKernel};
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

    /// A stand-in `rs_ensurecontiguous` UDF. Like the loader stub, the rule
    /// keys off the name only, so a plain SedonaScalarUDF carrying the
    /// canonical name is sufficient (the real impl is a plain sync
    /// ScalarUDFImpl in the `sedona-raster-functions` crate).
    fn fake_ensure_contiguous_udf() -> Arc<ScalarUDF> {
        let kernel: ScalarKernelRef = SimpleSedonaScalarKernel::new_ref(
            ArgMatcher::new(vec![ArgMatcher::is_raster()], SedonaType::Raster),
            Arc::new(|_, _| unreachable!("stub kernel; rewrite never invokes it")),
        );
        let udf = SedonaScalarUDF::new("rs_ensurecontiguous", vec![kernel], Volatility::Immutable);
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

    /// A `needs_contiguous`-only UDF accepting a raster, returning Int32 —
    /// models a kernel that reads contiguous bytes but assumes loaded input.
    fn needs_contiguous_udf(name: &str) -> Arc<ScalarUDF> {
        let kernel: ScalarKernelRef = SimpleSedonaScalarKernel::new_ref(
            ArgMatcher::new(
                vec![ArgMatcher::is_raster()],
                SedonaType::Arrow(DataType::Int32),
            ),
            Arc::new(|_, _| unreachable!("stub kernel; not invoked at plan time")),
        );
        let udf = SedonaScalarUDF::new(name, vec![kernel], Volatility::Immutable)
            .with_metadata(NEEDS_CONTIGUOUS_METADATA_KEY, "true");
        Arc::new(ScalarUDF::new_from_impl(udf))
    }

    /// A UDF tagged BOTH `needs_pixels` and `needs_contiguous` — models a
    /// GDAL-bridge export like RS_AsGeoTiff that loads bytes then hands them to
    /// GDAL contiguously.
    fn needs_pixels_and_contiguous_udf(name: &str) -> Arc<ScalarUDF> {
        let kernel: ScalarKernelRef = SimpleSedonaScalarKernel::new_ref(
            ArgMatcher::new(
                vec![ArgMatcher::is_raster()],
                SedonaType::Arrow(DataType::Int32),
            ),
            Arc::new(|_, _| unreachable!("stub kernel; not invoked at plan time")),
        );
        let udf = SedonaScalarUDF::new(name, vec![kernel], Volatility::Immutable)
            .with_metadata(NEEDS_PIXELS_METADATA_KEY, "true")
            .with_metadata(NEEDS_CONTIGUOUS_METADATA_KEY, "true");
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

    fn count_named(expr: &Expr, name: &str) -> usize {
        let mut n = 0;
        expr.apply(|e| {
            if let Expr::ScalarFunction(sf) = e {
                if sf.func.name() == name {
                    n += 1;
                }
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .unwrap();
        n
    }

    fn count_ensure_loaded(expr: &Expr) -> usize {
        count_named(expr, "rs_ensureloaded")
    }

    fn count_ensure_contiguous(expr: &Expr) -> usize {
        count_named(expr, "rs_ensurecontiguous")
    }

    /// Drive `rewrite_expr_node` with the given loader stub and a fresh
    /// contiguous stub. Existing call sites that only exercise the loader pass
    /// just the loader; the contiguous wrapper is supplied here.
    fn rewrite(expr: Expr, schema: &Arc<DFSchema>, udf: &Arc<ScalarUDF>) -> Expr {
        let contig = fake_ensure_contiguous_udf();
        rewrite_expr_node(expr, schema, udf, &contig).unwrap().data
    }

    /// Test-only shim preserving the pre-nesting `wrap_for_loading(arg, udf)`
    /// call shape used by the idempotency tests: wrap with the loader only.
    fn wrap_for_loading(arg: Expr, ensure_loaded_udf: &Arc<ScalarUDF>) -> Expr {
        let contig = fake_ensure_contiguous_udf();
        wrap_raster_arg(arg, true, false, ensure_loaded_udf, &contig)
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
        assert!(already_loaded(&args[0]), "raster arg should be wrapped");
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
            already_loaded(&args[0]),
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
        use crate::restore_metadata::restore_metadata_udf;
        use std::collections::HashMap;

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
        use datafusion::execution::session_state::SessionStateBuilder;

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
        use datafusion::execution::session_state::SessionStateBuilder;
        use datafusion_expr::registry::FunctionRegistry;
        use datafusion_expr::{EmptyRelation, LogicalPlanBuilder};

        // SessionState doubles as the OptimizerConfig and carries the
        // function registry the rule resolves both wrapper UDFs from.
        let mut state = SessionStateBuilder::new().with_default_features().build();
        state.register_udf(fake_ensure_loaded_udf()).unwrap();
        state.register_udf(fake_ensure_contiguous_udf()).unwrap();

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
            already_loaded(&args[0]),
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
        let contig = fake_ensure_contiguous_udf();

        // Initial expression: rs_mock(rast)
        let call = Expr::ScalarFunction(ScalarFunction {
            func: needs_bytes_udf("rs_mock"),
            args: vec![col("rast")],
        });

        // First pass via transform_up (matching the optimizer's pattern).
        let first_pass = call
            .transform_up(|e| rewrite_expr_node(e, &schema, &udf, &contig))
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
            .transform_up(|e| rewrite_expr_node(e, &schema, &udf, &contig))
            .unwrap();
        assert_eq!(
            count_ensure_loaded(&second_pass.data),
            1,
            "second pass should not add more wrappers"
        );
    }

    #[test]
    fn wraps_raster_arg_of_needs_contiguous_udf() {
        // A needs_contiguous-only UDF: its raster arg is wrapped in
        // rs_ensurecontiguous (and nothing else — no loader).
        let schema = raster_schema_named("rast");
        let udf = fake_ensure_loaded_udf();
        let call = Expr::ScalarFunction(ScalarFunction {
            func: needs_contiguous_udf("rs_mock"),
            args: vec![col("rast")],
        });
        let out = rewrite(call, &schema, &udf);
        assert_eq!(count_ensure_contiguous(&out), 1, "{out:?}");
        assert_eq!(count_ensure_loaded(&out), 0, "{out:?}");
        let Expr::ScalarFunction(ScalarFunction { args, .. }) = &out else {
            panic!("expected ScalarFunction, got {out:?}");
        };
        assert!(
            already_contiguous(&args[0]),
            "raster arg should be wrapped in rs_ensurecontiguous"
        );
    }

    #[test]
    fn wraps_needs_pixels_and_contiguous_as_ensurecontiguous_of_ensureloaded() {
        // A UDF tagged BOTH flags yields the nested form
        // rs_ensurecontiguous(rs_ensureloaded(rast)): load innermost, repack
        // outermost.
        let schema = raster_schema_named("rast");
        let udf = fake_ensure_loaded_udf();
        let call = Expr::ScalarFunction(ScalarFunction {
            func: needs_pixels_and_contiguous_udf("rs_mock"),
            args: vec![col("rast")],
        });
        let out = rewrite(call, &schema, &udf);
        assert_eq!(count_ensure_loaded(&out), 1, "{out:?}");
        assert_eq!(count_ensure_contiguous(&out), 1, "{out:?}");

        // Assert the exact nesting: arg = rs_ensurecontiguous(rs_ensureloaded(rast)),
        // aliased back to the original name.
        let Expr::ScalarFunction(ScalarFunction { args, .. }) = &out else {
            panic!("expected outer ScalarFunction, got {out:?}");
        };
        let inner = match &args[0] {
            Expr::Alias(alias) => alias.expr.as_ref(),
            other => other,
        };
        let Expr::ScalarFunction(ScalarFunction {
            func: outer_func,
            args: outer_args,
        }) = inner
        else {
            panic!("expected rs_ensurecontiguous call, got {inner:?}");
        };
        assert_eq!(outer_func.name(), "rs_ensurecontiguous");
        let Expr::ScalarFunction(ScalarFunction {
            func: loader_func, ..
        }) = &outer_args[0]
        else {
            panic!(
                "expected rs_ensureloaded nested inside, got {:?}",
                outer_args[0]
            );
        };
        assert_eq!(
            loader_func.name(),
            "rs_ensureloaded",
            "loader must be nested INSIDE the contiguous wrapper"
        );
    }

    #[test]
    fn needs_contiguous_leaves_non_raster_args_alone() {
        let schema = int_schema("n");
        let udf = fake_ensure_loaded_udf();
        let call = Expr::ScalarFunction(ScalarFunction {
            func: needs_contiguous_udf("rs_mock"),
            args: vec![col("n")],
        });
        let out = rewrite(call, &schema, &udf);
        assert_eq!(count_ensure_contiguous(&out), 0);
        assert_eq!(count_ensure_loaded(&out), 0);
    }

    #[test]
    fn does_not_rewrap_already_contiguous_arg() {
        // An arg already wrapped in rs_ensurecontiguous is not wrapped again by
        // a needs_contiguous UDF (idempotency across fixpoint passes), and the
        // aliased form is likewise recognised.
        let schema = raster_schema_named("rast");
        let udf = fake_ensure_loaded_udf();
        let already = wrap_raster_arg(
            col("rast"),
            false,
            true,
            &udf,
            &fake_ensure_contiguous_udf(),
        );
        let call = Expr::ScalarFunction(ScalarFunction {
            func: needs_contiguous_udf("rs_mock"),
            args: vec![already],
        });
        let out = rewrite(call, &schema, &udf);
        assert_eq!(
            count_ensure_contiguous(&out),
            1,
            "already-contiguous arg must not be re-wrapped: {out:?}"
        );
    }

    #[test]
    fn both_flags_idempotent_over_fully_wrapped_arg() {
        // Fixpoint re-run: the arg is already rs_ensurecontiguous(rs_ensureloaded(rast))
        // and the UDF needs both. Neither wrapper is re-added — already_loaded
        // looks THROUGH rs_ensurecontiguous, and already_contiguous matches the
        // outer wrapper.
        let schema = raster_schema_named("rast");
        let udf = fake_ensure_loaded_udf();
        let fully_wrapped =
            wrap_raster_arg(col("rast"), true, true, &udf, &fake_ensure_contiguous_udf());
        let call = Expr::ScalarFunction(ScalarFunction {
            func: needs_pixels_and_contiguous_udf("rs_mock"),
            args: vec![fully_wrapped],
        });
        let out = rewrite(call, &schema, &udf);
        assert_eq!(count_ensure_loaded(&out), 1, "no extra loader: {out:?}");
        assert_eq!(
            count_ensure_contiguous(&out),
            1,
            "no extra repacker: {out:?}"
        );
    }

    #[test]
    fn contiguous_wrapping_preserves_enclosing_expression_name() {
        // As with the loader, wrapping must not rename the enclosing call, or
        // DataFusion's optimize_projections invariant check fails for an
        // unaliased projection.
        let schema = raster_schema_named("rast");
        let udf = fake_ensure_loaded_udf();
        let call = Expr::ScalarFunction(ScalarFunction {
            func: needs_pixels_and_contiguous_udf("rs_mock"),
            args: vec![col("rast")],
        });
        let original_name = call.schema_name().to_string();
        let out = rewrite(call, &schema, &udf);
        assert_eq!(
            out.schema_name().to_string(),
            original_name,
            "nested wrapping must not rename the enclosing expression: {out:?}"
        );
    }
}
