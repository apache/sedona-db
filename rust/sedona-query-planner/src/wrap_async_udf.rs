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

//! DF-22662 workaround: logical optimizer rule that wraps async scalar UDF
//! calls with `sd_restore_metadata` to preserve field metadata.
//!
//! DataFusion's async scalar UDFs drop their `return_field` metadata at the
//! physical layer. This rule wraps every async UDF call with a sync
//! `sd_restore_metadata` UDF that re-stamps the metadata onto the output.
//!
//! **Removal:** once DataFusion #22662 is fixed and downstreamed, delete
//! this file and its `mod` declaration. `grep DF-22662` finds every site.

use std::sync::Arc;

use datafusion_common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion_common::{DFSchema, Result};
use datafusion_expr::async_udf::AsyncScalarUDF;
use datafusion_expr::expr::{Alias, ScalarFunction};
use datafusion_expr::expr_schema::ExprSchemable;
use datafusion_expr::{Expr, LogicalPlan};
use datafusion_optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};

use crate::restore_metadata::{restore_metadata_udf, RESTORE_METADATA_NAME};

/// Logical optimizer rule that wraps async scalar UDF calls with
/// `sd_restore_metadata` to preserve field metadata stripped at the
/// physical layer (DF-22662).
#[derive(Default, Debug)]
pub struct WrapAsyncUdfRule;

impl OptimizerRule for WrapAsyncUdfRule {
    fn name(&self) -> &str {
        "sedona.wrap_async_udf"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        // Bottom-up so nested UDFs are wrapped inside-out. Note: expression
        // rewriting uses transform_down to skip already-wrapped subtrees.
        Some(ApplyOrder::BottomUp)
    }

    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        // Type-check argument expressions against the merged schema of the
        // node's INPUTS (see ensure_loaded.rs for rationale).
        let inputs = plan.inputs();
        if inputs.is_empty() {
            return Ok(Transformed::no(plan));
        }
        let Some(schema) = merged_input_schema(&inputs) else {
            return Ok(Transformed::no(plan));
        };
        drop(inputs);

        plan.map_expressions(|e| {
            e.transform_down_up(skip_already_wrapped, |expr| wrap_async_udf(expr, &schema))
        })
    }
}

/// Merge the schemas of all inputs into one.
fn merged_input_schema(inputs: &[&LogicalPlan]) -> Option<Arc<DFSchema>> {
    let mut merged = inputs[0].schema().as_ref().clone();
    for input in &inputs[1..] {
        merged = merged.join(input.schema()).ok()?;
    }
    Some(Arc::new(merged))
}

/// Pre-order pass: skip children of `sd_restore_metadata` for idempotency.
fn skip_already_wrapped(expr: Expr) -> Result<Transformed<Expr>> {
    if let Expr::ScalarFunction(ref func_call) = expr {
        if func_call.func.name() == RESTORE_METADATA_NAME {
            // Already wrapped; skip children to avoid re-wrapping nested async UDFs.
            return Ok(Transformed::new(expr, false, TreeNodeRecursion::Jump));
        }
    }
    Ok(Transformed::no(expr))
}

/// Post-order pass: wrap async scalar UDF calls with `sd_restore_metadata`
/// to preserve the return field's metadata.
fn wrap_async_udf(expr: Expr, schema: &Arc<DFSchema>) -> Result<Transformed<Expr>> {
    let Expr::ScalarFunction(ref func_call) = expr else {
        return Ok(Transformed::no(expr));
    };

    // Skip sd_restore_metadata (shouldn't happen since pre-order skipped children,
    // but be defensive).
    if func_call.func.name() == RESTORE_METADATA_NAME {
        return Ok(Transformed::no(expr));
    }

    // Check if this is an async UDF by downcasting.
    let is_async = func_call
        .func
        .inner()
        .as_any()
        .downcast_ref::<AsyncScalarUDF>()
        .is_some();

    if !is_async {
        return Ok(Transformed::no(expr));
    }

    let (_, return_field) = expr.to_field(schema)?;
    let metadata = return_field.metadata().clone();
    if metadata.is_empty() {
        // No metadata to preserve; skip wrapping.
        return Ok(Transformed::no(expr));
    }

    // Get the original display name before wrapping so we can preserve it.
    let original_name = expr.schema_name().to_string();

    // Wrap the async UDF call with sd_restore_metadata, then alias it back
    // to the original name so the schema doesn't change.
    let restore_udf = restore_metadata_udf(metadata);
    let wrapped = Expr::ScalarFunction(ScalarFunction {
        func: restore_udf,
        args: vec![expr],
    });
    let aliased = Expr::Alias(Alias::new(wrapped, None::<&str>, original_name));

    Ok(Transformed::yes(aliased))
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::any::Any;
    use std::collections::HashMap;
    use std::hash::{Hash, Hasher};

    use arrow_schema::{DataType, Field, FieldRef, Schema};
    use async_trait::async_trait;
    use datafusion_expr::async_udf::{AsyncScalarUDF, AsyncScalarUDFImpl};
    use datafusion_expr::expr_schema::ExprSchemable;
    use datafusion_expr::{
        col, ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, Signature, Volatility,
    };

    /// A fake async UDF for testing.
    #[derive(Debug)]
    struct FakeAsyncUdf {
        signature: Signature,
        metadata: HashMap<String, String>,
    }

    impl FakeAsyncUdf {
        fn new(metadata: HashMap<String, String>) -> Self {
            Self {
                signature: Signature::any(1, Volatility::Stable),
                metadata,
            }
        }
    }

    impl PartialEq for FakeAsyncUdf {
        fn eq(&self, _other: &Self) -> bool {
            true
        }
    }
    impl Eq for FakeAsyncUdf {}
    impl Hash for FakeAsyncUdf {
        fn hash<H: Hasher>(&self, state: &mut H) {
            "fake_async".hash(state);
        }
    }

    impl datafusion_expr::ScalarUDFImpl for FakeAsyncUdf {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn name(&self) -> &str {
            "fake_async"
        }

        fn signature(&self) -> &Signature {
            &self.signature
        }

        fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
            Ok(DataType::Utf8)
        }

        fn return_field_from_args(&self, _args: ReturnFieldArgs) -> Result<FieldRef> {
            Ok(Arc::new(
                Field::new("output", DataType::Utf8, true).with_metadata(self.metadata.clone()),
            ))
        }

        fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
            unreachable!("async UDFs don't use invoke_with_args")
        }
    }

    #[async_trait]
    impl AsyncScalarUDFImpl for FakeAsyncUdf {
        async fn invoke_async_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
            unreachable!("test stub")
        }
    }

    fn fake_async_udf(metadata: HashMap<String, String>) -> Arc<ScalarUDF> {
        Arc::new(AsyncScalarUDF::new(Arc::new(FakeAsyncUdf::new(metadata))).into_scalar_udf())
    }

    #[test]
    fn wraps_async_udf_with_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "ARROW:extension:name".to_string(),
            "sedona.raster".to_string(),
        );

        let udf = fake_async_udf(metadata.clone());
        let input_schema = Arc::new(
            DFSchema::try_from(Schema::new(vec![Field::new("x", DataType::Utf8, true)])).unwrap(),
        );

        let expr = Expr::ScalarFunction(ScalarFunction {
            func: udf,
            args: vec![col("x")],
        });
        let original_name = expr.schema_name().to_string();

        let result = wrap_async_udf(expr, &input_schema).unwrap();
        assert!(result.transformed, "async UDF should be wrapped");

        // Verify the wrapper is an alias around sd_restore_metadata.
        let Expr::Alias(ref alias) = result.data else {
            panic!("expected Alias, got {:?}", result.data);
        };
        assert_eq!(
            alias.name, original_name,
            "alias should preserve original name"
        );
        let Expr::ScalarFunction(ref sf) = *alias.expr else {
            panic!("expected ScalarFunction inside alias");
        };
        assert_eq!(sf.func.name(), RESTORE_METADATA_NAME);

        // Verify the output field has the correct metadata.
        let (_, output_field) = result.data.to_field(&input_schema).unwrap();
        assert_eq!(
            output_field.metadata(),
            &metadata,
            "output field should have the restored metadata"
        );
    }

    #[test]
    fn skips_async_udf_without_metadata() {
        let udf = fake_async_udf(HashMap::new());
        let input_schema = Arc::new(
            DFSchema::try_from(Schema::new(vec![Field::new("x", DataType::Utf8, true)])).unwrap(),
        );

        let expr = Expr::ScalarFunction(ScalarFunction {
            func: udf,
            args: vec![col("x")],
        });

        let result = wrap_async_udf(expr, &input_schema).unwrap();
        assert!(
            !result.transformed,
            "async UDF without metadata should not be wrapped"
        );

        // Verify the output field has no metadata (since we didn't wrap).
        let (_, output_field) = result.data.to_field(&input_schema).unwrap();
        assert!(
            output_field.metadata().is_empty(),
            "output field should have no metadata"
        );
    }

    #[test]
    fn skips_already_wrapped() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "ARROW:extension:name".to_string(),
            "sedona.raster".to_string(),
        );

        let restore_udf = restore_metadata_udf(metadata.clone());

        let expr = Expr::ScalarFunction(ScalarFunction {
            func: restore_udf,
            args: vec![col("x")],
        });

        let result = skip_already_wrapped(expr).unwrap();
        assert!(!result.transformed, "already wrapped should be skipped");
        // Verify Jump is returned to skip children.
        assert_eq!(result.tnr, TreeNodeRecursion::Jump);
    }

    /// Verify that running the rewrite multiple times doesn't grow the nesting.
    /// This tests the full wrapped structure: Alias(sd_restore_metadata(async_udf(x)))
    #[test]
    fn idempotent_on_already_wrapped_async_udf() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "ARROW:extension:name".to_string(),
            "sedona.raster".to_string(),
        );

        let udf = fake_async_udf(metadata.clone());
        let input_schema = Arc::new(
            DFSchema::try_from(Schema::new(vec![Field::new("x", DataType::Utf8, true)])).unwrap(),
        );

        // Helper closure matching the optimizer's transform pattern.
        let rewrite = |e: Expr| -> Result<Transformed<Expr>> {
            e.transform_down_up(skip_already_wrapped, |expr| {
                wrap_async_udf(expr, &input_schema)
            })
        };

        // First pass: wrap the async UDF.
        let expr = Expr::ScalarFunction(ScalarFunction {
            func: udf,
            args: vec![col("x")],
        });
        let first_pass = rewrite(expr).unwrap();
        assert!(first_pass.transformed, "first pass should wrap");

        // Capture the structure after first pass.
        let first_pass_str = format!("{:?}", first_pass.data);

        // Second pass: should be a no-op.
        let second_pass = rewrite(first_pass.data).unwrap();

        let second_pass_str = format!("{:?}", second_pass.data);
        assert_eq!(
            first_pass_str, second_pass_str,
            "second pass should not modify the expression"
        );
    }

    /// Nested async UDFs should each be wrapped individually.
    #[test]
    fn wraps_nested_async_udfs() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "ARROW:extension:name".to_string(),
            "sedona.raster".to_string(),
        );

        let outer_udf = fake_async_udf(metadata.clone());
        let inner_udf = fake_async_udf(metadata.clone());
        let input_schema = Arc::new(
            DFSchema::try_from(Schema::new(vec![Field::new("x", DataType::Utf8, true)])).unwrap(),
        );

        // Construct: outer_async(inner_async(x))
        let inner_call = Expr::ScalarFunction(ScalarFunction {
            func: inner_udf,
            args: vec![col("x")],
        });
        let outer_call = Expr::ScalarFunction(ScalarFunction {
            func: outer_udf,
            args: vec![inner_call],
        });

        let result = outer_call
            .transform_down_up(skip_already_wrapped, |e| wrap_async_udf(e, &input_schema))
            .unwrap();
        assert!(result.transformed, "nested async UDFs should be wrapped");

        // Both inner and outer should be wrapped with sd_restore_metadata.
        // Structure: Alias(sd_restore_metadata(outer_async(Alias(sd_restore_metadata(inner_async(x))))))
        let result_str = format!("{:?}", result.data);
        let restore_count = result_str.matches("RestoreMetadata").count();
        assert_eq!(restore_count, 2, "both nested async UDFs should be wrapped");
    }
}
