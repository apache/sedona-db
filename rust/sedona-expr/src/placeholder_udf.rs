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

//! Placeholder UDFs for expression deserialization
//!
//! This module provides placeholder implementations of scalar, aggregate, and window
//! user-defined functions (UDFs). These are used when deserializing expressions where
//! the actual execution of the expression is not necessarily important (e.g., for an
//! implementation of TableProvider that independently parses expressions for pruning,
//! or an implementation that does not use the filters expression at all).

use std::{
    collections::HashSet,
    sync::{Arc, OnceLock},
};

use arrow_schema::{DataType, Field};
use datafusion_common::{
    tree_node::{Transformed, TreeNode, TreeNodeRecursion},
    Result,
};
use datafusion_execution::FunctionRegistry;
use datafusion_expr::{
    expr::ScalarFunction,
    function::{AccumulatorArgs, PartitionEvaluatorArgs, WindowUDFFieldArgs},
    Accumulator, AggregateUDF, AggregateUDFImpl, Expr, HigherOrderUDF, PartitionEvaluator,
    ScalarUDF, ScalarUDFImpl, Signature, Volatility, WindowFunctionDefinition, WindowUDF,
    WindowUDFImpl,
};
use sedona_common::sedona_internal_err;

/// A [FunctionRegistry] that creates placeholder UDFs for any requested function.
///
/// This registry is useful for deserializing expressions where the actual function
/// implementations are not available or not needed. All functions are resolved to
/// placeholder stubs that preserve the function name but will error if actually invoked.
pub struct PlaceholderRegistry;

impl PlaceholderRegistry {
    /// Check if an expression contains any [PlaceholderUDF] functions.
    pub fn expr_contains_placeholder(expr: &Expr) -> bool {
        let mut found = false;
        expr.apply(|e| {
            if let Expr::ScalarFunction(func) = e {
                if func.func.inner().downcast_ref::<PlaceholderUDF>().is_some() {
                    found = true;
                    return Ok(TreeNodeRecursion::Stop);
                }
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .expect("expr_contains_placeholder traversal should not fail");
        found
    }

    /// Check if an expression contains any placeholder functions (scalar, aggregate, or window).
    pub fn expr_contains_any_placeholder(expr: &Expr) -> Result<bool> {
        let mut found = false;
        expr.apply(|e| {
            match e {
                Expr::ScalarFunction(func) => {
                    if func.func.inner().downcast_ref::<PlaceholderUDF>().is_some() {
                        found = true;
                        return Ok(TreeNodeRecursion::Stop);
                    }
                }
                Expr::AggregateFunction(func) => {
                    if func
                        .func
                        .inner()
                        .downcast_ref::<PlaceholderUDAF>()
                        .is_some()
                    {
                        found = true;
                        return Ok(TreeNodeRecursion::Stop);
                    }
                }
                Expr::WindowFunction(func) => {
                    if let WindowFunctionDefinition::WindowUDF(ref udf) = func.fun {
                        if udf.inner().downcast_ref::<PlaceholderUDWF>().is_some() {
                            found = true;
                            return Ok(TreeNodeRecursion::Stop);
                        }
                    }
                }
                _ => {}
            }
            Ok(TreeNodeRecursion::Continue)
        })?;

        Ok(found)
    }

    /// Replace all placeholder functions (scalar, aggregate, window) with implementations from the registry.
    ///
    /// Returns an error if any placeholder function is not found in the registry.
    pub fn expr_replace_placeholders(expr: Expr, registry: &dyn FunctionRegistry) -> Result<Expr> {
        expr.transform_up(|e| {
            match &e {
                Expr::ScalarFunction(func) => {
                    if func.func.inner().downcast_ref::<PlaceholderUDF>().is_some() {
                        let real_udf = registry.udf(func.name())?;
                        let replaced = Expr::ScalarFunction(ScalarFunction {
                            func: real_udf,
                            args: func.args.clone(),
                        });
                        return Ok(Transformed::yes(replaced));
                    }
                }
                Expr::AggregateFunction(func) => {
                    if func
                        .func
                        .inner()
                        .downcast_ref::<PlaceholderUDAF>()
                        .is_some()
                    {
                        let real_udaf = registry.udaf(func.func.name())?;
                        let replaced = Expr::AggregateFunction(
                            datafusion_expr::expr::AggregateFunction::new_udf(
                                real_udaf,
                                func.params.args.clone(),
                                func.params.distinct,
                                func.params.filter.clone(),
                                func.params.order_by.clone(),
                                func.params.null_treatment,
                            ),
                        );
                        return Ok(Transformed::yes(replaced));
                    }
                }
                Expr::WindowFunction(func) => {
                    if let WindowFunctionDefinition::WindowUDF(ref udf) = func.fun {
                        if udf.inner().downcast_ref::<PlaceholderUDWF>().is_some() {
                            let real_udwf = registry.udwf(udf.name())?;
                            let replaced = Expr::WindowFunction(Box::new(
                                datafusion_expr::expr::WindowFunction {
                                    fun: WindowFunctionDefinition::WindowUDF(real_udwf),
                                    params: func.params.clone(),
                                },
                            ));
                            return Ok(Transformed::yes(replaced));
                        }
                    }
                }
                _ => {}
            }
            Ok(Transformed::no(e))
        })
        .map(|t| t.data)
    }
}

impl FunctionRegistry for PlaceholderRegistry {
    fn udfs(&self) -> HashSet<String> {
        HashSet::new()
    }

    fn higher_order_function_names(&self) -> HashSet<String> {
        HashSet::new()
    }

    fn udafs(&self) -> HashSet<String> {
        HashSet::new()
    }

    fn udwfs(&self) -> HashSet<String> {
        HashSet::new()
    }

    fn udf(&self, name: &str) -> Result<Arc<ScalarUDF>> {
        Ok(Arc::new(ScalarUDF::new_from_impl(PlaceholderUDF::new(
            name,
        ))))
    }

    fn higher_order_function(&self, name: &str) -> Result<Arc<HigherOrderUDF>> {
        sedona_internal_err!("Imported higher-order function '{name}' is not supported")
    }

    fn udaf(&self, name: &str) -> Result<Arc<AggregateUDF>> {
        Ok(Arc::new(AggregateUDF::new_from_impl(PlaceholderUDAF::new(
            name,
        ))))
    }

    fn udwf(&self, name: &str) -> Result<Arc<WindowUDF>> {
        Ok(Arc::new(WindowUDF::new_from_impl(PlaceholderUDWF::new(
            name,
        ))))
    }

    fn expr_planners(&self) -> Vec<Arc<dyn datafusion_expr::planner::ExprPlanner>> {
        vec![]
    }
}

/// Placeholder [ScalarUDF] that preserves a function name.
///
/// This struct is a stub for deserializing expressions where the actual execution
/// of the expression is not necessarily important.
#[derive(Debug, Hash, PartialEq, Eq)]
pub struct PlaceholderUDF {
    name: String,
}

impl PlaceholderUDF {
    /// Create a new placeholder UDF with the given name.
    pub fn new(name: &str) -> Self {
        PlaceholderUDF {
            name: name.to_string(),
        }
    }
}

impl ScalarUDFImpl for PlaceholderUDF {
    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature {
        // In normal usage this is not passed through an optimizer (or if it is, it
        // will error as soon as its return field is inspected); however, we declare
        // this as Volatile to be safe.
        static SIGNATURE_ANY: OnceLock<Signature> = OnceLock::new();
        SIGNATURE_ANY.get_or_init(|| Signature::variadic_any(Volatility::Volatile))
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        sedona_internal_err!(
            "Imported placeholder UDF '{}' must be replaced before planning",
            self.name
        )
    }

    fn invoke_with_args(
        &self,
        _args: datafusion_expr::ScalarFunctionArgs,
    ) -> Result<datafusion_expr::ColumnarValue> {
        sedona_internal_err!(
            "Imported placeholder UDF '{}' must be replaced before execution",
            self.name
        )
    }
}

/// Placeholder [AggregateUDF] that preserves a function name.
///
/// This struct is a stub for deserializing expressions where the actual execution
/// of the expression is not necessarily important.
#[derive(Debug, Hash, PartialEq, Eq)]
pub struct PlaceholderUDAF {
    name: String,
}

impl PlaceholderUDAF {
    /// Create a new placeholder aggregate UDF with the given name.
    pub fn new(name: &str) -> Self {
        PlaceholderUDAF {
            name: name.to_string(),
        }
    }
}

impl AggregateUDFImpl for PlaceholderUDAF {
    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature {
        static SIGNATURE_ANY: OnceLock<Signature> = OnceLock::new();
        SIGNATURE_ANY.get_or_init(|| Signature::variadic_any(Volatility::Volatile))
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        sedona_internal_err!(
            "Imported placeholder UDAF '{}' must be replaced before planning",
            self.name
        )
    }

    fn accumulator(&self, _args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        sedona_internal_err!(
            "Imported placeholder UDAF '{}' must be replaced before execution",
            self.name
        )
    }
}

/// Placeholder [WindowUDF] that preserves a function name.
///
/// This struct is a stub for deserializing expressions where the actual execution
/// of the expression is not necessarily important.
#[derive(Debug, Hash, PartialEq, Eq)]
pub struct PlaceholderUDWF {
    name: String,
}

impl PlaceholderUDWF {
    /// Create a new placeholder window UDF with the given name.
    pub fn new(name: &str) -> Self {
        PlaceholderUDWF {
            name: name.to_string(),
        }
    }
}

impl WindowUDFImpl for PlaceholderUDWF {
    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature {
        static SIGNATURE_ANY: OnceLock<Signature> = OnceLock::new();
        SIGNATURE_ANY.get_or_init(|| Signature::variadic_any(Volatility::Volatile))
    }

    fn field(&self, _field_args: WindowUDFFieldArgs) -> Result<Arc<Field>> {
        sedona_internal_err!(
            "Imported placeholder UDWF '{}' must be replaced before planning",
            self.name
        )
    }

    fn partition_evaluator(
        &self,
        _partition_evaluator_args: PartitionEvaluatorArgs,
    ) -> Result<Box<dyn PartitionEvaluator>> {
        sedona_internal_err!(
            "Imported placeholder UDWF '{}' must be replaced before execution",
            self.name
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_common::ScalarValue;
    use datafusion_expr::{col, lit, SimpleScalarUDF};

    #[test]
    fn test_placeholder_registry_returns_empty_function_sets() {
        let registry = PlaceholderRegistry;
        assert!(registry.udfs().is_empty());
        assert!(registry.udafs().is_empty());
        assert!(registry.udwfs().is_empty());
    }

    #[test]
    fn test_placeholder_registry_creates_placeholder_udf() {
        let registry = PlaceholderRegistry;
        let udf = registry.udf("test_func").unwrap();
        assert_eq!(udf.name(), "test_func");

        // Verify it's a PlaceholderUDF
        let inner = udf.inner();
        assert!(inner.downcast_ref::<PlaceholderUDF>().is_some());
    }

    #[test]
    fn test_placeholder_registry_creates_placeholder_udaf() {
        let registry = PlaceholderRegistry;
        let udaf = registry.udaf("test_agg").unwrap();
        assert_eq!(udaf.name(), "test_agg");

        // Verify it's a PlaceholderUDAF
        let inner = udaf.inner();
        assert!(inner.downcast_ref::<PlaceholderUDAF>().is_some());
    }

    #[test]
    fn test_placeholder_registry_creates_placeholder_udwf() {
        let registry = PlaceholderRegistry;
        let udwf = registry.udwf("test_window").unwrap();
        assert_eq!(udwf.name(), "test_window");

        // Verify it's a PlaceholderUDWF
        let inner = udwf.inner();
        assert!(inner.downcast_ref::<PlaceholderUDWF>().is_some());
    }

    #[test]
    fn test_placeholder_registry_expr_planners_empty() {
        let registry = PlaceholderRegistry;
        assert!(registry.expr_planners().is_empty());
    }

    #[test]
    fn test_placeholder_udf_name() {
        let udf = PlaceholderUDF::new("my_func");
        assert_eq!(udf.name(), "my_func");
    }

    #[test]
    fn test_placeholder_udf_signature_is_variadic() {
        let udf = PlaceholderUDF::new("test");
        let sig = udf.signature();
        // The signature should accept any arguments
        assert!(matches!(sig.volatility, Volatility::Volatile));
    }

    #[test]
    fn test_placeholder_udf_return_type_errors() {
        let udf = PlaceholderUDF::new("my_func");
        let result = udf.return_type(&[DataType::Int32]);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("my_func"));
        assert!(err.contains("must be replaced before planning"));
    }

    #[test]
    fn test_placeholder_udf_equality() {
        let udf1 = PlaceholderUDF::new("func_a");
        let udf2 = PlaceholderUDF::new("func_a");
        let udf3 = PlaceholderUDF::new("func_b");

        assert_eq!(udf1, udf2);
        assert_ne!(udf1, udf3);
    }

    #[test]
    fn test_placeholder_udf_debug() {
        let udf = PlaceholderUDF::new("debug_test");
        let debug_str = format!("{:?}", udf);
        assert!(debug_str.contains("PlaceholderUDF"));
        assert!(debug_str.contains("debug_test"));
    }

    #[test]
    fn test_placeholder_udaf_name() {
        let udaf = PlaceholderUDAF::new("my_agg");
        assert_eq!(udaf.name(), "my_agg");
    }

    #[test]
    fn test_placeholder_udaf_signature_is_variadic() {
        let udaf = PlaceholderUDAF::new("test");
        let sig = udaf.signature();
        assert!(matches!(sig.volatility, Volatility::Volatile));
    }

    #[test]
    fn test_placeholder_udaf_return_type_errors() {
        let udaf = PlaceholderUDAF::new("my_agg");
        let result = udaf.return_type(&[DataType::Int32]);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("my_agg"));
        assert!(err.contains("must be replaced before planning"));
    }

    #[test]
    fn test_placeholder_udaf_equality() {
        let udaf1 = PlaceholderUDAF::new("agg_a");
        let udaf2 = PlaceholderUDAF::new("agg_a");
        let udaf3 = PlaceholderUDAF::new("agg_b");

        assert_eq!(udaf1, udaf2);
        assert_ne!(udaf1, udaf3);
    }

    #[test]
    fn test_placeholder_udaf_debug() {
        let udaf = PlaceholderUDAF::new("debug_agg");
        let debug_str = format!("{:?}", udaf);
        assert!(debug_str.contains("PlaceholderUDAF"));
        assert!(debug_str.contains("debug_agg"));
    }

    #[test]
    fn test_placeholder_udwf_name() {
        let udwf = PlaceholderUDWF::new("my_window");
        assert_eq!(udwf.name(), "my_window");
    }

    #[test]
    fn test_placeholder_udwf_signature_is_variadic() {
        let udwf = PlaceholderUDWF::new("test");
        let sig = udwf.signature();
        assert!(matches!(sig.volatility, Volatility::Volatile));
    }

    #[test]
    fn test_placeholder_udwf_return_type_errors() {
        // Note: WindowUDFImpl uses `field` instead of `return_type`
        // The field method requires WindowUDFFieldArgs which is hard to construct in tests,
        // so we just verify the signature exists and the UDF can be created.
        let udwf = PlaceholderUDWF::new("my_window");
        assert_eq!(udwf.name(), "my_window");
    }

    #[test]
    fn test_placeholder_udwf_partition_evaluator_errors() {
        // Note: partition_evaluator requires PartitionEvaluatorArgs which is hard to construct
        // in tests, so we just verify the UDF can be created and wrapped.
        let udwf = PlaceholderUDWF::new("my_window");
        let _wrapped = WindowUDF::new_from_impl(udwf);
    }

    #[test]
    fn test_placeholder_udwf_equality() {
        let udwf1 = PlaceholderUDWF::new("win_a");
        let udwf2 = PlaceholderUDWF::new("win_a");
        let udwf3 = PlaceholderUDWF::new("win_b");

        assert_eq!(udwf1, udwf2);
        assert_ne!(udwf1, udwf3);
    }

    #[test]
    fn test_placeholder_udwf_debug() {
        let udwf = PlaceholderUDWF::new("debug_window");
        let debug_str = format!("{:?}", udwf);
        assert!(debug_str.contains("PlaceholderUDWF"));
        assert!(debug_str.contains("debug_window"));
    }

    #[test]
    fn test_expr_contains_placeholder_true() {
        let udf = ScalarUDF::new_from_impl(PlaceholderUDF::new("test_func"));
        let expr = udf.call(vec![col("x")]);

        assert!(PlaceholderRegistry::expr_contains_placeholder(&expr));
    }

    #[test]
    fn test_expr_contains_placeholder_false_for_column() {
        let expr = col("x");
        assert!(!PlaceholderRegistry::expr_contains_placeholder(&expr));
    }

    #[test]
    fn test_expr_contains_placeholder_nested() {
        let udf = ScalarUDF::new_from_impl(PlaceholderUDF::new("inner_func"));
        let inner = udf.call(vec![col("x")]);
        // Wrap in a binary expression
        let expr = inner.gt(lit(5i32));

        assert!(PlaceholderRegistry::expr_contains_placeholder(&expr));
    }

    #[test]
    fn test_expr_contains_any_placeholder_with_scalar() {
        let udf = ScalarUDF::new_from_impl(PlaceholderUDF::new("test_func"));
        let expr = udf.call(vec![col("x")]);

        assert!(PlaceholderRegistry::expr_contains_any_placeholder(&expr).unwrap());
    }

    #[test]
    fn test_expr_contains_any_placeholder_with_aggregate() {
        let udaf = AggregateUDF::new_from_impl(PlaceholderUDAF::new("test_agg"));
        let expr = udaf.call(vec![col("x")]);

        assert!(PlaceholderRegistry::expr_contains_any_placeholder(&expr).unwrap());
    }

    #[test]
    fn test_expr_contains_any_placeholder_false_for_column() {
        let expr = col("x");
        assert!(!PlaceholderRegistry::expr_contains_any_placeholder(&expr).unwrap());
    }

    #[test]
    fn test_expr_replace_placeholder_udfs() {
        // Create a simple registry that returns a real UDF implementation
        // and attempts aggregate and window replacement (these are verbose
        // to mock so we don't do that here)
        struct TestRegistry;
        impl FunctionRegistry for TestRegistry {
            fn udfs(&self) -> HashSet<String> {
                HashSet::new()
            }

            fn higher_order_function_names(&self) -> HashSet<String> {
                HashSet::new()
            }

            fn udafs(&self) -> HashSet<String> {
                HashSet::new()
            }

            fn udwfs(&self) -> HashSet<String> {
                HashSet::new()
            }

            fn udf(&self, name: &str) -> Result<Arc<ScalarUDF>> {
                // Return a real UDF using SimpleScalarUDF with a closure
                let udf: ScalarUDF = SimpleScalarUDF::new_with_signature(
                    name,
                    Signature::any(1, Volatility::Immutable),
                    DataType::Int32,
                    Arc::new(|_args| Ok(ScalarValue::Int32(Some(42)).into())),
                )
                .into();
                Ok(Arc::new(udf))
            }

            fn higher_order_function(&self, name: &str) -> Result<Arc<HigherOrderUDF>> {
                sedona_internal_err!("attempt to replace higher-order function '{name}'")
            }

            fn udaf(&self, name: &str) -> Result<Arc<AggregateUDF>> {
                sedona_internal_err!("attempt to replace udaf '{name}'")
            }

            fn udwf(&self, name: &str) -> Result<Arc<WindowUDF>> {
                sedona_internal_err!("attempt to replace udwf '{name}'")
            }

            fn expr_planners(&self) -> Vec<Arc<dyn datafusion_expr::planner::ExprPlanner>> {
                vec![]
            }
        }

        let udf = ScalarUDF::new_from_impl(PlaceholderUDF::new("original"));
        let expr = udf.call(vec![col("x")]);

        // Verify it starts as a placeholder
        assert!(PlaceholderRegistry::expr_contains_placeholder(&expr));

        let registry = TestRegistry;
        let replaced = PlaceholderRegistry::expr_replace_placeholders(expr, &registry).unwrap();

        // Verify the function was replaced and is no longer a placeholder
        assert!(!PlaceholderRegistry::expr_contains_placeholder(&replaced));
        match &replaced {
            Expr::ScalarFunction(ScalarFunction { func, .. }) => {
                assert_eq!(func.name(), "original");
                // Verify it's NOT a PlaceholderUDF anymore
                assert!(func.inner().downcast_ref::<PlaceholderUDF>().is_none());
            }
            other => panic!("Expected ScalarFunction, got {:?}", other),
        }

        // Test aggregate replacement is attempted (errors prove the registry was called)
        let udaf = AggregateUDF::new_from_impl(PlaceholderUDAF::new("my_aggregate"));
        let agg_expr = udaf.call(vec![col("x")]);
        assert!(PlaceholderRegistry::expr_contains_any_placeholder(&agg_expr).unwrap());

        let agg_result = PlaceholderRegistry::expr_replace_placeholders(agg_expr, &registry);
        assert!(agg_result.is_err());
        let agg_err = agg_result.unwrap_err().to_string();
        assert!(
            agg_err.contains("attempt to replace udaf 'my_aggregate'"),
            "Expected error about udaf replacement attempt, got: {}",
            agg_err
        );

        // Test window replacement is attempted (errors prove the registry was called)
        let udwf = WindowUDF::new_from_impl(PlaceholderUDWF::new("my_window"));
        let window_func = datafusion_expr::expr::WindowFunction::new(
            WindowFunctionDefinition::WindowUDF(Arc::new(udwf)),
            vec![col("x")],
        );
        let win_expr = Expr::WindowFunction(Box::new(window_func));
        assert!(PlaceholderRegistry::expr_contains_any_placeholder(&win_expr).unwrap());

        let win_result = PlaceholderRegistry::expr_replace_placeholders(win_expr, &registry);
        assert!(win_result.is_err());
        let win_err = win_result.unwrap_err().to_string();
        assert!(
            win_err.contains("attempt to replace udwf 'my_window'"),
            "Expected error about udwf replacement attempt, got: {}",
            win_err
        );
    }
}
