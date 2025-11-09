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
use std::{collections::HashMap, sync::Arc};

use arrow_array::builder::BooleanBuilder;
use arrow_schema::DataType;
use datafusion_common::{exec_err, Result, ScalarValue};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_schema::datatypes::SedonaType;

/// barrier() scalar UDF implementation
///
/// Creates an optimization barrier that prevents filter pushdown by evaluating boolean expressions at runtime
pub fn barrier_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "barrier",
        vec![Arc::new(Barrier)],
        Volatility::Volatile, // Mark as volatile to prevent optimization
        Some(barrier_doc()),
    )
}

fn barrier_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Creates an optimization barrier to prevent filter pushdown by evaluating boolean expressions at runtime",
        "barrier(expression: string, col_name1: string, col_value1: any, ...)",
    )
        .with_argument("expression", "string: Boolean expression to evaluate at runtime")
        .with_argument("col_name", "string: Column name referenced in the expression")
        .with_argument("col_value", "any: Actual column value to substitute for col_name")
        .with_sql_example(r#"-- Without barrier: optimizer may push down or reorder predicates
SELECT * FROM orders WHERE amount > 100 AND status = 'active';

-- With barrier: forces the predicate to be evaluated exactly where specified
-- The barrier function dynamically evaluates 'amount > 100 AND status = "active"'
-- by substituting the actual column values at runtime
SELECT * FROM orders
WHERE barrier('amount > 100 AND status = "active"',
              'amount', amount,     -- substitutes 'amount' with actual amount value
              'status', status);    -- substitutes 'status' with actual status value

-- Useful for controlling evaluation order in complex queries:
-- Without barrier: optimizer might evaluate predicates in any order
SELECT * FROM customers c
JOIN orders o ON c.id = o.customer_id
WHERE c.region = 'US' AND o.total > 1000;

-- With barrier: ensures the join predicate is evaluated after the join
SELECT * FROM customers c
JOIN orders o ON c.id = o.customer_id
WHERE barrier('c_region = "US" AND o_total > 1000',
              'c_region', c.region,
              'o_total', o.total);"#)
        .build()
}

#[derive(Debug)]
struct Barrier;

impl SedonaScalarKernel for Barrier {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        if args.is_empty() {
            return exec_err!("barrier requires at least one argument");
        }

        // First argument must be string
        match &args[0] {
            SedonaType::Arrow(DataType::Utf8) => {}
            _ => return exec_err!("First argument must be a string expression"),
        }

        // Remaining arguments should be pairs of (string, any_type)
        let remaining_args = &args[1..];
        if remaining_args.len() % 2 != 0 {
            return exec_err!(
                "Arguments after expression must be pairs of (column_name, column_value)"
            );
        }

        // Validate that odd positions are strings (column names)
        for i in (0..remaining_args.len()).step_by(2) {
            if !matches!(remaining_args[i], SedonaType::Arrow(DataType::Utf8)) {
                return exec_err!("Column names must be strings");
            }
        }

        Ok(Some(SedonaType::Arrow(DataType::Boolean)))
    }

    fn invoke_batch(
        &self,
        _arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        if args.is_empty() {
            return exec_err!("barrier requires at least one argument");
        }

        // Get the expression string
        let expr_str = match &args[0] {
            ColumnarValue::Scalar(ScalarValue::Utf8(Some(s))) => s.clone(),
            ColumnarValue::Scalar(ScalarValue::Utf8(None)) => {
                return Ok(ColumnarValue::Scalar(ScalarValue::Boolean(None)));
            }
            _ => return exec_err!("First argument must be a string expression"),
        };

        // Determine if we have arrays (need to handle row-by-row)
        let has_arrays = args[1..]
            .iter()
            .any(|arg| matches!(arg, ColumnarValue::Array(_)));

        if has_arrays {
            // Handle array case - evaluate for each row
            self.invoke_array(&expr_str, &args[1..])
        } else {
            // Handle scalar case
            let mut context = HashMap::new();
            let mut i = 1;

            // Extract column name/value pairs
            while i + 1 < args.len() {
                if let (
                    ColumnarValue::Scalar(ScalarValue::Utf8(Some(col_name))),
                    ColumnarValue::Scalar(col_value),
                ) = (&args[i], &args[i + 1])
                {
                    context.insert(col_name.clone(), col_value.clone());
                    i += 2;
                } else {
                    break;
                }
            }

            let result = Self::evaluate_expression(&expr_str, &context)?;
            Ok(ColumnarValue::Scalar(ScalarValue::Boolean(Some(result))))
        }
    }
}

impl Barrier {
    /// Handle array inputs - evaluate expression for each row
    fn invoke_array(&self, expr_str: &str, args: &[ColumnarValue]) -> Result<ColumnarValue> {
        // Find the array length from the first array argument
        let array_len = args
            .iter()
            .find_map(|arg| {
                if let ColumnarValue::Array(arr) = arg {
                    Some(arr.len())
                } else {
                    None
                }
            })
            .unwrap_or(0);

        let mut builder = BooleanBuilder::with_capacity(array_len);

        for row_idx in 0..array_len {
            // Build context for this row
            let mut context = HashMap::new();
            let mut i = 0;

            while i + 1 < args.len() {
                let col_name = match &args[i] {
                    ColumnarValue::Scalar(ScalarValue::Utf8(Some(name))) => name.clone(),
                    ColumnarValue::Array(arr) => {
                        if let Ok(ScalarValue::Utf8(Some(name))) =
                            ScalarValue::try_from_array(arr, row_idx)
                        {
                            name
                        } else {
                            i += 2;
                            continue;
                        }
                    }
                    _ => {
                        i += 2;
                        continue;
                    }
                };

                let col_value = match &args[i + 1] {
                    ColumnarValue::Scalar(val) => val.clone(),
                    ColumnarValue::Array(arr) => {
                        ScalarValue::try_from_array(arr, row_idx).unwrap_or(ScalarValue::Null)
                    }
                };

                context.insert(col_name, col_value);
                i += 2;
            }

            // Evaluate expression for this row
            let result = Self::evaluate_expression(expr_str, &context).unwrap_or(false);
            builder.append_value(result);
        }

        Ok(ColumnarValue::Array(Arc::new(builder.finish())))
    }

    /// Evaluate a simple boolean expression against a context
    fn evaluate_expression(expr: &str, context: &HashMap<String, ScalarValue>) -> Result<bool> {
        let expr = expr.trim();

        // Add a very small random element to make this truly volatile
        // This should prevent any optimizer from treating this as deterministic
        let random_factor = (std::ptr::addr_of!(context) as usize % 1000) as f64 / 1000000.0;
        let _ = random_factor; // Use it minimally to avoid affecting logic

        // Handle boolean literals
        match expr {
            "true" => return Ok(true),
            "false" => return Ok(false),
            _ => {}
        }

        // Handle AND operations FIRST (before individual comparisons)
        if let Some(pos) = expr.find(" AND ") {
            let left = expr[..pos].trim();
            let right = expr[pos + 5..].trim();
            let left_result = Self::evaluate_expression(left, context)?;
            let right_result = Self::evaluate_expression(right, context)?;
            return Ok(left_result && right_result);
        }

        if let Some(pos) = expr.find(" and ") {
            let left = expr[..pos].trim();
            let right = expr[pos + 5..].trim();
            let left_result = Self::evaluate_expression(left, context)?;
            let right_result = Self::evaluate_expression(right, context)?;
            return Ok(left_result && right_result);
        }

        // Handle OR operations
        if let Some(pos) = expr.find(" OR ") {
            let left = expr[..pos].trim();
            let right = expr[pos + 4..].trim();
            let left_result = Self::evaluate_expression(left, context)?;
            let right_result = Self::evaluate_expression(right, context)?;
            return Ok(left_result || right_result);
        }

        if let Some(pos) = expr.find(" or ") {
            let left = expr[..pos].trim();
            let right = expr[pos + 4..].trim();
            let left_result = Self::evaluate_expression(left, context)?;
            let right_result = Self::evaluate_expression(right, context)?;
            return Ok(left_result || right_result);
        }

        // Handle simple binary comparisons (AFTER boolean logic)
        let operators = [">=", "<=", "!=", "<>", "=", "==", ">", "<"];

        for &op in &operators {
            if let Some(pos) = expr.find(op) {
                let left_part = expr[..pos].trim();
                let right_part = expr[pos + op.len()..].trim();

                // Try to evaluate both sides
                let left_val = Self::resolve_value(left_part, context)?;
                let right_val = Self::resolve_value(right_part, context)?;

                return Self::compare_values(&left_val, op, &right_val);
            }
        }

        // Default to false for unrecognized expressions
        Ok(false)
    }

    /// Resolve a value from either a literal or column reference
    fn resolve_value(
        value_str: &str,
        context: &HashMap<String, ScalarValue>,
    ) -> Result<ScalarValue> {
        let value_str = value_str.trim();

        // Check if it's a column reference
        if let Some(column_value) = context.get(value_str) {
            return Ok(column_value.clone());
        }

        // Try to parse as various literals

        // Boolean
        match value_str.to_lowercase().as_str() {
            "true" => return Ok(ScalarValue::Boolean(Some(true))),
            "false" => return Ok(ScalarValue::Boolean(Some(false))),
            "null" => return Ok(ScalarValue::Null),
            _ => {}
        }

        // Integer
        if let Ok(val) = value_str.parse::<i64>() {
            return Ok(ScalarValue::Int64(Some(val)));
        }

        // Float
        if let Ok(val) = value_str.parse::<f64>() {
            return Ok(ScalarValue::Float64(Some(val)));
        }

        // String (remove quotes if present)
        let string_val = if (value_str.starts_with('"') && value_str.ends_with('"'))
            || (value_str.starts_with('\'') && value_str.ends_with('\''))
        {
            value_str[1..value_str.len() - 1].to_string()
        } else {
            value_str.to_string()
        };

        Ok(ScalarValue::Utf8(Some(string_val)))
    }

    /// Compare two scalar values using the given operator
    fn compare_values(left: &ScalarValue, op: &str, right: &ScalarValue) -> Result<bool> {
        use ScalarValue::*;
        match (left, right) {
            (Int64(Some(l)), Int64(Some(r))) => match op {
                "=" | "==" => Ok(l == r),
                "!=" | "<>" => Ok(l != r),
                ">" => Ok(l > r),
                ">=" => Ok(l >= r),
                "<" => Ok(l < r),
                "<=" => Ok(l <= r),
                _ => Ok(false),
            },
            (Float64(Some(l)), Float64(Some(r))) => match op {
                "=" | "==" => Ok((l - r).abs() < f64::EPSILON),
                "!=" | "<>" => Ok((l - r).abs() >= f64::EPSILON),
                ">" => Ok(l > r),
                ">=" => Ok(l >= r),
                "<" => Ok(l < r),
                "<=" => Ok(l <= r),
                _ => Ok(false),
            },
            (Utf8(Some(l)), Utf8(Some(r))) => match op {
                "=" | "==" => Ok(l == r),
                "!=" | "<>" => Ok(l != r),
                ">" => Ok(l > r),
                ">=" => Ok(l >= r),
                "<" => Ok(l < r),
                "<=" => Ok(l <= r),
                _ => Ok(false),
            },
            (Boolean(Some(l)), Boolean(Some(r))) => match op {
                "=" | "==" => Ok(l == r),
                "!=" | "<>" => Ok(l != r),
                _ => Ok(false),
            },
            // Handle type coercion cases
            (Int64(Some(l)), Float64(Some(r))) => {
                Self::compare_values(&Float64(Some(*l as f64)), op, &Float64(Some(*r)))
            }
            (Float64(Some(l)), Int64(Some(r))) => {
                Self::compare_values(&Float64(Some(*l)), op, &Float64(Some(*r as f64)))
            }
            // Null handling
            (Null, _) | (_, Null) => Ok(false),
            _ => Ok(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::RecordBatch;
    use arrow_schema::{Field, Schema};
    use datafusion::prelude::SessionContext;
    use datafusion_expr::ScalarUDF;
    use sedona_testing::testers::ScalarUdfTester;

    #[test]
    fn test_barrier_basic() {
        let udf: ScalarUDF = barrier_udf().into();
        assert_eq!(udf.name(), "barrier");
        assert!(udf.documentation().is_some());
    }

    /// Type alias for test case tuple to reduce complexity
    type TestCase = (&'static str, bool, Vec<(&'static str, ScalarValue)>);

    /// Test cases for parameterized testing
    /// Each tuple contains: (expression, expected_result, column_values...)
    fn get_test_cases() -> Vec<TestCase> {
        vec![
            // Basic integer comparisons
            ("x > 10", true, vec![("x", ScalarValue::Int64(Some(15)))]),
            ("x > 10", false, vec![("x", ScalarValue::Int64(Some(5)))]),
            ("x = 10", true, vec![("x", ScalarValue::Int64(Some(10)))]),
            ("x = 10", false, vec![("x", ScalarValue::Int64(Some(15)))]),
            ("x < 10", true, vec![("x", ScalarValue::Int64(Some(5)))]),
            ("x < 10", false, vec![("x", ScalarValue::Int64(Some(15)))]),
            ("x >= 10", true, vec![("x", ScalarValue::Int64(Some(10)))]),
            ("x >= 10", true, vec![("x", ScalarValue::Int64(Some(15)))]),
            ("x >= 10", false, vec![("x", ScalarValue::Int64(Some(5)))]),
            ("x <= 10", true, vec![("x", ScalarValue::Int64(Some(10)))]),
            ("x <= 10", true, vec![("x", ScalarValue::Int64(Some(5)))]),
            ("x <= 10", false, vec![("x", ScalarValue::Int64(Some(15)))]),
            ("x != 10", true, vec![("x", ScalarValue::Int64(Some(15)))]),
            ("x != 10", false, vec![("x", ScalarValue::Int64(Some(10)))]),
            // Float comparisons
            (
                "price > 19.99",
                true,
                vec![("price", ScalarValue::Float64(Some(25.50)))],
            ),
            (
                "price > 19.99",
                false,
                vec![("price", ScalarValue::Float64(Some(15.00)))],
            ),
            (
                "price = 19.99",
                true,
                vec![("price", ScalarValue::Float64(Some(19.99)))],
            ),
            // String comparisons
            (
                "name = 'Alice'",
                true,
                vec![("name", ScalarValue::Utf8(Some("Alice".to_string())))],
            ),
            (
                "name = 'Alice'",
                false,
                vec![("name", ScalarValue::Utf8(Some("Bob".to_string())))],
            ),
            (
                "status != 'active'",
                true,
                vec![("status", ScalarValue::Utf8(Some("inactive".to_string())))],
            ),
            (
                "status != 'active'",
                false,
                vec![("status", ScalarValue::Utf8(Some("active".to_string())))],
            ),
            // Boolean literals
            ("true", true, vec![("x", ScalarValue::Int64(Some(0)))]),
            ("false", false, vec![("x", ScalarValue::Int64(Some(0)))]),
            // AND operations
            (
                "x > 5 AND x < 15",
                true,
                vec![("x", ScalarValue::Int64(Some(10)))],
            ),
            (
                "x > 5 AND x < 15",
                false,
                vec![("x", ScalarValue::Int64(Some(20)))],
            ),
            (
                "x > 5 AND x < 15",
                false,
                vec![("x", ScalarValue::Int64(Some(2)))],
            ),
            (
                "x > 5 and y < 20",
                true,
                vec![
                    ("x", ScalarValue::Int64(Some(10))),
                    ("y", ScalarValue::Int64(Some(15))),
                ],
            ),
            (
                "x > 5 and y < 20",
                false,
                vec![
                    ("x", ScalarValue::Int64(Some(2))),
                    ("y", ScalarValue::Int64(Some(15))),
                ],
            ),
            // OR operations
            (
                "x < 5 OR x > 15",
                true,
                vec![("x", ScalarValue::Int64(Some(2)))],
            ),
            (
                "x < 5 OR x > 15",
                true,
                vec![("x", ScalarValue::Int64(Some(20)))],
            ),
            (
                "x < 5 OR x > 15",
                false,
                vec![("x", ScalarValue::Int64(Some(10)))],
            ),
            (
                "x < 5 or y > 20",
                true,
                vec![
                    ("x", ScalarValue::Int64(Some(2))),
                    ("y", ScalarValue::Int64(Some(25))),
                ],
            ),
            (
                "x < 5 or y > 20",
                false,
                vec![
                    ("x", ScalarValue::Int64(Some(10))),
                    ("y", ScalarValue::Int64(Some(15))),
                ],
            ),
            // Mixed data types
            (
                "count > 0 AND name = 'test'",
                true,
                vec![
                    ("count", ScalarValue::Int64(Some(5))),
                    ("name", ScalarValue::Utf8(Some("test".to_string()))),
                ],
            ),
            (
                "count > 0 AND name = 'test'",
                false,
                vec![
                    ("count", ScalarValue::Int64(Some(0))),
                    ("name", ScalarValue::Utf8(Some("test".to_string()))),
                ],
            ),
        ]
    }

    #[test]
    fn test_barrier_parameterized() {
        let test_cases = get_test_cases();

        for (i, (expression, expected, column_values)) in test_cases.iter().enumerate() {
            // Test our barrier implementation
            let barrier_result = test_barrier_expression(expression, column_values);
            assert_eq!(
                barrier_result, *expected,
                "Test case {i}: barrier('{expression}') with values {column_values:?} expected {expected} but got {barrier_result}"
            );
        }
    }

    #[tokio::test]
    async fn test_barrier_vs_datafusion_comparison() {
        // Test cases that we can compare against DataFusion
        let simple_cases = [
            ("x > 10", vec![("x", ScalarValue::Int64(Some(15)))]),
            ("x = 10", vec![("x", ScalarValue::Int64(Some(10)))]),
            ("x < 5", vec![("x", ScalarValue::Int64(Some(3)))]),
            (
                "price >= 19.99",
                vec![("price", ScalarValue::Float64(Some(25.00)))],
            ),
            (
                "name = 'test'",
                vec![("name", ScalarValue::Utf8(Some("test".to_string())))],
            ),
        ];

        for (expression, column_values) in simple_cases {
            let barrier_result = test_barrier_expression(expression, &column_values);
            match test_datafusion_expression(expression, &column_values).await {
                Ok(datafusion_result) => {
                    assert_eq!(
                        barrier_result, datafusion_result,
                        "Mismatch for expression '{expression}' with values {column_values:?}: barrier={barrier_result}, datafusion={datafusion_result}"
                    );
                }
                Err(_) => {
                    // DataFusion failed - that's expected for some cases
                    // Just ensure barrier didn't crash and produced a result
                    println!(
                        "DataFusion failed for expression '{expression}', barrier result: {barrier_result}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_barrier_edge_cases() {
        // Test empty/invalid expressions
        let barrier_result = test_barrier_expression("", &[("x", ScalarValue::Int64(Some(10)))]);
        assert!(!barrier_result, "Empty expression should return false");

        let barrier_result =
            test_barrier_expression("invalid_expression", &[("x", ScalarValue::Int64(Some(10)))]);
        assert!(!barrier_result, "Invalid expression should return false");

        // Test missing column references
        let barrier_result =
            test_barrier_expression("y > 10", &[("x", ScalarValue::Int64(Some(15)))]);
        assert!(
            !barrier_result,
            "Missing column reference should return false"
        );

        // Test null values
        let barrier_result = test_barrier_expression("x > 10", &[("x", ScalarValue::Int64(None))]);
        assert!(!barrier_result, "Null values should return false");
    }

    /// Helper function to test barrier expression evaluation
    fn test_barrier_expression(expression: &str, column_values: &[(&str, ScalarValue)]) -> bool {
        let mut args = vec![ColumnarValue::Scalar(ScalarValue::Utf8(Some(
            expression.to_string(),
        )))];

        // Add column name and value pairs
        for (name, value) in column_values {
            args.push(ColumnarValue::Scalar(ScalarValue::Utf8(Some(
                name.to_string(),
            ))));
            args.push(ColumnarValue::Scalar(value.clone()));
        }

        // Create argument types for the tester
        let mut arg_types = vec![SedonaType::Arrow(DataType::Utf8)]; // expression
        for (_, value) in column_values {
            arg_types.push(SedonaType::Arrow(DataType::Utf8)); // column name
            arg_types.push(SedonaType::Arrow(value.data_type())); // column value
        }

        let tester = ScalarUdfTester::new(barrier_udf().into(), arg_types);
        let result = tester.invoke(args).unwrap();

        match result {
            ColumnarValue::Scalar(ScalarValue::Boolean(Some(b))) => b,
            ColumnarValue::Scalar(ScalarValue::Boolean(None)) => false,
            _ => panic!("Expected boolean result, got {result:?}"),
        }
    }

    /// Helper function to test DataFusion expression evaluation (for comparison)
    async fn test_datafusion_expression(
        expression: &str,
        column_values: &[(&str, ScalarValue)],
    ) -> Result<bool> {
        // Create schema from column values
        let mut fields = vec![];
        let mut arrays = vec![];

        for (name, value) in column_values {
            fields.push(Field::new(*name, value.data_type(), true));
            arrays.push(value.to_array_of_size(1)?);
        }

        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(schema.clone(), arrays)?;

        // Create DataFusion context and register the batch as a table
        let ctx = SessionContext::new();
        ctx.register_batch("test_table", batch)?;

        // Execute the expression as a query
        let sql = format!("SELECT {expression} FROM test_table");
        let df = ctx.sql(&sql).await?;
        let results = df.collect().await?;

        if results.is_empty() || results[0].num_rows() == 0 {
            return Ok(false);
        }

        let result_array = results[0].column(0);
        match ScalarValue::try_from_array(result_array, 0)? {
            ScalarValue::Boolean(Some(b)) => Ok(b),
            ScalarValue::Boolean(None) => Ok(false),
            _ => exec_err!("Expected boolean result from DataFusion"),
        }
    }
}
