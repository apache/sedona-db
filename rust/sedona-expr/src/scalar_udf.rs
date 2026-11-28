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
use std::{collections::HashMap, fmt::Debug, sync::Arc};

use arrow_schema::{DataType, FieldRef};
use datafusion_common::config::ConfigOptions;
use datafusion_common::format::DEFAULT_CAST_OPTIONS;
use datafusion_common::{Result, ScalarValue, not_impl_err};
use datafusion_expr::{
    ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature,
    Volatility,
};
use sedona_common::sedona_internal_err;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

/// Shorthand for a [SedonaScalarKernel] reference
pub type ScalarKernelRef = Arc<dyn SedonaScalarKernel>;

/// Helper to resolve an iterable of kernels
pub trait IntoScalarKernelRefs {
    fn into_scalar_kernel_refs(self) -> Vec<ScalarKernelRef>;
}

impl IntoScalarKernelRefs for ScalarKernelRef {
    fn into_scalar_kernel_refs(self) -> Vec<ScalarKernelRef> {
        vec![self]
    }
}

impl IntoScalarKernelRefs for Vec<ScalarKernelRef> {
    fn into_scalar_kernel_refs(self) -> Vec<ScalarKernelRef> {
        self
    }
}

impl<T: SedonaScalarKernel + 'static> IntoScalarKernelRefs for T {
    fn into_scalar_kernel_refs(self) -> Vec<ScalarKernelRef> {
        vec![Arc::new(self)]
    }
}

impl<T: SedonaScalarKernel + 'static> IntoScalarKernelRefs for Vec<Arc<T>> {
    fn into_scalar_kernel_refs(self) -> Vec<ScalarKernelRef> {
        self.into_iter()
            .map(|item| item as ScalarKernelRef)
            .collect()
    }
}

/// Top-level scalar user-defined function
///
/// This struct implements datafusion's ScalarUDF and implements kernel dispatch
/// and argument wrapping/unwrapping while this is still necessary to support
/// user-defined types.
#[derive(Debug, Clone)]
pub struct SedonaScalarUDF {
    name: String,
    signature: Signature,
    kernels: Vec<ScalarKernelRef>,
    aliases: Vec<String>,
    /// Class-level, string-keyed metadata describing this UDF to the
    /// planner. Flags are set via [`SedonaScalarUDF::with_metadata`] and
    /// read back via [`SedonaScalarUDF::metadata`]; the planner keys off
    /// well-known entries (e.g. the raster `"needs_pixels"` flag, whose
    /// key is owned by `sedona-raster-functions`) without this crate
    /// knowing their meaning. The map shape leaves room for further
    /// planner-visible flags.
    metadata: HashMap<String, String>,
}

impl PartialEq for SedonaScalarUDF {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for SedonaScalarUDF {}

impl std::hash::Hash for SedonaScalarUDF {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

/// User-defined function implementation
///
/// A `SedonaScalarUdf` is comprised of one or more kernels, to which it dispatches
/// the first whose return_type returns `Some()`. Whereas a SeondaScalarUdf represents
/// a logical operation (e.g., ST_Intersects()), a kernel wraps the logic around a specific
/// implementation.
pub trait SedonaScalarKernel: Debug + Send + Sync {
    /// Calculate a return type given input types
    ///
    /// Returns Some(physical_type) if this kernel applies to the input types or
    /// None otherwise. This struct acts as a version of the Signature that can
    /// better accommodate the types we need to support (and might be able to be
    /// removed when there is better support for matching user-defined types/
    /// types with metadata in DataFusion).
    ///
    /// The [`ArgMatcher`] contains a set of helper functions to help implement this
    /// function.
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>>;

    /// Calculate a return type given input type and scalar arguments
    ///
    /// Most functions should implement [SedonaScalarKernel::return_type]; however, some functions
    /// (e.g., ST_SetSRID) calculate a return type based on the value of the argument if it is
    /// a constant. If this is implemented, [SedonaScalarKernel::return_type] will not be called.
    fn return_type_from_args_and_scalars(
        &self,
        args: &[SedonaType],
        _scalar_args: &[Option<&ScalarValue>],
    ) -> Result<Option<SedonaType>> {
        self.return_type(args)
    }

    /// Compute a batch of results
    ///
    /// Computes an output chunk based on the physical types of the input and the
    /// computed output type. The ColumnarValues passed are the "unwrapped" representation
    /// of any extension type (e.g., for Wkb the provided ColumnarValue will be Binary).
    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue>;

    fn invoke_batch_from_args(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
        _return_type: &SedonaType,
        _num_rows: usize,
        _config_options: Option<&ConfigOptions>,
    ) -> Result<ColumnarValue> {
        self.invoke_batch(arg_types, args)
    }
}

/// Type definition for a Scalar kernel implementation function
pub type SedonaScalarKernelImpl =
    Arc<dyn Fn(&[SedonaType], &[ColumnarValue]) -> Result<ColumnarValue> + Send + Sync>;

/// Scalar kernel based on a function for testing
pub struct SimpleSedonaScalarKernel {
    arg_matcher: ArgMatcher,
    fun: SedonaScalarKernelImpl,
}

impl Debug for SimpleSedonaScalarKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("SimpleSedonaScalarKernel").finish()
    }
}

impl SimpleSedonaScalarKernel {
    pub fn new_ref(arg_matcher: ArgMatcher, fun: SedonaScalarKernelImpl) -> ScalarKernelRef {
        Arc::new(Self { arg_matcher, fun })
    }
}

impl SedonaScalarKernel for SimpleSedonaScalarKernel {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        self.arg_matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        (self.fun)(arg_types, args)
    }
}

impl SedonaScalarUDF {
    /// Create a new SedonaScalarUDF
    pub fn new(
        name: &str,
        kernels: Vec<ScalarKernelRef>,
        volatility: Volatility,
    ) -> SedonaScalarUDF {
        let signature = Signature::user_defined(volatility);
        Self {
            name: name.to_string(),
            signature,
            kernels,
            aliases: vec![],
            metadata: HashMap::new(),
        }
    }

    /// Add aliases to an existing SedonaScalarUDF
    pub fn with_aliases(self, aliases: Vec<String>) -> SedonaScalarUDF {
        Self { aliases, ..self }
    }

    /// Set a class-level metadata entry on this UDF, returning the
    /// modified UDF. Metadata is planner-visible (e.g. the
    /// `RS_EnsureLoaded` optimizer rule reads the raster `"needs_pixels"`
    /// flag) and crosses the `sedona-extension` FFI boundary so
    /// plugin-defined UDFs can declare it too.
    pub fn with_metadata(
        mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> SedonaScalarUDF {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Class-level metadata map describing this UDF to the planner.
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }

    /// The kernels (overload implementations) this UDF dispatches to.
    ///
    /// Exposed so a UDF's kernels can be exported across the
    /// `sedona-extension` FFI boundary (e.g. handed to an out-of-tree
    /// plugin as capsules).
    pub fn kernels(&self) -> &[ScalarKernelRef] {
        &self.kernels
    }

    /// Replace this UDF's kernels while preserving its name, volatility,
    /// aliases, and metadata.
    pub fn with_kernels(self, kernels: Vec<ScalarKernelRef>) -> Self {
        Self { kernels, ..self }
    }

    /// Create a SedonaScalarUDF from a single kernel
    ///
    /// This constructor creates a [Volatility::Immutable] function with no documentation
    /// consisting of only the implementation provided.
    pub fn from_impl(name: &str, kernels: impl IntoScalarKernelRefs) -> SedonaScalarUDF {
        Self::new(
            name,
            kernels.into_scalar_kernel_refs(),
            Volatility::Immutable,
        )
    }

    /// Add a new kernel to a Scalar UDF
    ///
    /// Because kernels are resolved in reverse order, the new kernel will take
    /// precedence over any previously added kernels that apply to the same types.
    pub fn add_kernels(&mut self, kernels: impl IntoScalarKernelRefs) {
        for kernel in kernels.into_scalar_kernel_refs() {
            self.kernels.push(kernel);
        }
    }

    /// Find the first kernel that applies to `args`, or `None` if no kernel does
    fn first_matching_kernel(
        &self,
        args: &[SedonaType],
        scalars: &[Option<&ScalarValue>],
    ) -> Result<Option<(&dyn SedonaScalarKernel, SedonaType)>> {
        // Resolve kernels in reverse so that more recently added ones are resolved first
        for kernel in self.kernels.iter().rev() {
            if let Some(return_type) = kernel.return_type_from_args_and_scalars(args, scalars)? {
                return Ok(Some((kernel.as_ref(), return_type)));
            }
        }

        Ok(None)
    }

    /// The argument types a plan has to cast `args` to before this function can run
    /// on them, or `None` when a kernel accepts `args` as given
    ///
    /// Kernels are first matched against the argument types exactly as given.
    /// Only when no kernel matches are the integer and decimal arguments widened
    /// to `Float64` and the kernels tried again, so that a kernel declaring an
    /// exact `Float64` argument accepts an integer or decimal literal (e.g.,
    /// the `1` or `1.5` a Spark Connect client sends as `Int32`/`Decimal128`)
    /// without every such kernel having to handle the cast itself. Nothing else
    /// is coerced: a kernel that matches the arguments as given (including one
    /// that accepts any numeric type) sees them untouched, and when no kernel
    /// matches even after widening the error names the types as they were given.
    ///
    /// The widening is a planning-time decision: the `sedona.widen_numeric_args`
    /// analyzer rule in `sedona-query-planner` calls this and wraps the widened
    /// arguments in explicit casts, so the casts show in `EXPLAIN`, a literal is
    /// cast once when the plan is simplified rather than on every batch, and the
    /// kernel is invoked with the types it matched. [ScalarUDFImpl::return_type]
    /// resolves against the widened types too, so the schema DataFusion derives
    /// while building the plan (before any analyzer runs) already agrees with the
    /// analyzed plan.
    pub fn widened_arg_types(
        &self,
        args: &[SedonaType],
        scalars: &[Option<&ScalarValue>],
    ) -> Result<Option<Vec<SedonaType>>> {
        let (_, _, widened_args) = self.return_type_impl(args, scalars)?;
        Ok(widened_args)
    }

    /// Resolve the kernel and return type for a set of argument types
    ///
    /// The third element of the result is the argument types the kernel was
    /// matched against when they differ from `args`, as described in
    /// [Self::widened_arg_types], or `None` when a kernel matched `args` as given.
    fn return_type_impl(
        &self,
        args: &[SedonaType],
        scalars: &[Option<&ScalarValue>],
    ) -> Result<(&dyn SedonaScalarKernel, SedonaType, Option<Vec<SedonaType>>)> {
        if let Some((kernel, return_type)) = self.first_matching_kernel(args, scalars)? {
            return Ok((kernel, return_type, None));
        }

        if let Some(widened_args) = widen_numeric_args_to_float64(args) {
            // Kernels that compute a return type from a literal argument should see the
            // literal as the type they matched, so widen the scalars along with the types.
            let widened_scalars = scalars
                .iter()
                .zip(&widened_args)
                .map(|(scalar, widened_type)| match scalar {
                    Some(scalar) if scalar.data_type() != *widened_type.storage_type() => scalar
                        .cast_to_with_options(widened_type.storage_type(), &DEFAULT_CAST_OPTIONS)
                        .map(Some),
                    Some(scalar) => Ok(Some((*scalar).clone())),
                    None => Ok(None),
                })
                .collect::<Result<Vec<_>>>()?;
            let widened_scalar_refs = widened_scalars
                .iter()
                .map(|scalar| scalar.as_ref())
                .collect::<Vec<_>>();

            if let Some((kernel, return_type)) =
                self.first_matching_kernel(&widened_args, &widened_scalar_refs)?
            {
                return Ok((kernel, return_type, Some(widened_args)));
            }
        }

        let args_display = args
            .iter()
            .map(|arg| arg.logical_type_name())
            .collect::<Vec<_>>()
            .join(", ");

        not_impl_err!(
            "{}({args_display}): No kernel matching arguments",
            self.name
        )
    }
}

/// Whether an argument of this storage type is widened to `Float64` when no
/// kernel matches the arguments as given
///
/// Only integers and decimals qualify: both are representable as `Float64`
/// (decimals with more than about 15 significant digits lose precision in the
/// process). Floats are not widened, and extension types (geometry, geography,
/// raster) are never touched because they are not [SedonaType::Arrow].
fn widens_to_float64(data_type: &DataType) -> bool {
    data_type.is_integer()
        || matches!(
            data_type,
            DataType::Decimal32(_, _)
                | DataType::Decimal64(_, _)
                | DataType::Decimal128(_, _)
                | DataType::Decimal256(_, _)
        )
}

/// Replace every integer or decimal argument type with `Float64`
///
/// Returns `None` when there is nothing to widen so that the caller can skip
/// the second kernel lookup.
fn widen_numeric_args_to_float64(args: &[SedonaType]) -> Option<Vec<SedonaType>> {
    let mut widened_any = false;
    let widened = args
        .iter()
        .map(|arg| match arg {
            SedonaType::Arrow(data_type) if widens_to_float64(data_type) => {
                widened_any = true;
                SedonaType::Arrow(DataType::Float64)
            }
            other => other.clone(),
        })
        .collect();

    widened_any.then_some(widened)
}

impl ScalarUDFImpl for SedonaScalarUDF {
    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn documentation(&self) -> Option<&Documentation> {
        None
    }

    fn return_type(&self, _args: &[DataType]) -> Result<DataType> {
        sedona_internal_err!("Should not be called (use return_field_from_args())")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let arg_types = args
            .arg_fields
            .iter()
            .map(|field| SedonaType::from_storage_field(field))
            .collect::<Result<Vec<_>>>()?;
        let (_, out_type, _) = self.return_type_impl(&arg_types, args.scalar_arguments)?;
        Ok(Arc::new(out_type.to_storage_field("", true)?))
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        Ok(arg_types.to_vec())
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let arg_types = args
            .arg_fields
            .iter()
            .map(|field| SedonaType::from_storage_field(field))
            .collect::<Result<Vec<_>>>()?;

        let arg_scalars = args
            .args
            .iter()
            .map(|arg| {
                if let ColumnarValue::Scalar(scalar) = arg {
                    Some(scalar)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        let (kernel, return_type, widened_types) =
            self.return_type_impl(&arg_types, &arg_scalars)?;

        if let Some(widened_types) = widened_types {
            // The kernel matched only after widening, so the plan was expected to
            // cast these arguments before invoking this function. Running the kernel
            // on the unwidened values would hand it types it never agreed to.
            let args_display = arg_types
                .iter()
                .map(|arg| arg.logical_type_name())
                .collect::<Vec<_>>()
                .join(", ");
            let widened_display = widened_types
                .iter()
                .map(|arg| arg.logical_type_name())
                .collect::<Vec<_>>()
                .join(", ");
            return sedona_internal_err!(
                "{}({args_display}) needs its arguments cast to ({widened_display}) \
                 before it runs; the plan was not analyzed with the \
                 sedona.widen_numeric_args rule",
                self.name
            );
        }

        kernel.invoke_batch_from_args(
            &arg_types,
            &args.args,
            &return_type,
            args.number_rows,
            Some(&*args.config_options),
        )
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }
}

#[cfg(test)]
mod tests {

    use arrow_array::StringArray;
    use datafusion_common::{DFSchema, scalar::ScalarValue};
    use sedona_testing::testers::ScalarUdfTester;

    use datafusion_expr::{ExprSchemable, ScalarUDF, lit};
    use sedona_geometry::types::Edges;
    use sedona_schema::{crs::lnglat, datatypes::WKB_GEOMETRY};

    use super::*;

    #[test]
    fn metadata_defaults_empty_and_set_via_builder() {
        let udf = SedonaScalarUDF::new("u", vec![], Volatility::Immutable);
        assert!(udf.metadata().get("a_flag").is_none());

        let annotated = udf.with_metadata("a_flag", "true");
        assert_eq!(
            annotated.metadata().get("a_flag").map(String::as_str),
            Some("true")
        );
    }

    #[test]
    fn metadata_survives_with_aliases() {
        let udf = SedonaScalarUDF::new("u", vec![], Volatility::Immutable)
            .with_metadata("a_flag", "true")
            .with_aliases(vec!["u_alias".to_string()]);
        assert_eq!(
            udf.metadata().get("a_flag").map(String::as_str),
            Some("true")
        );
        assert_eq!(udf.aliases(), &["u_alias".to_string()]);
    }

    #[test]
    fn udf_empty() -> Result<()> {
        // UDF with no implementations
        let udf = SedonaScalarUDF::new("empty", vec![], Volatility::Immutable);
        assert_eq!(udf.name(), "empty");
        assert_eq!(udf.coerce_types(&[])?, vec![]);

        let tester = ScalarUdfTester::new(udf.into(), vec![]);

        let err = tester.return_type().unwrap_err();
        assert_eq!(err.message(), "empty(): No kernel matching arguments");

        let batch_err = tester.invoke_arrays(vec![]).unwrap_err();
        assert_eq!(batch_err.message(), "empty(): No kernel matching arguments");

        Ok(())
    }

    #[test]
    fn simple_udf() {
        // UDF with two implementations: one that matches any geometry and one that
        // matches a specific arrow type.
        let kernel_geo = SimpleSedonaScalarKernel::new_ref(
            ArgMatcher::new(
                vec![ArgMatcher::is_geometry_or_geography()],
                SedonaType::Arrow(DataType::Null),
            ),
            Arc::new(|_, _| Ok(ColumnarValue::Scalar(ScalarValue::Null))),
        );

        let kernel_arrow = SimpleSedonaScalarKernel::new_ref(
            ArgMatcher::new(
                vec![ArgMatcher::is_arrow(DataType::Boolean)],
                SedonaType::Arrow(DataType::Boolean),
            ),
            Arc::new(|_, _| Ok(ColumnarValue::Scalar(ScalarValue::Boolean(None)))),
        );

        let udf = SedonaScalarUDF::new(
            "simple_udf",
            vec![kernel_geo, kernel_arrow],
            Volatility::Immutable,
        );

        // Calling with a geo type should return a Null type
        let tester = ScalarUdfTester::new(udf.clone().into(), vec![WKB_GEOMETRY]);
        tester.assert_return_type(DataType::Null);
        assert_eq!(
            tester.invoke_scalar("POINT (0 1)").unwrap(),
            ScalarValue::Null
        );

        // Calling with a Boolean should result in a Boolean
        let tester = ScalarUdfTester::new(
            udf.clone().into(),
            vec![SedonaType::Arrow(DataType::Boolean)],
        );
        tester.assert_return_type(DataType::Boolean);
        assert_eq!(
            tester.invoke_scalar(true).unwrap(),
            ScalarValue::Boolean(None)
        );

        // Adding a new kernel should result in that kernel getting picked first
        let mut udf = udf.clone();
        udf.add_kernels(SimpleSedonaScalarKernel::new_ref(
            ArgMatcher::new(
                vec![ArgMatcher::is_arrow(DataType::Boolean)],
                SedonaType::Arrow(DataType::Utf8),
            ),
            Arc::new(|_, _| Ok(ColumnarValue::Scalar(ScalarValue::Utf8(None)))),
        ));

        // Now, calling with a Boolean should result in a Utf8
        let tester = ScalarUdfTester::new(
            udf.clone().into(),
            vec![SedonaType::Arrow(DataType::Boolean)],
        );
        tester.assert_return_type(DataType::Utf8);
    }

    #[test]
    fn widen_integer_and_decimal_args_to_float64_kernel() {
        // A kernel that accepts exactly Float64 and echoes its argument back
        let float64_kernel = SimpleSedonaScalarKernel::new_ref(
            ArgMatcher::new(
                vec![ArgMatcher::is_arrow(DataType::Float64)],
                SedonaType::Arrow(DataType::Float64),
            ),
            Arc::new(|arg_types, args| {
                assert_eq!(arg_types, &[SedonaType::Arrow(DataType::Float64)]);
                Ok(args[0].clone())
            }),
        );
        let udf = SedonaScalarUDF::from_impl("float64_only", float64_kernel);

        // Decimal128 and Int32 arguments resolve the Float64 kernel's return type,
        // and the plan is told to cast them to Float64 first
        for arg_type in [DataType::Decimal128(10, 2), DataType::Int32] {
            let tester =
                ScalarUdfTester::new(udf.clone().into(), vec![SedonaType::Arrow(arg_type)]);
            tester.assert_return_type(DataType::Float64);
        }
        assert_eq!(
            udf.widened_arg_types(&[SedonaType::Arrow(DataType::Decimal128(10, 2))], &[None])
                .unwrap(),
            Some(vec![SedonaType::Arrow(DataType::Float64)])
        );
        assert_eq!(
            udf.widened_arg_types(&[SedonaType::Arrow(DataType::Int32)], &[None])
                .unwrap(),
            Some(vec![SedonaType::Arrow(DataType::Float64)])
        );

        // Invoking on the unwidened values (a plan the analyzer rule did not see) is
        // refused rather than handing the kernel an Int32 it never matched
        let tester =
            ScalarUdfTester::new(udf.clone().into(), vec![SedonaType::Arrow(DataType::Int32)]);
        let err = tester.invoke_scalar(7_i32).unwrap_err();
        assert!(
            err.to_string().contains(
                "float64_only(int32) needs its arguments cast to (float64) before it runs; \
                 the plan was not analyzed with the sedona.widen_numeric_args rule"
            ),
            "{err}"
        );

        // Float64 passed as-is needs no widening and runs
        let tester = ScalarUdfTester::new(
            udf.clone().into(),
            vec![SedonaType::Arrow(DataType::Float64)],
        );
        assert_eq!(
            udf.widened_arg_types(&[SedonaType::Arrow(DataType::Float64)], &[None])
                .unwrap(),
            None
        );
        assert_eq!(
            tester.invoke_scalar(2.5).unwrap(),
            ScalarValue::Float64(Some(2.5))
        );
    }

    #[test]
    fn widen_only_numeric_args_next_to_geometry() {
        // Geometry arguments carry extension metadata that must survive the retry
        let kernel = SimpleSedonaScalarKernel::new_ref(
            ArgMatcher::new(
                vec![
                    ArgMatcher::is_geometry(),
                    ArgMatcher::is_arrow(DataType::Float64),
                ],
                SedonaType::Arrow(DataType::Float64),
            ),
            Arc::new(|_, _| unreachable!("Should not be executed")),
        );
        let udf = SedonaScalarUDF::from_impl("geom_and_float64", kernel);

        let tester = ScalarUdfTester::new(
            udf.clone().into(),
            vec![WKB_GEOMETRY, SedonaType::Arrow(DataType::Int64)],
        );
        tester.assert_return_type(DataType::Float64);
        assert_eq!(
            udf.widened_arg_types(
                &[WKB_GEOMETRY, SedonaType::Arrow(DataType::Int64)],
                &[None, None]
            )
            .unwrap(),
            Some(vec![WKB_GEOMETRY, SedonaType::Arrow(DataType::Float64)])
        );
    }

    #[test]
    fn widen_literal_seen_by_a_return_type_that_depends_on_it() {
        // A kernel whose return type is read off a Float64 literal sees the literal
        // as Float64 when it was given as an integer
        let kernel = SimpleSedonaScalarKernel::new_ref(
            ArgMatcher::new(
                vec![ArgMatcher::is_arrow(DataType::Float64)],
                SedonaType::Arrow(DataType::Float64),
            ),
            Arc::new(|_, _| unreachable!("Should not be executed")),
        );

        #[derive(Debug)]
        struct ReturnsUtf8ForLiteralOne(ScalarKernelRef);

        impl SedonaScalarKernel for ReturnsUtf8ForLiteralOne {
            fn return_type_from_args_and_scalars(
                &self,
                args: &[SedonaType],
                scalars: &[Option<&ScalarValue>],
            ) -> Result<Option<SedonaType>> {
                Ok(self.0.return_type(args)?.map(|return_type| {
                    if scalars[0] == Some(&ScalarValue::Float64(Some(1.0))) {
                        SedonaType::Arrow(DataType::Utf8)
                    } else {
                        return_type
                    }
                }))
            }

            fn return_type(&self, _args: &[SedonaType]) -> Result<Option<SedonaType>> {
                unreachable!("return_type_from_args_and_scalars is overridden")
            }

            fn invoke_batch(
                &self,
                _arg_types: &[SedonaType],
                _args: &[ColumnarValue],
            ) -> Result<ColumnarValue> {
                unreachable!("Should not be executed")
            }
        }

        let udf = SedonaScalarUDF::from_impl("literal_aware", ReturnsUtf8ForLiteralOne(kernel));
        let tester = ScalarUdfTester::new(udf.into(), vec![SedonaType::Arrow(DataType::Int32)]);
        assert_eq!(
            tester.return_type_with_scalar(Some(1_i32)).unwrap(),
            SedonaType::Arrow(DataType::Utf8)
        );
        assert_eq!(
            tester.return_type_with_scalar(Some(2_i32)).unwrap(),
            SedonaType::Arrow(DataType::Float64)
        );
    }

    #[test]
    fn no_widening_when_a_kernel_already_matches() {
        // A kernel accepting any numeric type must see the Decimal exactly as given
        let numeric_kernel = SimpleSedonaScalarKernel::new_ref(
            ArgMatcher::new(
                vec![ArgMatcher::is_numeric()],
                SedonaType::Arrow(DataType::Utf8),
            ),
            Arc::new(|arg_types, args| {
                assert_eq!(arg_types, &[SedonaType::Arrow(DataType::Decimal128(10, 2))]);
                assert_eq!(args[0].data_type(), DataType::Decimal128(10, 2));
                Ok(ColumnarValue::Scalar(ScalarValue::Utf8(Some(
                    "numeric".to_string(),
                ))))
            }),
        );
        let udf = SedonaScalarUDF::from_impl("numeric_any", numeric_kernel);

        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![SedonaType::Arrow(DataType::Decimal128(10, 2))],
        );
        tester.assert_return_type(DataType::Utf8);
        assert_eq!(
            tester
                .invoke(vec![ColumnarValue::Scalar(ScalarValue::Decimal128(
                    Some(150),
                    10,
                    2
                ))])
                .unwrap()
                .to_array(1)
                .unwrap()
                .as_ref(),
            &StringArray::from(vec!["numeric"])
        );

        // With both an integer kernel and a Float64 kernel, an Int32 argument still
        // picks the integer kernel because it matches without widening
        let mut udf = SedonaScalarUDF::from_impl(
            "int_or_float64",
            SimpleSedonaScalarKernel::new_ref(
                ArgMatcher::new(
                    vec![ArgMatcher::is_arrow(DataType::Float64)],
                    SedonaType::Arrow(DataType::Float64),
                ),
                Arc::new(|_, _| unreachable!("Float64 kernel should not be picked")),
            ),
        );
        udf.add_kernels(SimpleSedonaScalarKernel::new_ref(
            ArgMatcher::new(
                vec![ArgMatcher::is_integer()],
                SedonaType::Arrow(DataType::Int32),
            ),
            Arc::new(|_, args| Ok(args[0].clone())),
        ));
        let tester = ScalarUdfTester::new(udf.into(), vec![SedonaType::Arrow(DataType::Int32)]);
        tester.assert_return_type(DataType::Int32);
        assert_eq!(
            tester.invoke_scalar(7_i32).unwrap(),
            ScalarValue::Int32(Some(7))
        );
    }

    #[test]
    fn no_widening_of_non_numeric_args() {
        let float64_kernel = SimpleSedonaScalarKernel::new_ref(
            ArgMatcher::new(
                vec![ArgMatcher::is_arrow(DataType::Float64)],
                SedonaType::Arrow(DataType::Float64),
            ),
            Arc::new(|_, _| unreachable!("Should not be executed")),
        );
        let udf = SedonaScalarUDF::from_impl("float64_only", float64_kernel);

        // Utf8 is left alone and the original error stands
        let tester =
            ScalarUdfTester::new(udf.clone().into(), vec![SedonaType::Arrow(DataType::Utf8)]);
        let err = tester.return_type().unwrap_err();
        assert_eq!(
            err.message(),
            "float64_only(utf8): No kernel matching arguments"
        );

        // Float32 is not widened either
        let tester = ScalarUdfTester::new(
            udf.clone().into(),
            vec![SedonaType::Arrow(DataType::Float32)],
        );
        let err = tester.return_type().unwrap_err();
        assert_eq!(
            err.message(),
            "float64_only(float32): No kernel matching arguments"
        );

        // A widened argument next to one that still doesn't match reports the
        // types as given
        let udf = SedonaScalarUDF::from_impl(
            "two_float64",
            SimpleSedonaScalarKernel::new_ref(
                ArgMatcher::new(
                    vec![
                        ArgMatcher::is_arrow(DataType::Float64),
                        ArgMatcher::is_arrow(DataType::Float64),
                    ],
                    SedonaType::Arrow(DataType::Float64),
                ),
                Arc::new(|_, _| unreachable!("Should not be executed")),
            ),
        );
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![
                SedonaType::Arrow(DataType::Int32),
                SedonaType::Arrow(DataType::Utf8),
            ],
        );
        let err = tester.return_type().unwrap_err();
        assert_eq!(
            err.message(),
            "two_float64(int32, utf8): No kernel matching arguments"
        );
    }

    #[test]
    fn crs_propagation() {
        let geom_lnglat = SedonaType::Wkb(Edges::Planar, lnglat());
        let predicate_stub_impl = SimpleSedonaScalarKernel::new_ref(
            ArgMatcher::new(
                vec![ArgMatcher::is_geometry(), ArgMatcher::is_geometry()],
                SedonaType::Arrow(DataType::Boolean),
            ),
            Arc::new(|_arg_types, _args| unreachable!("Should not be executed")),
        );
        let predicate_stub = SedonaScalarUDF::from_impl("foofy", predicate_stub_impl);

        // None CRS to None CRS is OK
        let tester = ScalarUdfTester::new(
            predicate_stub.clone().into(),
            vec![WKB_GEOMETRY, WKB_GEOMETRY],
        );
        tester.assert_return_type(DataType::Boolean);

        // lnglat + lnglat is OK
        let tester = ScalarUdfTester::new(
            predicate_stub.clone().into(),
            vec![geom_lnglat.clone(), geom_lnglat.clone()],
        );
        tester.assert_return_type(DataType::Boolean);

        // Non-equal CRSes should error
        let tester = ScalarUdfTester::new(
            predicate_stub.clone().into(),
            vec![WKB_GEOMETRY, geom_lnglat.clone()],
        );
        let err = tester.return_type().unwrap_err();
        assert!(err.message().starts_with("Mismatched CRS arguments"));

        // When geometry is output, it should match the crses of the inputs
        let geom_out_impl = SimpleSedonaScalarKernel::new_ref(
            ArgMatcher::new(
                vec![ArgMatcher::is_geometry(), ArgMatcher::is_geometry()],
                WKB_GEOMETRY,
            ),
            Arc::new(|_arg_types, args| Ok(args[0].clone())),
        );
        let geom_out_stub = SedonaScalarUDF::from_impl("foofy", geom_out_impl);

        let tester = ScalarUdfTester::new(
            geom_out_stub.clone().into(),
            vec![geom_lnglat.clone(), geom_lnglat.clone()],
        );
        tester.assert_return_type(geom_lnglat.clone());
    }

    #[test]
    fn return_type_from_scalar_arg() {
        let udf: ScalarUDF = SedonaScalarUDF::from_impl("simple_cast", SimpleCast {}).into();
        let call = udf.call(vec![lit(10), lit("float32")]);
        let schema = DFSchema::empty();
        let call_field = call.to_field(&schema).unwrap();
        assert_eq!(
            (call_field.1.data_type(), call_field.1.is_nullable()),
            (&DataType::Float32, true)
        );
    }

    #[derive(Debug)]
    struct SimpleCast {}

    impl SimpleCast {
        fn parse_type(val: &ColumnarValue) -> Result<SedonaType> {
            if let ColumnarValue::Scalar(ScalarValue::Utf8(Some(scalar_arg1))) = val {
                match scalar_arg1.as_str() {
                    "float32" => return Ok(SedonaType::Arrow(DataType::Float32)),
                    "float64" => return Ok(SedonaType::Arrow(DataType::Float64)),
                    _ => {}
                }
            }

            sedona_internal_err!("unrecognized target value")
        }
    }

    impl SedonaScalarKernel for SimpleCast {
        fn return_type(&self, _args: &[SedonaType]) -> Result<Option<SedonaType>> {
            sedona_internal_err!("Should not be called")
        }

        fn return_type_from_args_and_scalars(
            &self,
            _args: &[SedonaType],
            scalar_args: &[Option<&ScalarValue>],
        ) -> Result<Option<SedonaType>> {
            let out_type = Self::parse_type(&ColumnarValue::Scalar(
                scalar_args[1].cloned().expect("arg1 as a scalar in test"),
            ))?;

            Ok(Some(out_type))
        }

        fn invoke_batch(
            &self,
            _arg_types: &[SedonaType],
            args: &[ColumnarValue],
        ) -> Result<ColumnarValue> {
            let out_type = Self::parse_type(&args[1])?;
            args[0].cast_to(out_type.storage_type(), None)
        }
    }
}
