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
use std::iter::zip;
use std::sync::Arc;
use std::{any::Any, fmt::Debug};

use arrow_schema::{DataType, Field, FieldRef};
use datafusion_common::{not_impl_err, plan_err, Result, ScalarValue};
use datafusion_expr::{
    ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature,
    Volatility,
};
use sedona_common::sedona_internal_err;
use sedona_schema::datatypes::{Edges, SedonaType};

pub type ScalarKernelRef = Arc<dyn SedonaScalarKernel + Send + Sync>;

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
    documentation: Option<Documentation>,
    aliases: Vec<String>,
}

/// User-defined function implementation
///
/// A `SedonaScalarUdf` is comprised of one or more kernels, to which it dispatches
/// the first whose return_type returns `Some()`. Whereas a SeondaScalarUdf represents
/// a logical operation (e.g., ST_Intersects()), a kernel wraps the logic around a specific
/// implementation.
pub trait SedonaScalarKernel: Debug {
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
}

/// Helper to match arguments and compute return types
#[derive(Debug)]
pub struct ArgMatcher {
    matchers: Vec<Arc<dyn TypeMatcher + Send + Sync>>,
    out_type: SedonaType,
}

impl ArgMatcher {
    /// Create a new ArgMatcher
    pub fn new(matchers: Vec<Arc<dyn TypeMatcher + Send + Sync>>, out_type: SedonaType) -> Self {
        Self { matchers, out_type }
    }

    /// Calculate a return type given input types
    ///
    /// Returns Some(physical_type) if this kernel applies to the input types or
    /// None otherwise. This function also checks that all input arguments have
    /// compatible CRSes and if so, applies the CRS to the output type.
    pub fn match_args(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        if !self.matches(args) {
            return Ok(None);
        }

        let geometry_arg_crses = args
            .iter()
            .filter(|arg_type| IsGeometryOrGeography {}.match_type(arg_type))
            .map(|arg_type| match arg_type {
                SedonaType::Wkb(_, crs) | SedonaType::WkbView(_, crs) => crs.clone(),
                _ => None,
            })
            .collect::<Vec<_>>();

        if geometry_arg_crses.is_empty() {
            return Ok(Some(self.out_type.clone()));
        }

        let out_crs = geometry_arg_crses[0].clone();
        for this_crs in geometry_arg_crses.into_iter().skip(1) {
            if out_crs != this_crs {
                let hint = "Use ST_Transform() or ST_SetSRID() to ensure arguments are compatible.";

                return match (out_crs, this_crs) {
                    (None, Some(rhs_crs)) => {
                        plan_err!("Mismatched CRS arguments: None vs {rhs_crs}\n{hint}")
                    }
                    (Some(lhs_crs), None) => {
                        plan_err!("Mismatched CRS arguments: {lhs_crs} vs None\n{hint}")
                    }
                    (Some(lhs_crs), Some(rhs_crs)) => {
                        plan_err!("Mismatched CRS arguments: {lhs_crs} vs {rhs_crs}\n{hint}")
                    }
                    _ => sedona_internal_err!("None vs. None should be considered equal"),
                };
            }
        }

        match &self.out_type {
            SedonaType::Wkb(edges, _) => Ok(Some(SedonaType::Wkb(*edges, out_crs))),
            SedonaType::WkbView(edges, _) => Ok(Some(SedonaType::WkbView(*edges, out_crs))),
            _ => Ok(Some(self.out_type.clone())),
        }
    }

    /// Check for an input type match
    ///
    /// Returns true if args applies to the input types.
    pub fn matches(&self, args: &[SedonaType]) -> bool {
        if args.len() > self.matchers.len() {
            return false;
        }

        let matcher_iter = self.matchers.iter();
        let mut arg_iter = args.iter().peekable();

        for matcher in matcher_iter {
            if let Some(arg) = arg_iter.peek() {
                if arg == &&SedonaType::Arrow(DataType::Null) || matcher.match_type(arg) {
                    arg_iter.next(); // Consume the argument
                    continue; // Move to the next matcher
                } else if matcher.is_optional() {
                    continue; // Skip the optional matcher
                } else {
                    return false; // Non-optional matcher failed
                }
            } else if matcher.is_optional() {
                continue; // Skip remaining optional matchers
            } else {
                return false; // Non-optional matcher failed with no arguments left
            }
        }

        // Ensure all arguments are consumed
        arg_iter.next().is_none()
    }

    /// Matches any argument
    pub fn is_any() -> Arc<dyn TypeMatcher + Send + Sync> {
        Arc::new(IsAny {})
    }

    /// Matches the given Arrow type using PartialEq
    pub fn is_arrow(data_type: DataType) -> Arc<dyn TypeMatcher + Send + Sync> {
        Arc::new(IsExact {
            exact_type: SedonaType::Arrow(data_type),
        })
    }

    /// Matches any geography or geometry argument without considering Crs
    pub fn is_geometry_or_geography() -> Arc<dyn TypeMatcher + Send + Sync> {
        Arc::new(IsGeometryOrGeography {})
    }

    /// Matches any geometry argument without considering Crs
    pub fn is_geometry() -> Arc<dyn TypeMatcher + Send + Sync> {
        Arc::new(IsGeometry {})
    }

    /// Matches any geography argument without considering Crs
    pub fn is_geography() -> Arc<dyn TypeMatcher + Send + Sync> {
        Arc::new(IsGeography {})
    }

    /// Matches any numeric argument
    pub fn is_numeric() -> Arc<dyn TypeMatcher + Send + Sync> {
        Arc::new(IsNumeric {})
    }

    /// Matches any string argument
    pub fn is_string() -> Arc<dyn TypeMatcher + Send + Sync> {
        Arc::new(IsString {})
    }

    /// Matches any binary argument
    pub fn is_binary() -> Arc<dyn TypeMatcher + Send + Sync> {
        Arc::new(IsBinary {})
    }

    /// Matches any boolean argument
    pub fn is_boolean() -> Arc<dyn TypeMatcher + Send + Sync> {
        Arc::new(IsBoolean {})
    }

    /// Matches any argument that is optional
    pub fn is_optional(
        matcher: Arc<dyn TypeMatcher + Send + Sync>,
    ) -> Arc<dyn TypeMatcher + Send + Sync> {
        Arc::new(OptionalMatcher { inner: matcher })
    }
}

pub trait TypeMatcher: Debug {
    fn match_type(&self, arg: &SedonaType) -> bool;
    fn is_optional(&self) -> bool {
        false
    }
}

#[derive(Debug)]
struct IsAny;

impl TypeMatcher for IsAny {
    fn match_type(&self, _arg: &SedonaType) -> bool {
        true
    }
}

#[derive(Debug)]
struct IsExact {
    exact_type: SedonaType,
}

impl TypeMatcher for IsExact {
    fn match_type(&self, arg: &SedonaType) -> bool {
        self.exact_type.match_signature(arg)
    }
}

#[derive(Debug)]
struct OptionalMatcher {
    inner: Arc<dyn TypeMatcher + Send + Sync>,
}

impl TypeMatcher for OptionalMatcher {
    fn match_type(&self, arg: &SedonaType) -> bool {
        self.inner.match_type(arg)
    }

    fn is_optional(&self) -> bool {
        true
    }
}

#[derive(Debug)]
struct IsGeometryOrGeography {}

impl TypeMatcher for IsGeometryOrGeography {
    fn match_type(&self, arg: &SedonaType) -> bool {
        matches!(arg, SedonaType::Wkb(_, _) | SedonaType::WkbView(_, _))
    }
}

#[derive(Debug)]
struct IsGeometry {}

impl TypeMatcher for IsGeometry {
    fn match_type(&self, arg: &SedonaType) -> bool {
        match arg {
            SedonaType::Wkb(edges, _) | SedonaType::WkbView(edges, _) => {
                matches!(edges, Edges::Planar)
            }
            _ => false,
        }
    }
}

#[derive(Debug)]
struct IsGeography {}

impl TypeMatcher for IsGeography {
    fn match_type(&self, arg: &SedonaType) -> bool {
        match arg {
            SedonaType::Wkb(edges, _) | SedonaType::WkbView(edges, _) => {
                matches!(edges, Edges::Spherical)
            }
            _ => false,
        }
    }
}

#[derive(Debug)]
struct IsNumeric {}

impl TypeMatcher for IsNumeric {
    fn match_type(&self, arg: &SedonaType) -> bool {
        match arg {
            SedonaType::Arrow(data_type) => data_type.is_numeric(),
            _ => false,
        }
    }
}

#[derive(Debug)]
struct IsString {}

impl TypeMatcher for IsString {
    fn match_type(&self, arg: &SedonaType) -> bool {
        match arg {
            SedonaType::Arrow(data_type) => {
                matches!(
                    data_type,
                    DataType::Utf8 | DataType::Utf8View | DataType::LargeUtf8
                )
            }
            _ => false,
        }
    }
}

#[derive(Debug)]
struct IsBinary {}

impl TypeMatcher for IsBinary {
    fn match_type(&self, arg: &SedonaType) -> bool {
        match arg {
            SedonaType::Arrow(data_type) => {
                matches!(data_type, DataType::Binary | DataType::BinaryView)
            }
            _ => false,
        }
    }
}

#[derive(Debug)]
struct IsBoolean {}

impl TypeMatcher for IsBoolean {
    fn match_type(&self, arg: &SedonaType) -> bool {
        match arg {
            SedonaType::Arrow(data_type) => {
                matches!(data_type, DataType::Boolean)
            }
            _ => false,
        }
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
        documentation: Option<Documentation>,
    ) -> SedonaScalarUDF {
        let signature = Signature::user_defined(volatility);
        Self {
            name: name.to_string(),
            signature,
            kernels,
            documentation,
            aliases: vec![],
        }
    }

    pub fn new_with_aliases(
        name: &str,
        kernels: Vec<ScalarKernelRef>,
        volatility: Volatility,
        documentation: Option<Documentation>,
        aliases: Vec<String>,
    ) -> SedonaScalarUDF {
        let signature = Signature::user_defined(volatility);
        Self {
            name: name.to_string(),
            signature,
            kernels,
            documentation,
            aliases,
        }
    }

    /// Create a new stub function
    ///
    /// Creates a new function that calculates a return type but fails when invoked with
    /// arguments. This is useful to create stub functions when it is expected that the
    /// actual functionality will be registered from one or more independent crates
    /// (e.g., ST_Intersects(), which may be implemented in sedona-geo or sedona-geography).
    pub fn new_stub(
        name: &str,
        arg_matcher: ArgMatcher,
        volatility: Volatility,
        documentation: Option<Documentation>,
    ) -> Self {
        let name_string = name.to_string();
        let stub_kernel = SimpleSedonaScalarKernel::new_ref(
            arg_matcher,
            Arc::new(move |arg_types, _| {
                not_impl_err!("Implementation for {name_string}({arg_types:?}) was not registered")
            }),
        );

        Self::new(name, vec![stub_kernel], volatility, documentation)
    }

    /// Create a SedonaScalarUDF from a single kernel
    ///
    /// This constructor creates a [Volatility::Immutable] function with no documentation
    /// consisting of only the implementation provided.
    pub fn from_kernel(name: &str, kernel: ScalarKernelRef) -> SedonaScalarUDF {
        Self::new(name, vec![kernel], Volatility::Immutable, None)
    }

    pub fn invoke_batch(
        &self,
        args: &[ColumnarValue],
        number_rows: usize,
    ) -> Result<ColumnarValue> {
        let arg_types: Vec<_> = args.iter().map(|arg| arg.data_type()).collect();
        let return_type = self.return_type(&arg_types)?;
        let arg_fields: Vec<_> = arg_types
            .into_iter()
            .map(|data_type| Arc::new(Field::new("", data_type, true)))
            .collect();

        let args = ScalarFunctionArgs {
            args: args.to_vec(),
            arg_fields,
            number_rows,
            return_field: Arc::new(Field::new("", return_type, true)),
        };

        self.invoke_with_args(args)
    }

    /// Add a new kernel to a Scalar UDF
    ///
    /// Because kernels are resolved in reverse order, the new kernel will take
    /// precedence over any previously added kernels that apply to the same types.
    pub fn add_kernel(&mut self, kernel: ScalarKernelRef) {
        self.kernels.push(kernel);
    }

    fn physical_types(args: &[DataType]) -> Result<Vec<SedonaType>> {
        args.iter().map(SedonaType::from_data_type).collect()
    }

    fn return_type_impl(
        &self,
        args: &[SedonaType],
        scalars: &[Option<&ScalarValue>],
    ) -> Result<(&dyn SedonaScalarKernel, SedonaType)> {
        // Resolve kernels in reverse so that more recently added ones are resolved first
        for kernel in self.kernels.iter().rev() {
            if let Some(return_type) = kernel.return_type_from_args_and_scalars(args, scalars)? {
                return Ok((kernel.as_ref(), return_type));
            }
        }

        not_impl_err!("{}({:?}): No kernel matching arguments", self.name, args)
    }
}

impl ScalarUDFImpl for SedonaScalarUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.documentation.as_ref()
    }

    fn return_type(&self, args: &[DataType]) -> Result<DataType> {
        let arg_types = Self::physical_types(args)?;
        let scalars = vec![None; args.len()];
        let (_, out_type) = self.return_type_impl(&arg_types, &scalars)?;
        Ok(out_type.data_type_maybe_deprecated())
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let arg_data_types: Vec<DataType> = args
            .arg_fields
            .iter()
            .map(|arg| arg.data_type().clone())
            .collect();
        let arg_types = Self::physical_types(&arg_data_types)?;
        let (_, out_type) = self.return_type_impl(&arg_types, args.scalar_arguments)?;
        Ok(Field::new("", out_type.data_type_maybe_deprecated(), true).into())
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        Ok(arg_types.to_vec())
    }

    fn invoke_with_args(&self, args: datafusion_expr::ScalarFunctionArgs) -> Result<ColumnarValue> {
        let arg_types: Vec<DataType> = args.args.iter().map(|arg| arg.data_type()).collect();
        let arg_physical_types = Self::physical_types(&arg_types)?;
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
        let (kernel, out_type) = self.return_type_impl(&arg_physical_types, &arg_scalars)?;
        let args_unwrapped: Result<Vec<ColumnarValue>, _> = zip(&arg_physical_types, &args.args)
            .map(|(a, b)| a.unwrap_arg(b))
            .collect();
        let result = kernel.invoke_batch(&arg_physical_types, &args_unwrapped?)?;
        out_type.wrap_arg(&result)
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }
}

#[cfg(test)]
mod tests {
    use datafusion_common::{scalar::ScalarValue, DFSchema};

    use datafusion_expr::{lit, ExprSchemable, ScalarUDF};
    use sedona_schema::{
        crs::lnglat,
        datatypes::{WKB_GEOGRAPHY, WKB_GEOMETRY},
    };

    use super::*;

    #[test]
    fn matchers() {
        assert!(ArgMatcher::is_arrow(DataType::Null).match_type(&SedonaType::Arrow(DataType::Null)));

        assert!(ArgMatcher::is_geometry_or_geography().match_type(&WKB_GEOMETRY));
        assert!(ArgMatcher::is_geometry_or_geography().match_type(&WKB_GEOGRAPHY));
        assert!(!ArgMatcher::is_geometry_or_geography()
            .match_type(&SedonaType::Arrow(DataType::Binary)));

        assert!(ArgMatcher::is_geometry().match_type(&WKB_GEOMETRY));
        assert!(!ArgMatcher::is_geometry().match_type(&WKB_GEOGRAPHY));

        assert!(ArgMatcher::is_geography().match_type(&WKB_GEOGRAPHY));
        assert!(!ArgMatcher::is_geography().match_type(&WKB_GEOMETRY));

        assert!(ArgMatcher::is_numeric().match_type(&SedonaType::Arrow(DataType::Int32)));
        assert!(ArgMatcher::is_numeric().match_type(&SedonaType::Arrow(DataType::Float64)));

        assert!(ArgMatcher::is_string().match_type(&SedonaType::Arrow(DataType::Utf8)));
        assert!(ArgMatcher::is_string().match_type(&SedonaType::Arrow(DataType::Utf8View)));
        assert!(ArgMatcher::is_string().match_type(&SedonaType::Arrow(DataType::LargeUtf8)));
        assert!(!ArgMatcher::is_string().match_type(&SedonaType::Arrow(DataType::Binary)));

        assert!(ArgMatcher::is_binary().match_type(&SedonaType::Arrow(DataType::Binary)));
        assert!(ArgMatcher::is_binary().match_type(&SedonaType::Arrow(DataType::BinaryView)));
        assert!(!ArgMatcher::is_binary().match_type(&SedonaType::Arrow(DataType::Utf8)));

        assert!(ArgMatcher::is_boolean().match_type(&SedonaType::Arrow(DataType::Boolean)));
        assert!(!ArgMatcher::is_boolean().match_type(&SedonaType::Arrow(DataType::Int32)));
    }

    #[test]
    fn optional_matcher() {
        let matcher = ArgMatcher::new(
            vec![
                ArgMatcher::is_geometry(),
                ArgMatcher::is_optional(ArgMatcher::is_boolean()),
                ArgMatcher::is_optional(ArgMatcher::is_numeric()),
            ],
            SedonaType::Arrow(DataType::Null),
        );

        // Match with all args present and matching
        assert!(matcher.matches(&[
            WKB_GEOMETRY,
            SedonaType::Arrow(DataType::Boolean),
            SedonaType::Arrow(DataType::Int32)
        ]));

        // Match when first argument present, second is None
        assert!(matcher.matches(&[WKB_GEOMETRY]));

        // Match when skip an optional arg
        assert!(matcher.matches(&[WKB_GEOMETRY, SedonaType::Arrow(DataType::Int32)]));

        // No match when first is None, second is present
        assert!(!matcher.matches(&[SedonaType::Arrow(DataType::Boolean)]));

        // No match when second argument is incorrect type
        assert!(!matcher.matches(&[WKB_GEOMETRY, WKB_GEOMETRY]));

        // No match when first argument is incorrect type
        assert!(!matcher.matches(&[
            SedonaType::Arrow(DataType::Boolean),
            SedonaType::Arrow(DataType::Boolean)
        ]));

        // No match when too many arguments
        assert!(!matcher.matches(&[
            WKB_GEOGRAPHY,
            SedonaType::Arrow(DataType::Boolean),
            SedonaType::Arrow(DataType::Int32),
            SedonaType::Arrow(DataType::Int32)
        ]));
    }

    #[test]
    fn arg_matcher_matches_null() {
        for type_matcher in [
            ArgMatcher::is_arrow(DataType::Null),
            ArgMatcher::is_arrow(DataType::Float32),
            ArgMatcher::is_geometry_or_geography(),
            ArgMatcher::is_geometry(),
            ArgMatcher::is_geography(),
            ArgMatcher::is_numeric(),
            ArgMatcher::is_string(),
            ArgMatcher::is_binary(),
            ArgMatcher::is_boolean(),
            ArgMatcher::is_optional(ArgMatcher::is_numeric()),
        ] {
            let matcher = ArgMatcher::new(vec![type_matcher], SedonaType::Arrow(DataType::Null));
            assert!(matcher.matches(&[SedonaType::Arrow(DataType::Null)]));
        }
    }

    #[test]
    fn udf_empty() -> Result<()> {
        // UDF with no implementations
        let udf = SedonaScalarUDF::new("empty", vec![], Volatility::Immutable, None);
        assert_eq!(udf.name(), "empty");
        let err = udf.return_type(&[]).unwrap_err();
        assert_eq!(err.message(), "empty([]): No kernel matching arguments");

        assert_eq!(udf.coerce_types(&[])?, vec![]);

        let batch_err = udf.invoke_batch(&[], 5).unwrap_err();
        assert_eq!(
            batch_err.message(),
            "empty([]): No kernel matching arguments"
        );

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
            None,
        );

        assert_eq!(udf.name(), "simple_udf");

        // Calling with a geo type should return a Null type
        let wkb_arrow = WKB_GEOMETRY.data_type_maybe_deprecated();
        let wkb_dummy_val = WKB_GEOMETRY
            .wrap_arg(&ColumnarValue::Scalar(ScalarValue::Binary(None)))
            .unwrap();

        assert_eq!(
            udf.return_type(std::slice::from_ref(&wkb_arrow)).unwrap(),
            DataType::Null
        );
        assert_eq!(
            udf.coerce_types(std::slice::from_ref(&wkb_arrow)).unwrap(),
            vec![wkb_arrow.clone()]
        );

        if let ColumnarValue::Scalar(scalar) = udf.invoke_batch(&[wkb_dummy_val], 5).unwrap() {
            assert_eq!(scalar, ScalarValue::Null);
        } else {
            panic!("Unexpected batch result");
        }

        // Calling with a Boolean should result in a Boolean
        let bool_arrow = DataType::Boolean;
        let bool_dummy_val = ColumnarValue::Scalar(ScalarValue::Boolean(None));
        assert_eq!(
            udf.coerce_types(std::slice::from_ref(&bool_arrow)).unwrap(),
            vec![bool_arrow.clone()]
        );

        assert_eq!(
            udf.return_type(std::slice::from_ref(&bool_arrow)).unwrap(),
            DataType::Boolean
        );

        if let ColumnarValue::Scalar(scalar) = udf.invoke_batch(&[bool_dummy_val], 5).unwrap() {
            assert_eq!(scalar, ScalarValue::Boolean(None));
        } else {
            panic!("Unexpected batch result");
        }

        // Calling with something where no types match should error
        let batch_err = udf.invoke_batch(&[], 5).unwrap_err();
        assert_eq!(
            batch_err.message(),
            "simple_udf([]): No kernel matching arguments"
        );

        // Adding a new kernel should result in that kernel getting picked first
        let mut udf = udf.clone();
        udf.add_kernel(SimpleSedonaScalarKernel::new_ref(
            ArgMatcher::new(
                vec![ArgMatcher::is_arrow(DataType::Boolean)],
                SedonaType::Arrow(DataType::Utf8),
            ),
            Arc::new(|_, _| Ok(ColumnarValue::Scalar(ScalarValue::Utf8(None)))),
        ));

        // Now, calling with a Boolean should result in a Utf8
        assert_eq!(
            udf.return_type(std::slice::from_ref(&bool_arrow)).unwrap(),
            DataType::Utf8
        );
    }

    #[test]
    fn stub() {
        let stub = SedonaScalarUDF::new_stub(
            "stubby",
            ArgMatcher::new(vec![], SedonaType::Arrow(DataType::Boolean)),
            Volatility::Immutable,
            None,
        );

        assert_eq!(stub.return_type(&[]).unwrap(), DataType::Boolean);
        let err = stub.invoke_batch(&[], 1).unwrap_err();
        assert_eq!(
            err.message(),
            "Implementation for stubby([]) was not registered"
        );
    }

    #[test]
    fn crs_propagation() {
        let geom_lnglat = SedonaType::Wkb(Edges::Planar, lnglat()).data_type_maybe_deprecated();

        let predicate_stub = SedonaScalarUDF::new_stub(
            "stubby",
            ArgMatcher::new(
                vec![ArgMatcher::is_geometry(), ArgMatcher::is_geometry()],
                SedonaType::Arrow(DataType::Boolean),
            ),
            Volatility::Immutable,
            None,
        );

        // None CRS to None CRS is OK
        assert_eq!(
            predicate_stub
                .return_type(&[WKB_GEOMETRY.data_type_maybe_deprecated(), WKB_GEOMETRY.data_type_maybe_deprecated()])
                .unwrap(),
            DataType::Boolean
        );

        // lnglat + lnglat is OK
        assert_eq!(
            predicate_stub
                .return_type(&[geom_lnglat.clone(), geom_lnglat.clone()])
                .unwrap(),
            DataType::Boolean
        );

        // Non-equal CRSes should error
        let err = predicate_stub
            .return_type(&[WKB_GEOMETRY.data_type_maybe_deprecated(), geom_lnglat.clone()])
            .unwrap_err();
        assert!(err.message().starts_with("Mismatched CRS arguments"));

        // When geometry is output, it should match the crses of the inputs
        let geom_out_stub = SedonaScalarUDF::new_stub(
            "stubby",
            ArgMatcher::new(
                vec![ArgMatcher::is_geometry(), ArgMatcher::is_geometry()],
                WKB_GEOMETRY,
            ),
            Volatility::Immutable,
            None,
        );

        assert_eq!(
            geom_out_stub
                .return_type(&[geom_lnglat.clone(), geom_lnglat.clone()])
                .unwrap(),
            geom_lnglat.clone()
        );
    }

    #[test]
    fn return_type_from_scalar_arg() {
        let udf: ScalarUDF =
            SedonaScalarUDF::from_kernel("simple_cast", Arc::new(SimpleCast {})).into();
        let call = udf.call(vec![lit(10), lit("float32")]);
        let schema = DFSchema::empty();
        assert_eq!(
            call.data_type_and_nullable(&schema).unwrap(),
            (DataType::Float32, true)
        );
    }

    #[derive(Debug)]
    struct SimpleCast {}

    impl SimpleCast {
        fn parse_type(val: &ColumnarValue) -> Result<SedonaType> {
            if let ColumnarValue::Scalar(ScalarValue::Utf8(Some(scalar_arg1))) = val {
                match scalar_arg1.as_str() {
                    "float32" => return Ok(DataType::Float32.try_into().unwrap()),
                    "float64" => return Ok(DataType::Float64.try_into().unwrap()),
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
            args[0].cast_to(&out_type.data_type_maybe_deprecated(), None)
        }
    }
}
