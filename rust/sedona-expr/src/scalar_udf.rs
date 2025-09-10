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
use std::{any::Any, fmt::Debug, sync::Arc};

use arrow_schema::{DataType, FieldRef};
use datafusion_common::{not_impl_err, Result, ScalarValue};
use datafusion_expr::{
    ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature,
    Volatility,
};
use sedona_common::sedona_internal_err;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

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

    /// Add a new kernel to a Scalar UDF
    ///
    /// Because kernels are resolved in reverse order, the new kernel will take
    /// precedence over any previously added kernels that apply to the same types.
    pub fn add_kernel(&mut self, kernel: ScalarKernelRef) {
        self.kernels.push(kernel);
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

    fn return_type(&self, _args: &[DataType]) -> Result<DataType> {
        sedona_internal_err!("Should not be called (use return_field_from_args())")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let arg_types = args
            .arg_fields
            .iter()
            .map(|field| SedonaType::from_storage_field(field))
            .collect::<Result<Vec<_>>>()?;
        let (_, out_type) = self.return_type_impl(&arg_types, args.scalar_arguments)?;
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

        let (kernel, _) = self.return_type_impl(&arg_types, &arg_scalars)?;
        kernel.invoke_batch(&arg_types, &args.args)
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }
}

#[cfg(test)]
mod tests {
    use datafusion_common::{scalar::ScalarValue, DFSchema};
    use sedona_testing::testers::ScalarUdfTester;

    use datafusion_expr::{lit, ExprSchemable, ScalarUDF};
    use sedona_schema::{
        crs::lnglat,
        datatypes::{Edges, WKB_GEOMETRY},
    };

    use super::*;

    #[test]
    fn udf_empty() -> Result<()> {
        // UDF with no implementations
        let udf = SedonaScalarUDF::new("empty", vec![], Volatility::Immutable, None);
        assert_eq!(udf.name(), "empty");
        assert_eq!(udf.coerce_types(&[])?, vec![]);

        let tester = ScalarUdfTester::new(udf.into(), vec![]);

        let err = tester.return_type().unwrap_err();
        assert_eq!(err.message(), "empty([]): No kernel matching arguments");

        let batch_err = tester.invoke_arrays(vec![]).unwrap_err();
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
        udf.add_kernel(SimpleSedonaScalarKernel::new_ref(
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
    fn stub() {
        let stub = SedonaScalarUDF::new_stub(
            "stubby",
            ArgMatcher::new(vec![], SedonaType::Arrow(DataType::Boolean)),
            Volatility::Immutable,
            None,
        );
        let tester = ScalarUdfTester::new(stub.into(), vec![]);
        tester.assert_return_type(DataType::Boolean);

        let err = tester.invoke_arrays(vec![]).unwrap_err();
        assert_eq!(
            err.message(),
            "Implementation for stubby([]) was not registered"
        );
    }

    #[test]
    fn crs_propagation() {
        let geom_lnglat = SedonaType::Wkb(Edges::Planar, lnglat());
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
        let geom_out_stub = SedonaScalarUDF::new_stub(
            "stubby",
            ArgMatcher::new(
                vec![ArgMatcher::is_geometry(), ArgMatcher::is_geometry()],
                WKB_GEOMETRY,
            ),
            Volatility::Immutable,
            None,
        );

        let tester = ScalarUdfTester::new(
            geom_out_stub.clone().into(),
            vec![geom_lnglat.clone(), geom_lnglat.clone()],
        );
        tester.assert_return_type(geom_lnglat.clone());
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
