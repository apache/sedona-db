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
use datafusion_common::{not_impl_err, Result};
use datafusion_expr::{
    function::{AccumulatorArgs, StateFieldsArgs},
    Accumulator, AggregateUDFImpl, Documentation, Signature, Volatility,
};
use sedona_common::sedona_internal_err;
use sedona_schema::datatypes::SedonaType;

use sedona_schema::matchers::ArgMatcher;

pub type SedonaAccumulatorRef = Arc<dyn SedonaAccumulator + Send + Sync>;

/// Top-level aggregate user-defined function
///
/// This struct implements datafusion's AggregateUDFImpl and implements kernel dispatch
/// such that implementations can be registered flexibly.
#[derive(Debug, Clone)]
pub struct SedonaAggregateUDF {
    name: String,
    signature: Signature,
    kernels: Vec<SedonaAccumulatorRef>,
    documentation: Option<Documentation>,
}

impl SedonaAggregateUDF {
    /// Create a new SedonaAggregateUDF
    pub fn new(
        name: &str,
        kernels: Vec<SedonaAccumulatorRef>,
        volatility: Volatility,
        documentation: Option<Documentation>,
    ) -> Self {
        let signature = Signature::user_defined(volatility);
        Self {
            name: name.to_string(),
            signature,
            kernels,
            documentation,
        }
    }

    /// Create a new stub aggregate function
    ///
    /// Creates a new aggregate function that calculates a return type but fails when
    /// invoked with arguments. This is useful to create stub functions when it is
    /// expected that the actual functionality will be registered from one or more
    /// independent crates (e.g., ST_Union_Agg(), which may be implemented in
    /// sedona-geo or sedona-geography).
    pub fn new_stub(
        name: &str,
        arg_matcher: ArgMatcher,
        volatility: Volatility,
        documentation: Option<Documentation>,
    ) -> Self {
        let stub_kernel = StubAccumulator::new(name.to_string(), arg_matcher);
        Self::new(name, vec![Arc::new(stub_kernel)], volatility, documentation)
    }

    /// Add a new kernel to an Aggregate UDF
    ///
    /// Because kernels are resolved in reverse order, the new kernel will take
    /// precedence over any previously added kernels that apply to the same types.
    pub fn add_kernel(&mut self, kernel: SedonaAccumulatorRef) {
        self.kernels.push(kernel);
    }

    // List the current kernels
    pub fn kernels(&self) -> &[SedonaAccumulatorRef] {
        &self.kernels
    }

    fn dispatch_impl(&self, args: &[SedonaType]) -> Result<(&dyn SedonaAccumulator, SedonaType)> {
        // Resolve kernels in reverse so that more recently added ones are resolved first
        for kernel in self.kernels.iter().rev() {
            if let Some(return_type) = kernel.return_type(args)? {
                return Ok((kernel.as_ref(), return_type));
            }
        }

        not_impl_err!("{}({:?}): No kernel matching arguments", self.name, args)
    }
}

impl AggregateUDFImpl for SedonaAggregateUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        Ok(arg_types.into())
    }

    fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<FieldRef>> {
        let arg_types = args
            .input_fields
            .iter()
            .map(|field| SedonaType::from_storage_field(field))
            .collect::<Result<Vec<_>>>()?;
        let (accumulator, _) = self.dispatch_impl(&arg_types)?;
        accumulator.state_fields(&arg_types)
    }

    fn return_field(&self, arg_fields: &[FieldRef]) -> Result<FieldRef> {
        let arg_types = arg_fields
            .iter()
            .map(|field| SedonaType::from_storage_field(field))
            .collect::<Result<Vec<_>>>()?;
        let (_, out_type) = self.dispatch_impl(&arg_types)?;
        Ok(Arc::new(out_type.to_storage_field("", true)?))
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        sedona_internal_err!("return_type() should not be called (use return_field())")
    }

    fn accumulator(&self, acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        let arg_fields = acc_args
            .exprs
            .iter()
            .map(|expr| expr.return_field(acc_args.schema))
            .collect::<Result<Vec<_>>>()?;
        let arg_types = arg_fields
            .iter()
            .map(|field| SedonaType::from_storage_field(field))
            .collect::<Result<Vec<_>>>()?;
        let (accumulator, output_type) = self.dispatch_impl(&arg_types)?;
        accumulator.accumulator(&arg_types, &output_type)
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.documentation.as_ref()
    }
}

pub trait SedonaAccumulator: Debug {
    /// Given input data types, calculate an output data type
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>>;

    /// Given input data types and previously-calculated output data type,
    /// resolve an [Accumulator]
    ///
    /// The Accumulator provides the underlying DataFusion implementation.
    /// The SedonaAccumulator does not perform any wrapping or unwrapping on the
    /// accumulator arguments or return values (in anticipation of wrapping/unwrapping
    /// being reverted in the near future).
    fn accumulator(
        &self,
        args: &[SedonaType],
        output_type: &SedonaType,
    ) -> Result<Box<dyn Accumulator>>;

    /// The fields representing the underlying serialized state of the Accumulator
    fn state_fields(&self, args: &[SedonaType]) -> Result<Vec<FieldRef>>;
}

#[derive(Debug)]
struct StubAccumulator {
    name: String,
    matcher: ArgMatcher,
}

impl StubAccumulator {
    fn new(name: String, matcher: ArgMatcher) -> Self {
        Self { name, matcher }
    }
}

impl SedonaAccumulator for StubAccumulator {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        self.matcher.match_args(args)
    }

    fn accumulator(
        &self,
        args: &[SedonaType],
        _output_type: &SedonaType,
    ) -> Result<Box<dyn Accumulator>> {
        not_impl_err!(
            "Implementation for {}({args:?}) was not registered",
            self.name
        )
    }

    fn state_fields(&self, _args: &[SedonaType]) -> Result<Vec<FieldRef>> {
        Ok(vec![])
    }
}

#[cfg(test)]
mod test {
    use sedona_testing::testers::AggregateUdfTester;

    use crate::aggregate_udf::SedonaAggregateUDF;

    use super::*;

    #[test]
    fn udaf_empty() -> Result<()> {
        // UDF with no implementations
        let udf = SedonaAggregateUDF::new("empty", vec![], Volatility::Immutable, None);
        assert_eq!(udf.name(), "empty");
        let err = udf.return_field(&[]).unwrap_err();
        assert_eq!(err.message(), "empty([]): No kernel matching arguments");
        assert!(udf.kernels().is_empty());
        assert_eq!(udf.coerce_types(&[])?, vec![]);

        let batch_err = udf.return_field(&[]).unwrap_err();
        assert_eq!(
            batch_err.message(),
            "empty([]): No kernel matching arguments"
        );

        Ok(())
    }

    #[test]
    fn stub() {
        let stub = SedonaAggregateUDF::new_stub(
            "stubby",
            ArgMatcher::new(vec![], SedonaType::Arrow(DataType::Boolean)),
            Volatility::Immutable,
            None,
        );

        // We registered the stub with zero arguments, so when we call it
        // with zero arguments it should calculate a return type but
        // produce our stub error message when used.
        let tester = AggregateUdfTester::new(stub.clone().into(), vec![]);
        assert_eq!(
            tester.return_type().unwrap(),
            SedonaType::Arrow(DataType::Boolean)
        );

        let err = tester.aggregate(&vec![]).unwrap_err();
        assert_eq!(
            err.message(),
            "Implementation for stubby([]) was not registered"
        );

        // If we call with anything else, we shouldn't be able to do anything
        let tester = AggregateUdfTester::new(
            stub.clone().into(),
            vec![SedonaType::Arrow(DataType::Binary)],
        );
        let err = tester.return_type().unwrap_err();
        assert_eq!(
            err.message(),
            "stubby([Arrow(Binary)]): No kernel matching arguments"
        );
    }
}
