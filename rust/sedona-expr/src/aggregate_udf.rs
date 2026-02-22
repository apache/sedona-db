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
    Accumulator, AggregateUDFImpl, Documentation, GroupsAccumulator, Signature, Volatility,
};
use sedona_common::sedona_internal_err;
use sedona_schema::datatypes::SedonaType;

/// Shorthand for a [SedonaAccumulator] reference
pub type SedonaAccumulatorRef = Arc<dyn SedonaAccumulator>;

/// Helper to resolve an iterable of accumulators
pub trait IntoSedonaAccumulatorRefs {
    fn into_sedona_accumulator_refs(self) -> Vec<SedonaAccumulatorRef>;
}

impl IntoSedonaAccumulatorRefs for SedonaAccumulatorRef {
    fn into_sedona_accumulator_refs(self) -> Vec<SedonaAccumulatorRef> {
        vec![self]
    }
}

impl IntoSedonaAccumulatorRefs for Vec<SedonaAccumulatorRef> {
    fn into_sedona_accumulator_refs(self) -> Vec<SedonaAccumulatorRef> {
        self
    }
}

impl<T: SedonaAccumulator + 'static> IntoSedonaAccumulatorRefs for T {
    fn into_sedona_accumulator_refs(self) -> Vec<SedonaAccumulatorRef> {
        vec![Arc::new(self)]
    }
}

impl<T: SedonaAccumulator + 'static> IntoSedonaAccumulatorRefs for Vec<Arc<T>> {
    fn into_sedona_accumulator_refs(self) -> Vec<SedonaAccumulatorRef> {
        self.into_iter()
            .map(|item| item as SedonaAccumulatorRef)
            .collect()
    }
}

/// Top-level aggregate user-defined function
///
/// This struct implements datafusion's AggregateUDFImpl and implements kernel dispatch
/// such that implementations can be registered flexibly.
#[derive(Debug, Clone)]
pub struct SedonaAggregateUDF {
    name: String,
    signature: Signature,
    kernels: Vec<SedonaAccumulatorRef>,
}

impl PartialEq for SedonaAggregateUDF {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for SedonaAggregateUDF {}

impl std::hash::Hash for SedonaAggregateUDF {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

impl SedonaAggregateUDF {
    /// Create a new SedonaAggregateUDF
    pub fn new(
        name: &str,
        kernels: impl IntoSedonaAccumulatorRefs,
        volatility: Volatility,
    ) -> Self {
        let signature = Signature::user_defined(volatility);
        Self {
            name: name.to_string(),
            signature,
            kernels: kernels.into_sedona_accumulator_refs(),
        }
    }

    /// Create a new immutable SedonaAggregateUDF
    pub fn from_impl(name: &str, kernels: impl IntoSedonaAccumulatorRefs) -> Self {
        Self::new(name, kernels, Volatility::Immutable)
    }

    /// Add a new kernel to an Aggregate UDF
    ///
    /// Because kernels are resolved in reverse order, the new kernel will take
    /// precedence over any previously added kernels that apply to the same types.
    pub fn add_kernel(&mut self, kernels: impl IntoSedonaAccumulatorRefs) {
        for kernel in kernels.into_sedona_accumulator_refs() {
            self.kernels.push(kernel);
        }
    }

    // List the current kernels
    pub fn kernels(&self) -> &[SedonaAccumulatorRef] {
        &self.kernels
    }

    fn accumulator_arg_types(args: &AccumulatorArgs) -> Result<Vec<SedonaType>> {
        let arg_fields = args
            .exprs
            .iter()
            .map(|expr| expr.return_field(args.schema))
            .collect::<Result<Vec<_>>>()?;
        arg_fields
            .iter()
            .map(|field| SedonaType::from_storage_field(field))
            .collect()
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

    fn groups_accumulator_supported(&self, args: AccumulatorArgs) -> bool {
        if let Ok(arg_types) = Self::accumulator_arg_types(&args) {
            if let Ok((accumulator, _)) = self.dispatch_impl(&arg_types) {
                return accumulator.groups_accumulator_supported(&arg_types);
            }
        }

        false
    }

    fn create_groups_accumulator(
        &self,
        args: AccumulatorArgs,
    ) -> Result<Box<dyn GroupsAccumulator>> {
        let arg_types = Self::accumulator_arg_types(&args)?;
        let (accumulator, output_type) = self.dispatch_impl(&arg_types)?;
        accumulator.groups_accumulator(&arg_types, &output_type)
    }

    fn accumulator(&self, acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        let arg_types = Self::accumulator_arg_types(&acc_args)?;
        let (accumulator, output_type) = self.dispatch_impl(&arg_types)?;
        accumulator.accumulator(&arg_types, &output_type)
    }

    fn documentation(&self) -> Option<&Documentation> {
        None
    }
}

pub trait SedonaAccumulator: Debug + Send + Sync {
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

    /// Given input data types, check if this implementation supports GroupsAccumulator
    fn groups_accumulator_supported(&self, _args: &[SedonaType]) -> bool {
        false
    }

    /// Given input data types, resolve a [GroupsAccumulator]
    ///
    /// A GroupsAccumulator is an important optimization for aggregating many small groups,
    /// particularly when such an aggregation is cheap. See the DataFusion documentation
    /// for details.
    fn groups_accumulator(
        &self,
        _args: &[SedonaType],
        _output_type: &SedonaType,
    ) -> Result<Box<dyn GroupsAccumulator>> {
        sedona_internal_err!("groups_accumulator not supported for {self:?}")
    }

    /// The fields representing the underlying serialized state of the Accumulator
    fn state_fields(&self, args: &[SedonaType]) -> Result<Vec<FieldRef>>;
}

#[cfg(test)]
mod test {
    use crate::aggregate_udf::SedonaAggregateUDF;

    use super::*;

    #[test]
    fn udaf_empty() -> Result<()> {
        // UDF with no implementations
        let udf = SedonaAggregateUDF::new(
            "empty",
            Vec::<SedonaAccumulatorRef>::new(),
            Volatility::Immutable,
        );
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
}
