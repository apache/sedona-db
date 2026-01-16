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

use std::{fmt::Debug, iter::zip, sync::Arc};

use arrow_array::{Array, ArrayRef, StructArray};
use arrow_buffer::NullBuffer;
use arrow_schema::{DataType, Field, FieldRef};
use datafusion_common::{
    cast::{as_string_view_array, as_struct_array},
    exec_err, DataFusionError, Result, ScalarValue,
};
use datafusion_expr::{Accumulator, ColumnarValue};
use sedona_common::sedona_internal_err;
use sedona_schema::{crs::deserialize_crs, datatypes::SedonaType, matchers::ArgMatcher};

use crate::aggregate_udf::{IntoSedonaAccumulatorRefs, SedonaAccumulator, SedonaAccumulatorRef};
use crate::scalar_udf::{IntoScalarKernelRefs, ScalarKernelRef, SedonaScalarKernel};

/// Wrap a [SedonaScalarKernel] to provide Item CRS type support
///
/// Most kernels that operate on geometry or geography in some way
/// can also support Item CRS inputs:
///
/// - Functions that return a non-spatial type whose value does not
///   depend on the input CRS only need to operate on the `item` portion
///   of any item_crs input (e.g., ST_Area()).
/// - Functions that accept two or more spatial arguments must have
///   compatible CRSes.
/// - Functions that return a geometry or geography must also return
///   an item_crs type where the output CRSes are propagated from the
///   input.
///
/// This kernel provides an automatic wrapper enforcing these rules.
/// It is appropriate for most functions except:
///
/// - Functions whose return value depends on the CRS
/// - Functions whose return value depends on the value of a scalar
///   argument
/// - Functions whose return CRS is not strictly propagated from the
///   CRSes of the arguments.
#[derive(Debug)]
pub struct ItemCrsKernel {
    inner: ScalarKernelRef,
}

impl ItemCrsKernel {
    /// Create a new [ScalarKernelRef] wrapping the input
    ///
    /// The resulting kernel matches arguments of the input with ItemCrs inputs
    /// but not those of the original kernel (i.e., a function needs both kernels
    /// to support both type-level and item-level CRSes).
    pub fn new_ref(inner: ScalarKernelRef) -> ScalarKernelRef {
        Arc::new(Self { inner })
    }

    /// Wrap a vector of kernels by appending all ItemCrs versions followed by
    /// the contents of inner
    ///
    /// This is the recommended way to add kernels when all of them should support
    /// ItemCrs inputs.
    pub fn wrap_impl(inner: impl IntoScalarKernelRefs) -> Vec<ScalarKernelRef> {
        let kernels = inner.into_scalar_kernel_refs();

        let mut out = Vec::with_capacity(kernels.len() * 2);

        // Add ItemCrsKernels first (so they will be resolved last)
        for inner_kernel in &kernels {
            out.push(ItemCrsKernel::new_ref(inner_kernel.clone()));
        }

        for inner_kernel in kernels {
            out.push(inner_kernel);
        }

        out
    }
}

impl SedonaScalarKernel for ItemCrsKernel {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        return_type_handle_item_crs(self.inner.as_ref(), args)
    }

    fn invoke_batch_from_args(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
        return_type: &SedonaType,
        num_rows: usize,
    ) -> Result<ColumnarValue> {
        invoke_handle_item_crs(self.inner.as_ref(), arg_types, args, return_type, num_rows)
    }

    fn invoke_batch(
        &self,
        _arg_types: &[SedonaType],
        _args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        sedona_internal_err!("Should not be called because invoke_batch_from_args() is implemented")
    }
}

/// Wrap a [SedonaAccumulator] to provide Item CRS type support
///
/// Most accumulators that operate on geometry or geography in some way
/// can also support Item CRS inputs:
///
/// - Accumulators that return a non-spatial type whose value does not
///   depend on the input CRS only need to operate on the `item` portion
///   of any item_crs input (e.g., ST_Analyze_Agg()).
/// - Accumulators that return a geometry or geography must also return
///   an item_crs type where the output CRSes are propagated from the
///   input.
/// - CRSes within a single group must be compatible
///
/// This accumulator provides an automatic wrapper enforcing these rules.
#[derive(Debug)]
pub struct ItemCrsSedonaAccumulator {
    inner: SedonaAccumulatorRef,
}

impl ItemCrsSedonaAccumulator {
    /// Create a new [SedonaAccumulatorRef] wrapping the input
    ///
    /// The resulting accumulator matches arguments of the input with ItemCrs inputs
    /// but not those of the original accumulator (i.e., an aggregate function needs both
    /// accumulators to support both type-level and item-level CRSes).
    pub fn new_ref(inner: SedonaAccumulatorRef) -> SedonaAccumulatorRef {
        Arc::new(Self { inner })
    }

    /// Wrap a vector of accumulators by appending all ItemCrs versions followed by
    /// the contents of inner
    ///
    /// This is the recommended way to add accumulators when all of them should support
    /// ItemCrs inputs.
    pub fn wrap_impl(inner: impl IntoSedonaAccumulatorRefs) -> Vec<SedonaAccumulatorRef> {
        let accumulators = inner.into_sedona_accumulator_refs();

        let mut out = Vec::with_capacity(accumulators.len() * 2);

        // Add ItemCrsAccumulators first (so they will be resolved last)
        for inner_accumulator in &accumulators {
            out.push(ItemCrsSedonaAccumulator::new_ref(inner_accumulator.clone()));
        }

        for inner_accumulator in accumulators {
            out.push(inner_accumulator);
        }

        out
    }
}

impl SedonaAccumulator for ItemCrsSedonaAccumulator {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        // We don't have any functions we can test this with yet, so for the moment only support
        // single-argument aggregations (slightly simpler).
        if args.len() != 1 {
            return Ok(None);
        }

        // This implementation doesn't apply to non-item crs types
        if !ArgMatcher::is_item_crs().match_type(&args[0]) {
            return Ok(None);
        }

        // Strip any CRS that might be present from the input type
        let item_arg_types = args
            .iter()
            .map(|arg_type| {
                parse_item_crs_arg_type_strip_crs(arg_type).map(|(item_type, _)| item_type)
            })
            .collect::<Result<Vec<_>>>()?;

        // Resolve the inner accumulator's return type.
        if let Some(item_type) = self.inner.return_type(&item_arg_types)? {
            let geo_matcher = ArgMatcher::is_geometry_or_geography();

            // If the inner output is item_crs, the output must also be item_crs. Otherwise
            // the output is left as is.
            if geo_matcher.match_type(&item_type) {
                Ok(Some(SedonaType::new_item_crs(&item_type)?))
            } else {
                Ok(Some(item_type))
            }
        } else {
            Ok(None)
        }
    }

    fn accumulator(
        &self,
        args: &[SedonaType],
        output_type: &SedonaType,
    ) -> Result<Box<dyn datafusion_expr::Accumulator>> {
        // Strip any CRS that might be present from the input type
        let item_arg_types = args
            .iter()
            .map(|arg_type| {
                parse_item_crs_arg_type_strip_crs(arg_type).map(|(item_type, _)| item_type)
            })
            .collect::<Result<Vec<_>>>()?;

        // Extract the item output type from the item_crs output type
        let (item_output_type, _) = parse_item_crs_arg_type(output_type)?;

        // Create the inner accumulator
        let inner = self.inner.accumulator(&item_arg_types, &item_output_type)?;

        Ok(Box::new(ItemCrsAccumulator {
            inner,
            item_output_type,
            crs: None,
        }))
    }

    fn state_fields(&self, args: &[SedonaType]) -> Result<Vec<FieldRef>> {
        // We need an extra state field to track the CRS of each group
        let mut fields = self.inner.state_fields(args)?;
        fields.push(Field::new("group_crs", DataType::Utf8View, true).into());
        Ok(fields)
    }
}

#[derive(Debug)]
struct ItemCrsAccumulator {
    /// The wrapped inner accumulator
    inner: Box<dyn Accumulator>,
    /// The item output type (without the item_crs wrapper)
    item_output_type: SedonaType,
    /// If any rows have been encountered, the CRS (the literal string "0" is used
    /// as a sentinel for "no CRS" because we have to serialize it (and None is
    /// reserved for "we haven't seen any rows yet"))
    crs: Option<String>,
}

impl Accumulator for ItemCrsAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        // The input is an item_crs struct array; extract the item and crs columns
        let struct_array = as_struct_array(&values[0])?;
        let item_array = struct_array.column(0).clone();
        let crs_array = as_string_view_array(struct_array.column(1))?;

        // Check and track CRS values
        if let Some(struct_nulls) = struct_array.nulls() {
            // Skip CRS values for null items
            for (is_valid, crs_value) in zip(struct_nulls, crs_array.iter()) {
                if is_valid {
                    self.merge_crs(crs_value.unwrap_or("0"))?;
                }
            }
        } else {
            // No nulls
            for crs_value in crs_array.iter() {
                self.merge_crs(crs_value.unwrap_or("0"))?;
            }
        }

        // Update the inner accumulator with just the item portion
        self.inner.update_batch(&[item_array])?;
        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let inner_result = self.inner.evaluate()?;

        // If the output type is not geometry or geography we can just return it
        if !matches!(
            self.item_output_type,
            SedonaType::Wkb(_, _) | SedonaType::WkbView(_, _)
        ) {
            return Ok(inner_result);
        }

        // Otherwise, prepare the item_crs result

        // Convert the sentinel back to None
        let crs_value = match &self.crs {
            Some(s) if s == "0" => None,
            Some(s) => Some(s.clone()),
            None => None,
        };

        // Create the item_crs struct scalar
        let item_crs_result = make_item_crs(
            &self.item_output_type,
            ColumnarValue::Scalar(inner_result),
            &ColumnarValue::Scalar(ScalarValue::Utf8View(crs_value)),
            None,
        )?;

        match item_crs_result {
            ColumnarValue::Scalar(scalar) => Ok(scalar),
            ColumnarValue::Array(_) => {
                sedona_internal_err!("Expected scalar result from make_item_crs")
            }
        }
    }

    fn size(&self) -> usize {
        self.inner.size() + size_of::<ItemCrsAccumulator>()
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let mut inner_state = self.inner.state()?;
        inner_state.push(ScalarValue::Utf8View(self.crs.clone()));
        Ok(inner_state)
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        // The CRS field is the last element of states
        if states.is_empty() {
            return sedona_internal_err!("Expected at least one state field");
        }
        let crs_array = as_string_view_array(states.last().unwrap())?;

        // Check and merge CRS values from the state
        for crs_str in crs_array.iter().flatten() {
            self.merge_crs(crs_str)?;
        }

        // Merge the inner state (excluding the CRS field)
        let inner_states = &states[..states.len() - 1];
        self.inner.merge_batch(inner_states)
    }
}

impl ItemCrsAccumulator {
    /// Merge a CRS value into the accumulator's tracked CRS
    ///
    /// Ensures all CRS values are compatible. Here "0" means an explicit
    /// null crs. This is because we have to serialize it somehow and None
    /// is reserved for the "we haven't seen a CRS yet".
    fn merge_crs(&mut self, crs_str: &str) -> Result<()> {
        match &self.crs {
            None => {
                // First CRS value encountered
                self.crs = Some(crs_str.to_string());
                Ok(())
            }
            Some(existing) if existing == crs_str => {
                // CRS is byte-for-byte equal, nothing to do
                Ok(())
            }
            Some(existing) => {
                // Check if CRSes are semantically equal
                let existing_crs = deserialize_crs(existing)?;
                let new_crs = deserialize_crs(crs_str)?;
                if existing_crs == new_crs {
                    Ok(())
                } else {
                    let existing_displ = existing_crs
                        .map(|c| c.to_string())
                        .unwrap_or("None".to_string());
                    let new_displ = new_crs.map(|c| c.to_string()).unwrap_or("None".to_string());
                    exec_err!("CRS values not equal: {existing_displ} vs {new_displ}")
                }
            }
        }
    }
}

/// Calculate a return type based on the underlying kernel
///
/// This function extracts the item portion of any item_crs input and
/// passes the result to the underlying kernel's return type implementation.
/// If the underlying kernel is going to return a geometry or geography type,
/// we wrap it in an item_crs type.
///
/// This function does not pass on input scalars, because those types of
/// functions as used in SedonaDB typically assign a type-level CRS.
/// Functions that use scalar inputs to calculate an output type need
/// to implement an item_crs implementation themselves.
fn return_type_handle_item_crs(
    kernel: &dyn SedonaScalarKernel,
    arg_types: &[SedonaType],
) -> Result<Option<SedonaType>> {
    let item_crs_matcher = ArgMatcher::is_item_crs();

    // If there are no item_crs arguments, this kernel never applies.
    if !arg_types
        .iter()
        .any(|arg_type| item_crs_matcher.match_type(arg_type))
    {
        return Ok(None);
    }

    // Extract the item types. This also strips the type-level CRS for any non item-crs
    // type, because any resulting geometry type should be CRS free.
    let item_arg_types = arg_types
        .iter()
        .map(|arg_type| parse_item_crs_arg_type_strip_crs(arg_type).map(|(item_type, _)| item_type))
        .collect::<Result<Vec<_>>>()?;

    // Any kernel that uses scalars to determine the output type is spurious here, so we
    // pretend that there aren't any for the purposes of computing the type.
    let scalar_args_none = (0..arg_types.len())
        .map(|_| None)
        .collect::<Vec<Option<&ScalarValue>>>();

    // If the wrapped kernel matches and returns a geometry type, that geometry type will be an
    // item/crs type. The new_item_crs() function handles stripping any CRS that might be present
    // in the output type.
    if let Some(item_type) =
        kernel.return_type_from_args_and_scalars(&item_arg_types, &scalar_args_none)?
    {
        let geo_matcher = ArgMatcher::is_geometry_or_geography();
        if geo_matcher.match_type(&item_type) {
            Ok(Some(SedonaType::new_item_crs(&item_type)?))
        } else {
            Ok(Some(item_type))
        }
    } else {
        Ok(None)
    }
}

/// Execute an underlying kernel
///
/// This function handles invoking the underlying kernel, which operates
/// only on the `item` portion of the `item_crs` type. Before executing,
/// this function handles ensuring that all CRSes are compatible, and,
/// if necessary, wrap a geometry or geography output in an item_crs
/// type.
fn invoke_handle_item_crs(
    kernel: &dyn SedonaScalarKernel,
    arg_types: &[SedonaType],
    args: &[ColumnarValue],
    return_type: &SedonaType,
    num_rows: usize,
) -> Result<ColumnarValue> {
    // Separate the argument types into item and Option<crs>
    // Don't strip the CRSes because we need them to compare with
    // the item-level CRSes to ensure they are equal.
    let arg_types_unwrapped = arg_types
        .iter()
        .map(parse_item_crs_arg_type)
        .collect::<Result<Vec<_>>>()?;

    let args_unwrapped = zip(&arg_types_unwrapped, args)
        .map(|(arg_type, arg)| {
            let (item_type, crs_type) = arg_type;
            parse_item_crs_arg(item_type, crs_type, arg)
        })
        .collect::<Result<Vec<_>>>()?;

    let crs_args = args_unwrapped
        .iter()
        .flat_map(|(_, crs_arg)| crs_arg)
        .collect::<Vec<_>>();

    let crs_result = ensure_crs_args_equal(&crs_args)?;

    let item_types = arg_types_unwrapped
        .iter()
        .map(|(item_type, _)| item_type.clone())
        .collect::<Vec<_>>();
    let item_args = args_unwrapped
        .iter()
        .map(|(item_arg, _)| item_arg.clone())
        .collect::<Vec<_>>();

    let item_arg_types_no_crs = arg_types
        .iter()
        .map(|arg_type| parse_item_crs_arg_type_strip_crs(arg_type).map(|(item_type, _)| item_type))
        .collect::<Result<Vec<_>>>()?;
    let out_item_type = match kernel.return_type(&item_arg_types_no_crs)? {
        Some(matched_item_type) => matched_item_type,
        None => return sedona_internal_err!("Expected inner kernel to match types {item_types:?}"),
    };

    let item_result =
        kernel.invoke_batch_from_args(&item_types, &item_args, return_type, num_rows)?;

    if ArgMatcher::is_geometry_or_geography().match_type(&out_item_type) {
        make_item_crs(&out_item_type, item_result, crs_result, None)
    } else {
        Ok(item_result)
    }
}

/// Create a new item_crs struct [ColumnarValue]
///
/// Optionally provide extra nulls (e.g., to satisfy a scalar function contract
/// where null inputs -> null outputs).
pub fn make_item_crs(
    item_type: &SedonaType,
    item_result: ColumnarValue,
    crs_result: &ColumnarValue,
    extra_nulls: Option<&NullBuffer>,
) -> Result<ColumnarValue> {
    let out_fields = vec![
        item_type.to_storage_field("item", true)?,
        Field::new("crs", DataType::Utf8View, true),
    ];

    let scalar_result = matches!(
        (&item_result, crs_result),
        (ColumnarValue::Scalar(_), ColumnarValue::Scalar(_))
    );

    let item_crs_arrays = ColumnarValue::values_to_arrays(&[item_result, crs_result.clone()])?;
    let item_array = &item_crs_arrays[0];
    let crs_array = &item_crs_arrays[1];
    let nulls = NullBuffer::union(item_array.nulls(), extra_nulls);

    let item_crs_array = StructArray::new(
        out_fields.into(),
        vec![item_array.clone(), crs_array.clone()],
        nulls,
    );

    if scalar_result {
        Ok(ScalarValue::Struct(Arc::new(item_crs_array)).into())
    } else {
        Ok(ColumnarValue::Array(Arc::new(item_crs_array)))
    }
}

/// Given an input type, separate it into an item and crs type (if the input
/// is an item_crs type). Otherwise, just return the item type as is.
fn parse_item_crs_arg_type(sedona_type: &SedonaType) -> Result<(SedonaType, Option<SedonaType>)> {
    if let SedonaType::Arrow(DataType::Struct(fields)) = sedona_type {
        let field_names = fields.iter().map(|f| f.name()).collect::<Vec<_>>();
        if field_names != ["item", "crs"] {
            return Ok((sedona_type.clone(), None));
        }

        let item = SedonaType::from_storage_field(&fields[0])?;
        let crs = SedonaType::from_storage_field(&fields[1])?;
        Ok((item, Some(crs)))
    } else {
        Ok((sedona_type.clone(), None))
    }
}

/// Given an input type, separate it into an item and crs type (if the input
/// is an item_crs type). Otherwise, just return the item type as is. This
/// version strips the CRS, which we need to do here before passing it to the
/// underlying kernel (which expects all input CRSes to match).
fn parse_item_crs_arg_type_strip_crs(
    sedona_type: &SedonaType,
) -> Result<(SedonaType, Option<SedonaType>)> {
    match sedona_type {
        SedonaType::Wkb(edges, _) => Ok((SedonaType::Wkb(*edges, None), None)),
        SedonaType::WkbView(edges, _) => Ok((SedonaType::WkbView(*edges, None), None)),
        SedonaType::Arrow(DataType::Struct(fields))
            if fields.iter().map(|f| f.name()).collect::<Vec<_>>() == vec!["item", "crs"] =>
        {
            let item = SedonaType::from_storage_field(&fields[0])?;
            let crs = SedonaType::from_storage_field(&fields[1])?;
            Ok((item, Some(crs)))
        }
        other => Ok((other.clone(), None)),
    }
}

/// Separate an argument into the item and its crs (if applicable). This
/// operates on the result of parse_item_crs_arg_type().
fn parse_item_crs_arg(
    item_type: &SedonaType,
    crs_type: &Option<SedonaType>,
    arg: &ColumnarValue,
) -> Result<(ColumnarValue, Option<ColumnarValue>)> {
    if crs_type.is_some() {
        return match arg {
            ColumnarValue::Array(array) => {
                let struct_array = as_struct_array(array)?;
                Ok((
                    ColumnarValue::Array(struct_array.column(0).clone()),
                    Some(ColumnarValue::Array(struct_array.column(1).clone())),
                ))
            }
            ColumnarValue::Scalar(scalar_value) => {
                if let ScalarValue::Struct(struct_array) = scalar_value {
                    let item_scalar = ScalarValue::try_from_array(struct_array.column(0), 0)?;
                    let crs_scalar = ScalarValue::try_from_array(struct_array.column(1), 0)?;
                    Ok((
                        ColumnarValue::Scalar(item_scalar),
                        Some(ColumnarValue::Scalar(crs_scalar)),
                    ))
                } else {
                    sedona_internal_err!(
                        "Expected struct scalar for item_crs but got {}",
                        scalar_value
                    )
                }
            }
        };
    }

    match item_type {
        SedonaType::Wkb(_, crs) | SedonaType::WkbView(_, crs) => {
            let crs_scalar = if let Some(crs) = crs {
                if let Some(auth_code) = crs.to_authority_code()? {
                    ScalarValue::Utf8View(Some(auth_code))
                } else {
                    ScalarValue::Utf8View(Some(crs.to_json()))
                }
            } else {
                ScalarValue::Utf8View(None)
            };

            Ok((arg.clone(), Some(ColumnarValue::Scalar(crs_scalar))))
        }
        _ => Ok((arg.clone(), None)),
    }
}

/// Ensures values representing CRSes all represent equivalent CRS values
fn ensure_crs_args_equal<'a>(crs_args: &[&'a ColumnarValue]) -> Result<&'a ColumnarValue> {
    match crs_args.len() {
        0 => sedona_internal_err!("Zero CRS arguments as input to item_crs"),
        1 => Ok(crs_args[0]),
        _ => {
            let crs_args_string = crs_args
                .iter()
                .map(|arg| arg.cast_to(&DataType::Utf8View, None))
                .collect::<Result<Vec<_>>>()?;
            let crs_arrays = ColumnarValue::values_to_arrays(&crs_args_string)?;
            for i in 1..crs_arrays.len() {
                ensure_crs_string_arrays_equal2(&crs_arrays[i - 1], &crs_arrays[i])?
            }

            Ok(crs_args[0])
        }
    }
}

// Checks two string view arrays for equality when each represents a string representation
// of a CRS
fn ensure_crs_string_arrays_equal2(lhs: &ArrayRef, rhs: &ArrayRef) -> Result<()> {
    for (lhs_item, rhs_item) in zip(as_string_view_array(lhs)?, as_string_view_array(rhs)?) {
        if lhs_item == rhs_item {
            // First check for byte-for-byte equality (faster and most likely)
            continue;
        }

        // Check the deserialized CRS values for equality
        if let (Some(lhs_item_str), Some(rhs_item_str)) = (lhs_item, rhs_item) {
            let lhs_crs = deserialize_crs(lhs_item_str)?;
            let rhs_crs = deserialize_crs(rhs_item_str)?;
            if lhs_crs == rhs_crs {
                continue;
            }
        }

        if lhs_item != rhs_item {
            return Err(DataFusionError::Execution(format!(
                "CRS values not equal: {lhs_item:?} vs {rhs_item:?}",
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::{
        crs::lnglat,
        datatypes::{Edges, SedonaType, WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS},
    };
    use sedona_testing::{
        create::create_array_item_crs, create::create_scalar_item_crs, testers::ScalarUdfTester,
    };

    use crate::scalar_udf::{SedonaScalarUDF, SimpleSedonaScalarKernel};

    use super::*;

    // A test function of something + geometry -> out_type
    fn test_udf(out_type: SedonaType) -> ScalarUDF {
        let geom_to_geom_kernel = SimpleSedonaScalarKernel::new_ref(
            ArgMatcher::new(
                vec![ArgMatcher::is_any(), ArgMatcher::is_geometry()],
                out_type,
            ),
            Arc::new(|_arg_types, args| Ok(args[0].clone())),
        );

        let crsified_kernel = ItemCrsKernel::new_ref(geom_to_geom_kernel);
        SedonaScalarUDF::from_impl("fun", crsified_kernel.clone()).into()
    }

    #[test]
    fn item_crs_kernel_no_match() {
        // A call with geometry + geometry should fail (this case would be handled by the
        // original kernel, not the item_crs kernel)
        let tester = ScalarUdfTester::new(test_udf(WKB_GEOMETRY), vec![WKB_GEOMETRY, WKB_GEOMETRY]);
        let err = tester.return_type().unwrap_err();
        assert_eq!(
            err.message(),
            "fun([Wkb(Planar, None), Wkb(Planar, None)]): No kernel matching arguments"
        );
    }

    #[rstest]
    fn item_crs_kernel_basic(
        #[values(
            (WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS.clone()),
            (WKB_GEOMETRY_ITEM_CRS.clone(), WKB_GEOMETRY),
            (WKB_GEOMETRY_ITEM_CRS.clone(), WKB_GEOMETRY_ITEM_CRS.clone())
        )]
        arg_types: (SedonaType, SedonaType),
    ) {
        // A call with geometry + item_crs or both item_crs should return item_crs
        let tester = ScalarUdfTester::new(test_udf(WKB_GEOMETRY), vec![arg_types.0, arg_types.1]);
        tester.assert_return_type(WKB_GEOMETRY_ITEM_CRS.clone());
        let result = tester
            .invoke_scalar_scalar("POINT (0 1)", "POINT (1 2)")
            .unwrap();
        assert_eq!(
            result,
            create_scalar_item_crs(Some("POINT (0 1)"), None, &WKB_GEOMETRY)
        );
    }

    #[test]
    fn item_crs_kernel_crs_values() {
        let tester = ScalarUdfTester::new(
            test_udf(WKB_GEOMETRY),
            vec![WKB_GEOMETRY_ITEM_CRS.clone(), WKB_GEOMETRY_ITEM_CRS.clone()],
        );
        tester.assert_return_type(WKB_GEOMETRY_ITEM_CRS.clone());

        let scalar_item_crs_4326 =
            create_scalar_item_crs(Some("POINT (0 1)"), Some("EPSG:4326"), &WKB_GEOMETRY);
        let scalar_item_crs_crs84 =
            create_scalar_item_crs(Some("POINT (0 1)"), Some("OGC:CRS84"), &WKB_GEOMETRY);
        let scalar_item_crs_3857 =
            create_scalar_item_crs(Some("POINT (0 1)"), Some("EPSG:3857"), &WKB_GEOMETRY);

        // Should be able to execute when both arguments have an equal
        // (but not necessarily identical) CRS
        let result = tester
            .invoke_scalar_scalar(scalar_item_crs_4326.clone(), scalar_item_crs_crs84.clone())
            .unwrap();
        assert_eq!(result, scalar_item_crs_4326);

        // We should get an error when the CRSes are not compatible
        let err = tester
            .invoke_scalar_scalar(scalar_item_crs_4326.clone(), scalar_item_crs_3857.clone())
            .unwrap_err();
        assert_eq!(
            err.message(),
            "CRS values not equal: Some(\"EPSG:4326\") vs Some(\"EPSG:3857\")"
        );
    }

    #[test]
    fn item_crs_kernel_crs_types() {
        let scalar_item_crs_4326 =
            create_scalar_item_crs(Some("POINT (0 1)"), Some("EPSG:4326"), &WKB_GEOMETRY);
        let scalar_item_crs_crs84 =
            create_scalar_item_crs(Some("POINT (0 1)"), Some("OGC:CRS84"), &WKB_GEOMETRY);
        let scalar_item_crs_3857 =
            create_scalar_item_crs(Some("POINT (0 1)"), Some("EPSG:3857"), &WKB_GEOMETRY);

        let sedona_type_lnglat = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(
            test_udf(WKB_GEOMETRY),
            vec![WKB_GEOMETRY_ITEM_CRS.clone(), sedona_type_lnglat.clone()],
        );
        tester.assert_return_type(WKB_GEOMETRY_ITEM_CRS.clone());

        // We should be able to execute item_crs + geometry when the crs compares equal
        let result = tester
            .invoke_scalar_scalar(scalar_item_crs_4326.clone(), "POINT (3 4)")
            .unwrap();
        assert_eq!(result, scalar_item_crs_4326);

        let result = tester
            .invoke_scalar_scalar(scalar_item_crs_crs84.clone(), "POINT (3 4)")
            .unwrap();
        assert_eq!(result, scalar_item_crs_crs84);

        // We should get an error when the CRSes are not compatible
        let err = tester
            .invoke_scalar_scalar(scalar_item_crs_3857.clone(), "POINT (3 4)")
            .unwrap_err();
        assert_eq!(
            err.message(),
            "CRS values not equal: Some(\"EPSG:3857\") vs Some(\"OGC:CRS84\")"
        );
    }

    #[test]
    fn item_crs_kernel_arrays() {
        let tester = ScalarUdfTester::new(
            test_udf(WKB_GEOMETRY),
            vec![WKB_GEOMETRY_ITEM_CRS.clone(), WKB_GEOMETRY_ITEM_CRS.clone()],
        );

        let array_item_crs_lnglat = create_array_item_crs(
            &[
                Some("POINT (0 1)"),
                Some("POINT (2 3)"),
                Some("POINT (3 4)"),
            ],
            [Some("EPSG:4326"), Some("EPSG:4326"), Some("EPSG:4326")],
            &WKB_GEOMETRY,
        );
        let scalar_item_crs_4326 =
            create_scalar_item_crs(Some("POINT (0 1)"), Some("EPSG:4326"), &WKB_GEOMETRY);
        let scalar_item_crs_3857 =
            create_scalar_item_crs(Some("POINT (0 1)"), Some("EPSG:3857"), &WKB_GEOMETRY);

        // This should succeed when all CRS combinations are compatible
        let result = tester
            .invoke_array_scalar(array_item_crs_lnglat.clone(), scalar_item_crs_4326.clone())
            .unwrap();
        assert_eq!(&result, &array_item_crs_lnglat);

        // This should fail otherwise
        let err = tester
            .invoke_array_scalar(array_item_crs_lnglat.clone(), scalar_item_crs_3857.clone())
            .unwrap_err();
        assert_eq!(
            err.message(),
            "CRS values not equal: Some(\"EPSG:4326\") vs Some(\"EPSG:3857\")"
        );
    }

    #[test]
    fn item_crs_kernel_non_spatial_args_and_result() {
        let scalar_item_crs =
            create_scalar_item_crs(Some("POINT (0 1)"), Some("EPSG:4326"), &WKB_GEOMETRY);

        let tester = ScalarUdfTester::new(
            test_udf(SedonaType::Arrow(DataType::Int32)),
            vec![
                SedonaType::Arrow(DataType::Int32),
                WKB_GEOMETRY_ITEM_CRS.clone(),
            ],
        );
        tester.assert_return_type(DataType::Int32);

        let result = tester.invoke_scalar_scalar(1234, scalar_item_crs).unwrap();
        assert_eq!(result, ScalarValue::Int32(Some(1234)))
    }

    #[test]
    fn crs_args_equal() {
        // Zero args
        let err = ensure_crs_args_equal(&[]).unwrap_err();
        assert!(err.message().contains("Zero CRS arguments"));

        let crs_lnglat = ColumnarValue::Scalar(ScalarValue::Utf8(Some("EPSG:4326".to_string())));
        let crs_also_lnglat =
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("OGC:CRS84".to_string())));
        let crs_other = ColumnarValue::Scalar(ScalarValue::Utf8(Some("EPSG:3857".to_string())));

        // One arg
        let result_one_arg = ensure_crs_args_equal(&[&crs_lnglat]).unwrap();
        assert!(std::ptr::eq(result_one_arg, &crs_lnglat));

        // Two args (equal)
        let result_two_args = ensure_crs_args_equal(&[&crs_lnglat, &crs_also_lnglat]).unwrap();
        assert!(std::ptr::eq(result_two_args, &crs_lnglat));

        // Two args (not equal)
        let err = ensure_crs_args_equal(&[&crs_lnglat, &crs_other]).unwrap_err();
        assert_eq!(
            err.message(),
            "CRS values not equal: Some(\"EPSG:4326\") vs Some(\"EPSG:3857\")"
        );
    }
}
