use std::{any::Any, sync::Arc};

use abi_stable::StableAbi;
use arrow_schema::{DataType, Field, FieldRef, Schema};
use datafusion::physical_plan::{expressions::Column, PhysicalExpr};
use datafusion_common::{DataFusionError, Result, ScalarValue};
use datafusion_expr::{
    function::{AccumulatorArgs, StateFieldsArgs},
    Accumulator, AggregateUDF, AggregateUDFImpl, ColumnarValue, ReturnFieldArgs,
    ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
};
use datafusion_ffi::{
    udaf::{FFI_AggregateUDF, ForeignAggregateUDF},
    udf::{FFI_ScalarUDF, ForeignScalarUDF},
};
use sedona_common::sedona_internal_err;
use sedona_schema::datatypes::SedonaType;

use sedona_expr::{
    aggregate_udf::{SedonaAccumulator, SedonaAccumulatorRef},
    scalar_udf::{ScalarKernelRef, SedonaScalarKernel},
};

/// A stable struct for sharing [SedonaScalarKernel]s across FFI boundaries
///
/// The primary interface for importing or exporting these is `.from()`
/// and `.into()` between the [FFI_SedonaScalarKernel] and the [ScalarKernelRef].
///
/// Internally this struct uses the [FFI_ScalarUDF] from DataFusion's FFI
/// library to avoid having to invent an FFI ourselves. Like the [FFI_ScalarUDF],
/// this struct is only convenient to use when the libraries on both sides of
/// a boundary are written in Rust. Because Rust makes it relatively easy to
/// wrap C or C++ libraries, this should not be a barrier for most types of
/// kernels we might want to implement; however, it is also an option to
/// create our own FFI using simpler primitives if using DataFusion's
/// introduces performance or implementation issues.
#[repr(C)]
#[derive(Debug, StableAbi)]
#[allow(non_camel_case_types)]
pub struct FFI_SedonaScalarKernel {
    inner: FFI_ScalarUDF,
}

impl From<ScalarKernelRef> for FFI_SedonaScalarKernel {
    fn from(value: ScalarKernelRef) -> Self {
        let exported = ScalarUDF::new_from_impl(ExportedScalarKernel::from(value));
        FFI_SedonaScalarKernel {
            inner: Arc::new(exported).into(),
        }
    }
}

impl TryFrom<&FFI_SedonaScalarKernel> for ScalarKernelRef {
    type Error = DataFusionError;

    fn try_from(value: &FFI_SedonaScalarKernel) -> Result<Self> {
        Ok(Arc::new(ImportedScalarKernel::try_from(value)?))
    }
}

impl TryFrom<FFI_SedonaScalarKernel> for ScalarKernelRef {
    type Error = DataFusionError;

    fn try_from(value: FFI_SedonaScalarKernel) -> Result<Self> {
        Self::try_from(&value)
    }
}

#[derive(Debug)]
struct ExportedScalarKernel {
    name: String,
    signature: Signature,
    sedona_impl: ScalarKernelRef,
}

impl From<ScalarKernelRef> for ExportedScalarKernel {
    fn from(value: ScalarKernelRef) -> Self {
        Self {
            name: "ExportedScalarKernel".to_string(),
            signature: Signature::any(0, datafusion_expr::Volatility::Volatile),
            sedona_impl: value,
        }
    }
}

impl ScalarUDFImpl for ExportedScalarKernel {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        sedona_internal_err!("should not be called")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let sedona_types = args
            .arg_fields
            .iter()
            .map(|f| SedonaType::from_storage_field(f))
            .collect::<Result<Vec<_>>>()?;
        match self.sedona_impl.return_type(&sedona_types)? {
            Some(output_type) => Ok(output_type.to_storage_field("", true)?.into()),
            // Sedona kernels return None to indicate the kernel doesn't apply to the inputs,
            // but the ScalarUDFImpl doesn't have a way to natively indicate that. We use
            // NotImplemented with a special message and catch it on the other side.
            None => Err(DataFusionError::NotImplemented(
                "::kernel does not match input args::".to_string(),
            )),
        }
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let sedona_types = args
            .arg_fields
            .iter()
            .map(|f| SedonaType::from_storage_field(f))
            .collect::<Result<Vec<_>>>()?;
        self.sedona_impl.invoke_batch(&sedona_types, &args.args)
    }
}

#[derive(Debug)]
struct ImportedScalarKernel {
    udf_impl: ScalarUDF,
}

impl TryFrom<&FFI_SedonaScalarKernel> for ImportedScalarKernel {
    type Error = DataFusionError;

    fn try_from(value: &FFI_SedonaScalarKernel) -> Result<Self> {
        let wrapped = ForeignScalarUDF::try_from(&value.inner)?;
        Ok(Self {
            udf_impl: ScalarUDF::new_from_impl(wrapped),
        })
    }
}

impl SedonaScalarKernel for ImportedScalarKernel {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let df_args = ReturnFieldArgs {
            arg_fields: &args
                .iter()
                .map(|arg| arg.to_storage_field("", true).map(Arc::new))
                .collect::<Result<Vec<_>>>()?,
            scalar_arguments: &[],
        };
        match self.udf_impl.return_field_from_args(df_args) {
            Ok(field) => Ok(Some(SedonaType::from_storage_field(&field)?)),
            Err(err) => {
                if matches!(err, DataFusionError::NotImplemented(_)) {
                    Ok(None)
                } else {
                    Err(err)
                }
            }
        }
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let arg_rows = Self::output_size(args);

        let scalar_fn_args = ScalarFunctionArgs {
            args: args.to_vec(),
            arg_fields: arg_types
                .iter()
                .map(|arg| arg.to_storage_field("", true).map(Arc::new))
                .collect::<Result<Vec<_>>>()?,
            number_rows: arg_rows.unwrap_or(1),
            // Wrapper code on the other side of this doesn't use this value
            return_field: Field::new("", DataType::Null, true).into(),
        };

        // DataFusion's FFI_ScalarUDF always returns array output but
        // our original UDFs were careful to return ScalarValues.
        match self.udf_impl.invoke_with_args(scalar_fn_args)? {
            ColumnarValue::Array(array) => match arg_rows {
                Some(_) => Ok(ColumnarValue::Array(array)),
                None => Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(
                    &array, 0,
                )?)),
            },
            ColumnarValue::Scalar(scalar_value) => {
                // This branch is probably never taken but may in the future
                Ok(ColumnarValue::Scalar(scalar_value))
            }
        }
    }
}

impl ImportedScalarKernel {
    fn output_size(args: &[ColumnarValue]) -> Option<usize> {
        for original_arg in args {
            if let ColumnarValue::Array(array) = original_arg {
                return Some(array.len());
            }
        }
        None
    }
}

/// A stable struct for sharing [SedonaAccumulator]s across FFI boundaries
///
/// The primary interface for importing or exporting these is `.from()`
/// and `.into()` between the [FFI_SedonaAggregateKernel] and the [SedonaAccumulatorRef].
///
/// Internally this struct uses the [FFI_AggregateUDF] from DataFusion's FFI
/// library to avoid having to invent an FFI ourselves. See [FFI_SedonaScalarKernel]
/// for general information about the rationale and usage of FFI implementations.
#[repr(C)]
#[derive(Debug, StableAbi)]
#[allow(non_camel_case_types)]
pub struct FFI_SedonaAggregateKernel {
    inner: FFI_AggregateUDF,
}

impl From<SedonaAccumulatorRef> for FFI_SedonaAggregateKernel {
    fn from(value: SedonaAccumulatorRef) -> Self {
        let exported: AggregateUDF = ExportedSedonaAccumulator::from(value).into();
        FFI_SedonaAggregateKernel {
            inner: Arc::new(exported).into(),
        }
    }
}

impl TryFrom<&FFI_SedonaAggregateKernel> for SedonaAccumulatorRef {
    type Error = DataFusionError;

    fn try_from(value: &FFI_SedonaAggregateKernel) -> Result<Self> {
        Ok(Arc::new(ImportedSedonaAccumulator::try_from(value)?))
    }
}

impl TryFrom<FFI_SedonaAggregateKernel> for SedonaAccumulatorRef {
    type Error = DataFusionError;

    fn try_from(value: FFI_SedonaAggregateKernel) -> Result<Self> {
        Self::try_from(&value)
    }
}

#[derive(Debug)]
struct ExportedSedonaAccumulator {
    name: String,
    signature: Signature,
    sedona_impl: SedonaAccumulatorRef,
}

impl From<SedonaAccumulatorRef> for ExportedSedonaAccumulator {
    fn from(value: SedonaAccumulatorRef) -> Self {
        Self {
            name: "ExportedSedonaAccumulator".to_string(),
            signature: Signature::any(0, datafusion_expr::Volatility::Volatile),
            sedona_impl: value,
        }
    }
}

impl AggregateUDFImpl for ExportedSedonaAccumulator {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    // We have to use return_type() with struct-wrapped types instead of
    // return_field() because the FFI Aggregate Function doesn't yet use
    // return_field().
    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        let sedona_types = arg_types
            .iter()
            .map(SedonaType::from_data_type)
            .collect::<Result<Vec<_>>>()?;
        match self.sedona_impl.return_type(&sedona_types)? {
            Some(output_type) => Ok(output_type.data_type()),
            // Sedona kernels return None to indicate the kernel doesn't apply to the inputs,
            // but the ScalarUDFImpl doesn't have a way to natively indicate that. We use
            // NotImplemented with a special message and catch it on the other side.
            None => Err(DataFusionError::NotImplemented(
                "::kernel does not match input args::".to_string(),
            )),
        }
    }

    fn accumulator(&self, acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        let arg_fields = acc_args
            .exprs
            .iter()
            .map(|expr| expr.return_field(acc_args.schema))
            .collect::<Result<Vec<_>>>()?;
        let sedona_types = arg_fields
            .iter()
            .map(|f| SedonaType::from_data_type(f.data_type()))
            .collect::<Result<Vec<_>>>()?;
        if let Some(output_type) = self.sedona_impl.return_type(&sedona_types)? {
            self.sedona_impl.accumulator(&sedona_types, &output_type)
        } else {
            Err(DataFusionError::NotImplemented(
                "::kernel does not match input args::".to_string(),
            ))
        }
    }

    fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<FieldRef>> {
        let sedona_types = args
            .input_fields
            .iter()
            .map(|f| SedonaType::from_data_type(f.data_type()))
            .collect::<Result<Vec<_>>>()?;
        self.sedona_impl.state_fields(&sedona_types)
    }
}

#[derive(Debug)]
struct ImportedSedonaAccumulator {
    aggregate_impl: AggregateUDF,
}

impl TryFrom<&FFI_SedonaAggregateKernel> for ImportedSedonaAccumulator {
    type Error = DataFusionError;

    fn try_from(value: &FFI_SedonaAggregateKernel) -> Result<Self> {
        let wrapped = ForeignAggregateUDF::try_from(&value.inner)?;
        Ok(Self {
            aggregate_impl: wrapped.into(),
        })
    }
}

impl SedonaAccumulator for ImportedSedonaAccumulator {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let arg_fields = args
            .iter()
            .map(|arg| Arc::new(Field::new("", arg.data_type(), true)))
            .collect::<Vec<_>>();

        match self.aggregate_impl.return_field(&arg_fields) {
            Ok(field) => Ok(Some(SedonaType::from_storage_field(&field)?)),
            Err(err) => {
                if matches!(err, DataFusionError::NotImplemented(_)) {
                    Ok(None)
                } else {
                    Err(err)
                }
            }
        }
    }

    fn accumulator(
        &self,
        args: &[SedonaType],
        output_type: &SedonaType,
    ) -> Result<Box<dyn Accumulator>> {
        let arg_fields = args
            .iter()
            .map(|arg| Arc::new(Field::new("", arg.data_type(), true)))
            .collect::<Vec<_>>();
        let mock_schema = Schema::new(arg_fields);
        let exprs = (0..mock_schema.fields().len())
            .map(|i| -> Arc<dyn PhysicalExpr> { Arc::new(Column::new("col", i)) })
            .collect::<Vec<_>>();

        let return_field = Field::new("", output_type.data_type(), true);

        let args = AccumulatorArgs {
            return_field: return_field.into(),
            schema: &mock_schema,
            ignore_nulls: true,
            order_bys: &[],
            is_reversed: false,
            name: "",
            is_distinct: false,
            exprs: &exprs,
        };

        self.aggregate_impl.accumulator(args)
    }

    fn state_fields(&self, args: &[SedonaType]) -> Result<Vec<FieldRef>> {
        let arg_fields = args
            .iter()
            .map(|arg| Arc::new(Field::new("", arg.data_type(), true)))
            .collect::<Vec<_>>();

        let state_field_args = StateFieldsArgs {
            name: "",
            input_fields: &arg_fields,
            return_field: Arc::new(Field::new("", DataType::Null, false)),
            ordering_fields: &[],
            is_distinct: false,
        };

        self.aggregate_impl.state_fields(state_field_args)
    }
}

#[cfg(test)]
mod test {
    use datafusion_expr::Volatility;
    use sedona_expr::{
        aggregate_udf::SedonaAggregateUDF,
        scalar_udf::{ArgMatcher, SedonaScalarUDF, SimpleSedonaScalarKernel},
    };
    use sedona_functions::st_envelope_aggr::st_envelope_aggr_udf;
    use sedona_schema::datatypes::WKB_GEOMETRY;
    use sedona_testing::{
        compare::{assert_scalar_equal, assert_value_equal},
        create::{create_array, create_array_value, create_scalar, create_scalar_value},
        testers::AggregateUdfTester,
    };

    use super::*;

    #[test]
    fn ffi_roundtrip() {
        let kernel = SimpleSedonaScalarKernel::new_ref(
            ArgMatcher::new(vec![ArgMatcher::is_geometry()], WKB_GEOMETRY),
            Arc::new(|_, args| Ok(args[0].clone())),
        );

        let scalar_value = create_scalar_value(Some("POINT (0 1)"), &WKB_GEOMETRY);
        let array_value = create_array_value(&[Some("POINT (0 1)"), None], &WKB_GEOMETRY);

        let udf_native = SedonaScalarUDF::new(
            "simple_udf",
            vec![kernel.clone()],
            Volatility::Immutable,
            None,
        );

        assert_value_equal(
            &udf_native
                .invoke_batch(std::slice::from_ref(&scalar_value), 1)
                .unwrap(),
            &scalar_value,
        );

        assert_value_equal(
            &udf_native
                .invoke_batch(std::slice::from_ref(&array_value), 1)
                .unwrap(),
            &array_value,
        );

        let ffi_kernel = FFI_SedonaScalarKernel::from(kernel.clone());
        let udf_from_ffi = SedonaScalarUDF::new(
            "simple_udf_from_ffi",
            vec![ffi_kernel.try_into().unwrap()],
            Volatility::Immutable,
            None,
        );

        assert_value_equal(
            &udf_from_ffi
                .invoke_batch(std::slice::from_ref(&scalar_value), 1)
                .unwrap(),
            &scalar_value,
        );

        assert_value_equal(
            &udf_from_ffi
                .invoke_batch(std::slice::from_ref(&array_value), 1)
                .unwrap(),
            &array_value,
        );
    }

    #[test]
    fn ffi_aggregate_roundtrip() {
        let agg = st_envelope_aggr_udf();
        let array_value = create_array(&[Some("POINT (0 1)"), None], &WKB_GEOMETRY);
        let scalar_envelope = create_scalar(Some("POINT (0 1)"), &WKB_GEOMETRY);

        // Check aggregation without FFI
        let tester = AggregateUdfTester::new(agg.clone().into(), vec![WKB_GEOMETRY]);
        assert_eq!(tester.return_type().unwrap(), WKB_GEOMETRY);
        assert_scalar_equal(
            &tester.aggregate(&vec![array_value.clone()]).unwrap(),
            &scalar_envelope,
        );

        // Check aggregation roundtrip through FFI
        let ffi_kernel = FFI_SedonaAggregateKernel::from(agg.kernels()[0].clone());
        let agg_from_ffi = SedonaAggregateUDF::new(
            "simple_agg_from_ffi",
            vec![ffi_kernel.try_into().unwrap()],
            Volatility::Immutable,
            None,
        );

        let tester = AggregateUdfTester::new(agg_from_ffi.into(), vec![WKB_GEOMETRY]);
        assert_eq!(tester.return_type().unwrap(), WKB_GEOMETRY);
        assert_scalar_equal(
            &tester.aggregate(&vec![array_value.clone()]).unwrap(),
            &scalar_envelope,
        );
    }
}
