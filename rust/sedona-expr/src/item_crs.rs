use std::{fmt::Debug, iter::zip, sync::Arc};

use arrow_array::{ArrayRef, StructArray};
use arrow_schema::{DataType, Field};
use datafusion_common::{
    cast::{as_string_view_array, as_struct_array},
    internal_err, unwrap_or_internal_err, DataFusionError, Result, ScalarValue,
};
use datafusion_expr::ColumnarValue;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

use crate::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};

#[derive(Debug)]
pub struct ItemCrsKernel {
    inner: ScalarKernelRef,
}

impl ItemCrsKernel {
    pub fn new_ref(inner: ScalarKernelRef) -> ScalarKernelRef {
        Arc::new(Self { inner })
    }
}

impl SedonaScalarKernel for ItemCrsKernel {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        return_type_handle_item_crs(self.inner.as_ref(), args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        invoke_handle_item_crs(self.inner.as_ref(), arg_types, args)
    }
}

/// Propagate item crs types where appropriate
///
/// Most kernels that operate on
pub fn return_type_handle_item_crs(
    kernel: &dyn SedonaScalarKernel,
    arg_types: &[SedonaType],
) -> Result<Option<SedonaType>> {
    let item_crs_matcher = ArgMatcher::is_item_crs();
    if !arg_types
        .iter()
        .any(|arg_type| item_crs_matcher.match_type(arg_type))
    {
        return kernel.return_type(arg_types);
    }

    let item_arg_types = arg_types
        .iter()
        .map(|arg_type| parse_item_crs_arg_type(arg_type).map(|(item_type, _)| item_type))
        .collect::<Result<Vec<_>>>()?;

    if let Some(item_type) = kernel.return_type(&item_arg_types)? {
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

pub fn invoke_handle_item_crs(
    kernel: &dyn SedonaScalarKernel,
    arg_types: &[SedonaType],
    args: &[ColumnarValue],
) -> Result<ColumnarValue> {
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

    let out_item_type = kernel.return_type(&item_types)?;
    let out_item_type = unwrap_or_internal_err!(out_item_type);

    let item_result = kernel.invoke_batch(&item_types, &item_args)?;

    if ArgMatcher::is_geometry_or_geography().match_type(&out_item_type) {
        make_item_crs(&out_item_type, item_result, crs_result)
    } else {
        Ok(item_result)
    }
}

pub fn make_item_crs(
    item_type: &SedonaType,
    item_result: ColumnarValue,
    crs_result: ColumnarValue,
) -> Result<ColumnarValue> {
    let out_fields = vec![
        item_type.to_storage_field("item", true)?,
        Field::new("crs", DataType::Utf8View, true),
    ];
    match item_result {
        ColumnarValue::Array(item_array) => {
            let nulls = item_array.nulls().cloned();
            let crs_array = crs_result.to_array(item_array.len())?;
            let item_crs_array =
                StructArray::new(out_fields.into(), vec![item_array, crs_array], nulls);
            Ok(ColumnarValue::Array(Arc::new(item_crs_array)))
        }
        ColumnarValue::Scalar(item_scalar) => {
            let item_array = item_scalar.to_array()?;
            let nulls = item_array.nulls().cloned();
            let item_crs_array = StructArray::try_new(
                out_fields.into(),
                vec![item_array, crs_result.to_array(1)?],
                nulls,
            )?;
            Ok(ScalarValue::Struct(Arc::new(item_crs_array)).into())
        }
    }
}

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
                    internal_err!(
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
                ScalarValue::Null
            };

            Ok((arg.clone(), Some(ColumnarValue::Scalar(crs_scalar))))
        }
        _ => Ok((arg.clone(), None)),
    }
}

fn ensure_crs_args_equal(crs_args: &[&ColumnarValue]) -> Result<ColumnarValue> {
    match crs_args.len() {
        0 => internal_err!("Zero CRS arguments as input to item_crs"),
        1 => Ok(crs_args[0].clone()),
        _ => {
            let crs_args_string = crs_args
                .iter()
                .map(|arg| arg.cast_to(&DataType::Utf8View, None))
                .collect::<Result<Vec<_>>>()?;
            let crs_arrays = ColumnarValue::values_to_arrays(&crs_args_string)?;
            for i in 1..crs_arrays.len() {
                ensure_crs_string_arrays_equal2(&crs_arrays[i - 1], &crs_arrays[i])?
            }

            Ok(crs_args[0].clone())
        }
    }
}

fn ensure_crs_string_arrays_equal2(lhs: &ArrayRef, rhs: &ArrayRef) -> Result<()> {
    for (lhs_item, rhs_item) in zip(as_string_view_array(lhs)?, as_string_view_array(rhs)?) {
        if lhs_item != rhs_item {
            return Err(DataFusionError::Execution(format!(
                "CRS values not equal: {lhs_item:?} vs {rhs_item:?}"
            )));
        }
    }

    Ok(())
}
