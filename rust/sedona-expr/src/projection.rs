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
use arrow_schema::{DataType, Field, FieldRef};
use datafusion_physical_expr::expressions::{Column, Literal};
use datafusion_physical_expr::{PhysicalExpr, ScalarFunctionExpr};
use sedona_schema::projection::wrap_schema_maybe_deprecated;
use std::any::Any;
use std::sync::Arc;

use arrow_array::{new_null_array, RecordBatch, StructArray};
use datafusion_common::ScalarValue;
use datafusion_common::{DFSchema, Result};
use datafusion_expr::{
    ColumnarValue, Expr, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
};
use sedona_schema::{extension_type::ExtensionType, projection::unwrap_schema_maybe_deprecated};

/// Implementation underlying wrap_df
///
/// Returns None if there is no need to wrap the input, or a list of expressions that
/// either pass along the existing column or a UDF call that applies the wrap.
pub fn wrap_expressions(schema: &DFSchema) -> Result<Option<Vec<Expr>>> {
    let wrap_udf = WrapExtensionUdfMaybeDeprecated::udf();
    let mut wrap_count = 0;

    let mut exprs = Vec::with_capacity(schema.fields().len());
    for i in 0..schema.fields().len() {
        let this_column = Expr::Column(schema.columns()[i].clone());
        let (this_qualifier, this_field) = schema.qualified_field(i);

        if let Some(ext) = ExtensionType::from_field(schema.field(i)) {
            let dummy_scalar = dummy_scalar_value(&ext.to_data_type())?;
            let wrap_call = wrap_udf
                .call(vec![this_column.clone(), Expr::Literal(dummy_scalar, None)])
                .alias_qualified(this_qualifier.cloned(), this_field.name());

            exprs.push(wrap_call);
            wrap_count += 1;
        } else {
            exprs.push(this_column.alias_qualified(this_qualifier.cloned(), this_field.name()));
        }
    }

    if wrap_count > 0 {
        Ok(Some(exprs))
    } else {
        Ok(None)
    }
}

/// Implementation underlying unwrap_df
///
/// Returns None if there is no need to unwrap the input, or a list of expressions that
/// either pass along the existing column or a UDF call that applies the unwrap.
/// Returns a DFSchema because the resulting schema based purely on the expressions would
/// otherwise not include field metadata.
pub fn unwrap_expressions(schema: &DFSchema) -> Result<Option<(DFSchema, Vec<Expr>)>> {
    let unwrap_udf = UnwrapExtensionUdf::udf();
    let mut exprs = Vec::with_capacity(schema.fields().len());
    let mut qualifiers = Vec::with_capacity(exprs.capacity());
    let mut unwrap_count = 0;

    for i in 0..schema.fields().len() {
        let this_column = Expr::Column(schema.columns()[i].clone());
        let (this_qualifier, this_field) = schema.qualified_field(i);
        qualifiers.push(this_qualifier.cloned());

        if ExtensionType::from_data_type(this_field.data_type()).is_some() {
            let unwrap_call = unwrap_udf
                .call(vec![this_column.clone()])
                .alias_qualified(this_qualifier.cloned(), this_field.name());

            exprs.push(unwrap_call);
            unwrap_count += 1;
        } else {
            exprs.push(this_column.alias_qualified(this_qualifier.cloned(), this_field.name()));
        }
    }

    if unwrap_count > 0 {
        let schema_unwrapped = unwrap_schema_maybe_deprecated(schema.as_arrow());
        let dfschema_unwrapped = DFSchema::from_field_specific_qualified_schema(
            qualifiers,
            &Arc::new(schema_unwrapped),
        )?;

        Ok(Some((dfschema_unwrapped, exprs)))
    } else {
        Ok(None)
    }
}

/// Wrap physical expressions
///
/// Conceptually identical to [wrap_expressions] except with a [PhysicalExpr]
/// for use in places like TableProviders that are required to generate physical
/// plans. Allowing the complex return type because this won't need to exist after
/// DataFusion 48 is released.
#[allow(clippy::type_complexity)]
pub fn wrap_physical_expressions(
    projected_storage_fields: &[FieldRef],
) -> Result<Option<Vec<(Arc<dyn PhysicalExpr>, String)>>> {
    let wrap_udf = Arc::new(WrapExtensionUdfMaybeDeprecated::udf());
    let wrap_udf_name = wrap_udf.name().to_string();
    let mut wrap_count = 0;
    let exprs: Result<Vec<_>> = projected_storage_fields
        .iter()
        .enumerate()
        .map(|(i, f)| -> Result<(Arc<dyn PhysicalExpr>, String)> {
            let column = Arc::new(Column::new(f.name(), i));

            if let Some(ext) = ExtensionType::from_field(f) {
                wrap_count += 1;
                let dummy_scalar = dummy_scalar_value(&ext.to_data_type())?;
                let dummy_literal = Arc::new(Literal::new(dummy_scalar));
                Ok((
                    Arc::new(ScalarFunctionExpr::new(
                        &wrap_udf_name,
                        wrap_udf.clone(),
                        vec![column, dummy_literal],
                        Arc::new(Field::new("", ext.to_data_type(), f.is_nullable())),
                    )),
                    f.name().to_string(),
                ))
            } else {
                Ok((column, f.name().to_string()))
            }
        })
        .collect();

    if wrap_count > 0 {
        Ok(Some(exprs?))
    } else {
        Ok(None)
    }
}

/// Wrap a record batch possibly containing extension types encoded as field metadata
///
/// The resulting batch will wrap columns with extension types as struct arrays
/// that can be passed to APIs that operate purely on ArrayRefs (e.g., UDFs).
/// This is the projection that should be applied when wrapping an input stream.
pub fn wrap_batch_maybe_deprecated(batch: RecordBatch) -> RecordBatch {
    let columns = batch
        .columns()
        .iter()
        .enumerate()
        .map(|(i, column)| {
            if let Some(ext) = ExtensionType::from_field(batch.schema().field(i)) {
                ext.wrap_array(column.clone()).unwrap()
            } else {
                column.clone()
            }
        })
        .collect();

    let schema = wrap_schema_maybe_deprecated(&batch.schema());
    RecordBatch::try_new(Arc::new(schema), columns).unwrap()
}

/// Unwrap a record batch such that the output expresses extension types as fields
///
/// The resulting output will have extension types represented with field metadata
/// instead of as wrapped structs. This is the projection that should be applied
/// when writing to output.
pub fn unwrap_batch_maybe_deprecated(batch: RecordBatch) -> RecordBatch {
    let columns: Vec<_> = batch
        .columns()
        .iter()
        .map(|column| {
            if ExtensionType::from_data_type(column.data_type()).is_some() {
                let struct_array = StructArray::from(column.to_data());
                struct_array.column(0).clone()
            } else {
                column.clone()
            }
        })
        .collect();

    let schema = unwrap_schema_maybe_deprecated(&batch.schema());
    RecordBatch::try_new(Arc::new(schema), columns).unwrap()
}

/// For passing to the WrapExtensionUdf as a way for it to know what the return type
/// should be
fn dummy_scalar_value(data_type: &DataType) -> Result<ScalarValue> {
    let dummy_array = new_null_array(data_type, 1);
    ScalarValue::try_from_array(&dummy_array, 0)
}

#[derive(Debug)]
pub struct WrapExtensionUdfMaybeDeprecated {
    signature: Signature,
}

impl WrapExtensionUdfMaybeDeprecated {
    pub fn udf() -> ScalarUDF {
        let signature = Signature::any(2, datafusion_expr::Volatility::Immutable);
        ScalarUDF::new_from_impl(Self { signature })
    }
}

impl ScalarUDFImpl for WrapExtensionUdfMaybeDeprecated {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "wrap_extension_internal"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, args: &[DataType]) -> Result<DataType> {
        debug_assert_eq!(args.len(), 2);
        Ok(args[1].clone())
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        if let Some(extension_type) = ExtensionType::from_data_type(&args.args[1].data_type()) {
            extension_type.wrap_arg(&args.args[0])
        } else {
            Ok(args.args[0].clone())
        }
    }
}

#[derive(Debug)]
pub struct UnwrapExtensionUdf {
    signature: Signature,
}

impl UnwrapExtensionUdf {
    pub fn udf() -> ScalarUDF {
        let signature = Signature::any(1, datafusion_expr::Volatility::Immutable);
        ScalarUDF::new_from_impl(Self { signature })
    }
}

impl ScalarUDFImpl for UnwrapExtensionUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "unwrap_extension_internal"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, args: &[DataType]) -> Result<DataType> {
        debug_assert_eq!(args.len(), 1);
        if let Some(extension_type) = ExtensionType::from_data_type(&args[0]) {
            Ok(extension_type.to_field("", true).data_type().clone())
        } else {
            Ok(args[0].clone())
        }
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        if let Some(extension) = ExtensionType::from_data_type(&args.args[0].data_type()) {
            extension.unwrap_arg(&args.args[0])
        } else {
            Ok(args.args[0].clone())
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::create_array;
    use arrow_schema::{DataType, Field, Schema};

    use super::*;

    /// An ExtensionType for tests
    pub fn geoarrow_wkt() -> ExtensionType {
        ExtensionType::new("geoarrow.wkt", DataType::Utf8, None)
    }

    #[test]
    fn batch_wrap_unwrap() {
        let schema = Schema::new(vec![
            Field::new("col1", DataType::Utf8, false),
            geoarrow_wkt().to_field("col2", true),
        ]);

        let col1 = create_array!(Utf8, ["POINT (0 1)", "POINT (2, 3)"]);
        let col2 = col1.clone();

        let batch = RecordBatch::try_new(schema.into(), vec![col1, col2]).unwrap();
        let batch_wrapped = wrap_batch_maybe_deprecated(batch.clone());
        assert_eq!(batch_wrapped.column(0).data_type(), &DataType::Utf8);
        assert!(batch_wrapped.column(1).data_type().is_nested());

        let batch_unwrapped = unwrap_batch_maybe_deprecated(batch_wrapped);
        assert_eq!(batch_unwrapped, batch);
    }
}
