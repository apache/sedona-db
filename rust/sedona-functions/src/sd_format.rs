use std::{sync::Arc, vec};

use crate::executor::WkbExecutor;
use arrow_array::{
    builder::StringBuilder, cast::AsArray, Array, GenericListArray, OffsetSizeTrait, StructArray,
};
use arrow_schema::{DataType, Field, Fields};
use datafusion_common::{
    error::{DataFusionError, Result},
    internal_err, ScalarValue,
};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use sedona_expr::scalar_udf::{ArgMatcher, SedonaScalarKernel, SedonaScalarUDF};
use sedona_schema::datatypes::SedonaType;

/// SD_Format() scalar UDF implementation
///
/// This function is invoked to obtain a proxy array with human-readable
/// output. For most arrays, this just returns the array (which will be
/// formatted using its storage type by the Arrow formatter).
pub fn sd_format_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "sd_format",
        vec![Arc::new(SDFormatDefault {})],
        Volatility::Immutable,
        Some(sd_format_doc()),
    )
}

fn sd_format_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Return a version of value suitable for formatting/display with
         the options provided. This is used to inject custom behaviour for a
         SedonaType specifically for formatting values.",
        "SD_Format (value: Any, [options: String])",
    )
    .with_argument("value", "Any: Any input value")
    .with_argument(
        "options",
        "
    String: JSON-encoded options. The following options are currently supported:

    - width_hint (numeric): The approximate width of the output. The value provided will
      typically be an overestimate and the value may be further abrevidated by
      the renderer. This value is purely a hint and may be ignored.",
    )
    .with_sql_example("SELECT SD_Format(ST_Point(1.0, 2.0, '{}'))")
    .build()
}

/// Default implementation that returns its input (i.e., by default, just
/// do whatever DataFusion would have done with the value and ignore any
/// options that were provided)
#[derive(Debug)]
struct SDFormatDefault {}

impl SedonaScalarKernel for SDFormatDefault {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let sedona_type = &args[0];
        let formatted_type = sedona_type_to_formatted_type(sedona_type)?;
        if formatted_type == *sedona_type {
            return Ok(None);
        }
        let matcher = ArgMatcher::new(
            vec![
                ArgMatcher::is_any(),
                ArgMatcher::is_optional(ArgMatcher::is_string()),
            ],
            formatted_type,
        );
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let mut maybe_width_hint: Option<usize> = None;
        if args.len() >= 2 {
            if let ColumnarValue::Scalar(ScalarValue::Utf8(Some(options_value))) =
                args[1].cast_to(&DataType::Utf8, None)?
            {
                let options: serde_json::Value = options_value
                    .parse()
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
                if let Some(width_hint_value) = options.get("width_hint") {
                    if let Some(width_hint_i64) = width_hint_value.as_i64() {
                        maybe_width_hint = Some(
                            width_hint_i64
                                .try_into()
                                .map_err(|e| DataFusionError::External(Box::new(e)))?,
                        );
                    }
                }
            }
        }

        columnar_value_to_formatted_value(&arg_types[0], &args[0], maybe_width_hint)
    }
}

fn sedona_type_to_formatted_type(sedona_type: &SedonaType) -> Result<SedonaType> {
    match sedona_type {
        SedonaType::Wkb(_, _) | SedonaType::WkbView(_, _) => Ok(SedonaType::Arrow(DataType::Utf8)),
        SedonaType::Arrow(arrow_type) => {
            // dive into the arrow type and translate geospatial types into Utf8
            match arrow_type {
                DataType::Struct(fields) => {
                    let mut new_fields = Vec::with_capacity(fields.len());
                    for field in fields {
                        let new_field = field_to_formatted_field(field)?;
                        new_fields.push(Arc::new(new_field));
                    }
                    Ok(SedonaType::Arrow(DataType::Struct(new_fields.into())))
                }
                DataType::List(field) => {
                    let new_field = field_to_formatted_field(field)?;
                    Ok(SedonaType::Arrow(DataType::List(Arc::new(new_field))))
                }
                DataType::ListView(field) => {
                    let new_field = field_to_formatted_field(field)?;
                    Ok(SedonaType::Arrow(DataType::ListView(Arc::new(new_field))))
                }
                _ => Ok(sedona_type.clone()),
            }
        }
    }
}

fn field_to_formatted_field(field: &Field) -> Result<Field> {
    let new_type = sedona_type_to_formatted_type(&SedonaType::from_data_type(field.data_type())?)?;
    let new_field = field.clone().with_data_type(new_type.data_type());
    Ok(new_field)
}

fn columnar_value_to_formatted_value(
    sedona_type: &SedonaType,
    columnar_value: &ColumnarValue,
    maybe_width_hint: Option<usize>,
) -> Result<ColumnarValue> {
    match sedona_type {
        SedonaType::Wkb(_, _) | SedonaType::WkbView(_, _) => {
            geospatial_value_to_formatted_value(sedona_type, columnar_value, maybe_width_hint)
        }
        SedonaType::Arrow(arrow_type) => match arrow_type {
            DataType::Struct(fields) => match columnar_value {
                ColumnarValue::Array(array) => {
                    let struct_array = array.as_struct();
                    let formatted_struct_array =
                        struct_value_to_formatted_value(fields, struct_array, maybe_width_hint)?;
                    Ok(ColumnarValue::Array(Arc::new(formatted_struct_array)))
                }
                ColumnarValue::Scalar(ScalarValue::Struct(struct_array)) => {
                    let formatted_struct_array =
                        struct_value_to_formatted_value(fields, struct_array, maybe_width_hint)?;
                    Ok(ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(
                        formatted_struct_array,
                    ))))
                }
                _ => internal_err!("Unsupported struct columnar value"),
            },
            DataType::List(field) => match columnar_value {
                ColumnarValue::Array(array) => {
                    let list_array = array.as_list::<i32>();
                    let formatted_list_array =
                        list_value_to_formatted_value(field, list_array, maybe_width_hint)?;
                    Ok(ColumnarValue::Array(Arc::new(formatted_list_array)))
                }
                ColumnarValue::Scalar(ScalarValue::List(list_array)) => {
                    let formatted_list_array =
                        list_value_to_formatted_value(field, list_array, maybe_width_hint)?;
                    Ok(ColumnarValue::Scalar(ScalarValue::List(Arc::new(
                        formatted_list_array,
                    ))))
                }
                _ => internal_err!("Unsupported list columnar value"),
            },
            _ => Ok(columnar_value.clone()),
        },
    }
}

/// Implementation format geometry or geography
///
/// This is very similar to ST_AsText except it respects the width_hint by
/// stopping the render for each item when too many characters have been written.
fn geospatial_value_to_formatted_value(
    sedona_type: &SedonaType,
    geospatial_value: &ColumnarValue,
    maybe_width_hint: Option<usize>,
) -> Result<ColumnarValue> {
    let arg_types: &[SedonaType] = std::slice::from_ref(sedona_type);
    let args: &[ColumnarValue] = std::slice::from_ref(geospatial_value);
    let executor = WkbExecutor::new(arg_types, args);

    let min_output_size = match maybe_width_hint {
        Some(width_hint) => executor.num_iterations() * width_hint,
        None => executor.num_iterations() * 25,
    };

    // Initialize an output builder of the appropriate type
    let mut builder = StringBuilder::with_capacity(executor.num_iterations(), min_output_size);

    executor.execute_wkb_void(|maybe_item| {
        match maybe_item {
            Some(item) => {
                let mut builder_wrapper =
                    LimitedSizeOutput::new(&mut builder, maybe_width_hint.unwrap_or(usize::MAX));

                // We ignore this error on purpose: we raised it on purpose to prevent
                // the WKT writer from writing too many characters
                #[allow(unused_must_use)]
                wkt::to_wkt::write_geometry(&mut builder_wrapper, &item);

                builder.append_value("");
            }
            None => builder.append_null(),
        };

        Ok(())
    })?;

    executor.finish(Arc::new(builder.finish()))
}

fn struct_value_to_formatted_value(
    fields: &Fields,
    struct_array: &StructArray,
    maybe_width_hint: Option<usize>,
) -> Result<StructArray> {
    let columns = struct_array.columns();

    let mut new_fields = Vec::with_capacity(columns.len());
    for (column, field) in columns.iter().zip(fields) {
        let new_field = field_to_formatted_field(field)?;
        let new_column = columnar_value_to_formatted_value(
            &SedonaType::from_data_type(field.data_type())?,
            &ColumnarValue::Array(Arc::clone(column)),
            maybe_width_hint,
        )?;

        let ColumnarValue::Array(new_array) = new_column else {
            return internal_err!("Expected Array");
        };

        new_fields.push((Arc::new(new_field), new_array));
    }

    Ok(StructArray::from(new_fields))
}

fn list_value_to_formatted_value<OffsetSize: OffsetSizeTrait>(
    field: &Field,
    list_array: &GenericListArray<OffsetSize>,
    maybe_width_hint: Option<usize>,
) -> Result<GenericListArray<OffsetSize>> {
    let values_array = list_array.values();
    let offsets = list_array.offsets();
    let nulls = list_array.nulls();

    let new_field = field_to_formatted_field(field)?;
    let new_columnar_value = columnar_value_to_formatted_value(
        &SedonaType::from_data_type(field.data_type())?,
        &ColumnarValue::Array(Arc::clone(values_array)),
        maybe_width_hint,
    )?;
    let ColumnarValue::Array(new_values_array) = new_columnar_value else {
        return internal_err!("Expected Array");
    };

    Ok(GenericListArray::<OffsetSize>::new(
        Arc::new(new_field),
        offsets.clone(),
        new_values_array,
        nulls.cloned(),
    ))
}

struct LimitedSizeOutput<'a, T> {
    inner: &'a mut T,
    current_item_size: usize,
    max_item_size: usize,
}

impl<'a, T> LimitedSizeOutput<'a, T> {
    pub fn new(inner: &'a mut T, max_item_size: usize) -> Self {
        Self {
            inner,
            current_item_size: 0,
            max_item_size,
        }
    }
}

impl<'a, T: std::fmt::Write> std::fmt::Write for LimitedSizeOutput<'a, T> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.inner.write_str(s)?;
        self.current_item_size += s.len();
        if self.current_item_size > self.max_item_size {
            Err(std::fmt::Error)
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{create_array, StringArray};
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::{
        WKB_GEOGRAPHY, WKB_GEOMETRY, WKB_VIEW_GEOGRAPHY, WKB_VIEW_GEOMETRY,
    };
    use sedona_testing::{create::create_array, testers::ScalarUdfTester};

    use super::*;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = sd_format_udf().into();
        assert_eq!(udf.name(), "sd_format");
        assert!(udf.documentation().is_some())
    }

    #[rstest]
    fn udf(
        #[values(WKB_GEOMETRY, WKB_GEOGRAPHY, WKB_VIEW_GEOMETRY, WKB_VIEW_GEOGRAPHY)]
        sedona_type: SedonaType,
    ) {
        use arrow_array::ArrayRef;

        let udf = sd_format_udf();
        let unary_tester = ScalarUdfTester::new(udf.clone().into(), vec![sedona_type.clone()]);
        let binary_tester = ScalarUdfTester::new(
            udf.clone().into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Utf8)],
        );

        // With omitted, Null, or invalid options, the output should be identical
        let wkt_values = vec![Some("POINT(1 2)"), None, Some("LINESTRING(3 5,7 8)")];
        let wkt_array = create_array(&wkt_values, &sedona_type);
        let expected_array: ArrayRef = Arc::new(wkt_values.iter().collect::<StringArray>());

        assert_eq!(
            &unary_tester.invoke_wkb_array(wkt_values.clone()).unwrap(),
            &expected_array
        );
        assert_eq!(
            &binary_tester
                .invoke_array_scalar(wkt_array.clone(), "{}")
                .unwrap(),
            &expected_array
        );
        assert_eq!(
            &binary_tester
                .invoke_array_scalar(wkt_array.clone(), ScalarValue::Null)
                .unwrap(),
            &expected_array
        );

        // Invalid options should error
        let err = binary_tester
            .invoke_array_scalar(wkt_array.clone(), r#"{"width_hint": -1}"#)
            .unwrap_err();
        assert_eq!(
            err.message(),
            "out of range integral type conversion attempted"
        );

        // For a very small width hint, we should get truncated values
        let expected_array: ArrayRef =
            create_array!(Utf8, [Some("POINT"), None, Some("LINESTRING")]);
        assert_eq!(
            &binary_tester
                .invoke_array_scalar(wkt_array.clone(), r#"{"width_hint": 3}"#)
                .unwrap(),
            &expected_array
        );
    }
}
