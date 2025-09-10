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
use std::{sync::Arc, vec};

use crate::executor::WkbExecutor;
use arrow_array::{
    builder::StringBuilder, cast::AsArray, Array, GenericListArray, GenericListViewArray,
    OffsetSizeTrait, StructArray,
};
use arrow_schema::{DataType, Field, Fields};
use datafusion_common::{
    error::{DataFusionError, Result},
    internal_err, ScalarValue,
};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

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

        let formatted_type = sedona_type_to_formatted_type(&arg_types[0])?;
        if formatted_type == arg_types[0] {
            // No change in type, the input data does not have geospatial columns that we can format,
            // so just return the input value
            return Ok(args[0].clone());
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
    let new_type = sedona_type_to_formatted_type(&SedonaType::from_storage_field(field)?)?;
    new_type.to_storage_field(field.name(), field.is_nullable())
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
            DataType::ListView(field) => match columnar_value {
                ColumnarValue::Array(array) => {
                    let list_array = array.as_list_view::<i32>();
                    let formatted_list_array =
                        list_view_value_to_formatted_value(field, list_array, maybe_width_hint)?;
                    Ok(ColumnarValue::Array(Arc::new(formatted_list_array)))
                }
                _ => internal_err!("Unsupported list view columnar value"),
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
        let sedona_type = SedonaType::from_storage_field(field)?;
        let new_column = columnar_value_to_formatted_value(
            &sedona_type,
            &ColumnarValue::Array(column.clone()),
            maybe_width_hint,
        )?;

        let ColumnarValue::Array(new_array) = new_column else {
            return internal_err!(
                "Expected Array in struct field formatting, got: {:?}",
                new_column
            );
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
    let sedona_type = SedonaType::from_storage_field(field)?;
    let new_columnar_value = columnar_value_to_formatted_value(
        &sedona_type,
        &ColumnarValue::Array(values_array.clone()),
        maybe_width_hint,
    )?;

    let ColumnarValue::Array(new_values_array) = new_columnar_value else {
        return internal_err!(
            "Expected Array when formatting list for field '{}', but got: {:?}",
            field.name(),
            new_columnar_value
        );
    };

    Ok(GenericListArray::<OffsetSize>::new(
        Arc::new(new_field),
        offsets.clone(),
        new_values_array,
        nulls.cloned(),
    ))
}

fn list_view_value_to_formatted_value<OffsetSize: OffsetSizeTrait>(
    field: &Field,
    list_view_array: &GenericListViewArray<OffsetSize>,
    maybe_width_hint: Option<usize>,
) -> Result<GenericListViewArray<OffsetSize>> {
    let values_array = list_view_array.values();
    let offsets = list_view_array.offsets();
    let sizes = list_view_array.sizes();
    let nulls = list_view_array.nulls();

    let new_field = field_to_formatted_field(field)?;
    let sedona_type = SedonaType::from_storage_field(field)?;
    let new_columnar_value = columnar_value_to_formatted_value(
        &sedona_type,
        &ColumnarValue::Array(values_array.clone()),
        maybe_width_hint,
    )?;

    let ColumnarValue::Array(new_values_array) = new_columnar_value else {
        return internal_err!(
            "Expected Array during list view formatting for field '{}' of type '{}'",
            field.name(),
            field.data_type()
        );
    };

    Ok(GenericListViewArray::<OffsetSize>::new(
        Arc::new(new_field),
        offsets.clone(),
        sizes.clone(),
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
    use arrow_array::{
        create_array, ArrayRef, Float64Array, Int32Array, ListArray, ListViewArray, StringArray,
        StructArray,
    };
    use arrow_schema::{DataType, Field};
    use datafusion::arrow::buffer::{OffsetBuffer, ScalarBuffer};
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::{
        WKB_GEOGRAPHY, WKB_GEOMETRY, WKB_VIEW_GEOGRAPHY, WKB_VIEW_GEOMETRY,
    };
    use sedona_testing::{create::create_array, testers::ScalarUdfTester};
    use std::sync::Arc;

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

    #[test]
    fn sd_format_does_not_format_non_spatial_columns() {
        let udf = sd_format_udf();

        // Define test cases as (description, array, expected_data_type)
        let test_cases: Vec<(&str, ArrayRef, DataType)> = vec![
            // Float64Array
            (
                "Float64Array",
                Arc::new(Float64Array::from(vec![Some(1.5), None, Some(3.16)])),
                DataType::Float64,
            ),
            // StructArray with mixed types
            (
                "StructArray",
                {
                    let struct_fields = vec![
                        Arc::new(Field::new("float_field", DataType::Float64, true)),
                        Arc::new(Field::new("int_field", DataType::Int32, false)),
                    ];
                    let float_col: ArrayRef =
                        Arc::new(Float64Array::from(vec![Some(1.1), Some(2.2), None]));
                    let int_col: ArrayRef = Arc::new(Int32Array::from(vec![10, 20, 30]));
                    Arc::new(StructArray::new(
                        struct_fields.clone().into(),
                        vec![float_col, int_col],
                        None,
                    ))
                },
                DataType::Struct(
                    vec![
                        Arc::new(Field::new("float_field", DataType::Float64, true)),
                        Arc::new(Field::new("int_field", DataType::Int32, false)),
                    ]
                    .into(),
                ),
            ),
            // String array using create_array! macro
            (
                "String array",
                create_array!(Utf8, [Some("hello"), None, Some("world")]),
                DataType::Utf8,
            ),
            // List array with Int32 elements
            (
                "List array",
                {
                    let int_values = Int32Array::from(vec![Some(42), None, Some(100), Some(200)]);
                    let field = Arc::new(Field::new("item", DataType::Int32, true));
                    let offsets = OffsetBuffer::new(vec![0, 2, 2, 4].into()); // [0,2), [2,2), [2,4)
                    Arc::new(ListArray::new(
                        field.clone(),
                        offsets,
                        Arc::new(int_values),
                        None,
                    ))
                },
                DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
            ),
            // List view array with Int32 elements
            (
                "List view array",
                {
                    let int_values = Int32Array::from(vec![Some(10), Some(20), Some(30)]);
                    let field = Arc::new(Field::new("item", DataType::Int32, true));
                    let offsets = ScalarBuffer::from(vec![0i32, 1i32, 2i32]); // Start offsets
                    let sizes = ScalarBuffer::from(vec![1i32, 1i32, 1i32]); // Sizes
                    Arc::new(ListViewArray::new(
                        field.clone(),
                        offsets,
                        sizes,
                        Arc::new(int_values),
                        None,
                    ))
                },
                DataType::ListView(Arc::new(Field::new("item", DataType::Int32, true))),
            ),
        ];

        for (description, test_array, expected_data_type) in test_cases {
            let tester = ScalarUdfTester::new(
                udf.clone().into(),
                vec![SedonaType::Arrow(expected_data_type.clone())],
            );
            let result = tester.invoke_array(test_array.clone()).unwrap();
            if !matches!(expected_data_type, DataType::ListView(_)) {
                assert_eq!(&result, &test_array, "Failed for test case: {description}");
            }
        }
    }

    #[rstest]
    fn sd_format_should_format_spatial_columns(
        #[values(WKB_GEOMETRY, WKB_GEOGRAPHY, WKB_VIEW_GEOMETRY, WKB_VIEW_GEOGRAPHY)]
        sedona_type: SedonaType,
    ) {
        let udf = sd_format_udf();

        // Create geometry storage array (without wrapping)
        let geometry_values = vec![Some("POINT(1 2)"), None, Some("LINESTRING(0 0, 1 1)")];
        let geometry_array = create_array(&geometry_values, &sedona_type);

        // Create non-spatial array
        let int_array: ArrayRef = Arc::new(Int32Array::from(vec![10, 20, 30]));
        let struct_fields = vec![
            Arc::new(sedona_type.to_storage_field("geom", true).unwrap()),
            Arc::new(Field::new("id", DataType::Int32, false)),
        ];
        let struct_array = StructArray::new(
            struct_fields.clone().into(),
            vec![geometry_array, int_array.clone()],
            None,
        );

        // Create tester
        let input_sedona_type = SedonaType::Arrow(DataType::Struct(struct_fields.into()));
        let tester = ScalarUdfTester::new(udf.clone().into(), vec![input_sedona_type]);

        // Test the function
        let result = tester.invoke_array(Arc::new(struct_array.clone())).unwrap();

        // Verify the result structure
        let result_struct = result.as_struct();
        assert_eq!(result_struct.num_columns(), 2);

        // First column should be formatted to UTF8 (geometry -> string)
        let geometry_column = result_struct.column(0);
        assert_eq!(geometry_column.data_type(), &DataType::Utf8);

        // Second column should remain Int32 (unchanged)
        let id_column = result_struct.column(1);
        assert_eq!(id_column.data_type(), &DataType::Int32);
        assert_eq!(id_column, &int_array);

        // Check if it's actually formatted
        if geometry_column.data_type() == &DataType::Utf8 {
            // Verify the geometry was actually formatted to WKT strings
            let string_array = geometry_column.as_string::<i32>();
            assert_wkt_values_match(string_array, &geometry_values);
        } else {
            // If not UTF8, this test should fail but let's see what we got
            panic!(
                "Geometry column was not formatted to UTF8. Got: {:?}",
                geometry_column.data_type()
            );
        }
    }

    #[rstest]
    fn sd_format_should_handle_both_spatial_and_non_spatial_columns(
        #[values(WKB_GEOMETRY, WKB_GEOGRAPHY, WKB_VIEW_GEOMETRY, WKB_VIEW_GEOGRAPHY)]
        sedona_type: SedonaType,
    ) {
        let udf = sd_format_udf();

        // Create geometry array
        let geog_values = vec![Some("POLYGON((0 0,1 0,1 1,0 1,0 0))"), Some("POINT(1 2)")];
        let geog_array = create_array(&geog_values, &sedona_type);

        // Create string array
        let name_array: ArrayRef =
            Arc::new(StringArray::from(vec![Some("feature1"), Some("feature2")]));

        // Create boolean array
        let active_array: ArrayRef = Arc::new(arrow_array::BooleanArray::from(vec![
            Some(true),
            Some(false),
        ]));

        // Create struct array with proper extension metadata
        let struct_fields = vec![
            Arc::new(sedona_type.to_storage_field("geom", true).unwrap()),
            Arc::new(Field::new("name", DataType::Utf8, true)),
            Arc::new(Field::new("active", DataType::Boolean, false)),
        ];
        let struct_array = StructArray::new(
            struct_fields.clone().into(),
            vec![geog_array, name_array.clone(), active_array.clone()],
            None,
        );

        // Create tester
        let input_sedona_type = SedonaType::Arrow(DataType::Struct(struct_fields.into()));
        let tester = ScalarUdfTester::new(udf.clone().into(), vec![input_sedona_type]);

        // Test the function
        let result = tester.invoke_array(Arc::new(struct_array)).unwrap();

        // Verify the result structure
        let result_struct = result.as_struct();
        assert_eq!(result_struct.num_columns(), 3);

        // Geography column should be formatted to UTF8
        let geog_column = result_struct.column(0);
        assert_eq!(geog_column.data_type(), &DataType::Utf8);

        // Name column should remain UTF8 (unchanged)
        let name_column = result_struct.column(1);
        assert_eq!(name_column.data_type(), &DataType::Utf8);
        assert_eq!(name_column, &name_array);

        // Active column should remain Boolean (unchanged)
        let active_column = result_struct.column(2);
        assert_eq!(active_column.data_type(), &DataType::Boolean);
        assert_eq!(active_column, &active_array);

        // Verify the geography was actually formatted to WKT strings
        let string_array = geog_column.as_string::<i32>();
        assert_wkt_values_match(string_array, &geog_values);
    }

    #[rstest]
    fn sd_format_should_format_spatial_lists(
        #[values(WKB_GEOMETRY, WKB_GEOGRAPHY, WKB_VIEW_GEOMETRY, WKB_VIEW_GEOGRAPHY)]
        sedona_type: SedonaType,
    ) -> Result<()> {
        let udf = sd_format_udf();

        // Create an array of WKB geometries using storage format
        let geom_values = vec![
            Some("POINT(1 2)"),
            Some("LINESTRING(0 0,1 1)"),
            None,
            Some("POLYGON((0 0,1 1,1 0,0 0))"),
        ];
        let geom_array = create_array(&geom_values, &sedona_type);

        // Create a simple list containing the geometry array
        let field = Arc::new(sedona_type.to_storage_field("geom", true).unwrap());
        let offsets = OffsetBuffer::new(vec![0, 2, 4].into());
        let list_array = ListArray::new(field, offsets, geom_array, None);

        // Create tester
        let input_sedona_type = SedonaType::Arrow(list_array.data_type().clone());
        let tester = ScalarUdfTester::new(udf.clone().into(), vec![input_sedona_type]);

        // Execute the UDF
        let result = tester.invoke_array(Arc::new(list_array));
        let output_array = result.unwrap();
        let formatted_list = output_array.as_any().downcast_ref::<ListArray>().unwrap();

        // Check that the list field type is now UTF8 (formatted from WKB)
        let list_field = formatted_list.data_type();
        if let DataType::List(inner_field) = list_field {
            assert_eq!(inner_field.data_type(), &DataType::Utf8);
        } else {
            panic!("Expected List data type, got: {list_field:?}");
        }

        // Check the actual formatted values in the list
        let values_array = formatted_list.values();
        if let Some(utf8_array) = values_array.as_any().downcast_ref::<StringArray>() {
            assert_wkt_values_match(utf8_array, &geom_values);
        } else {
            panic!(
                "Expected list elements to be formatted as UTF8 strings, got: {:?}",
                values_array.data_type()
            );
        }

        Ok(())
    }

    #[rstest]
    fn sd_format_should_format_spatial_list_views(
        #[values(WKB_GEOMETRY, WKB_GEOGRAPHY, WKB_VIEW_GEOMETRY, WKB_VIEW_GEOGRAPHY)]
        sedona_type: SedonaType,
    ) -> Result<()> {
        let udf = sd_format_udf();

        // Create an array of WKB geometries using storage format
        let geom_values = vec![
            Some("POINT(1 2)"),
            Some("LINESTRING(0 0,1 1)"),
            None,
            Some("POLYGON((0 0,1 1,1 0,0 0))"),
        ];
        let geom_array = create_array(&geom_values, &sedona_type);

        // Create a ListView containing the geometry array
        let field = Arc::new(sedona_type.to_storage_field("geom", true).unwrap());
        let offsets = ScalarBuffer::from(vec![0i32, 2i32]); // Two list views: [0,2) and [2,4)
        let sizes = ScalarBuffer::from(vec![2i32, 2i32]); // Each list view has 2 elements
        let list_view_array = ListViewArray::new(field, offsets, sizes, geom_array, None);

        // Create tester
        let input_sedona_type = SedonaType::Arrow(list_view_array.data_type().clone());
        let tester = ScalarUdfTester::new(udf.clone().into(), vec![input_sedona_type]);

        // Execute the UDF
        let result = tester.invoke_array(Arc::new(list_view_array));
        let output_array = result.unwrap();
        let formatted_list_view = output_array
            .as_any()
            .downcast_ref::<ListViewArray>()
            .unwrap();

        // Check that the list view field type is now UTF8 (formatted from WKB)
        let list_field = formatted_list_view.data_type();
        if let DataType::ListView(inner_field) = list_field {
            assert_eq!(inner_field.data_type(), &DataType::Utf8);
        } else {
            panic!("Expected ListView data type, got: {list_field:?}");
        }

        // Check the actual formatted values in the list view
        let values_array = formatted_list_view.values();
        if let Some(utf8_array) = values_array.as_any().downcast_ref::<StringArray>() {
            assert_wkt_values_match(utf8_array, &geom_values);
        } else {
            panic!(
                "Expected list view elements to be formatted as UTF8 strings, got: {:?}",
                values_array.data_type()
            );
        }

        Ok(())
    }

    #[rstest]
    fn sd_format_should_format_struct_containing_list_of_geometries(
        #[values(WKB_GEOMETRY, WKB_GEOGRAPHY, WKB_VIEW_GEOMETRY, WKB_VIEW_GEOGRAPHY)]
        sedona_type: SedonaType,
    ) -> Result<()> {
        let udf = sd_format_udf();

        // Create an array of WKB geometries
        let geom_values = vec![
            Some("POINT(1 2)"),
            Some("LINESTRING(0 0,1 1)"),
            None,
            Some("POLYGON((0 0,1 1,1 0,0 0))"),
        ];
        let geom_array = create_array(&geom_values, &sedona_type);

        // Create a list containing the geometry array
        let geom_list_field = Arc::new(sedona_type.to_storage_field("geom", true).unwrap());
        let geom_offsets = OffsetBuffer::new(vec![0, 4].into()); // One list containing all 4 geometries
        let geom_list_array = ListArray::new(geom_list_field, geom_offsets, geom_array, None);

        // Create other fields for the struct
        let name_array: ArrayRef = Arc::new(StringArray::from(vec![Some("feature_collection")]));
        let count_array: ArrayRef = Arc::new(Int32Array::from(vec![4]));

        // Create struct containing the list of geometries
        let struct_fields = vec![
            Arc::new(Field::new("name", DataType::Utf8, true)),
            Arc::new(Field::new(
                "geometries",
                DataType::List(Arc::new(
                    sedona_type.to_storage_field("geom", true).unwrap(),
                )),
                true,
            )),
            Arc::new(Field::new("count", DataType::Int32, false)),
        ];
        let struct_array = StructArray::new(
            struct_fields.clone().into(),
            vec![
                name_array.clone(),
                Arc::new(geom_list_array),
                count_array.clone(),
            ],
            None,
        );

        // Create tester
        let input_sedona_type = SedonaType::Arrow(DataType::Struct(struct_fields.into()));
        let tester = ScalarUdfTester::new(udf.clone().into(), vec![input_sedona_type]);

        // Test the function
        let result = tester.invoke_array(Arc::new(struct_array)).unwrap();

        // Verify the result structure
        let result_struct = result.as_struct();
        assert_eq!(result_struct.num_columns(), 3);

        // Name column should remain UTF8 (unchanged)
        let name_column = result_struct.column(0);
        assert_eq!(name_column.data_type(), &DataType::Utf8);
        assert_eq!(name_column, &name_array);

        // Geometries column should be a list of UTF8 (formatted)
        let geometries_column = result_struct.column(1);
        if let DataType::List(inner_field) = geometries_column.data_type() {
            assert_eq!(inner_field.data_type(), &DataType::Utf8);
        } else {
            panic!(
                "Expected List data type, got: {:?}",
                geometries_column.data_type()
            );
        }

        // Count column should remain Int32 (unchanged)
        let count_column = result_struct.column(2);
        assert_eq!(count_column.data_type(), &DataType::Int32);
        assert_eq!(count_column, &count_array);

        // Verify the geometries were actually formatted to WKT strings
        let formatted_list = geometries_column
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let values_array = formatted_list.values();
        if let Some(utf8_array) = values_array.as_any().downcast_ref::<StringArray>() {
            assert_wkt_values_match(utf8_array, &geom_values);
        } else {
            panic!(
                "Expected list elements to be formatted as UTF8 strings, got: {:?}",
                values_array.data_type()
            );
        }

        Ok(())
    }

    #[rstest]
    fn sd_format_should_format_list_of_structs_containing_geometry(
        #[values(WKB_GEOMETRY, WKB_GEOGRAPHY, WKB_VIEW_GEOMETRY, WKB_VIEW_GEOGRAPHY)]
        sedona_type: SedonaType,
    ) -> Result<()> {
        let udf = sd_format_udf();

        // Create geometry arrays for each struct element
        let geom_values = vec![Some("POINT(1 2)"), Some("LINESTRING(0 0,1 1)")];
        let geom_array = create_array(&geom_values, &sedona_type);

        // Create other field arrays for the struct elements
        let name_array: ArrayRef = Arc::new(StringArray::from(vec![
            Some("point_feature"),
            Some("line_feature"),
        ]));
        let id_array: ArrayRef = Arc::new(Int32Array::from(vec![101, 102]));

        // Create struct array containing geometry field
        let struct_fields = vec![
            Arc::new(Field::new("id", DataType::Int32, false)),
            Arc::new(sedona_type.to_storage_field("geom", true).unwrap()),
            Arc::new(Field::new("name", DataType::Utf8, true)),
        ];
        let struct_array = StructArray::new(
            struct_fields.clone().into(),
            vec![id_array.clone(), geom_array, name_array.clone()],
            None,
        );

        // Create a list containing the struct array
        let list_field = Arc::new(Field::new(
            "feature",
            DataType::Struct(struct_fields.into()),
            true,
        ));
        let list_offsets = OffsetBuffer::new(vec![0, 2].into()); // One list containing 2 structs
        let list_array = ListArray::new(list_field, list_offsets, Arc::new(struct_array), None);

        // Create tester
        let input_sedona_type = SedonaType::Arrow(list_array.data_type().clone());
        let tester = ScalarUdfTester::new(udf.clone().into(), vec![input_sedona_type]);

        // Test the function
        let result = tester.invoke_array(Arc::new(list_array)).unwrap();

        // Verify the result structure
        let result_list = result.as_any().downcast_ref::<ListArray>().unwrap();

        // Check that the list field type is a struct with formatted geometry field
        let list_field = result_list.data_type();
        if let DataType::List(inner_field) = list_field {
            if let DataType::Struct(struct_fields) = inner_field.data_type() {
                // Find the geometry field and verify it's been formatted to UTF8
                let geom_field = struct_fields.iter().find(|f| f.name() == "geom").unwrap();
                assert_eq!(geom_field.data_type(), &DataType::Utf8);
            } else {
                panic!(
                    "Expected Struct data type inside List, got: {:?}",
                    inner_field.data_type()
                );
            }
        } else {
            panic!("Expected List data type, got: {list_field:?}");
        }

        // Verify the actual struct values and their geometry formatting
        let struct_values = result_list.values().as_struct();
        assert_eq!(struct_values.num_columns(), 3);

        // ID column should remain Int32 (unchanged)
        let id_column = struct_values.column(0);
        assert_eq!(id_column.data_type(), &DataType::Int32);
        assert_eq!(id_column, &id_array);

        // Geometry column should be formatted to UTF8
        let geometry_column = struct_values.column(1);
        assert_eq!(geometry_column.data_type(), &DataType::Utf8);

        // Name column should remain UTF8 (unchanged)
        let name_column = struct_values.column(2);
        assert_eq!(name_column.data_type(), &DataType::Utf8);
        assert_eq!(name_column, &name_array);

        // Verify the geometries were actually formatted to WKT strings
        let string_array = geometry_column.as_string::<i32>();
        assert_wkt_values_match(string_array, &geom_values);

        Ok(())
    }

    /// Helper function to verify that actual WKT values match expected values,
    /// handling the normalization of comma spacing in WKT output
    fn assert_wkt_values_match(actual_array: &StringArray, expected_values: &[Option<&str>]) {
        for (i, expected) in expected_values.iter().enumerate() {
            match expected {
                Some(expected_value) => {
                    let actual_value = actual_array.value(i);
                    // Note: WKT output may not have spaces after commas
                    let normalized_expected = expected_value.replace(", ", ",");
                    assert_eq!(actual_value, normalized_expected);
                }
                None => assert!(actual_array.is_null(i)),
            }
        }
    }
}
