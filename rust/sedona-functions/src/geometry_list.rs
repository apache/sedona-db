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

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BinaryArray, StringViewArray, StructArray, builder::StringViewBuilder,
};
use arrow_buffer::OffsetBuffer;
use arrow_schema::{DataType, Field};
use datafusion_common::{
    Result, ScalarValue,
    cast::{as_string_view_array, as_struct_array},
};
use datafusion_expr::ColumnarValue;
use sedona_common::sedona_internal_err;
use sedona_schema::datatypes::SedonaType;

/// Input for functions that return lists containing geometry values.
/// Unwraps item-level CRS inputs and repeats each row's CRS for its members.
/// Keeping the CRS on each geometry allows list elements to be used by other UDFs.
pub(crate) struct GeometryListInput {
    pub types: [SedonaType; 1],
    pub args: [ColumnarValue; 1],
    crs: Option<StringViewArray>,
}

impl GeometryListInput {
    pub fn new(arg_type: &SedonaType, arg: &ColumnarValue) -> Result<Self> {
        if !arg_type.is_item_crs() {
            return Ok(Self {
                types: [arg_type.clone()],
                args: [arg.clone()],
                crs: None,
            });
        }

        let num_rows = match arg {
            ColumnarValue::Array(array) => array.len(),
            ColumnarValue::Scalar(_) => 1,
        };
        let array = arg.to_array(num_rows)?;
        let array = as_struct_array(&array)?;
        // Propagate the struct's validity before the executor reads its WKB.
        let (fields, columns) = array.flatten();
        let item = match arg {
            ColumnarValue::Array(_) => ColumnarValue::Array(columns[0].clone()),
            ColumnarValue::Scalar(_) => ScalarValue::try_from_array(&columns[0], 0)?.into(),
        };
        Ok(Self {
            types: [SedonaType::from_storage_field(&fields[0])?],
            args: [item],
            crs: Some(as_string_view_array(&columns[1])?.clone()),
        })
    }

    /// Wrap flattened WKB members with the CRS of their parent input row.
    /// Offsets refer to the outer list, including empty and null rows.
    pub fn finish_values(
        &self,
        values: BinaryArray,
        offsets: &OffsetBuffer<i32>,
        field: &Field,
    ) -> Result<ArrayRef> {
        let Some(crs) = &self.crs else {
            return Ok(Arc::new(values));
        };
        let DataType::Struct(fields) = field.data_type() else {
            return sedona_internal_err!("Expected item_crs field for geometry list members");
        };
        let mut crs_builder = StringViewBuilder::with_capacity(values.len());
        for (crs, offsets) in crs.iter().zip(offsets.windows(2)) {
            let count = (offsets[1] - offsets[0]) as usize;
            if let Some(crs) = crs {
                // Reuse the string view for every member, including long CRS definitions.
                crs_builder.try_append_value_n(crs, count)?;
            } else {
                for _ in 0..count {
                    crs_builder.append_null();
                }
            }
        }
        let nulls = values.nulls().cloned();
        Ok(Arc::new(StructArray::try_new(
            fields.clone(),
            vec![Arc::new(values), Arc::new(crs_builder.finish())],
            nulls,
        )?))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Array, ArrayRef, ListArray, StructArray, UInt32Array};
    use arrow_buffer::NullBuffer;
    use datafusion_common::{ScalarValue, cast::as_struct_array};
    use rstest::rstest;
    use sedona_schema::datatypes::{
        SedonaType, WKB_GEOGRAPHY, WKB_GEOMETRY, WKB_LARGE_GEOGRAPHY, WKB_LARGE_GEOMETRY,
        WKB_VIEW_GEOGRAPHY, WKB_VIEW_GEOMETRY,
    };
    use sedona_testing::{
        compare::assert_array_equal, create::create_array_item_crs, testers::ScalarUdfTester,
    };

    use crate::{st_dump::st_dump_udf, st_geometries::st_geometries_udf};

    #[rstest]
    fn item_crs_members(
        #[values(false, true)] dump: bool,
        #[values(
            WKB_GEOMETRY,
            WKB_GEOGRAPHY,
            WKB_LARGE_GEOMETRY,
            WKB_LARGE_GEOGRAPHY,
            WKB_VIEW_GEOMETRY,
            WKB_VIEW_GEOGRAPHY
        )]
        input_type: SedonaType,
    ) {
        let udf = if dump {
            st_dump_udf()
        } else {
            st_geometries_udf()
        };
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![SedonaType::new_item_crs(&input_type).unwrap()],
        );
        let output_type = match &input_type {
            SedonaType::Wkb(edges, _)
            | SedonaType::WkbLarge(edges, _)
            | SedonaType::WkbView(edges, _) => SedonaType::Wkb(*edges, None),
            _ => unreachable!(),
        };
        let wkts = [
            Some("MULTIPOINT (1 2, 3 4)"),
            None,
            Some("GEOMETRYCOLLECTION EMPTY"),
            Some("GEOMETRYCOLLECTION (POINT (5 6), MULTIPOINT (7 8, 9 10))"),
            Some("POINT EMPTY"),
            Some("POINT (11 12)"),
        ];
        let crses = [
            Some("EPSG:4326"),
            Some("EPSG:3857"),
            Some("EPSG:3857"),
            Some("OGC:CRS84"),
            None,
            Some("urn:ogc:def:crs:EPSG::4326"),
        ];
        let input = create_array_item_crs(&wkts, crses, &input_type);
        let expected: Vec<Vec<&str>> = vec![
            vec!["POINT (1 2)", "POINT (3 4)"],
            vec![],
            vec![],
            if dump {
                vec!["POINT (5 6)", "POINT (7 8)", "POINT (9 10)"]
            } else {
                vec!["POINT (5 6)", "MULTIPOINT (7 8, 9 10)"]
            },
            vec!["POINT EMPTY"],
            vec!["POINT (11 12)"],
        ];
        let paths: Vec<Vec<Vec<u32>>> = vec![
            vec![vec![1], vec![2]],
            vec![],
            vec![],
            vec![vec![1], vec![2, 1], vec![2, 2]],
            vec![vec![]],
            vec![vec![]],
        ];

        // Check arrays (including nonzero input offsets) and the scalar path.
        for start in [0, 1] {
            let result = tester
                .invoke_array(input.slice(start, input.len() - start))
                .unwrap();
            assert_eq!(
                result.data_type(),
                tester.return_type().unwrap().storage_type()
            );
            let result = result.as_any().downcast_ref::<ListArray>().unwrap();
            for row in start..input.len() {
                assert_row(
                    result,
                    row - start,
                    wkts[row].is_none(),
                    &expected[row],
                    &paths[row],
                    crses[row],
                    &output_type,
                    dump,
                );
            }
        }
        for row in 0..input.len() {
            let result = tester
                .invoke_scalar(ScalarValue::try_from_array(&input, row).unwrap())
                .unwrap()
                .to_array()
                .unwrap();
            let result = result.as_any().downcast_ref::<ListArray>().unwrap();
            assert_row(
                result,
                0,
                wkts[row].is_none(),
                &expected[row],
                &paths[row],
                crses[row],
                &output_type,
                dump,
            );
        }

        // An empty batch must retain the nested geometry metadata.
        let result = tester.invoke_array(input.slice(0, 0)).unwrap();
        assert_eq!(result.len(), 0);
        assert_eq!(
            result.data_type(),
            tester.return_type().unwrap().storage_type()
        );

        // Parent validity can mask a non-null child geometry.
        let parent_null = as_struct_array(&input).unwrap();
        let parent_null: ArrayRef = Arc::new(StructArray::new(
            parent_null.fields().clone(),
            parent_null.columns().to_vec(),
            Some(NullBuffer::from(vec![false; input.len()])),
        ));
        let result = tester.invoke_array(parent_null.clone()).unwrap();
        assert_eq!(result.null_count(), input.len());
        let result = tester
            .invoke_scalar(ScalarValue::try_from_array(&parent_null, 0).unwrap())
            .unwrap();
        assert!(result.is_null());
    }

    #[allow(clippy::too_many_arguments)]
    fn assert_row(
        result: &ListArray,
        row: usize,
        is_null: bool,
        wkts: &[&str],
        paths: &[Vec<u32>],
        crs: Option<&str>,
        output_type: &SedonaType,
        dump: bool,
    ) {
        assert_eq!(result.is_null(row), is_null);
        if is_null {
            return;
        }
        let values = result.value(row);
        let (geometries, field) = if dump {
            let values = as_struct_array(&values).unwrap();
            let path_lists = values
                .column(0)
                .as_any()
                .downcast_ref::<ListArray>()
                .unwrap();
            for (i, path) in paths.iter().enumerate() {
                let path_value = path_lists.value(i);
                let path_value = path_value.as_any().downcast_ref::<UInt32Array>().unwrap();
                assert_eq!(path_value.values().as_ref(), path);
            }
            (values.column(1).clone(), values.fields()[1].clone())
        } else {
            let arrow_schema::DataType::List(field) = result.data_type() else {
                unreachable!()
            };
            (values, field.clone())
        };
        assert_eq!(
            SedonaType::from_storage_field(&field).unwrap(),
            SedonaType::new_item_crs(output_type).unwrap()
        );
        assert_array_equal(
            &geometries,
            &create_array_item_crs(
                &wkts.iter().map(|wkt| Some(*wkt)).collect::<Vec<_>>(),
                std::iter::repeat_n(crs, wkts.len()),
                output_type,
            ),
        );
    }
}
