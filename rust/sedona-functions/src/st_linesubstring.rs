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
use crate::executor::WkbExecutor;
use arrow_array::builder::BinaryBuilder;
use arrow_schema::DataType;
use datafusion_common::{error::Result, DataFusionError, ScalarValue};
use datafusion_expr::{ColumnarValue, Volatility};
use geo_traits::{CoordTrait, Dimensions, GeometryTrait, GeometryType, LineStringTrait};
use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{SedonaScalarKernel, SedonaScalarUDF},
};
use sedona_geometry::error::SedonaGeometryError;
use sedona_geometry::wkb_factory::{write_wkb_coord_trait, write_wkb_linestring_header, write_wkb_point_header};
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};
use std::{io::Write, sync::Arc};
use arrow_array::Array;

#[derive(Debug)]
struct STLineSubstring;
pub fn st_line_substring_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_linesubstring",
        ItemCrsKernel::wrap_impl(vec![Arc::new(STLineSubstring)]),
        Volatility::Immutable,
    )
}

impl SedonaScalarKernel for STLineSubstring {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![
                ArgMatcher::is_geometry(),
                ArgMatcher::is_numeric(),
                ArgMatcher::is_numeric(),
            ],
            WKB_GEOMETRY,
        );
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);
        let mut builder = BinaryBuilder::new();

        // 1. Convert parameters into clean abstract arrays
        let batch_rows = executor.num_iterations();
        let start_array = args[1].cast_to(&DataType::Float64, None)?.to_array(batch_rows)?;
        let end_array = args[2].cast_to(&DataType::Float64, None)?.to_array(batch_rows)?;

        // 2. Downcast them to Arrow Float64Arrays so they can be read by row index
        let start_floats = start_array.as_any().downcast_ref::<arrow_array::Float64Array>().unwrap();
        let end_floats = end_array.as_any().downcast_ref::<arrow_array::Float64Array>().unwrap();

        unsafe fn interpolate<C: CoordTrait<T = f64>>(
            p1: C,
            p2: C,
            fraction: f64,
            dim: Dimensions,
            buf: &mut impl Write,
        ) -> Result<(), SedonaGeometryError> {
            for i in 0..dim.size() {
                let v =
                    p1.nth_unchecked(i) + (p2.nth_unchecked(i) - p1.nth_unchecked(i)) * fraction;
                buf.write_all(&v.to_le_bytes())?;
            }
            Ok(())
        }
        let mut row_idx = 0;
        executor.execute_wkb_void(|maybe_wkb| unsafe {
            let mut wkb_body = Vec::new();
            let mut point_count = 0u32;
            // Fetch the unique start/end fraction values for THIS specific row
            let s_f_opt = if start_floats.is_null(row_idx) { None } else { Some(start_floats.value(row_idx)) };
            let e_f_opt = if end_floats.is_null(row_idx) { None } else { Some(end_floats.value(row_idx)) };
            row_idx += 1; // Increment for the next iteration

            let (s_f, e_f) = match (s_f_opt, e_f_opt) {
                (Some(s), Some(e)) => (s, e),
                _ => {
                    builder.append_null();
                    return Ok(());
                }
            };

            if let Some(wkb) = maybe_wkb {
                if let GeometryType::LineString(line) = wkb.as_type() {
                    let num_coords = line.num_coords();
                    let dim = line.dim();

                    let mut cumulative_distances = Vec::with_capacity(num_coords);
                    let mut total_length = 0.0;
                    cumulative_distances.push(0.0);

                    for i in 0..(num_coords - 1) {
                        let p1 = line.coord(i).unwrap();
                        let p2 = line.coord(i + 1).unwrap();
                        let dist = ((p2.x() - p1.x()).powi(2) + (p2.y() - p1.y()).powi(2)).sqrt();
                        total_length += dist;
                        cumulative_distances.push(total_length);
                    }

                    let start_dist = s_f * total_length;
                    let end_dist = e_f * total_length;

                    for i in 0..(num_coords - 1) {
                        let d1 = cumulative_distances[i];
                        let d2 = cumulative_distances[i + 1];
                        let p1 = line.coord(i).unwrap();
                        let p2 = line.coord(i + 1).unwrap();

                        if start_dist >= d1 && start_dist <= d2 {
                            let segment_len = d2 - d1;
                            let fraction = if segment_len > 0.0 {
                                (start_dist - d1) / segment_len
                            } else {
                                0.0
                            };
                            interpolate(p1, p2, fraction, dim, &mut wkb_body).map_err(|e| {
                                DataFusionError::Internal(format!(
                                    "Sedona interpolation failed: {}",
                                    e
                                ))
                            })?;
                            point_count += 1;
                        }

                        if d1 > start_dist && d1 < end_dist {
                            write_wkb_coord_trait(&mut wkb_body, &p1).map_err(|e| {
                                DataFusionError::Internal(format!("WKB write failed: {}", e))
                            })?;
                            point_count += 1;
                        }

                        if end_dist >= d1 && end_dist <= d2 {
                            let segment_len = d2 - d1;
                            let fraction = if segment_len > 0.0 {
                                (end_dist - d1) / segment_len
                            } else {
                                0.0
                            };
                            interpolate(p1, p2, fraction, dim, &mut wkb_body).map_err(|e| {
                                DataFusionError::Internal(format!(
                                    "Sedona interpolation failed: {}",
                                    e
                                ))
                            })?;
                            point_count += 1;
                        }
                    }

                    if point_count > 0 {
                        let mut final_wkb = Vec::new();

                        if s_f == e_f {
                            // POINT Result
                            write_wkb_point_header(&mut final_wkb, dim)
                                .map_err(|e| DataFusionError::Internal(e.to_string()))?;
                            let coord_bytes = dim.size() * 8;
                            if wkb_body.len() >= coord_bytes {
                                final_wkb.extend_from_slice(&wkb_body[..coord_bytes]);
                            }
                        } else {
                            // LINESTRING Result
                            write_wkb_linestring_header(&mut final_wkb, dim, point_count as usize)
                                .map_err(|e| DataFusionError::Internal(e.to_string()))?;

                            final_wkb.extend_from_slice(&wkb_body);
                        }
                        builder.append_value(final_wkb);
                    } else {
                        builder.append_null();
                    }
                } else {
                    builder.append_null();
                }
            } else {
                builder.append_null();
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}
#[cfg(test)]
mod tests {
    use arrow_array::{ArrayRef, Float64Array};
    use arrow_schema::DataType;
    use datafusion_common::scalar::ScalarValue;
    use datafusion_expr::{ColumnarValue, ScalarUDF};
    use rstest::rstest;
    use sedona_schema::datatypes::{SedonaType, WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS};
    use sedona_testing::{
        compare::{assert_array_equal, assert_scalar_equal},
        testers::ScalarUdfTester,
    };
    use std::sync::Arc;

    use super::*;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = st_line_substring_udf().into();
        assert_eq!(udf.name(), "st_linesubstring");
    }

    #[rstest]
    fn udf(
        #[values(WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS.clone())]
        sedona_type: SedonaType,
    ) {
        let udf = st_line_substring_udf();
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![
                sedona_type,
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64)
            ]
        );

        let actual_2d = tester.invoke_scalar_scalar_scalar("LINESTRING(0 0, 10 10)", 0.0, 0.5).unwrap();
        let expected_2d = tester.invoke_wkb_scalar(Some("LINESTRING(0 0, 5 5)")).unwrap();
        assert_scalar_equal(&actual_2d, &expected_2d);


        let actual_z = tester.invoke_scalar_scalar_scalar("LINESTRING Z (0 10 20, 10 20 30)", 0.5, 1.0).unwrap();
        let expected_z = tester.invoke_wkb_scalar(Some("LINESTRING Z (5 15 25, 10 20 30)")).unwrap();
        assert_scalar_equal(&actual_z, &expected_z);

        let actual_mid = tester.invoke_scalar_scalar_scalar("LINESTRING Z (0 0 0, 10 10 10)", 0.5, 0.8).unwrap();
        let expected_mid = tester.invoke_wkb_scalar(Some("LINESTRING Z (5 5 5, 8 8 8)")).unwrap();
        assert_scalar_equal(&actual_mid, &expected_mid);


        let actual_point = tester.invoke_scalar_scalar_scalar("LINESTRING(0 0, 10 10)", 0.5, 0.5).unwrap();
        let expected_point = tester.invoke_wkb_scalar(Some("POINT(5 5)")).unwrap();
        assert_scalar_equal(&actual_point, &expected_point);


        let geoms_input = tester.invoke_wkb_array(vec![
            Some("LINESTRING(0 0, 10 10)"),
            None,
            Some("LINESTRING(0 0, 10 10)")
        ]).unwrap();

        let starts_input: ArrayRef = Arc::new(Float64Array::from(vec![Some(0.0), Some(0.0), Some(0.5)]));
        let ends_input: ArrayRef = Arc::new(Float64Array::from(vec![Some(0.5), Some(1.0), Some(0.5)]));

        let expected_array = tester.invoke_wkb_array(vec![
            Some("LINESTRING(0 0, 5 5)"),
            None,
            Some("POINT(5 5)")
        ]).unwrap();

        let actual_array = match tester.invoke(vec![
            ColumnarValue::Array(geoms_input),
            ColumnarValue::Array(starts_input),
            ColumnarValue::Array(ends_input)
        ]).unwrap() {
            ColumnarValue::Array(arr) => arr,
            _ => panic!("Expected array block context output"),
        };

        assert_array_equal(&actual_array, &expected_array);
    }

    #[test]
    fn aliases() {
        let udf: ScalarUDF = st_line_substring_udf().into();
        assert_eq!(udf.name(), "st_linesubstring");
    }
}