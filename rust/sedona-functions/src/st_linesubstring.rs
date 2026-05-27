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
use arrow_array::Array;
use arrow_schema::DataType;
use datafusion_common::{error::Result, exec_datafusion_err, exec_err};
use datafusion_expr::{ColumnarValue, Volatility};
use geo_traits::{CoordTrait, Dimensions, GeometryTrait, GeometryType, LineStringTrait};
use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{SedonaScalarKernel, SedonaScalarUDF},
};
use sedona_geometry::error::SedonaGeometryError;
use sedona_geometry::wkb_factory::{
    write_wkb_coord_trait, write_wkb_linestring_header, write_wkb_point_header,
};
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};
use std::{io::Write, sync::Arc};

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
        let start_array = args[1]
            .cast_to(&DataType::Float64, None)?
            .to_array(batch_rows)?;
        let end_array = args[2]
            .cast_to(&DataType::Float64, None)?
            .to_array(batch_rows)?;

        // 2. Downcast them to Arrow Float64Arrays so they can be read by row index
        let start_floats = start_array
            .as_any()
            .downcast_ref::<arrow_array::Float64Array>()
            .unwrap();
        let end_floats = end_array
            .as_any()
            .downcast_ref::<arrow_array::Float64Array>()
            .unwrap();

        let mut start_floats_iter = start_floats.iter();
        let mut end_floats_iter = end_floats.iter();

        let mut wkb_body = Vec::new();
        let mut cumulative_distances = Vec::new();

        executor.execute_wkb_void(|maybe_wkb| {
            let (Some(wkb), Some(s_f), Some(e_f)) = (
                maybe_wkb,
                start_floats_iter.next().unwrap(),
                end_floats_iter.next().unwrap(),
            ) else {
                builder.append_null();
                return Ok(());
            };

            let GeometryType::LineString(line) = wkb.as_type() else {
                return exec_err!("Can't compute substring of non-line input");
            };

            if !(0.0..=1.0).contains(&s_f) {
                return exec_err!("start_fraction must be between 0.0 and 1.0 (got {s_f}");
            }

            if !(0.0..=1.0).contains(&e_f) {
                return exec_err!("end_fraction must be between 0.0 and 1.0 (got {e_f}");
            }

            if e_f < s_f {
                return exec_err!(
                    "end_fraction must be greater than start_fraction (got {e_f} < {s_f})"
                );
            }

            invoke_scalar(
                line,
                s_f,
                e_f,
                &mut wkb_body,
                &mut cumulative_distances,
                &mut builder,
            )
            .map_err(|e| exec_datafusion_err!("Error executing st_linesubstring: {e}"))?;
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(
    line: &impl LineStringTrait<T = f64>,
    s_f: f64,
    e_f: f64,
    wkb_body: &mut Vec<u8>,
    cumulative_distances: &mut Vec<f64>,
    builder: &mut BinaryBuilder,
) -> Result<(), SedonaGeometryError> {
    let mut point_count = 0u32;
    wkb_body.clear();
    cumulative_distances.clear();

    let num_coords = line.num_coords();
    let dim = line.dim();

    // Empty or degenerate input returns NULL here
    if num_coords <= 1 {
        builder.append_null();
        return Ok(());
    }

    let mut total_length = 0.0;
    cumulative_distances.push(0.0);

    for i in 0..(num_coords - 1) {
        let (p1, p2) = unsafe { (line.coord_unchecked(i), line.coord_unchecked(i + 1)) };
        let dist = ((p2.x() - p1.x()).powi(2) + (p2.y() - p1.y()).powi(2)).sqrt();
        total_length += dist;
        cumulative_distances.push(total_length);
    }

    let start_dist = s_f * total_length;
    let end_dist = e_f * total_length;

    // Use binary search to find the segment range to scan.
    // For start: find the segment containing start_dist (where d[i] <= start_dist < d[i+1])
    // For end: find the segment containing end_dist (where d[i] < end_dist <= d[i+1])
    let num_segments = num_coords - 1;

    let first_segment = cumulative_distances
        .partition_point(|&d| d <= start_dist)
        .saturating_sub(1)
        .min(num_segments - 1);

    let last_segment = cumulative_distances
        .partition_point(|&d| d < end_dist)
        .saturating_sub(1)
        .min(num_segments - 1);

    for i in first_segment..=last_segment {
        let d1 = cumulative_distances[i];
        let d2 = cumulative_distances[i + 1];
        let (p1, p2) = unsafe { (line.coord_unchecked(i), line.coord_unchecked(i + 1)) };

        if start_dist >= d1 && start_dist < d2 {
            let segment_len = d2 - d1;
            let fraction = if segment_len > 0.0 {
                (start_dist - d1) / segment_len
            } else {
                0.0
            };

            unsafe {
                interpolate(&p1, &p2, fraction, dim, wkb_body)?;
            }
            point_count += 1;
        }

        if d1 > start_dist && d1 < end_dist {
            write_wkb_coord_trait(wkb_body, &p1)?;
            point_count += 1;
        }

        if end_dist > d1 && end_dist <= d2 {
            let segment_len = d2 - d1;
            let fraction = if segment_len > 0.0 {
                (end_dist - d1) / segment_len
            } else {
                0.0
            };

            unsafe {
                interpolate(&p1, &p2, fraction, dim, wkb_body)?;
            }

            point_count += 1;
        }
    }

    if point_count > 0 {
        if s_f == e_f {
            // POINT Result
            write_wkb_point_header(builder, dim)?;
            let coord_bytes = dim.size() * size_of::<f64>();
            if wkb_body.len() >= coord_bytes {
                builder.write_all(&wkb_body[..coord_bytes])?;
            }
        } else {
            // LINESTRING Result
            write_wkb_linestring_header(builder, dim, point_count as usize)?;
            builder.write_all(wkb_body)?;
        }
    }

    builder.append_value([]);
    Ok(())
}

unsafe fn interpolate<C: CoordTrait<T = f64>>(
    p1: &C,
    p2: &C,
    fraction: f64,
    dim: Dimensions,
    buf: &mut impl Write,
) -> Result<(), SedonaGeometryError> {
    for i in 0..dim.size() {
        let v = p1.nth_unchecked(i) + (p2.nth_unchecked(i) - p1.nth_unchecked(i)) * fraction;
        buf.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{ArrayRef, Float64Array};
    use arrow_schema::DataType;
    use datafusion_common::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::{SedonaType, WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS};
    use sedona_testing::create::create_array;
    use sedona_testing::{compare::assert_array_equal, testers::ScalarUdfTester};
    use std::sync::Arc;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = st_line_substring_udf().into();
        assert_eq!(udf.name(), "st_linesubstring");
    }

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS.clone())] sedona_type: SedonaType) {
        let udf = st_line_substring_udf();
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![
                sedona_type.clone(),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
            ],
        );

        let actual_2d = tester
            .invoke_scalar_scalar_scalar("LINESTRING(0 0, 10 10)", 0.0, 0.5)
            .unwrap();
        tester.assert_scalar_result_equals(actual_2d, "LINESTRING(0 0, 5 5)");

        let actual_z = tester
            .invoke_scalar_scalar_scalar("LINESTRING Z (0 10 20, 10 20 30)", 0.5, 1.0)
            .unwrap();
        tester.assert_scalar_result_equals(actual_z, "LINESTRING Z (5 15 25, 10 20 30)");

        let actual_mid = tester
            .invoke_scalar_scalar_scalar("LINESTRING Z (0 0 0, 10 10 10)", 0.5, 0.8)
            .unwrap();
        tester.assert_scalar_result_equals(actual_mid, "LINESTRING Z (5 5 5, 8 8 8)");

        let actual_point = tester
            .invoke_scalar_scalar_scalar("LINESTRING(0 0, 10 10)", 0.5, 0.5)
            .unwrap();
        tester.assert_scalar_result_equals(actual_point, "POINT (5 5)");

        let actual_empty = tester
            .invoke_scalar_scalar_scalar("LINESTRING EMPTY", 0.0, 1.0)
            .unwrap();
        tester.assert_scalar_result_equals(actual_empty, ScalarValue::Null);

        let geoms_input = create_array(
            &[
                Some("LINESTRING(0 0, 10 10)"),
                None,
                Some("LINESTRING(0 0, 10 10)"),
            ],
            &sedona_type,
        );

        let starts_input: ArrayRef =
            Arc::new(Float64Array::from(vec![Some(0.0), Some(0.0), Some(0.5)]));
        let ends_input: ArrayRef =
            Arc::new(Float64Array::from(vec![Some(0.5), Some(1.0), Some(0.5)]));

        let expected_array = create_array(
            &[Some("LINESTRING(0 0, 5 5)"), None, Some("POINT(5 5)")],
            &sedona_type,
        );

        let actual_array = tester
            .invoke_arrays(vec![geoms_input, starts_input, ends_input])
            .unwrap();

        assert_array_equal(&actual_array, &expected_array);
    }
}
