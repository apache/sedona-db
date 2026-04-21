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
use arrow_array::builder::BinaryBuilder;
use arrow_schema::DataType;
use datafusion_common::{error::Result, ScalarValue};
use datafusion_expr::{ColumnarValue, Volatility};
use geo_traits::{CoordTrait, GeometryTrait, LineStringTrait,GeometryType};
use sedona_common::sedona_internal_err;
use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{SedonaScalarKernel, SedonaScalarUDF},
};
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};
use std::{io::Write, sync::Arc};

use crate::executor::WkbExecutor;

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
                ArgMatcher::is_numeric()
            ],
            WKB_GEOMETRY,
        );
        matcher.match_args(args)
    }


    fn invoke_batch(&self, arg_types: &[SedonaType], args: &[ColumnarValue]) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);
        let mut builder = BinaryBuilder::new();


        let start_frac: Option<f64> = match &args[1].cast_to(&DataType::Float64, None)? {
            ColumnarValue::Scalar(ScalarValue::Float64(s)) => *s,
            _ => None,
        };


        let end_frac: Option<f64> = match &args[2].cast_to(&DataType::Float64, None)? {
            ColumnarValue::Scalar(ScalarValue::Float64(e)) => *e,
            _ => None,
        };
        fn interpolate<C: CoordTrait<T = f64>>(p1: C, p2: C, fraction: f64) -> (f64, f64) {
            let x = p1.x() + (p2.x() - p1.x()) * fraction;
            let y = p1.y() + (p2.y() - p1.y()) * fraction;
            (x, y)
        }
        executor.execute_wkb_void(|maybe_wkb| {
            let mut new_coords = Vec::new();
            if let Some(wkb) = maybe_wkb {
                if let GeometryType::LineString(line) = wkb.as_type() {
                    let num_coords = line.num_coords() as i64;

                    let mut cumulative_distances = Vec::with_capacity(num_coords as usize);
                    let mut total_length = 0.0;
                    cumulative_distances.push(0.0);
                    let (s_frac, e_frac) = match (start_frac, end_frac) {
                        (Some(s), Some(e)) => (s, e),
                        _ => {
                            builder.append_null();
                            return Ok(());
                        }
                    };
                    for i in 0..(num_coords as usize - 1) {
                        let p1 = line.coord(i).unwrap();
                        let p2 = line.coord(i + 1).unwrap();


                        let dist = ((p2.x() - p1.x()).powi(2) + (p2.y() - p1.y()).powi(2)).sqrt();
                        total_length += dist;
                        cumulative_distances.push(total_length);
                    }
                    let start_dist = s_frac * total_length;
                    let end_dist = e_frac * total_length;


                    for i in 0..(num_coords as usize - 1) {
                        let d1 = cumulative_distances[i];
                        let d2 = cumulative_distances[i + 1];
                        let p1 = line.coord(i).unwrap();
                        let p2 = line.coord(i + 1).unwrap();

                        if start_dist >= d1 && start_dist <= d2 {
                            let segment_len = d2 - d1;
                            let fraction = if segment_len > 0.0 { (start_dist - d1) / segment_len } else { 0.0 };
                            new_coords.push(interpolate(p1, p2, fraction));
                        }

                        if d1 > start_dist && d1 < end_dist {
                            new_coords.push((p1.x(), p1.y()));
                        }

                        if end_dist >= d1 && end_dist <= d2 {
                            let segment_len = d2 - d1;
                            let fraction = if segment_len > 0.0 { (end_dist - d1) / segment_len } else { 0.0 };
                            new_coords.push(interpolate(p1, p2, fraction));
                        }
                    }
                }

            }
            if !new_coords.is_empty() {
                // write byte order (1 = little endian)
                builder.write_all(&[1u8])?;


                let type_id: u32 = 2;
                builder.write_all(&type_id.to_le_bytes())?;


                let num_points = new_coords.len() as u32;
                builder.write_all(&num_points.to_le_bytes())?;


                for (x, y) in new_coords {

                    builder.write_all(&x.to_le_bytes())?;
                    builder.write_all(&y.to_le_bytes())?;
                }

                builder.append_value([]);
            } else {
                builder.append_null();
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}