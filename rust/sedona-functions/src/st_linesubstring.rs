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
use sedona_geometry::wkb_factory::write_wkb_coord_trait;
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

        let start_frac: Option<f64> = match &args[1].cast_to(&DataType::Float64, None)? {
            ColumnarValue::Scalar(ScalarValue::Float64(s)) => *s,
            _ => None,
        };

        let end_frac: Option<f64> = match &args[2].cast_to(&DataType::Float64, None)? {
            ColumnarValue::Scalar(ScalarValue::Float64(e)) => *e,
            _ => None,
        };

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

        executor.execute_wkb_void(|maybe_wkb| unsafe {
            let mut wkb_body = Vec::new();
            let mut point_count = 0u32;

            let (s_f, e_f) = match (start_frac, end_frac) {
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

                    // 1. Handle Empty Geometry (Fixes Panic)
                    if num_coords == 0 {
                        let mut empty_wkb = Vec::with_capacity(9);
                        empty_wkb.push(1u8); // Little Endian
                        empty_wkb.extend_from_slice(&2u32.to_le_bytes()); // Type: LineString
                        empty_wkb.extend_from_slice(&0u32.to_le_bytes()); // Count: 0
                        builder.append_value(empty_wkb);
                        return Ok(());
                    }

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

                    // 2. Build Header inside the 'line' scope (Fixes "cannot find dim/line")
                    if point_count > 0 {
                        let mut final_wkb = Vec::new();
                        final_wkb.push(1u8); // Little Endian

                        if s_f == e_f {
                            // POINT Result (Fixes point vs line test)
                            let p_type: u32 = if dim == Dimensions::Xyz { 1001 } else { 1 };
                            final_wkb.extend_from_slice(&p_type.to_le_bytes());
                            let coord_bytes = dim.size() * 8;
                            if wkb_body.len() >= coord_bytes {
                                final_wkb.extend_from_slice(&wkb_body[..coord_bytes]);
                            }
                        } else {
                            // LINESTRING Result (Fixes Z-coordinate drop)
                            let l_type: u32 = if dim == Dimensions::Xyz { 1002 } else { 2 };
                            final_wkb.extend_from_slice(&l_type.to_le_bytes());
                            final_wkb.extend_from_slice(&point_count.to_le_bytes());
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
