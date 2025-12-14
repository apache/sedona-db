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

use std::{fmt::Debug, sync::Arc};

use arrow_array::builder::UInt64Builder;
use arrow_schema::DataType;
use datafusion_common::{DataFusionError, Result};
use datafusion_expr::ColumnarValue;
use sedona_expr::scalar_udf::SedonaScalarKernel;
use sedona_functions::executor::WkbBytesExecutor;
use sedona_geometry::{transform::CrsEngine, wkb_header::WkbHeader};
use sedona_schema::{crs::lnglat, datatypes::SedonaType, matchers::ArgMatcher};

use crate::st_transform::with_global_proj_engine;

/// Generic scalar kernel for sd_order based on the first coordinate
/// of a geometry projected to lon/lat
///
/// This [SedonaScalarKernel] requires the actual function (e.g., S2, H3,
/// or A5 cell identifier) to be provided but takes care of the extraction
/// of the first coordinate and projecting to lon/lat space. The provided
/// function must return a `u64`.
pub struct OrderLngLat<F> {
    order_fn: F,
}

impl<F: Fn((f64, f64)) -> u64> OrderLngLat<F> {
    /// Create a new kernel from the required function type
    pub fn new(order_fn: F) -> Self {
        Self { order_fn }
    }
}

impl<F: Fn((f64, f64)) -> u64> Debug for OrderLngLat<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrderLngLat").finish()
    }
}

impl<F: Fn((f64, f64)) -> u64> SedonaScalarKernel for OrderLngLat<F> {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry_or_geography()],
            SedonaType::Arrow(DataType::UInt64),
        );
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        // Extract the source CRS, checking for lon/lat to see if we can avoid
        // a transformation. If the CRS is missing we also skip any particular
        // transform (although the resulting sort may not be effective).
        let maybe_src_crs = match &arg_types[0] {
            SedonaType::Wkb(_, maybe_crs) | SedonaType::WkbView(_, maybe_crs)
                if maybe_crs != &lnglat() =>
            {
                maybe_crs.as_ref().map(|crs| crs.to_crs_string())
            }
            _ => None,
        };

        let executor = WkbBytesExecutor::new(arg_types, args);
        let mut builder = UInt64Builder::with_capacity(executor.num_iterations());

        // If we have a source CRS (i.e., the source CRS was present and not lon/lat already)
        // resolve the transform and apply it to the first coord before applying the order_fn.
        // Otherwise, skip the transform and go straight to the order_fn. This approach allows
        // this to be used even if PROJ isn't available (as long as the data were lon/lat
        // already).
        if let Some(src_crs) = maybe_src_crs {
            with_global_proj_engine(|engine| {
                let to_lnglat = engine
                    .get_transform_crs_to_crs(&src_crs, "OGC:CRS84", None, "")
                    .map_err(|e| DataFusionError::Execution(format!("{e}")))?;

                executor.execute_wkb_void(|maybe_wkb| {
                    match maybe_wkb {
                        Some(wkb_bytes) => {
                            let header = WkbHeader::try_new(wkb_bytes)
                                .map_err(|e| DataFusionError::Execution(format!("{e}")))?;
                            let mut first_xy = header.first_xy();
                            to_lnglat
                                .transform_coord(&mut first_xy)
                                .map_err(|e| DataFusionError::Execution(format!("{e}")))?;
                            let order = (self.order_fn)(first_xy);
                            builder.append_value(order);
                        }
                        None => builder.append_null(),
                    }

                    Ok(())
                })?;

                Ok(())
            })?;
        } else {
            executor.execute_wkb_void(|maybe_wkb| {
                match maybe_wkb {
                    Some(wkb_bytes) => {
                        let header = WkbHeader::try_new(wkb_bytes)
                            .map_err(|e| DataFusionError::Execution(format!("{e}")))?;
                        let first_xy = header.first_xy();
                        let order = (self.order_fn)(first_xy);
                        builder.append_value(order);
                    }
                    None => builder.append_null(),
                }

                Ok(())
            })?;
        }

        executor.finish(Arc::new(builder.finish()))
    }
}

#[cfg(test)]
mod test {

    use arrow_array::{create_array, ArrayRef};
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{Edges, WKB_GEOMETRY};
    use sedona_testing::{create::create_array, testers::ScalarUdfTester};

    use super::*;

    #[test]
    fn order_geometry() {
        // For testing, sort by first (rounded) x value
        let kernel = OrderLngLat::new(|(lng, lat)| {
            if lng.is_nan() || lat.is_nan() {
                u64::MAX
            } else {
                lng as u64
            }
        });
        let udf = SedonaScalarUDF::from_kernel("sd_order", Arc::new(kernel));

        let array = create_array(
            &[
                // POINT (1 2) in EPSG:3857
                Some("POINT (111320 222685)"),
                Some("POINT EMPTY"),
                None,
                // POINT (0 1) in EPSG:3857
                Some("POINT (0 111326)"),
            ],
            &WKB_GEOMETRY,
        );

        // Check the None Crs case
        let tester = ScalarUdfTester::new(udf.clone().into(), vec![WKB_GEOMETRY]);
        tester.assert_return_type(DataType::UInt64);

        let result = tester.invoke_array(array.clone()).unwrap();
        let expected =
            create_array!(UInt64, [Some(111320), Some(u64::MAX), None, Some(0)]) as ArrayRef;
        assert_eq!(&result, &expected);

        // Check the "already lnglat" case
        let tester = ScalarUdfTester::new(
            udf.clone().into(),
            vec![SedonaType::Wkb(Edges::Planar, lnglat())],
        );
        tester.assert_return_type(DataType::UInt64);

        let result = tester.invoke_array(array.clone()).unwrap();
        let expected =
            create_array!(UInt64, [Some(111320), Some(u64::MAX), None, Some(0)]) as ArrayRef;
        assert_eq!(&result, &expected);

        // Check the "not already lnglat" case
        let crs = sedona_schema::crs::deserialize_crs("EPSG:3857").unwrap();
        let tester = ScalarUdfTester::new(
            udf.clone().into(),
            vec![SedonaType::Wkb(Edges::Planar, crs)],
        );
        tester.assert_return_type(DataType::UInt64);

        let result = tester.invoke_array(array.clone()).unwrap();
        let expected = create_array!(UInt64, [Some(1), Some(u64::MAX), None, Some(0)]) as ArrayRef;
        assert_eq!(&result, &expected);
    }
}
