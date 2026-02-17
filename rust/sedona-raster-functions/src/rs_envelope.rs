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

use crate::executor::RasterExecutor;
use arrow_array::builder::{BinaryBuilder, StringViewBuilder};
use datafusion_common::DataFusionError;
use datafusion_common::Result;
use datafusion_common::ScalarValue;
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_expr::item_crs::make_item_crs;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_geometry::wkb_factory::write_wkb_polygon;
use sedona_raster::affine_transformation::to_world_coordinate;
use sedona_raster::traits::RasterRef;
use sedona_schema::datatypes::Edges;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

/// RS_Envelope() scalar UDF documentation
///
/// Returns the envelope (bounding box) of the given raster as a WKB Polygon.
pub fn rs_envelope_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_envelope",
        vec![Arc::new(RsEnvelope {})],
        Volatility::Immutable,
        None,
    )
}

#[derive(Debug)]
struct RsEnvelope {}

impl SedonaScalarKernel for RsEnvelope {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let out_type = SedonaType::new_item_crs(&SedonaType::Wkb(Edges::Planar, None))?;
        let matcher = ArgMatcher::new(vec![ArgMatcher::is_raster()], out_type);

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        // 1 (byte order) + 4 (type) + 4 (num rings) + 4 (num points) + 80 (5 points * 16 bytes)
        let bytes_per_poly = 93;
        let mut builder = BinaryBuilder::with_capacity(
            executor.num_iterations(),
            executor.num_iterations() * bytes_per_poly,
        );
        let mut crs_builder = StringViewBuilder::with_capacity(executor.num_iterations());

        executor.execute_raster_void(|_i, raster_opt| {
            match raster_opt {
                Some(raster) => {
                    write_envelope_wkb(raster, &mut builder)?;
                    builder.append_value([]);
                    crs_builder.append_value(raster.crs().unwrap_or("0"));
                }
                None => {
                    builder.append_null();
                    crs_builder.append_null();
                }
            }
            Ok(())
        })?;

        let item_array = builder.finish();
        let item_result = executor.finish(Arc::new(item_array))?;
        let crs_array = crs_builder.finish();
        let crs_value = if matches!(item_result, ColumnarValue::Scalar(_)) {
            ColumnarValue::Scalar(ScalarValue::try_from_array(&crs_array, 0)?)
        } else {
            ColumnarValue::Array(Arc::new(crs_array))
        };

        make_item_crs(
            &SedonaType::Wkb(Edges::Planar, None),
            item_result,
            &crs_value,
            None,
        )
    }
}

/// Write WKB for the axis-aligned bounding box (envelope) of the raster.
///
/// This computes the four corners of the raster in world coordinates, then
/// derives the min/max X and Y to produce an axis-aligned bounding box.
/// For skewed/rotated rasters, this differs from the convex hull.
fn write_envelope_wkb(raster: &dyn RasterRef, out: &mut impl std::io::Write) -> Result<()> {
    let width = raster.metadata().width() as i64;
    let height = raster.metadata().height() as i64;

    // Compute the four corners in world coordinates
    let (ulx, uly) = to_world_coordinate(raster, 0, 0);
    let (urx, ury) = to_world_coordinate(raster, width, 0);
    let (lrx, lry) = to_world_coordinate(raster, width, height);
    let (llx, lly) = to_world_coordinate(raster, 0, height);

    // Compute the axis-aligned bounding box
    let min_x = ulx.min(urx).min(lrx).min(llx);
    let max_x = ulx.max(urx).max(lrx).max(llx);
    let min_y = uly.min(ury).min(lry).min(lly);
    let max_y = uly.max(ury).max(lry).max(lly);

    write_wkb_polygon(
        out,
        [
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y),
            (min_x, min_y),
        ]
        .into_iter(),
    )
    .map_err(|e| DataFusionError::External(e.into()))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::RASTER;
    use sedona_schema::datatypes::WKB_GEOMETRY;
    use sedona_testing::compare::assert_array_equal;
    use sedona_testing::create::create_array_item_crs;
    use sedona_testing::rasters::generate_test_rasters;
    use sedona_testing::testers::ScalarUdfTester;

    #[test]
    fn udf_docs() {
        let udf: ScalarUDF = rs_envelope_udf().into();
        assert_eq!(udf.name(), "rs_envelope");
    }

    #[rstest]
    fn udf_invoke() {
        let udf = rs_envelope_udf();
        let tester = ScalarUdfTester::new(udf.into(), vec![RASTER]);

        // i=0: no skew, i=1: null, i=2: skewed (scale=0.2,-0.4, skew=0.06,0.08)
        let rasters = generate_test_rasters(3, Some(1)).unwrap();

        // Reference values verified against PostGIS ST_Envelope:
        //
        // Raster 0 (i=0): width=1, height=2, ul=(1,2), scale=(0.1,-0.2), skew=(0,0)
        //   No skew, so envelope = convex hull
        //
        // Raster 2 (i=2): width=3, height=4, ul=(3,4), scale=(0.2,-0.4), skew=(0.06,0.08)
        //   Corners: (3,4), (3.6,4.24), (3.84,2.64), (3.24,2.4)
        //   AABB: x=[3, 3.84], y=[2.4, 4.24]
        let expected = &create_array_item_crs(
            &[
                Some("POLYGON ((1 1.6, 1.1 1.6, 1.1 2, 1 2, 1 1.6))"),
                None,
                Some("POLYGON ((3 2.4, 3.84 2.4, 3.84 4.24, 3 4.24, 3 2.4))"),
            ],
            [Some("OGC:CRS84"), None, Some("OGC:CRS84")],
            &WKB_GEOMETRY,
        );

        let result = tester.invoke_array(Arc::new(rasters)).unwrap();

        assert_array_equal(&result, expected);
    }
}
