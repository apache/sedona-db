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
use arrow_array::builder::BinaryBuilder;
use datafusion_common::DataFusionError;
use datafusion_common::Result;
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
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
        Some(rs_envelope_doc()),
    )
}

fn rs_envelope_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns the envelope of the raster as a Geometry.".to_string(),
        "RS_Envelope(raster: Raster)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_sql_example("SELECT RS_Envelope(RS_Example())".to_string())
    .build()
}

#[derive(Debug)]
struct RsEnvelope {}

impl SedonaScalarKernel for RsEnvelope {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_raster()],
            SedonaType::Wkb(Edges::Planar, None),
        );

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

        executor.execute_raster_void(|_i, raster_opt| {
            match raster_opt {
                Some(raster) => {
                    create_envelope_wkb(&raster, &mut builder)?;
                    builder.append_value([]);
                }
                None => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

/// Create WKB for a polygon for the raster
fn create_envelope_wkb(raster: &dyn RasterRef, out: &mut impl std::io::Write) -> Result<()> {
    // Compute the four corners of the raster in world coordinates.
    // Due to skew/rotation in the affine transformation, each corner must be
    // computed individually.

    let width = raster.metadata().width() as i64;
    let height = raster.metadata().height() as i64;

    // Compute the four corners in pixel coordinates:
    // Upper-left (0, 0), Upper-right (width, 0), Lower-right (width, height), Lower-left (0, height)
    let (ulx, uly) = to_world_coordinate(raster, 0, 0);
    let (urx, ury) = to_world_coordinate(raster, width, 0);
    let (lrx, lry) = to_world_coordinate(raster, width, height);
    let (llx, lly) = to_world_coordinate(raster, 0, height);

    write_wkb_polygon(
        out,
        [(ulx, uly), (urx, ury), (lrx, lry), (llx, lly), (ulx, uly)].into_iter(),
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
    use sedona_testing::create::create_array;
    use sedona_testing::rasters::generate_test_rasters;
    use sedona_testing::testers::ScalarUdfTester;

    #[test]
    fn udf_docs() {
        let udf: ScalarUDF = rs_envelope_udf().into();
        assert_eq!(udf.name(), "rs_envelope");
        assert!(udf.documentation().is_some());
    }

    #[rstest]
    fn udf_invoke() {
        let udf = rs_envelope_udf();
        let tester = ScalarUdfTester::new(udf.into(), vec![RASTER]);

        let rasters = generate_test_rasters(3, Some(0)).unwrap();

        // Corners computed using gdal:
        // Raster 1:
        // Envelope corner coordinates (X, Y):
        // (2.00000000, 3.00000000)
        // (2.20000000, 3.08000000)
        // (2.29000000, 2.48000000)
        // (2.09000000, 2.40000000)
        //
        // Raster 2:
        // (3.00000000, 4.00000000)
        // (3.60000000, 4.24000000)
        // (3.84000000, 2.64000000)
        // (3.24000000, 2.40000000)
        let expected = &create_array(
            &[
                None,
                Some("POLYGON ((2.0 3.0, 2.2 3.08, 2.29 2.48, 2.09 2.4, 2.0 3.0))"),
                Some("POLYGON ((3.0 4.0, 3.6 4.24, 3.84 2.64, 3.24 2.4, 3.0 4.0))"),
            ],
            &WKB_GEOMETRY,
        );

        let result = tester.invoke_array(Arc::new(rasters)).unwrap();

        assert_array_equal(&result, expected);
    }
}
