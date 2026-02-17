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

use crate::executor::RasterExecutor;
use arrow_array::builder::Float64Builder;
use arrow_schema::DataType;
use datafusion_common::error::Result;
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::affine_transformation::rotation;
use sedona_raster::traits::RasterRef;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

/// RS_UpperLeftX() scalar UDF implementation
///
/// Extract the raster's upper left corner's
/// X coordinate
pub fn rs_upperleftx_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_upperleftx",
        vec![Arc::new(RsGeoTransform {
            param: GeoTransformParam::UpperLeftX,
        })],
        Volatility::Immutable,
    )
}

/// RS_UpperLeftY() scalar UDF implementation
///
/// Extract the raster's upper left corner's
/// Y coordinate
pub fn rs_upperlefty_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_upperlefty",
        vec![Arc::new(RsGeoTransform {
            param: GeoTransformParam::UpperLeftY,
        })],
        Volatility::Immutable,
    )
}

/// RS_ScaleX() scalar UDF implementation
///
/// Extract the raster's pixel width or scale parameter
/// in the X direction
pub fn rs_scalex_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_scalex",
        vec![Arc::new(RsGeoTransform {
            param: GeoTransformParam::ScaleX,
        })],
        Volatility::Immutable,
    )
}

/// RS_ScaleY() scalar UDF implementation
///
/// Extract the raster's pixel height or scale
/// parameter in the Y direction
pub fn rs_scaley_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_scaley",
        vec![Arc::new(RsGeoTransform {
            param: GeoTransformParam::ScaleY,
        })],
        Volatility::Immutable,
    )
}

/// RS_SkewX() scalar UDF implementation
///
/// Extract the raster's X skew (rotation) parameter
/// from the geotransform
pub fn rs_skewx_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_skewx",
        vec![Arc::new(RsGeoTransform {
            param: GeoTransformParam::SkewX,
        })],
        Volatility::Immutable,
    )
}

/// RS_SkewY() scalar UDF implementation
///
/// Extract the raster's Y skew (rotation) parameter
/// from the geotransform.
pub fn rs_skewy_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_skewy",
        vec![Arc::new(RsGeoTransform {
            param: GeoTransformParam::SkewY,
        })],
        Volatility::Immutable,
    )
}

/// RS_Rotation() scalar UDF implementation
///
/// Calculate the uniform rotation of the raster
/// in radians based on the skew parameters.
pub fn rs_rotation_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_rotation",
        vec![Arc::new(RsGeoTransform {
            param: GeoTransformParam::Rotation,
        })],
        Volatility::Immutable,
    )
}

#[derive(Debug, Clone)]
enum GeoTransformParam {
    Rotation,
    ScaleX,
    ScaleY,
    SkewX,
    SkewY,
    UpperLeftX,
    UpperLeftY,
}

#[derive(Debug)]
struct RsGeoTransform {
    param: GeoTransformParam,
}

impl SedonaScalarKernel for RsGeoTransform {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_raster()],
            SedonaType::Arrow(DataType::Float64),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let mut builder = Float64Builder::with_capacity(executor.num_iterations());

        executor.execute_raster_void(|_i, raster_opt| {
            match raster_opt {
                None => builder.append_null(),
                Some(raster) => {
                    let metadata = raster.metadata();
                    match self.param {
                        GeoTransformParam::Rotation => {
                            let rotation = rotation(raster);
                            builder.append_value(rotation);
                        }
                        GeoTransformParam::ScaleX => builder.append_value(metadata.scale_x()),
                        GeoTransformParam::ScaleY => builder.append_value(metadata.scale_y()),
                        GeoTransformParam::SkewX => builder.append_value(metadata.skew_x()),
                        GeoTransformParam::SkewY => builder.append_value(metadata.skew_y()),
                        GeoTransformParam::UpperLeftX => {
                            builder.append_value(metadata.upper_left_x())
                        }
                        GeoTransformParam::UpperLeftY => {
                            builder.append_value(metadata.upper_left_y())
                        }
                    }
                }
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Float64Array;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::RASTER;
    use sedona_testing::compare::assert_array_equal;
    use sedona_testing::rasters::generate_test_rasters;
    use sedona_testing::testers::ScalarUdfTester;

    #[test]
    fn udf_info() {
        let udf: ScalarUDF = rs_rotation_udf().into();
        assert_eq!(udf.name(), "rs_rotation");

        let udf: ScalarUDF = rs_scalex_udf().into();
        assert_eq!(udf.name(), "rs_scalex");

        let udf: ScalarUDF = rs_scaley_udf().into();
        assert_eq!(udf.name(), "rs_scaley");

        let udf: ScalarUDF = rs_skewx_udf().into();
        assert_eq!(udf.name(), "rs_skewx");

        let udf: ScalarUDF = rs_skewy_udf().into();
        assert_eq!(udf.name(), "rs_skewy");

        let udf: ScalarUDF = rs_upperleftx_udf().into();
        assert_eq!(udf.name(), "rs_upperleftx");

        let udf: ScalarUDF = rs_upperlefty_udf().into();
        assert_eq!(udf.name(), "rs_upperlefty");
    }

    #[rstest]
    fn udf_invoke(
        #[values(
            GeoTransformParam::Rotation,
            GeoTransformParam::ScaleX,
            GeoTransformParam::ScaleY,
            GeoTransformParam::SkewX,
            GeoTransformParam::SkewY,
            GeoTransformParam::UpperLeftX,
            GeoTransformParam::UpperLeftY
        )]
        g: GeoTransformParam,
    ) {
        let udf = match g {
            GeoTransformParam::Rotation => rs_rotation_udf(),
            GeoTransformParam::ScaleX => rs_scalex_udf(),
            GeoTransformParam::ScaleY => rs_scaley_udf(),
            GeoTransformParam::SkewX => rs_skewx_udf(),
            GeoTransformParam::SkewY => rs_skewy_udf(),
            GeoTransformParam::UpperLeftX => rs_upperleftx_udf(),
            GeoTransformParam::UpperLeftY => rs_upperlefty_udf(),
        };
        let tester = ScalarUdfTester::new(udf.into(), vec![RASTER]);

        let rasters = generate_test_rasters(3, Some(1)).unwrap();
        let expected_values = match g {
            GeoTransformParam::Rotation => vec![Some(-0.0), None, Some(-0.29145679447786704)],
            GeoTransformParam::ScaleX => vec![Some(0.1), None, Some(0.2)],
            GeoTransformParam::ScaleY => vec![Some(-0.2), None, Some(-0.4)],
            GeoTransformParam::SkewX => vec![Some(0.0), None, Some(0.06)],
            GeoTransformParam::SkewY => vec![Some(0.0), None, Some(0.08)],
            GeoTransformParam::UpperLeftX => vec![Some(1.0), None, Some(3.0)],
            GeoTransformParam::UpperLeftY => vec![Some(2.0), None, Some(4.0)],
        };

        let expected: Arc<dyn arrow_array::Array> = Arc::new(Float64Array::from(expected_values));

        let result = tester.invoke_array(Arc::new(rasters)).unwrap();
        assert_array_equal(&result, &expected);
    }
}
