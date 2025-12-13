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
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
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
        Some(rs_upperleftx_doc()),
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
        Some(rs_upperlefty_doc()),
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
        Some(rs_scalex_doc()),
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
        Some(rs_scaley_doc()),
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
        Some(rs_skewx_doc()),
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
        Some(rs_skewy_doc()),
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
        Some(rs_rotation_doc()),
    )
}

fn rs_upperleftx_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns the X coordinate of the upper-left corner of the raster.".to_string(),
        "RS_UpperLeftX(raster: Raster)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_sql_example("SELECT RS_UpperLeftX(RS_Example())".to_string())
    .build()
}

fn rs_upperlefty_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns the Y coordinate of the upper-left corner of the raster.".to_string(),
        "RS_UpperLeftY(raster: Raster)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_sql_example("SELECT RS_UpperLeftY(RS_Example())".to_string())
    .build()
}

fn rs_scalex_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns the pixel width of the raster in CRS units.".to_string(),
        "RS_ScaleX(raster: Raster)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_sql_example("SELECT RS_ScaleX(RS_Example())".to_string())
    .build()
}

fn rs_scaley_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns the pixel height of the raster in CRS units.".to_string(),
        "RS_ScaleY(raster: Raster)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_sql_example("SELECT RS_ScaleY(RS_Example())".to_string())
    .build()
}

fn rs_skewx_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns the X skew or rotation parameter.".to_string(),
        "RS_SkewX(raster: Raster)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_sql_example("SELECT RS_SkewX(RS_Example())".to_string())
    .build()
}

fn rs_skewy_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns the Y skew or rotation parameter.".to_string(),
        "RS_SkewY(raster: Raster)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_sql_example("SELECT RS_SkewY(RS_Example())".to_string())
    .build()
}

fn rs_rotation_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns the uniform rotation of the raster in radians.".to_string(),
        "RS_Rotation(raster: Raster)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_sql_example("SELECT RS_Rotation(RS_Example())".to_string())
    .build()
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
                            let rotation = rotation(&raster);
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
        assert!(udf.documentation().is_some());

        let udf: ScalarUDF = rs_scalex_udf().into();
        assert_eq!(udf.name(), "rs_scalex");
        assert!(udf.documentation().is_some());

        let udf: ScalarUDF = rs_scaley_udf().into();
        assert_eq!(udf.name(), "rs_scaley");
        assert!(udf.documentation().is_some());

        let udf: ScalarUDF = rs_skewx_udf().into();
        assert_eq!(udf.name(), "rs_skewx");
        assert!(udf.documentation().is_some());

        let udf: ScalarUDF = rs_skewy_udf().into();
        assert_eq!(udf.name(), "rs_skewy");
        assert!(udf.documentation().is_some());

        let udf: ScalarUDF = rs_upperleftx_udf().into();
        assert_eq!(udf.name(), "rs_upperleftx");
        assert!(udf.documentation().is_some());

        let udf: ScalarUDF = rs_upperlefty_udf().into();
        assert_eq!(udf.name(), "rs_upperlefty");
        assert!(udf.documentation().is_some());
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
            GeoTransformParam::ScaleX => vec![Some(0.0), None, Some(0.2)],
            GeoTransformParam::ScaleY => vec![Some(-0.0), None, Some(-0.4)],
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
