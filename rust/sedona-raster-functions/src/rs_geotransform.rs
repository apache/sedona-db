// Licensed to the Apache Software Foundatio//pub fn rs_upperleftx_udf() -> SedonaScapub fn rs_scalex_udf() pub fn rs_scaley_udf() -pub fn rs_skewx_udf() ->pub fn rs_skewy_udf() -> SedonaScalarUDF {
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

fn rs_upperleftx_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns the X coordinate of the upper-left corner of the raster.".to_string(),
        "RS_UpperLeftX(raster: Raster)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_sql_example("SELECT RS_UpperLeftX(raster)".to_string())
    .build()
}

fn rs_upperlefty_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns the Y coordinate of the upper-left corner of the raster.".to_string(),
        "RS_UpperLeftY(raster: Raster)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_sql_example("SELECT RS_UpperLeftY(raster)".to_string())
    .build()
}

fn rs_scalex_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns the pixel width of the raster in CRS units.".to_string(),
        "RS_ScaleX(raster: Raster)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_sql_example("SELECT RS_ScaleX(raster)".to_string())
    .build()
}

fn rs_scaley_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns the pixel height of the raster in CRS units.".to_string(),
        "RS_ScaleY(raster: Raster)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_sql_example("SELECT RS_ScaleY(raster)".to_string())
    .build()
}

fn rs_skewx_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns the X skew or rotation parameter.".to_string(),
        "RS_SkewX(raster: Raster)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_sql_example("SELECT RS_SkewX(raster)".to_string())
    .build()
}

fn rs_skewy_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns the Y skew or rotation parameter.".to_string(),
        "RS_SkewY(raster: Raster)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_sql_example("SELECT RS_SkewY(raster)".to_string())
    .build()
}

#[derive(Debug, Clone)]
enum GeoTransformParam {
    UpperLeftX,
    UpperLeftY,
    ScaleX,
    ScaleY,
    SkewX,
    SkewY,
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
                        GeoTransformParam::UpperLeftX => {
                            builder.append_value(metadata.upper_left_x())
                        }
                        GeoTransformParam::UpperLeftY => {
                            builder.append_value(metadata.upper_left_y())
                        }
                        GeoTransformParam::ScaleX => builder.append_value(metadata.scale_x()),
                        GeoTransformParam::ScaleY => builder.append_value(metadata.scale_y()),
                        GeoTransformParam::SkewX => builder.append_value(metadata.skew_x()),
                        GeoTransformParam::SkewY => builder.append_value(metadata.skew_y()),
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
    use arrow_array::{Array, Float64Array};
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::RASTER;
    use sedona_testing::rasters::generate_test_rasters;

    #[test]
    fn udf_size() {
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
            GeoTransformParam::UpperLeftX,
            GeoTransformParam::UpperLeftY,
            GeoTransformParam::ScaleX,
            GeoTransformParam::ScaleY,
            GeoTransformParam::SkewX,
            GeoTransformParam::SkewY
        )]
        g: GeoTransformParam,
    ) {
        let kernel = RsGeoTransform { param: g.clone() };
        // 3 rasters, second one is null
        let rasters = generate_test_rasters(3, Some(1)).unwrap();

        // Create the UDF and invoke it
        let args = [ColumnarValue::Array(Arc::new(rasters))];
        let arg_types = vec![RASTER];

        let result = kernel.invoke_batch(&arg_types, &args).unwrap();

        // Check the result
        if let ColumnarValue::Array(result_array) = result {
            let array = result_array
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();

            assert_eq!(array.len(), 3);

            match g.clone() {
                GeoTransformParam::UpperLeftX => assert_eq!(array.value(0), 1.0),
                GeoTransformParam::UpperLeftY => assert_eq!(array.value(0), 2.0),
                GeoTransformParam::ScaleX => assert_eq!(array.value(0), 0.0),
                GeoTransformParam::ScaleY => assert_eq!(array.value(0), 0.0),
                GeoTransformParam::SkewX => assert_eq!(array.value(0), 0.0),
                GeoTransformParam::SkewY => assert_eq!(array.value(0), 0.0),
            }
            assert!(array.is_null(1)); // Second raster is null
            match g.clone() {
                GeoTransformParam::UpperLeftX => assert_eq!(array.value(2), 3.0),
                GeoTransformParam::UpperLeftY => assert_eq!(array.value(2), 4.0),
                GeoTransformParam::ScaleX => assert_eq!(array.value(2), 0.2),
                GeoTransformParam::ScaleY => assert_eq!(array.value(2), 0.4),
                GeoTransformParam::SkewX => assert_eq!(array.value(2), 0.6),
                GeoTransformParam::SkewY => assert_eq!(array.value(2), 0.8),
            }
        } else {
            panic!("Expected array result");
        }
    }
}
