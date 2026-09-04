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
use std::f64::consts::FRAC_PI_2;
use std::{sync::Arc, vec};

use crate::executor::RasterExecutor;
use arrow_array::builder::Float64Builder;
use arrow_array::{ArrayRef, StructArray};
use arrow_buffer::NullBufferBuilder;
use arrow_schema::{DataType, Field, Fields};
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

/// RS_GeoTransform() scalar UDF implementation
///
/// Returns the raster's geotransform decomposed into pixel magnitudes along
/// the transformed i/j axes, rotation and axis-separation angles, and the
/// upper-left offsets, as a struct matching Sedona Spark's `RS_GeoTransform`.
pub fn rs_geotransform_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_geotransform",
        vec![Arc::new(RsGeoTransformComposite {})],
        Volatility::Immutable,
    )
}

fn geotransform_fields() -> Fields {
    Fields::from(vec![
        Field::new("magnitudeI", DataType::Float64, false),
        Field::new("magnitudeJ", DataType::Float64, false),
        Field::new("thetaI", DataType::Float64, false),
        Field::new("thetaIJ", DataType::Float64, false),
        Field::new("offsetX", DataType::Float64, false),
        Field::new("offsetY", DataType::Float64, false),
    ])
}

/// Decompose a GDAL-ordered geotransform into the six values Sedona Spark's
/// `RS_GeoTransform` reports. The math (including the sign conventions of the
/// two `acos` sign tests) mirrors Sedona Spark's
/// `RasterAccessors#getGeoTransform`, which is the parity target; note that
/// `thetaI` therefore only agrees with `RS_Rotation` when `|skewX| == |skewY|`.
fn decompose_geotransform(gt: &[f64]) -> [f64; 6] {
    let (offset_x, scale_x, skew_x) = (gt[0], gt[1], gt[2]);
    let (offset_y, skew_y, scale_y) = (gt[3], gt[4], gt[5]);

    // Pixel sizes along the transformed i (west-east) and j (north-south) axes
    let magnitude_i = (scale_x * scale_x + skew_y * skew_y).sqrt();
    let magnitude_j = (scale_y * scale_y + skew_x * skew_x).sqrt();

    // Rotation of the raster (radians, positive clockwise)
    let mut theta_i = (scale_x / magnitude_i).acos();
    if (skew_y / magnitude_i).acos() < FRAC_PI_2 {
        theta_i = -theta_i;
    }

    // Angle from the transformed i axis to the transformed j axis (radians,
    // positive counter-clockwise)
    let mut theta_ij = ((scale_x * skew_x + skew_y * scale_y) / (magnitude_i * magnitude_j)).acos();
    if ((-skew_y * skew_x + scale_x * scale_y) / (magnitude_i * magnitude_j)).acos() > FRAC_PI_2 {
        theta_ij = -theta_ij;
    }

    [
        magnitude_i,
        magnitude_j,
        theta_i,
        theta_ij,
        offset_x,
        offset_y,
    ]
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
                Some(raster) => match self.param {
                    GeoTransformParam::Rotation => {
                        let rotation = rotation(raster);
                        builder.append_value(rotation);
                    }
                    GeoTransformParam::ScaleX => builder.append_value(raster.transform()[1]),
                    GeoTransformParam::ScaleY => builder.append_value(raster.transform()[5]),
                    GeoTransformParam::SkewX => builder.append_value(raster.transform()[2]),
                    GeoTransformParam::SkewY => builder.append_value(raster.transform()[4]),
                    GeoTransformParam::UpperLeftX => builder.append_value(raster.transform()[0]),
                    GeoTransformParam::UpperLeftY => builder.append_value(raster.transform()[3]),
                },
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

#[derive(Debug)]
struct RsGeoTransformComposite {}

impl SedonaScalarKernel for RsGeoTransformComposite {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_raster()],
            SedonaType::Arrow(DataType::Struct(geotransform_fields())),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let num_iterations = executor.num_iterations();
        let mut builders: Vec<Float64Builder> = (0..6)
            .map(|_| Float64Builder::with_capacity(num_iterations))
            .collect();
        let mut validity = NullBufferBuilder::new(num_iterations);

        executor.execute_raster_void(|_i, raster_opt| {
            match raster_opt {
                None => {
                    validity.append_null();
                    // The fields are non-nullable, so null rows carry a
                    // placeholder in every child under a null struct slot.
                    for builder in builders.iter_mut() {
                        builder.append_value(0.0);
                    }
                }
                Some(raster) => {
                    validity.append_non_null();
                    let components = decompose_geotransform(raster.transform());
                    for (builder, component) in builders.iter_mut().zip(components) {
                        builder.append_value(component);
                    }
                }
            }
            Ok(())
        })?;

        let arrays: Vec<ArrayRef> = builders
            .iter_mut()
            .map(|builder| Arc::new(builder.finish()) as ArrayRef)
            .collect();
        let struct_array = StructArray::try_new(geotransform_fields(), arrays, validity.finish())?;
        executor.finish(Arc::new(struct_array))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Float64Array;
    use arrow_buffer::NullBuffer;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_raster::builder::RasterBuilder;
    use sedona_schema::datatypes::RASTER;
    use sedona_schema::raster::BandDataType;
    use sedona_testing::compare::assert_array_equal;
    use sedona_testing::rasters::generate_test_rasters;
    use sedona_testing::testers::ScalarUdfTester;

    #[test]
    fn udf_info() {
        let udf: ScalarUDF = rs_rotation_udf().into();
        assert_eq!(udf.name(), "rs_rotation");

        let udf: ScalarUDF = rs_geotransform_udf().into();
        assert_eq!(udf.name(), "rs_geotransform");

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

    /// A north-up raster, a null, and a sheared raster with distinct skews
    /// (equal skews are the degenerate regime where the magnitudes coincide
    /// and thetaIJ collapses to +/-pi/2). The transforms are chosen so every
    /// expected component below is exactly representable: the magnitudes are
    /// square roots of perfect squares (3-4-5 and 5-12-13) and the `acos`
    /// inputs are exact one-rounding quotients.
    fn build_composite_test_rasters() -> arrow_array::StructArray {
        let mut builder = RasterBuilder::new(3);
        for transform in [
            Some([1.0, 3.0, 0.0, 2.0, 0.0, -4.0]),
            None,
            Some([10.0, 4.0, 5.0, 20.0, 3.0, -12.0]),
        ] {
            match transform {
                None => builder.append_null().unwrap(),
                Some([ulx, scale_x, skew_x, uly, skew_y, scale_y]) => {
                    builder
                        .start_raster_2d(2, 2, ulx, uly, scale_x, scale_y, skew_x, skew_y, None)
                        .unwrap();
                    builder.start_band_2d(BandDataType::UInt8, None).unwrap();
                    builder.band_data_writer().append_value([0u8; 4]);
                    builder.finish_band().unwrap();
                    builder.finish_raster().unwrap();
                }
            }
        }
        builder.finish().unwrap()
    }

    #[test]
    fn udf_invoke_composite() {
        let tester = ScalarUdfTester::new(rs_geotransform_udf().into(), vec![RASTER]);

        // North-up: magnitudes are |scaleX|/|scaleY|, thetaI is acos(1) = 0,
        // and thetaIJ is acos(0) = pi/2 negated by its sign test (the
        // i-to-j separation of a y-down raster is -90 degrees).
        let north_up = [3.0, 4.0, 0.0, -FRAC_PI_2, 1.0, 2.0];
        // Sheared with distinct skews: magnitudeI = sqrt(4^2 + 3^2) = 5,
        // magnitudeJ = sqrt(12^2 + 5^2) = 13, thetaI = acos(4/5) negated
        // because skewY > 0, and thetaIJ = acos(-16/65) negated because its
        // sign test acos(-63/65) exceeds pi/2.
        let skewed = [
            5.0,
            13.0,
            -(4.0f64 / 5.0).acos(),
            -(-16.0f64 / 65.0).acos(),
            10.0,
            20.0,
        ];

        let columns: Vec<ArrayRef> = (0..6)
            .map(|i| Arc::new(Float64Array::from(vec![north_up[i], 0.0, skewed[i]])) as ArrayRef)
            .collect();
        let expected: Arc<dyn arrow_array::Array> = Arc::new(
            StructArray::try_new(
                geotransform_fields(),
                columns,
                Some(NullBuffer::from(vec![true, false, true])),
            )
            .unwrap(),
        );

        let result = tester
            .invoke_array(Arc::new(build_composite_test_rasters()))
            .unwrap();
        assert_array_equal(&result, &expected);
    }
}
