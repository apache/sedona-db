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
use datafusion_common::{error::Result, exec_err, DataFusionError, ScalarValue};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use geo_traits::{
    CoordTrait, GeometryCollectionTrait as _, GeometryTrait, LineStringTrait,
    MultiLineStringTrait as _, MultiPointTrait as _, MultiPolygonTrait as _, PointTrait,
    PolygonTrait as _,
};
use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{SedonaScalarKernel, SedonaScalarUDF},
};
use sedona_geometry::wkb_factory::{
    write_wkb_coord, write_wkb_coord_trait, write_wkb_empty_point,
    write_wkb_geometrycollection_header, write_wkb_linestring_header,
    write_wkb_multilinestring_header, write_wkb_multipoint_header, write_wkb_multipolygon_header,
    write_wkb_point_header, write_wkb_polygon_header, write_wkb_polygon_ring_header,
    WKB_MIN_PROBABLE_BYTES,
};
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};
use std::{io::Write, sync::Arc};

use crate::executor::WkbExecutor;

/// ST_Affine() scalar UDF
///
/// Native implementation for affine transformation
pub fn st_affine_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_affine",
        ItemCrsKernel::wrap_impl(vec![
            Arc::new(STAffine { is_3d: true }),
            Arc::new(STAffine { is_3d: false }),
        ]),
        Volatility::Immutable,
        Some(st_affine_doc()),
    )
}

fn st_affine_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Apply an affine transformation to the given geometry.",
        "ST_Affine (geom: Geometry, a: Double, b: Double, c: Double, d: Double, e: Double, f: Double, g: Double, h: Double, i: Double, xOff: Double, yOff: Double, zOff: Double)",
    )
    .with_argument("geom", "geometry: Input geometry")
    .with_argument("a", "a component of the affine matrix")
    .with_argument("b", "a component of the affine matrix")
    .with_argument("c", "a component of the affine matrix")
    .with_argument("d", "a component of the affine matrix")
    .with_argument("e", "a component of the affine matrix")
    .with_argument("f", "a component of the affine matrix")
    .with_argument("g", "a component of the affine matrix")
    .with_argument("h", "a component of the affine matrix")
    .with_argument("i", "a component of the affine matrix")
    .with_argument("xOff", "X offset")
    .with_argument("yOff", "Y offset")
    .with_argument("zOff", "Z offset")
    .with_sql_example("SELECT ST_Affine(ST_GeomFromText('POLYGON Z ((1 0 1, 1 1 1, 2 2 2, 1 0 1))'), 1, 2, 4, 1, 1, 2, 3, 2, 5, 4, 8, 3)")
    .build()
}

#[derive(Debug)]
struct STAffine {
    is_3d: bool,
}

impl SedonaScalarKernel for STAffine {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let arg_matchers = if self.is_3d {
            vec![
                ArgMatcher::is_geometry(),
                ArgMatcher::is_numeric(), // a
                ArgMatcher::is_numeric(), // b
                ArgMatcher::is_numeric(), // c
                ArgMatcher::is_numeric(), // d
                ArgMatcher::is_numeric(), // e
                ArgMatcher::is_numeric(), // f
                ArgMatcher::is_numeric(), // g
                ArgMatcher::is_numeric(), // h
                ArgMatcher::is_numeric(), // i
                ArgMatcher::is_numeric(), // xOff
                ArgMatcher::is_numeric(), // yOff
                ArgMatcher::is_numeric(), // zOff
            ]
        } else {
            vec![
                ArgMatcher::is_geometry(),
                ArgMatcher::is_numeric(), // a
                ArgMatcher::is_numeric(), // b
                ArgMatcher::is_numeric(), // d
                ArgMatcher::is_numeric(), // e
                ArgMatcher::is_numeric(), // xOff
                ArgMatcher::is_numeric(), // yOff
            ]
        };

        let matcher = ArgMatcher::new(arg_matchers, WKB_GEOMETRY);

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);
        let mut builder = BinaryBuilder::with_capacity(
            executor.num_iterations(),
            WKB_MIN_PROBABLE_BYTES * executor.num_iterations(),
        );

        let a = get_scalar_f64(args, 1)?;
        let b = get_scalar_f64(args, 2)?;
        let c = get_scalar_f64(args, 3)?;
        let d = get_scalar_f64(args, 4)?;
        let e = get_scalar_f64(args, 5)?;
        let f = get_scalar_f64(args, 6)?;
        let g = get_scalar_f64(args, 7)?;
        let h = get_scalar_f64(args, 8)?;
        let i = get_scalar_f64(args, 9)?;
        let x_offset = get_scalar_f64(args, 10)?;
        let y_offset = get_scalar_f64(args, 11)?;
        let z_offset = get_scalar_f64(args, 12)?;

        let mat = glam::DAffine3::from_cols_array(&[
            a, b, c, d, e, f, g, h, i, x_offset, y_offset, z_offset,
        ]);

        executor.execute_wkb_void(|maybe_wkb| {
            match maybe_wkb {
                Some(wkb) => {
                    match (self.is_3d, wkb.dim()) {
                        (true, geo_traits::Dimensions::Xyz) => {}
                        (true, geo_traits::Dimensions::Xyzm) => {}
                        (true, _) => {
                            return exec_err!("The geometry is 2D while the affine matrix is 3D")
                        }
                        (false, geo_traits::Dimensions::Xy) => {}
                        (false, geo_traits::Dimensions::Xym) => {}
                        (false, _) => {
                            return exec_err!("The geometry is 3D while the affine matrix is 2D")
                        }
                    }
                    invoke_scalar(&wkb, &mut builder, &mat)?;
                    builder.append_value([]);
                }
                None => builder.append_null(),
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn get_scalar_f64(args: &[ColumnarValue], index: usize) -> Result<f64> {
    let v = match args.get(index) {
        Some(ColumnarValue::Scalar(scalar_value)) => match scalar_value {
            ScalarValue::Float16(Some(v)) => f64::from(*v),
            ScalarValue::Float32(Some(v)) => f64::from(*v),
            ScalarValue::Float64(Some(v)) => f64::from(*v),
            ScalarValue::Int8(Some(v)) => f64::from(*v),
            ScalarValue::Int16(Some(v)) => f64::from(*v),
            ScalarValue::Int32(Some(v)) => f64::from(*v),
            ScalarValue::Int64(Some(v)) => *v as f64, // lossy conversion
            ScalarValue::UInt8(Some(v)) => f64::from(*v),
            ScalarValue::UInt16(Some(v)) => f64::from(*v),
            ScalarValue::UInt32(Some(v)) => f64::from(*v),
            ScalarValue::UInt64(Some(v)) => *v as f64, // lossy conversion
            _ => return exec_err!("Affine matrix must be numeric"),
        },
        _ => return exec_err!("Affine matrix must be numeric"),
    };

    if v.is_nan() {
        return exec_err!("Affine matrix must not contain NAN");
    }

    Ok(v)
}

fn invoke_scalar(
    geom: &impl GeometryTrait<T = f64>,
    writer: &mut impl Write,
    mat: &glam::DAffine3,
) -> Result<()> {
    let dims = geom.dim();
    match geom.as_type() {
        geo_traits::GeometryType::Point(pt) => {
            if pt.coord().is_some() {
                write_wkb_point_header(writer, dims)
                    .map_err(|e| DataFusionError::Execution(e.to_string()))?;
                write_wkb_coord_trait(writer, &pt.coord().unwrap())
                    .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            } else {
                write_wkb_empty_point(writer, dims)
                    .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            }
        }

        geo_traits::GeometryType::MultiPoint(multi_point) => {
            write_wkb_multipoint_header(writer, dims, multi_point.points().count())
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            for pt in multi_point.points() {
                invoke_scalar(&pt, writer, mat)?;
            }
        }

        geo_traits::GeometryType::LineString(ls) => {
            write_wkb_linestring_header(writer, dims, ls.coords().count())
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            write_transformed_coords(writer, ls.coords(), mat)?;
        }

        geo_traits::GeometryType::Polygon(pgn) => {
            let num_rings = pgn.interiors().count() + pgn.exterior().is_some() as usize;
            write_wkb_polygon_header(writer, dims, num_rings)
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;

            if let Some(exterior) = pgn.exterior() {
                write_transformed_ring(writer, exterior, mat)?;
            }

            for interior in pgn.interiors() {
                write_transformed_ring(writer, interior, mat)?;
            }
        }

        geo_traits::GeometryType::MultiLineString(mls) => {
            write_wkb_multilinestring_header(writer, dims, mls.line_strings().count())
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            for ls in mls.line_strings() {
                invoke_scalar(&ls, writer, mat)?;
            }
        }

        geo_traits::GeometryType::MultiPolygon(mpgn) => {
            write_wkb_multipolygon_header(writer, dims, mpgn.polygons().count())
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            for pgn in mpgn.polygons() {
                invoke_scalar(&pgn, writer, mat)?;
            }
        }

        geo_traits::GeometryType::GeometryCollection(gcn) => {
            write_wkb_geometrycollection_header(writer, dims, gcn.geometries().count())
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            for geom in gcn.geometries() {
                invoke_scalar(&geom, writer, mat)?;
            }
        }

        _ => {
            return Err(DataFusionError::Execution(
                "Unsupported geometry type for reversal operation".to_string(),
            ));
        }
    }
    Ok(())
}

fn write_transformed_ring(
    writer: &mut impl Write,
    ring: impl LineStringTrait<T = f64>,
    affine: &glam::DAffine3,
) -> Result<()> {
    write_wkb_polygon_ring_header(writer, ring.coords().count())
        .map_err(|e| DataFusionError::Execution(e.to_string()))?;
    write_transformed_coords(writer, ring.coords(), affine)
}

fn write_transformed_coords<I>(
    writer: &mut impl Write,
    coords: I,
    affine: &glam::DAffine3,
) -> Result<()>
where
    I: DoubleEndedIterator,
    I::Item: CoordTrait<T = f64>,
{
    coords.into_iter().try_for_each(|coord| {
        let transformed = affine.transform_point3(glam::DVec3::new(
            coord.x(),
            coord.y(),
            coord.nth(2).unwrap_or(f64::NAN),
        ));
        write_wkb_coord(writer, (transformed.x, transformed.y, transformed.z))
            .map_err(|e| DataFusionError::Execution(e.to_string()))
    })
}
