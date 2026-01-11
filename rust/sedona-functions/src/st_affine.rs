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
use arrow_array::{builder::BinaryBuilder, types::Float64Type, Array, PrimitiveArray};
use arrow_schema::DataType;
use datafusion_common::{
    cast::as_float64_array, error::Result, exec_err, internal_err, DataFusionError, ScalarValue,
};
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
    write_wkb_coord, write_wkb_empty_point, write_wkb_geometrycollection_header,
    write_wkb_linestring_header, write_wkb_multilinestring_header, write_wkb_multipoint_header,
    write_wkb_multipolygon_header, write_wkb_point_header, write_wkb_polygon_header,
    write_wkb_polygon_ring_header, WKB_MIN_PROBABLE_BYTES,
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

        let array_args = args[1..]
            .iter()
            .map(|arg| {
                arg.cast_to(&DataType::Float64, None)?
                    .to_array(executor.num_iterations())
            })
            .collect::<Result<Vec<Arc<dyn Array>>>>()?;

        let mut affine_iter = DAffine3Iterator::new(&array_args)?;

        executor.execute_wkb_void(|maybe_wkb| {
            match maybe_wkb {
                Some(wkb) => {
                    let mat = affine_iter.next().unwrap();

                    let dim = wkb.dim();
                    if matches!(
                        dim,
                        geo_traits::Dimensions::Xyz | geo_traits::Dimensions::Xyzm
                    ) {
                        invoke_scalar(&wkb, &mut builder, &mat, &dim)?;
                        builder.append_value([]);
                    } else {
                        return exec_err!("The geometry is 2D while the affine matrix is 3D");
                    }
                }
                None => builder.append_null(),
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(
    geom: &impl GeometryTrait<T = f64>,
    writer: &mut impl Write,
    mat: &glam::DAffine3,
    dim: &geo_traits::Dimensions,
) -> Result<()> {
    let dims = geom.dim();
    match geom.as_type() {
        geo_traits::GeometryType::Point(pt) => {
            if pt.coord().is_some() {
                write_wkb_point_header(writer, dims)
                    .map_err(|e| DataFusionError::Execution(e.to_string()))?;
                write_transformed_coord_3d(writer, pt.coord().unwrap(), mat, dim)?;
            } else {
                write_wkb_empty_point(writer, dims)
                    .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            }
        }

        geo_traits::GeometryType::MultiPoint(multi_point) => {
            write_wkb_multipoint_header(writer, dims, multi_point.points().count())
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            for pt in multi_point.points() {
                invoke_scalar(&pt, writer, mat, dim)?;
            }
        }

        geo_traits::GeometryType::LineString(ls) => {
            write_wkb_linestring_header(writer, dims, ls.coords().count())
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            write_transformed_coords(writer, ls.coords(), mat, dim)?;
        }

        geo_traits::GeometryType::Polygon(pgn) => {
            let num_rings = pgn.interiors().count() + pgn.exterior().is_some() as usize;
            write_wkb_polygon_header(writer, dims, num_rings)
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;

            if let Some(exterior) = pgn.exterior() {
                write_transformed_ring(writer, exterior, mat, dim)?;
            }

            for interior in pgn.interiors() {
                write_transformed_ring(writer, interior, mat, dim)?;
            }
        }

        geo_traits::GeometryType::MultiLineString(mls) => {
            write_wkb_multilinestring_header(writer, dims, mls.line_strings().count())
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            for ls in mls.line_strings() {
                invoke_scalar(&ls, writer, mat, dim)?;
            }
        }

        geo_traits::GeometryType::MultiPolygon(mpgn) => {
            write_wkb_multipolygon_header(writer, dims, mpgn.polygons().count())
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            for pgn in mpgn.polygons() {
                invoke_scalar(&pgn, writer, mat, dim)?;
            }
        }

        geo_traits::GeometryType::GeometryCollection(gcn) => {
            write_wkb_geometrycollection_header(writer, dims, gcn.geometries().count())
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            for geom in gcn.geometries() {
                invoke_scalar(&geom, writer, mat, dim)?;
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
    dim: &geo_traits::Dimensions,
) -> Result<()> {
    write_wkb_polygon_ring_header(writer, ring.coords().count())
        .map_err(|e| DataFusionError::Execution(e.to_string()))?;
    write_transformed_coords(writer, ring.coords(), affine, dim)
}

fn write_transformed_coords<I>(
    writer: &mut impl Write,
    coords: I,
    affine: &glam::DAffine3,
    dim: &geo_traits::Dimensions,
) -> Result<()>
where
    I: DoubleEndedIterator,
    I::Item: CoordTrait<T = f64>,
{
    coords
        .into_iter()
        .try_for_each(|coord| write_transformed_coord_3d(writer, coord, affine, dim))
}

fn write_transformed_coord_2d<C>(
    writer: &mut impl Write,
    coord: C,
    affine: &glam::DAffine2,
    dim: &geo_traits::Dimensions,
) -> Result<()>
where
    C: CoordTrait<T = f64>,
{
    let transformed = affine.transform_point2(glam::DVec2::new(coord.x(), coord.y()));

    match dim {
        geo_traits::Dimensions::Xy => write_wkb_coord(writer, (transformed.x, transformed.y))
            .map_err(|e| DataFusionError::Execution(e.to_string())),
        geo_traits::Dimensions::Xym => {
            // Preserve m value
            let m = coord.nth(3).unwrap();
            write_wkb_coord(writer, (transformed.x, transformed.y, m))
                .map_err(|e| DataFusionError::Execution(e.to_string()))
        }
        _ => {
            return internal_err!("3D dimension is passed to 2D affine transformation.");
        }
    }
}

fn write_transformed_coord_3d<C>(
    writer: &mut impl Write,
    coord: C,
    affine: &glam::DAffine3,
    dim: &geo_traits::Dimensions,
) -> Result<()>
where
    C: CoordTrait<T = f64>,
{
    let transformed = affine.transform_point3(glam::DVec3::new(
        coord.x(),
        coord.y(),
        coord.nth(2).unwrap(),
    ));

    match dim {
        geo_traits::Dimensions::Xyz => {
            write_wkb_coord(writer, (transformed.x, transformed.y, transformed.z))
                .map_err(|e| DataFusionError::Execution(e.to_string()))
        }
        geo_traits::Dimensions::Xyzm => {
            // Preserve m value
            let m = coord.nth(3).unwrap();
            write_wkb_coord(writer, (transformed.x, transformed.y, transformed.z, m))
                .map_err(|e| DataFusionError::Execution(e.to_string()))
        }
        _ => {
            return internal_err!("2D dimension is passed to 3D affine transformation.");
        }
    }
}

struct DAffine2Iterator<'a> {
    index: usize,
    a: &'a PrimitiveArray<Float64Type>,
    b: &'a PrimitiveArray<Float64Type>,
    d: &'a PrimitiveArray<Float64Type>,
    e: &'a PrimitiveArray<Float64Type>,
    x_offset: &'a PrimitiveArray<Float64Type>,
    y_offset: &'a PrimitiveArray<Float64Type>,
}

impl<'a> DAffine2Iterator<'a> {
    fn new(array_args: &'a [Arc<dyn Array>]) -> Result<Self> {
        if array_args.len() != 6 {
            return internal_err!("Invalid number of arguments are passed");
        }

        let a = as_float64_array(&array_args[0])?;
        let b = as_float64_array(&array_args[1])?;
        let d = as_float64_array(&array_args[2])?;
        let e = as_float64_array(&array_args[3])?;
        let x_offset = as_float64_array(&array_args[4])?;
        let y_offset = as_float64_array(&array_args[5])?;

        Ok(Self {
            index: 0,
            a,
            b,
            d,
            e,
            x_offset,
            y_offset,
        })
    }
}

impl<'a> Iterator for DAffine2Iterator<'a> {
    type Item = glam::DAffine2;

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.index;
        self.index += 1;
        Some(glam::DAffine2 {
            matrix2: glam::DMat2 {
                x_axis: glam::DVec2 {
                    x: self.a.value(i),
                    y: self.b.value(i),
                },
                y_axis: glam::DVec2 {
                    x: self.d.value(i),
                    y: self.e.value(i),
                },
            },
            translation: glam::DVec2 {
                x: self.x_offset.value(i),
                y: self.y_offset.value(i),
            },
        })
    }
}

struct DAffine3Iterator<'a> {
    index: usize,
    a: &'a PrimitiveArray<Float64Type>,
    b: &'a PrimitiveArray<Float64Type>,
    c: &'a PrimitiveArray<Float64Type>,
    d: &'a PrimitiveArray<Float64Type>,
    e: &'a PrimitiveArray<Float64Type>,
    f: &'a PrimitiveArray<Float64Type>,
    g: &'a PrimitiveArray<Float64Type>,
    h: &'a PrimitiveArray<Float64Type>,
    i: &'a PrimitiveArray<Float64Type>,
    x_offset: &'a PrimitiveArray<Float64Type>,
    y_offset: &'a PrimitiveArray<Float64Type>,
    z_offset: &'a PrimitiveArray<Float64Type>,
}

impl<'a> DAffine3Iterator<'a> {
    fn new(array_args: &'a [Arc<dyn Array>]) -> Result<Self> {
        if array_args.len() != 12 {
            return internal_err!("Invalid number of arguments are passed");
        }

        let a = as_float64_array(&array_args[0])?;
        let b = as_float64_array(&array_args[1])?;
        let c = as_float64_array(&array_args[2])?;
        let d = as_float64_array(&array_args[3])?;
        let e = as_float64_array(&array_args[4])?;
        let f = as_float64_array(&array_args[5])?;
        let g = as_float64_array(&array_args[6])?;
        let h = as_float64_array(&array_args[7])?;
        let i = as_float64_array(&array_args[8])?;
        let x_offset = as_float64_array(&array_args[9])?;
        let y_offset = as_float64_array(&array_args[10])?;
        let z_offset = as_float64_array(&array_args[11])?;

        Ok(Self {
            index: 0,
            a,
            b,
            c,
            d,
            e,
            f,
            g,
            h,
            i,
            x_offset,
            y_offset,
            z_offset,
        })
    }
}

impl<'a> Iterator for DAffine3Iterator<'a> {
    type Item = glam::DAffine3;

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.index;
        self.index += 1;
        Some(glam::DAffine3 {
            matrix3: glam::DMat3 {
                x_axis: glam::DVec3 {
                    x: self.a.value(i),
                    y: self.b.value(i),
                    z: self.c.value(i),
                },
                y_axis: glam::DVec3 {
                    x: self.d.value(i),
                    y: self.e.value(i),
                    z: self.f.value(i),
                },
                z_axis: glam::DVec3 {
                    x: self.g.value(i),
                    y: self.h.value(i),
                    z: self.i.value(i),
                },
            },
            translation: glam::DVec3 {
                x: self.x_offset.value(i),
                y: self.y_offset.value(i),
                z: self.z_offset.value(i),
            },
        })
    }
}
