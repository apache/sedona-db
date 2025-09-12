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

use crate::executor::WkbExecutor;
use arrow_array::builder::BinaryBuilder;
use datafusion_common::error::{DataFusionError, Result};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use geo_traits::{
    CoordTrait, Dimensions, GeometryCollectionTrait, GeometryTrait, GeometryType, LineStringTrait,
    MultiLineStringTrait, MultiPointTrait, MultiPolygonTrait, PointTrait, PolygonTrait,
};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
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
use wkb::reader::Wkb;

use sedona_geometry::error::SedonaGeometryError;

/// ST_FlipCoordinates() scalar UDF implementation
///
/// An implementation of flip coordinates
pub fn st_flipcoordinates_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_flipcoordinates",
        vec![Arc::new(STFlipCoordinates {})],
        Volatility::Immutable,
        Some(st_flipcoordinates_doc()),
    )
}

fn st_flipcoordinates_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns a version of the given geometry with X and Y axis flipped.",
        "ST_FlipCoordinates(A:geometry)",
    )
    .with_argument("geom", "geometry: Input geometry")
    .with_sql_example("SELECT ST_FlipCoordinates(df.geometry)")
    .build()
}

#[derive(Debug)]
struct STFlipCoordinates {}

impl SedonaScalarKernel for STFlipCoordinates {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(vec![ArgMatcher::is_geometry()], WKB_GEOMETRY);

        if matcher.match_args(args).is_ok() {
            // keep the CRS in the output type if present
            Ok(Some(args[0].clone()))
        } else {
            Ok(None)
        }
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

        executor.execute_wkb_void(|maybe_item| {
            match maybe_item {
                Some(item) => {
                    invoke_scalar(&item, &mut builder)?;
                    builder.append_value([]);
                }
                None => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(wkb: &Wkb, writer: &mut impl std::io::Write) -> Result<()> {
    swap_yx(wkb, writer).map_err(|e| DataFusionError::External(e.into()))
}

fn swap_yx(
    geom: impl GeometryTrait<T = f64>,
    writer: &mut impl std::io::Write,
) -> Result<(), SedonaGeometryError> {
    let dims = geom.dim();

    match geom.as_type() {
        GeometryType::Point(pt) => {
            if pt.coord().is_some() {
                write_wkb_point_header(writer, dims)?;
                swap(pt.coord().unwrap(), dims, writer)?;
            } else {
                write_wkb_empty_point(writer, dims)?;
            }
        }
        GeometryType::LineString(ls) => {
            write_wkb_linestring_header(writer, ls.dim(), ls.coords().count())?;
            for coord in ls.coords() {
                swap(coord, dims, writer)?;
            }
        }
        // Similar pattern for other geometry types...
        GeometryType::Polygon(pl) => {
            let num_rings = pl.interiors().count() + pl.exterior().is_some() as usize;
            write_wkb_polygon_header(writer, pl.dim(), num_rings)?;

            if let Some(exterior) = pl.exterior() {
                write_wkb_polygon_ring_header(writer, exterior.coords().count())?;
                for coord in exterior.coords() {
                    swap(coord, dims, writer)?;
                }
            }

            for interior in pl.interiors() {
                write_wkb_polygon_ring_header(writer, interior.coords().count())?;
                for coord in interior.coords() {
                    swap(coord, dims, writer)?;
                }
            }
        }
        GeometryType::MultiPoint(multi_pt) => {
            write_wkb_multipoint_header(writer, dims, multi_pt.points().count())?;
            for pt in multi_pt.points() {
                swap_yx(pt, writer)?;
            }
        }
        GeometryType::MultiLineString(multi_ls) => {
            write_wkb_multilinestring_header(writer, dims, multi_ls.line_strings().count())?;
            for ls in multi_ls.line_strings() {
                swap_yx(ls, writer)?;
            }
        }
        GeometryType::MultiPolygon(multi_pl) => {
            write_wkb_multipolygon_header(writer, dims, multi_pl.polygons().count())?;
            for pl in multi_pl.polygons() {
                swap_yx(pl, writer)?;
            }
        }
        GeometryType::GeometryCollection(collection) => {
            write_wkb_geometrycollection_header(writer, dims, collection.geometries().count())?;
            for geom in collection.geometries() {
                swap_yx(geom, writer)?;
            }
        }
        _ => {
            return Err(SedonaGeometryError::Invalid(
                "GeometryType not supported for transform".to_string(),
            ))
        }
    };

    Ok(())
}

fn swap<C: CoordTrait<T = f64>>(
    coord: C,
    dims: Dimensions,
    writer: &mut impl std::io::Write,
) -> Result<(), SedonaGeometryError> {
    match dims {
        Dimensions::Xy => write_wkb_coord(writer, (coord.y(), coord.x())),
        Dimensions::Xyz | Dimensions::Xym => {
            write_wkb_coord(writer, (coord.y(), coord.x(), coord.nth_or_panic(2)))
        }
        Dimensions::Xyzm => write_wkb_coord(
            writer,
            (
                coord.y(),
                coord.x(),
                coord.nth_or_panic(2),
                coord.nth_or_panic(3),
            ),
        ),
        _ => Err(SedonaGeometryError::Invalid(
            "Unsupported geometry dimension".to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::{
        compare::assert_array_equal, create::create_array, testers::ScalarUdfTester,
    };

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = st_flipcoordinates_udf().into();
        assert_eq!(udf.name(), "st_flipcoordinates");
        assert!(udf.documentation().is_some());
    }

    #[rstest]
    fn udf_invoke(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester =
            ScalarUdfTester::new(st_flipcoordinates_udf().into(), vec![sedona_type.clone()]);
        tester.assert_return_type(sedona_type.clone());

        let result = tester.invoke_scalar("POINT (1 3)").unwrap();
        tester.assert_scalar_result_equals(result, "POINT (3 1)");

        let input_wkt = vec![
            None,
            Some("POINT (1 2)"),
            Some("POINT Z(1 2 3)"),
            Some("LINESTRING (10 0, 1 3)"),
            Some("LINESTRING M(10 0 5, 1 3 6)"),
            Some("POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0), (0 1, 1 1, 1 0, 0 1))"),
            Some("GEOMETRYCOLLECTION (POINT (7 5), LINESTRING (-1 -3, 1 2))"),
            Some("MULTIPONT ZM((1 2 3 4), (5 6 7 8))"),
            Some("MULTILINESTRING ((0 0, 1 3), (10 0, 1 3))"),
            Some("POINT EMPTY"),
            Some("LINESTRING EMPTY"),
            Some("POLYGON EMPTY"),
            Some("MULTIPOINT EMPTY"),
            Some("MULTILINESTRING EMPTY"),
            Some("MULTIPOLYGON EMPTY"),
            Some("GEOMETRYCOLLECTION EMPTY"),
        ];
        let expected = create_array(
            &[
                None,
                Some("POINT (2 1)"),
                Some("POINT Z(2 1 3)"),
                Some("LINESTRING (0 10, 3 1)"),
                Some("LINESTRING M(0 10 5, 3 1 6)"),
                Some("POLYGON ((0 0, 0 2, 2 2, 2 0, 0 0), (1 0, 1 1, 0 1, 1 0))"),
                Some("GEOMETRYCOLLECTION (POINT (5 7), LINESTRING (-3 -1, 2 1))"),
                Some("MULTIPONT ZM((2 1 3 4), (6 5 7 8))"),
                Some("MULTILINESTRING ((0 0, 3 1), (0 10, 3 1))"),
                Some("POINT EMPTY"),
                Some("LINESTRING EMPTY"),
                Some("POLYGON EMPTY"),
                Some("MULTIPOINT EMPTY"),
                Some("MULTILINESTRING EMPTY"),
                Some("MULTIPOLYGON EMPTY"),
                Some("GEOMETRYCOLLECTION EMPTY"),
            ],
            &WKB_GEOMETRY,
        );
        assert_array_equal(&tester.invoke_wkb_array(input_wkt).unwrap(), &expected);
    }
}
