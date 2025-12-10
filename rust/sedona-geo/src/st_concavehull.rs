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

#![allow(dead_code)]
use std::sync::Arc;

use arrow_array::builder::BinaryBuilder;
use datafusion_common::error::Result;
use datafusion_common::ScalarValue;
use datafusion_common::{
    cast::{as_boolean_array, as_float64_array},
    DataFusionError,
};
use datafusion_expr::ColumnarValue;
use geo::{ConcaveHull, CoordsIter};
use geo_traits::{GeometryCollectionTrait, GeometryTrait, PointTrait};
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_geometry::wkb_factory::{
    write_wkb_coord_trait, write_wkb_empty_point, write_wkb_point_header, WKB_MIN_PROBABLE_BYTES,
};
use sedona_schema::datatypes::SedonaType;
use sedona_schema::{datatypes::WKB_GEOMETRY, matchers::ArgMatcher};
use wkb::writer::WriteOptions;

use crate::to_geo::GeoTypesExecutor;

/// ST_ConcaveHull implementation using [ConcaveHull]
pub fn st_concavehull_impl() -> ScalarKernelRef {
    Arc::new(STConcaveHull {})
}

#[derive(Debug)]
struct STConcaveHull {}

impl SedonaScalarKernel for STConcaveHull {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry(), ArgMatcher::is_numeric()],
            WKB_GEOMETRY,
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        invoke_batch_impl(arg_types, args)
    }
}

fn invoke_batch_impl(arg_types: &[SedonaType], args: &[ColumnarValue]) -> Result<ColumnarValue> {
    let executor = GeoTypesExecutor::new(arg_types, args);
    let mut builder = BinaryBuilder::with_capacity(
        executor.num_iterations(),
        WKB_MIN_PROBABLE_BYTES * executor.num_iterations(),
    );

    // Extract Args
    let pct_convex_val = args[1]
        .cast_to(&arrow_schema::DataType::Float64, None)?
        .to_array(executor.num_iterations())?;
    let pct_convex_array = as_float64_array(&pct_convex_val)?;
    let mut pct_convex_iter = pct_convex_array.iter();

    let allow_holes_val = args
        .get(2)
        .unwrap_or(&ColumnarValue::Scalar(ScalarValue::Boolean(Some(false))))
        .cast_to(&arrow_schema::DataType::Boolean, None)?
        .to_array(executor.num_iterations())?;
    let allow_holes_array = as_boolean_array(&allow_holes_val)?;
    let mut allow_holes_iter = allow_holes_array.iter();

    executor.execute_wkb_void(|maybe_wkb| {
        match (
            maybe_wkb,
            pct_convex_iter.next().unwrap(),
            allow_holes_iter.next().unwrap(),
        ) {
            (Some(wkb), Some(pct_convex), Some(allow_holes)) => {
                invoke_scalar(&wkb, pct_convex, allow_holes, &mut builder)?;
                builder.append_value([]);
            }
            _ => builder.append_null(),
        }

        Ok(())
    })?;

    executor.finish(Arc::new(builder.finish()))
}

fn invoke_scalar(
    geom: &geo_types::Geometry<f64>,
    pct_convex: f64,
    _allow_holes: bool,
    writer: &mut impl std::io::Write,
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

        geo_traits::GeometryType::MultiPoint(mpt) => {
            let hull = mpt.concave_hull(pct_convex);
            write_concave_hull(writer, hull)?;
        }

        geo_traits::GeometryType::LineString(ls) => {
            let hull = ls.concave_hull(pct_convex);
            write_concave_hull(writer, hull)?;
        }

        geo_traits::GeometryType::Polygon(pgn) => {
            let hull = pgn.concave_hull(pct_convex);
            write_concave_hull(writer, hull)?;
        }

        geo_traits::GeometryType::MultiLineString(mls) => {
            let hull = mls.concave_hull(pct_convex);
            write_concave_hull(writer, hull)?;
        }

        geo_traits::GeometryType::MultiPolygon(mpgn) => {
            let hull = mpgn.concave_hull(pct_convex);
            write_concave_hull(writer, hull)?;
        }

        geo_traits::GeometryType::GeometryCollection(gcn) => {
            let coords: Vec<geo_types::Coord> = gcn
                .geometries()
                .flat_map(|geom| geom.coords_iter())
                .collect();

            let multi_point = geo_types::MultiPoint::new(
                coords.into_iter().map(geo_types::Point::from).collect(),
            );

            let hull = multi_point.concave_hull(pct_convex);
            write_concave_hull(writer, hull)?;
        }

        _ => {
            return Err(DataFusionError::Execution(
                "Unsupported geometry type for concave hull".to_string(),
            ))
        }
    }

    Ok(())
}

fn write_concave_hull<W: std::io::Write>(
    writer: &mut W,
    hull: impl GeometryTrait<T = f64>,
) -> Result<()> {
    wkb::writer::write_geometry(
        writer,
        &hull,
        &WriteOptions {
            endianness: wkb::Endianness::LittleEndian,
        },
    )
    .map_err(|e| DataFusionError::Execution(e.to_string()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use arrow_schema::DataType;
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::SedonaType;
    use sedona_schema::datatypes::WKB_VIEW_GEOMETRY;
    use sedona_testing::compare::assert_scalar_equal_wkb_geometry_topologically;
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_kernel("st_concavehull", st_concavehull_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Float64)],
        );
        assert_eq!(tester.return_type().unwrap(), WKB_GEOMETRY);

        let result = tester
            .invoke_scalar_scalar("LINESTRING (50 50, 150 150, 150 50)", 0.1)
            .unwrap();
        assert_scalar_equal_wkb_geometry_topologically(
            &result,
            Some("POLYGON ((50 50, 150 150, 150 50, 50 50))"),
        );

        let result = tester
            .invoke_scalar_scalar("LINESTRING (100 150, 50 60, 70 80, 160 170)", 0.3)
            .unwrap();
        assert_scalar_equal_wkb_geometry_topologically(
            &result,
            Some("POLYGON ((70 80, 50 60, 100 150, 160 170, 70 80))"),
        );

        let result = tester
            .invoke_scalar_scalar("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))", 0.1)
            .unwrap();
        assert_scalar_equal_wkb_geometry_topologically(
            &result,
            Some("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"),
        );

        let result = tester
            .invoke_scalar_scalar("MULTIPOINT ((0 0), (10 0), (0 10), (10 10), (5 5))", 1.0)
            .unwrap();
        assert_scalar_equal_wkb_geometry_topologically(
            &result,
            Some("POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))"),
        );

        let result = tester
            .invoke_scalar_scalar(
                "MULTIPOLYGON (((2 2, 2 5, 5 5, 5 2, 2 2)), ((6 3, 8 3, 8 1, 6 1, 6 3)))",
                0.1,
            )
            .unwrap();
        assert_scalar_equal_wkb_geometry_topologically(
            &result,
            // PostGis returns `MULTIPOLYGON(((5 2,6 3,8 3,8 1,6 1,5 2)),((2 5,5 5,5 2,2 2,2 5)))`
            // but `geo` concavehull method returns a polygon for everything
            Some("POLYGON ((6 1, 8 1, 8 3, 6 3, 6 3, 5 5, 2 5, 2 2, 5 2, 6 1))"),
        );

        let result = tester
            .invoke_scalar_scalar(
                "GEOMETRYCOLLECTION (MULTIPOINT((1 1), (3 3)), POINT(5 6), LINESTRING(4 5, 5 6))",
                0.1,
            )
            .unwrap();
        assert_scalar_equal_wkb_geometry_topologically(
            &result,
            Some("POLYGON ((3 3, 1 1, 4 5, 5 6, 3 3))"),
        );
    }
}
