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
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_geometry::{
    error::SedonaGeometryError,
    transform::{transform, CrsTransform},
    wkb_factory::WKB_MIN_PROBABLE_BYTES,
};

use sedona_schema::datatypes::WKB_GEOGRAPHY;
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};
use wkb::reader::Wkb;

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
        let geom_matcher = ArgMatcher::new(vec![ArgMatcher::is_geometry()], WKB_GEOMETRY);
        let geog_matcher = ArgMatcher::new(vec![ArgMatcher::is_geography()], WKB_GEOGRAPHY);
        let matched_geom = geom_matcher.match_args(args)?;
        let matched_geog = geog_matcher.match_args(args)?;

        match (matched_geom, matched_geog) {
            (Some(geom_result), _) => Ok(Some(geom_result)),
            (_, Some(geog_result)) => Ok(Some(geog_result)),
            _ => Ok(None),
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

        let mut transform = SwapXy {};

        executor.execute_wkb_void(|maybe_item| {
            match maybe_item {
                Some(item) => {
                    invoke_scalar(&item, &mut transform, &mut builder)?;
                    builder.append_value([]);
                }
                None => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(
    wkb: &Wkb,
    swap_transform: &mut SwapXy,
    writer: &mut impl std::io::Write,
) -> Result<(), DataFusionError> {
    transform(wkb, swap_transform, writer).map_err(|e| DataFusionError::External(e.into()))?;
    Ok(())
}

#[derive(Debug)]
struct SwapXy {}
impl CrsTransform for SwapXy {
    fn transform_coord(
        &self,
        coord: &mut (f64, f64),
    ) -> std::result::Result<(), SedonaGeometryError> {
        let (x, y) = *coord;
        *coord = (y, x);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::crs::lnglat;
    use sedona_schema::datatypes::SedonaType::Wkb;
    use sedona_schema::datatypes::{Edges, WKB_VIEW_GEOMETRY};
    use sedona_testing::{
        compare::assert_array_equal, create::create_array, testers::ScalarUdfTester,
    };

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = st_flipcoordinates_udf().into();
        assert_eq!(udf.name(), "st_flipcoordinates");
        assert!(udf.documentation().is_some());
    }

    #[test]
    fn udf_return_type() {
        let tester = ScalarUdfTester::new(st_flipcoordinates_udf().into(), vec![WKB_GEOGRAPHY]);
        tester.assert_return_type(WKB_GEOGRAPHY);

        let tester = ScalarUdfTester::new(st_flipcoordinates_udf().into(), vec![WKB_GEOMETRY]);
        tester.assert_return_type(WKB_GEOMETRY);

        let tester = ScalarUdfTester::new(st_flipcoordinates_udf().into(), vec![WKB_VIEW_GEOMETRY]);
        tester.assert_return_type(WKB_GEOMETRY);

        let tester = ScalarUdfTester::new(
            st_flipcoordinates_udf().into(),
            vec![Wkb(Edges::Planar, lnglat())],
        );
        tester.assert_return_type(Wkb(Edges::Planar, lnglat()));
    }

    #[rstest]
    fn udf_invoke(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY, WKB_GEOGRAPHY)] sedona_type: SedonaType,
    ) {
        let tester =
            ScalarUdfTester::new(st_flipcoordinates_udf().into(), vec![sedona_type.clone()]);

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
            Some("MULTIPOINT ZM(1 2 3 4, 5 6 7 8)"),
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
                Some("MULTIPOINT ZM(2 1 3 4, 6 5 7 8)"),
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
