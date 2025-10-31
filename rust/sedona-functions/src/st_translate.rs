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
use arrow_schema::DataType;
use datafusion_common::{cast::as_float64_array, error::Result, DataFusionError};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};

use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_geometry::{
    error::SedonaGeometryError,
    transform::{transform, CrsTransform},
    wkb_factory::WKB_MIN_PROBABLE_BYTES,
};
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};
use std::{iter::zip, sync::Arc};

use crate::executor::WkbExecutor;

/// ST_Translate() scalar UDF
pub fn st_translate_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_translate",
        vec![Arc::new(STTranslate)],
        Volatility::Immutable,
        Some(st_translate_doc()),
    )
}

fn st_translate_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Update coordinates of geom by a fixed offset",
        "ST_Translate (geom: Geometry, deltax: numeric, deltay: numeric)",
    )
    .with_argument("geom", "geometry: Input geometry")
    .with_argument("deltax", "numeric: X value difference")
    .with_argument("deltay", "numeric: Y value difference")
    .with_sql_example("SELECT ST_Translate(ST_GeomFromWKT('LINESTRING(0 1, 2 3, 4 5)'), 2.0, 3.0)")
    .build()
}

#[derive(Debug)]
struct STTranslate;

impl SedonaScalarKernel for STTranslate {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![
                ArgMatcher::is_geometry(),
                ArgMatcher::is_numeric(),
                ArgMatcher::is_numeric(),
            ],
            WKB_GEOMETRY,
        );

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

        let deltax = args[1]
            .cast_to(&DataType::Float64, None)?
            .to_array(executor.num_iterations())?;
        let deltay = args[2]
            .cast_to(&DataType::Float64, None)?
            .to_array(executor.num_iterations())?;
        let deltax_array = as_float64_array(&deltax)?;
        let deltay_array = as_float64_array(&deltay)?;
        let mut delta_iter = zip(deltax_array, deltay_array);

        executor.execute_wkb_void(|maybe_wkb| {
            let (deltax, deltay) = delta_iter.next().unwrap();
            match (maybe_wkb, deltax, deltay) {
                (Some(wkb), Some(deltax), Some(deltay)) => {
                    let trans = Translate { deltax, deltay };
                    transform(wkb, &trans, &mut builder)
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;
                    builder.append_value([]);
                }
                _ => {
                    builder.append_null();
                }
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

#[derive(Debug)]
struct Translate {
    deltax: f64,
    deltay: f64,
}

impl CrsTransform for Translate {
    fn transform_coord(&self, coord: &mut (f64, f64)) -> Result<(), SedonaGeometryError> {
        coord.0 += self.deltax;
        coord.1 += self.deltay;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::create_array;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::WKB_VIEW_GEOMETRY;
    use sedona_testing::{
        compare::assert_array_equal, create::create_array, testers::ScalarUdfTester,
    };

    use super::*;

    #[test]
    fn udf_metadata() {
        let st_translate_udf: ScalarUDF = st_translate_udf().into();
        assert_eq!(st_translate_udf.name(), "st_translate");
        assert!(st_translate_udf.documentation().is_some());
    }

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester = ScalarUdfTester::new(
            st_translate_udf().into(),
            vec![
                sedona_type.clone(),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
            ],
        );
        tester.assert_return_type(WKB_GEOMETRY);

        let points = create_array(
            &[
                None,
                Some("POINT EMPTY"),
                Some("POINT EMPTY"),
                Some("POINT EMPTY"),
                Some("POINT Z EMPTY"),
                Some("POINT (0 1)"),
                Some("POINT (2 3)"),
                Some("POINT Z (4 5 6)"),
            ],
            &sedona_type,
        );

        let dx = create_array!(
            Float64,
            [
                Some(1.0),
                None,
                Some(1.0),
                Some(1.0),
                Some(1.0),
                Some(1.0),
                Some(1.0),
                Some(1.0)
            ]
        );
        let dy = create_array!(
            Float64,
            [
                Some(2.0),
                Some(2.0),
                None,
                Some(2.0),
                Some(2.0),
                Some(2.0),
                Some(2.0),
                Some(2.0)
            ]
        );

        let expected = create_array(
            &[
                None,
                None,
                None,
                Some("POINT EMPTY"),
                Some("POINT Z EMPTY"),
                Some("POINT (1 3)"),
                Some("POINT (3 5)"),
                Some("POINT Z (5 7 6)"),
            ],
            &WKB_GEOMETRY,
        );

        let result = tester.invoke_arrays(vec![points, dx, dy]).unwrap();
        assert_array_equal(&result, &expected);
    }
}
