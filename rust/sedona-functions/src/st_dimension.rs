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
use std::sync::Arc;

use crate::executor::WkbExecutor;
use arrow_array::builder::Int8Builder;
use arrow_schema::DataType;
use datafusion_common::error::Result;
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use geo_traits::{GeometryCollectionTrait, GeometryTrait, GeometryType};
use sedona_common::sedona_internal_err;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};
use wkb::reader::Wkb;

pub fn st_dimension_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_dimension",
        vec![Arc::new(STDimension {})],
        Volatility::Immutable,
        Some(st_dimension_doc()),
    )
}

fn st_dimension_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Return the dimension of the geometry",
        "ST_Dimension (A: Geometry)",
    )
    .with_argument("geom", "geometry: Input geometry")
    .with_sql_example("SELECT ST_Dimension(ST_GeomFromWKT('POLYGON EMPTY'))")
    .build()
}

#[derive(Debug)]
struct STDimension {}

impl SedonaScalarKernel for STDimension {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry()],
            SedonaType::Arrow(DataType::Int8),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);
        let mut builder = Int8Builder::with_capacity(executor.num_iterations());

        executor.execute_wkb_void(|maybe_item| {
            match maybe_item {
                Some(item) => {
                    builder.append_value(invoke_scalar(&item)?);
                }
                None => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(item: &Wkb) -> Result<i8> {
    match item.as_type() {
        GeometryType::Point(_) | GeometryType::MultiPoint(_) => Ok(0),
        GeometryType::LineString(_) | GeometryType::MultiLineString(_) => Ok(1),
        GeometryType::Polygon(_) | GeometryType::MultiPolygon(_) => Ok(2),
        GeometryType::GeometryCollection(collection) => {
            let mut highest_dim = 0;
            for geom in collection.geometries() {
                highest_dim = highest_dim.max(invoke_scalar(geom)?);
            }
            Ok(highest_dim)
        }
        _ => sedona_internal_err!("Invalid geometry type"),
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{create_array as arrow_array, ArrayRef};
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::{compare::assert_array_equal, testers::ScalarUdfTester};

    use super::*;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = st_dimension_udf().into();
        assert_eq!(udf.name(), "st_dimension");
        assert!(udf.documentation().is_some());
    }

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        use datafusion_common::ScalarValue;

        let tester = ScalarUdfTester::new(st_dimension_udf().into(), vec![sedona_type.clone()]);

        tester.assert_return_type(DataType::Int8);

        let result = tester.invoke_wkb_scalar(Some("POINT (1 2)")).unwrap();
        tester.assert_scalar_result_equals(result, ScalarValue::Int8(Some(0)));

        let result = tester.invoke_wkb_scalar(None).unwrap();
        tester.assert_scalar_result_equals(result, ScalarValue::Null);

        let input_wkt = vec![
            None,
            Some("POINT EMPTY"),
            Some("MULTIPOINT EMPTY"),
            Some("LINESTRING EMPTY"),
            Some("MULTILINESTRING EMPTY"),
            Some("POLYGON EMPTY"),
            Some("MULTIPOLYGON EMPTY"),
            Some("POINT (1 2)"),
            Some("MULTIPOINT ((0 0))"),
            Some("LINESTRING (1 2, 2 2)"),
            Some("MULTILINESTRING ((0 0, 1 0), (1 1, 0 1))"),
            Some("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"),
            Some("MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)))"),
            Some("GEOMETRYCOLLECTION EMPTY"),
            Some("GEOMETRYCOLLECTION (POINT (1 2))"),
            Some("GEOMETRYCOLLECTION (POINT (1 2), LINESTRING EMPTY)"),
            Some("GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (1 2, 2 2), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)))"),
            Some("GEOMETRYCOLLECTION (POINT (1 2), GEOMETRYCOLLECTION (LINESTRING (1 2, 2 2)))"),
        ];
        let expected: ArrayRef = arrow_array!(
            Int8,
            [
                None,
                Some(0),
                Some(0),
                Some(1),
                Some(1),
                Some(2),
                Some(2),
                Some(0),
                Some(0),
                Some(1),
                Some(1),
                Some(2),
                Some(2),
                Some(0),
                Some(0),
                Some(1),
                Some(2),
                Some(1)
            ]
        );
        assert_array_equal(&tester.invoke_wkb_array(input_wkt).unwrap(), &expected);
    }
}
