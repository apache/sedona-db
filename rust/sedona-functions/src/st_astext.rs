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
use arrow_array::builder::StringBuilder;
use arrow_schema::DataType;
use datafusion_common::error::{DataFusionError, Result};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

/// ST_AsText() scalar UDF implementation
///
/// An implementation of WKT writing using GeoRust's wkt crate.
pub fn st_astext_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new_with_aliases(
        "st_astext",
        vec![Arc::new(STAsText {})],
        Volatility::Immutable,
        Some(st_astext_doc()),
        vec!["st_aswkt".to_string()],
    )
}

fn st_astext_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Return the Well-Known Text string representation of a geometry or geography",
        "ST_AsText (A: Geometry)",
    )
    .with_argument("geom", "geometry: Input geometry or geography")
    .with_sql_example("SELECT ST_AsText(ST_Point(1.0, 2.0))")
    .with_related_udf("ST_GeomFromWKT")
    .build()
}

#[derive(Debug)]
struct STAsText {}

impl SedonaScalarKernel for STAsText {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry_or_geography()],
            SedonaType::Arrow(DataType::Utf8),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);

        // Estimate the minimum probable memory requirement of the output.
        // Here, the shortest full precision non-empty/non-null value would be
        // POINT (<16 digits> <16 digits>), or ~25 bytes.
        let min_probable_wkt_size = executor.num_iterations() * 25;

        // Initialize an output builder of the appropriate type
        let mut builder =
            StringBuilder::with_capacity(executor.num_iterations(), min_probable_wkt_size);

        executor.execute_wkb_void(|maybe_item| {
            match maybe_item {
                Some(item) => {
                    wkt::to_wkt::write_geometry(&mut builder, &item)
                        .map_err(|err| DataFusionError::External(Box::new(err)))?;
                    builder.append_value("");
                }
                None => builder.append_null(),
            };

            Ok(())
        })?;

        // Create the output array
        executor.finish(Arc::new(builder.finish()))
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{create_array, ArrayRef};
    use datafusion_common::scalar::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::{
        WKB_GEOGRAPHY, WKB_GEOMETRY, WKB_VIEW_GEOGRAPHY, WKB_VIEW_GEOMETRY,
    };
    use sedona_testing::{
        compare::{assert_array_equal, assert_scalar_equal},
        testers::ScalarUdfTester,
    };

    use super::*;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = st_astext_udf().into();
        assert_eq!(udf.name(), "st_astext");
        assert!(udf.documentation().is_some())
    }

    #[rstest]
    fn udf(
        #[values(WKB_GEOMETRY, WKB_GEOGRAPHY, WKB_VIEW_GEOMETRY, WKB_VIEW_GEOGRAPHY)]
        sedona_type: SedonaType,
    ) {
        let udf = st_astext_udf();
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type]);

        assert_scalar_equal(
            &tester.invoke_wkb_scalar(Some("POINT (1 2)")).unwrap(),
            &ScalarValue::Utf8(Some("POINT(1 2)".to_string())),
        );

        assert_scalar_equal(
            &tester.invoke_wkb_scalar(None).unwrap(),
            &ScalarValue::Utf8(None),
        );

        let expected_array: ArrayRef =
            create_array!(Utf8, [Some("POINT(1 2)"), None, Some("POINT(3 5)")]);
        assert_array_equal(
            &tester
                .invoke_wkb_array(vec![Some("POINT(1 2)"), None, Some("POINT(3 5)")])
                .unwrap(),
            &expected_array,
        );
    }

    #[test]
    fn aliases() {
        let udf: ScalarUDF = st_astext_udf().into();
        assert!(udf.aliases().contains(&"st_aswkt".to_string()));
    }
}
