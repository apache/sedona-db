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
use arrow_schema::DataType;
use datafusion_expr::Volatility;
use sedona_expr::scalar_udf::SedonaScalarUDF;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

/// ST_Equals() scalar UDF stub
pub fn st_equals_udf() -> SedonaScalarUDF {
    predicate_stub_udf("ST_Equals")
}

/// ST_Intersects() scalar UDF stub
pub fn st_intersects_udf() -> SedonaScalarUDF {
    predicate_stub_udf("ST_Intersects")
}

/// ST_Disjoint() scalar UDF stub
pub fn st_disjoint_udf() -> SedonaScalarUDF {
    predicate_stub_udf("ST_Disjoint")
}

/// ST_Contains() scalar UDF stub
pub fn st_contains_udf() -> SedonaScalarUDF {
    predicate_stub_udf("ST_Contains")
}

/// ST_Within() scalar UDF stub
pub fn st_within_udf() -> SedonaScalarUDF {
    predicate_stub_udf("ST_Within")
}

/// ST_Covers() scalar UDF stub
pub fn st_covers_udf() -> SedonaScalarUDF {
    predicate_stub_udf("ST_Covers")
}

/// ST_CoveredBy() scalar UDF stub
pub fn st_covered_by_udf() -> SedonaScalarUDF {
    predicate_stub_udf("ST_CoveredBy")
}

/// ST_Touches() scalar UDF stub
pub fn st_touches_udf() -> SedonaScalarUDF {
    predicate_stub_udf("ST_Touches")
}

/// ST_KNN() scalar UDF stub
///
/// This is a stub function that defines the signature and documentation for ST_KNN
/// but does not contain an actual implementation. The real k-nearest neighbors logic
/// is handled by the spatial join execution engine.
pub fn st_knn_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new_stub(
        "st_knn",
        ArgMatcher::new(
            vec![
                ArgMatcher::is_geometry_or_geography(),
                ArgMatcher::is_geometry_or_geography(),
                ArgMatcher::is_numeric(),
                ArgMatcher::is_boolean(),
            ],
            SedonaType::Arrow(DataType::Boolean),
        ),
        Volatility::Immutable,
        None,
    )
}

pub fn predicate_stub_udf(name: &str) -> SedonaScalarUDF {
    SedonaScalarUDF::new_stub(
        &name.to_lowercase(),
        ArgMatcher::new(
            vec![
                ArgMatcher::is_geometry_or_geography(),
                ArgMatcher::is_geometry_or_geography(),
            ],
            SedonaType::Arrow(DataType::Boolean),
        ),
        Volatility::Immutable,
        None,
    )
}

#[cfg(test)]
mod tests {
    use datafusion_expr::ScalarUDF;

    use super::*;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = st_intersects_udf().into();
        assert_eq!(udf.name(), "st_intersects");
    }
}
