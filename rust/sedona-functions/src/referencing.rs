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
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};

/// ST_LineLocatePoint() scalar UDF implementation
pub fn st_line_locate_point_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new_stub(
        "st_linelocatepoint",
        ArgMatcher::new(
            vec![ArgMatcher::is_geometry(), ArgMatcher::is_geometry()],
            SedonaType::Arrow(DataType::Float64),
        ),
        Volatility::Immutable,
        None,
    )
}

/// ST_LineInterpolatePoint() scalar UDF implementation
pub fn st_line_interpolate_point_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new_stub(
        "st_lineinterpolatepoint",
        ArgMatcher::new(
            vec![ArgMatcher::is_geometry(), ArgMatcher::is_numeric()],
            WKB_GEOMETRY,
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
        let udf: ScalarUDF = st_line_interpolate_point_udf().into();
        assert_eq!(udf.name(), "st_lineinterpolatepoint");

        let udf: ScalarUDF = st_line_locate_point_udf().into();
        assert_eq!(udf.name(), "st_linelocatepoint");
    }
}
