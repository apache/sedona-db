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
use std::vec;

use datafusion_expr::Volatility;
use sedona_expr::scalar_udf::SedonaScalarUDF;
use sedona_schema::{datatypes::WKB_GEOMETRY, matchers::ArgMatcher};

/// St_Transform() UDF implementation
///
/// An implementation of intersection calculation.
pub fn st_transform_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new_stub(
        "st_transform",
        ArgMatcher::new(
            vec![
                ArgMatcher::is_geometry(),
                ArgMatcher::or(vec![ArgMatcher::is_string(), ArgMatcher::is_numeric()]),
                ArgMatcher::optional(ArgMatcher::or(vec![
                    ArgMatcher::is_string(),
                    ArgMatcher::is_numeric(),
                ])),
                ArgMatcher::optional(ArgMatcher::is_boolean()),
            ],
            WKB_GEOMETRY,
        ),
        Volatility::Immutable,
        None,
    )
}

#[cfg(test)]
mod test {
    use super::*;
    use datafusion_expr::ScalarUDFImpl;

    #[test]
    fn udf_metadata() {
        let udf: SedonaScalarUDF = st_transform_udf();
        assert_eq!(udf.name(), "st_transform");
        assert!(udf.documentation().is_none());
    }
}
