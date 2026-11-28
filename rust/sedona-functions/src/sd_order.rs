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

use datafusion_expr::Volatility;
use sedona_expr::scalar_udf::SedonaScalarUDF;

/// SD_Order() scalar UDF implementation
///
/// Registered kernels produce native ordering keys for supported input types.
/// The logical optimizer inserts this call only when an overload matches.
/// Geometry/geography ordering is registered when S2 is enabled; other types
/// can opt in by registering their own kernels.
pub fn sd_order_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new("sd_order", vec![], Volatility::Immutable)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::DataType;
    use datafusion_expr::ScalarUDF;
    use sedona_schema::datatypes::SedonaType;
    use sedona_testing::testers::ScalarUdfTester;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = sd_order_udf().into();
        assert_eq!(udf.name(), "sd_order");
    }

    #[test]
    fn no_identity_fallback() {
        let tester = ScalarUdfTester::new(
            sd_order_udf().into(),
            vec![SedonaType::Arrow(DataType::Utf8)],
        );
        assert_eq!(
            tester.return_type().unwrap_err().message(),
            "sd_order(utf8): No kernel matching arguments"
        );
    }
}
