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

use arrow_schema::DataType;
use datafusion_common::plan_err;
use sedona_expr::scalar_udf::{SedonaScalarUDF, SimpleSedonaScalarKernel};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

/// ST_Relate() scalar UDF stub
pub fn st_relate_udf() -> SedonaScalarUDF {
    let stub_impl = SimpleSedonaScalarKernel::new_ref(
        ArgMatcher::new(
            vec![ArgMatcher::is_geometry(), ArgMatcher::is_geometry()],
            SedonaType::Arrow(DataType::Utf8),
        ),
        Arc::new(|_arg_types, _args| plan_err!("ST_Relate() is not yet implemented")),
    );
    SedonaScalarUDF::from_impl("st_relate", stub_impl)
}

#[cfg(test)]
mod tests {
    use datafusion_expr::ScalarUDF;

    use super::*;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = st_relate_udf().into();
        assert_eq!(udf.name(), "st_relate");
    }
}
