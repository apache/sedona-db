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

/// ST_KNN() scalar UDF stub
///
/// This is a stub function that defines the signature and documentation for ST_KNN
/// but does not contain an actual implementation. The real k-nearest neighbors logic
/// is handled by the spatial join execution engine.
pub fn st_knn_udf() -> SedonaScalarUDF {
    let stub_impl = SimpleSedonaScalarKernel::new_ref(
        ArgMatcher::new(
            vec![
                ArgMatcher::is_geometry_or_geography(),
                ArgMatcher::is_geometry_or_geography(),
                ArgMatcher::is_numeric(),
                ArgMatcher::is_boolean(),
            ],
            SedonaType::Arrow(DataType::Boolean),
        ),
        Arc::new(|_arg_types, _args| plan_err!("Can't execute ST_KNN() outside a spatial join")),
    );
    SedonaScalarUDF::from_impl("st_knn", stub_impl)
}

#[cfg(test)]
mod tests {
    use datafusion_expr::ScalarUDF;

    use super::*;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = st_knn_udf().into();
        assert_eq!(udf.name(), "st_knn");
    }
}
