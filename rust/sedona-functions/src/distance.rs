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

/// ST_Distance() scalar UDF stub
pub fn st_distance_udf() -> SedonaScalarUDF {
    distance_stub_udf("ST_Distance")
}

/// ST_DistanceSphere() scalar UDF stub
pub fn st_distance_sphere_udf() -> SedonaScalarUDF {
    distance_stub_udf("ST_DistanceSphere")
}

/// ST_DistanceSpheroid() scalar UDF stub
pub fn st_distance_spheroid_udf() -> SedonaScalarUDF {
    distance_stub_udf("ST_DistanceSpheroid")
}

/// ST_MaxDistance() scalar UDF stub
pub fn st_max_distance_udf() -> SedonaScalarUDF {
    distance_stub_udf("ST_MaxDistance")
}

/// ST_HausdorffDistance() scalar UDF stub
pub fn st_hausdorff_distance_udf() -> SedonaScalarUDF {
    distance_stub_udf("ST_HausdorffDistance")
}

/// ST_FrechetDistance() scalar UDF stub
pub fn st_frechet_distance_udf() -> SedonaScalarUDF {
    distance_stub_udf("ST_FrechetDistance")
}

pub fn distance_stub_udf(name: &str) -> SedonaScalarUDF {
    SedonaScalarUDF::new_stub(
        &name.to_lowercase(),
        ArgMatcher::new(
            vec![
                ArgMatcher::is_geometry_or_geography(),
                ArgMatcher::is_geometry_or_geography(),
            ],
            SedonaType::Arrow(DataType::Float64),
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
        let udf: ScalarUDF = st_distance_udf().into();
        assert_eq!(udf.name(), "st_distance");
    }
}
