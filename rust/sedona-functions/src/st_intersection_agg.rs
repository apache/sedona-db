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
use sedona_expr::aggregate_udf::SedonaAggregateUDF;
use sedona_schema::{
    datatypes::{Edges, SedonaType},
    matchers::ArgMatcher,
};

/// ST_Intersection_Agg() aggregate UDF implementation
///
/// An implementation of intersection calculation.
pub fn st_intersection_agg_udf() -> SedonaAggregateUDF {
    SedonaAggregateUDF::new_stub(
        "st_intersection_agg",
        ArgMatcher::new(
            vec![ArgMatcher::is_geometry_or_geography()],
            SedonaType::Wkb(Edges::Planar, None),
        ),
        Volatility::Immutable,
    )
}

#[cfg(test)]
mod test {
    use datafusion_expr::AggregateUDF;

    use super::*;

    #[test]
    fn udf_metadata() {
        let udf: AggregateUDF = st_intersection_agg_udf().into();
        assert_eq!(udf.name(), "st_intersection_agg");
    }
}
