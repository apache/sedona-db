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
use arrow_array::builder::UInt32Builder;
use std::{sync::Arc, vec};

use crate::executor::WkbExecutor;
use arrow_schema::DataType;
use datafusion_common::{DataFusionError, Result};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use sedona_expr::scalar_udf::{ArgMatcher, SedonaScalarKernel, SedonaScalarUDF};
use sedona_schema::datatypes::SedonaType;

/// ST_Srid() scalar UDF implementation
///
/// Scalar function to return the SRID of a geometry or geography
pub fn st_srid_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_srid",
        vec![Arc::new(STSRID {})],
        Volatility::Immutable,
        Some(st_srid_doc()),
    )
}

fn st_srid_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Return the spatial reference system identifier (SRID) of the geometry.",
        "ST_SRID (geom: Geometry)",
    )
    .with_argument("geom", "geometry: Input geometry or geography")
    .with_sql_example("SELECT ST_SRID(polygon))".to_string())
    .build()
}

#[derive(Debug)]
struct STSRID {}

impl SedonaScalarKernel for STSRID {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry_or_geography()],
            DataType::Utf8.try_into()?,
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);
        let mut builder = UInt32Builder::with_capacity(executor.num_iterations());
        let srid_opt = match &arg_types[0] {
            SedonaType::Wkb(_, Some(crs)) | SedonaType::WkbView(_, Some(crs)) => match crs.srid() {
                Some(srid) => Some(srid),
                None => return Err(DataFusionError::Execution("CRS has no SRID".to_string())),
            },
            _ => Some(0),
        };

        executor.execute_wkb_void(|maybe_wkb| {
            match maybe_wkb {
                Some(_wkb) => {
                    builder.append_option(srid_opt);
                }
                _ => builder.append_null(),
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

#[cfg(test)]
mod test {
    use arrow_array::create_array;
    use datafusion_expr::ScalarUDF;
    use sedona_schema::crs::{deserialize_crs, OGC_CRS84_PROJJSON};
    use sedona_schema::datatypes::Edges;
    use sedona_testing::{compare::assert_value_equal, create::create_array_value};
    use std::str::FromStr;

    use super::*;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = st_srid_udf().into();
        assert_eq!(udf.name(), "st_srid");
        assert!(udf.documentation().is_some())
    }

    #[test]
    fn udf() {
        let udf = st_srid_udf();

        // Call with a geometry with no CRS set
        let geom_none = create_array_value(
            &[None, Some("POINT (0 1)")],
            &SedonaType::Wkb(Edges::Planar, None),
        );
        let expected = ColumnarValue::Array(create_array!(UInt32, [None, Some(0)]));
        assert_value_equal(&udf.invoke_batch(&[geom_none], 1).unwrap(), &expected);

        // Call with a geometry with a CRS set
        let crs_value = serde_json::Value::String("EPSG:4837".to_string());
        let crs = deserialize_crs(&crs_value).unwrap();
        let geom_4326 = create_array_value(
            &[None, Some("POINT (0 1)")],
            &SedonaType::Wkb(Edges::Planar, Some(crs.unwrap())),
        );
        let expected = ColumnarValue::Array(create_array!(UInt32, [None, Some(4837)]));
        assert_value_equal(&udf.invoke_batch(&[geom_4326], 1).unwrap(), &expected);

        // Call with a CRS with no SRID (should error)
        let crs_value = serde_json::Value::from_str(OGC_CRS84_PROJJSON);
        let crs = deserialize_crs(&crs_value.unwrap()).unwrap();
        let geom_no_srid = create_array_value(
            &[None, Some("POINT (0 1)")],
            &SedonaType::Wkb(Edges::Planar, Some(crs.unwrap())),
        );
        let result = udf.invoke_batch(&[geom_no_srid], 1);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("CRS has no SRID"));
    }
}
