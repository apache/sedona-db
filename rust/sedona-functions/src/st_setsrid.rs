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

use arrow_schema::DataType;
use datafusion_common::{error::Result, DataFusionError, ScalarValue};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use sedona_common::sedona_internal_err;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_geometry::transform::CrsEngine;
use sedona_schema::{crs::deserialize_crs, datatypes::SedonaType, matchers::ArgMatcher};

/// ST_SetSRID() scalar UDF implementation
///
/// An implementation of ST_SetSRID providing a scalar function implementation
/// based on an optional [CrsEngine]. If provided, it will be used to validate
/// the provided SRID (otherwise, all SRID input is applied without validation).
pub fn st_set_srid_with_engine_udf(
    engine: Option<Arc<dyn CrsEngine + Send + Sync>>,
) -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_setsrid",
        vec![Arc::new(STSetSRID { engine })],
        Volatility::Immutable,
        Some(set_srid_doc()),
    )
}

/// ST_SetCRS() scalar UDF implementation without CRS validation
///
/// An implementation of ST_SetCRS providing a scalar function implementation
/// based on an optional [CrsEngine]. If provided, it will be used to validate
/// The provided CRS (otherwise, all CRS input is applied without validation).
pub fn st_set_crs_with_engine_udf(
    engine: Option<Arc<dyn CrsEngine + Send + Sync>>,
) -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_setcrs",
        vec![Arc::new(STSetCRS { engine })],
        Volatility::Immutable,
        Some(set_crs_doc()),
    )
}

/// ST_SetSRID() scalar UDF implementation without CRS validation
///
/// See [st_set_srid_with_engine_udf] for a validating version of this function
pub fn st_set_srid_udf() -> SedonaScalarUDF {
    st_set_srid_with_engine_udf(None)
}

/// ST_SetCRS() scalar UDF implementation without CRS validation
///
/// See [st_set_crs_with_engine_udf] for a validating version of this function
pub fn st_set_crs_udf() -> SedonaScalarUDF {
    st_set_crs_with_engine_udf(None)
}

fn set_srid_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Sets the spatial reference system identifier (SRID) of the geometry.",
        "ST_SetSRID (geom: Geometry, srid: Integer)",
    )
    .with_argument("geom", "geometry: Input geometry or geography")
    .with_argument("srid", "srid: EPSG code to set (e.g., 4326)")
    .with_sql_example(
        "SELECT ST_SetSRID(ST_GeomFromWKT('POINT (-64.363049 45.091501)'), 4326)".to_string(),
    )
    .build()
}

fn set_crs_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Set CRS information for a geometry or geography",
        "ST_SetCrs (geom: Geometry, crs: String)",
    )
    .with_argument("geom", "geometry: Input geometry or geography")
    .with_argument(
        "crs",
        "string: Coordinate reference system identifier (e.g., 'OGC:CRS84')",
    )
    .with_sql_example(
        "SELECT ST_SetCrs(ST_GeomFromWKT('POINT (-64.363049 45.091501)'), 'OGC:CRS84')".to_string(),
    )
    .build()
}

#[derive(Debug)]
struct STSetSRID {
    engine: Option<Arc<dyn CrsEngine + Send + Sync>>,
}

impl SedonaScalarKernel for STSetSRID {
    fn return_type_from_args_and_scalars(
        &self,
        args: &[SedonaType],
        scalar_args: &[Option<&ScalarValue>],
    ) -> Result<Option<SedonaType>> {
        if args.len() != 2
            || !(ArgMatcher::is_numeric().match_type(&args[1])
                || ArgMatcher::is_null().match_type(&args[1]))
        {
            return Ok(None);
        }
        determine_return_type(args, scalar_args, self.engine.as_ref())
    }

    fn invoke_batch(
        &self,
        _arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        Ok(args[0].clone())
    }

    fn return_type(&self, _args: &[SedonaType]) -> Result<Option<SedonaType>> {
        sedona_internal_err!(
            "Should not be called because return_type_from_args_and_scalars() is implemented"
        )
    }
}

#[derive(Debug)]
struct STSetCRS {
    engine: Option<Arc<dyn CrsEngine + Send + Sync>>,
}

impl SedonaScalarKernel for STSetCRS {
    fn return_type_from_args_and_scalars(
        &self,
        args: &[SedonaType],
        scalar_args: &[Option<&ScalarValue>],
    ) -> Result<Option<SedonaType>> {
        if args.len() != 2
            || !(ArgMatcher::is_string().match_type(&args[1])
                || ArgMatcher::is_null().match_type(&args[1]))
        {
            return Ok(None);
        }
        determine_return_type(args, scalar_args, self.engine.as_ref())
    }

    fn invoke_batch(
        &self,
        _arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        Ok(args[0].clone())
    }

    fn return_type(&self, _args: &[SedonaType]) -> Result<Option<SedonaType>> {
        sedona_internal_err!(
            "Should not be called because return_type_from_args_and_scalars() is implemented"
        )
    }
}

/// Validate a CRS string
///
/// If an engine is provided, the engine will be used to validate the CRS. If absent,
/// the CRS will only be validated using the basic checks in [deserialize_crs].
pub fn validate_crs(
    crs: &str,
    maybe_engine: Option<&Arc<dyn CrsEngine + Send + Sync>>,
) -> Result<()> {
    if let Some(engine) = maybe_engine {
        engine
            .as_ref()
            .get_transform_crs_to_crs(crs, crs, None, "")
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
    }

    Ok(())
}

fn determine_return_type(
    args: &[SedonaType],
    scalar_args: &[Option<&ScalarValue>],
    maybe_engine: Option<&Arc<dyn CrsEngine + Send + Sync>>,
) -> Result<Option<SedonaType>> {
    if !ArgMatcher::is_geometry_or_geography().match_type(&args[0]) {
        return Ok(None);
    }

    if let Some(scalar_crs) = scalar_args[1] {
        if let ScalarValue::Utf8(maybe_crs) = scalar_crs.cast_to(&DataType::Utf8)? {
            let new_crs = match maybe_crs {
                Some(crs) => {
                    if crs == "0" {
                        None
                    } else {
                        validate_crs(&crs, maybe_engine)?;
                        deserialize_crs(&serde_json::Value::String(crs))?
                    }
                }
                None => None,
            };

            match args[0] {
                SedonaType::Wkb(edges, _) => return Ok(Some(SedonaType::Wkb(edges, new_crs))),
                SedonaType::WkbView(edges, _) => {
                    return Ok(Some(SedonaType::WkbView(edges, new_crs)))
                }
                _ => {}
            }
        }
    }

    sedona_internal_err!("Unexpected argument types: {}, {}", args[0], args[1])
}

#[cfg(test)]
mod test {
    use std::rc::Rc;

    use arrow_schema::Field;
    use datafusion_expr::{ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF};
    use sedona_geometry::{error::SedonaGeometryError, transform::CrsTransform};
    use sedona_schema::{
        crs::lnglat,
        datatypes::{Edges, WKB_GEOMETRY},
    };
    use sedona_testing::{compare::assert_value_equal, create::create_scalar_value};

    use super::*;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = st_set_srid_udf().into();
        assert_eq!(udf.name(), "st_setsrid");
        assert!(udf.documentation().is_some());

        let udf: ScalarUDF = st_set_crs_udf().into();
        assert_eq!(udf.name(), "st_setcrs");
        assert!(udf.documentation().is_some());
    }

    #[test]
    fn udf_srid() {
        let udf: ScalarUDF = st_set_srid_udf().into();

        let wkb_lnglat = SedonaType::Wkb(Edges::Planar, lnglat());
        let geom_arg = create_scalar_value(Some("POINT (0 1)"), &WKB_GEOMETRY);
        let geom_lnglat = create_scalar_value(Some("POINT (0 1)"), &wkb_lnglat);

        let srid_scalar = ScalarValue::UInt32(Some(4326));
        let unset_scalar = ScalarValue::UInt32(Some(0));
        let null_srid_scalar = ScalarValue::UInt32(None);

        // Call with an integer code destination (should result in a lnglat crs)
        let (return_type, result) = call_udf(
            &udf,
            geom_arg.clone(),
            &[WKB_GEOMETRY, SedonaType::Arrow(DataType::UInt32)],
            srid_scalar.clone(),
        )
        .unwrap();
        assert_eq!(return_type, wkb_lnglat);
        assert_value_equal(&result, &geom_lnglat);

        // Call with an integer code of 0 (should unset the output crs)
        let (return_type, result) = call_udf(
            &udf,
            geom_lnglat.clone(),
            &[WKB_GEOMETRY, SedonaType::Arrow(DataType::UInt32)],
            unset_scalar.clone(),
        )
        .unwrap();
        assert_eq!(return_type, WKB_GEOMETRY);
        assert_value_equal(&result, &geom_arg);

        // Call with a null srid (should *not* set the output crs)
        let (return_type, result) = call_udf(
            &udf,
            geom_arg.clone(),
            &[WKB_GEOMETRY, SedonaType::Arrow(DataType::UInt32)],
            null_srid_scalar.clone(),
        )
        .unwrap();
        assert_eq!(return_type, WKB_GEOMETRY);
        assert_value_equal(&result, &geom_arg);
    }

    #[test]
    fn udf_crs() {
        let udf: ScalarUDF = st_set_crs_udf().into();

        let wkb_lnglat = SedonaType::Wkb(Edges::Planar, lnglat());
        let geom_arg = create_scalar_value(Some("POINT (0 1)"), &WKB_GEOMETRY);
        let geom_lnglat = create_scalar_value(Some("POINT (0 1)"), &wkb_lnglat);

        let good_crs_scalar = ScalarValue::Utf8(Some("EPSG:4326".to_string()));
        let null_crs_scalar = ScalarValue::Utf8(None);
        let questionable_crs_scalar = ScalarValue::Utf8(Some("gazornenplat".to_string()));

        // Call with a string scalar destination
        let (return_type, result) = call_udf(
            &udf,
            geom_arg.clone(),
            &[WKB_GEOMETRY, SedonaType::Arrow(DataType::Utf8)],
            good_crs_scalar.clone(),
        )
        .unwrap();
        assert_eq!(return_type, wkb_lnglat);
        assert_value_equal(&result, &geom_lnglat);

        // Call with a null scalar destination (should *not* set the output crs)
        let (return_type, result) = call_udf(
            &udf,
            geom_arg.clone(),
            &[WKB_GEOMETRY, SedonaType::Arrow(DataType::Utf8)],
            null_crs_scalar.clone(),
        )
        .unwrap();
        assert_eq!(return_type, WKB_GEOMETRY);
        assert_value_equal(&result, &geom_arg);

        // Ensure that an engine can reject a CRS if the UDF was constructed with one
        let udf_with_validation: ScalarUDF =
            st_set_crs_with_engine_udf(Some(Arc::new(ExtremelyUnusefulEngine {}))).into();
        let err = call_udf(
            &udf_with_validation,
            geom_arg.clone(),
            &[WKB_GEOMETRY, SedonaType::Arrow(DataType::Utf8)],
            questionable_crs_scalar.clone(),
        )
        .unwrap_err();
        assert_eq!(err.message(), "Unknown geometry error")
    }

    fn call_udf(
        udf: &ScalarUDF,
        arg: ColumnarValue,
        arg_type: &[SedonaType],
        to: ScalarValue,
    ) -> Result<(SedonaType, ColumnarValue)> {
        let SedonaType::Arrow(datatype) = &arg_type[1] else {
            return Err(DataFusionError::Internal(
                "Expected SedonaType::Arrow, but found a different variant".to_string(),
            ));
        };
        let arg_fields = vec![
            Arc::new(arg_type[0].to_storage_field("", true)?),
            Field::new("", datatype.clone(), true).into(),
        ];
        let return_field_args = ReturnFieldArgs {
            arg_fields: &arg_fields,
            scalar_arguments: &[None, Some(&to)],
        };

        let return_field = udf.return_field_from_args(return_field_args)?;
        let return_type = SedonaType::from_storage_field(&return_field)?;

        let args = ScalarFunctionArgs {
            args: vec![arg, to.into()],
            arg_fields,
            number_rows: 1,
            return_field,
        };

        let value = udf.invoke_with_args(args)?;
        Ok((return_type, value))
    }

    #[derive(Debug)]
    struct ExtremelyUnusefulEngine {}

    impl CrsEngine for ExtremelyUnusefulEngine {
        fn get_transform_crs_to_crs(
            &self,
            _from: &str,
            _to: &str,
            _area_of_interest: Option<sedona_geometry::bounding_box::BoundingBox>,
            _options: &str,
        ) -> Result<Rc<dyn CrsTransform>, SedonaGeometryError> {
            Err(SedonaGeometryError::Unknown)
        }

        fn get_transform_pipeline(
            &self,
            _pipeline: &str,
            _options: &str,
        ) -> Result<Rc<dyn CrsTransform>, SedonaGeometryError> {
            Err(SedonaGeometryError::Unknown)
        }
    }
}
