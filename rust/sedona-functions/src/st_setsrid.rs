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
use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, OnceLock},
};

use arrow_array::{
    builder::{BinaryBuilder, NullBufferBuilder},
    new_null_array, ArrayRef, StringViewArray,
};
use arrow_buffer::NullBuffer;
use arrow_schema::DataType;
use datafusion_common::config::ConfigOptions;
use datafusion_common::{
    cast::{as_int64_array, as_string_view_array},
    error::Result,
    DataFusionError, ScalarValue,
};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use sedona_common::sedona_internal_err;
use sedona_expr::{
    item_crs::{
        make_item_crs, parse_item_crs_arg, parse_item_crs_arg_type,
        parse_item_crs_arg_type_strip_crs,
    },
    scalar_udf::{ScalarKernelRef, SedonaScalarKernel, SedonaScalarUDF},
};
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
    .with_sql_example("SELECT ST_GeomFromWKT('POINT (-64.363049 45.091501)', 4326)".to_string())
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
            || !(ArgMatcher::is_integer().match_type(&args[1])
                || ArgMatcher::is_null().match_type(&args[1]))
        {
            return Ok(None);
        }
        determine_return_type(args, scalar_args, self.engine.as_ref())
    }

    fn invoke_batch_from_args(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
        return_type: &SedonaType,
        _num_rows: usize,
        _config_options: Option<&ConfigOptions>,
    ) -> Result<ColumnarValue> {
        let (item_type, maybe_crs_type) = parse_item_crs_arg_type_strip_crs(&arg_types[0])?;
        let (item_arg, _) = parse_item_crs_arg(&item_type, &maybe_crs_type, &args[0])?;

        invoke_set_crs(
            &item_type,
            item_arg,
            &args[1],
            return_type,
            self.engine.as_ref(),
        )
    }

    fn return_type(&self, _args: &[SedonaType]) -> Result<Option<SedonaType>> {
        sedona_internal_err!(
            "Should not be called because return_type_from_args_and_scalars() is implemented"
        )
    }

    fn invoke_batch(
        &self,
        _arg_types: &[SedonaType],
        _args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        sedona_internal_err!("Should not be called because invoke_batch_from_args() is implemented")
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

    fn invoke_batch_from_args(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
        return_type: &SedonaType,
        _num_rows: usize,
        _config_options: Option<&ConfigOptions>,
    ) -> Result<ColumnarValue> {
        let (item_type, maybe_crs_type) = parse_item_crs_arg_type_strip_crs(&arg_types[0])?;
        let (item_arg, _) = parse_item_crs_arg(&item_type, &maybe_crs_type, &args[0])?;

        invoke_set_crs(
            &item_type,
            item_arg,
            &args[1],
            return_type,
            self.engine.as_ref(),
        )
    }

    fn return_type(&self, _args: &[SedonaType]) -> Result<Option<SedonaType>> {
        sedona_internal_err!(
            "Should not be called because return_type_from_args_and_scalars() is implemented"
        )
    }

    fn invoke_batch(
        &self,
        _arg_types: &[SedonaType],
        _args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        sedona_internal_err!("Should not be called because invoke_batch_from_args() is implemented")
    }
}

/// Shared invoke logic for both STSetSRID and STSetCRS.
///
/// When the SRID/CRS is a column (array), builds an `item_crs` struct with per-row CRS.
/// When the SRID/CRS is a scalar, the CRS is already baked into the return type and the
/// geometry is returned as-is. If the scalar SRID/CRS is NULL, the result is NULL per
/// standard SQL NULL propagation semantics.
fn invoke_set_crs(
    item_type: &SedonaType,
    item_arg: ColumnarValue,
    crs_arg: &ColumnarValue,
    return_type: &SedonaType,
    maybe_engine: Option<&Arc<dyn CrsEngine + Send + Sync>>,
) -> Result<ColumnarValue> {
    let item_crs_matcher = ArgMatcher::is_item_crs();
    if item_crs_matcher.match_type(return_type) {
        let normalized_crs_value = normalize_crs_array(crs_arg, maybe_engine)?;
        make_item_crs(
            item_type,
            item_arg,
            &ColumnarValue::Array(normalized_crs_value),
            crs_input_nulls(crs_arg),
        )
    } else if matches!(crs_arg, ColumnarValue::Scalar(sv) if sv.is_null()) {
        // Scalar NULL SRID/CRS: propagate NULL per SQL NULL semantics.
        let storage_type = return_type.storage_type();
        match &item_arg {
            ColumnarValue::Array(array) => Ok(ColumnarValue::Array(new_null_array(
                storage_type,
                array.len(),
            ))),
            ColumnarValue::Scalar(_) => {
                Ok(ColumnarValue::Scalar(ScalarValue::try_from(storage_type)?))
            }
        }
    } else {
        Ok(item_arg)
    }
}

fn determine_return_type(
    args: &[SedonaType],
    scalar_args: &[Option<&ScalarValue>],
    maybe_engine: Option<&Arc<dyn CrsEngine + Send + Sync>>,
) -> Result<Option<SedonaType>> {
    let (item_type, _) = parse_item_crs_arg_type_strip_crs(&args[0])?;

    // If this is not geometry or geography and/or this is not an item_crs of one,
    // this kernel does not apply.
    if !ArgMatcher::is_geometry_or_geography().match_type(&item_type) {
        return Ok(None);
    }

    if let Some(scalar_crs) = scalar_args[1] {
        if let ScalarValue::Utf8(maybe_crs) = scalar_crs.cast_to(&DataType::Utf8)? {
            let new_crs = match maybe_crs {
                Some(crs) => {
                    validate_crs(&crs, maybe_engine)?;
                    deserialize_crs(&crs)?
                }
                None => None,
            };

            match item_type {
                SedonaType::Wkb(edges, _) => Ok(Some(SedonaType::Wkb(edges, new_crs))),
                SedonaType::WkbView(edges, _) => Ok(Some(SedonaType::WkbView(edges, new_crs))),
                _ => sedona_internal_err!("Unexpected argument types: {}, {}", args[0], args[1]),
            }
        } else {
            sedona_internal_err!("Unexpected return type of cast to string")
        }
    } else {
        Ok(Some(SedonaType::new_item_crs(&item_type)?))
    }
}

/// [SedonaScalarKernel] wrapper that handles the SRID argument for constructors like ST_Point
#[derive(Debug)]
pub(crate) struct SRIDifiedKernel {
    inner: ScalarKernelRef,
}

impl SRIDifiedKernel {
    pub(crate) fn new(inner: ScalarKernelRef) -> Self {
        Self { inner }
    }
}

impl SedonaScalarKernel for SRIDifiedKernel {
    fn return_type_from_args_and_scalars(
        &self,
        args: &[SedonaType],
        scalar_args: &[Option<&ScalarValue>],
    ) -> Result<Option<SedonaType>> {
        // args should consist of the original args and one extra arg for
        // specifying CRS. So, first, validate the length and separate these.
        //
        // [arg0, arg1, ..., crs_arg];
        //  ^^^^^^^^^^^^^^^
        //     orig_args
        let orig_args_len = match (args.len(), scalar_args.len()) {
            (0, 0) => return Ok(None),
            (l1, l2) if l1 == l2 => l1 - 1,
            _ => return sedona_internal_err!("Arg types and arg values have different lengths"),
        };

        let orig_args = &args[..orig_args_len];
        let orig_scalar_args = &scalar_args[..orig_args_len];

        // Invoke the original return_type_from_args_and_scalars() first before checking the CRS argument
        let mut inner_result = match self
            .inner
            .return_type_from_args_and_scalars(orig_args, orig_scalar_args)?
        {
            Some(sedona_type) => sedona_type,
            // if no match, quit here. Since the CRS arg is also an unintended
            // one, validating it would be a cryptic error to the user.
            None => return Ok(None),
        };

        // If we have a scalar CRS, the output type has a type-level CRS
        if let Some(scalar_crs) = scalar_args[orig_args_len] {
            let new_crs = match scalar_crs.cast_to(&DataType::Utf8) {
                Ok(ScalarValue::Utf8(Some(crs))) => {
                    if crs == "0" {
                        None
                    } else {
                        validate_crs(&crs, None)?;
                        deserialize_crs(&crs)?
                    }
                }
                Ok(ScalarValue::Utf8(None)) => None,
                Ok(_) | Err(_) => {
                    return sedona_internal_err!("Can't cast Crs {scalar_crs:?} to Utf8")
                }
            };

            match &mut inner_result {
                SedonaType::Wkb(_, crs) => *crs = new_crs,
                SedonaType::WkbView(_, crs) => *crs = new_crs,
                _ => {
                    return sedona_internal_err!("Return type must be Wkb or WkbView");
                }
            }

            Ok(Some(inner_result))
        } else {
            // If we have a column CRS, the output has an item level CRS

            // Assert that the output type had a Crs of None. If the inner kernel returned
            // a specific output CRS it is likely that the SRIDified kernel is not appropriate.
            match &inner_result {
                SedonaType::Wkb(_, crs) | SedonaType::WkbView(_, crs) => {
                    if crs.is_some() {
                        return sedona_internal_err!(
                            "Return type of SRIDifiedKernel inner specified an explicit CRS"
                        );
                    }
                }
                _ => {
                    return sedona_internal_err!("Return type must be Wkb or WkbView");
                }
            }

            Ok(Some(SedonaType::new_item_crs(&inner_result)?))
        }
    }

    fn invoke_batch_from_args(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
        return_type: &SedonaType,
        _num_rows: usize,
        _config_options: Option<&ConfigOptions>,
    ) -> Result<ColumnarValue> {
        let orig_args_len = arg_types.len() - 1;
        let orig_arg_types = &arg_types[..orig_args_len];
        let orig_args = &args[..orig_args_len];

        // Invoke the inner UDF first to propagate any errors even when the CRS is NULL.
        // Note that, this behavior is different from PostGIS.
        let result = self.inner.invoke_batch(orig_arg_types, orig_args)?;

        // If the CRS input is a scalar, we can return the inner result as-is except
        // for the NULL case.
        if let ColumnarValue::Scalar(sc) = &args[orig_args_len] {
            // If the specified SRID is NULL, the result is also NULL.
            if sc.is_null() {
                // Create the same length of NULLs as the original result.
                let len = match &result {
                    ColumnarValue::Array(array) => array.len(),
                    ColumnarValue::Scalar(_) => 1,
                };

                let mut builder = BinaryBuilder::with_capacity(len, 0);
                for _ in 0..len {
                    builder.append_null();
                }
                let new_array = builder.finish();
                return Ok(ColumnarValue::Array(Arc::new(new_array)));
            }

            Ok(result)
        } else {
            let (item_type, _) = parse_item_crs_arg_type(return_type)?;
            let normalized_crs_value = normalize_crs_array(&args[orig_args_len], None)?;
            make_item_crs(
                &item_type,
                result,
                &ColumnarValue::Array(normalized_crs_value),
                crs_input_nulls(&args[orig_args_len]),
            )
        }
    }

    fn invoke_batch(
        &self,
        _arg_types: &[SedonaType],
        _args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        sedona_internal_err!("Should not be called because invoke_batch_from_args() is implemented")
    }

    fn return_type(&self, _args: &[SedonaType]) -> Result<Option<SedonaType>> {
        sedona_internal_err!(
            "Should not be called because return_type_from_args_and_scalars() is implemented"
        )
    }
}

static SINGLE_NULL_BUFFER: OnceLock<NullBuffer> = OnceLock::new();

fn crs_input_nulls(crs_value: &ColumnarValue) -> Option<&NullBuffer> {
    match crs_value {
        ColumnarValue::Array(array) => array.nulls(),
        ColumnarValue::Scalar(scalar_value) => {
            if scalar_value.is_null() {
                let null_buffer = SINGLE_NULL_BUFFER.get_or_init(|| {
                    let mut builder = NullBufferBuilder::new(1);
                    builder.append(false);
                    builder
                        .finish()
                        .expect("Failed to build single null buffer")
                });
                Some(null_buffer)
            } else {
                None
            }
        }
    }
}

/// Given an SRID or CRS array, compute the final crs array to put in the item_crs struct
///
/// For SRID arrays, this is `EPSG:<srid>` except for the SRID of 0 (which maps
/// to a null value in the CRS array) and 4326 (which maps to a value of OGC:CRS84
/// in the CRS array).
///
/// For CRS arrays of strings, this function attempts to abbreviate any inputs. For example,
/// PROJJSON input will attempt to be abbreviated to authority:code if possible (or left
/// as is otherwise). The special value "0" maps to a null value in the CRS array.
fn normalize_crs_array(
    crs_value: &ColumnarValue,
    maybe_engine: Option<&Arc<dyn CrsEngine + Send + Sync>>,
) -> Result<ArrayRef> {
    match crs_value.data_type() {
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64 => {
            // Local cache to avoid re-validating inputs
            let mut known_valid = HashSet::new();

            let int_value = crs_value.cast_to(&DataType::Int64, None)?;
            let int_array_ref = ColumnarValue::values_to_arrays(&[int_value])?;
            let int_array = as_int64_array(&int_array_ref[0])?;
            let utf8_view_array = int_array
                .iter()
                .map(|maybe_srid| -> Result<Option<String>> {
                    if let Some(srid) = maybe_srid {
                        if srid == 0 {
                            return Ok(None);
                        } else if srid == 4326 {
                            return Ok(Some("OGC:CRS84".to_string()));
                        }

                        let auth_code = format!("EPSG:{srid}");
                        if !known_valid.contains(&srid) {
                            validate_crs(&auth_code, maybe_engine)?;
                            known_valid.insert(srid);
                        }

                        Ok(Some(auth_code))
                    } else {
                        Ok(None)
                    }
                })
                .collect::<Result<StringViewArray>>()?;

            Ok(Arc::new(utf8_view_array))
        }
        _ => {
            let mut known_abbreviated = HashMap::<String, String>::new();

            let string_value = crs_value.cast_to(&DataType::Utf8View, None)?;
            let string_array_ref = ColumnarValue::values_to_arrays(&[string_value])?;
            let string_view_array = as_string_view_array(&string_array_ref[0])?;
            let utf8_view_array = string_view_array
                .iter()
                .map(|maybe_crs| -> Result<Option<String>> {
                    if let Some(crs_str) = maybe_crs {
                        if crs_str == "0" {
                            return Ok(None);
                        }

                        if let Some(abbreviated_crs) = known_abbreviated.get(crs_str) {
                            Ok(Some(abbreviated_crs.clone()))
                        } else if let Some(crs) = deserialize_crs(crs_str)? {
                            let abbreviated_crs =
                                if let Some(auth_code) = crs.to_authority_code()? {
                                    auth_code
                                } else {
                                    crs_str.to_string()
                                };

                            known_abbreviated.insert(crs.to_string(), abbreviated_crs.clone());
                            Ok(Some(abbreviated_crs))
                        } else {
                            Ok(None)
                        }
                    } else {
                        Ok(None)
                    }
                })
                .collect::<Result<StringViewArray>>()?;

            Ok(Arc::new(utf8_view_array))
        }
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

#[cfg(test)]
mod test {
    use std::rc::Rc;

    use arrow_array::{create_array, ArrayRef};
    use arrow_schema::Field;
    use datafusion_common::config::ConfigOptions;
    use datafusion_expr::{ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF};
    use rstest::rstest;
    use sedona_geometry::{error::SedonaGeometryError, transform::CrsTransform};
    use sedona_schema::{
        crs::lnglat,
        datatypes::{Edges, WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS},
    };
    use sedona_testing::{
        compare::assert_value_equal,
        create::{create_array, create_array_item_crs, create_scalar_value},
        testers::ScalarUdfTester,
    };

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

        // Call with a null srid (result should be NULL per SQL NULL propagation)
        let (return_type, result) = call_udf(
            &udf,
            geom_arg.clone(),
            &[WKB_GEOMETRY, SedonaType::Arrow(DataType::UInt32)],
            null_srid_scalar.clone(),
        )
        .unwrap();
        assert_eq!(return_type, WKB_GEOMETRY);
        let null_geom = create_scalar_value(None, &WKB_GEOMETRY);
        assert_value_equal(&result, &null_geom);
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

        // Call with a null scalar destination (result should be NULL per SQL NULL propagation)
        let (return_type, result) = call_udf(
            &udf,
            geom_arg.clone(),
            &[WKB_GEOMETRY, SedonaType::Arrow(DataType::Utf8)],
            null_crs_scalar.clone(),
        )
        .unwrap();
        assert_eq!(return_type, WKB_GEOMETRY);
        let null_geom = create_scalar_value(None, &WKB_GEOMETRY);
        assert_value_equal(&result, &null_geom);

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

    #[rstest]
    fn udf_item_srid_output(
        #[values(WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS.clone())] sedona_type: SedonaType,
    ) {
        let tester = ScalarUdfTester::new(
            st_set_srid_udf().into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Int32)],
        );
        tester.assert_return_type(WKB_GEOMETRY_ITEM_CRS.clone());

        let geometry_array = create_array(
            &[
                Some("POINT (0 1)"),
                Some("POINT (2 3)"),
                Some("POINT (4 5)"),
                Some("POINT (6 7)"),
                Some("POINT (8 9)"),
            ],
            &sedona_type,
        );
        let crs_array =
            create_array!(Int32, [Some(4326), Some(3857), Some(3857), Some(0), None]) as ArrayRef;

        let result = tester
            .invoke_array_array(geometry_array, crs_array)
            .unwrap();
        assert_eq!(
            &result,
            &create_array_item_crs(
                &[
                    Some("POINT (0 1)"),
                    Some("POINT (2 3)"),
                    Some("POINT (4 5)"),
                    Some("POINT (6 7)"),
                    None,
                ],
                [
                    Some("OGC:CRS84"),
                    Some("EPSG:3857"),
                    Some("EPSG:3857"),
                    None,
                    None
                ],
                &WKB_GEOMETRY
            )
        );
    }

    #[rstest]
    fn udf_item_crs_output(
        #[values(WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS.clone())] sedona_type: SedonaType,
    ) {
        let tester = ScalarUdfTester::new(
            st_set_crs_udf().into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Utf8)],
        );
        tester.assert_return_type(WKB_GEOMETRY_ITEM_CRS.clone());

        let geometry_array = create_array(
            &[
                Some("POINT (0 1)"),
                Some("POINT (2 3)"),
                Some("POINT (4 5)"),
                Some("POINT (6 7)"),
                Some("POINT (8 9)"),
            ],
            &sedona_type,
        );
        let crs_array = create_array!(
            Utf8,
            [
                Some("EPSG:4326"),
                Some("EPSG:3857"),
                Some("EPSG:3857"),
                Some("0"),
                None
            ]
        ) as ArrayRef;

        let result = tester
            .invoke_array_array(geometry_array, crs_array)
            .unwrap();
        assert_eq!(
            &result,
            &create_array_item_crs(
                &[
                    Some("POINT (0 1)"),
                    Some("POINT (2 3)"),
                    Some("POINT (4 5)"),
                    Some("POINT (6 7)"),
                    None
                ],
                [
                    Some("OGC:CRS84"),
                    Some("EPSG:3857"),
                    Some("EPSG:3857"),
                    None,
                    None
                ],
                &WKB_GEOMETRY
            )
        );
    }

    fn call_udf(
        udf: &ScalarUDF,
        arg: ColumnarValue,
        arg_type: &[SedonaType],
        to: ScalarValue,
    ) -> Result<(SedonaType, ColumnarValue)> {
        let SedonaType::Arrow(datatype) = &arg_type[1] else {
            return sedona_internal_err!(
                "Expected SedonaType::Arrow, but found a different variant"
            );
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
            config_options: Arc::new(ConfigOptions::default()),
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
