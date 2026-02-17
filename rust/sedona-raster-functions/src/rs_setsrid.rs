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

use arrow_array::{Array, ArrayRef, StringViewArray, StructArray};
use arrow_buffer::NullBuffer;
use arrow_schema::DataType;
use datafusion_common::cast::{as_int64_array, as_string_view_array};
use datafusion_common::error::Result;
use datafusion_common::{exec_err, DataFusionError, ScalarValue};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_geometry::transform::CrsEngine;
use sedona_schema::crs::{CachedCrsNormalization, CachedSRIDToCrs};
use sedona_schema::datatypes::SedonaType;
use sedona_schema::matchers::ArgMatcher;
use sedona_schema::raster::{raster_indices, RasterSchema};

/// RS_SetSRID() scalar UDF implementation
///
/// An implementation of RS_SetSRID providing a scalar function implementation
/// based on an optional [CrsEngine]. If provided, it will be used to validate
/// the provided SRID (otherwise, all SRID input is applied without validation).
pub fn rs_set_srid_with_engine_udf(
    engine: Option<Arc<dyn CrsEngine + Send + Sync>>,
) -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_setsrid",
        vec![Arc::new(RsSetSrid { engine })],
        Volatility::Immutable,
        Some(rs_set_srid_doc()),
    )
}

/// RS_SetCRS() scalar UDF implementation
///
/// An implementation of RS_SetCRS providing a scalar function implementation
/// based on an optional [CrsEngine]. If provided, it will be used to validate
/// the provided CRS (otherwise, all CRS input is applied without validation).
pub fn rs_set_crs_with_engine_udf(
    engine: Option<Arc<dyn CrsEngine + Send + Sync>>,
) -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_setcrs",
        vec![Arc::new(RsSetCrs { engine })],
        Volatility::Immutable,
        Some(rs_set_crs_doc()),
    )
}

/// RS_SetSRID() scalar UDF implementation without CRS validation
///
/// See [rs_set_srid_with_engine_udf] for a validating version of this function
pub fn rs_set_srid_udf() -> SedonaScalarUDF {
    rs_set_srid_with_engine_udf(None)
}

/// RS_SetCRS() scalar UDF implementation without CRS validation
///
/// See [rs_set_crs_with_engine_udf] for a validating version of this function
pub fn rs_set_crs_udf() -> SedonaScalarUDF {
    rs_set_crs_with_engine_udf(None)
}

fn rs_set_srid_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Set the spatial reference system identifier (SRID) of the raster".to_string(),
        "RS_SetSRID(raster: Raster, srid: Integer)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_argument("srid", "Integer: EPSG code to set (e.g., 4326)")
    .with_sql_example("SELECT RS_SetSRID(RS_Example(), 3857)".to_string())
    .build()
}

fn rs_set_crs_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Set the coordinate reference system (CRS) of the raster".to_string(),
        "RS_SetCRS(raster: Raster, crs: String)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_argument(
        "crs",
        "String: Coordinate reference system identifier (e.g., 'OGC:CRS84', 'EPSG:4326')",
    )
    .with_sql_example("SELECT RS_SetCRS(RS_Example(), 'EPSG:3857')".to_string())
    .build()
}

// ---------------------------------------------------------------------------
// RS_SetSRID kernel
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct RsSetSrid {
    engine: Option<Arc<dyn CrsEngine + Send + Sync>>,
}

impl SedonaScalarKernel for RsSetSrid {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_raster(), ArgMatcher::is_integer()],
            SedonaType::Raster,
        );
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        _arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let raster_arg = &args[0];
        let srid_arg = &args[1];

        let input_nulls = extract_input_nulls(srid_arg);

        // Convert SRID integer(s) to CRS string(s)
        let crs_columnar = srid_to_crs_columnar(srid_arg, self.engine.as_ref())?;

        replace_raster_crs(raster_arg, &crs_columnar, input_nulls)
    }
}

// ---------------------------------------------------------------------------
// RS_SetCRS kernel
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct RsSetCrs {
    engine: Option<Arc<dyn CrsEngine + Send + Sync>>,
}

impl SedonaScalarKernel for RsSetCrs {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_raster(), ArgMatcher::is_string()],
            SedonaType::Raster,
        );
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        _arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let raster_arg = &args[0];
        let crs_arg = &args[1];

        let input_nulls = extract_input_nulls(crs_arg);

        // Normalize the CRS string(s) — abbreviate PROJJSON to authority:code, map "0" to null
        let crs_columnar = normalize_crs_columnar(crs_arg, self.engine.as_ref())?;

        replace_raster_crs(raster_arg, &crs_columnar, input_nulls)
    }
}

// ---------------------------------------------------------------------------
// Core: zero-copy CRS column swap
// ---------------------------------------------------------------------------

/// Replace the CRS column of a raster StructArray with a new CRS value.
///
/// This is a zero-copy operation for the metadata and bands columns:
/// we clone the Arc pointers for columns 0 (metadata) and 2 (bands),
/// and only rebuild column 1 (CRS) from the provided value.
///
/// When `input_nulls` is provided, rows where the original SRID/CRS input was
/// null will have the entire raster nulled out (not just the CRS column).
fn replace_raster_crs(
    raster_arg: &ColumnarValue,
    crs_array: &StringViewArray,
    input_nulls: Option<NullBuffer>,
) -> Result<ColumnarValue> {
    match raster_arg {
        ColumnarValue::Array(raster_array) => {
            let raster_struct = raster_array
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(|| {
                    datafusion_common::DataFusionError::Internal(
                        "Expected StructArray for raster data".to_string(),
                    )
                })?;

            let num_rows = raster_struct.len();
            let new_crs: ArrayRef = broadcast_string_view(crs_array, num_rows);
            let new_struct = swap_crs_column(raster_struct, new_crs)?;

            let input_nulls = input_nulls.map(|nulls| {
                if nulls.len() == 1 && num_rows != 1 {
                    if nulls.is_valid(0) {
                        NullBuffer::new_valid(num_rows)
                    } else {
                        NullBuffer::new_null(num_rows)
                    }
                } else {
                    nulls
                }
            });

            // Merge input nulls: rows where the SRID/CRS input was null become null rasters
            let merged_nulls = NullBuffer::union(new_struct.nulls(), input_nulls.as_ref());
            let new_struct = StructArray::new(
                RasterSchema::fields(),
                new_struct.columns().to_vec(),
                merged_nulls,
            );

            Ok(ColumnarValue::Array(Arc::new(new_struct)))
        }
        ColumnarValue::Scalar(ScalarValue::Struct(arc_struct)) => {
            let new_crs: ArrayRef = Arc::new(crs_array.clone());
            let new_struct = swap_crs_column(arc_struct.as_ref(), new_crs)?;

            // Merge input nulls: null SRID/CRS input produces a null raster
            let merged_nulls = NullBuffer::union(new_struct.nulls(), input_nulls.as_ref());
            let new_struct = StructArray::new(
                RasterSchema::fields(),
                new_struct.columns().to_vec(),
                merged_nulls,
            );

            Ok(ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(
                new_struct,
            ))))
        }
        ColumnarValue::Scalar(ScalarValue::Null) => Ok(ColumnarValue::Scalar(ScalarValue::Null)),
        _ => exec_err!("Expected raster (Struct) input for RS_SetSRID/RS_SetCRS"),
    }
}

/// Broadcast a `StringViewArray` to a target length.
///
/// If the array already has the target length, it is returned as-is (clone of Arc).
/// Otherwise the array must have length 1, and its single value is repeated.
fn broadcast_string_view(array: &StringViewArray, len: usize) -> ArrayRef {
    if array.len() == len {
        return Arc::new(array.clone());
    }
    debug_assert_eq!(array.len(), 1);
    if array.is_null(0) {
        Arc::new(StringViewArray::new_null(len))
    } else {
        let value = array.value(0);
        Arc::new(std::iter::repeat_n(Some(value), len).collect::<StringViewArray>())
    }
}

/// Swap only the CRS column of a raster StructArray, keeping all other columns intact.
fn swap_crs_column(raster_struct: &StructArray, new_crs_array: ArrayRef) -> Result<StructArray> {
    let mut columns: Vec<ArrayRef> = raster_struct.columns().to_vec();
    columns[raster_indices::CRS] = new_crs_array;
    Ok(StructArray::new(
        RasterSchema::fields(),
        columns,
        raster_struct.nulls().cloned(),
    ))
}

/// Extract a [NullBuffer] from the original SRID/CRS input argument.
///
/// For arrays, this returns the array's null buffer directly.
/// For scalars, this returns a single-element null buffer if the scalar is null.
///
/// This is used to distinguish "input was null" (which should null the raster)
/// from "input mapped to null CRS" (e.g. SRID=0 or CRS="0", which should
/// clear the CRS but preserve the raster).
fn extract_input_nulls(input: &ColumnarValue) -> Option<NullBuffer> {
    match input {
        ColumnarValue::Array(array) => array.nulls().cloned(),
        ColumnarValue::Scalar(scalar) => {
            if scalar.is_null() {
                Some(NullBuffer::new_null(1))
            } else {
                None
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SRID-to-CRS conversion
// ---------------------------------------------------------------------------

/// Convert an SRID integer ColumnarValue to a CRS StringViewArray ColumnarValue.
///
/// Uses [CachedSRIDToCrs] to avoid repeated validation of the same SRID within a batch.
///
/// Mapping:
/// - 0 -> null (no CRS)
/// - 4326 -> "OGC:CRS84"
/// - other -> "EPSG:{srid}"
fn srid_to_crs_columnar(
    srid_arg: &ColumnarValue,
    maybe_engine: Option<&Arc<dyn CrsEngine + Send + Sync>>,
) -> Result<StringViewArray> {
    let mut srid_to_crs = CachedSRIDToCrs::new();

    // Cast to Int64 for uniform handling
    let int_value = srid_arg.cast_to(&DataType::Int64, None)?;
    let int_array_ref = ColumnarValue::values_to_arrays(&[int_value])?;
    let int_array = as_int64_array(&int_array_ref[0])?;

    let crs_array: StringViewArray = int_array
        .iter()
        .map(|maybe_srid| -> Result<Option<String>> {
            if let Some(srid) = maybe_srid {
                let Some(auth_code) = srid_to_crs.get_crs(srid)? else {
                    return Ok(None);
                };
                validate_crs(&auth_code, maybe_engine)?;
                Ok(Some(auth_code))
            } else {
                Ok(None)
            }
        })
        .collect::<Result<_>>()?;

    Ok(crs_array)
}

// ---------------------------------------------------------------------------
// CRS string normalization
// ---------------------------------------------------------------------------

/// Normalize a CRS string ColumnarValue — abbreviate PROJJSON to authority:code
/// where possible, and map "0" to null.
///
/// Uses [CachedCrsNormalization] to avoid repeated deserialization of the same CRS
/// string within a batch.
fn normalize_crs_columnar(
    crs_arg: &ColumnarValue,
    _maybe_engine: Option<&Arc<dyn CrsEngine + Send + Sync>>,
) -> Result<StringViewArray> {
    let mut crs_norm = CachedCrsNormalization::new();

    let string_value = crs_arg.cast_to(&DataType::Utf8View, None)?;
    let string_array_ref = ColumnarValue::values_to_arrays(&[string_value])?;
    let string_view_array = as_string_view_array(&string_array_ref[0])?;

    let crs_array: StringViewArray = string_view_array
        .iter()
        .map(|maybe_crs| -> Result<Option<String>> {
            if let Some(crs_str) = maybe_crs {
                let normalized = crs_norm.normalize(crs_str)?;
                Ok(normalized)
            } else {
                Ok(None)
            }
        })
        .collect::<Result<_>>()?;

    Ok(crs_array)
}

/// Validate a CRS string
///
/// If an engine is provided, the engine will be used to validate the CRS.
/// Otherwise, no additional validation is performed (basic deserialization
/// checks are handled by the cache structs).
fn validate_crs(crs: &str, maybe_engine: Option<&Arc<dyn CrsEngine + Send + Sync>>) -> Result<()> {
    if let Some(engine) = maybe_engine {
        engine
            .as_ref()
            .get_transform_crs_to_crs(crs, crs, None, "")
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::StructArray;
    use datafusion_common::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use sedona_raster::array::RasterStructArray;
    use sedona_raster::traits::RasterRef;
    use sedona_schema::datatypes::RASTER;
    use sedona_testing::rasters::generate_test_rasters;
    use sedona_testing::testers::ScalarUdfTester;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = rs_set_srid_udf().into();
        assert_eq!(udf.name(), "rs_setsrid");
        assert!(udf.documentation().is_some());

        let udf: ScalarUDF = rs_set_crs_udf().into();
        assert_eq!(udf.name(), "rs_setcrs");
        assert!(udf.documentation().is_some());
    }

    #[test]
    fn set_srid_array() {
        let udf: ScalarUDF = rs_set_srid_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::UInt32)]);

        tester.assert_return_type(RASTER);

        // Generate rasters with OGC:CRS84 and set SRID to 3857
        let rasters = generate_test_rasters(3, Some(1)).unwrap();
        let result = tester
            .invoke_array_scalar(Arc::new(rasters), 3857u32)
            .unwrap();

        // Verify CRS was changed to EPSG:3857
        let result_struct = result.as_any().downcast_ref::<StructArray>().unwrap();
        let raster_array = RasterStructArray::new(result_struct);
        assert_eq!(raster_array.len(), 3);

        let raster0 = raster_array.get(0).unwrap();
        assert_eq!(raster0.crs(), Some("EPSG:3857"));

        // Null raster at index 1 should remain null
        assert!(raster_array.is_null(1));

        let raster2 = raster_array.get(2).unwrap();
        assert_eq!(raster2.crs(), Some("EPSG:3857"));
    }

    #[test]
    fn set_srid_4326_maps_to_ogc_crs84() {
        let udf: ScalarUDF = rs_set_srid_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::UInt32)]);

        let rasters = generate_test_rasters(1, None).unwrap();
        let result = tester
            .invoke_array_scalar(Arc::new(rasters), 4326u32)
            .unwrap();

        let result_struct = result.as_any().downcast_ref::<StructArray>().unwrap();
        let raster_array = RasterStructArray::new(result_struct);
        let raster = raster_array.get(0).unwrap();
        assert_eq!(raster.crs(), Some("OGC:CRS84"));
    }

    #[test]
    fn set_srid_zero_clears_crs() {
        let udf: ScalarUDF = rs_set_srid_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::UInt32)]);

        let rasters = generate_test_rasters(1, None).unwrap();
        let result = tester.invoke_array_scalar(Arc::new(rasters), 0u32).unwrap();

        let result_struct = result.as_any().downcast_ref::<StructArray>().unwrap();
        let raster_array = RasterStructArray::new(result_struct);
        let raster = raster_array.get(0).unwrap();
        // CRS should be None (null) for SRID 0
        assert_eq!(raster.crs(), None);
    }

    #[test]
    fn set_crs_array() {
        let udf: ScalarUDF = rs_set_crs_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Utf8)]);

        tester.assert_return_type(RASTER);

        let rasters = generate_test_rasters(3, Some(1)).unwrap();
        let result = tester
            .invoke_array_scalar(Arc::new(rasters), "EPSG:3857")
            .unwrap();

        let result_struct = result.as_any().downcast_ref::<StructArray>().unwrap();
        let raster_array = RasterStructArray::new(result_struct);
        assert_eq!(raster_array.len(), 3);

        let raster0 = raster_array.get(0).unwrap();
        assert_eq!(raster0.crs(), Some("EPSG:3857"));

        assert!(raster_array.is_null(1));

        let raster2 = raster_array.get(2).unwrap();
        assert_eq!(raster2.crs(), Some("EPSG:3857"));
    }

    #[test]
    fn set_crs_epsg_4326_normalizes_to_ogc_crs84() {
        let udf: ScalarUDF = rs_set_crs_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Utf8)]);

        let rasters = generate_test_rasters(1, None).unwrap();
        let result = tester
            .invoke_array_scalar(Arc::new(rasters), "EPSG:4326")
            .unwrap();

        let result_struct = result.as_any().downcast_ref::<StructArray>().unwrap();
        let raster_array = RasterStructArray::new(result_struct);
        let raster = raster_array.get(0).unwrap();
        // EPSG:4326 should normalize to OGC:CRS84
        assert_eq!(raster.crs(), Some("OGC:CRS84"));
    }

    #[test]
    fn set_crs_zero_clears_crs() {
        let udf: ScalarUDF = rs_set_crs_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Utf8)]);

        let rasters = generate_test_rasters(1, None).unwrap();
        let result = tester.invoke_array_scalar(Arc::new(rasters), "0").unwrap();

        let result_struct = result.as_any().downcast_ref::<StructArray>().unwrap();
        let raster_array = RasterStructArray::new(result_struct);
        let raster = raster_array.get(0).unwrap();
        assert_eq!(raster.crs(), None);
    }

    #[test]
    fn set_srid_preserves_metadata_and_bands() {
        let udf: ScalarUDF = rs_set_srid_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::UInt32)]);

        let rasters = generate_test_rasters(3, Some(1)).unwrap();
        let original_array = RasterStructArray::new(&rasters);

        let result = tester
            .invoke_array_scalar(Arc::new(rasters.clone()), 3857u32)
            .unwrap();
        let result_struct = result.as_any().downcast_ref::<StructArray>().unwrap();
        let result_array = RasterStructArray::new(result_struct);

        // Verify non-null rasters have same metadata and band data
        for i in [0, 2] {
            let original = original_array.get(i).unwrap();
            let modified = result_array.get(i).unwrap();

            // Metadata preserved
            assert_eq!(original.metadata().width(), modified.metadata().width());
            assert_eq!(original.metadata().height(), modified.metadata().height());
            assert_eq!(
                original.metadata().upper_left_x(),
                modified.metadata().upper_left_x()
            );
            assert_eq!(
                original.metadata().upper_left_y(),
                modified.metadata().upper_left_y()
            );

            // Band data preserved
            let orig_bands = original.bands();
            let mod_bands = modified.bands();
            assert_eq!(orig_bands.len(), mod_bands.len());
            for band_idx in 0..orig_bands.len() {
                let orig_band = orig_bands.band(band_idx + 1).unwrap();
                let mod_band = mod_bands.band(band_idx + 1).unwrap();
                assert_eq!(orig_band.data(), mod_band.data());
                assert_eq!(
                    orig_band.metadata().data_type().unwrap(),
                    mod_band.metadata().data_type().unwrap()
                );
            }

            // CRS changed
            assert_eq!(modified.crs(), Some("EPSG:3857"));
            assert_ne!(original.crs(), modified.crs());
        }
    }

    #[test]
    fn set_srid_scalar_null_raster() {
        let udf: ScalarUDF = rs_set_srid_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Int32)]);

        let result = tester
            .invoke_scalar_scalar(ScalarValue::Null, 3857)
            .unwrap();
        // ScalarValue::Null gets cast to a typed null raster struct by the tester,
        // so the result is a null struct entry (not ScalarValue::Null).
        match result {
            ScalarValue::Struct(s) => assert!(s.is_null(0), "Expected null raster at index 0"),
            other => panic!("Expected struct scalar, got {other:?}"),
        }
    }

    #[test]
    fn set_crs_scalar_null_raster() {
        let udf: ScalarUDF = rs_set_crs_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Utf8)]);

        let result = tester
            .invoke_scalar_scalar(ScalarValue::Null, "EPSG:4326")
            .unwrap();
        // ScalarValue::Null gets cast to a typed null raster struct by the tester,
        // so the result is a null struct entry (not ScalarValue::Null).
        match result {
            ScalarValue::Struct(s) => assert!(s.is_null(0), "Expected null raster at index 0"),
            other => panic!("Expected struct scalar, got {other:?}"),
        }
    }

    #[test]
    fn set_srid_with_null_srid() {
        let udf: ScalarUDF = rs_set_srid_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Int32)]);

        let rasters = generate_test_rasters(3, Some(1)).unwrap();
        let null_srid = ScalarValue::Int32(None);

        let result = tester
            .invoke_array_scalar(Arc::new(rasters), null_srid)
            .unwrap();
        let raster_array =
            RasterStructArray::new(result.as_any().downcast_ref::<StructArray>().unwrap());
        for i in 0..raster_array.len() {
            assert!(
                raster_array.is_null(i),
                "Expected null raster at index {i} for null SRID input"
            );
        }
    }

    #[test]
    fn set_crs_with_null_crs() {
        let udf: ScalarUDF = rs_set_crs_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Utf8)]);

        let rasters = generate_test_rasters(3, Some(1)).unwrap();
        let null_crs = ScalarValue::Utf8(None);

        let result = tester
            .invoke_array_scalar(Arc::new(rasters), null_crs)
            .unwrap();
        let raster_array =
            RasterStructArray::new(result.as_any().downcast_ref::<StructArray>().unwrap());
        for i in 0..raster_array.len() {
            assert!(
                raster_array.is_null(i),
                "Expected null raster at index {i} for null SRID input"
            );
        }
    }

    #[test]
    fn set_srid_array_with_null_srid_per_row() {
        let udf: ScalarUDF = rs_set_srid_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Int32)]);

        // 3 rasters (null at index 1), SRIDs: [Some(3857), Some(4326), None]
        let rasters = generate_test_rasters(3, Some(1)).unwrap();
        let srid_array: ArrayRef = Arc::new(arrow_array::Int32Array::from(vec![
            Some(3857),
            Some(4326),
            None,
        ]));

        let result = tester
            .invoke_array_array(Arc::new(rasters), srid_array)
            .unwrap();
        let result_struct = result.as_any().downcast_ref::<StructArray>().unwrap();
        let raster_array = RasterStructArray::new(result_struct);

        // Row 0: valid raster + valid SRID -> EPSG:3857
        let raster0 = raster_array.get(0).unwrap();
        assert_eq!(raster0.crs(), Some("EPSG:3857"));

        // Row 1: null raster (from input) -> still null
        assert!(raster_array.is_null(1));

        // Row 2: valid raster + null SRID -> null raster
        assert!(
            raster_array.is_null(2),
            "Expected null raster at index 2 (null SRID input)"
        );
    }

    #[test]
    fn set_crs_array_with_null_crs_per_row() {
        let udf: ScalarUDF = rs_set_crs_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Utf8)]);

        // 3 rasters (null at index 1), CRS strings: [Some("EPSG:3857"), Some("OGC:CRS84"), None]
        let rasters = generate_test_rasters(3, Some(1)).unwrap();
        let crs_array: ArrayRef = Arc::new(arrow_array::StringArray::from(vec![
            Some("EPSG:3857"),
            Some("OGC:CRS84"),
            None,
        ]));

        let result = tester
            .invoke_array_array(Arc::new(rasters), crs_array)
            .unwrap();
        let result_struct = result.as_any().downcast_ref::<StructArray>().unwrap();
        let raster_array = RasterStructArray::new(result_struct);

        // Row 0: valid raster + valid CRS -> EPSG:3857
        let raster0 = raster_array.get(0).unwrap();
        assert_eq!(raster0.crs(), Some("EPSG:3857"));

        // Row 1: null raster (from input) -> still null
        assert!(raster_array.is_null(1));

        // Row 2: valid raster + null CRS -> null raster
        assert!(
            raster_array.is_null(2),
            "Expected null raster at index 2 (null CRS input)"
        );
    }
}
