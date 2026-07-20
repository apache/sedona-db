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

//! RS_MapAlgebra UDF — evaluate a per-pixel expression over a raster's bands.
//!
//! **Experimental.** The expression dialect and the options argument may change.
//!
//! `RS_MapAlgebra(raster, expr [, options])` evaluates `expr` once per pixel and
//! writes the result into a single-band output raster that keeps the input's
//! geotransform, CRS, and spatial extent.
//!
//! # Expression dialect
//!
//! `expr` is a single expression (not a statement script) evaluated by the
//! [`evalexpr`] crate. It must evaluate to a number or boolean; the value is
//! coerced to the output pixel type. Supported syntax includes arithmetic
//! (`+ - * / % ^`), comparison (`== != < <= > >=`), boolean (`&& || !`), the
//! `min`/`max`/`if(cond, a, b)` functions, and the `math::*` functions
//! (`math::sqrt`, `math::ln`, ...); see the `evalexpr` documentation for the
//! full grammar.
//!
//! Variables available to the expression:
//! - `rast0`, `rast1`, ..., `rastN` — the value of band N (0-based) at the
//!   current pixel. `rast` is an alias for `rast0`.
//! - `x`, `y` — the current pixel's column and row (0-based).
//! - `width`, `height` — the raster's pixel dimensions.
//!
//! All variables are bound as floating point, so mixing them with integer
//! literals (e.g. `rast0 * 2`) yields a floating-point result and `/` is always
//! floating-point division.
//!
//! This dialect differs from Sedona Spark's `RS_MapAlgebra`, which runs a
//! Jiffle script referencing bands as `rast[0]` and assigning to `out[0]`.
//!
//! # Materialization
//!
//! Each band the expression references is materialized as a contiguous `f64`
//! array before evaluation, and the whole output band is built in memory before
//! it is moved into the Arrow array. Evaluation is also `f64`-only, so `Int64`
//! and `UInt64` inputs whose magnitude exceeds 2^53 lose precision on the way in
//! and are not representable exactly on the way out. A native implementation
//! evaluating directly over the N-D `nd_buffer` strides, per data type, would
//! avoid both the whole-band materialization and the `f64` round-trip.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::ArrayRef;
use arrow_buffer::Buffer;
use arrow_schema::DataType;
use datafusion_common::cast::as_string_array;
use datafusion_common::config::ConfigOptions;
use datafusion_common::error::Result;
use datafusion_common::{exec_datafusion_err, exec_err, ScalarValue};
use datafusion_expr::{ColumnarValue, Volatility};
use evalexpr::{build_operator_tree, ContextWithMutableVariables, HashMapContext, Value};
use serde::Deserialize;

use crate::rs_ensure_loaded::{NEEDS_PIXELS_METADATA_KEY, RETURNS_BYTES_METADATA_KEY};
use crate::RasterExecutor;
use sedona_common::sedona_internal_err;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::array::RasterRefImpl;
use sedona_raster::builder::RasterBuilder;
use sedona_raster::traits::{nodata_f64_to_bytes, BandRef, RasterRef};
use sedona_schema::datatypes::{SedonaType, RASTER};
use sedona_schema::matchers::ArgMatcher;
use sedona_schema::raster::BandDataType;

/// RS_MapAlgebra() scalar UDF implementation.
///
/// Signatures:
/// - `RS_MapAlgebra(raster, expr)` — 2 args
/// - `RS_MapAlgebra(raster, expr, options)` — 3 args, `options` a JSON string
pub fn rs_map_algebra_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_mapalgebra",
        vec![
            Arc::new(RsMapAlgebra { arg_count: 2 }),
            Arc::new(RsMapAlgebra { arg_count: 3 }),
        ],
        Volatility::Immutable,
    )
    // Reads band pixels (so the planner materializes OutDb rasters via
    // RS_EnsureLoaded first) and emits a fresh InDb raster (so its output is
    // already loaded and isn't wrapped again).
    .with_metadata(NEEDS_PIXELS_METADATA_KEY, "true")
    .with_metadata(RETURNS_BYTES_METADATA_KEY, "true")
}

/// Kernel implementation for RS_MapAlgebra.
#[derive(Debug)]
struct RsMapAlgebra {
    /// Number of arguments in the matched signature (2 or 3).
    arg_count: usize,
}

/// Options for [`rs_map_algebra_udf`], deserialized from the JSON `options`
/// argument. Every field is optional; an empty object (`{}`) is the default.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
struct MapAlgebraOptions {
    /// Output pixel type (e.g. `"F"`, `"D"`, `"I"`, `"uint8"`). Defaults to the
    /// input raster's first band data type.
    pixel_type: Option<String>,
    /// nodata value recorded on the output band. It is stored in the band
    /// metadata verbatim; it is not written into any pixel. A value that is not
    /// exactly representable in the output pixel type is an error rather than a
    /// silent truncation.
    nodata: Option<f64>,
}

impl SedonaScalarKernel for RsMapAlgebra {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matchers = match self.arg_count {
            2 => vec![ArgMatcher::is_raster(), ArgMatcher::is_string()],
            3 => vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_string(),
                ArgMatcher::is_string(),
            ],
            _ => {
                return sedona_internal_err!(
                    "RS_MapAlgebra: unexpected arg_count {}",
                    self.arg_count
                );
            }
        };
        ArgMatcher::new(matchers, RASTER).match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        self.invoke_batch_from_args(arg_types, args, &SedonaType::Arrow(DataType::Null), 0, None)
    }

    fn invoke_batch_from_args(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
        _return_type: &SedonaType,
        _num_rows: usize,
        _config_options: Option<&ConfigOptions>,
    ) -> Result<ColumnarValue> {
        let num_iterations = RasterExecutor::num_iterations_over(args);

        // `expr` at index 1, optional `options` JSON at index 2.
        let expr_array = args[1]
            .clone()
            .cast_to(&DataType::Utf8, None)?
            .into_array(num_iterations)?;
        let expr_array = as_string_array(&expr_array)?;

        let options_array = if self.arg_count >= 3 {
            args[2]
                .clone()
                .cast_to(&DataType::Utf8, None)?
                .into_array(num_iterations)?
        } else {
            ScalarValue::Utf8(None).to_array_of_size(num_iterations)?
        };
        let options_array = as_string_array(&options_array)?;

        let mut expr_iter = expr_array.iter();
        let mut options_iter = options_array.iter();

        let mut builder = RasterBuilder::new(num_iterations);

        // The executor only needs the raster argument; expr/options iterate in
        // lockstep alongside it. `finish_over` still considers the full arg list
        // so a per-row expr/options column over a scalar raster yields an array.
        let executor = RasterExecutor::new_with_num_iterations(
            std::slice::from_ref(&arg_types[0]),
            std::slice::from_ref(&args[0]),
            num_iterations,
        );

        executor.execute_raster_void(|_i, raster_opt| {
            let expr_opt = expr_iter.next().flatten();
            let options_opt = options_iter.next().flatten();

            let (Some(raster), Some(expr)) = (raster_opt, expr_opt) else {
                builder.append_null()?;
                return Ok(());
            };

            let options = match options_opt {
                Some(json) => parse_options(json)?,
                None => MapAlgebraOptions::default(),
            };

            evaluate_map_algebra(&mut builder, raster, expr, &options)
        })?;

        let out: ArrayRef = Arc::new(builder.finish()?);
        RasterExecutor::finish_over(args, out)
    }
}

/// Deserialize the JSON `options` argument.
fn parse_options(json: &str) -> Result<MapAlgebraOptions> {
    serde_json::from_str(json)
        .map_err(|e| exec_datafusion_err!("RS_MapAlgebra: invalid options JSON: {e}"))
}

/// Evaluate `expr` over `raster` and append the resulting single-band raster to
/// `builder`.
fn evaluate_map_algebra(
    builder: &mut RasterBuilder,
    raster: &RasterRefImpl<'_>,
    expr: &str,
    options: &MapAlgebraOptions,
) -> Result<()> {
    let compiled = build_operator_tree(expr).map_err(|e| {
        exec_datafusion_err!("RS_MapAlgebra: failed to parse expression {expr:?}: {e}")
    })?;

    // The spatial dims are X-first (`[x, y]`); `width`/`height` come from them.
    let spatial_dims = raster.spatial_dims();
    let spatial_shape = raster.spatial_shape();
    if spatial_dims.len() != 2 || spatial_shape.len() != 2 {
        return exec_err!("RS_MapAlgebra: expected a 2-D raster (spatial dims {spatial_dims:?})");
    }
    let width = usize::try_from(spatial_shape[0])
        .map_err(|_| exec_datafusion_err!("RS_MapAlgebra: invalid raster width"))?;
    let height = usize::try_from(spatial_shape[1])
        .map_err(|_| exec_datafusion_err!("RS_MapAlgebra: invalid raster height"))?;
    let pixel_count = width
        .checked_mul(height)
        .ok_or_else(|| exec_datafusion_err!("RS_MapAlgebra: raster pixel count overflows"))?;

    let bands = raster.bands();
    let num_bands = bands.len();
    if num_bands == 0 {
        return exec_err!("RS_MapAlgebra: raster has no bands");
    }

    // Output pixel type: explicit option, else the first band's data type.
    let out_data_type = match &options.pixel_type {
        Some(pt) => parse_pixel_type(pt)?,
        None => bands
            .band(1)
            .map_err(|e| exec_datafusion_err!("RS_MapAlgebra: failed to read band 1: {e}"))?
            .data_type(),
    };

    // nodata: exact-or-error, never a silent lossy f64 round-trip.
    let nodata_bytes = match options.nodata {
        Some(value) => Some(
            nodata_f64_to_bytes(value, &out_data_type)
                .map_err(|e| exec_datafusion_err!("RS_MapAlgebra: invalid nodata value: {e}"))?,
        ),
        None => None,
    };

    // Bind only the variables the expression actually references. Decode each
    // referenced band once (a band may be referenced by both `rast` and
    // `rast0`), then map every referenced variable name to its band's data.
    let referenced: HashSet<&str> = compiled.iter_variable_identifiers().collect();
    let mut decoded: HashMap<usize, Vec<f64>> = HashMap::new();
    let mut band_bindings: Vec<(String, usize)> = Vec::new();
    for &ident in &referenced {
        let Some(band_index) = parse_band_variable(ident) else {
            continue;
        };
        if band_index >= num_bands {
            return exec_err!(
                "RS_MapAlgebra: expression references band {band_index} \
                 ({ident}) but the raster has {num_bands} band(s)"
            );
        }
        if let Entry::Vacant(entry) = decoded.entry(band_index) {
            let band = bands.band(band_index + 1).map_err(|e| {
                exec_datafusion_err!("RS_MapAlgebra: failed to read band {band_index}: {e}")
            })?;
            entry.insert(decode_band_to_f64(band.as_ref(), pixel_count)?);
        }
        band_bindings.push((ident.to_string(), band_index));
    }

    let use_x = referenced.contains("x");
    let use_y = referenced.contains("y");

    let mut context = HashMapContext::new();
    if referenced.contains("width") {
        set_float(&mut context, "width", width as f64)?;
    }
    if referenced.contains("height") {
        set_float(&mut context, "height", height as f64)?;
    }

    // Evaluate each pixel into a scratch `f64` buffer, then coerce the whole
    // buffer to the output pixel type in one pass. The per-pixel variable binds
    // allocate a fresh key string each call — an artifact of the expression
    // engine's owned-String context API, and the dominant per-pixel cost.
    let mut results = vec![0.0_f64; pixel_count];
    for (idx, slot) in results.iter_mut().enumerate() {
        if use_x {
            set_float(&mut context, "x", (idx % width) as f64)?;
        }
        if use_y {
            set_float(&mut context, "y", (idx / width) as f64)?;
        }
        for (name, band_index) in &band_bindings {
            set_float(&mut context, name, decoded[band_index][idx])?;
        }
        let value = compiled.eval_with_context(&context).map_err(|e| {
            exec_datafusion_err!("RS_MapAlgebra: failed to evaluate expression: {e}")
        })?;
        *slot = value_to_f64(&value)?;
    }

    let out_bytes = encode_results(&results, out_data_type);

    // Build the output raster: same geometry as the input, one derived band.
    // The band is C-order (`[y, x]`) matching the raster's spatial dims.
    let transform: [f64; 6] = raster
        .transform()
        .try_into()
        .map_err(|_| exec_datafusion_err!("RS_MapAlgebra: raster transform is not 6 elements"))?;
    builder
        .start_raster_nd(&transform, &spatial_dims, spatial_shape, raster.crs())
        .map_err(|e| exec_datafusion_err!("RS_MapAlgebra: failed to start raster: {e}"))?;

    let band_dims = [raster.y_dim(), raster.x_dim()];
    let band_shape = [height as i64, width as i64];
    builder
        .start_band_nd(
            None,
            &band_dims,
            &band_shape,
            out_data_type,
            nodata_bytes.as_deref(),
            None,
            None,
        )
        .map_err(|e| exec_datafusion_err!("RS_MapAlgebra: failed to start band: {e}"))?;

    // Move the finished band bytes into the Arrow buffer as a view (a refcount
    // bump) rather than copying them through the builder.
    let len = u32::try_from(out_bytes.len()).map_err(|_| {
        exec_datafusion_err!(
            "RS_MapAlgebra: band data of {} bytes exceeds the binary-view limit",
            out_bytes.len()
        )
    })?;
    let buffer = Buffer::from(out_bytes);
    builder
        .append_band_data_buffer(&buffer, 0, len)
        .map_err(|e| exec_datafusion_err!("RS_MapAlgebra: failed to append band data: {e}"))?;
    builder
        .finish_band()
        .map_err(|e| exec_datafusion_err!("RS_MapAlgebra: failed to finish band: {e}"))?;
    builder
        .finish_raster()
        .map_err(|e| exec_datafusion_err!("RS_MapAlgebra: failed to finish raster: {e}"))?;

    Ok(())
}

/// Parse a band variable identifier (`rast`, `rast0`, `rast1`, ...) into a
/// 0-based band index. `rast` is an alias for `rast0`. Returns `None` for any
/// other identifier (e.g. `x`, `width`).
fn parse_band_variable(ident: &str) -> Option<usize> {
    let suffix = ident.strip_prefix("rast")?;
    if suffix.is_empty() {
        Some(0)
    } else {
        suffix.parse::<usize>().ok()
    }
}

/// Bind `name` to a floating-point value in the evaluation context.
fn set_float(context: &mut HashMapContext, name: &str, value: f64) -> Result<()> {
    context
        .set_value(name.to_string(), Value::Float(value))
        .map_err(|e| exec_datafusion_err!("RS_MapAlgebra: failed to bind variable {name}: {e}"))
}

/// Convert an expression result to `f64`. Booleans map to `1.0`/`0.0`.
fn value_to_f64(value: &Value) -> Result<f64> {
    match value {
        Value::Float(f) => Ok(*f),
        Value::Int(i) => Ok(*i as f64),
        Value::Boolean(b) => Ok(if *b { 1.0 } else { 0.0 }),
        other => exec_err!(
            "RS_MapAlgebra: expression must evaluate to a number or boolean, got {other:?}"
        ),
    }
}

/// Decode a 2-D band's contiguous bytes into one `f64` per pixel, row-major.
fn decode_band_to_f64(band: &dyn BandRef, pixel_count: usize) -> Result<Vec<f64>> {
    if band.ndim() != 2 {
        return exec_err!(
            "RS_MapAlgebra: only 2-D bands are supported, got a {}-D band with dims {:?}",
            band.ndim(),
            band.dim_names()
        );
    }
    let data_type = band.data_type();
    let byte_size = data_type.byte_size();
    let nd_buffer = band
        .nd_buffer()
        .map_err(|e| exec_datafusion_err!("RS_MapAlgebra: failed to read band: {e}"))?;
    let bytes = nd_buffer
        .as_contiguous()
        .map_err(|e| exec_datafusion_err!("RS_MapAlgebra: band is not contiguous: {e}"))?;
    let expected = pixel_count
        .checked_mul(byte_size)
        .ok_or_else(|| exec_datafusion_err!("RS_MapAlgebra: band size overflows"))?;
    if bytes.len() != expected {
        return exec_err!(
            "RS_MapAlgebra: band byte length {} does not match {} pixels of {:?}",
            bytes.len(),
            pixel_count,
            data_type
        );
    }
    Ok(decode_le_f64(bytes, data_type, pixel_count))
}

/// Decode `pixel_count` little-endian values of `data_type` to `f64`. `bytes`
/// must be exactly `pixel_count * data_type.byte_size()` long (validated by the
/// caller), so every slice index below is in bounds.
fn decode_le_f64(bytes: &[u8], data_type: BandDataType, pixel_count: usize) -> Vec<f64> {
    macro_rules! read_le {
        ($ty:ty, $size:literal) => {{
            let mut out = Vec::with_capacity(pixel_count);
            for i in 0..pixel_count {
                let offset = i * $size;
                let mut arr = [0u8; $size];
                arr.copy_from_slice(&bytes[offset..offset + $size]);
                out.push(<$ty>::from_le_bytes(arr) as f64);
            }
            out
        }};
    }
    match data_type {
        BandDataType::UInt8 => bytes[..pixel_count].iter().map(|&b| b as f64).collect(),
        BandDataType::Int8 => bytes[..pixel_count]
            .iter()
            .map(|&b| b as i8 as f64)
            .collect(),
        BandDataType::UInt16 => read_le!(u16, 2),
        BandDataType::Int16 => read_le!(i16, 2),
        BandDataType::UInt32 => read_le!(u32, 4),
        BandDataType::Int32 => read_le!(i32, 4),
        BandDataType::UInt64 => read_le!(u64, 8),
        BandDataType::Int64 => read_le!(i64, 8),
        BandDataType::Float32 => read_le!(f32, 4),
        BandDataType::Float64 => read_le!(f64, 8),
    }
}

/// Coerce per-pixel `f64` results to the little-endian bytes of `data_type`.
///
/// Integer types truncate toward zero and saturate at the type bounds (a C-style
/// cast; NaN maps to 0), matching numpy's `astype` for in-range values. `Float32`
/// narrows to the nearest `f32`. This coercion is intrinsic to map algebra: the
/// expression produces `f64` and the chosen output type stores it.
fn encode_results(results: &[f64], data_type: BandDataType) -> Vec<u8> {
    let mut out = Vec::with_capacity(results.len() * data_type.byte_size());
    macro_rules! write_le {
        ($ty:ty) => {{
            for &value in results {
                out.extend_from_slice(&(value as $ty).to_le_bytes());
            }
        }};
    }
    match data_type {
        BandDataType::UInt8 => write_le!(u8),
        BandDataType::Int8 => write_le!(i8),
        BandDataType::UInt16 => write_le!(u16),
        BandDataType::Int16 => write_le!(i16),
        BandDataType::UInt32 => write_le!(u32),
        BandDataType::Int32 => write_le!(i32),
        BandDataType::UInt64 => write_le!(u64),
        BandDataType::Int64 => write_le!(i64),
        BandDataType::Float32 => write_le!(f32),
        BandDataType::Float64 => write_le!(f64),
    }
    out
}

/// Parse a pixel-type string to a [`BandDataType`], accepting the short GDAL-style
/// codes and the full type names. Mirrors the pixel-type vocabulary of
/// `RS_AsRaster`.
fn parse_pixel_type(value: &str) -> Result<BandDataType> {
    match value.trim().to_ascii_lowercase().as_str() {
        "d" | "float64" => Ok(BandDataType::Float64),
        "f" | "float32" => Ok(BandDataType::Float32),
        "i" | "int32" => Ok(BandDataType::Int32),
        "ui" | "uint32" => Ok(BandDataType::UInt32),
        "s" | "int16" => Ok(BandDataType::Int16),
        "us" | "uint16" => Ok(BandDataType::UInt16),
        "b" | "uint8" => Ok(BandDataType::UInt8),
        "i8" | "int8" => Ok(BandDataType::Int8),
        "u64" | "uint64" => Ok(BandDataType::UInt64),
        "i64" | "int64" => Ok(BandDataType::Int64),
        other => exec_err!(
            "RS_MapAlgebra: unsupported pixel type {other:?} (expected one of \
             B/I8/S/US/I/UI/I64/U64/F/D or the full type name)"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::StringArray;
    use datafusion_expr::ScalarUDF;
    use sedona_testing::raster_spec::{
        assert_raster_scalar_equals, assert_rasters_equal, raster_array, RasterSpec,
    };
    use sedona_testing::testers::ScalarUdfTester;

    /// A 3x2 EPSG:4326 raster with one Float64 band holding values 1..=6
    /// (row-major), world extent x in [0, 3], y in [0, 2].
    fn f64_raster() -> RasterSpec {
        RasterSpec::d2(3, 2)
            .crs(Some("EPSG:4326"))
            .transform([0.0, 1.0, 0.0, 2.0, 0.0, -1.0])
            .band_values(&[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0])
    }

    /// Invoke `RS_MapAlgebra` on a single scalar raster. `options = None` drives
    /// the 2-arg kernel; `Some(json)` drives the 3-arg kernel.
    fn invoke(raster: &RasterSpec, expr: &str, options: Option<&str>) -> Result<ScalarValue> {
        let udf: ScalarUDF = rs_map_algebra_udf().into();
        let result = match options {
            None => {
                let tester =
                    ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Utf8)]);
                tester.invoke_arrays(vec![
                    Arc::new(raster.build()),
                    Arc::new(StringArray::from(vec![expr])),
                ])?
            }
            Some(opts) => {
                let tester = ScalarUdfTester::new(
                    udf,
                    vec![
                        RASTER,
                        SedonaType::Arrow(DataType::Utf8),
                        SedonaType::Arrow(DataType::Utf8),
                    ],
                );
                tester.invoke_arrays(vec![
                    Arc::new(raster.build()),
                    Arc::new(StringArray::from(vec![expr])),
                    Arc::new(StringArray::from(vec![opts])),
                ])?
            }
        };
        ScalarValue::try_from_array(&result, 0)
    }

    #[test]
    fn test_parse_pixel_type() {
        assert_eq!(parse_pixel_type("B").unwrap(), BandDataType::UInt8);
        assert_eq!(parse_pixel_type("i8").unwrap(), BandDataType::Int8);
        assert_eq!(parse_pixel_type("S").unwrap(), BandDataType::Int16);
        assert_eq!(parse_pixel_type("US").unwrap(), BandDataType::UInt16);
        assert_eq!(parse_pixel_type("I").unwrap(), BandDataType::Int32);
        assert_eq!(parse_pixel_type("UI").unwrap(), BandDataType::UInt32);
        assert_eq!(parse_pixel_type("I64").unwrap(), BandDataType::Int64);
        assert_eq!(parse_pixel_type("u64").unwrap(), BandDataType::UInt64);
        assert_eq!(parse_pixel_type("F").unwrap(), BandDataType::Float32);
        assert_eq!(parse_pixel_type("float64").unwrap(), BandDataType::Float64);
        assert!(parse_pixel_type("nope").is_err());
    }

    #[test]
    fn test_parse_band_variable() {
        assert_eq!(parse_band_variable("rast"), Some(0));
        assert_eq!(parse_band_variable("rast0"), Some(0));
        assert_eq!(parse_band_variable("rast3"), Some(3));
        assert_eq!(parse_band_variable("x"), None);
        assert_eq!(parse_band_variable("width"), None);
        assert_eq!(parse_band_variable("rastfoo"), None);
    }

    #[test]
    fn test_value_to_f64() {
        assert_eq!(value_to_f64(&Value::Float(1.5)).unwrap(), 1.5);
        assert_eq!(value_to_f64(&Value::Int(7)).unwrap(), 7.0);
        assert_eq!(value_to_f64(&Value::Boolean(true)).unwrap(), 1.0);
        assert_eq!(value_to_f64(&Value::Boolean(false)).unwrap(), 0.0);
        assert!(value_to_f64(&Value::String("x".into())).is_err());
    }

    #[test]
    fn test_encode_results_truncates_and_saturates() {
        // Integer output truncates toward zero and saturates at the type bounds.
        assert_eq!(
            encode_results(&[3.9, -3.9, 300.0, -1.0], BandDataType::UInt8),
            vec![3u8, 0, 255, 0]
        );
        assert_eq!(
            encode_results(&[3.9, -3.9], BandDataType::Int32)
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect::<Vec<_>>(),
            vec![3, -3]
        );
        // Float output narrows exactly for representable values.
        assert_eq!(
            encode_results(&[1.5, -2.25], BandDataType::Float32)
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect::<Vec<_>>(),
            vec![1.5f32, -2.25]
        );
    }

    #[test]
    fn test_decode_le_f64_round_trips_each_dtype() {
        // Encode known values into a band, decode them back, and confirm the
        // f64 view matches — parameterized over every data type.
        for dtype in [
            BandDataType::UInt8,
            BandDataType::Int8,
            BandDataType::UInt16,
            BandDataType::Int16,
            BandDataType::UInt32,
            BandDataType::Int32,
            BandDataType::UInt64,
            BandDataType::Int64,
            BandDataType::Float32,
            BandDataType::Float64,
        ] {
            let expected = [1.0, 2.0, 3.0, 4.0];
            let bytes = encode_results(&expected, dtype);
            let decoded = decode_le_f64(&bytes, dtype, expected.len());
            assert_eq!(decoded, expected, "round-trip failed for {dtype:?}");
        }
    }

    #[test]
    fn test_map_algebra_scale_and_offset() {
        // `rast0 * 2 + 1` over 1..=6 -> 3, 5, 7, 9, 11, 13. Output Float32 is
        // exact for these integer-valued results, so assert exact equality.
        let scalar = invoke(
            &f64_raster(),
            "rast0 * 2 + 1",
            Some(r#"{"pixel_type": "F"}"#),
        )
        .unwrap();
        assert_raster_scalar_equals(
            &scalar,
            &RasterSpec::d2(3, 2)
                .crs(Some("EPSG:4326"))
                .transform([0.0, 1.0, 0.0, 2.0, 0.0, -1.0])
                .band_values(&[3.0f32, 5.0, 7.0, 9.0, 11.0, 13.0]),
        );
    }

    #[test]
    fn test_map_algebra_defaults_to_input_dtype() {
        // No pixel_type: the output inherits the input band's UInt8 type.
        // `rast0 + 1` over 0..=5 -> 1..=6 (values are the RasterSpec defaults).
        let input = RasterSpec::d2(3, 2)
            .crs(Some("EPSG:4326"))
            .transform([0.0, 1.0, 0.0, 2.0, 0.0, -1.0])
            .band_values(&[0u8, 1, 2, 3, 4, 5]);
        let scalar = invoke(&input, "rast0 + 1", None).unwrap();
        assert_raster_scalar_equals(
            &scalar,
            &RasterSpec::d2(3, 2)
                .crs(Some("EPSG:4326"))
                .transform([0.0, 1.0, 0.0, 2.0, 0.0, -1.0])
                .band_values(&[1u8, 2, 3, 4, 5, 6]),
        );
    }

    #[test]
    fn test_map_algebra_records_nodata() {
        // The nodata option is recorded on the output band metadata.
        let scalar = invoke(
            &f64_raster(),
            "rast0",
            Some(r#"{"pixel_type": "D", "nodata": 0.0}"#),
        )
        .unwrap();
        assert_raster_scalar_equals(
            &scalar,
            &RasterSpec::d2(3, 2)
                .crs(Some("EPSG:4326"))
                .transform([0.0, 1.0, 0.0, 2.0, 0.0, -1.0])
                .band_values(&[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0])
                .nodata(0.0f64),
        );
    }

    #[test]
    fn test_map_algebra_pixel_coordinates() {
        // `x + y * width` numbers pixels 0..=5 in row-major order, exercising the
        // x/y/width variables. Output Int32.
        let scalar = invoke(
            &f64_raster(),
            "x + y * width",
            Some(r#"{"pixel_type": "I"}"#),
        )
        .unwrap();
        assert_raster_scalar_equals(
            &scalar,
            &RasterSpec::d2(3, 2)
                .crs(Some("EPSG:4326"))
                .transform([0.0, 1.0, 0.0, 2.0, 0.0, -1.0])
                .band_values(&[0i32, 1, 2, 3, 4, 5]),
        );
    }

    #[test]
    fn test_map_algebra_two_band_ndvi() {
        // NDVI (rast1 - rast0) / (rast1 + rast0) with red=1, nir=3 everywhere ->
        // (3-1)/(3+1) = 0.5 for every pixel. Exact in Float64.
        let input = RasterSpec::d2(2, 1)
            .crs(Some("EPSG:4326"))
            .transform([0.0, 1.0, 0.0, 1.0, 0.0, -1.0])
            .band_values(&[1.0f64, 1.0])
            .band_values(&[3.0f64, 3.0]);
        let scalar = invoke(
            &input,
            "(rast1 - rast0) / (rast1 + rast0)",
            Some(r#"{"pixel_type": "D"}"#),
        )
        .unwrap();
        assert_raster_scalar_equals(
            &scalar,
            &RasterSpec::d2(2, 1)
                .crs(Some("EPSG:4326"))
                .transform([0.0, 1.0, 0.0, 1.0, 0.0, -1.0])
                .band_values(&[0.5f64, 0.5]),
        );
    }

    #[test]
    fn test_map_algebra_null_raster_is_null() {
        // A NULL raster input yields a NULL output row.
        let udf: ScalarUDF = rs_map_algebra_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Utf8)]);
        let result = tester
            .invoke_arrays(vec![
                Arc::new(raster_array([None])),
                Arc::new(StringArray::from(vec!["rast0 * 2"])),
            ])
            .unwrap();
        assert_rasters_equal(&result, &[None]);
    }

    #[test]
    fn test_map_algebra_bad_expression_errors() {
        let err = invoke(&f64_raster(), "rast0 * (", None)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("failed to parse expression"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn test_map_algebra_band_out_of_range_errors() {
        // The single-band raster has no band 5.
        let err = invoke(&f64_raster(), "rast5 + 1", None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("references band 5"), "unexpected: {err}");
    }

    #[test]
    fn test_map_algebra_bad_pixel_type_errors() {
        let err = invoke(&f64_raster(), "rast0", Some(r#"{"pixel_type": "nope"}"#))
            .unwrap_err()
            .to_string();
        assert!(err.contains("unsupported pixel type"), "unexpected: {err}");
    }

    #[test]
    fn test_map_algebra_lossy_nodata_errors() {
        // 0.5 is not exactly representable in a UInt8 band, so recording it as
        // nodata is an error rather than a silent truncation.
        let input = RasterSpec::d2(2, 1)
            .crs(Some("EPSG:4326"))
            .transform([0.0, 1.0, 0.0, 1.0, 0.0, -1.0])
            .band_values(&[1u8, 2]);
        let err = invoke(&input, "rast0", Some(r#"{"nodata": 0.5}"#))
            .unwrap_err()
            .to_string();
        assert!(err.contains("invalid nodata value"), "unexpected: {err}");
    }

    #[test]
    fn test_map_algebra_unknown_option_errors() {
        let err = invoke(&f64_raster(), "rast0", Some(r#"{"bogus": 1}"#))
            .unwrap_err()
            .to_string();
        assert!(err.contains("invalid options JSON"), "unexpected: {err}");
    }

    #[test]
    fn test_udf_name() {
        let udf: ScalarUDF = rs_map_algebra_udf().into();
        assert_eq!(udf.name(), "rs_mapalgebra");
    }
}
