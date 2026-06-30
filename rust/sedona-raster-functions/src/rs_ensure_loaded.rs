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

//! `RS_EnsureLoaded(raster) -> raster` — async UDF that materialises
//! the pixel bytes of any OutDb bands in the input raster column.
//!
//! Walks every input row, identifies bands whose `data` column is empty
//! (the schema-OutDb discriminator), groups them by `outdb_format`,
//! dispatches each via the [`RasterLoaderRegistry`] held on `SedonaContext`,
//! and assembles an output `RecordBatch` of the same row count whose
//! `data` columns are populated with the loaded bytes. InDb bands pass
//! through unchanged. Other band/raster metadata is preserved verbatim.

use std::any::Any;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};

use arrow_array::{Array, ArrayRef, StructArray};
use arrow_schema::{DataType, FieldRef};
use async_trait::async_trait;
use datafusion_common::config::ConfigOptions;
use datafusion_common::{plan_err, Result};
use datafusion_expr::async_udf::AsyncScalarUDFImpl;
use datafusion_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility,
};
use sedona_common::{sedona_internal_datafusion_err, sedona_internal_err};
use sedona_raster::array::RasterStructArray;
use sedona_raster::builder::RasterBuilder;
use sedona_raster::raster_loader::{
    AsyncRasterLoader, RasterLoadRequest, RasterLoaderConfig, RasterLoaderRegistry,
};
use sedona_raster::traits::RasterRef;
use sedona_raster::view_entries::ViewEntry;

/// `SedonaScalarUDF` metadata key marking a UDF whose kernels read raster
/// pixel bytes. A raster function sets it (value `"true"`) via
/// `with_metadata`; the `RS_EnsureLoaded` optimizer rule keys off it to
/// decide whether to wrap raster arguments with byte materialisation.
///
/// This crate owns the key. The optimizer rule lives in
/// `sedona-query-planner`, which can't depend on this crate, so it carries
/// a duplicate of the same string literal — keep the two in sync.
pub const NEEDS_PIXELS_METADATA_KEY: &str = "needs_pixels";

/// `SedonaScalarUDF` metadata key marking a UDF whose returned raster is
/// already fully materialised in-database. A raster function sets it (value
/// `"true"`) via `with_metadata`; the `RS_EnsureLoaded` optimizer rule keys
/// off it to skip wrapping an argument that is itself such a call (its result
/// is already loaded, so a wrap would be redundant — and, being async, would
/// nest unhoistably; see apache/datafusion#20031).
///
/// Only set this on functions that guarantee in-database output for loaded
/// input. Like [`NEEDS_PIXELS_METADATA_KEY`], the optimizer rule carries a
/// duplicate of the string literal — keep the two in sync.
pub const RETURNS_BYTES_METADATA_KEY: &str = "returns_bytes";

/// Async UDF that resolves OutDb bands by dispatching through the
/// [`RasterLoaderRegistry`] stashed in `ConfigOptions` as a
/// [`RasterLoaderConfig`] extension. The UDF instance itself is
/// session-agnostic — it pulls the registry handle out of
/// `args.config_options.extensions.get::<RasterLoaderConfig>()` at
/// dispatch time. This matches DataFusion's
/// `AsyncScalarUDFImpl::invoke_async_with_args` surface (only
/// `Arc<ConfigOptions>` is reachable from the async fn) and mirrors how
/// `SedonaRuntime` flows through the session's options.
#[derive(Debug)]
pub struct RsEnsureLoaded {
    signature: Signature,
}

impl Default for RsEnsureLoaded {
    fn default() -> Self {
        Self::new()
    }
}

impl RsEnsureLoaded {
    pub fn new() -> Self {
        Self {
            // `any(1, ...)` accepts whatever single-arg type the caller
            // passes; we validate "argument is a Raster Struct" in
            // `return_type` and at runtime. Using `Signature::any` (vs.
            // `Signature::user_defined`) sidesteps DataFusion's
            // `coerce_types` call path, which `AsyncScalarUDF` doesn't
            // delegate to the inner impl.
            //
            // `Stable` (not `Volatile`) so DataFusion's CSE pass can
            // deduplicate identical RS_EnsureLoaded(col) calls injected
            // by the analyzer rule. Semantic: within a single query the
            // byte materialisation is deterministic for fixed inputs;
            // across queries the underlying storage may change, so the
            // result isn't `Immutable`.
            signature: Signature::any(1, Volatility::Stable),
        }
    }
}

/// Pull the shared registry handle out of a `ConfigOptions`. Returns a
/// helpful error if the [`RasterLoaderConfig`] extension isn't installed
/// — that only happens if a caller bypasses `SedonaContext::new` to
/// build their own session, in which case naming the extension is the
/// right diagnostic.
fn registry_handle_from_config(
    config: &ConfigOptions,
) -> Result<Arc<RwLock<RasterLoaderRegistry>>> {
    config
        .extensions
        .get::<RasterLoaderConfig>()
        .map(|cfg| cfg.registry.handle())
        .ok_or_else(|| {
            sedona_internal_datafusion_err!(
                "RasterLoaderConfig is not registered in this session's ConfigOptions; \
                 RS_EnsureLoaded cannot dispatch without it. Use SedonaContext::new() \
                 or insert the extension manually."
            )
        })
}

fn lookup_loader(
    registry: &Arc<RwLock<RasterLoaderRegistry>>,
    format: Option<&str>,
) -> Result<Arc<dyn AsyncRasterLoader>> {
    let guard = registry.read().map_err(|e| {
        sedona_internal_datafusion_err!("raster loader registry lock poisoned: {e}")
    })?;
    // The registry owns format resolution + the missing-loader diagnostic.
    guard.get_or_error(format)
}

/// True if `view` is the canonical identity over `source_shape` (each visible
/// axis maps to the same source axis, full extent, unit step). An empty view
/// is treated as identity. Used to detect whether a loader returned an
/// already-resolved (identity) layout we can build directly.
fn view_is_identity(view: &[ViewEntry], source_shape: &[i64]) -> bool {
    view.is_empty()
        || (view.len() == source_shape.len()
            && view.iter().enumerate().all(|(i, v)| {
                v.source_axis == i as i64
                    && v.start == 0
                    && v.step == 1
                    && v.steps == source_shape[i]
            }))
}

// One RsEnsureLoaded per session by construction — equality and hash
// are by identity (i.e. by name). DataFusion needs these to deduplicate
// `ScalarUDF` instances in the function registry; the struct holds no
// per-session state of its own.
impl PartialEq for RsEnsureLoaded {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}
impl Eq for RsEnsureLoaded {}
impl Hash for RsEnsureLoaded {
    fn hash<H: Hasher>(&self, state: &mut H) {
        "rs_ensureloaded".hash(state);
    }
}

impl ScalarUDFImpl for RsEnsureLoaded {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "rs_ensureloaded"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        // Never called in practice — `return_field_from_args` below is the
        // authoritative output-type source and carries the raster
        // extension metadata that a bare `DataType` would drop. Provided
        // only to satisfy the trait.
        sedona_internal_err!(
            "RS_EnsureLoaded::return_type should not be called; return_field_from_args is authoritative"
        )
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        // Identity on schema: the output raster has the same fields as the
        // input — only the `data` column's bytes change. Return the input
        // field verbatim so its `"sedona.raster"` extension metadata
        // survives; building a fresh `Field` from the bare `DataType`
        // (as the default `return_type`-based path does) would strip the
        // extension and downstream code would stop recognising the column
        // as a Raster.
        if args.arg_fields.len() != 1 {
            return plan_err!(
                "RS_EnsureLoaded expects exactly one argument, got {}",
                args.arg_fields.len()
            );
        }
        let field = &args.arg_fields[0];
        if !matches!(field.data_type(), DataType::Struct(_)) {
            return plan_err!(
                "RS_EnsureLoaded expects a Raster (Struct) argument, got {}",
                field.data_type()
            );
        }
        Ok(Arc::clone(field))
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        // DataFusion routes async UDFs through `invoke_async_with_args`
        // on the AsyncFuncExec node; this sync entry should never be
        // called for an `AsyncScalarUDF`-wrapped impl.
        sedona_internal_err!(
            "RS_EnsureLoaded is async; AsyncFuncExec should have dispatched to invoke_async_with_args"
        )
    }
}

#[async_trait]
impl AsyncScalarUDFImpl for RsEnsureLoaded {
    /// Materialising OutDb bytes is per-row I/O, so favour larger input
    /// batches over DataFusion's default to amortise loader dispatch and
    /// keep the async pipeline fed.
    fn ideal_batch_size(&self) -> Option<usize> {
        Some(1024)
    }

    async fn invoke_async_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        if args.args.len() != 1 {
            return sedona_internal_err!("RS_EnsureLoaded() expects a single argument");
        }

        let input_array = args.args[0].to_array(args.number_rows)?;
        let registry = registry_handle_from_config(&args.config_options)?;
        let output = ensure_loaded(&input_array, |format| lookup_loader(&registry, format)).await?;
        Ok(ColumnarValue::Array(output))
    }
}

/// Sequentially resolve OutDb bands in `input` and return a new raster
/// StructArray with `data` populated.
///
/// Sequential rather than `buffer_unordered` for the first cut: holding
/// borrows from the input across the `loader.load(...).await` point is
/// tricky enough with one outstanding future that we'd rather extract
/// owned metadata, dispatch, and move on. Parallel fan-out is a follow-up
/// optimisation that doesn't change the trait surface or the registry
/// contract.
async fn ensure_loaded<F>(input_array: &ArrayRef, mut lookup: F) -> Result<ArrayRef>
where
    F: FnMut(Option<&str>) -> Result<Arc<dyn AsyncRasterLoader>>,
{
    let input_struct = input_array
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| {
            sedona_internal_datafusion_err!(
                "RS_EnsureLoaded: expected StructArray input, got {:?}",
                input_array.data_type()
            )
        })?;

    let rasters = RasterStructArray::try_new(input_struct)?;
    // Shared band `data` column of the input; addressed per band via
    // `rasters.band_data_row(..)` for zero-copy InDb passthrough below.
    let band_data_array = rasters.band_data_array();
    let mut builder = RasterBuilder::new(rasters.len());

    for raster_idx in 0..rasters.len() {
        if rasters.is_null(raster_idx) {
            builder.append_null().map_err(|e| {
                sedona_internal_datafusion_err!("RS_EnsureLoaded: append_null failed: {e}")
            })?;
            continue;
        }

        let raster = rasters.get(raster_idx).map_err(|e| {
            sedona_internal_datafusion_err!(
                "RS_EnsureLoaded: bad input raster row {raster_idx}: {e}"
            )
        })?;

        // Owned per-row metadata so the borrows don't span the per-band
        // `await` points further down.
        let transform: [f64; 6] = raster.transform().try_into().map_err(|_| {
            sedona_internal_datafusion_err!(
                "RS_EnsureLoaded: raster row {raster_idx} transform is not 6 elements"
            )
        })?;
        let spatial_dims_owned: Vec<String> = raster
            .spatial_dims()
            .iter()
            .map(|s| s.to_string())
            .collect();
        let spatial_dims: Vec<&str> = spatial_dims_owned.iter().map(String::as_str).collect();
        let spatial_shape: Vec<i64> = raster.spatial_shape().to_vec();
        let crs: Option<String> = raster.crs().map(|s| s.to_string());

        builder
            .start_raster_nd(&transform, &spatial_dims, &spatial_shape, crs.as_deref())
            .map_err(|e| {
                sedona_internal_datafusion_err!(
                    "RS_EnsureLoaded: start_raster_nd failed at row {raster_idx}: {e}"
                )
            })?;

        let num_bands = raster.num_bands();
        for band_idx in 0..num_bands {
            // Extract everything we need from the band as owned data
            // before any `await`, so the future is straightforwardly Send.
            let band_name = raster.band_name(band_idx).map(|s| s.to_string());
            let (
                dim_names_owned,
                source_shape,
                view_owned,
                data_type,
                nodata,
                outdb_uri,
                outdb_format,
                is_indb,
            ) = {
                let band = raster.band(band_idx).map_err(|e| {
                    sedona_internal_datafusion_err!(
                        "RS_EnsureLoaded: bad input band ({raster_idx},{band_idx}): {e}"
                    )
                })?;
                let dim_names_owned: Vec<String> =
                    band.dim_names().iter().map(|s| s.to_string()).collect();
                let source_shape: Vec<i64> = band.raw_source_shape().to_vec();
                // Passed to the loader so it can (eventually) prune I/O; today
                // every loader ignores it and returns the full source.
                let view_owned: Vec<ViewEntry> = band.view().to_vec();
                let data_type = band.data_type();
                let nodata: Option<Vec<u8>> = band.nodata().map(|b| b.to_vec());
                let outdb_uri: Option<String> = band.outdb_uri().map(|s| s.to_string());
                let outdb_format: Option<String> = band.outdb_format().map(|s| s.to_string());
                let is_indb = band.is_indb();
                // A non-identity view can't be passed through yet: the rebuild
                // via `start_band_nd` below emits an identity band, so the view
                // would be silently dropped — the zero-copy passthrough shares
                // the source bytes but does not carry the view. Unreachable
                // today (the read boundary rejects non-identity views), but
                // guard it loudly so that when views land this surfaces as the
                // internal gap it is. Fixed alongside the view-preserving
                // passthrough
                // https://github.com/apache/sedona-db/pull/897
                if is_indb && !view_is_identity(&view_owned, &source_shape) {
                    return sedona_internal_err!(
                        "RS_EnsureLoaded: InDb band ({raster_idx},{band_idx}) has a \
                         non-identity view; view-preserving passthrough is not \
                         implemented yet"
                    );
                }
                (
                    dim_names_owned,
                    source_shape,
                    view_owned,
                    data_type,
                    nodata,
                    outdb_uri,
                    outdb_format,
                    is_indb,
                )
            };

            let dim_names: Vec<&str> = dim_names_owned.iter().map(String::as_str).collect();
            builder
                .start_band_nd(
                    band_name.as_deref(),
                    &dim_names,
                    &source_shape,
                    data_type,
                    nodata.as_deref(),
                    outdb_uri.as_deref(),
                    outdb_format.as_deref(),
                )
                .map_err(|e| {
                    sedona_internal_datafusion_err!(
                        "RS_EnsureLoaded: start_band_nd failed at ({raster_idx},{band_idx}): {e}"
                    )
                })?;

            // Resolve and append the bytes. Both paths are zero-copy: the
            // source/loaded `Buffer` is shared into the output `BinaryView`
            // column by block refcounting rather than re-copied.
            if is_indb {
                // InDb passthrough: share the input row's backing buffer. The
                // view is identity (guarded above), so the rebuilt
                // identity-view output band is byte- and semantics-equivalent.
                let src_row = rasters.band_data_row(raster_idx, band_idx);
                builder
                    .append_band_data_from(band_data_array, src_row)
                    .map_err(|e| {
                        sedona_internal_datafusion_err!(
                            "RS_EnsureLoaded: InDb passthrough failed at \
                             ({raster_idx},{band_idx}): {e}"
                        )
                    })?;
            } else {
                // `outdb_format` may be unset (None) — e.g. RS_FromPath emits
                // null and relies on the catch-all GDAL loader. The registry
                // resolves None against each loader's `supports_format`.
                let format = outdb_format.as_deref();
                let uri = outdb_uri.as_deref().ok_or_else(|| {
                    sedona_internal_datafusion_err!(
                        "RS_EnsureLoaded: OutDb band ({raster_idx},{band_idx}) has empty data \
                         but no outdb_uri set"
                    )
                })?;
                let loader = lookup(format)?;
                let req = RasterLoadRequest {
                    uri,
                    dim_names: &dim_names,
                    source_shape: &source_shape,
                    view: &view_owned,
                    data_type,
                };

                // This is currently loading one request at a time; however, it is more efficient
                // for some loaders to process more than one request at once.
                let result = loader.load(&[&req]).await.map_err(|e| {
                    sedona_internal_datafusion_err!(
                        "RS_EnsureLoaded: loader '{}' failed on \
                         band ({raster_idx},{band_idx}): {e}",
                        loader.name()
                    )
                })?;

                // Check that the loader returned the correct number of results
                if result.len() != 1 {
                    return sedona_internal_err!(
                        "RS_EnsureLoaded: loader {} returned incorrect result count",
                        loader.name()
                    );
                }

                // We can only build identity-view output bands today
                // (`start_band_nd`). A loader that resolved/cropped the view —
                // returning a different `source_shape` or a non-identity
                // `view` — needs `start_band_with_view`. Reserved for
                // https://github.com/apache/sedona-db/issues/897.
                let result = &result[0];
                if result.source_shape != source_shape
                    || !view_is_identity(&result.view, &result.source_shape)
                {
                    return sedona_internal_err!(
                        "RS_EnsureLoaded: band ({raster_idx},{band_idx}) loader returned a \
                         resolved/non-identity view; lazy view resolution is not yet supported"
                    );
                }
                // Validate the loaded length so an under-sized loader output
                // surfaces here, not as garbage bytes downstream.
                let expected_bytes = source_shape
                    .iter()
                    .try_fold(1i64, |acc, &d| acc.checked_mul(d))
                    .and_then(|elems| elems.checked_mul(data_type.byte_size() as i64))
                    .ok_or_else(|| {
                        sedona_internal_datafusion_err!(
                            "RS_EnsureLoaded: band ({raster_idx},{band_idx}) byte count \
                             overflows i64"
                        )
                    })?;
                let got = result.bytes.len();
                if got as i64 != expected_bytes {
                    return sedona_internal_err!(
                        "RS_EnsureLoaded: band ({raster_idx},{band_idx}) expected \
                         {expected_bytes} bytes but loader returned {got}"
                    );
                }
                builder
                    .append_band_data_buffer(&result.bytes, 0, got as u32)
                    .map_err(|e| {
                        sedona_internal_datafusion_err!(
                            "RS_EnsureLoaded: OutDb append failed at \
                             ({raster_idx},{band_idx}): {e}"
                        )
                    })?;
            }
            builder.finish_band().map_err(|e| {
                sedona_internal_datafusion_err!(
                    "RS_EnsureLoaded: finish_band failed at ({raster_idx},{band_idx}): {e}"
                )
            })?;
        }

        builder.finish_raster().map_err(|e| {
            sedona_internal_datafusion_err!(
                "RS_EnsureLoaded: finish_raster failed at row {raster_idx}: {e}"
            )
        })?;
    }

    let output_struct = builder.finish().map_err(|e| {
        sedona_internal_datafusion_err!("RS_EnsureLoaded: builder.finish failed: {e}")
    })?;
    Ok(Arc::new(output_struct) as ArrayRef)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    use arrow_array::Array;
    use arrow_buffer::Buffer;
    use sedona_raster::array::RasterStructArray;
    use sedona_raster::builder::RasterBuilder;
    use sedona_raster::raster_loader::{RasterLoadResult, RasterLoaderRegistry};
    use sedona_raster::traits::RasterRef;
    use sedona_schema::raster::BandDataType;

    /// Records load requests and returns a deterministic byte pattern.
    #[derive(Debug, Default)]
    struct RecordingLoader {
        seen: Mutex<Vec<(String, Vec<i64>, BandDataType)>>,
    }

    #[async_trait]
    impl AsyncRasterLoader for RecordingLoader {
        fn name(&self) -> &str {
            "recording"
        }
        fn supports_format(&self, _format: Option<&str>) -> bool {
            true
        }
        async fn load(
            &self,
            reqs: &[&RasterLoadRequest],
        ) -> Result<Vec<RasterLoadResult>, arrow_schema::ArrowError> {
            let mut results = Vec::with_capacity(reqs.len());
            for req in reqs {
                self.seen.lock().unwrap().push((
                    req.uri.to_string(),
                    req.source_shape.to_vec(),
                    req.data_type,
                ));
                let elements: i64 = req.source_shape.iter().copied().product();
                let len = elements as usize * req.data_type.byte_size();
                // Fill with a recognisable pattern: byte i = (i % 251) as u8.
                let bytes: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
                results.push(RasterLoadResult::unresolved(Buffer::from_vec(bytes), req));
            }
            Ok(results)
        }
    }

    /// Build a 1-row raster with one OutDb band ready for the loader to
    /// materialise.
    fn build_outdb_input(uri: &str, format: &str, source_shape: &[i64]) -> StructArray {
        let mut b = RasterBuilder::new(1);
        b.start_raster_nd(
            &[0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
            &["y", "x"],
            source_shape,
            None,
        )
        .unwrap();
        b.start_band_nd(
            Some("band0"),
            &["y", "x"],
            source_shape,
            BandDataType::UInt8,
            None,
            Some(uri),
            Some(format),
        )
        .unwrap();
        // OutDb bands write empty data.
        b.band_data_writer().append_value([0u8; 0]);
        b.finish_band().unwrap();
        b.finish_raster().unwrap();
        b.finish().unwrap()
    }

    /// Build a 1-row raster with one InDb band — bytes are inline,
    /// `outdb_uri`/`outdb_format` are null.
    fn build_indb_input(source_shape: &[i64], data: &[u8]) -> StructArray {
        let mut b = RasterBuilder::new(1);
        b.start_raster_nd(
            &[0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
            &["y", "x"],
            source_shape,
            None,
        )
        .unwrap();
        b.start_band_nd(
            Some("band0"),
            &["y", "x"],
            source_shape,
            BandDataType::UInt8,
            None,
            None,
            None,
        )
        .unwrap();
        b.band_data_writer().append_value(data);
        b.finish_band().unwrap();
        b.finish_raster().unwrap();
        b.finish().unwrap()
    }

    fn registry_with(loader: Arc<dyn AsyncRasterLoader>) -> Arc<RwLock<RasterLoaderRegistry>> {
        let mut reg = RasterLoaderRegistry::new();
        reg.register(loader);
        Arc::new(RwLock::new(reg))
    }

    /// Regression guard: `RS_EnsureLoaded`'s declared output field must
    /// keep the `"sedona.raster"` extension metadata. If it ever reverts
    /// to a bare-`DataType` return path the output column stops being
    /// recognised as a Raster, and the analyzer rule (which wraps raster
    /// args of needs_bytes UDFs) would both fail to detect already-wrapped
    /// args and break downstream raster kernels reading the result.
    #[test]
    fn return_field_preserves_raster_extension() {
        use datafusion_expr::ReturnFieldArgs;
        use sedona_schema::datatypes::SedonaType;

        let raster_field = SedonaType::Raster.to_storage_field("rast", true).unwrap();
        let arg_fields = [Arc::new(raster_field)];
        let args = ReturnFieldArgs {
            arg_fields: &arg_fields,
            scalar_arguments: &[None],
        };

        let out = RsEnsureLoaded::new().return_field_from_args(args).unwrap();

        // The output must round-trip back to SedonaType::Raster — proving
        // the extension type survived, not just the raw Struct DataType.
        assert!(
            matches!(SedonaType::from_storage_field(&out), Ok(SedonaType::Raster)),
            "output field lost its raster extension: {out:?}"
        );
    }

    #[test]
    fn return_field_rejects_non_raster_arg() {
        use arrow_schema::{DataType, Field};
        use datafusion_expr::ReturnFieldArgs;

        let arg_fields = [Arc::new(Field::new("n", DataType::Int32, true))];
        let args = ReturnFieldArgs {
            arg_fields: &arg_fields,
            scalar_arguments: &[None],
        };
        let err = RsEnsureLoaded::new()
            .return_field_from_args(args)
            .unwrap_err()
            .to_string();
        assert!(err.contains("Raster"), "{err}");
    }

    #[tokio::test]
    async fn ensure_loaded_populates_outdb_band_data() {
        let input_struct = build_outdb_input("file:///tmp/foo.tif", "mock", &[2, 3]);
        let input: ArrayRef = Arc::new(input_struct);

        let loader: Arc<RecordingLoader> = Arc::new(RecordingLoader::default());
        let loader_dyn: Arc<dyn AsyncRasterLoader> = loader.clone();
        let reg = registry_with(loader_dyn);

        let out = ensure_loaded(&input, |fmt| reg.read().unwrap().get_or_error(fmt))
            .await
            .unwrap();

        let out_struct = out.as_any().downcast_ref::<StructArray>().unwrap();
        let out_rasters = RasterStructArray::try_new(out_struct).unwrap();
        assert_eq!(out_rasters.len(), 1);
        let r = out_rasters.get(0).unwrap();
        let band = r.band(0).unwrap();
        // Loader filled 6 bytes (2 × 3 × UInt8) with the (i % 251) pattern.
        assert_eq!(
            band.nd_buffer().unwrap().as_contiguous().unwrap(),
            &[0, 1, 2, 3, 4, 5]
        );
        // outdb_uri / outdb_format are preserved as provenance.
        assert_eq!(band.outdb_uri(), Some("file:///tmp/foo.tif"));
        assert_eq!(band.outdb_format(), Some("mock"));

        // Loader saw one request.
        let seen = loader.seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].0, "file:///tmp/foo.tif");
        assert_eq!(seen[0].1, vec![2, 3]);
        assert_eq!(seen[0].2, BandDataType::UInt8);
    }

    #[tokio::test]
    async fn ensure_loaded_passes_through_indb_bands_without_calling_loader() {
        let pixels: Vec<u8> = (10..16).collect(); // 6 bytes
        let input_struct = build_indb_input(&[2, 3], &pixels);
        let input: ArrayRef = Arc::new(input_struct);

        let loader: Arc<RecordingLoader> = Arc::new(RecordingLoader::default());
        let loader_dyn: Arc<dyn AsyncRasterLoader> = loader.clone();
        let reg = registry_with(loader_dyn);

        let out = ensure_loaded(&input, |fmt| reg.read().unwrap().get_or_error(fmt))
            .await
            .unwrap();

        let out_struct = out.as_any().downcast_ref::<StructArray>().unwrap();
        let out_rasters = RasterStructArray::try_new(out_struct).unwrap();
        let r = out_rasters.get(0).unwrap();
        let band = r.band(0).unwrap();
        assert_eq!(band.nd_buffer().unwrap().as_contiguous().unwrap(), &pixels);

        // Loader was never called.
        assert!(loader.seen.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn ensure_loaded_indb_passthrough_is_zero_copy() {
        // 32 bytes (> the 12-byte inline threshold) so the band data is
        // block-backed and eligible for buffer sharing rather than a copy.
        let pixels: Vec<u8> = (0..32).collect();
        let input_struct = build_indb_input(&[4, 8], &pixels);

        // Pointer to the input band's backing bytes, captured before the input
        // is moved into the call (the backing Buffer is refcounted, so the move
        // doesn't reallocate).
        let in_ptr = {
            let in_rasters = RasterStructArray::try_new(&input_struct).unwrap();
            let r = in_rasters.get(0).unwrap();
            let band = r.band(0).unwrap();
            let ndb = band.nd_buffer().unwrap();
            ndb.as_contiguous().unwrap().as_ptr()
        };

        let input: ArrayRef = Arc::new(input_struct);
        let loader: Arc<RecordingLoader> = Arc::new(RecordingLoader::default());
        let loader_dyn: Arc<dyn AsyncRasterLoader> = loader.clone();
        let reg = registry_with(loader_dyn);

        let out = ensure_loaded(&input, |fmt| reg.read().unwrap().get_or_error(fmt))
            .await
            .unwrap();

        let out_struct = out.as_any().downcast_ref::<StructArray>().unwrap();
        let out_rasters = RasterStructArray::try_new(out_struct).unwrap();
        let r = out_rasters.get(0).unwrap();
        let band = r.band(0).unwrap();
        let out_bytes = band.nd_buffer().unwrap().as_contiguous().unwrap();

        assert_eq!(out_bytes, pixels.as_slice());
        assert_eq!(
            out_bytes.as_ptr(),
            in_ptr,
            "InDb passthrough must share the source buffer (zero-copy), not re-copy"
        );
        assert!(loader.seen.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn ensure_loaded_outdb_append_is_zero_copy() {
        // A loader that hands back a stable, pre-allocated buffer; the output
        // band must reference that same allocation rather than copy it.
        #[derive(Debug)]
        struct StableBufferLoader {
            buffer: Buffer,
        }
        #[async_trait]
        impl AsyncRasterLoader for StableBufferLoader {
            fn name(&self) -> &str {
                "stable"
            }
            fn supports_format(&self, _format: Option<&str>) -> bool {
                true
            }
            async fn load(
                &self,
                req: &[&RasterLoadRequest],
            ) -> Result<Vec<RasterLoadResult>, arrow_schema::ArrowError> {
                assert_eq!(req.len(), 1);
                Ok(vec![RasterLoadResult::unresolved(
                    self.buffer.clone(),
                    req[0],
                )])
            }
        }

        // 32 bytes (> inline threshold) so the loaded buffer is shared.
        let bytes: Vec<u8> = (0..32).collect();
        let buffer = Buffer::from_vec(bytes.clone());
        let loaded_ptr = buffer.as_ptr();

        let input_struct = build_outdb_input("file:///tmp/foo.tif", "mock", &[4, 8]);
        let input: ArrayRef = Arc::new(input_struct);
        let loader_dyn: Arc<dyn AsyncRasterLoader> = Arc::new(StableBufferLoader { buffer });
        let reg = registry_with(loader_dyn);

        let out = ensure_loaded(&input, |fmt| reg.read().unwrap().get_or_error(fmt))
            .await
            .unwrap();

        let out_struct = out.as_any().downcast_ref::<StructArray>().unwrap();
        let out_rasters = RasterStructArray::try_new(out_struct).unwrap();
        let r = out_rasters.get(0).unwrap();
        let band = r.band(0).unwrap();
        let out_bytes = band.nd_buffer().unwrap().as_contiguous().unwrap();

        assert_eq!(out_bytes, bytes.as_slice());
        assert_eq!(
            out_bytes.as_ptr(),
            loaded_ptr,
            "OutDb append must share the loader's buffer (zero-copy), not re-copy"
        );
    }

    #[tokio::test]
    async fn ensure_loaded_errors_when_format_not_registered() {
        let input_struct = build_outdb_input("s3://bucket/foo.zarr", "zarr", &[2, 3]);
        let input: ArrayRef = Arc::new(input_struct);

        let reg: Arc<RwLock<RasterLoaderRegistry>> =
            Arc::new(RwLock::new(RasterLoaderRegistry::new()));

        let err = ensure_loaded(&input, |fmt| reg.read().unwrap().get_or_error(fmt))
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("zarr"),
            "expected error to mention missing format 'zarr', got: {msg}"
        );
    }

    #[tokio::test]
    async fn ensure_loaded_errors_on_undersized_loader_output() {
        let input_struct = build_outdb_input("file:///tmp/foo.tif", "mock", &[2, 3]);
        let input: ArrayRef = Arc::new(input_struct);

        #[derive(Debug, Default)]
        struct ShortLoader;

        #[async_trait]
        impl AsyncRasterLoader for ShortLoader {
            fn name(&self) -> &str {
                "short"
            }
            fn supports_format(&self, _format: Option<&str>) -> bool {
                true
            }
            async fn load(
                &self,
                reqs: &[&RasterLoadRequest],
            ) -> Result<Vec<RasterLoadResult>, arrow_schema::ArrowError> {
                // Return one too few bytes (5 instead of 6) for each request.
                Ok(reqs
                    .iter()
                    .map(|req| RasterLoadResult::unresolved(Buffer::from_vec(vec![0u8; 5]), req))
                    .collect())
            }
        }

        let loader_dyn: Arc<dyn AsyncRasterLoader> = Arc::new(ShortLoader);
        let reg = registry_with(loader_dyn);

        let err = ensure_loaded(&input, |fmt| reg.read().unwrap().get_or_error(fmt))
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("expected") && msg.contains("loader returned"),
            "expected diagnostic about expected vs actual loader bytes, got: {msg}"
        );
    }

    #[tokio::test]
    async fn ensure_loaded_errors_when_loader_reports_non_identity_view() {
        // A loader that reports a non-identity view exercises the deferred
        // #897 path: we can't build a viewed output band yet, so the caller
        // must surface a clear error rather than silently dropping the view.
        let input_struct = build_outdb_input("file:///tmp/foo.tif", "mock", &[2, 3]);
        let input: ArrayRef = Arc::new(input_struct);

        #[derive(Debug, Default)]
        struct ViewReportingLoader;
        #[async_trait]
        impl AsyncRasterLoader for ViewReportingLoader {
            fn name(&self) -> &str {
                "view-reporting"
            }
            fn supports_format(&self, _format: Option<&str>) -> bool {
                true
            }
            async fn load(
                &self,
                reqs: &[&RasterLoadRequest],
            ) -> Result<Vec<RasterLoadResult>, arrow_schema::ArrowError> {
                Ok(reqs
                    .iter()
                    .map(|req| {
                        let elements: i64 = req.source_shape.iter().product();
                        let len = elements as usize * req.data_type.byte_size();
                        RasterLoadResult {
                            bytes: Buffer::from_vec(vec![0u8; len]),
                            source_shape: req.source_shape.to_vec(),
                            // Non-identity: first axis sliced to a single step.
                            view: vec![
                                ViewEntry {
                                    source_axis: 0,
                                    start: 0,
                                    step: 1,
                                    steps: 1,
                                },
                                ViewEntry {
                                    source_axis: 1,
                                    start: 0,
                                    step: 1,
                                    steps: 3,
                                },
                            ],
                        }
                    })
                    .collect())
            }
        }

        let loader_dyn: Arc<dyn AsyncRasterLoader> = Arc::new(ViewReportingLoader);
        let reg = registry_with(loader_dyn);
        let err = ensure_loaded(&input, |fmt| reg.read().unwrap().get_or_error(fmt))
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("lazy view resolution is not yet supported"),
            "expected the deferred view-resolution error, got: {msg}"
        );
    }

    #[tokio::test]
    async fn ensure_loaded_preserves_null_raster_rows() {
        // Build a 2-row input: one OutDb band, one null raster row.
        let mut b = RasterBuilder::new(2);
        b.start_raster_nd(&[0.0, 1.0, 0.0, 0.0, 0.0, -1.0], &["y", "x"], &[2, 3], None)
            .unwrap();
        b.start_band_nd(
            Some("band0"),
            &["y", "x"],
            &[2, 3],
            BandDataType::UInt8,
            None,
            Some("file:///tmp/foo.tif"),
            Some("mock"),
        )
        .unwrap();
        b.band_data_writer().append_value([0u8; 0]);
        b.finish_band().unwrap();
        b.finish_raster().unwrap();
        b.append_null().unwrap();
        let input_struct = b.finish().unwrap();
        let input: ArrayRef = Arc::new(input_struct);

        let loader_dyn: Arc<dyn AsyncRasterLoader> = Arc::new(RecordingLoader::default());
        let reg = registry_with(loader_dyn);

        let out = ensure_loaded(&input, |fmt| reg.read().unwrap().get_or_error(fmt))
            .await
            .unwrap();

        assert_eq!(out.len(), 2);
        assert!(!out.is_null(0));
        assert!(out.is_null(1));
    }
}
