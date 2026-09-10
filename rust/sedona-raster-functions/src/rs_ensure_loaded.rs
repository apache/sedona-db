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

use std::collections::HashMap;
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
use sedona_raster::builder::{RasterBuilder, RasterOverrides};
use sedona_raster::raster_loader::{
    AsyncRasterLoader, RasterLoadRequest, RasterLoadResult, RasterLoaderConfig,
    RasterLoaderRegistry,
};
use sedona_raster::traits::{BandOverrides, Override, RasterRef};
use sedona_raster::view_entries::ViewEntries;
use sedona_schema::raster::BandDataType;

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

/// Where a band's bytes come from once the batch has been planned.
enum BandBytes {
    /// InDb: the input band is derived into the output as-is — metadata,
    /// view, and a zero-copy share of its bytes — via [`BandRef::copy_into`].
    InDb,
    /// OutDb: result `pos` of the bundled `load()` issued for loader `group`.
    OutDb { group: usize, pos: usize },
}

/// One OutDb band's load request, owned so it can be assembled before any
/// `await` (keeping the future straightforwardly `Send`) and borrowed by
/// the [`RasterLoadRequest`] slice for the duration of the loader call.
struct LoadRequestPlan {
    uri: String,
    dim_names: Vec<String>,
    source_shape: Vec<i64>,
    view: ViewEntries,
    data_type: BandDataType,
}

/// One bundled `load()` call: every OutDb band of the batch whose
/// `outdb_format` resolved to `loader`, in `(raster_idx, band_idx)` order.
struct LoaderGroup {
    loader: Arc<dyn AsyncRasterLoader>,
    requests: Vec<LoadRequestPlan>,
}

/// `RasterLoadRequest::dim_names` is parallel to the *source* shape, while
/// [`BandRef::dim_names`] names the *visible* axes. The two only differ when
/// the view permutes axes: map each visible name back to the source axis it
/// reads from, falling back to visible order if the view is not a
/// permutation of the source axes.
fn source_order_dim_names(visible: &[&str], view: &ViewEntries) -> Vec<String> {
    let visible_order = || visible.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    if view.len() != visible.len() {
        return visible_order();
    }
    let mut by_source: Vec<Option<String>> = vec![None; visible.len()];
    for (name, entry) in visible.iter().zip(view.iter()) {
        let Some(slot) = usize::try_from(entry.source_axis)
            .ok()
            .and_then(|axis| by_source.get_mut(axis))
        else {
            return visible_order();
        };
        if slot.is_some() {
            return visible_order();
        }
        *slot = Some(name.to_string());
    }
    by_source.into_iter().flatten().collect()
}

/// Resolve OutDb bands in `input` and return a new raster StructArray with
/// `data` populated.
///
/// Three passes:
///
/// 1. **Plan.** Walk every raster/band and assign each OutDb band to a
///    loader group — one group per distinct `outdb_format` (a batch can mix
///    `None` → GDAL, `"zarr"`, and extension loaders). Each format is
///    resolved through `lookup` once per call, not once per band.
/// 2. **Load.** Issue **one** [`AsyncRasterLoader::load`] per group carrying
///    every request for that loader. Backends that batch (one blocking
///    hop, one store/array open, one Python call) only see the win when
///    the caller hands them the whole batch, so this is where the
///    bundling happens. Groups are awaited in turn; fan-out *within* a
///    group is the loader's job.
/// 3. **Build.** Walk the rasters again and assemble the output: InDb
///    bands are derived as-is (zero-copy, view preserved); OutDb bands
///    take the loaded bytes with the layout the loader reported.
///
/// Peak memory is unchanged from a per-band dispatch: the output column
/// holds every loaded buffer at once either way.
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

    // ---- Pass 1: plan -------------------------------------------------
    let mut bands_by_raster: Vec<Option<Vec<BandBytes>>> = Vec::with_capacity(rasters.len());
    let mut groups: Vec<LoaderGroup> = Vec::new();
    // Each distinct `outdb_format` is resolved through the registry once per
    // call, and groups are keyed on the *resolved loader*: two formats that
    // resolve to the same backend (e.g. `None` and "geotiff" both landing on
    // the GDAL catch-all) share one `load()`.
    let mut loader_by_format: HashMap<Option<String>, Arc<dyn AsyncRasterLoader>> = HashMap::new();
    let mut group_by_loader: HashMap<usize, usize> = HashMap::new();

    for raster_idx in 0..rasters.len() {
        if rasters.is_null(raster_idx) {
            bands_by_raster.push(None);
            continue;
        }

        let raster = rasters.get(raster_idx).map_err(|e| {
            sedona_internal_datafusion_err!(
                "RS_EnsureLoaded: bad input raster row {raster_idx}: {e}"
            )
        })?;

        let mut bands = Vec::with_capacity(raster.num_bands());
        for band_idx in 0..raster.num_bands() {
            let band = raster.band(band_idx).map_err(|e| {
                sedona_internal_datafusion_err!(
                    "RS_EnsureLoaded: bad input band ({raster_idx},{band_idx}): {e}"
                )
            })?;
            if band.is_indb() {
                bands.push(BandBytes::InDb);
                continue;
            }

            let uri = band.outdb_uri().ok_or_else(|| {
                sedona_internal_datafusion_err!(
                    "RS_EnsureLoaded: OutDb band ({raster_idx},{band_idx}) has empty data \
                     but no outdb_uri set"
                )
            })?;
            // `outdb_format` may be unset (None) — e.g. RS_FromPath emits
            // null and relies on the catch-all GDAL loader. The registry
            // resolves None against each loader's `supports_format`.
            let outdb_format = band.outdb_format().map(str::to_string);
            let loader = match loader_by_format.get(&outdb_format) {
                Some(loader) => Arc::clone(loader),
                None => {
                    let loader = lookup(outdb_format.as_deref())?;
                    loader_by_format.insert(outdb_format, Arc::clone(&loader));
                    loader
                }
            };
            let loader_id = Arc::as_ptr(&loader) as *const () as usize;
            let group = *group_by_loader.entry(loader_id).or_insert_with(|| {
                groups.push(LoaderGroup {
                    loader,
                    requests: Vec::new(),
                });
                groups.len() - 1
            });
            let requests = &mut groups[group].requests;
            bands.push(BandBytes::OutDb {
                group,
                pos: requests.len(),
            });
            requests.push(LoadRequestPlan {
                uri: uri.to_string(),
                dim_names: source_order_dim_names(&band.dim_names(), band.view()),
                // Raw source extent plus the band's view: a loader may honour
                // the view and return only the visible region, or ignore it
                // and return the full source. Its `RasterLoadResult` says
                // which, and pass 3 builds the output band from that.
                source_shape: band.raw_source_shape().to_vec(),
                view: band.view().clone(),
                data_type: band.data_type(),
            });
        }
        bands_by_raster.push(Some(bands));
    }

    // ---- Pass 2: load, one call per loader -----------------------------
    let mut group_results: Vec<Vec<RasterLoadResult>> = Vec::with_capacity(groups.len());
    for group in &groups {
        // `RasterLoadRequest` borrows; keep the borrowed `&str` dim names
        // alive alongside the requests for the duration of the call.
        let dim_name_refs: Vec<Vec<&str>> = group
            .requests
            .iter()
            .map(|plan| plan.dim_names.iter().map(String::as_str).collect())
            .collect();
        let requests: Vec<RasterLoadRequest<'_>> = group
            .requests
            .iter()
            .zip(&dim_name_refs)
            .map(|(plan, dim_names)| RasterLoadRequest {
                uri: &plan.uri,
                dim_names,
                source_shape: &plan.source_shape,
                view: &plan.view,
                data_type: plan.data_type,
            })
            .collect();
        let request_refs: Vec<&RasterLoadRequest<'_>> = requests.iter().collect();

        // `load` returns one result per request, in request order (every
        // backend does; the count is checked below).
        let results = group.loader.load(&request_refs).await.map_err(|e| {
            sedona_internal_datafusion_err!(
                "RS_EnsureLoaded: loader '{}' failed loading {} band(s): {e}",
                group.loader.name(),
                request_refs.len()
            )
        })?;
        if results.len() != request_refs.len() {
            return sedona_internal_err!(
                "RS_EnsureLoaded: loader '{}' returned {} result(s) for {} request(s)",
                group.loader.name(),
                results.len(),
                request_refs.len()
            );
        }
        group_results.push(results);
    }

    // ---- Pass 3: build ------------------------------------------------
    let mut builder = RasterBuilder::new(rasters.len());
    for (raster_idx, bands) in bands_by_raster.iter().enumerate() {
        let Some(bands) = bands else {
            builder.append_null().map_err(|e| {
                sedona_internal_datafusion_err!("RS_EnsureLoaded: append_null failed: {e}")
            })?;
            continue;
        };

        let raster = rasters.get(raster_idx).map_err(|e| {
            sedona_internal_datafusion_err!(
                "RS_EnsureLoaded: bad input raster row {raster_idx}: {e}"
            )
        })?;
        builder
            .start_raster_from(&raster, RasterOverrides::default())
            .map_err(|e| {
                sedona_internal_datafusion_err!(
                    "RS_EnsureLoaded: start_raster_from failed at row {raster_idx}: {e}"
                )
            })?;

        for (band_idx, bytes) in bands.iter().enumerate() {
            let band = raster.band(band_idx).map_err(|e| {
                sedona_internal_datafusion_err!(
                    "RS_EnsureLoaded: bad input band ({raster_idx},{band_idx}): {e}"
                )
            })?;
            match *bytes {
                BandBytes::InDb => {
                    band.copy_into(&mut builder, BandOverrides::default())
                        .map_err(|e| {
                            sedona_internal_datafusion_err!(
                                "RS_EnsureLoaded: InDb passthrough failed at \
                                 ({raster_idx},{band_idx}): {e}"
                            )
                        })?;
                }
                BandBytes::OutDb { group, pos } => {
                    let result = &group_results[group][pos];
                    // `result.source_shape` + `result.view` describe the bytes
                    // the loader actually produced (full source with the
                    // band's view echoed, or the visible region under an
                    // identity view), so the output band carries exactly that
                    // pair. Validate the byte count against it so an
                    // under-sized loader output surfaces here, not as garbage
                    // bytes downstream.
                    let expected_bytes = result
                        .source_shape
                        .iter()
                        .try_fold(1i64, |acc, &d| acc.checked_mul(d))
                        .and_then(|elems| elems.checked_mul(band.data_type().byte_size() as i64))
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
                    // Whichever layout the loader chose, it must not change
                    // what the band exposes. `finish_raster` only validates
                    // the spatial axes against the raster, so check the whole
                    // visible shape here.
                    let visible = result.view.visible_shape();
                    if visible != band.shape() {
                        return sedona_internal_err!(
                            "RS_EnsureLoaded: band ({raster_idx},{band_idx}) loader returned \
                             visible shape {visible:?}, expected {:?}",
                            band.shape()
                        );
                    }
                    // Derive the output band from the input band, swapping in
                    // the layout the loader produced and its bytes (shared
                    // zero-copy into the output `BinaryView` column).
                    band.copy_into(
                        &mut builder,
                        BandOverrides {
                            source_shape: Some(&result.source_shape),
                            view: Override::Set(&result.view),
                            data: Override::Set(&result.bytes),
                            ..Default::default()
                        },
                    )
                    .map_err(|e| {
                        sedona_internal_datafusion_err!(
                            "RS_EnsureLoaded: OutDb derive failed at \
                             ({raster_idx},{band_idx}): {e}"
                        )
                    })?;
                }
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
    use sedona_raster::view_entries::ViewEntry;
    use std::sync::Mutex;

    use arrow_array::Array;
    use arrow_buffer::Buffer;
    use sedona_raster::array::RasterStructArray;
    use sedona_raster::builder::{RasterBuilder, StartBandArgs};
    use sedona_raster::raster_loader::{RasterLoadResult, RasterLoaderRegistry};
    use sedona_raster::traits::RasterRef;
    use sedona_schema::raster::BandDataType;

    /// `(uri, source_shape, data_type, dim_names)` of one recorded request.
    type SeenRequest = (String, Vec<i64>, BandDataType, Vec<String>);

    /// Records load requests and returns a deterministic byte pattern.
    #[derive(Debug, Default)]
    struct RecordingLoader {
        seen: Mutex<Vec<SeenRequest>>,
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
                    req.dim_names.iter().map(|s| s.to_string()).collect(),
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
        b.start_band(StartBandArgs {
            name: Some("band0"),
            outdb_uri: Some(uri),
            outdb_format: Some(format),
            ..StartBandArgs::new(&["y", "x"], source_shape, BandDataType::UInt8)
        })
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
        b.start_band(StartBandArgs {
            name: Some("band0"),
            ..StartBandArgs::new(&["y", "x"], source_shape, BandDataType::UInt8)
        })
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

    /// `[4, 8]` source (32 bytes, so block-backed rather than inline) with
    /// the view selecting only source row 1, so the band's visible shape is
    /// `[1, 8]` and the raster's spatial shape matches it.
    fn row1_view() -> ViewEntries {
        ViewEntries::new(vec![
            ViewEntry {
                source_axis: 0,
                start: 1,
                step: 1,
                steps: 1,
            },
            ViewEntry {
                source_axis: 1,
                start: 0,
                step: 1,
                steps: 8,
            },
        ])
    }

    /// One-row raster whose single `[4, 8]` UInt8 band carries [`row1_view`].
    /// `data` is `Some` for an InDb band, `None` for an OutDb one.
    fn build_viewed_input(data: Option<&[u8]>) -> ArrayRef {
        let view = row1_view();
        let mut b = RasterBuilder::new(1);
        b.start_raster_nd(&[0.0, 1.0, 0.0, 0.0, 0.0, -1.0], &["y", "x"], &[1, 8], None)
            .unwrap();
        b.start_band(StartBandArgs {
            name: Some("band0"),
            view: Some(&view),
            outdb_uri: data.is_none().then_some("file:///tmp/foo.tif"),
            outdb_format: data.is_none().then_some("mock"),
            ..StartBandArgs::new(&["y", "x"], &[4, 8], BandDataType::UInt8)
        })
        .unwrap();
        b.band_data_writer().append_value(data.unwrap_or(&[]));
        b.finish_band().unwrap();
        b.finish_raster().unwrap();
        Arc::new(b.finish().unwrap())
    }

    #[tokio::test]
    async fn ensure_loaded_keeps_view_when_loader_returns_full_source() {
        // The loader ignores the view and returns the whole `[2, 3]` source
        // (`RasterLoadResult::unresolved` echoes the request's view). The
        // output band must keep the band's view over the full bytes.
        let input = build_viewed_input(None);
        let loader: Arc<RecordingLoader> = Arc::new(RecordingLoader::default());
        let reg = registry_with(loader.clone());

        let out = ensure_loaded(&input, |fmt| reg.read().unwrap().get_or_error(fmt))
            .await
            .unwrap();

        let out_struct = out.as_any().downcast_ref::<StructArray>().unwrap();
        let out_rasters = RasterStructArray::try_new(out_struct).unwrap();
        let r = out_rasters.get(0).unwrap();
        let band = r.band(0).unwrap();
        assert_eq!(band.shape(), &[1, 8]);
        assert_eq!(band.raw_source_shape(), &[4, 8]);
        assert_eq!(band.view(), &row1_view());
        assert!(band.is_indb());
        // Full source bytes were kept; the view selects row 1 of them.
        let full: Vec<u8> = (0..32).collect();
        assert_eq!(band.nd_buffer().unwrap().buffer, full.as_slice());
        assert_eq!(
            band.nd_buffer().unwrap().as_contiguous().unwrap(),
            &full[8..16]
        );
        // The loader was asked for the raw source under the band's view.
        let seen = loader.seen.lock().unwrap();
        assert_eq!(seen[0].1, vec![4, 8]);
    }

    #[tokio::test]
    async fn ensure_loaded_accepts_loader_resolved_view() {
        // A loader that honours the view returns only the visible region with
        // an identity view over it; the output band is then a plain `[1, 3]`.
        #[derive(Debug)]
        struct ResolvingLoader;
        #[async_trait]
        impl AsyncRasterLoader for ResolvingLoader {
            fn name(&self) -> &str {
                "resolving"
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
                        let visible = req.view.visible_shape();
                        RasterLoadResult {
                            bytes: Buffer::from_vec((100u8..108).collect()),
                            view: ViewEntries::identity_for_shape(&visible),
                            source_shape: visible,
                        }
                    })
                    .collect())
            }
        }

        let input = build_viewed_input(None);
        let reg = registry_with(Arc::new(ResolvingLoader));
        let out = ensure_loaded(&input, |fmt| reg.read().unwrap().get_or_error(fmt))
            .await
            .unwrap();

        let out_struct = out.as_any().downcast_ref::<StructArray>().unwrap();
        let out_rasters = RasterStructArray::try_new(out_struct).unwrap();
        let r = out_rasters.get(0).unwrap();
        let band = r.band(0).unwrap();
        assert_eq!(band.shape(), &[1, 8]);
        assert_eq!(band.raw_source_shape(), &[1, 8]);
        assert!(band.view().is_identity(&[1, 8]));
        assert_eq!(
            band.nd_buffer().unwrap().as_contiguous().unwrap(),
            &(100u8..108).collect::<Vec<u8>>()[..]
        );
    }

    #[tokio::test]
    async fn ensure_loaded_passes_through_indb_band_with_non_identity_view() {
        // 32-byte source (block-backed, so sharing is observable); the view
        // selects row 1 of the `[4, 8]` grid.
        let source: Vec<u8> = (0..32).collect();
        let input = build_viewed_input(Some(&source));
        let in_ptr = {
            let s = input.as_any().downcast_ref::<StructArray>().unwrap();
            let rs = RasterStructArray::try_new(s).unwrap();
            let r = rs.get(0).unwrap();
            let band = r.band(0).unwrap();
            band.nd_buffer().unwrap().buffer.as_ptr()
        };
        let loader: Arc<RecordingLoader> = Arc::new(RecordingLoader::default());
        let reg = registry_with(loader.clone());

        let out = ensure_loaded(&input, |fmt| reg.read().unwrap().get_or_error(fmt))
            .await
            .unwrap();

        let out_struct = out.as_any().downcast_ref::<StructArray>().unwrap();
        let out_rasters = RasterStructArray::try_new(out_struct).unwrap();
        let r = out_rasters.get(0).unwrap();
        let band = r.band(0).unwrap();
        assert_eq!(band.shape(), &[1, 8]);
        assert_eq!(band.raw_source_shape(), &[4, 8]);
        assert_eq!(band.view(), &row1_view());
        assert_eq!(
            band.nd_buffer().unwrap().as_contiguous().unwrap(),
            &source[8..16]
        );
        assert_eq!(
            band.nd_buffer().unwrap().buffer.as_ptr(),
            in_ptr,
            "viewed InDb passthrough must share the source buffer (zero-copy)"
        );
        assert!(loader.seen.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn ensure_loaded_preserves_null_raster_rows() {
        // Build a 2-row input: one OutDb band, one null raster row.
        let mut b = RasterBuilder::new(2);
        b.start_raster_nd(&[0.0, 1.0, 0.0, 0.0, 0.0, -1.0], &["y", "x"], &[2, 3], None)
            .unwrap();
        b.start_band(StartBandArgs {
            name: Some("band0"),
            outdb_uri: Some("file:///tmp/foo.tif"),
            outdb_format: Some("mock"),
            ..StartBandArgs::new(&["y", "x"], &[2, 3], BandDataType::UInt8)
        })
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

    /// Records each `load` call's request URIs and fills every returned
    /// buffer with a marker byte parsed from the URI's `#<n>` suffix, so a
    /// test can verify both bundling (one call carrying every request) and
    /// that results are scattered back to the right band.
    #[derive(Debug)]
    struct MarkerLoader {
        name: String,
        formats: Vec<String>,
        calls: Mutex<Vec<Vec<String>>>,
    }

    impl MarkerLoader {
        fn new(name: &str, formats: &[&str]) -> Self {
            Self {
                name: name.to_string(),
                formats: formats.iter().map(|s| s.to_string()).collect(),
                calls: Mutex::default(),
            }
        }

        fn marker(uri: &str) -> u8 {
            uri.rsplit('#').next().unwrap().parse().unwrap()
        }
    }

    #[async_trait]
    impl AsyncRasterLoader for MarkerLoader {
        fn name(&self) -> &str {
            &self.name
        }
        fn supports_format(&self, format: Option<&str>) -> bool {
            format.is_some_and(|f| self.formats.iter().any(|x| x == f))
        }
        async fn load(
            &self,
            reqs: &[&RasterLoadRequest],
        ) -> Result<Vec<RasterLoadResult>, arrow_schema::ArrowError> {
            self.calls
                .lock()
                .unwrap()
                .push(reqs.iter().map(|r| r.uri.to_string()).collect());
            Ok(reqs
                .iter()
                .map(|req| {
                    let elements: i64 = req.source_shape.iter().product();
                    let len = elements as usize * req.data_type.byte_size();
                    RasterLoadResult::unresolved(
                        Buffer::from_vec(vec![Self::marker(req.uri); len]),
                        req,
                    )
                })
                .collect())
        }
    }

    enum BandSpec {
        InDb(Vec<u8>),
        OutDb { uri: String, format: String },
    }

    fn outdb(format: &str, marker: u8) -> BandSpec {
        BandSpec::OutDb {
            uri: format!("mock://{format}#{marker}"),
            format: format.to_string(),
        }
    }

    /// Build an N-row input of `[2, 3]` UInt8 bands; a `None` row is a null
    /// raster.
    fn build_rows(rows: &[Option<Vec<BandSpec>>]) -> ArrayRef {
        let mut b = RasterBuilder::new(rows.len());
        for row in rows {
            let Some(bands) = row else {
                b.append_null().unwrap();
                continue;
            };
            b.start_raster_nd(&[0.0, 1.0, 0.0, 0.0, 0.0, -1.0], &["y", "x"], &[2, 3], None)
                .unwrap();
            for (i, spec) in bands.iter().enumerate() {
                let name = format!("band{i}");
                let base = StartBandArgs::new(&["y", "x"], &[2, 3], BandDataType::UInt8);
                match spec {
                    BandSpec::InDb(pixels) => {
                        b.start_band(StartBandArgs {
                            name: Some(&name),
                            ..base
                        })
                        .unwrap();
                        b.band_data_writer().append_value(pixels);
                    }
                    BandSpec::OutDb { uri, format } => {
                        b.start_band(StartBandArgs {
                            name: Some(&name),
                            outdb_uri: Some(uri),
                            outdb_format: Some(format),
                            ..base
                        })
                        .unwrap();
                        b.band_data_writer().append_value([0u8; 0]);
                    }
                }
                b.finish_band().unwrap();
            }
            b.finish_raster().unwrap();
        }
        Arc::new(b.finish().unwrap())
    }

    fn band_bytes(out: &ArrayRef, raster_idx: usize, band_idx: usize) -> Vec<u8> {
        let s = out.as_any().downcast_ref::<StructArray>().unwrap();
        let rasters = RasterStructArray::try_new(s).unwrap();
        let r = rasters.get(raster_idx).unwrap();
        let band = r.band(band_idx).unwrap();
        band.nd_buffer().unwrap().as_contiguous().unwrap().to_vec()
    }

    #[tokio::test]
    async fn ensure_loaded_issues_one_load_per_loader_for_the_whole_batch() {
        // 3 rows × 2 OutDb bands of one format → exactly one load() carrying
        // all 6 requests in (row, band) order.
        let input = build_rows(&[
            Some(vec![outdb("mock", 1), outdb("mock", 2)]),
            Some(vec![outdb("mock", 3), outdb("mock", 4)]),
            Some(vec![outdb("mock", 5), outdb("mock", 6)]),
        ]);
        let loader = Arc::new(MarkerLoader::new("marker", &["mock"]));
        let reg = registry_with(loader.clone());

        let out = ensure_loaded(&input, |fmt| reg.read().unwrap().get_or_error(fmt))
            .await
            .unwrap();

        let calls = loader.calls.lock().unwrap();
        assert_eq!(
            calls.len(),
            1,
            "expected one bundled load(), got {}",
            calls.len()
        );
        assert_eq!(
            calls[0],
            (1..=6)
                .map(|m| format!("mock://mock#{m}"))
                .collect::<Vec<_>>()
        );
        drop(calls);
        for (r, b, m) in [
            (0, 0, 1u8),
            (0, 1, 2),
            (1, 0, 3),
            (1, 1, 4),
            (2, 0, 5),
            (2, 1, 6),
        ] {
            assert_eq!(band_bytes(&out, r, b), vec![m; 6], "raster {r} band {b}");
        }
    }

    #[tokio::test]
    async fn ensure_loaded_groups_by_loader_and_scatters_results_in_order() {
        // Two loaders, three formats, interleaved across rows and bands, plus
        // a null row. Each loader is called exactly once, each format is
        // resolved through the registry exactly once, and every band gets
        // its own loader's bytes back.
        // `alpha` claims two formats: both must land in its single call.
        let input = build_rows(&[
            Some(vec![outdb("a", 1), outdb("b", 2)]),
            None,
            Some(vec![outdb("b", 3), outdb("a", 4)]),
            Some(vec![outdb("c", 5)]),
        ]);
        let alpha = Arc::new(MarkerLoader::new("alpha", &["a", "c"]));
        let beta = Arc::new(MarkerLoader::new("beta", &["b"]));
        let mut reg = RasterLoaderRegistry::new();
        reg.register(alpha.clone());
        reg.register(beta.clone());

        let mut lookups = 0usize;
        let out = ensure_loaded(&input, |fmt| {
            lookups += 1;
            reg.get_or_error(fmt)
        })
        .await
        .unwrap();

        assert_eq!(lookups, 3, "each distinct format resolves once per call");
        assert_eq!(
            *alpha.calls.lock().unwrap(),
            vec![vec![
                "mock://a#1".to_string(),
                "mock://a#4".to_string(),
                "mock://c#5".to_string()
            ]],
            "two formats resolving to one loader must share a single load()"
        );
        assert_eq!(
            *beta.calls.lock().unwrap(),
            vec![vec!["mock://b#2".to_string(), "mock://b#3".to_string()]]
        );
        assert!(out.is_null(1));
        for (r, b, m) in [(0, 0, 1u8), (0, 1, 2), (2, 0, 3), (2, 1, 4), (3, 0, 5)] {
            assert_eq!(band_bytes(&out, r, b), vec![m; 6], "raster {r} band {b}");
        }
    }

    #[tokio::test]
    async fn ensure_loaded_mixes_indb_and_outdb_bands_within_a_raster() {
        let pixels: Vec<u8> = (10..16).collect();
        let input = build_rows(&[Some(vec![
            BandSpec::InDb(pixels.clone()),
            outdb("mock", 9),
            BandSpec::InDb(pixels.clone()),
        ])]);
        let loader = Arc::new(MarkerLoader::new("marker", &["mock"]));
        let reg = registry_with(loader.clone());

        let out = ensure_loaded(&input, |fmt| reg.read().unwrap().get_or_error(fmt))
            .await
            .unwrap();

        assert_eq!(
            *loader.calls.lock().unwrap(),
            vec![vec!["mock://mock#9".to_string()]]
        );
        assert_eq!(band_bytes(&out, 0, 0), pixels);
        assert_eq!(band_bytes(&out, 0, 1), vec![9u8; 6]);
        assert_eq!(band_bytes(&out, 0, 2), pixels);
    }

    #[tokio::test]
    async fn ensure_loaded_rejects_loader_returning_wrong_result_count() {
        #[derive(Debug)]
        struct ExtraResultLoader;
        #[async_trait]
        impl AsyncRasterLoader for ExtraResultLoader {
            fn name(&self) -> &str {
                "extra"
            }
            fn supports_format(&self, _format: Option<&str>) -> bool {
                true
            }
            async fn load(
                &self,
                reqs: &[&RasterLoadRequest],
            ) -> Result<Vec<RasterLoadResult>, arrow_schema::ArrowError> {
                // One result too many.
                Ok(reqs
                    .iter()
                    .chain(reqs.iter().take(1))
                    .map(|req| RasterLoadResult::unresolved(Buffer::from_vec(vec![0u8; 6]), req))
                    .collect())
            }
        }

        let input = build_rows(&[Some(vec![outdb("mock", 1)])]);
        let reg = registry_with(Arc::new(ExtraResultLoader));
        let err = ensure_loaded(&input, |fmt| reg.read().unwrap().get_or_error(fmt))
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("returned 2 result(s) for 1 request(s)"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn ensure_loaded_names_loader_and_request_count_on_load_failure() {
        #[derive(Debug)]
        struct FailingLoader;
        #[async_trait]
        impl AsyncRasterLoader for FailingLoader {
            fn name(&self) -> &str {
                "failing"
            }
            fn supports_format(&self, _format: Option<&str>) -> bool {
                true
            }
            async fn load(
                &self,
                _reqs: &[&RasterLoadRequest],
            ) -> Result<Vec<RasterLoadResult>, arrow_schema::ArrowError> {
                Err(arrow_schema::ArrowError::ExternalError("boom".into()))
            }
        }

        let input = build_rows(&[Some(vec![outdb("mock", 1)]), Some(vec![outdb("mock", 2)])]);
        let reg = registry_with(Arc::new(FailingLoader));
        let err = ensure_loaded(&input, |fmt| reg.read().unwrap().get_or_error(fmt))
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("loader 'failing' failed loading 2 band(s)") && err.contains("boom"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn ensure_loaded_rejects_loader_that_changes_a_non_spatial_visible_shape() {
        // A ["t", "y", "x"] band, source [3, 1, 8], view slicing `t` to one
        // step (visible [1, 1, 8]). `finish_raster` only checks y/x, so a
        // loader that hands back the full source under an identity view would
        // silently turn the band into [3, 1, 8] without this guard.
        let view = ViewEntries::new(vec![
            ViewEntry {
                source_axis: 0,
                start: 1,
                step: 1,
                steps: 1,
            },
            ViewEntry {
                source_axis: 1,
                start: 0,
                step: 1,
                steps: 1,
            },
            ViewEntry {
                source_axis: 2,
                start: 0,
                step: 1,
                steps: 8,
            },
        ]);
        let mut b = RasterBuilder::new(1);
        b.start_raster_nd(&[0.0, 1.0, 0.0, 0.0, 0.0, -1.0], &["y", "x"], &[1, 8], None)
            .unwrap();
        b.start_band(StartBandArgs {
            view: Some(&view),
            outdb_uri: Some("file:///tmp/cube.zarr"),
            outdb_format: Some("mock"),
            ..StartBandArgs::new(&["t", "y", "x"], &[3, 1, 8], BandDataType::UInt8)
        })
        .unwrap();
        b.band_data_writer().append_value([0u8; 0]);
        b.finish_band().unwrap();
        b.finish_raster().unwrap();
        let input: ArrayRef = Arc::new(b.finish().unwrap());

        #[derive(Debug)]
        struct IdentityViewLoader;
        #[async_trait]
        impl AsyncRasterLoader for IdentityViewLoader {
            fn name(&self) -> &str {
                "identity-view"
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
                    .map(|req| RasterLoadResult {
                        bytes: Buffer::from_vec(vec![0u8; 24]),
                        source_shape: req.source_shape.to_vec(),
                        view: ViewEntries::identity_for_shape(req.source_shape),
                    })
                    .collect())
            }
        }

        let reg = registry_with(Arc::new(IdentityViewLoader));
        let err = ensure_loaded(&input, |fmt| reg.read().unwrap().get_or_error(fmt))
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("visible shape [3, 1, 8], expected [1, 1, 8]"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn ensure_loaded_sends_source_order_dim_names_for_a_transposed_view() {
        // Source axes are (y=4, x=8); the view transposes them so the band's
        // visible axes are ["x", "y"] with shape [8, 4]. The request must
        // name the axes in *source* order, parallel to `source_shape`.
        let transposed = ViewEntries::new(vec![
            ViewEntry {
                source_axis: 1,
                start: 0,
                step: 1,
                steps: 8,
            },
            ViewEntry {
                source_axis: 0,
                start: 0,
                step: 1,
                steps: 4,
            },
        ]);
        let mut b = RasterBuilder::new(1);
        b.start_raster_nd(&[0.0, 1.0, 0.0, 0.0, 0.0, -1.0], &["y", "x"], &[4, 8], None)
            .unwrap();
        b.start_band(StartBandArgs {
            view: Some(&transposed),
            outdb_uri: Some("file:///tmp/foo.tif"),
            outdb_format: Some("mock"),
            ..StartBandArgs::new(&["x", "y"], &[4, 8], BandDataType::UInt8)
        })
        .unwrap();
        b.band_data_writer().append_value([0u8; 0]);
        b.finish_band().unwrap();
        b.finish_raster().unwrap();
        let input: ArrayRef = Arc::new(b.finish().unwrap());

        let loader: Arc<RecordingLoader> = Arc::new(RecordingLoader::default());
        let reg = registry_with(loader.clone());
        let out = ensure_loaded(&input, |fmt| reg.read().unwrap().get_or_error(fmt))
            .await
            .unwrap();

        let seen = loader.seen.lock().unwrap();
        assert_eq!(seen[0].1, vec![4, 8]);
        assert_eq!(seen[0].3, vec!["y".to_string(), "x".to_string()]);
        drop(seen);
        // The band itself is unchanged: still transposed, still [8, 4].
        let out_struct = out.as_any().downcast_ref::<StructArray>().unwrap();
        let out_rasters = RasterStructArray::try_new(out_struct).unwrap();
        let r = out_rasters.get(0).unwrap();
        let band = r.band(0).unwrap();
        assert_eq!(band.dim_names(), vec!["x", "y"]);
        assert_eq!(band.shape(), &[8, 4]);
        assert_eq!(band.view(), &transposed);
    }

    #[test]
    fn source_order_dim_names_falls_back_to_visible_order_for_non_permutations() {
        // Identity view: unchanged.
        let identity = ViewEntries::identity_for_shape(&[4, 8]);
        assert_eq!(
            source_order_dim_names(&["y", "x"], &identity),
            vec!["y", "x"]
        );
        // Two visible axes reading the same source axis: not a permutation.
        let doubled = ViewEntries::new(vec![
            ViewEntry {
                source_axis: 0,
                start: 0,
                step: 1,
                steps: 4,
            },
            ViewEntry {
                source_axis: 0,
                start: 0,
                step: 1,
                steps: 4,
            },
        ]);
        assert_eq!(
            source_order_dim_names(&["a", "b"], &doubled),
            vec!["a", "b"]
        );
    }

    #[tokio::test]
    async fn ensure_loaded_handles_an_empty_batch() {
        let input: ArrayRef = Arc::new(RasterBuilder::new(0).finish().unwrap());
        let loader: Arc<RecordingLoader> = Arc::new(RecordingLoader::default());
        let reg = registry_with(loader.clone());
        let out = ensure_loaded(&input, |fmt| reg.read().unwrap().get_or_error(fmt))
            .await
            .unwrap();
        assert_eq!(out.len(), 0);
        assert!(loader.seen.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn ensure_loaded_handles_a_raster_with_no_bands() {
        let mut b = RasterBuilder::new(1);
        b.start_raster_nd(&[0.0, 1.0, 0.0, 0.0, 0.0, -1.0], &["y", "x"], &[2, 3], None)
            .unwrap();
        b.finish_raster().unwrap();
        let input: ArrayRef = Arc::new(b.finish().unwrap());
        let loader: Arc<RecordingLoader> = Arc::new(RecordingLoader::default());
        let reg = registry_with(loader.clone());
        let out = ensure_loaded(&input, |fmt| reg.read().unwrap().get_or_error(fmt))
            .await
            .unwrap();
        let out_struct = out.as_any().downcast_ref::<StructArray>().unwrap();
        let out_rasters = RasterStructArray::try_new(out_struct).unwrap();
        assert_eq!(out_rasters.len(), 1);
        assert!(!out.is_null(0));
        assert_eq!(out_rasters.get(0).unwrap().num_bands(), 0);
        assert!(loader.seen.lock().unwrap().is_empty());
    }
}
