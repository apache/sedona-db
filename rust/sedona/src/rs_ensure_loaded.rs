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
//! dispatches each via the [`OutDbLoaderRegistry`] held on `SedonaContext`,
//! and assembles an output `RecordBatch` of the same row count whose
//! `data` columns are populated with the loaded bytes. InDb bands pass
//! through unchanged. Other band/raster metadata is preserved verbatim.

use std::any::Any;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};

use arrow_array::{Array, ArrayRef, StructArray};
use arrow_buffer::Buffer;
use arrow_schema::DataType;
use async_trait::async_trait;
use datafusion_common::{plan_err, DataFusionError, Result};
use datafusion_expr::async_udf::AsyncScalarUDFImpl;
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility};
use sedona_common::sedona_internal_datafusion_err;
use sedona_raster::array::RasterStructArray;
use sedona_raster::builder::RasterBuilder;
use sedona_raster::outdb_loader::{AsyncByteLoader, OutDbLoadRequest, OutDbLoaderRegistry};
use sedona_raster::traits::RasterRef;

/// Constant exposed so callers (the analyzer rule from a later commit,
/// and tests that need to reference the UDF by name) don't drift on
/// spelling.
pub const RS_ENSURE_LOADED_NAME: &str = "rs_ensureloaded";

/// Async UDF that resolves OutDb bands by dispatching through an
/// [`OutDbLoaderRegistry`]. Owns an `Arc<RwLock<…>>` clone of the
/// session's registry; lookups happen under a brief read lock, then
/// the actual I/O runs without holding the lock.
pub struct RsEnsureLoaded {
    signature: Signature,
    registry: Arc<RwLock<OutDbLoaderRegistry>>,
}

impl RsEnsureLoaded {
    pub fn new(registry: Arc<RwLock<OutDbLoaderRegistry>>) -> Self {
        Self {
            // user_defined signature so DataFusion delegates argument
            // type resolution to `return_field_from_args`; we just want
            // "takes one Raster, returns one Raster"
            signature: Signature::user_defined(Volatility::Volatile),
            registry,
        }
    }

    fn registry_get(&self, format: &str) -> Result<Arc<dyn AsyncByteLoader>> {
        let guard = self
            .registry
            .read()
            .map_err(|e| sedona_internal_datafusion_err!("OutDb registry lock poisoned: {e}"))?;
        if let Some(loader) = guard.get(format) {
            return Ok(loader);
        }
        // Build a diagnostic that lists registered formats so users
        // know what plugins are loaded.
        let registered: Vec<String> = guard.formats().map(String::from).collect();
        let registered_msg = if registered.is_empty() {
            "no OutDb loaders are registered in this session".to_string()
        } else {
            format!("registered formats: {}", registered.join(", "))
        };
        Err(DataFusionError::Plan(format!(
            "no OutDb loader registered for format '{format}' — {registered_msg}"
        )))
    }
}

impl fmt::Debug for RsEnsureLoaded {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RsEnsureLoaded")
            .field("registry", &*self.registry.read().map_err(|_| fmt::Error)?)
            .finish()
    }
}

// One RsEnsureLoaded per session by construction — equality and hash are
// by identity (i.e. by name). DataFusion needs these to deduplicate
// `ScalarUDF` instances in the function registry; the actual `registry`
// field is per-session and shouldn't participate in identity.
impl PartialEq for RsEnsureLoaded {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}
impl Eq for RsEnsureLoaded {}
impl Hash for RsEnsureLoaded {
    fn hash<H: Hasher>(&self, state: &mut H) {
        RS_ENSURE_LOADED_NAME.hash(state);
    }
}

impl ScalarUDFImpl for RsEnsureLoaded {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        RS_ENSURE_LOADED_NAME
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        // Output shape mirrors input — the schema is preserved; only the
        // `data` column's contents change. We require exactly one Struct
        // argument shaped like a raster.
        if arg_types.len() != 1 {
            return plan_err!(
                "RS_EnsureLoaded expects exactly one argument, got {}",
                arg_types.len()
            );
        }
        match &arg_types[0] {
            dt @ DataType::Struct(_) => Ok(dt.clone()),
            other => plan_err!("RS_EnsureLoaded expects a Raster (Struct) argument, got {other}"),
        }
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        // DataFusion routes async UDFs through `invoke_async_with_args`
        // on the AsyncFuncExec node; this sync entry should never be
        // called for an `AsyncScalarUDF`-wrapped impl.
        Err(sedona_internal_datafusion_err!(
            "RS_EnsureLoaded is async; AsyncFuncExec should have dispatched to invoke_async_with_args"
        ))
    }
}

#[async_trait]
impl AsyncScalarUDFImpl for RsEnsureLoaded {
    async fn invoke_async_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let input_array = match args.args.into_iter().next() {
            Some(ColumnarValue::Array(arr)) => arr,
            Some(ColumnarValue::Scalar(_)) => {
                return Err(sedona_internal_datafusion_err!(
                    "RS_EnsureLoaded does not support scalar inputs; pass a column reference"
                ))
            }
            None => {
                return Err(sedona_internal_datafusion_err!(
                    "RS_EnsureLoaded received zero arguments"
                ))
            }
        };

        let output = ensure_loaded(&input_array, &self.registry, |format| {
            self.registry_get(format)
        })
        .await?;

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
async fn ensure_loaded<F>(
    input_array: &ArrayRef,
    _registry: &Arc<RwLock<OutDbLoaderRegistry>>,
    mut lookup: F,
) -> Result<ArrayRef>
where
    F: FnMut(&str) -> Result<Arc<dyn AsyncByteLoader>>,
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

    let rasters = RasterStructArray::new(input_struct);
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
                data_type,
                nodata,
                outdb_uri,
                outdb_format,
                indb_bytes,
            ) = {
                let band = raster.band(band_idx).map_err(|e| {
                    sedona_internal_datafusion_err!(
                        "RS_EnsureLoaded: bad input band ({raster_idx},{band_idx}): {e}"
                    )
                })?;
                let dim_names_owned: Vec<String> =
                    band.dim_names().iter().map(|s| s.to_string()).collect();
                let source_shape: Vec<u64> = band.raw_source_shape().to_vec();
                let data_type = band.data_type();
                let nodata: Option<Vec<u8>> = band.nodata().map(|b| b.to_vec());
                let outdb_uri: Option<String> = band.outdb_uri().map(|s| s.to_string());
                let outdb_format: Option<String> = band.outdb_format().map(|s| s.to_string());
                // For InDb bands, copy bytes into an owned Buffer.
                // `Buffer::from_vec` is zero-copy ownership transfer of
                // the Vec; the per-row clone of `band.data()` itself is
                // the one InDb copy we accept for sequential simplicity.
                // Parallel fan-out + zero-copy borrowing of the input
                // column is a follow-up optimisation.
                let indb_bytes: Option<Buffer> = if band.is_indb() {
                    Some(Buffer::from_vec(band.data().to_vec()))
                } else {
                    None
                };
                (
                    dim_names_owned,
                    source_shape,
                    data_type,
                    nodata,
                    outdb_uri,
                    outdb_format,
                    indb_bytes,
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

            // Resolve the bytes: InDb passes through; OutDb dispatches.
            let resolved: Buffer = if let Some(buf) = indb_bytes {
                buf
            } else {
                let format = outdb_format.as_deref().ok_or_else(|| {
                    sedona_internal_datafusion_err!(
                        "RS_EnsureLoaded: OutDb band ({raster_idx},{band_idx}) has empty data \
                         but no outdb_format set"
                    )
                })?;
                let uri = outdb_uri.as_deref().ok_or_else(|| {
                    sedona_internal_datafusion_err!(
                        "RS_EnsureLoaded: OutDb band ({raster_idx},{band_idx}) has empty data \
                         but no outdb_uri set"
                    )
                })?;
                let loader = lookup(format)?;
                let req = OutDbLoadRequest {
                    uri,
                    dim_names: &dim_names,
                    source_shape: &source_shape,
                    data_type,
                };
                loader.load(&req).await.map_err(|e| {
                    sedona_internal_datafusion_err!(
                        "RS_EnsureLoaded: loader for format '{format}' failed on \
                         band ({raster_idx},{band_idx}): {e}"
                    )
                })?
            };

            // Validate the resolved length so an under-sized loader output
            // surfaces here, not as garbage bytes downstream.
            let expected_bytes = source_shape
                .iter()
                .try_fold(1u64, |acc, &d| acc.checked_mul(d))
                .and_then(|elems| elems.checked_mul(data_type.byte_size() as u64))
                .ok_or_else(|| {
                    sedona_internal_datafusion_err!(
                        "RS_EnsureLoaded: band ({raster_idx},{band_idx}) byte count overflows u64"
                    )
                })?;
            let got = resolved.len();
            if got as u64 != expected_bytes {
                return Err(sedona_internal_datafusion_err!(
                    "RS_EnsureLoaded: band ({raster_idx},{band_idx}) expected {expected_bytes} \
                     bytes but loader returned {got}"
                ));
            }

            builder.band_data_writer().append_value(resolved.as_slice());
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
    use sedona_raster::array::RasterStructArray;
    use sedona_raster::builder::RasterBuilder;
    use sedona_raster::outdb_loader::OutDbLoaderRegistry;
    use sedona_raster::traits::RasterRef;
    use sedona_schema::raster::BandDataType;

    /// Records load requests and returns a deterministic byte pattern.
    #[derive(Default)]
    struct RecordingLoader {
        seen: Mutex<Vec<(String, Vec<u64>, BandDataType)>>,
    }

    #[async_trait]
    impl AsyncByteLoader for RecordingLoader {
        async fn load(
            &self,
            req: &OutDbLoadRequest<'_>,
        ) -> Result<Buffer, arrow_schema::ArrowError> {
            self.seen.lock().unwrap().push((
                req.uri.to_string(),
                req.source_shape.to_vec(),
                req.data_type,
            ));
            let elements: u64 = req.source_shape.iter().copied().product();
            let len = elements as usize * req.data_type.byte_size();
            // Fill with a recognisable pattern: byte i = (i % 251) as u8.
            let bytes: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
            Ok(Buffer::from_vec(bytes))
        }
    }

    /// Build a 1-row raster with one OutDb band ready for the loader to
    /// materialise.
    fn build_outdb_input(uri: &str, format: &str, source_shape: &[u64]) -> StructArray {
        let mut b = RasterBuilder::new(1);
        b.start_raster_nd(
            &[0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
            &["y", "x"],
            &source_shape.iter().map(|&v| v as i64).collect::<Vec<_>>(),
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
    fn build_indb_input(source_shape: &[u64], data: &[u8]) -> StructArray {
        let mut b = RasterBuilder::new(1);
        b.start_raster_nd(
            &[0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
            &["y", "x"],
            &source_shape.iter().map(|&v| v as i64).collect::<Vec<_>>(),
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

    fn registry_with(
        format: &str,
        loader: Arc<dyn AsyncByteLoader>,
    ) -> Arc<RwLock<OutDbLoaderRegistry>> {
        let mut reg = OutDbLoaderRegistry::new();
        reg.register(format, loader);
        Arc::new(RwLock::new(reg))
    }

    #[tokio::test]
    async fn ensure_loaded_populates_outdb_band_data() {
        let input_struct = build_outdb_input("file:///tmp/foo.tif", "mock", &[2, 3]);
        let input: ArrayRef = Arc::new(input_struct);

        let loader: Arc<RecordingLoader> = Arc::new(RecordingLoader::default());
        let loader_dyn: Arc<dyn AsyncByteLoader> = loader.clone();
        let reg = registry_with("mock", loader_dyn);

        let out = ensure_loaded(&input, &reg, |fmt| {
            reg.read()
                .unwrap()
                .get(fmt)
                .ok_or_else(|| datafusion_common::DataFusionError::Plan(format!("no '{fmt}'")))
        })
        .await
        .unwrap();

        let out_struct = out.as_any().downcast_ref::<StructArray>().unwrap();
        let out_rasters = RasterStructArray::new(out_struct);
        assert_eq!(out_rasters.len(), 1);
        let r = out_rasters.get(0).unwrap();
        let band = r.band(0).unwrap();
        // Loader filled 6 bytes (2 × 3 × UInt8) with the (i % 251) pattern.
        assert_eq!(band.data(), &[0, 1, 2, 3, 4, 5]);
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
        let loader_dyn: Arc<dyn AsyncByteLoader> = loader.clone();
        let reg = registry_with("mock", loader_dyn);

        let out = ensure_loaded(&input, &reg, |fmt| {
            reg.read()
                .unwrap()
                .get(fmt)
                .ok_or_else(|| datafusion_common::DataFusionError::Plan(format!("no '{fmt}'")))
        })
        .await
        .unwrap();

        let out_struct = out.as_any().downcast_ref::<StructArray>().unwrap();
        let out_rasters = RasterStructArray::new(out_struct);
        let r = out_rasters.get(0).unwrap();
        let band = r.band(0).unwrap();
        assert_eq!(band.data(), &pixels);

        // Loader was never called.
        assert!(loader.seen.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn ensure_loaded_errors_when_format_not_registered() {
        let input_struct = build_outdb_input("s3://bucket/foo.zarr", "zarr", &[2, 3]);
        let input: ArrayRef = Arc::new(input_struct);

        let reg: Arc<RwLock<OutDbLoaderRegistry>> =
            Arc::new(RwLock::new(OutDbLoaderRegistry::new()));

        let err = ensure_loaded(&input, &reg, |fmt| {
            reg.read().unwrap().get(fmt).ok_or_else(|| {
                datafusion_common::DataFusionError::Plan(format!(
                    "no OutDb loader registered for format '{fmt}'"
                ))
            })
        })
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

        #[derive(Default)]
        struct ShortLoader;

        #[async_trait]
        impl AsyncByteLoader for ShortLoader {
            async fn load(
                &self,
                _req: &OutDbLoadRequest<'_>,
            ) -> Result<Buffer, arrow_schema::ArrowError> {
                // Return one too few bytes (5 instead of 6).
                Ok(Buffer::from_vec(vec![0u8; 5]))
            }
        }

        let loader_dyn: Arc<dyn AsyncByteLoader> = Arc::new(ShortLoader);
        let reg = registry_with("mock", loader_dyn);

        let err = ensure_loaded(&input, &reg, |fmt| {
            reg.read()
                .unwrap()
                .get(fmt)
                .ok_or_else(|| datafusion_common::DataFusionError::Plan(format!("no '{fmt}'")))
        })
        .await
        .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("expected") && msg.contains("loader returned"),
            "expected diagnostic about expected vs actual loader bytes, got: {msg}"
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

        let loader_dyn: Arc<dyn AsyncByteLoader> = Arc::new(RecordingLoader::default());
        let reg = registry_with("mock", loader_dyn);

        let out = ensure_loaded(&input, &reg, |fmt| {
            reg.read()
                .unwrap()
                .get(fmt)
                .ok_or_else(|| datafusion_common::DataFusionError::Plan(format!("no '{fmt}'")))
        })
        .await
        .unwrap();

        assert_eq!(out.len(), 2);
        assert!(!out.is_null(0));
        assert!(out.is_null(1));
    }
}
