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

//! `RS_EnsureContiguous(raster) -> raster` — sync UDF that repacks any
//! strided in-database band into a packed row-major (C-order) layout, so a
//! downstream kernel that hands bytes to a contiguous consumer (GDAL, an
//! export writer) can borrow them zero-copy via
//! [`NdBuffer::as_contiguous`](sedona_raster::traits::NdBuffer::as_contiguous).
//!
//! Repacking is pure CPU, so this UDF is synchronous — unlike the async
//! `RS_EnsureLoaded`, it needs no registry, config extension, or I/O. Each
//! input band takes one of three paths:
//!
//! - a contiguous in-database band (including a contiguous *non-identity*
//!   view, e.g. an outer-axis slice) passes through zero-copy, view intact —
//!   the whole point is for this to be a no-op when nothing needs repacking;
//! - a strided in-database band (a non-unit inner stride, a broadcast, a
//!   permutation, or a reversed axis) is materialized into a fresh packed
//!   buffer with an identity view over its visible shape;
//! - an out-of-database band has no loaded bytes to repack, so it passes
//!   through unchanged. The `RS_EnsureLoaded` normalization runs inside this
//!   one when a kernel needs both, so a contiguous-only kernel never actually
//!   sees an unloaded band — this arm is a defensive identity passthrough.
//!
//! Other band/raster metadata is preserved verbatim.

use std::any::Any;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, StructArray};
use arrow_buffer::Buffer;
use arrow_schema::{DataType, FieldRef};
use datafusion_common::{plan_err, Result};
use datafusion_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility,
};
use sedona_common::{sedona_internal_datafusion_err, sedona_internal_err};
use sedona_raster::array::RasterStructArray;
use sedona_raster::builder::{RasterBuilder, StartBandArgs};
use sedona_raster::traits::{BandOverrides, RasterRef};

/// `SedonaScalarUDF` metadata key marking a UDF whose kernels require their
/// raster's band bytes to be laid out contiguously (they call
/// [`NdBuffer::as_contiguous`](sedona_raster::traits::NdBuffer::as_contiguous),
/// directly or through the GDAL bridge). A raster function sets it (value
/// `"true"`) via `with_metadata`; the `RS_EnsureLoaded` optimizer rule keys off
/// it to wrap raster arguments with `RS_EnsureContiguous`.
///
/// This crate owns the key. The optimizer rule lives in
/// `sedona-query-planner`, which can't depend on this crate, so it carries
/// a duplicate of the same string literal — keep the two in sync.
pub const NEEDS_CONTIGUOUS_METADATA_KEY: &str = "needs_contiguous";

/// Sync UDF that repacks strided in-database bands into a packed row-major
/// layout. Stateless and session-agnostic — repacking is pure CPU, so unlike
/// `RS_EnsureLoaded` it needs no registry or config extension.
#[derive(Debug)]
pub struct RsEnsureContiguous {
    signature: Signature,
}

impl Default for RsEnsureContiguous {
    fn default() -> Self {
        Self::new()
    }
}

impl RsEnsureContiguous {
    pub fn new() -> Self {
        Self {
            // `any(1, ...)` accepts whatever single-arg type the caller passes;
            // "argument is a Raster Struct" is validated in
            // `return_field_from_args` and at runtime.
            //
            // `Stable` (not `Volatile`) so DataFusion's CSE pass can
            // deduplicate identical `RS_EnsureContiguous(col)` calls the
            // optimizer rule injects. Within a single query the repack is
            // deterministic for fixed inputs; across queries the underlying
            // storage may change, so the result isn't `Immutable`.
            signature: Signature::any(1, Volatility::Stable),
        }
    }
}

// One RsEnsureContiguous per session by construction — equality and hash are
// by identity (i.e. by name). DataFusion's `ScalarUDFImpl` requires `Eq + Hash`
// (via its `DynEq`/`DynHash` supertraits) to deduplicate `ScalarUDF` instances
// in the function registry; the struct holds no per-session state of its own.
impl PartialEq for RsEnsureContiguous {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}
impl Eq for RsEnsureContiguous {}
impl Hash for RsEnsureContiguous {
    fn hash<H: Hasher>(&self, state: &mut H) {
        "rs_ensurecontiguous".hash(state);
    }
}

impl ScalarUDFImpl for RsEnsureContiguous {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "rs_ensurecontiguous"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        // Never called in practice — `return_field_from_args` below is the
        // authoritative output-type source and carries the raster extension
        // metadata that a bare `DataType` would drop. Provided only to satisfy
        // the trait.
        sedona_internal_err!(
            "RS_EnsureContiguous::return_type should not be called; return_field_from_args is authoritative"
        )
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        // Identity on schema: the output raster has the same fields as the
        // input — only some bands' `data` bytes are repacked (and, for a
        // materialized band, its view collapses to the identity sentinel).
        // Return the input field verbatim so its `"sedona.raster"` extension
        // metadata survives; building a fresh `Field` from the bare
        // `DataType` would strip the extension and downstream code would stop
        // recognising the column as a Raster.
        if args.arg_fields.len() != 1 {
            return plan_err!(
                "RS_EnsureContiguous expects exactly one argument, got {}",
                args.arg_fields.len()
            );
        }
        let field = &args.arg_fields[0];
        if !matches!(field.data_type(), DataType::Struct(_)) {
            return plan_err!(
                "RS_EnsureContiguous expects a Raster (Struct) argument, got {}",
                field.data_type()
            );
        }
        Ok(Arc::clone(field))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        if args.args.len() != 1 {
            return sedona_internal_err!("RS_EnsureContiguous() expects a single argument");
        }
        let input_array = args.args[0].to_array(args.number_rows)?;
        let output = ensure_contiguous(&input_array)?;
        Ok(ColumnarValue::Array(output))
    }
}

/// Repack strided in-database bands in `input` and return a new raster
/// StructArray of the same row count. Contiguous in-database bands and
/// out-of-database bands pass through unchanged (zero-copy).
fn ensure_contiguous(input_array: &ArrayRef) -> Result<ArrayRef> {
    let input_struct = input_array
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| {
            sedona_internal_datafusion_err!(
                "RS_EnsureContiguous: expected StructArray input, got {:?}",
                input_array.data_type()
            )
        })?;

    let rasters = RasterStructArray::try_new(input_struct)?;
    let mut builder = RasterBuilder::new(rasters.len());

    for raster_idx in 0..rasters.len() {
        if rasters.is_null(raster_idx) {
            builder.append_null().map_err(|e| {
                sedona_internal_datafusion_err!("RS_EnsureContiguous: append_null failed: {e}")
            })?;
            continue;
        }

        let raster = rasters.get(raster_idx).map_err(|e| {
            sedona_internal_datafusion_err!(
                "RS_EnsureContiguous: bad input raster row {raster_idx}: {e}"
            )
        })?;

        // Owned per-row metadata; keeps the borrow plumbing simple and mirrors
        // RS_EnsureLoaded's shape.
        let transform: [f64; 6] = raster.transform().try_into().map_err(|_| {
            sedona_internal_datafusion_err!(
                "RS_EnsureContiguous: raster row {raster_idx} transform is not 6 elements"
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
                    "RS_EnsureContiguous: start_raster_nd failed at row {raster_idx}: {e}"
                )
            })?;

        let num_bands = raster.num_bands();
        for band_idx in 0..num_bands {
            let band_name = raster.band_name(band_idx).map(|s| s.to_string());
            let band = raster.band(band_idx).map_err(|e| {
                sedona_internal_datafusion_err!(
                    "RS_EnsureContiguous: bad input band ({raster_idx},{band_idx}): {e}"
                )
            })?;

            if band.is_indb() {
                let ndb = band.nd_buffer().map_err(|e| {
                    sedona_internal_datafusion_err!(
                        "RS_EnsureContiguous: nd_buffer failed at ({raster_idx},{band_idx}): {e}"
                    )
                })?;
                if ndb.is_contiguous() {
                    // Contiguous in-database band — including a contiguous
                    // non-identity view (an outer-axis slice): share the source
                    // bytes zero-copy and carry the view through unchanged.
                    // Collapsing the view here would defeat the point (this must
                    // be a no-op when nothing needs repacking).
                    band.copy_into(
                        &mut builder,
                        BandOverrides {
                            name: band_name.as_deref(),
                            ..Default::default()
                        },
                    )
                    .map_err(|e| {
                        sedona_internal_datafusion_err!(
                            "RS_EnsureContiguous: contiguous passthrough failed at \
                             ({raster_idx},{band_idx}): {e}"
                        )
                    })?;
                    builder.finish_band().map_err(|e| {
                        sedona_internal_datafusion_err!(
                            "RS_EnsureContiguous: finish_band failed at ({raster_idx},{band_idx}): {e}"
                        )
                    })?;
                } else {
                    // Strided in-database band: repack the visible region into a
                    // packed row-major buffer with an identity view. The packed
                    // bytes ARE the finished output, so build them once and move
                    // the allocation into Arrow (`Buffer::from_vec`) rather than
                    // materialize into reusable scratch and copy out — a move is
                    // strictly cheaper than keeping a scratch buffer alive and
                    // copying each band through it.
                    let dim_names_owned: Vec<String> =
                        band.dim_names().iter().map(|s| s.to_string()).collect();
                    let dim_names: Vec<&str> = dim_names_owned.iter().map(String::as_str).collect();
                    // The packed bytes are the visible region in identity layout,
                    // so the emitted band's source_shape is the VISIBLE shape,
                    // not the original (larger) `raw_source_shape`.
                    let visible_shape: Vec<i64> = band.shape().to_vec();
                    let data_type = band.data_type();
                    let nodata: Option<Vec<u8>> = band.nodata().map(|b| b.to_vec());
                    let outdb_uri: Option<String> = band.outdb_uri().map(|s| s.to_string());
                    let outdb_format: Option<String> = band.outdb_format().map(|s| s.to_string());

                    let packed = Buffer::from_vec(ndb.materialize_contiguous());
                    let len = u32::try_from(packed.len()).map_err(|_| {
                        sedona_internal_datafusion_err!(
                            "RS_EnsureContiguous: band ({raster_idx},{band_idx}) packed length \
                             {} exceeds u32",
                            packed.len()
                        )
                    })?;

                    builder
                        .start_band(StartBandArgs {
                            name: band_name.as_deref(),
                            nodata: nodata.as_deref(),
                            outdb_uri: outdb_uri.as_deref(),
                            outdb_format: outdb_format.as_deref(),
                            // `view: None` — identity over the packed visible shape.
                            ..StartBandArgs::new(&dim_names, &visible_shape, data_type)
                        })
                        .map_err(|e| {
                            sedona_internal_datafusion_err!(
                                "RS_EnsureContiguous: start_band failed at \
                                 ({raster_idx},{band_idx}): {e}"
                            )
                        })?;
                    builder
                        .append_band_data_buffer(&packed, 0, len)
                        .map_err(|e| {
                            sedona_internal_datafusion_err!(
                                "RS_EnsureContiguous: append_band_data_buffer failed at \
                             ({raster_idx},{band_idx}): {e}"
                            )
                        })?;
                    builder.finish_band().map_err(|e| {
                        sedona_internal_datafusion_err!(
                            "RS_EnsureContiguous: finish_band failed at ({raster_idx},{band_idx}): {e}"
                        )
                    })?;
                }
            } else {
                // Out-of-database band: no loaded bytes to repack, so pass the
                // band through with its metadata, view, and outdb hints intact.
                band.copy_into(
                    &mut builder,
                    BandOverrides {
                        name: band_name.as_deref(),
                        ..Default::default()
                    },
                )
                .map_err(|e| {
                    sedona_internal_datafusion_err!(
                        "RS_EnsureContiguous: OutDb passthrough failed at \
                         ({raster_idx},{band_idx}): {e}"
                    )
                })?;
                builder.finish_band().map_err(|e| {
                    sedona_internal_datafusion_err!(
                        "RS_EnsureContiguous: finish_band failed at ({raster_idx},{band_idx}): {e}"
                    )
                })?;
            }
        }

        builder.finish_raster().map_err(|e| {
            sedona_internal_datafusion_err!(
                "RS_EnsureContiguous: finish_raster failed at row {raster_idx}: {e}"
            )
        })?;
    }

    let output_struct = builder.finish().map_err(|e| {
        sedona_internal_datafusion_err!("RS_EnsureContiguous: builder.finish failed: {e}")
    })?;
    Ok(Arc::new(output_struct) as ArrayRef)
}

#[cfg(test)]
mod tests {
    use super::*;

    use sedona_raster::builder::{RasterBuilder, StartBandArgs};
    use sedona_raster::view_entries::ViewEntry;
    use sedona_schema::raster::BandDataType;

    const TRANSFORM: [f64; 6] = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];

    /// Terse `ViewEntry` constructor.
    fn ve(source_axis: i64, start: i64, step: i64, steps: i64) -> ViewEntry {
        ViewEntry {
            source_axis,
            start,
            step,
            steps,
        }
    }

    /// Build a single-raster, single-band `UInt8` array with an explicit view
    /// over `source_shape`. Empty top-level spatial dims impose no
    /// spatial-shape constraint on the (arbitrary) view being exercised.
    fn build_viewed(
        source_shape: &[i64],
        dim_names: &[&str],
        view: &[ViewEntry],
        data: Vec<u8>,
    ) -> StructArray {
        let mut b = RasterBuilder::new(1);
        b.start_raster_nd(&TRANSFORM, &[], &[], None).unwrap();
        b.start_band(StartBandArgs {
            name: Some("band0"),
            view: Some(view),
            ..StartBandArgs::new(dim_names, source_shape, BandDataType::UInt8)
        })
        .unwrap();
        b.band_data_writer().append_value(data);
        b.finish_band().unwrap();
        b.finish_raster().unwrap();
        b.finish().unwrap()
    }

    /// Run the UDF body over a one-row (or multi-row) raster StructArray and
    /// return the output raster array for further assertions.
    fn run(input_struct: StructArray) -> ArrayRef {
        let input: ArrayRef = Arc::new(input_struct);
        ensure_contiguous(&input).unwrap()
    }

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

        let out = RsEnsureContiguous::new()
            .return_field_from_args(args)
            .unwrap();

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
        let err = RsEnsureContiguous::new()
            .return_field_from_args(args)
            .unwrap_err()
            .to_string();
        assert!(err.contains("Raster"), "{err}");
    }

    #[test]
    fn strided_band_is_materialized_to_identity_layout() {
        // Every-other slice of [0..8]: visible values 1, 3, 5. The output band
        // must be packed (as_contiguous succeeds), equal the gather, and carry
        // an identity view whose source_shape is the visible [3], not [8].
        let input = build_viewed(&[8], &["x"], &[ve(0, 1, 2, 3)], (0u8..8).collect());
        let out = run(input);

        let out_struct = out.as_any().downcast_ref::<StructArray>().unwrap();
        let out_rasters = RasterStructArray::try_new(out_struct).unwrap();
        let r = out_rasters.get(0).unwrap();
        let band = r.band(0).unwrap();

        assert_eq!(band.shape(), &[3]);
        // Identity layout: source_shape collapsed to the visible shape.
        assert_eq!(band.raw_source_shape(), &[3]);
        let buf = band.nd_buffer().unwrap();
        assert!(buf.is_contiguous(), "materialized band must be contiguous");
        assert_eq!(buf.offset, 0);
        assert_eq!(buf.as_contiguous().unwrap(), &[1, 3, 5]);
        // Band name preserved.
        assert_eq!(r.band_name(0), Some("band0"));
    }

    #[test]
    fn broadcast_band_is_materialized() {
        // Broadcast source [1, 3] over 5 rows (zero outer stride) → 15 bytes,
        // each visible row [10, 20, 30]. Materialized output is packed.
        let input = build_viewed(
            &[1, 3],
            &["row", "col"],
            &[ve(0, 0, 0, 5), ve(1, 0, 1, 3)],
            vec![10u8, 20, 30],
        );
        let out = run(input);

        let out_struct = out.as_any().downcast_ref::<StructArray>().unwrap();
        let out_rasters = RasterStructArray::try_new(out_struct).unwrap();
        let raster = out_rasters.get(0).unwrap();
        let band = raster.band(0).unwrap();

        assert_eq!(band.shape(), &[5, 3]);
        assert_eq!(band.raw_source_shape(), &[5, 3]);
        let buf = band.nd_buffer().unwrap();
        assert!(buf.is_contiguous());
        assert_eq!(
            buf.as_contiguous().unwrap(),
            &[10, 20, 30, 10, 20, 30, 10, 20, 30, 10, 20, 30, 10, 20, 30]
        );
    }

    #[test]
    fn permuted_and_sliced_band_is_materialized() {
        // 2-D source [Y=4, X=3], data 0..12, viewed as [X, Y] with Y sliced
        // start=1 step=2 steps=2 → the C-order gather is [3, 9, 4, 10, 5, 11].
        let input = build_viewed(
            &[4, 3],
            &["x", "y"],
            &[ve(1, 0, 1, 3), ve(0, 1, 2, 2)],
            (0u8..12).collect(),
        );
        let out = run(input);

        let out_struct = out.as_any().downcast_ref::<StructArray>().unwrap();
        let out_rasters = RasterStructArray::try_new(out_struct).unwrap();
        let raster = out_rasters.get(0).unwrap();
        let band = raster.band(0).unwrap();

        assert_eq!(band.shape(), &[3, 2]);
        let buf = band.nd_buffer().unwrap();
        assert!(buf.is_contiguous());
        assert_eq!(buf.as_contiguous().unwrap(), &[3, 9, 4, 10, 5, 11]);
    }

    #[test]
    fn contiguous_identity_band_is_zero_copy_no_op() {
        // 16-byte identity band (> the 12-byte inline threshold, so block-backed
        // and shareable). The output band must reference the SAME backing buffer
        // — a repack here would be a wasteful copy.
        let input = build_viewed(&[16], &["x"], &[ve(0, 0, 1, 16)], (0u8..16).collect());

        let in_ptr = {
            let in_rasters = RasterStructArray::try_new(&input).unwrap();
            let raster = in_rasters.get(0).unwrap();
            let band = raster.band(0).unwrap();
            band.nd_buffer().unwrap().buffer.as_ptr()
        };

        let out = run(input);
        let out_struct = out.as_any().downcast_ref::<StructArray>().unwrap();
        let out_rasters = RasterStructArray::try_new(out_struct).unwrap();
        let raster = out_rasters.get(0).unwrap();
        let band = raster.band(0).unwrap();
        let buf = band.nd_buffer().unwrap();

        assert!(buf.is_contiguous());
        assert_eq!(buf.as_contiguous().unwrap(), (0u8..16).collect::<Vec<_>>());
        assert_eq!(
            buf.buffer.as_ptr(),
            in_ptr,
            "contiguous passthrough must share the source buffer (zero-copy), not repack"
        );
    }

    #[test]
    fn contiguous_non_identity_view_passes_through_preserving_view() {
        // Outer-axis slice of a [4, 4] source to its first 2 rows: a
        // non-identity view whose bytes are still C-order packed from offset 0.
        // The output must preserve the view AND share the buffer zero-copy — the
        // 16-byte source is block-backed.
        let input = build_viewed(
            &[4, 4],
            &["y", "x"],
            &[ve(0, 0, 1, 2), ve(1, 0, 1, 4)],
            (0u8..16).collect(),
        );

        let in_ptr = {
            let in_rasters = RasterStructArray::try_new(&input).unwrap();
            let raster = in_rasters.get(0).unwrap();
            let band = raster.band(0).unwrap();
            band.nd_buffer().unwrap().buffer.as_ptr()
        };

        let out = run(input);
        let out_struct = out.as_any().downcast_ref::<StructArray>().unwrap();
        let out_rasters = RasterStructArray::try_new(out_struct).unwrap();
        let raster = out_rasters.get(0).unwrap();
        let band = raster.band(0).unwrap();

        // View preserved (not collapsed): visible [2, 4] over source [4, 4].
        assert_eq!(band.view(), &[ve(0, 0, 1, 2), ve(1, 0, 1, 4)]);
        assert_eq!(band.shape(), &[2, 4]);
        assert_eq!(band.raw_source_shape(), &[4, 4]);
        let buf = band.nd_buffer().unwrap();
        assert!(buf.is_contiguous());
        assert_eq!(buf.as_contiguous().unwrap(), (0u8..8).collect::<Vec<_>>());
        assert_eq!(
            buf.buffer.as_ptr(),
            in_ptr,
            "contiguous non-identity view must pass through zero-copy"
        );
    }

    #[test]
    fn multiband_mixes_passthrough_and_materialize() {
        // Two bands sharing a source buffer: band 0 is an identity (contiguous)
        // 16-byte band, band 1 is an every-other strided slice. Band 0 passes
        // through; band 1 is materialized.
        let mut b = RasterBuilder::new(1);
        b.start_raster_nd(&TRANSFORM, &[], &[], None).unwrap();
        // Band 0: identity over [16].
        b.start_band(StartBandArgs {
            name: Some("keep"),
            ..StartBandArgs::new(&["x"], &[16], BandDataType::UInt8)
        })
        .unwrap();
        b.band_data_writer()
            .append_value((0u8..16).collect::<Vec<u8>>());
        b.finish_band().unwrap();
        // Band 1: strided over [8] → visible [4] = indices 0, 2, 4, 6.
        b.start_band(StartBandArgs {
            name: Some("repack"),
            view: Some(&[ve(0, 0, 2, 4)]),
            ..StartBandArgs::new(&["x"], &[8], BandDataType::UInt8)
        })
        .unwrap();
        b.band_data_writer()
            .append_value((0u8..8).collect::<Vec<u8>>());
        b.finish_band().unwrap();
        b.finish_raster().unwrap();
        let input = b.finish().unwrap();

        let out = run(input);
        let out_struct = out.as_any().downcast_ref::<StructArray>().unwrap();
        let out_rasters = RasterStructArray::try_new(out_struct).unwrap();
        let r = out_rasters.get(0).unwrap();
        assert_eq!(r.num_bands(), 2);

        let keep = r.band(0).unwrap();
        assert_eq!(r.band_name(0), Some("keep"));
        assert!(keep.nd_buffer().unwrap().is_contiguous());
        assert_eq!(
            keep.nd_buffer().unwrap().as_contiguous().unwrap(),
            (0u8..16).collect::<Vec<_>>()
        );

        let repack = r.band(1).unwrap();
        assert_eq!(r.band_name(1), Some("repack"));
        assert_eq!(repack.shape(), &[4]);
        assert_eq!(repack.raw_source_shape(), &[4]);
        let buf = repack.nd_buffer().unwrap();
        assert!(buf.is_contiguous());
        assert_eq!(buf.as_contiguous().unwrap(), &[0, 2, 4, 6]);
    }

    #[test]
    fn preserves_null_raster_rows() {
        // 2-row input: one strided band, one null raster row.
        let mut b = RasterBuilder::new(2);
        b.start_raster_nd(&TRANSFORM, &[], &[], None).unwrap();
        b.start_band(StartBandArgs {
            name: Some("band0"),
            view: Some(&[ve(0, 1, 2, 3)]),
            ..StartBandArgs::new(&["x"], &[8], BandDataType::UInt8)
        })
        .unwrap();
        b.band_data_writer()
            .append_value((0u8..8).collect::<Vec<u8>>());
        b.finish_band().unwrap();
        b.finish_raster().unwrap();
        b.append_null().unwrap();
        let input = b.finish().unwrap();

        let out = run(input);
        assert_eq!(out.len(), 2);
        assert!(!out.is_null(0));
        assert!(out.is_null(1));

        let out_struct = out.as_any().downcast_ref::<StructArray>().unwrap();
        let out_rasters = RasterStructArray::try_new(out_struct).unwrap();
        let raster = out_rasters.get(0).unwrap();
        let band = raster.band(0).unwrap();
        assert_eq!(
            band.nd_buffer().unwrap().as_contiguous().unwrap(),
            &[1, 3, 5]
        );
    }
}
