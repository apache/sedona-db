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

//! Zarr group → N-D raster streaming reader.
//!
//! [`ZarrChunkReader`] walks the group's chunk grid lazily and emits one
//! raster row per chunk position, with one band per array in the group.
//! Each row carries an `outdb_uri` chunk anchor
//! (`zarr://<store-uri>/<array-path>#chunk=i0,i1,...`); the `data`
//! column stays empty until the async resolver lands and dereferences
//! the anchor to bytes on demand.

use std::sync::Arc;

use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::{ArrowError, Schema, SchemaRef};
use sedona_common::sedona_internal_datafusion_err;
use sedona_raster::affine_transformation::AffineMatrix;
use sedona_raster::builder::RasterBuilder;
use sedona_raster::traits::is_spatial_dim_pair;
use sedona_schema::datatypes::SedonaType;
use sedona_schema::raster::BandDataType;
use zarrs::array::Array;
#[cfg(test)]
use zarrs::array::ArrayBytes;
use zarrs::group::Group;
use zarrs::node::NodeMetadata;
use zarrs::storage::{AsyncReadableListableStorage, AsyncReadableListableStorageTraits};

use crate::coords;
use crate::dtype::zarr_to_band_data_type;
use crate::geozarr::{crs_from_cf_attributes, geotransform_from_cf_attributes, GroupGeoMetadata};
use crate::source_uri::build_chunk_anchor;

/// Streaming reader over the chunk grid of a Zarr group.
///
/// Each `next()` call emits one `RecordBatch` containing up to
/// `batch_size` rows; one row per chunk position. The reader holds the
/// open group, parsed metadata, and the current chunk-grid position —
/// metadata parsing happens once in [`ZarrChunkReader::try_new`] and
/// the per-row work is just transform arithmetic + anchor URI
/// formatting.
///
/// Rows always emit OutDb-style: `data` is empty, `outdb_uri` carries
/// a chunk anchor that the async raster byte loader (registered separately)
/// resolves to bytes on demand.
pub struct ZarrChunkReader {
    schema: SchemaRef,
    group_uri: String,
    array_infos: Vec<ArrayInfo>,
    geo: GroupGeoMetadata,
    group_transform: [f64; 6],
    spatial_dim_indices: Vec<usize>,
    /// Owned copy of `array_infos[0].dim_names[i]` for `i` in
    /// `spatial_dim_indices`. Kept here so `next()` doesn't need to
    /// rebuild the `&str` slice each row.
    spatial_dims_names: Vec<String>,
    chunk_spatial_shape: Vec<i64>,
    /// Current position in the chunk grid (row-major, C-order).
    chunk_indices: Vec<u64>,
    /// Whether the grid has been exhausted.
    exhausted: bool,
    batch_size: usize,
}

impl std::fmt::Debug for ZarrChunkReader {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("ZarrChunkReader")
            .field("group_uri", &self.group_uri)
            .field("num_arrays", &self.array_infos.len())
            .field("chunk_indices", &self.chunk_indices)
            .field("exhausted", &self.exhausted)
            .field("batch_size", &self.batch_size)
            .finish()
    }
}

impl ZarrChunkReader {
    /// Open a Zarr group and prepare a streaming reader over its chunk
    /// grid. Performs all I/O for metadata up front (open group, parse
    /// attributes, list arrays, validate constraints) so per-batch work
    /// is cheap.
    ///
    /// `arrays`:
    /// - `None` — read every multi-dimensional array. 1-D arrays
    ///   (typical xarray coord variables) are auto-skipped.
    /// - `Some(names)` — read exactly the named arrays. Unknown names
    ///   error. 1-D arrays are always rejected (a raster band needs
    ///   ≥ 2 dimensions); naming one explicitly errors with a clear
    ///   message.
    ///
    /// `batch_size` controls how many chunk rows are emitted per
    /// `RecordBatch`. Must be ≥ 1; callers typically pass
    /// `SessionConfig::batch_size` (defaults to 8192).
    pub async fn try_new(
        storage: AsyncReadableListableStorage,
        group_uri: &str,
        arrays: Option<&[String]>,
        batch_size: usize,
    ) -> Result<Self, ArrowError> {
        let batch_size = batch_size.max(1);
        let OpenedGroup {
            array_infos,
            geo,
            group_transform,
            spatial_dim_indices,
        } = open_and_validate(storage, group_uri, arrays).await?;

        let (spatial_dims_names, chunk_spatial_shape) = raster_spatial_metadata(
            &array_infos[0].dim_names,
            &array_infos[0].chunk_shape,
            &spatial_dim_indices,
        );

        let raster_field = SedonaType::Raster
            .to_storage_field("raster", true)
            .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
        let schema = Arc::new(Schema::new(vec![raster_field]));

        let chunk_indices = vec![0u64; array_infos[0].chunk_grid_shape.len()];

        Ok(Self {
            schema,
            group_uri: group_uri.to_string(),
            array_infos,
            geo,
            group_transform,
            spatial_dim_indices,
            spatial_dims_names,
            chunk_spatial_shape,
            chunk_indices,
            exhausted: false,
            batch_size,
        })
    }

    /// Append one raster row at the current `chunk_indices` to the
    /// in-progress builder. Does not advance the cursor.
    fn emit_one_row(&self, builder: &mut RasterBuilder) -> Result<(), ArrowError> {
        let row_transform = compute_row_transform(
            &self.group_transform,
            &self.chunk_indices,
            &self.array_infos[0].chunk_shape,
            &self.spatial_dim_indices,
        );
        let spatial_dims_ref: Vec<&str> =
            self.spatial_dims_names.iter().map(String::as_str).collect();
        let crs_str = self.geo.crs.as_deref();
        builder.start_raster_nd(
            &row_transform,
            &spatial_dims_ref,
            &self.chunk_spatial_shape,
            crs_str,
        )?;

        for info in &self.array_infos {
            let dim_names_ref: Vec<&str> = info.dim_names.iter().map(String::as_str).collect();
            let nodata_ref = info.nodata.as_deref();
            // Every band gets its chunk-anchor URI populated as
            // provenance metadata. `data.is_empty()` is the InDb/OutDb
            // discriminator; this reader always emits empty `data` and
            // defers pixel-byte resolution to the raster byte loader.
            let anchor = build_chunk_anchor(&self.group_uri, &info.path, &self.chunk_indices);
            let source_shape: Vec<i64> = info.chunk_shape.iter().map(|&n| n as i64).collect();
            builder.start_band_nd(
                Some(info.path.as_str()),
                &dim_names_ref,
                &source_shape,
                info.data_type,
                nodata_ref,
                Some(anchor.as_str()),
                Some("zarr"),
            )?;
            builder.band_data_writer().append_value([0u8; 0]);
            builder.finish_band()?;
        }
        builder.finish_raster()
    }
}

impl Iterator for ZarrChunkReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }
        let mut builder = RasterBuilder::new(self.batch_size);
        let mut rows_emitted = 0usize;
        while rows_emitted < self.batch_size {
            if let Err(e) = self.emit_one_row(&mut builder) {
                self.exhausted = true;
                return Some(Err(e));
            }
            rows_emitted += 1;
            if !advance_chunk_indices(
                &mut self.chunk_indices,
                &self.array_infos[0].chunk_grid_shape,
            ) {
                self.exhausted = true;
                break;
            }
        }
        if rows_emitted == 0 {
            return None;
        }
        let struct_arr = match builder.finish() {
            Ok(arr) => arr,
            Err(e) => return Some(Err(e)),
        };
        Some(RecordBatch::try_new(
            self.schema.clone(),
            vec![Arc::new(struct_arr)],
        ))
    }
}

impl RecordBatchReader for ZarrChunkReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

/// Per-array metadata extracted once at group open and reused for every
/// chunk position. Caching this avoids re-reading Zarr metadata for each
/// of the (potentially thousands of) chunk rows.
struct ArrayInfo {
    /// Array path within the store, used to build chunk anchor URIs and
    /// surface in band names.
    path: String,
    /// SedonaDB BandDataType corresponding to this array's zarrs dtype.
    data_type: BandDataType,
    /// Dimension names in array order. Required to be `Some(_)` for every
    /// dim; missing names error at validation time.
    dim_names: Vec<String>,
    /// Inner chunk grid shape, one entry per dimension. Used to enumerate
    /// chunk positions and validated to match across arrays.
    chunk_grid_shape: Vec<u64>,
    /// Chunk shape (elements per chunk per dim). Same for every chunk
    /// position — ragged final chunks are not emitted as separate short
    /// rows.
    chunk_shape: Vec<u64>,
    /// Full array shape (extent per dim). Used to validate that a spatial
    /// coordinate array's length matches the data extent before deriving a
    /// geotransform from it.
    shape: Vec<u64>,
    /// Encoded fill value in native-endian byte representation, for the
    /// `nodata` field. None when the array has no fill value declared.
    nodata: Option<Vec<u8>>,
}

/// Bundle of group-level state returned by [`open_and_validate`].
struct OpenedGroup {
    array_infos: Vec<ArrayInfo>,
    geo: GroupGeoMetadata,
    group_transform: [f64; 6],
    spatial_dim_indices: Vec<usize>,
}

/// Open the Zarr group, parse and validate group metadata, and return
/// everything `ZarrChunkReader` needs to iterate without further I/O
/// (apart from per-chunk byte fetches by future resolvers).
async fn open_and_validate(
    storage: AsyncReadableListableStorage,
    group_uri: &str,
    arrays_filter: Option<&[String]>,
) -> Result<OpenedGroup, ArrowError> {
    let group = Group::async_open(storage.clone(), "/").await.map_err(|e| {
        ArrowError::ExternalError(Box::new(sedona_internal_datafusion_err!(
            "failed to open Zarr group at {group_uri}: {e}"
        )))
    })?;

    let mut geo = GroupGeoMetadata::from_attributes(group.attributes())?;

    let arrays = match arrays_filter {
        // Explicit filter: open each named array directly, skipping the
        // listing step. This is the only viable path against backends
        // that don't expose directory listing (plain HTTPS without
        // WebDAV, S3-via-HttpStore) and is strictly less I/O than
        // list-then-filter when the user already knows what they want.
        // A 1-D array named explicitly is a user error — surface it
        // immediately rather than letting the spatial-dim resolver
        // fail with a confusing message downstream.
        Some(names) => {
            let arrays = open_named_arrays(&storage, names, group_uri).await?;
            for (name, array) in names.iter().zip(arrays.iter()) {
                if array.shape().len() < 2 {
                    return Err(ArrowError::InvalidArgumentError(format!(
                        "array {name:?} has rank {} (shape {:?}); a raster band \
                         requires at least 2 dimensions and cannot be read.",
                        array.shape().len(),
                        array.shape()
                    )));
                }
            }
            arrays
        }
        // Discovery: list direct children of the group and try to open
        // each as an array. Per-array open failures are logged + skipped
        // rather than poisoning the whole group, so a single malformed
        // sibling (e.g. ITS_LIVE's U-typed coord variables with null
        // fill values) doesn't take the whole read offline. 1-D arrays
        // (typical xarray coord variables) are silently dropped so a
        // canonical xarray layout reads cleanly.
        None => {
            let arrays = enumerate_child_arrays(&group, &storage, group_uri).await?;
            if arrays.is_empty() {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "Zarr group at {group_uri} has no child arrays"
                )));
            }
            let kept: Vec<_> = arrays.into_iter().filter(|a| a.shape().len() > 1).collect();
            if kept.is_empty() {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "Zarr group at {group_uri} contains only 1-D arrays (typical \
                     xarray-style coord variables); a raster band requires at least \
                     2 dimensions, so this group has nothing readable as a raster"
                )));
            }
            kept
        }
    };

    // CF / rioxarray groups declare their CRS — and often the affine
    // `GeoTransform` — on a separate scalar grid-mapping variable (commonly
    // `spatial_ref`) rather than in the group attributes. When the group
    // attributes left either unresolved, consult that variable before we drop
    // the open array handles.
    let cf_transform = if geo.crs.is_none() || geo.transform.is_none() {
        let cf = cf_georeference_from_grid_mapping(&storage, &arrays).await;
        if geo.crs.is_none() {
            geo.crs = cf.crs;
        }
        // Only relevant when the group declared no `spatial:transform`; otherwise
        // the explicit group transform wins in `resolve_group_transform`.
        geo.transform.is_none().then_some(cf.transform).flatten()
    } else {
        None
    };

    let array_infos = collect_array_infos(arrays)?;
    validate_group_constraints(&array_infos)?;

    // Spatial-dim resolution. Two configurations are accepted:
    //   - dim_names ends with a recognized spatial pair — ["y", "x"],
    //     ["lat", "lon"], or ["latitude", "longitude"] (canonical for
    //     georeferenced 2-D and time-series rasters); the spatial extent
    //     is the chunk's last two dims.
    //   - `spatial:dims` attribute on the group explicitly names them.
    // Anything else errors with a clear message — silently picking dims
    // would produce wrong per-row transforms.
    let spatial_dim_indices =
        resolve_spatial_dim_indices(&array_infos[0].dim_names, geo.spatial_dims.as_deref())?;

    // Effective transform + CRS. An explicit transform wins (GeoZarr
    // `spatial:transform`, then the CF `GeoTransform`); failing that a
    // `spatial:bbox`; failing that the spatial coordinate arrays (the common
    // CF / non-GeoZarr case); failing that identity pixel coords.
    let group_transform = resolve_group_transform(
        &storage,
        group_uri,
        &mut geo,
        cf_transform,
        &array_infos,
        &spatial_dim_indices,
    )
    .await?;

    // chunk_grid_shape comes from untrusted Zarr metadata; bound-check the
    // product so a hostile or malformed grid can't make per-batch
    // RasterBuilder capacity (or downstream consumers) overflow.
    array_infos[0]
        .chunk_grid_shape
        .iter()
        .try_fold(1usize, |acc, &n| {
            usize::try_from(n).ok().and_then(|n| acc.checked_mul(n))
        })
        .ok_or_else(|| {
            ArrowError::InvalidArgumentError(format!(
                "chunk grid shape {:?} overflows usize",
                array_infos[0].chunk_grid_shape
            ))
        })?;

    Ok(OpenedGroup {
        array_infos,
        geo,
        group_transform,
        spatial_dim_indices,
    })
}

/// Resolve the group's GDAL geotransform. An explicit `spatial:transform` wins;
/// failing that a `spatial:bbox`; failing that the spatial coordinate arrays;
/// failing that the identity pixel transform. Infers and sets `geo.crs` from the
/// spatial dim names when deriving from a bbox or coordinate arrays and none is
/// declared.
async fn resolve_group_transform(
    storage: &AsyncReadableListableStorage,
    group_uri: &str,
    geo: &mut GroupGeoMetadata,
    cf_transform: Option<[f64; 6]>,
    array_infos: &[ArrayInfo],
    spatial_dim_indices: &[usize],
) -> Result<[f64; 6], ArrowError> {
    // An explicit transform wins, GeoZarr `spatial:transform` first and then the
    // CF `GeoTransform` from the grid-mapping variable (both already GDAL order).
    if let Some(t) = geo.transform {
        return Ok(t);
    }
    if let Some(t) = cf_transform {
        log::debug!(
            "Zarr group at {group_uri} has no `spatial:transform`; using the CF \
             `GeoTransform` from the grid-mapping variable"
        );
        return Ok(t);
    }
    // spatial_dim_indices is validated upstream to be [y_index, x_index].
    let y_axis = spatial_dim_indices[0];
    let x_axis = spatial_dim_indices[1];

    // A declared `spatial:bbox` is a stronger georeferencing signal than CF
    // coordinate arrays, so try it first. An unusable bbox is non-fatal — warn
    // and fall through to the coordinate arrays.
    if let Some(t) = transform_from_bbox(group_uri, geo, array_infos, y_axis, x_axis) {
        return Ok(t);
    }
    transform_from_coordinate_arrays(storage, group_uri, geo, array_infos, y_axis, x_axis).await
}

/// Try to derive a transform from `spatial:bbox` and the array's *own* shape (no
/// separate `spatial:shape` attribute that could drift). Returns `None` — so the
/// caller falls back to coordinate arrays — when there is no bbox or it is
/// unusable. Infers `geo.crs` from the spatial dim names when none is declared.
fn transform_from_bbox(
    group_uri: &str,
    geo: &mut GroupGeoMetadata,
    array_infos: &[ArrayInfo],
    y_axis: usize,
    x_axis: usize,
) -> Option<[f64; 6]> {
    let bbox = geo.bbox?;
    let height = array_infos[0].shape[y_axis];
    let width = array_infos[0].shape[x_axis];
    match AffineMatrix::from_bbox_and_spatial_shape(
        bbox,
        height,
        width,
        geo.registration.as_deref(),
    ) {
        Ok(matrix) => {
            // Mirror the coordinate-array path: when the group declares no CRS,
            // infer a geographic one from the spatial dim names (lat/lon).
            // Generic y/x stay CRS-less (attach via RS_SetCRS).
            if geo.crs.is_none() {
                if let Some(inferred) = coords::infer_geographic_crs(
                    &array_infos[0].dim_names[y_axis],
                    &array_infos[0].dim_names[x_axis],
                ) {
                    geo.crs = Some(inferred.to_string());
                }
            }
            log::debug!(
                "Zarr group at {group_uri} has no `spatial:transform`; derived a \
                 geotransform from `spatial:bbox` and the {height}x{width} array shape"
            );
            Some(matrix.to_gdal_geotransform())
        }
        Err(e) => {
            log::warn!(
                "Zarr group at {group_uri}: `spatial:bbox` is unusable ({e}); \
                 falling back to coordinate arrays for georeferencing"
            );
            None
        }
    }
}

/// Derive a transform from the spatial coordinate arrays, falling back to the
/// identity pixel transform when they are absent or unusable. Errors if a CRS is
/// declared but no transform can be derived (silently using pixel coordinates
/// would produce wrong spatial-join results). Infers `geo.crs` from the spatial
/// dim names when none is declared and the coordinates are usable.
async fn transform_from_coordinate_arrays(
    storage: &AsyncReadableListableStorage,
    group_uri: &str,
    geo: &mut GroupGeoMetadata,
    array_infos: &[ArrayInfo],
    y_axis: usize,
    x_axis: usize,
) -> Result<[f64; 6], ArrowError> {
    let y_name = array_infos[0].dim_names[y_axis].clone();
    let x_name = array_infos[0].dim_names[x_axis].clone();
    let x_vals = coords::read_coord_values(storage, &x_name).await?;
    let y_vals = coords::read_coord_values(storage, &y_name).await?;

    // A coordinate array must span the data extent along its axis; a length
    // mismatch is a malformed coord variable that would yield a wrong scale, so
    // refuse to derive a transform from it.
    let x_ok = x_vals
        .as_ref()
        .is_none_or(|v| v.len() as u64 == array_infos[0].shape[x_axis]);
    let y_ok = y_vals
        .as_ref()
        .is_none_or(|v| v.len() as u64 == array_infos[0].shape[y_axis]);
    if !(x_ok && y_ok) {
        log::warn!(
            "Zarr group at {group_uri}: a spatial coordinate array's length does not \
             match the data extent; ignoring coordinates for georeferencing"
        );
    }
    let derived = match (x_ok && y_ok, x_vals, y_vals) {
        (true, Some(x), Some(y)) => coords::transform_from_coords(&x, &y),
        _ => None,
    };
    match derived {
        Some(t) => {
            // Regular CF coordinate arrays imply geographic lon/lat; infer the
            // CRS from the dim names only when none was declared. Generic y/x stay
            // CRS-less (attach via RS_SetCRS).
            let crs_note = if let Some(declared) = geo.crs.as_deref() {
                format!("keeping the declared CRS {declared:?}")
            } else if let Some(inferred) = coords::infer_geographic_crs(&y_name, &x_name) {
                geo.crs = Some(inferred.to_string());
                format!("inferred CRS {inferred} from the dim names")
            } else {
                "no CRS inferred (spatial dims are not lat/lon) — set one with RS_SetCRS"
                    .to_string()
            };
            log::debug!(
                "Zarr group at {group_uri} has no `spatial:transform`; derived a \
                 geotransform from the {x_name:?}/{y_name:?} coordinate arrays; {crs_note}"
            );
            Ok(t)
        }
        // CRS-without-transform and no usable coordinate arrays is almost always
        // malformed metadata; error rather than silently using pixel coordinates
        // in spatial joins.
        None if geo.crs.is_some() => Err(ArrowError::InvalidArgumentError(format!(
            "Zarr group at {group_uri} declares a CRS but has neither a \
             `spatial:transform` attribute nor regularly-spaced numeric spatial \
             coordinate arrays; refusing to fall back to the identity transform \
             because that would silently produce wrong results in spatial joins. \
             Declare `spatial:transform`, provide regular coordinate arrays, or \
             remove the CRS to read this as a non-georeferenced datacube."
        ))),
        // No CRS and no usable coordinates: index space, with a breadcrumb so
        // spatial-join surprises are debuggable.
        None => {
            log::warn!(
                "Zarr group at {group_uri} has no `spatial:transform` and no usable \
                 spatial coordinate arrays; falling back to the identity \
                 pixel-coordinate transform [0, 1, 0, 0, 0, -1]. Spatial operations \
                 against this raster will use pixel coordinates."
            );
            Ok([0.0, 1.0, 0.0, 0.0, 0.0, -1.0])
        }
    }
}

/// Open arrays at the explicit paths requested by the caller, skipping
/// the group listing step entirely. Used when `arrays_filter` is `Some`
/// — the canonical path for backends that can't list (plain HTTPS / S3
/// behind HttpStore) and a strict subset of work for any other backend.
async fn open_named_arrays(
    storage: &AsyncReadableListableStorage,
    names: &[String],
    group_uri: &str,
) -> Result<Vec<Array<dyn AsyncReadableListableStorageTraits>>, ArrowError> {
    let mut out = Vec::with_capacity(names.len());
    for name in names {
        let normalized = name.trim_start_matches('/');
        let path = format!("/{normalized}");
        let array = Array::async_open(storage.clone(), &path)
            .await
            .map_err(|e| {
                ArrowError::InvalidArgumentError(format!(
                    "Zarr group at {group_uri} has no array named {name:?} \
                     (or its metadata could not be parsed): {e}"
                ))
            })?;
        out.push(array);
    }
    Ok(out)
}

/// CRS and/or affine transform read from a CF grid-mapping variable.
#[derive(Default)]
struct CfGeoreference {
    crs: Option<String>,
    transform: Option<[f64; 6]>,
}

/// Resolve the CRS and affine `GeoTransform` from a CF / rioxarray grid-mapping
/// variable.
///
/// rioxarray writes the CRS (and usually the `GeoTransform`) to a scalar
/// variable (conventionally `spatial_ref`, sometimes `crs`) carrying `crs_wkt` /
/// `spatial_ref` and `GeoTransform` attributes, linked from each data variable
/// via the CF `grid_mapping` attribute — or, as rioxarray does, listed in the
/// data variable's `coordinates` attribute. That variable is a 0-D scalar, so it
/// never appears among the rank ≥ 2 data arrays and must be opened on its own.
///
/// Candidate names are gathered in priority order — `grid_mapping` targets
/// first, then `coordinates` tokens, then the conventional `spatial_ref` /
/// `crs` — and the first candidate that opens and carries either a CRS or a
/// `GeoTransform` wins (both are read from that one variable). A candidate that
/// is missing, unreadable, or carries neither is skipped, so a group without a
/// grid-mapping variable simply stays unreferenced.
async fn cf_georeference_from_grid_mapping(
    storage: &AsyncReadableListableStorage,
    data_arrays: &[Array<dyn AsyncReadableListableStorageTraits>],
) -> CfGeoreference {
    let candidates = cf_crs_candidate_names(data_arrays.iter().map(|a| a.attributes()));
    for name in candidates {
        let path = format!("/{}", name.trim_start_matches('/'));
        let array = match Array::async_open(storage.clone(), &path).await {
            Ok(array) => array,
            // An absent candidate and a transient/permission failure look
            // alike here; georeferencing is optional so we keep trying, but
            // leave a breadcrumb so a genuinely broken store stays traceable.
            Err(e) => {
                log::debug!("Zarr grid-mapping candidate {name:?}: open failed: {e}");
                continue;
            }
        };
        let attrs = array.attributes();
        let crs = match crs_from_cf_attributes(attrs) {
            Ok(crs) => crs,
            // A malformed CRS attribute on a candidate variable shouldn't take
            // the whole read offline; log and treat it as absent.
            Err(e) => {
                log::warn!("Zarr grid-mapping variable {name:?}: ignoring CRS: {e}");
                None
            }
        };
        let transform = geotransform_from_cf_attributes(attrs);
        // The first variable that carries either is the grid-mapping variable;
        // both come from it, so stop here.
        if crs.is_some() || transform.is_some() {
            return CfGeoreference { crs, transform };
        }
    }
    CfGeoreference::default()
}

/// Gather candidate grid-mapping variable names from the data arrays'
/// attributes, in the order they should be tried: every `grid_mapping`
/// target, then every `coordinates` token, then the conventional
/// `spatial_ref` / `crs`. De-duplicated, first occurrence wins.
///
/// CF 1.7+ allows an extended `grid_mapping` form, `"<crs var>: <coord>
/// <coord> ..."`, where the grid-mapping variable name carries a trailing
/// colon; we strip it so that name resolves. The colon-less coordinate names
/// in that form are tried too — harmless, they carry no CRS attribute.
fn cf_crs_candidate_names<'a>(
    array_attrs: impl Iterator<Item = &'a serde_json::Map<String, serde_json::Value>>,
) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    let mut push = |s: &str| {
        for tok in s.split_whitespace() {
            let tok = tok.trim_end_matches(':');
            if !tok.is_empty() && !names.iter().any(|n| n == tok) {
                names.push(tok.to_string());
            }
        }
    };
    let attrs: Vec<_> = array_attrs.collect();
    for key in ["grid_mapping", "coordinates"] {
        for a in &attrs {
            if let Some(s) = a.get(key).and_then(|v| v.as_str()) {
                push(s);
            }
        }
    }
    for conventional in ["spatial_ref", "crs"] {
        push(conventional);
    }
    names
}

/// Discover the group's direct child arrays.
///
/// Discovery goes through zarrs's [`Group::async_children`], which uses
/// the group's V3 `consolidated_metadata` block when present (no
/// per-node storage reads) and otherwise lists the store. Listing is
/// unsupported on some backends — plain HTTPS / S3-behind-`HttpStore`
/// answer directory listing with `405 Method Not Allowed` — so a store
/// that has neither consolidated metadata nor listing yields an
/// actionable error pointing at the `arrays` option rather than a raw
/// store error.
///
/// Each array is then built from its already-parsed metadata. Per-array
/// failures are logged at warn level and the child is skipped — a single
/// malformed sibling array (e.g. an xarray-style coord variable with a
/// dtype zarrs can't yet parse) can no longer poison the whole group.
/// Subgroups are skipped via the `NodeMetadata::Group` arm.
async fn enumerate_child_arrays(
    group: &Group<dyn AsyncReadableListableStorageTraits>,
    storage: &AsyncReadableListableStorage,
    group_uri: &str,
) -> Result<Vec<Array<dyn AsyncReadableListableStorageTraits>>, ArrowError> {
    let children = group.async_children(false).await.map_err(|e| {
        ArrowError::ExternalError(Box::new(sedona_internal_datafusion_err!(
            "failed to discover child arrays of Zarr group at {group_uri}: {e}. \
             If this store has no consolidated metadata and does not support \
             directory listing (for example a plain HTTPS server), pass the \
             `arrays` option to name the arrays to read."
        )))
    })?;

    let mut arrays = Vec::with_capacity(children.len());
    for node in children {
        let path = node.path().to_string();
        match NodeMetadata::from(node) {
            NodeMetadata::Array(metadata) => {
                match Array::new_with_metadata(storage.clone(), &path, metadata) {
                    Ok(array) => arrays.push(array),
                    Err(e) => log::warn!(
                        "skipping Zarr child {path} in {group_uri}: not usable as an array ({e})"
                    ),
                }
            }
            NodeMetadata::Group(_) => {}
        }
    }
    Ok(arrays)
}

/// Collect per-array metadata from open zarrs `Array` handles.
///
/// Sorts arrays by path so band ordering across rows is deterministic
/// (zarrs's underlying store listing order is implementation-defined —
/// filesystem stores currently happen to enumerate alphabetically, but
/// that's not part of the contract we want consumers to rely on).
fn collect_array_infos(
    mut arrays: Vec<Array<dyn AsyncReadableListableStorageTraits>>,
) -> Result<Vec<ArrayInfo>, ArrowError> {
    arrays.sort_by(|a, b| a.path().as_str().cmp(b.path().as_str()));
    let mut out = Vec::with_capacity(arrays.len());
    for array in arrays {
        let path = array.path().to_string();
        let data_type = zarr_to_band_data_type(array.data_type())?;
        let dim_names = resolve_dim_names(&array, &path)?;
        let chunk_grid_shape = array.chunk_grid_shape().to_vec();
        let chunk_shape = array
            .chunk_shape(&vec![0u64; chunk_grid_shape.len()])
            .map_err(|e| {
                ArrowError::ExternalError(Box::new(sedona_internal_datafusion_err!(
                    "array {path}: failed to query chunk shape: {e}"
                )))
            })?
            .iter()
            .map(|n| n.get())
            .collect();
        let shape = array.shape().to_vec();
        let fill_bytes = array.fill_value().as_ne_bytes();
        let nodata = if fill_bytes.is_empty() {
            None
        } else {
            Some(fill_bytes.to_vec())
        };
        out.push(ArrayInfo {
            path,
            data_type,
            dim_names,
            chunk_grid_shape,
            chunk_shape,
            shape,
            nodata,
        });
    }
    Ok(out)
}

/// Resolve dimension names for an array, supporting both Zarr v3
/// (first-class `dimension_names` field) and Zarr v2 with the xarray
/// `_ARRAY_DIMENSIONS` attribute. Errors if neither carries a complete
/// set of named dimensions matching the array's rank.
fn resolve_dim_names<S: ?Sized>(array: &Array<S>, path: &str) -> Result<Vec<String>, ArrowError> {
    let rank = array.shape().len();

    if let Some(names) = array.dimension_names() {
        return names
            .iter()
            .enumerate()
            .map(|(i, n)| {
                n.clone().ok_or_else(|| {
                    ArrowError::InvalidArgumentError(format!(
                        "array {path}: dimension {i} has no name; every Zarr array \
                         dimension must be named",
                    ))
                })
            })
            .collect();
    }

    // Zarr v2 fallback: xarray's `_ARRAY_DIMENSIONS` convention. v2 had no
    // first-class dimension_names field and zarrs's v2->v3 converter
    // preserves attributes verbatim without lifting this one.
    if let Some(value) = array.attributes().get("_ARRAY_DIMENSIONS") {
        let arr = value.as_array().ok_or_else(|| {
            ArrowError::InvalidArgumentError(format!(
                "array {path}: _ARRAY_DIMENSIONS must be a JSON array of strings; got {value}"
            ))
        })?;
        if arr.len() != rank {
            return Err(ArrowError::InvalidArgumentError(format!(
                "array {path}: _ARRAY_DIMENSIONS has {} entries but array has rank {rank}",
                arr.len()
            )));
        }
        return arr
            .iter()
            .enumerate()
            .map(|(i, v)| {
                v.as_str().map(str::to_string).ok_or_else(|| {
                    ArrowError::InvalidArgumentError(format!(
                        "array {path}: _ARRAY_DIMENSIONS[{i}] must be a string; got {v}"
                    ))
                })
            })
            .collect();
    }

    Err(ArrowError::InvalidArgumentError(format!(
        "array {path}: no dimension names found. Zarr v3 arrays must set \
         `dimension_names`; Zarr v2 arrays must set the `_ARRAY_DIMENSIONS` \
         attribute (xarray convention)."
    )))
}

/// Enforce group constraints. All arrays must agree on chunk grid shape,
/// chunk shape, and dimension names. We do NOT enforce shared element
/// shape (`array.shape()`) because users routinely group arrays with
/// the same chunk grid but different totals (e.g. a coord variable with
/// one fewer dim is rejected here anyway by the dim-name check).
fn validate_group_constraints(infos: &[ArrayInfo]) -> Result<(), ArrowError> {
    let first = &infos[0];
    for other in &infos[1..] {
        if other.chunk_grid_shape != first.chunk_grid_shape {
            return Err(ArrowError::InvalidArgumentError(format!(
                "arrays {} and {} have different chunk grid shapes ({:?} vs {:?}); \
                 every array in the group must share the same chunk grid. \
                 Pass `arrays = [...]` to read only compatible arrays.",
                first.path, other.path, first.chunk_grid_shape, other.chunk_grid_shape
            )));
        }
        if other.chunk_shape != first.chunk_shape {
            return Err(ArrowError::InvalidArgumentError(format!(
                "arrays {} and {} have different chunk shapes ({:?} vs {:?}); \
                 every array in the group must share the same chunk shape. \
                 Pass `arrays = [...]` to read only compatible arrays.",
                first.path, other.path, first.chunk_shape, other.chunk_shape
            )));
        }
        if other.dim_names != first.dim_names {
            return Err(ArrowError::InvalidArgumentError(format!(
                "arrays {} and {} have different dimension names ({:?} vs {:?}); \
                 every array in the group must declare identical dim_names. \
                 Pass `arrays = [...]` to read only compatible arrays.",
                first.path, other.path, first.dim_names, other.dim_names
            )));
        }
    }
    Ok(())
}

/// Pick the `(y_index, x_index)` axes of an array's dim_names.
///
/// If `spatial_dims` is provided via the group's `spatial:dims` attribute,
/// look up those names by position. Otherwise, default to the last two
/// dims and require they form a recognized spatial pair — `[y, x]`,
/// `[lat, lon]`, or `[latitude, longitude]` (in that order), the common
/// CF / GeoZarr-2D conventions. Anything else errors.
fn resolve_spatial_dim_indices(
    dim_names: &[String],
    spatial_dims: Option<&[String]>,
) -> Result<Vec<usize>, ArrowError> {
    if let Some(spatial) = spatial_dims {
        let mut idx = Vec::with_capacity(spatial.len());
        for s in spatial {
            let i = dim_names.iter().position(|n| n == s).ok_or_else(|| {
                ArrowError::InvalidArgumentError(format!(
                    "spatial:dims declared name {s:?} not found in array dim_names {dim_names:?}",
                ))
            })?;
            idx.push(i);
        }
        return Ok(idx);
    }
    let n = dim_names.len();
    if n < 2 {
        return Err(ArrowError::InvalidArgumentError(format!(
            "at least 2 dimensions are required to resolve spatial axes; got {dim_names:?}",
        )));
    }
    if !is_spatial_dim_pair(&dim_names[n - 2], &dim_names[n - 1]) {
        return Err(ArrowError::InvalidArgumentError(format!(
            "the last two dim_names must be a recognized spatial pair \
             ([\"y\", \"x\"], [\"lat\", \"lon\"], or [\"latitude\", \"longitude\"]) \
             when `spatial:dims` is not declared; got {dim_names:?}",
        )));
    }
    Ok(vec![n - 2, n - 1])
}

/// Build the raster-level `(spatial_dims, spatial_shape)` for a chunk in the
/// X-first GIS order that `RasterRef::width()`/`height()`/`x_dim()`/`y_dim()`
/// and the GDAL geotransform expect (X/width at index 0).
///
/// `spatial_dim_indices` is in file (C-)order — the slowest-varying spatial
/// axis first, i.e. `[y_index, x_index]` for the 2-D case (see
/// [`resolve_spatial_dim_indices`]). Bands keep their natural C-order
/// `dim_names`/`shape` (matching the chunk's physical pixel layout); only
/// this logical raster-level descriptor is reordered, so reversing the
/// index list yields fastest-axis-first `[x, y]`. No pixel data is moved.
fn raster_spatial_metadata(
    dim_names: &[String],
    chunk_shape: &[u64],
    spatial_dim_indices: &[usize],
) -> (Vec<String>, Vec<i64>) {
    spatial_dim_indices
        .iter()
        .rev()
        .map(|&i| (dim_names[i].clone(), chunk_shape[i] as i64))
        .unzip()
}

/// Per-chunk transform: translate the group's transform so the chunk's
/// `[0, 0]` element maps to the chunk's spatial origin.
fn compute_row_transform(
    group_transform: &[f64; 6],
    chunk_indices: &[u64],
    chunk_shape: &[u64],
    spatial_dim_indices: &[usize],
) -> [f64; 6] {
    // GDAL GeoTransform layout: [origin_x, scale_x, skew_x, origin_y, skew_y, scale_y].
    // Translation along x = chunk_x_index × chunk_x_size in pixel-coordinate space,
    // converted to world coordinates via the affine.
    //
    // spatial_dim_indices is validated upstream to be [y_index, x_index].
    // Index 0 is the y axis, index 1 is the x axis.
    let y_axis = spatial_dim_indices[0];
    let x_axis = spatial_dim_indices[1];
    let x_offset = (chunk_indices[x_axis] * chunk_shape[x_axis]) as f64;
    let y_offset = (chunk_indices[y_axis] * chunk_shape[y_axis]) as f64;
    let [ox, sx, kx, oy, ky, sy] = *group_transform;
    [
        ox + sx * x_offset + kx * y_offset,
        sx,
        kx,
        oy + ky * x_offset + sy * y_offset,
        ky,
        sy,
    ]
}

/// Advance `chunk_indices` row-major over `chunk_grid_shape`. Returns
/// `true` while there are positions left, `false` when the grid is
/// exhausted (and the indices wrap back to all-zero).
fn advance_chunk_indices(chunk_indices: &mut [u64], chunk_grid_shape: &[u64]) -> bool {
    for i in (0..chunk_indices.len()).rev() {
        chunk_indices[i] += 1;
        if chunk_indices[i] < chunk_grid_shape[i] {
            return true;
        }
        chunk_indices[i] = 0;
    }
    false
}

/// Retrieve a single chunk's bytes as a fresh `Vec<u8>`.
///
/// Uses `ArrayBytes::Fixed`, so this errors for variable-length element
/// types — those don't have a `BandDataType` counterpart anyway, so the
/// dtype check in `collect_array_infos` rejects them upstream.
///
/// The sync chunk-decode path, exercised by the unit test below against a
/// filesystem fixture. The production byte loader (`raster_loader::ZarrLoader`)
/// reads over async object_store storage and so retrieves chunks directly
/// via `Array::async_retrieve_chunk` rather than through this helper.
#[cfg(test)]
fn retrieve_chunk_bytes<S>(array: &Array<S>, chunk_indices: &[u64]) -> Result<Vec<u8>, ArrowError>
where
    S: ?Sized + zarrs::storage::ReadableStorageTraits + 'static,
{
    let bytes = array
        .retrieve_chunk::<ArrayBytes<'static>>(chunk_indices)
        .map_err(|e| {
            ArrowError::ExternalError(Box::new(sedona_internal_datafusion_err!(
                "failed to retrieve chunk {:?} from {}: {e}",
                chunk_indices,
                array.path()
            )))
        })?;
    let raw = bytes.into_fixed().map_err(|_| {
        ArrowError::InvalidArgumentError(format!(
            "array {}: variable-length chunk bytes are not supported",
            array.path()
        ))
    })?;
    Ok(raw.into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tempfile::TempDir;
    use zarrs::array::data_type;
    use zarrs::array::ArrayBuilder;
    use zarrs_filesystem::FilesystemStore;

    /// Direct coverage for `retrieve_chunk_bytes`. The function is the
    /// only pixel-byte read primitive in the crate today; previously it
    /// was exercised through `group_to_indb_rasters` integration tests,
    /// which are gone now that the loader only emits OutDb anchors. The
    /// follow-up `RS_EnsureLoaded` resolver will call this directly.
    #[test]
    fn retrieve_chunk_bytes_returns_decoded_chunk() {
        let tmp = TempDir::new().unwrap();
        let store = Arc::new(FilesystemStore::new(tmp.path()).unwrap());

        // 1×4 UInt8 array with chunks of size [1, 2] → chunk grid [1, 2].
        let arr = ArrayBuilder::new(vec![1u64, 4u64], vec![1u64, 2u64], data_type::uint8(), 0u8)
            .dimension_names(Some(["y", "x"]))
            .build(store.clone(), "/band")
            .unwrap();
        arr.store_metadata().unwrap();
        arr.store_chunk(&[0u64, 0u64], vec![10u8, 11u8]).unwrap();
        arr.store_chunk(&[0u64, 1u64], vec![20u8, 21u8]).unwrap();

        let chunk_0 = retrieve_chunk_bytes(&arr, &[0, 0]).unwrap();
        assert_eq!(chunk_0, vec![10u8, 11u8]);

        let chunk_1 = retrieve_chunk_bytes(&arr, &[0, 1]).unwrap();
        assert_eq!(chunk_1, vec![20u8, 21u8]);
    }

    #[test]
    fn advance_chunk_indices_walks_row_major() {
        // 2×3 grid; outer axis varies slowest.
        let shape = vec![2u64, 3u64];
        let mut idx = vec![0u64, 0u64];
        let mut visited = vec![idx.clone()];
        while advance_chunk_indices(&mut idx, &shape) {
            visited.push(idx.clone());
        }
        // Expected row-major traversal of a 2×3 grid (last axis fastest):
        //   (0,0) (0,1) (0,2) (1,0) (1,1) (1,2)
        let expected = vec![
            vec![0, 0],
            vec![0, 1],
            vec![0, 2],
            vec![1, 0],
            vec![1, 1],
            vec![1, 2],
        ];
        assert_eq!(visited, expected);
    }

    #[test]
    fn advance_chunk_indices_signals_exhaustion_via_wraparound() {
        // After the last position (1,2) the next advance must return false
        // and reset back to all-zero.
        let shape = vec![2u64, 3u64];
        let mut idx = vec![1u64, 2u64];
        assert!(!advance_chunk_indices(&mut idx, &shape));
        assert_eq!(idx, vec![0, 0]);
    }

    #[test]
    fn advance_chunk_indices_single_position_grid_exits_immediately() {
        let shape = vec![1u64];
        let mut idx = vec![0u64];
        assert!(!advance_chunk_indices(&mut idx, &shape));
    }

    #[test]
    fn compute_row_transform_translates_to_chunk_origin_no_skew() {
        // 2-D y,x array with chunk [2, 2] and group origin (10, 20).
        // Chunk (1, 2) in row-major should map to origin (10 + 2*2, 20 + 2*1*(-1))
        // for transform [10, 1, 0, 20, 0, -1]: x_off = 4, y_off = 2.
        let group_t = [10.0, 1.0, 0.0, 20.0, 0.0, -1.0];
        let chunk_shape = vec![2u64, 2u64];
        let chunk_idx = vec![1u64, 2u64];
        let t = compute_row_transform(&group_t, &chunk_idx, &chunk_shape, &[0, 1]);
        // y_axis=0, x_axis=1 → x_off=2*2=4, y_off=1*2=2
        assert_eq!(t[0], 10.0 + 4.0); // origin_x
        assert_eq!(t[3], 20.0 - 2.0); // origin_y after y_off=2 with sy=-1
                                      // Scale/skew carry through unchanged.
        assert_eq!(t[1], 1.0);
        assert_eq!(t[2], 0.0);
        assert_eq!(t[4], 0.0);
        assert_eq!(t[5], -1.0);
    }

    #[test]
    fn resolve_spatial_dim_indices_default_yx() {
        let names = vec!["time".into(), "y".into(), "x".into()];
        let idx = resolve_spatial_dim_indices(&names, None).unwrap();
        assert_eq!(idx, vec![1, 2]);
    }

    #[test]
    fn raster_spatial_metadata_is_x_first_for_nonsquare_chunk() {
        // Band is C-order [time, lat, lon] with a deliberately non-square
        // chunk: lat=2 (height), lon=3 (width). resolve_spatial_dim_indices
        // returns [lat_idx, lon_idx] = [1, 2] (y-first, file order).
        let dim_names = vec!["time".to_string(), "lat".to_string(), "lon".to_string()];
        let chunk_shape = vec![1u64, 2, 3];
        let (dims, shape) = raster_spatial_metadata(&dim_names, &chunk_shape, &[1, 2]);
        // Raster-level descriptor must be X-first so width()=shape[0]=lon=3
        // and height()=shape[1]=lat=2 (not the band's C-order [lat, lon]).
        assert_eq!(dims, vec!["lon".to_string(), "lat".to_string()]);
        assert_eq!(shape, vec![3, 2]);
    }

    #[test]
    fn resolve_spatial_dim_indices_default_latlon() {
        let names = vec!["time".into(), "lat".into(), "lon".into()];
        let idx = resolve_spatial_dim_indices(&names, None).unwrap();
        assert_eq!(idx, vec![1, 2]);
    }

    #[test]
    fn resolve_spatial_dim_indices_default_latitude_longitude() {
        let names = vec!["latitude".into(), "longitude".into()];
        let idx = resolve_spatial_dim_indices(&names, None).unwrap();
        assert_eq!(idx, vec![0, 1]);
    }

    #[test]
    fn resolve_spatial_dim_indices_default_rejects_wrong_order() {
        let names = vec!["x".into(), "y".into()];
        let err = resolve_spatial_dim_indices(&names, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("[\"y\", \"x\"]"), "{err}");
    }

    #[test]
    fn resolve_spatial_dim_indices_explicit_lookup() {
        let names = vec!["lat".into(), "lon".into(), "time".into()];
        let spatial = vec!["lat".to_string(), "lon".to_string()];
        let idx = resolve_spatial_dim_indices(&names, Some(&spatial)).unwrap();
        assert_eq!(idx, vec![0, 1]);
    }

    fn attrs(json: serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
        json.as_object().unwrap().clone()
    }

    #[test]
    fn cf_crs_candidate_names_prioritizes_grid_mapping_then_coordinates() {
        // grid_mapping target wins, coordinates tokens follow (real coord
        // vars like `time` get tried too — harmless, they carry no CRS), and
        // the conventional names are appended last.
        let a = attrs(serde_json::json!({
            "grid_mapping": "crs",
            "coordinates": "spatial_ref time"
        }));
        let got = cf_crs_candidate_names(std::iter::once(&a));
        assert_eq!(got, vec!["crs", "spatial_ref", "time"]);
    }

    #[test]
    fn cf_crs_candidate_names_falls_back_to_conventional() {
        // No grid_mapping / coordinates anywhere: try the conventional names.
        let a = attrs(serde_json::json!({"units": "K"}));
        let got = cf_crs_candidate_names(std::iter::once(&a));
        assert_eq!(got, vec!["spatial_ref", "crs"]);
    }

    #[test]
    fn cf_crs_candidate_names_strips_extended_grid_mapping_colon() {
        // CF 1.7+ extended form: the colon-suffixed token is the grid-mapping
        // variable; the rest are the coordinates it applies to.
        let a = attrs(serde_json::json!({"grid_mapping": "crs: lon lat"}));
        let got = cf_crs_candidate_names(std::iter::once(&a));
        // `crs` (colon stripped) first, then the coord tokens, then the
        // conventional `spatial_ref` (`crs` already present, deduped).
        assert_eq!(got, vec!["crs", "lon", "lat", "spatial_ref"]);
    }

    #[test]
    fn cf_crs_candidate_names_dedupes_across_arrays() {
        let a = attrs(serde_json::json!({"coordinates": "spatial_ref time"}));
        let b = attrs(serde_json::json!({"coordinates": "spatial_ref lat lon"}));
        let got = cf_crs_candidate_names([&a, &b].into_iter());
        assert_eq!(got, vec!["spatial_ref", "time", "lat", "lon", "crs"]);
    }

    #[test]
    fn resolve_spatial_dim_indices_explicit_missing_errors() {
        let names = vec!["a".into(), "b".into()];
        let spatial = vec!["nope".to_string()];
        let err = resolve_spatial_dim_indices(&names, Some(&spatial))
            .unwrap_err()
            .to_string();
        assert!(err.contains("nope"), "{err}");
    }
}
