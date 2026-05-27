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
use sedona_raster::builder::RasterBuilder;
use sedona_schema::datatypes::SedonaType;
use sedona_schema::raster::BandDataType;
use zarrs::array::{Array, ArrayBytes};
use zarrs::group::Group;
use zarrs_filesystem::FilesystemStore;

use crate::dtype::zarr_to_band_data_type;
use crate::geozarr::GroupGeoMetadata;
use crate::source_uri::{build_chunk_anchor, group_uri_to_filesystem_path};

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
/// a chunk anchor that the async OutDb resolver (registered separately)
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
    pub fn try_new(
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
        } = open_and_validate(group_uri, arrays)?;

        let spatial_dims_names: Vec<String> = spatial_dim_indices
            .iter()
            .map(|&i| array_infos[0].dim_names[i].clone())
            .collect();
        let chunk_spatial_shape: Vec<i64> = spatial_dim_indices
            .iter()
            .map(|&i| array_infos[0].chunk_shape[i] as i64)
            .collect();

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
            // defers pixel-byte resolution to the OutDb resolver.
            let anchor = build_chunk_anchor(&self.group_uri, &info.path, &self.chunk_indices);
            builder.start_band_nd(
                Some(info.path.as_str()),
                &dim_names_ref,
                &info.chunk_shape,
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
fn open_and_validate(
    group_uri: &str,
    arrays_filter: Option<&[String]>,
) -> Result<OpenedGroup, ArrowError> {
    let fs_path = group_uri_to_filesystem_path(group_uri)?;
    let store = FilesystemStore::new(&fs_path).map_err(|e| {
        ArrowError::ExternalError(Box::new(sedona_internal_datafusion_err!(
            "failed to open Zarr filesystem store at {}: {e}",
            fs_path.display()
        )))
    })?;
    let storage: Arc<FilesystemStore> = Arc::new(store);

    let group = Group::open(storage.clone(), "/").map_err(|e| {
        ArrowError::ExternalError(Box::new(sedona_internal_datafusion_err!(
            "failed to open Zarr group at {group_uri}: {e}"
        )))
    })?;

    let geo = GroupGeoMetadata::from_attributes(group.attributes())?;

    // CRS-without-transform is almost always a malformed-metadata bug —
    // the user thinks they have full georef but downstream spatial joins
    // will silently use the identity-pixel-coords default. Error loudly
    // so they fix the metadata rather than getting empty result sets.
    if geo.crs.is_some() && geo.transform.is_none() {
        return Err(ArrowError::InvalidArgumentError(format!(
            "Zarr group at {group_uri} declares a CRS but no `spatial:transform` \
             attribute; refusing to fall back to the identity transform because \
             that would silently produce wrong results in spatial joins. Declare \
             `spatial:transform` on the group or remove the CRS to read this as \
             a non-georeferenced datacube."
        )));
    }

    let arrays = group.child_arrays().map_err(|e| {
        ArrowError::ExternalError(Box::new(sedona_internal_datafusion_err!(
            "failed to enumerate child arrays in {group_uri}: {e}"
        )))
    })?;
    if arrays.is_empty() {
        return Err(ArrowError::InvalidArgumentError(format!(
            "Zarr group at {group_uri} has no child arrays"
        )));
    }

    let arrays = select_arrays(arrays, arrays_filter, group_uri)?;
    let array_infos = collect_array_infos(arrays)?;
    validate_group_constraints(&array_infos)?;

    // Spatial-dim resolution. Two configurations are accepted:
    //   - dim_names ends with ["y", "x"] (canonical for georeferenced
    //     2-D and time-series rasters); the spatial extent is the chunk's
    //     last two dims.
    //   - `spatial:dims` attribute on the group explicitly names them.
    // Anything else errors with a clear message — silently picking dims
    // would produce wrong per-row transforms.
    let spatial_dim_indices =
        resolve_spatial_dim_indices(&array_infos[0].dim_names, geo.spatial_dims.as_deref())?;

    let group_transform = match geo.transform {
        Some(t) => t,
        None => {
            // Both `spatial:transform` and `proj:*` are absent (the
            // CRS-only case errored above). Fall back to identity pixel
            // coordinates and breadcrumb a warning so spatial-join
            // surprises are debuggable.
            log::warn!(
                "Zarr group at {group_uri} has no `spatial:transform`; falling back \
                 to the identity pixel-coordinate transform [0, 1, 0, 0, 0, -1]. \
                 Spatial operations against this raster will use pixel coordinates."
            );
            [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]
        }
    };

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

/// Collect per-array metadata from open zarrs `Array` handles.
///
/// Sorts arrays by path so band ordering across rows is deterministic
/// (zarrs's underlying store listing order is implementation-defined —
/// filesystem stores currently happen to enumerate alphabetically, but
/// that's not part of the contract we want consumers to rely on).
fn collect_array_infos(
    mut arrays: Vec<Array<FilesystemStore>>,
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
            nodata,
        });
    }
    Ok(out)
}

/// Apply the array-selection rules. 1-D arrays (typical xarray-style
/// coord variables) are always dropped — a raster band requires at
/// least 2 dimensions, so reading a 1-D array could never succeed.
/// - Explicit filter: keep arrays whose path (with leading `/` stripped)
///   matches one of the requested names. Unknown names error so users
///   don't silently get an empty group from a typo; naming a 1-D array
///   errors with a clear message rather than producing a confusing
///   "no spatial axes" failure downstream.
/// - No filter: read every multi-dimensional array.
fn select_arrays(
    arrays: Vec<Array<FilesystemStore>>,
    filter: Option<&[String]>,
    group_uri: &str,
) -> Result<Vec<Array<FilesystemStore>>, ArrowError> {
    if let Some(names) = filter {
        let available: Vec<String> = arrays
            .iter()
            .map(|a| a.path().as_str().trim_start_matches('/').to_string())
            .collect();
        for requested in names {
            let needle = requested.trim_start_matches('/');
            match arrays
                .iter()
                .find(|a| a.path().as_str().trim_start_matches('/') == needle)
            {
                None => {
                    return Err(ArrowError::InvalidArgumentError(format!(
                        "Zarr group at {group_uri} has no array named {requested:?}; \
                         available arrays: {available:?}"
                    )));
                }
                Some(a) if a.shape().len() < 2 => {
                    return Err(ArrowError::InvalidArgumentError(format!(
                        "array {requested:?} has rank {} (shape {:?}); a raster band \
                         requires at least 2 dimensions and cannot be read.",
                        a.shape().len(),
                        a.shape()
                    )));
                }
                Some(_) => {}
            }
        }
        let kept: Vec<_> = arrays
            .into_iter()
            .filter(|a| {
                let path = a.path().as_str().trim_start_matches('/').to_string();
                names.iter().any(|n| n.trim_start_matches('/') == path)
            })
            .collect();
        return Ok(kept);
    }

    let kept: Vec<_> = arrays.into_iter().filter(|a| a.shape().len() > 1).collect();
    if kept.is_empty() {
        return Err(ArrowError::InvalidArgumentError(format!(
            "Zarr group at {group_uri} contains only 1-D arrays (typical \
             xarray-style coord variables); a raster band requires at least \
             2 dimensions, so this group has nothing readable as a raster"
        )));
    }
    Ok(kept)
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
/// dims and require they be named `y` and `x` (in that order) — the
/// canonical GeoZarr-2D convention. Anything else errors.
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
    if dim_names[n - 2] != "y" || dim_names[n - 1] != "x" {
        return Err(ArrowError::InvalidArgumentError(format!(
            "the last two dim_names must be [\"y\", \"x\"] when `spatial:dims` is \
             not declared; got {dim_names:?}",
        )));
    }
    Ok(vec![n - 2, n - 1])
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
/// The only pixel-byte read primitive in the crate. Consumed by the
/// async OutDb byte loader (`outdb_loader::ZarrLoader`).
pub(crate) fn retrieve_chunk_bytes(
    array: &Array<FilesystemStore>,
    chunk_indices: &[u64],
) -> Result<Vec<u8>, ArrowError> {
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
