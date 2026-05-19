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

//! Zarr group → N-D raster `StructArray` entry points.
//!
//! Both `group_to_indb_rasters` and `group_to_outdb_rasters` produce the
//! same row shape: one raster row per chunk position, with one band per
//! array in the group. They differ only in how each row's pixel bytes
//! are delivered:
//!
//! - **InDb** — every chunk is fetched eagerly and copied into the
//!   Arrow `data` column. Heavy for large datacubes; intended for
//!   snapshots.
//! - **OutDb** — `data` is left empty; each band's `outdb_uri` carries a
//!   chunk anchor (`zarr://<store-uri>/<array-path>#chunk=i0,i1,...`).
//!   Byte resolution awaits the format-keyed dispatch work in a
//!   follow-up PR.

use std::sync::Arc;

use arrow_array::StructArray;
use arrow_schema::ArrowError;
use sedona_common::sedona_internal_datafusion_err;
use sedona_raster::builder::RasterBuilder;
use sedona_schema::raster::BandDataType;
use zarrs::array::{Array, ArrayBytes};
use zarrs::group::Group;
use zarrs_filesystem::FilesystemStore;

use crate::dtype::zarr_to_band_data_type;
use crate::geozarr::GroupGeoMetadata;
use crate::source_uri::{build_chunk_anchor, group_uri_to_filesystem_path};

/// Open a Zarr group and eagerly fetch every chunk's bytes into the
/// returned `StructArray`. Each row holds one chunk position's data
/// across every array in the group.
pub fn group_to_indb_rasters(group_uri: &str) -> Result<StructArray, ArrowError> {
    build_rasters(group_uri, Mode::InDb)
}

/// Open a Zarr group and emit one row per chunk position with chunk-anchor
/// URIs in each band's `outdb_uri`. The `data` column is empty; bytes
/// resolve on demand through whichever OutDb loader is registered for
/// the `zarr` format.
pub fn group_to_outdb_rasters(group_uri: &str) -> Result<StructArray, ArrowError> {
    build_rasters(group_uri, Mode::OutDb)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    InDb,
    OutDb,
}

/// Per-array metadata extracted once at group open and reused for every
/// chunk position. Caching this avoids re-reading Zarr metadata for each
/// of the (potentially thousands of) chunk rows.
struct ArrayInfo {
    /// Array path within the store, used to build chunk anchor URIs and
    /// surface in band names.
    path: String,
    /// Open zarrs handle.
    array: Array<FilesystemStore>,
    /// SedonaDB BandDataType corresponding to this array's zarrs dtype.
    data_type: BandDataType,
    /// Dimension names in array order. Required to be `Some(_)` for every
    /// dim; missing names error at validation time.
    dim_names: Vec<String>,
    /// Inner chunk grid shape, one entry per dimension. Used to enumerate
    /// chunk positions and validated to match across arrays.
    chunk_grid_shape: Vec<u64>,
    /// Chunk shape (elements per chunk per dim). Same for every chunk
    /// position in Phase 1 (no ragged final chunks emitted as separate
    /// short rows).
    chunk_shape: Vec<u64>,
    /// Encoded fill value in native-endian byte representation, for the
    /// `nodata` field. None when the array has no fill value declared.
    nodata: Option<Vec<u8>>,
}

fn build_rasters(group_uri: &str, mode: Mode) -> Result<StructArray, ArrowError> {
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

    let array_infos = collect_array_infos(arrays)?;
    validate_group_constraints(&array_infos)?;

    // Spatial-dim resolution. Phase 1 supports two configurations:
    //   - dim_names ends with ["y", "x"] (canonical for georeferenced
    //     2-D and time-series rasters); the spatial extent is the chunk's
    //     last two dims.
    //   - `spatial:dims` attribute on the group explicitly names them.
    // Anything else errors with a clear message — silently picking dims
    // would produce wrong per-row transforms.
    let spatial_dim_indices =
        resolve_spatial_dim_indices(&array_infos[0].dim_names, geo.spatial_dims.as_deref())?;
    let spatial_dims_names: Vec<&str> = spatial_dim_indices
        .iter()
        .map(|&i| array_infos[0].dim_names[i].as_str())
        .collect();
    let chunk_spatial_shape: Vec<i64> = spatial_dim_indices
        .iter()
        .map(|&i| array_infos[0].chunk_shape[i] as i64)
        .collect();

    let group_transform = geo.transform.unwrap_or([0.0, 1.0, 0.0, 0.0, 0.0, -1.0]);

    let total_rows = array_infos[0].chunk_grid_shape.iter().product::<u64>() as usize;
    let mut builder = RasterBuilder::new(total_rows);

    // Walk the chunk grid in row-major (C-order) order. The outer-most
    // axis varies slowest, the innermost fastest — same convention used
    // for byte strides in `BandRefImpl`.
    let mut chunk_indices = vec![0u64; array_infos[0].chunk_grid_shape.len()];
    loop {
        let row_transform = compute_row_transform(
            &group_transform,
            &chunk_indices,
            &array_infos[0].chunk_shape,
            &spatial_dim_indices,
        );
        let crs_str = geo.crs.as_deref();
        builder.start_raster_nd(
            &row_transform,
            &spatial_dims_names,
            &chunk_spatial_shape,
            crs_str,
        )?;

        for info in &array_infos {
            let dim_names_ref: Vec<&str> = info.dim_names.iter().map(String::as_str).collect();
            let nodata_ref = info.nodata.as_deref();
            let anchor;
            let (outdb_uri_arg, outdb_format_arg) = match mode {
                Mode::InDb => (None, None),
                Mode::OutDb => {
                    anchor = build_chunk_anchor(group_uri, &info.path, &chunk_indices);
                    (Some(anchor.as_str()), Some("zarr"))
                }
            };
            builder.start_band_nd(
                Some(info.path.as_str()),
                &dim_names_ref,
                &info.chunk_shape,
                info.data_type,
                nodata_ref,
                outdb_uri_arg,
                outdb_format_arg,
            )?;
            match mode {
                Mode::InDb => {
                    let bytes = retrieve_chunk_bytes(&info.array, &chunk_indices)?;
                    builder.band_data_writer().append_value(&bytes);
                }
                Mode::OutDb => {
                    // Schema-OutDb: empty `data` column. Byte resolution
                    // routes through the OutDb loader when a downstream
                    // consumer calls contiguous_data / nd_buffer.
                    builder.band_data_writer().append_value([0u8; 0]);
                }
            }
            builder.finish_band()?;
        }
        builder.finish_raster()?;

        if !advance_chunk_indices(&mut chunk_indices, &array_infos[0].chunk_grid_shape) {
            break;
        }
    }

    builder.finish()
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
        let dim_names = match array.dimension_names() {
            Some(names) => names
                .iter()
                .enumerate()
                .map(|(i, n)| {
                    n.clone().ok_or_else(|| {
                        ArrowError::InvalidArgumentError(format!(
                            "array {path}: dimension {i} has no name; Phase 1 requires every \
                             Zarr array dimension to be named",
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
            None => {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "array {path}: dimension_names is absent; Phase 1 requires every Zarr \
                     array to declare dimension_names",
                )));
            }
        };
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
            array,
            data_type,
            dim_names,
            chunk_grid_shape,
            chunk_shape,
            nodata,
        });
    }
    Ok(out)
}

/// Enforce Phase 1 group constraints. All arrays must agree on chunk grid
/// shape, chunk shape, and dimension names. We do NOT enforce shared
/// element shape (`array.shape()`) because users routinely group
/// arrays with the same chunk grid but different totals (e.g. a coord
/// variable with one fewer dim is rejected here anyway by the dim-name
/// check).
fn validate_group_constraints(infos: &[ArrayInfo]) -> Result<(), ArrowError> {
    let first = &infos[0];
    for other in &infos[1..] {
        if other.chunk_grid_shape != first.chunk_grid_shape {
            return Err(ArrowError::InvalidArgumentError(format!(
                "arrays {} and {} have different chunk grid shapes ({:?} vs {:?}); \
                 Phase 1 requires a shared chunk grid across the group",
                first.path, other.path, first.chunk_grid_shape, other.chunk_grid_shape
            )));
        }
        if other.chunk_shape != first.chunk_shape {
            return Err(ArrowError::InvalidArgumentError(format!(
                "arrays {} and {} have different chunk shapes ({:?} vs {:?}); \
                 Phase 1 requires a shared chunk shape across the group",
                first.path, other.path, first.chunk_shape, other.chunk_shape
            )));
        }
        if other.dim_names != first.dim_names {
            return Err(ArrowError::InvalidArgumentError(format!(
                "arrays {} and {} have different dimension names ({:?} vs {:?}); \
                 Phase 1 requires identical dim_names across the group",
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
            "Phase 1 requires at least 2 dimensions to resolve spatial axes; got {dim_names:?}",
        )));
    }
    if dim_names[n - 2] != "y" || dim_names[n - 1] != "x" {
        return Err(ArrowError::InvalidArgumentError(format!(
            "Phase 1 expects the last two dim_names to be [\"y\", \"x\"] when \
             `spatial:dims` is not declared; got {dim_names:?}",
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
    // Phase 1 assumes spatial_dim_indices == [y_index, x_index] (validated
    // upstream). Index 0 is the y axis, index 1 is the x axis.
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
/// Phase 1 uses `ArrayBytes::Fixed`, so this errors for variable-length
/// element types — those don't have a `BandDataType` counterpart anyway,
/// so the dtype check in `collect_array_infos` rejects them upstream.
fn retrieve_chunk_bytes(
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
            "array {}: variable-length chunk bytes not supported in Phase 1",
            array.path()
        ))
    })?;
    Ok(raw.into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

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
