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

//! Deriving a 2-D geotransform from a Zarr group's 1-D spatial coordinate
//! arrays, for CF-style datasets that carry coordinate variables but no
//! explicit `spatial:transform`.
//!
//! An affine transform can only represent a *uniform* grid, so the only
//! usable coordinate arrays are numeric ones with a constant step. We read the
//! values, confirm regular spacing, and synthesize the transform — treating
//! the coordinate values as cell centers (a half-step shift gives the cell-edge
//! origin GDAL expects). `latitude`/`longitude`-named axes additionally imply
//! geographic `EPSG:4326`; generic `y`/`x` get the transform but no CRS (the
//! coordinate space is unlabeled — attach a CRS with `RS_SetCRS`).

use arrow_schema::ArrowError;
use sedona_common::sedona_internal_datafusion_err;
use zarrs::array::data_type::{
    Float32DataType, Float64DataType, Int16DataType, Int32DataType, Int64DataType, Int8DataType,
    UInt16DataType, UInt32DataType, UInt64DataType, UInt8DataType,
};
use zarrs::array::{Array, ArraySubset};
use zarrs::storage::AsyncReadableListableStorage;

/// Relative tolerance for the regular-spacing check, generous enough to absorb
/// float32 coordinate rounding (e.g. a 0.25° step stored as `f32`).
const STEP_REL_TOL: f64 = 1e-3;

/// Read a 1-D coordinate array's values as `f64`.
///
/// Returns `Ok(None)` — not an error — when the array is absent, not 1-D, or
/// has a non-numeric dtype, so the caller can fall back gracefully.
pub(crate) async fn read_coord_values(
    storage: &AsyncReadableListableStorage,
    name: &str,
) -> Result<Option<Vec<f64>>, ArrowError> {
    let path = format!("/{}", name.trim_start_matches('/'));
    let Ok(array) = Array::async_open(storage.clone(), &path).await else {
        return Ok(None);
    };
    if array.shape().len() != 1 {
        return Ok(None);
    }
    let subset = ArraySubset::new_with_shape(array.shape().to_vec());

    macro_rules! retrieve_as {
        ($t:ty) => {{
            let values: Vec<$t> = array
                .async_retrieve_array_subset::<Vec<$t>>(&subset)
                .await
                .map_err(|e| {
                    ArrowError::ExternalError(Box::new(sedona_internal_datafusion_err!(
                        "failed to read coordinate array {path}: {e}"
                    )))
                })?;
            // Lossless for the float dtypes and exact for the small-magnitude
            // integer coordinates seen in practice; `as f64` can lose precision
            // only for i64/u64 magnitudes above 2^53, which a coordinate axis
            // would never reach.
            values.into_iter().map(|v| v as f64).collect::<Vec<f64>>()
        }};
    }

    let dt = array.data_type();
    let values = if dt.is::<Float64DataType>() {
        retrieve_as!(f64)
    } else if dt.is::<Float32DataType>() {
        retrieve_as!(f32)
    } else if dt.is::<Int64DataType>() {
        retrieve_as!(i64)
    } else if dt.is::<Int32DataType>() {
        retrieve_as!(i32)
    } else if dt.is::<Int16DataType>() {
        retrieve_as!(i16)
    } else if dt.is::<Int8DataType>() {
        retrieve_as!(i8)
    } else if dt.is::<UInt64DataType>() {
        retrieve_as!(u64)
    } else if dt.is::<UInt32DataType>() {
        retrieve_as!(u32)
    } else if dt.is::<UInt16DataType>() {
        retrieve_as!(u16)
    } else if dt.is::<UInt8DataType>() {
        retrieve_as!(u8)
    } else {
        // Non-numeric coordinate (e.g. fixed-length string, datetime): not
        // usable for a numeric affine, so fall back rather than erroring.
        return Ok(None);
    };
    Ok(Some(values))
}

/// The uniform step of a regularly-spaced coordinate array, or `None` if it is
/// too short, zero-width, or irregular (beyond [`STEP_REL_TOL`]).
pub(crate) fn regular_step(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    // An interior NaN/inf would slip past the per-gap check below (every
    // comparison against NaN is false), so reject non-finite values up front.
    if values.iter().any(|v| !v.is_finite()) {
        return None;
    }
    // Average step, so float rounding doesn't bias the result toward the first
    // gap; then confirm every gap matches it within tolerance.
    let step = (values[values.len() - 1] - values[0]) / (values.len() - 1) as f64;
    if step == 0.0 || !step.is_finite() {
        return None;
    }
    let tol = step.abs() * STEP_REL_TOL;
    if values
        .windows(2)
        .any(|w| ((w[1] - w[0]) - step).abs() > tol)
    {
        return None;
    }
    Some(step)
}

/// Synthesize a GDAL geotransform
/// `[origin_x, scale_x, skew_x, origin_y, skew_y, scale_y]` from regular 1-D
/// `x` (column) and `y` (row) coordinate *cell-center* values.
///
/// Coordinates are cell centers, so each origin is shifted half a step back to
/// the cell edge. 1-D rectilinear coordinates carry no rotation, so the skews
/// are zero. Returns `None` if either axis is not regularly spaced.
pub(crate) fn transform_from_coords(x: &[f64], y: &[f64]) -> Option<[f64; 6]> {
    let scale_x = regular_step(x)?;
    let scale_y = regular_step(y)?;
    Some([
        x[0] - scale_x / 2.0,
        scale_x,
        0.0,
        y[0] - scale_y / 2.0,
        0.0,
        scale_y,
    ])
}

/// Infer a geographic CRS from the spatial dimension names: `lat`/`latitude`
/// paired with `lon`/`longitude` implies `EPSG:4326`. Generic `y`/`x` carry no
/// implied CRS — the derived transform is still valid, just in an unlabeled
/// coordinate space (attach a CRS with `RS_SetCRS`).
pub(crate) fn infer_geographic_crs(y_name: &str, x_name: &str) -> Option<&'static str> {
    let is_lat = |s: &str| matches!(s.to_ascii_lowercase().as_str(), "lat" | "latitude");
    let is_lon = |s: &str| matches!(s.to_ascii_lowercase().as_str(), "lon" | "longitude");
    (is_lat(y_name) && is_lon(x_name)).then_some("EPSG:4326")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regular_step_detects_uniform_spacing() {
        assert_eq!(regular_step(&[0.0, 0.25, 0.5, 0.75]), Some(0.25));
        // Descending (e.g. north-to-south latitude) yields a negative step.
        assert_eq!(regular_step(&[45.0, 44.75, 44.5]), Some(-0.25));
        // float32-style rounding within tolerance still reads as regular; the
        // returned step is the average, so compare approximately.
        let step = regular_step(&[0.0, 0.1, 0.2000001, 0.3]).unwrap();
        assert!((step - 0.1).abs() < 1e-6, "step={step}");
    }

    #[test]
    fn regular_step_rejects_irregular_or_degenerate() {
        assert_eq!(regular_step(&[0.0, 1.0, 3.0]), None); // irregular
        assert_eq!(regular_step(&[5.0]), None); // too short
        assert_eq!(regular_step(&[2.0, 2.0, 2.0]), None); // zero step
    }

    #[test]
    fn regular_step_rejects_non_finite() {
        // A non-finite value anywhere (incl. an interior one, which the per-gap
        // check can't catch) must be rejected.
        assert_eq!(regular_step(&[0.0, f64::NAN, 2.0]), None);
        assert_eq!(regular_step(&[0.0, 1.0, f64::INFINITY]), None);
        assert_eq!(regular_step(&[f64::NAN, 1.0, 2.0]), None);
    }

    #[test]
    fn transform_from_coords_shifts_to_cell_edge() {
        // Hurricane-style: lon 265..320 step 0.25, lat 45..15 step -0.25.
        let lon: Vec<f64> = (0..221).map(|i| 265.0 + i as f64 * 0.25).collect();
        let lat: Vec<f64> = (0..121).map(|i| 45.0 - i as f64 * 0.25).collect();
        let t = transform_from_coords(&lon, &lat).unwrap();
        assert_eq!(t, [264.875, 0.25, 0.0, 45.125, 0.0, -0.25]);
    }

    #[test]
    fn transform_from_coords_none_when_irregular() {
        assert_eq!(transform_from_coords(&[0.0, 1.0, 3.0], &[0.0, 1.0]), None);
    }

    #[test]
    fn infer_crs_only_for_lat_lon() {
        assert_eq!(
            infer_geographic_crs("latitude", "longitude"),
            Some("EPSG:4326")
        );
        assert_eq!(infer_geographic_crs("lat", "lon"), Some("EPSG:4326"));
        assert_eq!(infer_geographic_crs("y", "x"), None);
    }
}
