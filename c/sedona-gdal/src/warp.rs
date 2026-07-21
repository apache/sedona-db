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

//! GDAL warp / reproject API wrappers.
//!
//! Sits alongside the RasterIO resampling in [`crate::raster::rasterband`]: this
//! is the (re)gridding path that can move the output origin off the source grid,
//! grow the output extent beyond the source footprint, and change the CRS —
//! things RasterIO's read-into-a-buffer cannot do.

use std::ffi::CString;
use std::os::raw::c_void;
use std::ptr::{null, null_mut};

use crate::dataset::Dataset;
use crate::errors::{GdalError, Result};
use crate::gdal_api::{call_gdal_api, GdalApi};
use crate::gdal_dyn_bindgen::{CE_Failure, CE_None};
use crate::geo_transform::GeoTransform;
use crate::raster::types::ResampleAlg;
use crate::spatial_ref::SpatialRef;

/// Reproject/warp `src` into the already-created `dst` dataset.
///
/// Both datasets carry their own spatial reference (set via `set_projection` /
/// `set_spatial_ref`), so the warp reprojects from the source SRS to the
/// destination SRS, resampling with `alg`. The output grid — origin, pixel size,
/// dimensions — is whatever `dst` was created with.
///
/// Destination cells that the (reprojected) source footprint does not cover are
/// left **untouched**: `GDALReprojectImage` with no warp options writes only
/// covered pixels. Callers that grow the extent must therefore pre-fill `dst`'s
/// band buffers with the desired background/nodata value before warping.
pub fn reproject_image(
    api: &'static GdalApi,
    src: &Dataset,
    dst: &Dataset,
    alg: ResampleAlg,
) -> Result<()> {
    let gra = alg.to_gdal_warp().ok_or_else(|| {
        GdalError::BadArgument(format!(
            "resample algorithm {alg:?} is not supported by the warp API"
        ))
    })?;

    let rv = unsafe {
        call_gdal_api!(
            api,
            GDALReprojectImage,
            src.c_dataset(),
            null(), // src WKT: NULL means "use the source dataset's own SRS"
            dst.c_dataset(),
            null(), // dst WKT: NULL means "use the destination dataset's own SRS"
            gra,
            0.0,        // warp memory limit (0 = GDAL default)
            0.0,        // max error in pixels (0 = exact transformer, no approximation)
            None,       // progress callback
            null_mut(), // progress arg
            null_mut()  // GDALWarpOptions* (NULL = defaults)
        )
    };
    if rv != CE_None {
        return Err(api.last_cpl_err(CE_Failure as u32));
    }
    Ok(())
}

/// A GDAL image-to-image reprojection transformer.
///
/// Owns the argument returned by `GDALCreateGenImgProjTransformer` and destroys
/// it on drop. Its sole use here is feeding [`suggested_warp_output`], which asks
/// GDAL what output grid a reprojection into the target CRS should use.
pub struct GenImgProjTransformer {
    api: &'static GdalApi,
    handle: *mut c_void,
}

// SAFETY: `GenImgProjTransformer` uniquely owns its transformer handle and only
// moves that ownership across threads; the handle is destroyed exactly once on
// drop and this wrapper offers no shared concurrent access, so `Send` is sound.
unsafe impl Send for GenImgProjTransformer {}

impl Drop for GenImgProjTransformer {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { call_gdal_api!(self.api, GDALDestroyGenImgProjTransformer, self.handle) };
        }
    }
}

impl GenImgProjTransformer {
    /// Create a transformer mapping `src` (using its own SRS) into `dst_srs`.
    pub fn new(api: &'static GdalApi, src: &Dataset, dst_srs: &SpatialRef) -> Result<Self> {
        let dst_wkt = dst_srs.to_wkt()?;
        let c_dst_wkt = CString::new(dst_wkt)?;
        let handle = unsafe {
            call_gdal_api!(
                api,
                GDALCreateGenImgProjTransformer,
                src.c_dataset(),
                null(),             // src WKT: use the source dataset's own SRS
                null_mut(),         // dst dataset: none (target given as WKT below)
                c_dst_wkt.as_ptr(), // dst WKT
                0,                  // bGCPUseOK
                0.0,                // dfGCPErrorThreshold
                0                   // nOrder
            )
        };
        if handle.is_null() {
            return Err(api.last_null_pointer_err("GDALCreateGenImgProjTransformer"));
        }
        Ok(Self { api, handle })
    }
}

/// Compute the output geotransform and pixel dimensions a reprojection of `src`
/// through `transformer` should use (GDAL's `GDALSuggestedWarpOutput`).
///
/// The returned geotransform is north-up (no skew) in the target CRS; this is
/// the same computation `rasterio.warp.calculate_default_transform` performs.
pub fn suggested_warp_output(
    api: &'static GdalApi,
    src: &Dataset,
    transformer: &GenImgProjTransformer,
) -> Result<(GeoTransform, usize, usize)> {
    let pfn = api.gen_img_proj_transform_fn()?;

    let mut gt: GeoTransform = [0.0; 6];
    let mut n_pixels: i32 = 0;
    let mut n_lines: i32 = 0;
    let rv = unsafe {
        call_gdal_api!(
            api,
            GDALSuggestedWarpOutput,
            src.c_dataset(),
            pfn,
            transformer.handle,
            gt.as_mut_ptr(),
            &mut n_pixels,
            &mut n_lines
        )
    };
    if rv != CE_None {
        return Err(api.last_cpl_err(CE_Failure as u32));
    }
    if n_pixels <= 0 || n_lines <= 0 {
        return Err(GdalError::BadArgument(format!(
            "GDALSuggestedWarpOutput returned non-positive size {n_pixels}x{n_lines}"
        )));
    }
    Ok((gt, n_pixels as usize, n_lines as usize))
}

#[cfg(all(test, feature = "gdal-sys"))]
mod tests {
    use crate::global::with_global_gdal;
    use crate::raster::types::{Buffer, GdalDataType, ResampleAlg};

    #[test]
    fn reproject_same_crs_upsamples_into_dst_grid() {
        with_global_gdal(|gdal| {
            // 2x1 source, values [10, 20], extent x[0,4] y[0,2], EPSG:4326.
            let src = gdal
                .create_mem_dataset(2, 1, 1, GdalDataType::UInt8)
                .unwrap();
            src.set_geo_transform(&[0.0, 2.0, 0.0, 2.0, 0.0, -2.0])
                .unwrap();
            src.set_projection("EPSG:4326").unwrap();
            let mut buf = Buffer::new((2, 1), vec![10u8, 20]);
            src.rasterband(1)
                .unwrap()
                .write((0, 0), (2, 1), &mut buf)
                .unwrap();

            // 4x2 destination on the same extent (extent-preserving upsample).
            let dst = gdal
                .create_mem_dataset(4, 2, 1, GdalDataType::UInt8)
                .unwrap();
            dst.set_geo_transform(&[0.0, 1.0, 0.0, 2.0, 0.0, -1.0])
                .unwrap();
            dst.set_projection("EPSG:4326").unwrap();

            gdal.reproject_image(&src, &dst, ResampleAlg::NearestNeighbour)
                .unwrap();

            let out = dst
                .rasterband(1)
                .unwrap()
                .read_as::<u8>((0, 0), (4, 2), (4, 2), None)
                .unwrap();
            // Nearest 2x integer upsample replicates each source pixel 2x2.
            assert_eq!(out.data(), [10, 10, 20, 20, 10, 10, 20, 20]);
        })
        .unwrap();
    }

    #[test]
    fn suggested_warp_output_reprojects_extent() {
        with_global_gdal(|gdal| {
            // A small EPSG:4326 raster; suggest the output grid for EPSG:3857.
            let src = gdal
                .create_mem_dataset(8, 8, 1, GdalDataType::UInt8)
                .unwrap();
            src.set_geo_transform(&[10.0, 0.5, 0.0, 50.0, 0.0, -0.5])
                .unwrap();
            src.set_projection("EPSG:4326").unwrap();

            let dst_srs = gdal.spatial_ref_from_definition("EPSG:3857").unwrap();
            let (gt, px, ln) = gdal.suggested_warp_output(&src, &dst_srs).unwrap();

            assert!(px > 0 && ln > 0);
            // Web Mercator origin x for 10 deg lon is ~1.11e6 m; the suggested
            // grid is north-up (no skew) and its pixel size is positive/negative.
            assert!(
                gt[2] == 0.0 && gt[4] == 0.0,
                "suggested transform is north-up"
            );
            assert!(gt[1] > 0.0 && gt[5] < 0.0, "positive x res, negative y res");
            assert!(
                gt[0] > 1.0e6 && gt[0] < 1.2e6,
                "3857 origin x near 1.11e6, got {}",
                gt[0]
            );
        })
        .unwrap();
    }
}
