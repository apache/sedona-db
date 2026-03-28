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

//! Fast affine-transformer rasterize wrapper.
//!
//! GDALRasterizeGeometries() will internally call GDALCreateGenImgProjTransformer2()
//! if pfnTransformer is NULL, even in the common case where only a GeoTransform-based
//! affine conversion from georeferenced coords to pixel/line is needed.
//!
//! This module supplies a minimal GDALTransformerFunc that applies the dataset
//! GeoTransform (and its inverse), avoiding expensive transformer creation.

use std::ffi::{c_char, c_int, c_void};
use std::ptr;

use crate::dataset::Dataset;
use crate::errors::{GdalError, Result};
use crate::gdal_api::{call_gdal_api, GdalApi};
use crate::gdal_dyn_bindgen::CE_None;
use crate::geo_transform::{GeoTransform, GeoTransformEx};
use crate::vector::geometry::Geometry;

#[repr(C)]
struct AffineTransformArg {
    gt: GeoTransform,
    inv_gt: GeoTransform,
}

unsafe extern "C" fn affine_transformer(
    p_transformer_arg: *mut c_void,
    b_dst_to_src: c_int,
    n_point_count: c_int,
    x: *mut f64,
    y: *mut f64,
    _z: *mut f64,
    pan_success: *mut c_int,
) -> c_int {
    if p_transformer_arg.is_null() || x.is_null() || y.is_null() || pan_success.is_null() {
        return 0;
    }
    if n_point_count < 0 {
        return 0;
    }

    // Treat transformer arg as immutable.
    let arg = &*(p_transformer_arg as *const AffineTransformArg);
    let affine = if b_dst_to_src == 0 {
        &arg.inv_gt
    } else {
        &arg.gt
    };

    let n = n_point_count as usize;
    for i in 0..n {
        // SAFETY: x/y/pan_success are assumed to point to arrays of length n_point_count.
        let xin = unsafe { *x.add(i) };
        let yin = unsafe { *y.add(i) };
        let (xout, yout) = affine.apply(xin, yin);
        unsafe {
            *x.add(i) = xout;
            *y.add(i) = yout;
            *pan_success.add(i) = 1;
        }
    }

    1
}

/// Rasterize geometries using the dataset geotransform as the transformer.
/// Geometry coordinates must already be in the dataset georeferenced coordinate space.
pub fn rasterize_affine(
    api: &'static GdalApi,
    dataset: &Dataset,
    bands: &[usize],
    geometries: &[Geometry],
    burn_values: &[f64],
    all_touched: bool,
) -> Result<()> {
    if bands.is_empty() {
        return Err(GdalError::BadArgument(
            "`bands` must not be empty".to_string(),
        ));
    }
    if burn_values.len() != geometries.len() {
        return Err(GdalError::BadArgument(format!(
            "Burn values length ({}) must match geometries length ({})",
            burn_values.len(),
            geometries.len()
        )));
    }

    let raster_count = dataset.raster_count();
    for band in bands {
        let is_good = *band > 0 && *band <= raster_count;
        if !is_good {
            return Err(GdalError::BadArgument(format!(
                "Band index {} is out of bounds",
                *band
            )));
        }
    }

    let bands_i32: Vec<c_int> = bands.iter().map(|&band| band as c_int).collect();

    // Keep this stack-allocated option array instead of going through `CslStringList`
    // so the affine fast path avoids extra allocation overhead for tiny geometries.
    let mut c_options: [*mut c_char; 2] = if all_touched {
        [c"ALL_TOUCHED=TRUE".as_ptr() as *mut c_char, ptr::null_mut()]
    } else {
        [
            c"ALL_TOUCHED=FALSE".as_ptr() as *mut c_char,
            ptr::null_mut(),
        ]
    };

    let geometries_c: Vec<_> = geometries.iter().map(|geo| geo.c_geometry()).collect();
    let burn_values_expanded: Vec<f64> = burn_values
        .iter()
        .flat_map(|burn| std::iter::repeat_n(burn, bands_i32.len()))
        .copied()
        .collect();

    let gt = dataset.geo_transform().map_err(|_e| {
        GdalError::BadArgument(
            "Missing geotransform: only geotransform-based affine rasterize is supported"
                .to_string(),
        )
    })?;
    let inv_gt = gt.invert().map_err(|_e| {
        GdalError::BadArgument(
            "Non-invertible geotransform: only geotransform-based affine rasterize is supported"
                .to_string(),
        )
    })?;
    let mut arg = AffineTransformArg { gt, inv_gt };

    unsafe {
        let error = call_gdal_api!(
            api,
            GDALRasterizeGeometries,
            dataset.c_dataset(),
            bands_i32.len() as c_int,
            bands_i32.as_ptr(),
            geometries_c.len() as c_int,
            geometries_c.as_ptr(),
            // SAFETY: `affine_transformer` has the GDAL transformer callback ABI.
            // This binding models the raw C API slot as `void*`, and on supported
            // targets we rely on the platform ABI allowing this callback pointer to
            // pass through that raw field unchanged.
            (affine_transformer as *const ()).cast::<c_void>() as *mut c_void,
            (&mut arg as *mut AffineTransformArg).cast::<c_void>(),
            burn_values_expanded.as_ptr(),
            // SAFETY: GDAL reads option strings through this mutable `char**` API.
            // The pointed-to string literals remain valid for the call and are not
            // expected to be modified by `GDALRasterizeGeometries`.
            c_options.as_mut_ptr(),
            ptr::null_mut(),
            ptr::null_mut()
        );
        if error != CE_None {
            return Err(api.last_cpl_err(error as u32));
        }
    }
    Ok(())
}

#[cfg(all(test, feature = "gdal-sys"))]
mod tests {
    use super::*;

    use crate::driver::{Driver, DriverManager};
    use crate::global::with_global_gdal_api;
    use crate::raster::rasterize::{rasterize, RasterizeOptions};
    use crate::raster::types::Buffer;

    fn mem_driver(api: &'static GdalApi) -> Driver {
        DriverManager::get_driver_by_name(api, "MEM").unwrap()
    }

    fn make_dataset_u8(
        api: &'static GdalApi,
        width: usize,
        height: usize,
        gt: GeoTransform,
    ) -> Result<Dataset> {
        let driver = mem_driver(api);
        let ds = driver.create_with_band_type::<u8>("", width, height, 1)?;
        ds.set_geo_transform(&gt)?;
        let band = ds.rasterband(1)?;
        let mut buf = Buffer::new((width, height), vec![0u8; width * height]);
        band.write((0, 0), (width, height), &mut buf)?;
        Ok(ds)
    }

    fn read_u8(ds: &Dataset, width: usize, height: usize) -> Vec<u8> {
        let band = ds.rasterband(1).unwrap();
        let buf = band
            .read_as::<u8>((0, 0), (width, height), (width, height), None)
            .unwrap();
        buf.data().to_vec()
    }

    fn poly_from_pixel_rect(
        api: &'static GdalApi,
        gt: &GeoTransform,
        x0: f64,
        y0: f64,
        x1: f64,
        y1: f64,
    ) -> Geometry {
        let (wx0, wy0) = gt.apply(x0, y0);
        let (wx1, wy1) = gt.apply(x1, y0);
        let (wx2, wy2) = gt.apply(x1, y1);
        let (wx3, wy3) = gt.apply(x0, y1);
        let wkt =
            format!("POLYGON (({wx0} {wy0}, {wx1} {wy1}, {wx2} {wy2}, {wx3} {wy3}, {wx0} {wy0}))");
        Geometry::from_wkt(api, &wkt).unwrap()
    }

    fn line_from_pixel_points(
        api: &'static GdalApi,
        gt: &GeoTransform,
        pts: &[(f64, f64)],
    ) -> Geometry {
        assert!(pts.len() >= 2);
        let mut s = String::from("LINESTRING (");
        for (i, (px, py)) in pts.iter().copied().enumerate() {
            let (wx, wy) = gt.apply(px, py);
            if i > 0 {
                s.push_str(", ");
            }
            s.push_str(&format!("{wx} {wy}"));
        }
        s.push(')');
        Geometry::from_wkt(api, &s).unwrap()
    }

    #[test]
    fn test_rasterize_affine_matches_baseline_north_up() {
        with_global_gdal_api(|api| {
            let (w, h) = (32usize, 24usize);
            let gt: GeoTransform = [100.0, 2.0, 0.0, 200.0, 0.0, -2.0];

            let geom_baseline = poly_from_pixel_rect(api, &gt, 3.2, 4.7, 20.4, 18.1);
            let geom_affine = poly_from_pixel_rect(api, &gt, 3.2, 4.7, 20.4, 18.1);

            let ds_baseline = make_dataset_u8(api, w, h, gt).unwrap();
            let ds_affine = make_dataset_u8(api, w, h, gt).unwrap();

            let geom_refs: Vec<&Geometry> = vec![&geom_baseline];
            rasterize(api, &ds_baseline, &[1], &geom_refs, &[1.0], None).unwrap();
            rasterize_affine(api, &ds_affine, &[1], &[geom_affine], &[1.0], false).unwrap();

            assert_eq!(read_u8(&ds_affine, w, h), read_u8(&ds_baseline, w, h));
        })
        .unwrap();
    }

    #[test]
    fn test_rasterize_affine_matches_baseline_rotated_gt_all_touched() {
        with_global_gdal_api(|api| {
            let (w, h) = (40usize, 28usize);
            // Rotated/skewed GeoTransform.
            let gt: GeoTransform = [10.0, 1.2, 0.15, 50.0, -0.1, -1.1];

            let geom_baseline = poly_from_pixel_rect(api, &gt, 5.25, 4.5, 25.75, 20.25);
            let geom_affine = poly_from_pixel_rect(api, &gt, 5.25, 4.5, 25.75, 20.25);

            let ds_baseline = make_dataset_u8(api, w, h, gt).unwrap();
            let ds_affine = make_dataset_u8(api, w, h, gt).unwrap();

            let geom_refs: Vec<&Geometry> = vec![&geom_baseline];
            rasterize(
                api,
                &ds_baseline,
                &[1],
                &geom_refs,
                &[1.0],
                Some(RasterizeOptions {
                    all_touched: true,
                    ..Default::default()
                }),
            )
            .unwrap();
            rasterize_affine(api, &ds_affine, &[1], &[geom_affine], &[1.0], true).unwrap();

            assert_eq!(read_u8(&ds_affine, w, h), read_u8(&ds_baseline, w, h));
        })
        .unwrap();
    }

    #[test]
    fn test_rasterize_affine_matches_baseline_linestring() {
        with_global_gdal_api(|api| {
            let (w, h) = (64usize, 48usize);
            // Rotated/skewed GeoTransform.
            let gt: GeoTransform = [5.0, 1.0, 0.2, 100.0, -0.15, -1.05];

            // A polyline with many vertices, defined in pixel/line space.
            let mut pts: Vec<(f64, f64)> = Vec::new();
            for i in 0..200 {
                let t = i as f64 / 199.0;
                let x = 2.625 + t * ((w as f64) - 5.25);
                let y = 5.25 + (t * 6.0).sin() * 8.0 + t * ((h as f64) - 12.25);
                pts.push((x, y));
            }
            let geom_baseline = line_from_pixel_points(api, &gt, &pts);
            let geom_affine = line_from_pixel_points(api, &gt, &pts);

            let ds_baseline = make_dataset_u8(api, w, h, gt).unwrap();
            let ds_affine = make_dataset_u8(api, w, h, gt).unwrap();

            let geom_refs: Vec<&Geometry> = vec![&geom_baseline];
            rasterize(api, &ds_baseline, &[1], &geom_refs, &[1.0], None).unwrap();
            rasterize_affine(api, &ds_affine, &[1], &[geom_affine], &[1.0], false).unwrap();

            let got = read_u8(&ds_affine, w, h);
            let expected = read_u8(&ds_baseline, w, h);
            if got != expected {
                let mut diffs = Vec::new();
                for (i, (a, b)) in got
                    .iter()
                    .copied()
                    .zip(expected.iter().copied())
                    .enumerate()
                {
                    if a != b {
                        let x = i % w;
                        let y = i / w;
                        diffs.push((x, y, a, b));
                    }
                }
                panic!(
                    "raster mismatch: {} differing pixels; first 10: {:?}",
                    diffs.len(),
                    &diffs[..diffs.len().min(10)]
                );
            }
        })
        .unwrap();
    }

    #[test]
    fn test_rasterize_affine_fails_on_noninvertible_gt() {
        with_global_gdal_api(|api| {
            let (w, h) = (8usize, 8usize);
            let gt: GeoTransform = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
            let ds = make_dataset_u8(api, w, h, gt).unwrap();
            let geom = Geometry::from_wkt(api, "POINT (0 0)").unwrap();
            let err = rasterize_affine(api, &ds, &[1], &[geom], &[1.0], true).unwrap_err();
            match err {
                GdalError::BadArgument(msg) => {
                    assert!(msg.contains("Non-invertible geotransform"));
                }
                other => panic!("Unexpected error: {other:?}"),
            }
        })
        .unwrap();
    }
}
