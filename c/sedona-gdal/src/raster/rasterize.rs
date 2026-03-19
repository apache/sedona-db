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

//! Ported (and contains copied code) from georust/gdal:
//! <https://github.com/georust/gdal/blob/v0.19.0/src/raster/rasterize.rs>.
//! Original code is licensed under MIT.

use std::ptr;

use crate::cpl::CslStringList;
use crate::dataset::Dataset;
use crate::errors::{GdalError, Result};
use crate::gdal_api::{call_gdal_api, GdalApi};
use crate::gdal_dyn_bindgen::*;
use crate::vector::geometry::Geometry;

/// Source of burn values.
#[derive(Copy, Clone, Debug)]
pub enum BurnSource {
    /// Use whatever `burn_values` argument is supplied to `rasterize`.
    UserSupplied,

    /// Add the geometry's Z value to whatever `burn_values` argument
    /// is supplied to `rasterize`.
    Z,
}

/// Algorithm for merging new raster values with existing values.
#[derive(Copy, Clone, Debug)]
pub enum MergeAlgorithm {
    /// Overwrite existing value (default).
    Replace,
    /// Add new value to existing value (useful for heatmaps).
    Add,
}

/// Optimization mode for rasterization.
#[derive(Copy, Clone, Debug)]
pub enum OptimizeMode {
    /// Let GDAL decide (default).
    Automatic,
    /// Force raster-based scan (iterates over pixels).
    Raster,
    /// Force vector-based scan (iterates over geometry edges).
    Vector,
}

/// Options that specify how to rasterize geometries.
#[derive(Copy, Clone, Debug)]
pub struct RasterizeOptions {
    /// Set to `true` to set all pixels touched by the line or polygons,
    /// not just those whose center is within the polygon or that are
    /// selected by Bresenham's line algorithm. Defaults to `false`.
    pub all_touched: bool,

    /// May be set to `BurnSource::Z` to use the Z values of the geometries.
    /// `burn_value` is added to this before burning. Defaults to
    /// `BurnSource::UserSupplied` in which case just the `burn_value` is burned.
    pub source: BurnSource,

    /// May be `MergeAlgorithm::Replace` (default) or `MergeAlgorithm::Add`.
    /// `Replace` overwrites existing values; `Add` adds to them.
    pub merge_algorithm: MergeAlgorithm,

    /// The height in lines of the chunk to operate on. `0` (default) lets GDAL
    /// choose based on cache size. Not used in `OPTIM=RASTER` mode.
    pub chunk_y_size: usize,

    /// Optimization mode for rasterization.
    pub optimize: OptimizeMode,
}

impl Default for RasterizeOptions {
    fn default() -> Self {
        RasterizeOptions {
            all_touched: false,
            source: BurnSource::UserSupplied,
            merge_algorithm: MergeAlgorithm::Replace,
            chunk_y_size: 0,
            optimize: OptimizeMode::Automatic,
        }
    }
}

impl RasterizeOptions {
    /// Build a GDAL option list from these rasterize options.
    pub fn to_options_list(self) -> Result<CslStringList> {
        let mut options = CslStringList::with_capacity(5);

        options.set_name_value(
            "ALL_TOUCHED",
            if self.all_touched { "TRUE" } else { "FALSE" },
        )?;

        options.set_name_value(
            "MERGE_ALG",
            match self.merge_algorithm {
                MergeAlgorithm::Replace => "REPLACE",
                MergeAlgorithm::Add => "ADD",
            },
        )?;

        options.set_name_value("CHUNKYSIZE", &self.chunk_y_size.to_string())?;

        options.set_name_value(
            "OPTIM",
            match self.optimize {
                OptimizeMode::Automatic => "AUTO",
                OptimizeMode::Raster => "RASTER",
                OptimizeMode::Vector => "VECTOR",
            },
        )?;

        if let BurnSource::Z = self.source {
            options.set_name_value("BURN_VALUE_FROM", "Z")?;
        }

        Ok(options)
    }
}

/// Rasterize geometries into the selected dataset bands.
/// Supply one burn value per geometry; values are replicated across the target bands.
pub fn rasterize(
    api: &'static GdalApi,
    dataset: &Dataset,
    band_list: &[i32],
    geometries: &[&Geometry],
    burn_values: &[f64],
    options: Option<RasterizeOptions>,
) -> Result<()> {
    if band_list.is_empty() {
        return Err(GdalError::BadArgument(
            "`band_list` must not be empty".to_string(),
        ));
    }
    if burn_values.len() != geometries.len() {
        return Err(GdalError::BadArgument(format!(
            "burn_values length ({}) must match geometries length ({})",
            burn_values.len(),
            geometries.len()
        )));
    }
    let raster_count = dataset.raster_count();
    for &band in band_list {
        let is_good = band > 0 && (band as usize) <= raster_count;
        if !is_good {
            return Err(GdalError::BadArgument(format!(
                "Band index {} is out of bounds",
                band
            )));
        }
    }

    let geom_handles: Vec<OGRGeometryH> = geometries.iter().map(|g| g.c_geometry()).collect();

    // Replicate each burn value across all bands, matching the GDAL C API
    // contract that expects nGeomCount * nBandCount burn values.
    let expanded_burn_values: Vec<f64> = burn_values
        .iter()
        .flat_map(|burn| std::iter::repeat_n(burn, band_list.len()))
        .copied()
        .collect();

    let opts = options.unwrap_or_default();
    let csl = opts.to_options_list()?;

    let n_band_count: i32 = band_list.len().try_into()?;
    let n_geom_count: i32 = geom_handles.len().try_into()?;

    let rv = unsafe {
        call_gdal_api!(
            api,
            GDALRasterizeGeometries,
            dataset.c_dataset(),
            n_band_count,
            band_list.as_ptr(),
            n_geom_count,
            geom_handles.as_ptr(),
            ptr::null_mut(), // pfnTransformer
            ptr::null_mut(), // pTransformArg
            expanded_burn_values.as_ptr(),
            csl.as_ptr(),
            ptr::null_mut(), // pfnProgress
            ptr::null_mut()  // pProgressData
        )
    };
    if rv != CE_None {
        return Err(api.last_cpl_err(rv as u32));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::DriverManager;
    use crate::global::with_global_gdal_api;

    #[test]
    fn test_rasterizeoptions_as_ptr() {
        let c_options = RasterizeOptions::default().to_options_list().unwrap();
        assert_eq!(
            c_options.fetch_name_value("ALL_TOUCHED"),
            Some("FALSE".to_string())
        );
        assert_eq!(c_options.fetch_name_value("BURN_VALUE_FROM"), None);
        assert_eq!(
            c_options.fetch_name_value("MERGE_ALG"),
            Some("REPLACE".to_string())
        );
        assert_eq!(
            c_options.fetch_name_value("CHUNKYSIZE"),
            Some("0".to_string())
        );
        assert_eq!(
            c_options.fetch_name_value("OPTIM"),
            Some("AUTO".to_string())
        );
    }

    #[cfg(feature = "gdal-sys")]
    #[test]
    fn test_rasterize() {
        with_global_gdal_api(|api| {
            let wkt = "POLYGON ((2 2, 2 4.25, 4.25 4.25, 4.25 2, 2 2))";
            let poly = Geometry::from_wkt(api, wkt).unwrap();

            let driver = DriverManager::get_driver_by_name(api, "MEM").unwrap();
            let dataset = driver.create("", 5, 5, 1).unwrap();

            let bands = [1];
            let geometries = [&poly];
            let burn_values = [1.0];
            rasterize(api, &dataset, &bands, &geometries, &burn_values, None).unwrap();

            let rb = dataset.rasterband(1).unwrap();
            let values = rb.read_as::<u8>((0, 0), (5, 5), (5, 5), None).unwrap();
            assert_eq!(
                values.data(),
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
            );
        })
        .unwrap();
    }
}
