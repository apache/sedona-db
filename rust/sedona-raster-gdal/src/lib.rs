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

//! GDAL integration for Apache SedonaDB raster types.
//!
//! This crate provides GDAL-based utilities for raster operations:
//!
//! ## Dataset Conversion (`dataset` module)
//! - For in-db rasters: Creates a GDALDataset backed by the MEM driver, with zero-copy
//!   access to the band data stored in the Arrow array.
//! - For out-db rasters: Creates a GDALDataset as a VRT (Virtual Raster) that references
//!   the external data sources.
//!
//! ## UDF Functions
//! - `RS_FromPath`: Load out-db raster from file path
//! - `RS_FromGDALRaster`: Parse binary content using GDAL driver as in-db raster
//! - `RS_AsGeoTiff`: Export raster as GeoTiff binary

pub mod raster_band_reader;
pub mod rs_as_geotiff;
pub mod rs_as_raster;
pub mod rs_clip;
pub mod rs_from_gdal_raster;
pub mod rs_from_path;
pub mod rs_geotiff_tiles;
pub mod rs_map_algebra;
pub mod rs_metadata;
pub mod rs_polygonize;
pub mod rs_value;
pub mod rs_zonal_stats;
pub mod utils;

mod gdal_common;
mod gdal_dataset_provider;

// Re-export main dataset conversion functions
pub use gdal_common::{
    band_data_type_to_gdal, bytes_to_f64, gdal_to_band_data_type, gdal_type_byte_size,
    nodata_bytes_to_f64, nodata_f64_to_bytes,
};

// Expose provider/cache initializers for callers that need GDAL datasets from a `RasterRef`.
// Crate-internal callers construct providers from an explicit `Gdal` plus `thread_local_cache()`.

// Re-export utility functions
pub use utils::load_as_indb_raster;

// Re-export UDF constructors
pub use rs_as_geotiff::{rs_as_geotiff_udf, CompressionType};
pub use rs_as_raster::rs_as_raster_udf;
pub use rs_clip::rs_clip_udf;
pub use rs_from_gdal_raster::rs_from_gdal_raster_udf;
pub use rs_from_path::rs_from_path_udf;
pub use rs_map_algebra::rs_map_algebra_udf;
pub use rs_metadata::rs_metadata_udf;
pub use rs_polygonize::rs_polygonize_udf;
pub use rs_value::rs_value_udf;
pub use rs_zonal_stats::{rs_zonal_stats_all_udf, rs_zonal_stats_udf, StatType, ZonalStatistics};

// Re-export UDTF constructors
pub use rs_geotiff_tiles::rs_geotiff_tiles_udtf;

/// Returns all GDAL-based raster UDFs
pub fn all_gdal_udfs() -> Vec<datafusion_expr::ScalarUDF> {
    vec![
        rs_from_path_udf().into(),
        rs_from_gdal_raster_udf().into(),
        rs_as_geotiff_udf().into(),
        rs_as_raster_udf().into(),
        rs_value_udf().into(),
        rs_polygonize_udf().into(),
        rs_clip_udf().into(),
        rs_zonal_stats_udf().into(),
        rs_zonal_stats_all_udf().into(),
        rs_map_algebra_udf().into(),
        rs_metadata_udf().into(),
    ]
}
