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

use std::ptr;

use crate::cpl::CslStringList;
use crate::errors::Result;
use crate::gdal_api::{call_gdal_api, GdalApi};
use crate::gdal_dyn_bindgen::*;
use crate::raster::rasterband::RasterBand;
use crate::vector::layer::Layer;

#[derive(Clone, Debug, Default)]
pub struct PolygonizeOptions {
    /// Use 8 connectedness (diagonal pixels are considered connected).
    ///
    /// If `false` (default), 4 connectedness is used.
    pub eight_connected: bool,

    /// Name of a dataset from which to read the geotransform.
    ///
    /// This is useful if the source band has no related dataset, which is typical for mask bands.
    ///
    /// Corresponds to GDAL's `DATASET_FOR_GEOREF=dataset_name` option.
    pub dataset_for_georef: Option<String>,

    /// Interval in number of features at which transactions must be flushed.
    ///
    /// - `0` means that no transactions are opened.
    /// - a negative value means a single transaction.
    ///
    /// Corresponds to GDAL's `COMMIT_INTERVAL=num` option.
    pub commit_interval: Option<i64>,
}

impl PolygonizeOptions {
    /// Build a GDAL option list from these polygonize options.
    pub fn to_options_list(&self) -> Result<CslStringList> {
        let mut options = CslStringList::new();

        if self.eight_connected {
            options.set_name_value("8CONNECTED", "8")?;
        }

        if let Some(ref ds) = self.dataset_for_georef {
            options.set_name_value("DATASET_FOR_GEOREF", ds)?;
        }

        if let Some(interval) = self.commit_interval {
            options.set_name_value("COMMIT_INTERVAL", &interval.to_string())?;
        }

        Ok(options)
    }
}

/// Polygonize a raster band into a vector layer.
/// This uses `GDALPolygonize`, which reads source pixels as integers.
pub fn polygonize(
    api: &'static GdalApi,
    src_band: &RasterBand<'_>,
    mask_band: Option<&RasterBand<'_>>,
    out_layer: &Layer<'_>,
    pixel_value_field: i32,
    options: &PolygonizeOptions,
) -> Result<()> {
    let mask = mask_band.map_or(ptr::null_mut(), |b| b.c_rasterband());
    let csl = options.to_options_list()?;

    let rv = unsafe {
        call_gdal_api!(
            api,
            GDALPolygonize,
            src_band.c_rasterband(),
            mask,
            out_layer.c_layer(),
            pixel_value_field,
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

/// Polygonize a raster band into a vector layer.
/// This uses `GDALFPolygonize`, which reads source pixels as floats.
pub fn fpolygonize(
    api: &'static GdalApi,
    src_band: &RasterBand<'_>,
    mask_band: Option<&RasterBand<'_>>,
    out_layer: &Layer<'_>,
    pixel_value_field: i32,
    options: &PolygonizeOptions,
) -> Result<()> {
    let mask = mask_band.map_or(ptr::null_mut(), |b| b.c_rasterband());
    let csl = options.to_options_list()?;

    let rv = unsafe {
        call_gdal_api!(
            api,
            GDALFPolygonize,
            src_band.c_rasterband(),
            mask,
            out_layer.c_layer(),
            pixel_value_field,
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
    use gdal_sys::GDALDatasetGetLayer;

    #[test]
    fn test_polygonizeoptions_as_ptr() {
        let c_options = PolygonizeOptions::default().to_options_list().unwrap();
        assert_eq!(c_options.fetch_name_value("8CONNECTED"), None);
        assert_eq!(c_options.fetch_name_value("DATASET_FOR_GEOREF"), None);
        assert_eq!(c_options.fetch_name_value("COMMIT_INTERVAL"), None);

        let c_options = PolygonizeOptions {
            eight_connected: true,
            dataset_for_georef: Some("/vsimem/georef.tif".to_string()),
            commit_interval: Some(12345),
        }
        .to_options_list()
        .unwrap();
        assert_eq!(c_options.fetch_name_value("8CONNECTED"), Some("8".into()));
        assert_eq!(
            c_options.fetch_name_value("DATASET_FOR_GEOREF"),
            Some("/vsimem/georef.tif".into())
        );
        assert_eq!(
            c_options.fetch_name_value("COMMIT_INTERVAL"),
            Some("12345".into())
        );
    }

    #[cfg(feature = "gdal-sys")]
    #[test]
    fn test_polygonize_connectivity_affects_regions() {
        use crate::dataset::LayerOptions;
        use crate::driver::DriverManager;
        use crate::global::with_global_gdal_api;
        use crate::raster::types::Buffer;
        use crate::vector::feature::FieldDefn;
        use crate::vsi::with_memfile;

        with_global_gdal_api(|api| {
            let mem_driver = DriverManager::get_driver_by_name(api, "MEM").unwrap();
            let flatgeobuf_driver = DriverManager::get_driver_by_name(api, "FlatGeobuf").unwrap();
            let raster_ds = mem_driver.create("", 3, 3, 1).unwrap();
            let band = raster_ds.rasterband(1).unwrap();

            // 3x3 raster with diagonal 1s:
            // 1 0 0
            // 0 1 0
            // 0 0 1
            let mut data = Buffer::new((3, 3), vec![1u8, 0, 0, 0, 1, 0, 0, 0, 1]);
            band.write((0, 0), (3, 3), &mut data).unwrap();

            with_memfile(
                api,
                "/vsimem/test_polygonize_connectivity_four.fgb",
                |layer_4_path| {
                    {
                        let vector_ds_4 =
                            flatgeobuf_driver.create_vector_only(layer_4_path).unwrap();

                        // 4-connected output
                        let layer_4 = vector_ds_4
                            .create_layer(LayerOptions {
                                name: "four",
                                srs: None,
                                ty: OGRwkbGeometryType::wkbPolygon,
                                options: None,
                            })
                            .unwrap();
                        let field_defn =
                            FieldDefn::new(api, "val", OGRFieldType::OFTInteger).unwrap();
                        layer_4.create_field(&field_defn).unwrap();

                        polygonize(api, &band, None, &layer_4, 0, &PolygonizeOptions::default())
                            .unwrap();
                    }

                    let reopened_4 = crate::dataset::Dataset::open_ex(
                        api,
                        layer_4_path,
                        crate::gdal_dyn_bindgen::GDAL_OF_VECTOR
                            | crate::gdal_dyn_bindgen::GDAL_OF_READONLY,
                        None,
                        None,
                        None,
                    )
                    .unwrap();
                    let c_layer_4 = unsafe { GDALDatasetGetLayer(reopened_4.c_dataset(), 0) };
                    assert!(!c_layer_4.is_null());
                    let mut read_layer_4 =
                        crate::vector::layer::Layer::new(api, c_layer_4, &reopened_4);
                    let ones_4 = read_layer_4
                        .features()
                        .filter_map(|f| f.field_as_integer(0))
                        .filter(|v| *v == 1)
                        .count();
                    assert_eq!(ones_4, 3);
                },
            );

            // 8-connected output
            with_memfile(
                api,
                "/vsimem/test_polygonize_connectivity_eight.fgb",
                |layer_8_path| {
                    {
                        let vector_ds_8 =
                            flatgeobuf_driver.create_vector_only(layer_8_path).unwrap();
                        let layer_8 = vector_ds_8
                            .create_layer(LayerOptions {
                                name: "eight",
                                srs: None,
                                ty: OGRwkbGeometryType::wkbPolygon,
                                options: None,
                            })
                            .unwrap();
                        let field_defn =
                            FieldDefn::new(api, "val", OGRFieldType::OFTInteger).unwrap();
                        layer_8.create_field(&field_defn).unwrap();

                        polygonize(
                            api,
                            &band,
                            None,
                            &layer_8,
                            0,
                            &PolygonizeOptions {
                                eight_connected: true,
                                dataset_for_georef: None,
                                commit_interval: None,
                            },
                        )
                        .unwrap();
                    }

                    let reopened_8 = crate::dataset::Dataset::open_ex(
                        api,
                        layer_8_path,
                        crate::gdal_dyn_bindgen::GDAL_OF_VECTOR
                            | crate::gdal_dyn_bindgen::GDAL_OF_READONLY,
                        None,
                        None,
                        None,
                    )
                    .unwrap();
                    let c_layer_8 = unsafe { GDALDatasetGetLayer(reopened_8.c_dataset(), 0) };
                    assert!(!c_layer_8.is_null());
                    let mut read_layer_8 =
                        crate::vector::layer::Layer::new(api, c_layer_8, &reopened_8);
                    let ones_8 = read_layer_8
                        .features()
                        .filter_map(|f| f.field_as_integer(0))
                        .filter(|v| *v == 1)
                        .count();
                    assert_eq!(ones_8, 1);
                },
            );
        })
        .unwrap();
    }

    #[cfg(feature = "gdal-sys")]
    #[test]
    fn test_polygonize_with_mask_band_restricts_output() {
        use crate::dataset::LayerOptions;
        use crate::driver::DriverManager;
        use crate::global::with_global_gdal_api;
        use crate::raster::types::Buffer;
        use crate::vector::feature::FieldDefn;
        use crate::vsi::with_memfile;

        with_global_gdal_api(|api| {
            let mem_driver = DriverManager::get_driver_by_name(api, "MEM").unwrap();
            let raster_ds = mem_driver.create("", 3, 3, 2).unwrap();

            let value_band = raster_ds.rasterband(1).unwrap();
            let mask_band = raster_ds.rasterband(2).unwrap();

            // Value band: all 7s.
            let mut values = Buffer::new((3, 3), vec![7u8; 9]);
            value_band.write((0, 0), (3, 3), &mut values).unwrap();

            // Mask: only the center pixel is included.
            let mut mask = Buffer::new((3, 3), vec![0u8, 0, 0, 0, 1, 0, 0, 0, 0]);
            mask_band.write((0, 0), (3, 3), &mut mask).unwrap();

            with_memfile(api, "/vsimem/test_polygonize_mask.fgb", |fgb_path| {
                let flatgeobuf_driver =
                    DriverManager::get_driver_by_name(api, "FlatGeobuf").unwrap();
                {
                    let vector_ds = flatgeobuf_driver.create_vector_only(fgb_path).unwrap();

                    let layer = vector_ds
                        .create_layer(LayerOptions {
                            name: "masked",
                            srs: None,
                            ty: OGRwkbGeometryType::wkbPolygon,
                            options: None,
                        })
                        .unwrap();
                    let field_defn = FieldDefn::new(api, "val", OGRFieldType::OFTInteger).unwrap();
                    layer.create_field(&field_defn).unwrap();

                    polygonize(
                        api,
                        &value_band,
                        Some(&mask_band),
                        &layer,
                        0,
                        &PolygonizeOptions::default(),
                    )
                    .unwrap();
                }

                let reopened = crate::dataset::Dataset::open_ex(
                    api,
                    fgb_path,
                    crate::gdal_dyn_bindgen::GDAL_OF_VECTOR
                        | crate::gdal_dyn_bindgen::GDAL_OF_READONLY,
                    None,
                    None,
                    None,
                )
                .unwrap();
                let c_layer = unsafe { GDALDatasetGetLayer(reopened.c_dataset(), 0) };
                assert!(!c_layer.is_null());
                let mut read_layer = crate::vector::layer::Layer::new(api, c_layer, &reopened);
                assert_eq!(read_layer.feature_count(true), 1);
                let only_val = read_layer.features().next().unwrap().field_as_integer(0);
                assert_eq!(only_val, Some(7));
            });
        })
        .unwrap();
    }
}
