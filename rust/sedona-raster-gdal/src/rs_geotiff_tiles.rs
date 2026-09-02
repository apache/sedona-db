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

//! GeoTIFF tiles format spec and `rs_geotiff_tiles` UDTF.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{
    builder::StringBuilder, builder::UInt32Builder, ArrayRef, RecordBatch, RecordBatchIterator,
    RecordBatchReader,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::catalog::{TableFunctionImpl, TableProvider};
use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion_common::{exec_datafusion_err, plan_err, Result, ScalarValue, Statistics};
use datafusion_expr::Expr;
use sedona_datasource::format::ExternalFileFormat;
use sedona_datasource::spec::{ExternalFormatSpec, Object, OpenReaderArgs, SupportsRepartition};
use sedona_gdal::spatial_ref::SpatialRef;
use sedona_raster::builder::{RasterBuilder, StartBandArgs};
use sedona_raster::view_entries::ViewEntry;

use crate::gdal_common::{
    gdal_to_band_data_type, nodata_f64_to_bytes, normalize_outdb_source_path, open_gdal_dataset,
    with_gdal,
};
use crate::utils::Grid;

/// Returns the fixed 4-column schema produced for GeoTIFF tile readers:
/// - `path`: Utf8
/// - `x`: UInt32
/// - `y`: UInt32
/// - `rast`: `sedona.raster` extension type
pub fn geotiff_tile_schema() -> SchemaRef {
    let rast_field = sedona_schema::datatypes::RASTER
        .to_storage_field("rast", false)
        .expect("raster storage field");
    Arc::new(Schema::new(vec![
        Field::new("path", DataType::Utf8, false),
        Field::new("x", DataType::UInt32, false),
        Field::new("y", DataType::UInt32, false),
        rast_field,
    ]))
}

/// [`ExternalFormatSpec`] implementation for reading GeoTIFF files as tiled rasters.
#[derive(Debug, Clone)]
pub struct GeoTiffTilesSpec {
    extension: String,
}

impl GeoTiffTilesSpec {
    pub fn new(extension: impl Into<String>) -> Self {
        Self {
            extension: extension.into(),
        }
    }
}

impl Default for GeoTiffTilesSpec {
    fn default() -> Self {
        Self::new("tif")
    }
}

#[async_trait]
impl ExternalFormatSpec for GeoTiffTilesSpec {
    async fn infer_schema(&self, _location: &Object) -> Result<Schema> {
        Ok((*geotiff_tile_schema()).clone())
    }

    async fn open_reader(
        &self,
        args: &OpenReaderArgs,
    ) -> Result<Box<dyn RecordBatchReader + Send>> {
        let path = args
            .src
            .to_url_string()
            .or_else(|| {
                args.src
                    .meta
                    .as_ref()
                    .map(|m| m.location.as_ref().to_string())
            })
            .ok_or_else(|| {
                exec_datafusion_err!(
                    "Cannot determine location for GeoTIFF source object: {:?}",
                    args.src
                )
            })?;

        let schema = geotiff_tile_schema();
        let batch = build_batch_for_file(&path, schema.clone())?;
        let batch = match batch {
            Some(mut b) => {
                if let Some(projection) = &args.file_projection {
                    b = b.project(projection)?;
                }
                b
            }
            None => {
                let s = match &args.file_projection {
                    Some(proj) => Arc::new(schema.project(proj)?),
                    None => schema,
                };
                RecordBatch::new_empty(s)
            }
        };

        let batch_schema = batch.schema();
        let batch_size = args.batch_size.unwrap_or(usize::MAX);
        let batches: Vec<RecordBatch> = if batch.num_rows() <= batch_size || batch_size == 0 {
            vec![batch]
        } else {
            let mut offset = 0;
            let mut result = Vec::new();
            while offset < batch.num_rows() {
                let len = (batch.num_rows() - offset).min(batch_size);
                result.push(batch.slice(offset, len));
                offset += len;
            }
            result
        };

        Ok(Box::new(RecordBatchIterator::new(
            batches.into_iter().map(Ok),
            batch_schema,
        )))
    }

    fn with_options(
        &self,
        options: &HashMap<String, String>,
    ) -> Result<Arc<dyn ExternalFormatSpec>> {
        if let Some(k) = options.keys().next() {
            return plan_err!("Unsupported option for GeoTiffTilesSpec: '{k}'");
        }
        Ok(Arc::new(self.clone()))
    }

    fn extension(&self) -> &str {
        &self.extension
    }

    fn list_single_object(&self) -> bool {
        false
    }

    fn supports_repartition(&self) -> SupportsRepartition {
        SupportsRepartition::None
    }

    async fn infer_stats(&self, _location: &Object, table_schema: &Schema) -> Result<Statistics> {
        Ok(Statistics::new_unknown(table_schema))
    }
}

/// Constructs a `RecordBatch` containing tiles for the GeoTIFF at `path`.
pub fn build_batch_for_file(
    path: impl AsRef<Path>,
    schema: SchemaRef,
) -> Result<Option<RecordBatch>> {
    let path_str = path.as_ref().to_string_lossy().to_string();
    let normalized_path = normalize_outdb_source_path(&path_str);
    with_gdal(|gdal| {
        let ds = open_gdal_dataset(gdal, &normalized_path, None)
            .map_err(|e| exec_datafusion_err!("Failed to open GeoTIFF {path_str}: {e}"))?;
        let (width, height) = ds.raster_size();

        let band_count = ds.raster_count();
        if band_count == 0 {
            return Ok(None);
        }

        let band1 = ds
            .rasterband(1)
            .map_err(|e| exec_datafusion_err!("Failed to get band 1 for {path_str}: {e}"))?;
        let (block_x, block_y) = band1.block_size();
        let block_x = block_x.max(1) as u32;
        let block_y = block_y.max(1) as u32;

        let tiles_x = div_ceil_u32(width as u32, block_x);
        let tiles_y = div_ceil_u32(height as u32, block_y);

        let geotransform = ds
            .geo_transform()
            .map_err(|e| exec_datafusion_err!("Failed to get geotransform for {path_str}: {e}"))?;

        let crs = ds
            .spatial_ref()
            .ok()
            .and_then(|sr: SpatialRef| sr.to_projjson().ok());

        let total_tiles = (tiles_x * tiles_y) as usize;
        let mut path_builder =
            StringBuilder::with_capacity(total_tiles, total_tiles * normalized_path.len());
        let mut x_builder = UInt32Builder::with_capacity(total_tiles);
        let mut y_builder = UInt32Builder::with_capacity(total_tiles);
        let mut rast_builder = RasterBuilder::new(total_tiles);

        for tile_y in 0..tiles_y {
            for tile_x in 0..tiles_x {
                let px = tile_x * block_x;
                let py = tile_y * block_y;

                let tw = (width as u32).saturating_sub(px).min(block_x);
                let th = (height as u32).saturating_sub(py).min(block_y);
                if tw == 0 || th == 0 {
                    continue;
                }

                let tile_transform = [
                    geotransform[0] + (px as f64) * geotransform[1] + (py as f64) * geotransform[2],
                    geotransform[1],
                    geotransform[2],
                    geotransform[3] + (px as f64) * geotransform[4] + (py as f64) * geotransform[5],
                    geotransform[4],
                    geotransform[5],
                ];

                path_builder.append_value(&normalized_path);
                x_builder.append_value(tile_x);
                y_builder.append_value(tile_y);

                Grid::from_gdal(tile_transform, tw as usize, th as usize)
                    .start_raster_into(&mut rast_builder, crs.as_deref())
                    .map_err(|e| {
                        exec_datafusion_err!(
                            "Failed to start raster for {path_str} tile ({tile_x},{tile_y}): {e}"
                        )
                    })?;

                for band_idx in 1..=band_count {
                    let band = ds.rasterband(band_idx).map_err(|e| {
                        exec_datafusion_err!("Failed to get band {band_idx} for {path_str}: {e}")
                    })?;

                    let gdal_type = band.band_type();
                    let band_data_type = gdal_to_band_data_type(gdal_type).map_err(|_| {
                        exec_datafusion_err!(
                            "Unsupported band data type {gdal_type:?} for {path_str} band {band_idx}"
                        )
                    })?;

                    let nodata_bytes = band
                        .no_data_value()
                        .map(|v| nodata_f64_to_bytes(v, &band_data_type));

                    let outdb_uri = format!("{normalized_path}#band={band_idx}");
                    let view = [
                        ViewEntry {
                            source_axis: 0,
                            start: py as i64,
                            step: 1,
                            steps: th as i64,
                        },
                        ViewEntry {
                            source_axis: 1,
                            start: px as i64,
                            step: 1,
                            steps: tw as i64,
                        },
                    ];

                    rast_builder
                        .start_band(StartBandArgs {
                            nodata: nodata_bytes.as_deref(),
                            outdb_uri: Some(&outdb_uri),
                            view: Some(&view),
                            ..StartBandArgs::new(
                                &["y", "x"],
                                &[height as i64, width as i64],
                                band_data_type,
                            )
                        })
                        .map_err(|e| {
                            exec_datafusion_err!(
                                "Failed to start band {band_idx} for {path_str}: {e}"
                            )
                        })?;

                    rast_builder.band_data_writer().append_value([]);

                    rast_builder.finish_band().map_err(|e| {
                        exec_datafusion_err!("Failed to finish band {band_idx} for {path_str}: {e}")
                    })?;
                }

                rast_builder.finish_raster().map_err(|e| {
                    exec_datafusion_err!(
                        "Failed to finish raster for {path_str} tile ({tile_x},{tile_y}): {e}"
                    )
                })?;
            }
        }

        let rast_array: ArrayRef = Arc::new(
            rast_builder
                .finish()
                .map_err(|e| exec_datafusion_err!("Failed to build rasters: {e}"))?,
        );
        let path_array: ArrayRef = Arc::new(path_builder.finish());
        let x_array: ArrayRef = Arc::new(x_builder.finish());
        let y_array: ArrayRef = Arc::new(y_builder.finish());

        let batch = RecordBatch::try_new(schema, vec![path_array, x_array, y_array, rast_array])
            .map_err(|e| exec_datafusion_err!("Failed to create RecordBatch: {e}"))?;

        Ok(Some(batch))
    })
}

/// Returns a [`TableFunctionImpl`] for `rs_geotiff_tiles(path[, recursive])`.
pub fn rs_geotiff_tiles_udtf() -> Arc<dyn TableFunctionImpl> {
    Arc::new(RsGeoTiffTilesFunction {})
}

#[derive(Debug)]
pub struct RsGeoTiffTilesFunction {}

impl TableFunctionImpl for RsGeoTiffTilesFunction {
    fn call(&self, exprs: &[Expr]) -> Result<Arc<dyn TableProvider>> {
        if exprs.is_empty() || exprs.len() > 2 {
            return plan_err!(
                "rs_geotiff_tiles() expected 1 or 2 arguments (path[, recursive]) but got {}",
                exprs.len()
            );
        }

        let path = match &exprs[0] {
            Expr::Literal(ScalarValue::Utf8(Some(s)), _) => s.clone(),
            Expr::Literal(ScalarValue::Utf8View(Some(s)), _) => s.to_string(),
            Expr::Literal(ScalarValue::LargeUtf8(Some(s)), _) => s.clone(),
            other => {
                return plan_err!("rs_geotiff_tiles() expected literal string path but got {other}")
            }
        };

        let recursive = if exprs.len() == 2 {
            match &exprs[1] {
                Expr::Literal(ScalarValue::Boolean(Some(v)), _) => *v,
                other => {
                    return plan_err!(
                        "rs_geotiff_tiles() expected literal boolean recursive but got {other}"
                    )
                }
            }
        } else {
            false
        };

        let table_url = if recursive {
            let path_buf = Path::new(&path);
            if path_buf.is_dir() {
                let trimmed = path.trim_end_matches(['/', '\\']);
                ListingTableUrl::parse(format!("{trimmed}/**"))?
            } else {
                ListingTableUrl::parse(&path)?
            }
        } else {
            let path_buf = Path::new(&path);
            if path_buf.is_dir() {
                let trimmed = path.trim_end_matches(['/', '\\']);
                ListingTableUrl::parse(format!("{trimmed}/*"))?
            } else {
                ListingTableUrl::parse(&path)?
            }
        };

        let spec = Arc::new(GeoTiffTilesSpec::new(""));
        let format = Arc::new(ExternalFileFormat::new(spec));
        let listing_options = ListingOptions::new(format).with_file_extension("");
        let schema = geotiff_tile_schema();
        let config = ListingTableConfig::new(table_url)
            .with_listing_options(listing_options)
            .with_schema(schema);

        let provider = ListingTable::try_new(config)?;
        Ok(Arc::new(provider))
    }
}

fn div_ceil_u32(n: u32, d: u32) -> u32 {
    if d == 0 {
        return 0;
    }
    n.div_ceil(d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::prelude::SessionContext;
    use sedona_gdal::raster::types::Buffer;
    use std::path::PathBuf;
    use tempfile::tempdir;

    fn write_test_geotiff(base: &Path, name: &str) -> PathBuf {
        let path = base.join(name);
        let path_str = path.to_string_lossy().to_string();
        with_gdal(|gdal| {
            let driver = gdal.get_driver_by_name("GTiff").unwrap();
            let dataset = driver
                .create_with_band_type::<u8>(&path_str, 10, 10, 1)
                .unwrap();
            dataset
                .set_geo_transform(&[0.0, 1.0, 0.0, 10.0, 0.0, -1.0])
                .unwrap();
            let band = dataset.rasterband(1).unwrap();
            let mut buffer = Buffer::new((10, 10), (0..100u8).collect::<Vec<_>>());
            band.write((0, 0), (10, 10), &mut buffer).unwrap();
            Ok::<_, datafusion_common::DataFusionError>(())
        })
        .unwrap();
        path
    }

    #[tokio::test]
    async fn udtf_registration_smoke() {
        let ctx = SessionContext::new();
        ctx.register_udtf("rs_geotiff_tiles", rs_geotiff_tiles_udtf());
    }

    #[test]
    fn rast_field_has_raster_metadata() {
        let schema = geotiff_tile_schema();
        let rast_field = schema.field_with_name("rast").unwrap();
        let sedona_type = sedona_schema::datatypes::SedonaType::from_storage_field(rast_field)
            .expect("sedona type");
        assert_eq!(sedona_type, sedona_schema::datatypes::RASTER);
        assert_eq!(
            rast_field
                .metadata()
                .get("ARROW:extension:name")
                .map(|s| s.as_str()),
            Some("sedona.raster")
        );
    }

    #[tokio::test]
    async fn spec_reads_geotiff_tiles_via_sql() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        let file_path = write_test_geotiff(base, "test.tif");

        let ctx = SessionContext::new();
        ctx.register_udtf("rs_geotiff_tiles", rs_geotiff_tiles_udtf());

        let df = ctx
            .sql(&format!(
                "SELECT path, x, y FROM rs_geotiff_tiles('{}')",
                file_path.to_string_lossy()
            ))
            .await
            .unwrap();

        let batches = df.collect().await.unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_columns(), 3);
        assert!(batches[0].num_rows() >= 1);
    }

    #[tokio::test]
    async fn spec_reads_geotiff_directory_via_sql() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        write_test_geotiff(base, "test1.tif");
        write_test_geotiff(base, "test2.tif");

        let ctx = SessionContext::new();
        ctx.register_udtf("rs_geotiff_tiles", rs_geotiff_tiles_udtf());

        let df = ctx
            .sql(&format!(
                "SELECT path, x, y FROM rs_geotiff_tiles('{}', false)",
                base.to_string_lossy()
            ))
            .await
            .unwrap();

        let batches = df.collect().await.unwrap();
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 2);
    }
}
