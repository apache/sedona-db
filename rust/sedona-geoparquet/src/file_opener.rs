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
use std::{collections::HashMap, sync::Arc};

use arrow_array::{Array, RecordBatch};
use arrow_schema::{DataType, SchemaRef};
use datafusion::datasource::{
    listing::PartitionedFile,
    physical_plan::{parquet::ParquetAccessPlan, FileOpenFuture, FileOpener},
};
use datafusion_common::{
    cast::{as_binary_array, as_binary_view_array, as_large_binary_array},
    exec_err, Result,
};
use datafusion_datasource_parquet::metadata::DFParquetMetadata;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_plan::metrics::{
    ExecutionPlanMetricsSet, MetricBuilder, MetricType, MetricValue, PruningMetrics,
};
use futures::StreamExt;
use object_store::ObjectStore;
use parquet::{
    basic::LogicalType,
    file::{
        metadata::{ParquetMetaData, RowGroupMetaData},
        statistics::Statistics,
    },
    geospatial::statistics::GeospatialStatistics,
};
use sedona_expr::{
    spatial_filter::{SpatialFilter, TableGeoStatistics},
    statistics::GeoStatistics,
};
use sedona_geometry::{
    bounding_box::BoundingBox,
    interval::{Interval, IntervalTrait},
    types::{GeometryTypeAndDimensions, GeometryTypeAndDimensionsSet},
};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

use crate::{
    metadata::{GeoParquetColumnEncoding, GeoParquetMetadata},
    options::TableGeoParquetOptions,
};

#[derive(Clone)]
pub(crate) struct GeoParquetFileOpenerMetrics {
    /// How many file ranges are pruned or matched by [`SpatialFilter`]
    ///
    /// Note on "file range": an opener may read only part of a file rather than the
    /// entire file; that portion is referred to as the "file range". See [`PartitionedFile`]
    /// for details.
    files_ranges_spatial_pruned: PruningMetrics,
    /// How many row groups are pruned or matched by [`SpatialFilter`]
    ///
    /// Note: row groups skipped during the file-level pruning step are not counted
    /// again here.
    row_groups_spatial_pruned: PruningMetrics,
}

impl GeoParquetFileOpenerMetrics {
    pub fn new(execution_plan_global_metrics: &ExecutionPlanMetricsSet) -> Self {
        let files_ranges_spatial_pruned = PruningMetrics::new();
        MetricBuilder::new(execution_plan_global_metrics)
            .with_type(MetricType::SUMMARY)
            .build(MetricValue::PruningMetrics {
                name: "files_ranges_spatial_pruned".into(),
                pruning_metrics: files_ranges_spatial_pruned.clone(),
            });

        let row_groups_spatial_pruned = PruningMetrics::new();
        MetricBuilder::new(execution_plan_global_metrics)
            .with_type(MetricType::SUMMARY)
            .build(MetricValue::PruningMetrics {
                name: "row_groups_spatial_pruned".into(),
                pruning_metrics: row_groups_spatial_pruned.clone(),
            });

        Self {
            files_ranges_spatial_pruned,
            row_groups_spatial_pruned,
        }
    }
}

/// Geo-aware [FileOpener] implementing file and row group pruning
///
/// Pruning happens (for Parquet) in the [FileOpener], so we implement
/// that here, too.
#[derive(Clone)]
pub(crate) struct GeoParquetFileOpener {
    pub inner: Arc<dyn FileOpener>,
    pub object_store: Arc<dyn ObjectStore>,
    pub metadata_size_hint: Option<usize>,
    pub predicate: Option<Arc<dyn PhysicalExpr>>,
    pub file_schema: SchemaRef,
    pub enable_pruning: bool,
    pub metrics: GeoParquetFileOpenerMetrics,
    pub options: TableGeoParquetOptions,
}

impl FileOpener for GeoParquetFileOpener {
    fn open(&self, file: PartitionedFile) -> Result<FileOpenFuture> {
        let self_clone = self.clone();

        Ok(Box::pin(async move {
            let parquet_metadata =
                DFParquetMetadata::new(&self_clone.object_store, &file.object_meta)
                    .with_metadata_size_hint(self_clone.metadata_size_hint)
                    .fetch_metadata()
                    .await?;

            let mut access_plan = ParquetAccessPlan::new_all(parquet_metadata.num_row_groups());

            let maybe_geoparquet_metadata = GeoParquetMetadata::try_from_parquet_metadata(
                &parquet_metadata,
                self_clone.options.geometry_columns.as_ref(),
            )?;

            if self_clone.enable_pruning {
                if let Some(predicate) = self_clone.predicate.as_ref() {
                    let spatial_filter = SpatialFilter::try_from_expr(predicate)?;

                    if let Some(geoparquet_metadata) = maybe_geoparquet_metadata.as_ref() {
                        filter_access_plan_using_geoparquet_file_metadata(
                            &self_clone.file_schema,
                            &mut access_plan,
                            &spatial_filter,
                            geoparquet_metadata,
                            &self_clone.metrics,
                        )?;

                        filter_access_plan_using_geoparquet_covering(
                            &self_clone.file_schema,
                            &mut access_plan,
                            &spatial_filter,
                            geoparquet_metadata,
                            &parquet_metadata,
                            &self_clone.metrics,
                        )?;

                        filter_access_plan_using_native_geostats(
                            &self_clone.file_schema,
                            &mut access_plan,
                            &spatial_filter,
                            &parquet_metadata,
                            &self_clone.metrics,
                        )?;
                    }
                }
            }

            // When we have built-in GEOMETRY/GEOGRAPHY types, we can filter the access plan
            // from the native GeoStatistics here.

            // We could also consider filtering using null_count here in the future (i.e.,
            // skip row groups that are all null)
            let file = file.with_extensions(Arc::new(access_plan));
            let stream = self_clone.inner.open(file)?.await?;

            // Validate geometry columns when enabled from read option.
            let validation_columns = if self_clone.options.validate {
                maybe_geoparquet_metadata
                    .as_ref()
                    .map(|metadata| wkb_validation_columns(&self_clone.file_schema, metadata))
                    .unwrap_or_default()
            } else {
                Vec::new()
            };

            if !self_clone.options.validate || validation_columns.is_empty() {
                return Ok(stream);
            }

            let validated_stream = stream.map(move |batch_result| {
                let batch = batch_result?;
                validate_wkb_batch(&batch, &validation_columns)?;
                Ok(batch)
            });

            Ok(Box::pin(validated_stream))
        }))
    }
}

fn wkb_validation_columns(
    file_schema: &SchemaRef,
    metadata: &GeoParquetMetadata,
) -> Vec<(usize, String)> {
    file_schema
        .fields()
        .iter()
        .enumerate()
        .filter_map(|(column_index, field)| {
            metadata
                .columns
                .get(field.name())
                .and_then(|column_metadata| {
                    if matches!(column_metadata.encoding, GeoParquetColumnEncoding::WKB) {
                        Some((column_index, field.name().clone()))
                    } else {
                        None
                    }
                })
        })
        .collect()
}

fn validate_wkb_batch(batch: &RecordBatch, validation_columns: &[(usize, String)]) -> Result<()> {
    for (column_index, column_name) in validation_columns {
        let column = batch.column(*column_index);
        validate_wkb_array(column.as_ref(), column_name)?;
    }
    Ok(())
}

fn validate_wkb_array(array: &dyn Array, column_name: &str) -> Result<()> {
    match array.data_type() {
        DataType::Binary => {
            let array = as_binary_array(array)?;
            validate_wkb_values(array.iter(), column_name)?;
        }
        DataType::LargeBinary => {
            let array = as_large_binary_array(array)?;
            validate_wkb_values(array.iter(), column_name)?;
        }
        DataType::BinaryView => {
            let array = as_binary_view_array(array)?;
            validate_wkb_values(array.iter(), column_name)?;
        }
        other => {
            return exec_err!(
                "Expected Binary/LargeBinary/BinaryView storage for WKB validation in column '{}' but got {}",
                column_name,
                other
            );
        }
    }

    Ok(())
}

fn validate_wkb_values<'a>(
    values: impl IntoIterator<Item = Option<&'a [u8]>>,
    column_name: &str,
) -> Result<()> {
    for (row_index, maybe_wkb) in values.into_iter().enumerate() {
        if let Some(wkb_bytes) = maybe_wkb {
            if let Err(e) = wkb::reader::read_wkb(wkb_bytes) {
                return exec_err!(
                    "WKB validation failed for column '{}' at row {}: {}",
                    column_name,
                    row_index,
                    e
                );
            }
        }
    }

    Ok(())
}

/// Filter an access plan using the GeoParquet file metadata
///
/// Inspects the GeoParquetMetadata for a bbox at the column metadata level
/// and skips all row groups in the file if the spatial_filter evaluates to false.
fn filter_access_plan_using_geoparquet_file_metadata(
    file_schema: &SchemaRef,
    access_plan: &mut ParquetAccessPlan,
    spatial_filter: &SpatialFilter,
    metadata: &GeoParquetMetadata,
    metrics: &GeoParquetFileOpenerMetrics,
) -> Result<()> {
    let column_geo_stats = geoparquet_file_geo_stats(file_schema, metadata)?;
    let table_geo_stats =
        TableGeoStatistics::try_from_stats_and_schema(&column_geo_stats, file_schema)?;
    if !spatial_filter.evaluate(&table_geo_stats)? {
        metrics.files_ranges_spatial_pruned.add_pruned(1);
        for i in access_plan.row_group_indexes() {
            access_plan.skip(i);
        }
    } else {
        metrics.files_ranges_spatial_pruned.add_matched(1);
    }

    Ok(())
}

/// Filter an access plan using the GeoParquet bbox covering, if present
///
/// Iterates through an existing access plan and skips row groups based on
/// the statistics of bbox columns (if specified in the GeoParquet column metadata).
fn filter_access_plan_using_geoparquet_covering(
    file_schema: &SchemaRef,
    access_plan: &mut ParquetAccessPlan,
    spatial_filter: &SpatialFilter,
    metadata: &GeoParquetMetadata,
    parquet_metadata: &ParquetMetaData,
    metrics: &GeoParquetFileOpenerMetrics,
) -> Result<()> {
    let row_group_indices_to_scan = access_plan.row_group_indexes();

    // What we're about to do is a bit of work, so skip it if we can.
    if row_group_indices_to_scan.is_empty() {
        return Ok(());
    }

    // The GeoParquetMetadata refers to bbox covering columns as e.g. {"xmin": ["bbox", "xmin"]},
    // but we need flattened integer references to retrieve min/max statistics for each of these.
    let covering_specs = parse_column_coverings(file_schema, parquet_metadata, metadata)?;

    // If there are no covering specs, don't iterate through the row groups
    // This has the side-effect of ensuring the row_groups_spatial_pruned matched count is not
    // double counted except in the rare case where we prune based on both.
    if covering_specs.iter().all(|spec| spec.is_none()) {
        return Ok(());
    }

    // Iterate through the row groups
    for i in row_group_indices_to_scan {
        // Generate row group statistics based on the covering statistics
        let row_group_column_geo_stats =
            row_group_covering_geo_stats(parquet_metadata.row_group(i), &covering_specs);
        let row_group_geo_stats = TableGeoStatistics::try_from_stats_and_schema(
            &row_group_column_geo_stats,
            file_schema,
        )?;

        // Evaluate predicate!
        if !spatial_filter.evaluate(&row_group_geo_stats)? {
            metrics.row_groups_spatial_pruned.add_pruned(1);
            access_plan.skip(i);
        } else {
            metrics.row_groups_spatial_pruned.add_matched(1);
        }
    }

    Ok(())
}

/// Filter an access plan using the Parquet GeoStatistics, if present
///
/// Iterates through an existing access plan and skips row groups based on
/// the Parquet format GeoStatistics (i.e., Geometry/Geography Parquet types).
fn filter_access_plan_using_native_geostats(
    file_schema: &SchemaRef,
    access_plan: &mut ParquetAccessPlan,
    spatial_filter: &SpatialFilter,
    parquet_metadata: &ParquetMetaData,
    metrics: &GeoParquetFileOpenerMetrics,
) -> Result<()> {
    let row_group_indices_to_scan = access_plan.row_group_indexes();

    // What we're about to do is a bit of work, so skip it if we can.
    if row_group_indices_to_scan.is_empty() {
        return Ok(());
    }

    // Get the indices we need to index in to the Parquet column()s.
    // For schemas with no nested columns, this will be a sequential
    // range of 0..n.
    let top_level_indices = top_level_column_indices(parquet_metadata);

    // If there are no native geometry or geography logical types at the
    // top level indices, don't iterate through the row groups. This has the side-effect
    // of ensuring the row_groups_spatial_pruned matched count is not double counted except
    // in the rare case where we prune based on both.
    let parquet_schema = parquet_metadata.file_metadata().schema_descr();
    if top_level_indices.iter().all(|i| {
        !matches!(
            parquet_schema.column(*i).logical_type_ref(),
            Some(LogicalType::Geometry { .. }) | Some(LogicalType::Geography { .. })
        )
    }) {
        return Ok(());
    }

    // Iterate through the row groups
    for i in row_group_indices_to_scan {
        // Generate row group statistics based on the covering statistics
        let row_group_column_geo_stats =
            row_group_native_geo_stats(parquet_metadata.row_group(i), &top_level_indices);
        let row_group_geo_stats = TableGeoStatistics::try_from_stats_and_schema(
            &row_group_column_geo_stats,
            file_schema,
        )?;

        // Evaluate predicate!
        if !spatial_filter.evaluate(&row_group_geo_stats)? {
            metrics.row_groups_spatial_pruned.add_pruned(1);
            access_plan.skip(i);
        } else {
            metrics.row_groups_spatial_pruned.add_matched(1);
        }
    }

    Ok(())
}

/// Calculates a Vec of [GeoStatistics] based on GeoParquet file-level metadata
///
/// Each element is either a [GeoStatistics] populated with a [BoundingBox]
/// or [GeoStatistics::unspecified], which is a value that will ensure that
/// any spatial predicate that references those statistics will evaluate to
/// true.
fn geoparquet_file_geo_stats(
    file_schema: &SchemaRef,
    metadata: &GeoParquetMetadata,
) -> Result<Vec<GeoStatistics>> {
    file_schema
        .fields()
        .iter()
        .map(|field| -> Result<GeoStatistics> {
            // If this column is in the GeoParquet metadata, construct actual statistics
            // (otherwise, construct unspecified statistics)
            if let Some(column_metadata) = metadata.columns.get(field.name()) {
                Ok(column_metadata.to_geo_statistics())
            } else {
                Ok(GeoStatistics::unspecified())
            }
        })
        .collect()
}

/// Calculates a Vec of [GeoStatistics] based on GeoParquet covering columns
///
/// Each element is either a [GeoStatistics] populated with a [BoundingBox]
/// or [GeoStatistics::unspecified], which is a value that will ensure that
/// any spatial predicate that references those statistics will evaluate to
/// true.
fn row_group_covering_geo_stats(
    row_group_metadata: &RowGroupMetaData,
    covering_specs: &[Option<[usize; 4]>],
) -> Vec<GeoStatistics> {
    covering_specs
        .iter()
        .map(|covering_column_indices| {
            if let Some(indices) = covering_column_indices {
                let stats = indices
                    .map(|j| row_group_metadata.column(j).statistics())
                    .map(|maybe_stats| match maybe_stats {
                        Some(stats) => column_stats_min_max(stats),
                        None => None,
                    });
                match (stats[0], stats[1], stats[2], stats[3]) {
                    (Some(xmin), Some(ymin), Some(xmax), Some(ymax)) => {
                        GeoStatistics::unspecified()
                            .with_bbox(Some(BoundingBox::xy((xmin.0, xmax.1), (ymin.0, ymax.1))))
                    }
                    _ => GeoStatistics::unspecified(),
                }
            } else {
                GeoStatistics::unspecified()
            }
        })
        .collect()
}

/// Parse raw Parquet Statistics into a (min, max) tuple
///
/// If statistics are present, are exact, and are of the appropriate type,
/// return a tuple of the min, max values. Returns `None` otherwise.
fn column_stats_min_max(stats: &Statistics) -> Option<(f64, f64)> {
    match stats {
        Statistics::Float(value_statistics) => {
            if value_statistics.min_is_exact() && value_statistics.max_is_exact() {
                // *_is_exact() checks that values are present so we can unwrap them
                Some((
                    *value_statistics.min_opt().unwrap() as f64,
                    *value_statistics.max_opt().unwrap() as f64,
                ))
            } else {
                None
            }
        }
        Statistics::Double(value_statistics) => {
            if value_statistics.min_is_exact() && value_statistics.max_is_exact() {
                Some((
                    *value_statistics.min_opt().unwrap(),
                    *value_statistics.max_opt().unwrap(),
                ))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Calculates column indices for xmin, xmax, ymin, and ymax
///
/// We need to build a map from Vec<String> (i.e., ["bbox", "xmin"]) to column integers,
/// because that is how column statistics are retrieved from a row group.
/// Nested column statistics aren't handled by arrow-rs or datafusion:
/// (e.g. https://github.com/apache/arrow-rs/pull/7365), but we might need to do this
/// anyway.
fn parse_column_coverings(
    file_schema: &SchemaRef,
    parquet_metadata: &ParquetMetaData,
    metadata: &GeoParquetMetadata,
) -> Result<Vec<Option<[usize; 4]>>> {
    let mut column_index_map: HashMap<Vec<String>, usize> = HashMap::new();
    let schema_descr = parquet_metadata.file_metadata().schema_descr();
    for (i, col) in schema_descr.columns().iter().enumerate() {
        let path_vec = col.path().parts().to_vec();
        column_index_map.insert(path_vec, i);
    }

    file_schema
        .fields()
        .iter()
        .map(|field| -> Result<_> {
            if !metadata.columns.contains_key(field.name()) {
                return Ok(None);
            }

            if let Some(covering_spec) = metadata.bbox_covering(Some(field.name()))? {
                let indices = [
                    covering_spec.xmin,
                    covering_spec.ymin,
                    covering_spec.xmax,
                    covering_spec.ymax,
                ]
                .map(|path| column_index_map.get(&path));

                match (indices[0], indices[1], indices[2], indices[3]) {
                    (Some(xmin), Some(ymin), Some(xmax), Some(ymax)) => {
                        Ok(Some([*xmin, *ymin, *xmax, *ymax]))
                    }
                    _ => Ok(None),
                }
            } else {
                Ok(None)
            }
        })
        .collect()
}

/// Calculates a Vec of [GeoStatistics] based on Parquet-native GeoStatistics
///
/// Each element is either a [GeoStatistics] populated with a [BoundingBox]
/// or [GeoStatistics::unspecified], which is a value that will ensure that
/// any spatial predicate that references those statistics will evaluate to
/// true.
fn row_group_native_geo_stats(
    row_group_metadata: &RowGroupMetaData,
    column_indices: &[usize],
) -> Vec<GeoStatistics> {
    column_indices
        .iter()
        .map(|column_index| {
            let native_geo_stats_opt = row_group_metadata.column(*column_index).geo_statistics();
            native_geo_stats_opt
                .map(parquet_geo_stats_to_sedona_geo_stats)
                .unwrap_or(GeoStatistics::unspecified())
        })
        .collect()
}

/// Convert Parquet [GeospatialStatistics] into Sedona [GeoStatistics]
///
/// This also sanity checks the Parquet statistics for non-finite or non-sensical
/// ranges, treating the information as unknown if it fails the sanity check.
fn parquet_geo_stats_to_sedona_geo_stats(
    parquet_geo_stats: &GeospatialStatistics,
) -> GeoStatistics {
    let mut out = GeoStatistics::unspecified();

    if let Some(native_bbox) = parquet_geo_stats.bounding_box() {
        let x_range = (native_bbox.get_xmin(), native_bbox.get_xmax());
        let y_range = (native_bbox.get_ymin(), native_bbox.get_ymax());
        let z_range = match (native_bbox.get_zmin(), native_bbox.get_zmax()) {
            (Some(lo), Some(hi)) => Some(Interval::new(lo, hi)),
            _ => None,
        };
        let m_range = match (native_bbox.get_mmin(), native_bbox.get_mmax()) {
            (Some(lo), Some(hi)) => Some(Interval::new(lo, hi)),
            _ => None,
        };

        let bbox = BoundingBox::xyzm(x_range, y_range, z_range, m_range);

        // Sanity check the bbox statistics. If the sanity check fails, don't set
        // a bounding box for pruning. Note that the x width can be < 0 (wraparound).
        let mut bbox_is_valid =
            bbox.x().width().is_finite() && bbox.y().width().is_finite() && bbox.y().width() >= 0.0;
        if let Some(z) = bbox.z() {
            bbox_is_valid = bbox_is_valid && z.width().is_finite() && z.width() >= 0.0;
        }
        if let Some(m) = bbox.m() {
            bbox_is_valid = bbox_is_valid && m.width().is_finite() && m.width() >= 0.0;
        }

        if bbox_is_valid {
            out = out.with_bbox(Some(bbox));
        }
    }

    if let Some(native_geometry_types) = parquet_geo_stats.geospatial_types() {
        let mut geometry_types = GeometryTypeAndDimensionsSet::new();
        let mut geometry_types_valid = true;
        for wkb_id in native_geometry_types {
            if *wkb_id < 0 {
                geometry_types_valid = false;
                break;
            }

            match GeometryTypeAndDimensions::try_from_wkb_id(*wkb_id as u32) {
                Ok(type_and_dim) => geometry_types.insert_or_ignore(&type_and_dim),
                Err(_) => {
                    geometry_types_valid = false;
                    break;
                }
            }
        }

        if !geometry_types.is_empty() && geometry_types_valid {
            out = out.with_geometry_types(Some(geometry_types))
        }
    }

    out
}

/// Calculates column indices for top-level columns of file_schema
///
/// We need to build a list of top-level indices, where the indices refer to the
/// flattened list of columns (e.g., `.column(i)` in row group metadata).
fn top_level_column_indices(parquet_metadata: &ParquetMetaData) -> Vec<usize> {
    let mut top_level_indices = Vec::new();
    let schema_descr = parquet_metadata.file_metadata().schema_descr();
    for (i, col) in schema_descr.columns().iter().enumerate() {
        let path_vec = col.path().parts();
        if path_vec.len() == 1 {
            top_level_indices.push(i);
        }
    }

    top_level_indices
}

/// Returns true if there are any fields with GeoArrow metadata
///
/// This is used to defer to the parent implementation if there is no
/// geometry present in the data source.
pub fn storage_schema_contains_geo(schema: &SchemaRef) -> bool {
    let matcher = ArgMatcher::is_geometry_or_geography();
    for field in schema.fields() {
        match SedonaType::from_storage_field(field) {
            Ok(sedona_type) => {
                if matcher.match_type(&sedona_type) {
                    return true;
                }
            }
            Err(_) => return false,
        }
    }

    false
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, BinaryArray, BinaryViewArray, Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::{
        arrow::ArrowSchemaConverter,
        file::{
            metadata::{
                ColumnChunkMetaData, FileMetaData, ParquetMetaDataBuilder, RowGroupMetaData,
            },
            statistics::ValueStatistics,
        },
    };
    use sedona_schema::datatypes::{WKB_GEOGRAPHY, WKB_GEOMETRY};

    use super::*;

    #[test]
    fn file_stats() {
        let file_schema = Schema::new(vec![
            Field::new("not_geo", DataType::Binary, true),
            WKB_GEOMETRY.to_storage_field("some_geo", true).unwrap(),
        ]);

        let geoparquet_metadata_with_stats = GeoParquetMetadata::try_new(
            r#"{
                "columns": {
                    "some_geo": {
                        "encoding": "WKB",
                        "geometry_types": ["Point Z", "Linestring ZM"],
                        "bbox": [1.0, 2.0, 3.0, 4.0]
                    }
                },
                "version": "1.1.0",
                "primary_column": "some_geo"
            }"#,
        )
        .unwrap();

        let geoparquet_metadata_without_stats = GeoParquetMetadata::try_new(
            r#"{
                "columns": {
                    "some_geo": {
                        "encoding": "WKB",
                        "geometry_types": []
                    }
                },
                "version": "1.1.0",
                "primary_column": "some_geo"
            }"#,
        )
        .unwrap();

        assert_eq!(
            geoparquet_file_geo_stats(&file_schema.clone().into(), &geoparquet_metadata_with_stats)
                .unwrap(),
            vec![
                GeoStatistics::unspecified(),
                GeoStatistics::unspecified()
                    .with_bbox(Some(BoundingBox::xy((1.0, 3.0), (2.0, 4.0))))
                    .try_with_str_geometry_types(Some(&["Point Z", "Linestring ZM"]))
                    .unwrap()
            ]
        );

        assert_eq!(
            geoparquet_file_geo_stats(
                &file_schema.clone().into(),
                &geoparquet_metadata_without_stats
            )
            .unwrap(),
            vec![GeoStatistics::unspecified(), GeoStatistics::unspecified()]
        );
    }

    #[test]
    fn row_group_stats() {
        let file_schema = file_schema_with_covering();
        let parquet_schema = Arc::new(
            ArrowSchemaConverter::new()
                .convert(file_schema.as_ref())
                .unwrap(),
        );
        let covering_specs = [None, Some([2, 3, 4, 5]), None];

        let row_group_metadata_without_covering_stats =
            RowGroupMetaData::builder(parquet_schema.clone())
                .set_column_metadata(
                    (0..6)
                        .map(|i| {
                            ColumnChunkMetaData::builder(parquet_schema.column(i))
                                .build()
                                .unwrap()
                        })
                        .collect(),
                )
                .build()
                .unwrap();

        assert_eq!(
            row_group_covering_geo_stats(
                &row_group_metadata_without_covering_stats,
                &covering_specs
            ),
            vec![
                GeoStatistics::unspecified(),
                GeoStatistics::unspecified(),
                GeoStatistics::unspecified()
            ]
        );

        let xmin_stats = Statistics::Double(ValueStatistics::new(
            Some(-2.0),
            Some(-1.0),
            None,
            None,
            false,
        ));
        let ymin_stats = Statistics::Double(ValueStatistics::new(
            Some(0.0),
            Some(1.0),
            None,
            None,
            false,
        ));
        let xmax_stats = Statistics::Double(ValueStatistics::new(
            Some(2.0),
            Some(3.0),
            None,
            None,
            false,
        ));
        let ymax_stats = Statistics::Double(ValueStatistics::new(
            Some(4.0),
            Some(5.0),
            None,
            None,
            false,
        ));

        let row_group_metadata_with_covering_stats =
            RowGroupMetaData::builder(parquet_schema.clone())
                .add_column_metadata(
                    ColumnChunkMetaData::builder(parquet_schema.column(0))
                        .build()
                        .unwrap(),
                )
                .add_column_metadata(
                    ColumnChunkMetaData::builder(parquet_schema.column(1))
                        .build()
                        .unwrap(),
                )
                .add_column_metadata(
                    ColumnChunkMetaData::builder(parquet_schema.column(2))
                        .set_statistics(xmin_stats)
                        .build()
                        .unwrap(),
                )
                .add_column_metadata(
                    ColumnChunkMetaData::builder(parquet_schema.column(3))
                        .set_statistics(ymin_stats)
                        .build()
                        .unwrap(),
                )
                .add_column_metadata(
                    ColumnChunkMetaData::builder(parquet_schema.column(4))
                        .set_statistics(xmax_stats)
                        .build()
                        .unwrap(),
                )
                .add_column_metadata(
                    ColumnChunkMetaData::builder(parquet_schema.column(5))
                        .set_statistics(ymax_stats)
                        .build()
                        .unwrap(),
                )
                .build()
                .unwrap();

        assert_eq!(
            row_group_covering_geo_stats(&row_group_metadata_with_covering_stats, &covering_specs),
            vec![
                GeoStatistics::unspecified(),
                GeoStatistics::unspecified()
                    .with_bbox(Some(BoundingBox::xy((-2.0, 3.0), (0.0, 5.0)))),
                GeoStatistics::unspecified()
            ]
        );
    }

    #[test]
    fn column_stats() {
        // Good: Double + both min and max present
        assert_eq!(
            column_stats_min_max(&Statistics::Double(ValueStatistics::new(
                Some(-1.0),
                Some(1.0),
                None,
                None,
                false
            ))),
            Some((-1.0, 1.0))
        );

        // Good: Float + both min and max present
        assert_eq!(
            column_stats_min_max(&Statistics::Float(ValueStatistics::new(
                Some(-1.0),
                Some(1.0),
                None,
                None,
                false
            ))),
            Some((-1.0, 1.0))
        );

        // Bad: Double with no min value
        assert_eq!(
            column_stats_min_max(&Statistics::Double(ValueStatistics::new(
                None,
                Some(1.0),
                None,
                None,
                false
            ))),
            None
        );

        // Bad: Double with no max value
        assert_eq!(
            column_stats_min_max(&Statistics::Double(ValueStatistics::new(
                Some(-1.0),
                None,
                None,
                None,
                false
            ))),
            None
        );

        // Bad: Float with no min value
        assert_eq!(
            column_stats_min_max(&Statistics::Float(ValueStatistics::new(
                None,
                Some(1.0),
                None,
                None,
                false
            ))),
            None
        );

        // Bad: Float with no max value
        assert_eq!(
            column_stats_min_max(&Statistics::Float(ValueStatistics::new(
                Some(-1.0),
                None,
                None,
                None,
                false
            ))),
            None
        );

        // Bad: not Float or Double
        assert_eq!(
            column_stats_min_max(&Statistics::Int32(ValueStatistics::new(
                Some(-1),
                Some(1),
                None,
                None,
                false
            ))),
            None
        );
    }

    #[test]
    fn parse_coverings() {
        let geoparquet_metadata_without_covering = GeoParquetMetadata::try_new(
            r#"{
                "columns": {
                    "some_geo": {
                        "encoding": "WKB",
                        "geometry_types": []
                    }
                },
                "version": "1.1.0",
                "primary_column": "some_geo"
            }"#,
        )
        .unwrap();

        let geoparquet_metadata_with_invalid_covering = GeoParquetMetadata::try_new(
            r#"{
                "columns": {
                    "some_geo": {
                        "encoding": "WKB",
                        "geometry_types": [],
                        "covering": {
                            "bbox": {
                                "xmin": ["column_that_does_not_exist", "xmin"],
                                "ymin": ["column_that_does_not_exist", "ymin"],
                                "xmax": ["column_that_does_not_exist", "xmax"],
                                "ymax": ["column_that_does_not_exist", "ymax"]
                            }
                        }
                    }
                },
                "version": "1.1.0",
                "primary_column": "some_geo"
            }"#,
        )
        .unwrap();

        let parquet_schema = ArrowSchemaConverter::new()
            .convert(file_schema_with_covering().as_ref())
            .unwrap();
        let parquet_file_metadata =
            FileMetaData::new(0, 0, None, None, Arc::new(parquet_schema), None);
        let parquet_metadata = ParquetMetaDataBuilder::new(parquet_file_metadata).build();

        // Should return the correct (flattened) indices
        assert_eq!(
            parse_column_coverings(
                &file_schema_with_covering(),
                &parquet_metadata,
                &geoparquet_metadata_with_covering()
            )
            .unwrap(),
            vec![None, Some([2, 3, 4, 5]), None]
        );

        // If there is no covering in the GeoParquet metadata, should return all Nones
        assert_eq!(
            parse_column_coverings(
                &file_schema_with_covering(),
                &parquet_metadata,
                &geoparquet_metadata_without_covering
            )
            .unwrap(),
            vec![None, None, None]
        );

        // If there is a covering in the GeoParquet metadata but the columns don't exist,
        // we should also return all Nones
        // If there is no covering in the GeoParquet metadata, should return all Nones
        assert_eq!(
            parse_column_coverings(
                &file_schema_with_covering(),
                &parquet_metadata,
                &geoparquet_metadata_with_invalid_covering
            )
            .unwrap(),
            vec![None, None, None]
        );
    }

    #[test]
    fn schema_contains_geo() {
        let other_field = Field::new("not_geo", arrow_schema::DataType::Binary, true);
        let geometry_field = WKB_GEOMETRY.to_storage_field("geom", true).unwrap();
        let geography_field = WKB_GEOGRAPHY.to_storage_field("geog", true).unwrap();

        assert!(!storage_schema_contains_geo(&Schema::new([]).into()));
        assert!(!storage_schema_contains_geo(
            &Schema::new(vec![other_field.clone()]).into()
        ));

        assert!(storage_schema_contains_geo(
            &Schema::new(vec![geometry_field.clone()]).into()
        ));
        assert!(storage_schema_contains_geo(
            &Schema::new(vec![geography_field.clone()]).into()
        ));
    }

    #[test]
    fn parquet_geo_stats_empty() {
        let parquet_stats = GeospatialStatistics::new(None, None);
        let result = parquet_geo_stats_to_sedona_geo_stats(&parquet_stats);
        assert_eq!(result, GeoStatistics::unspecified());
    }

    #[test]
    fn parquet_geo_stats_valid_bbox_xy() {
        let parquet_stats = GeospatialStatistics::new(
            Some(parquet::geospatial::bounding_box::BoundingBox::new(
                -180.0, 180.0, -90.0, 90.0,
            )),
            None,
        );
        let result = parquet_geo_stats_to_sedona_geo_stats(&parquet_stats);
        assert_eq!(
            result,
            GeoStatistics::unspecified()
                .with_bbox(Some(BoundingBox::xy((-180.0, 180.0), (-90.0, 90.0))))
        );
    }

    #[test]
    fn parquet_geo_stats_valid_bbox_xyz() {
        let parquet_stats = GeospatialStatistics::new(
            Some(
                parquet::geospatial::bounding_box::BoundingBox::new(-180.0, 180.0, -90.0, 90.0)
                    .with_zrange(0.0, 1000.0),
            ),
            None,
        );
        let result = parquet_geo_stats_to_sedona_geo_stats(&parquet_stats);
        let expected_bbox = BoundingBox::xyzm(
            (-180.0, 180.0),
            (-90.0, 90.0),
            Some(Interval::new(0.0, 1000.0)),
            None,
        );
        assert_eq!(
            result,
            GeoStatistics::unspecified().with_bbox(Some(expected_bbox))
        );
    }

    #[test]
    fn parquet_geo_stats_valid_bbox_xyzm() {
        let parquet_stats = GeospatialStatistics::new(
            Some(
                parquet::geospatial::bounding_box::BoundingBox::new(-180.0, 180.0, -90.0, 90.0)
                    .with_zrange(0.0, 1000.0)
                    .with_mrange(0.0, 100.0),
            ),
            None,
        );
        let result = parquet_geo_stats_to_sedona_geo_stats(&parquet_stats);
        let expected_bbox = BoundingBox::xyzm(
            (-180.0, 180.0),
            (-90.0, 90.0),
            Some(Interval::new(0.0, 1000.0)),
            Some(Interval::new(0.0, 100.0)),
        );
        assert_eq!(
            result,
            GeoStatistics::unspecified().with_bbox(Some(expected_bbox))
        );
    }

    #[test]
    fn parquet_geo_stats_invalid_bbox_infinite_x() {
        let parquet_stats = GeospatialStatistics::new(
            Some(parquet::geospatial::bounding_box::BoundingBox::new(
                f64::NEG_INFINITY,
                f64::INFINITY,
                -90.0,
                90.0,
            )),
            None,
        );
        let result = parquet_geo_stats_to_sedona_geo_stats(&parquet_stats);
        // Should return unspecified because x width is not finite
        assert_eq!(result.bbox(), None);
    }

    #[test]
    fn parquet_geo_stats_invalid_bbox_nan_x() {
        let parquet_stats = GeospatialStatistics::new(
            Some(parquet::geospatial::bounding_box::BoundingBox::new(
                f64::NAN,
                f64::NAN,
                -90.0,
                90.0,
            )),
            None,
        );
        let result = parquet_geo_stats_to_sedona_geo_stats(&parquet_stats);
        // Should return unspecified because x width is not finite
        assert_eq!(result.bbox(), None);
    }

    #[test]
    fn parquet_geo_stats_invalid_bbox_infinite_y() {
        let parquet_stats = GeospatialStatistics::new(
            Some(parquet::geospatial::bounding_box::BoundingBox::new(
                -180.0,
                180.0,
                f64::NEG_INFINITY,
                f64::INFINITY,
            )),
            None,
        );
        let result = parquet_geo_stats_to_sedona_geo_stats(&parquet_stats);
        // Should return unspecified because y width is not finite
        assert_eq!(result.bbox(), None);
    }

    #[test]
    fn parquet_geo_stats_invalid_bbox_negative_y_width() {
        let parquet_stats = GeospatialStatistics::new(
            Some(parquet::geospatial::bounding_box::BoundingBox::new(
                -180.0, 180.0, 1.0, -1.0,
            )),
            None,
        );
        let result = parquet_geo_stats_to_sedona_geo_stats(&parquet_stats);
        // Should return unspecified because y width is less than zero
        assert_eq!(result.bbox(), None);
    }

    #[test]
    fn parquet_geo_stats_invalid_bbox_infinite_z() {
        let parquet_stats = GeospatialStatistics::new(
            Some(
                parquet::geospatial::bounding_box::BoundingBox::new(-180.0, 180.0, -90.0, 90.0)
                    .with_zrange(f64::NEG_INFINITY, f64::INFINITY),
            ),
            None,
        );
        let result = parquet_geo_stats_to_sedona_geo_stats(&parquet_stats);
        // Should return unspecified because z width is not finite
        assert_eq!(result.bbox(), None);
    }

    #[test]
    fn parquet_geo_stats_invalid_bbox_negative_z_width() {
        let parquet_stats = GeospatialStatistics::new(
            Some(
                parquet::geospatial::bounding_box::BoundingBox::new(-180.0, 180.0, -90.0, 90.0)
                    .with_zrange(100.0, -100.0),
            ),
            None,
        );
        let result = parquet_geo_stats_to_sedona_geo_stats(&parquet_stats);
        // Should return unspecified because z width is less than 0
        assert_eq!(result.bbox(), None);
    }

    #[test]
    fn parquet_geo_stats_invalid_bbox_infinite_m() {
        let parquet_stats = GeospatialStatistics::new(
            Some(
                parquet::geospatial::bounding_box::BoundingBox::new(-180.0, 180.0, -90.0, 90.0)
                    .with_mrange(f64::NEG_INFINITY, f64::INFINITY),
            ),
            None,
        );
        let result = parquet_geo_stats_to_sedona_geo_stats(&parquet_stats);
        // Should return unspecified because m width is not finite
        assert_eq!(result.bbox(), None);
    }

    #[test]
    fn parquet_geo_stats_invalid_bbox_negative_m_width() {
        let parquet_stats = GeospatialStatistics::new(
            Some(
                parquet::geospatial::bounding_box::BoundingBox::new(-180.0, 180.0, -90.0, 90.0)
                    .with_mrange(50.0, -50.0),
            ),
            None,
        );
        let result = parquet_geo_stats_to_sedona_geo_stats(&parquet_stats);
        // Should return unspecified because m width is less than 0
        assert_eq!(result.bbox(), None);
    }

    #[test]
    fn parquet_geo_stats_valid_geometry_types() {
        let parquet_stats = GeospatialStatistics::new(None, Some(vec![1, 2, 3]));
        let result = parquet_geo_stats_to_sedona_geo_stats(&parquet_stats);
        assert!(result.geometry_types().is_some());
    }

    #[test]
    fn parquet_geo_stats_invalid_geometry_types_negative() {
        let parquet_stats = GeospatialStatistics::new(None, Some(vec![1, -1, 3]));
        let result = parquet_geo_stats_to_sedona_geo_stats(&parquet_stats);
        // Should return unspecified geometry types because of negative value
        assert_eq!(result.geometry_types(), None);
    }

    #[test]
    fn parquet_geo_stats_invalid_geometry_types_unknown_wkb_id() {
        let parquet_stats = GeospatialStatistics::new(None, Some(vec![1, 999999]));
        let result = parquet_geo_stats_to_sedona_geo_stats(&parquet_stats);
        // Should return unspecified geometry types because of unknown WKB id
        assert_eq!(result.geometry_types(), None);
    }

    #[test]
    fn parquet_geo_stats_empty_geometry_types() {
        let parquet_stats = GeospatialStatistics::new(None, Some(vec![]));
        let result = parquet_geo_stats_to_sedona_geo_stats(&parquet_stats);
        // Empty geometry types should result in None
        assert_eq!(result.geometry_types(), None);
    }

    #[test]
    fn parquet_geo_stats_combined_valid() {
        let parquet_stats = GeospatialStatistics::new(
            Some(parquet::geospatial::bounding_box::BoundingBox::new(
                -180.0, 180.0, -90.0, 90.0,
            )),
            Some(vec![1, 2]), // Point, LineString
        );
        let result = parquet_geo_stats_to_sedona_geo_stats(&parquet_stats);
        assert!(result.bbox().is_some());
        assert!(result.geometry_types().is_some());
    }

    #[test]
    fn parquet_geo_stats_valid_bbox_invalid_geometry_types() {
        let parquet_stats = GeospatialStatistics::new(
            Some(parquet::geospatial::bounding_box::BoundingBox::new(
                -180.0, 180.0, -90.0, 90.0,
            )),
            Some(vec![-1]), // Invalid negative
        );
        let result = parquet_geo_stats_to_sedona_geo_stats(&parquet_stats);
        // Bbox should be valid, geometry types should be None
        assert!(result.bbox().is_some());
        assert_eq!(result.geometry_types(), None);
    }

    #[test]
    fn parquet_geo_stats_invalid_bbox_valid_geometry_types() {
        let parquet_stats = GeospatialStatistics::new(
            Some(parquet::geospatial::bounding_box::BoundingBox::new(
                f64::NAN,
                180.0,
                -90.0,
                90.0,
            )),
            Some(vec![1, 2]),
        );
        let result = parquet_geo_stats_to_sedona_geo_stats(&parquet_stats);
        // Bbox should be None, geometry types should be valid
        assert_eq!(result.bbox(), None);
        assert!(result.geometry_types().is_some());
    }

    #[test]
    fn validate_wkb_array_binary() {
        let valid_point_wkb: [u8; 21] = [
            0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40,
        ];

        let valid_array: BinaryArray = [Some(valid_point_wkb.as_slice()), None].iter().collect();
        validate_wkb_array(&valid_array, "geom").unwrap();

        let invalid_array: BinaryArray = [Some(&b"\x01"[..]), None].iter().collect();
        let err = validate_wkb_array(&invalid_array, "geom").unwrap_err();
        assert!(err.to_string().contains("WKB validation failed"));
    }

    #[test]
    fn validate_wkb_array_binary_view() {
        let valid_point_wkb: [u8; 21] = [
            0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40,
        ];

        let valid_array: BinaryViewArray =
            [Some(valid_point_wkb.as_slice()), None].iter().collect();
        validate_wkb_array(&valid_array, "geom").unwrap();

        let invalid_array: BinaryViewArray = [Some(&b"\x01"[..]), None].iter().collect();
        let err = validate_wkb_array(&invalid_array, "geom").unwrap_err();
        assert!(err.to_string().contains("WKB validation failed"));
    }

    #[test]
    fn validate_wkb_batch_errors_on_invalid_wkb() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new("geom", DataType::Binary, true),
        ]));

        let id_column: ArrayRef = Arc::new(Int64Array::from(vec![Some(1)]));
        let geom_array: BinaryArray = [Some(&b"\x01"[..])].iter().collect();
        let geom_column: ArrayRef = Arc::new(geom_array);

        let batch = RecordBatch::try_new(schema, vec![id_column, geom_column]).unwrap();
        let validation_columns = vec![(1, "geom".to_string())];
        let err = validate_wkb_batch(&batch, &validation_columns).unwrap_err();
        assert!(err.to_string().contains("WKB validation failed"));
    }

    fn file_schema_with_covering() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("not_geo", DataType::Binary, true),
            WKB_GEOMETRY.to_storage_field("some_geo", true).unwrap(),
            Field::new(
                "bbox",
                DataType::Struct(
                    vec![
                        Field::new("xmin", DataType::Float64, true),
                        Field::new("ymin", DataType::Float64, true),
                        Field::new("xmax", DataType::Float64, true),
                        Field::new("ymax", DataType::Float64, true),
                    ]
                    .into(),
                ),
                true,
            ),
        ]))
    }

    fn geoparquet_metadata_with_covering() -> GeoParquetMetadata {
        GeoParquetMetadata::try_new(
            r#"{
                "columns": {
                    "some_geo": {
                        "encoding": "WKB",
                        "geometry_types": [],
                        "covering": {
                            "bbox": {
                                "xmin": ["bbox", "xmin"],
                                "ymin": ["bbox", "ymin"],
                                "xmax": ["bbox", "xmax"],
                                "ymax": ["bbox", "ymax"]
                            }
                        }
                    }
                },
                "version": "1.1.0",
                "primary_column": "some_geo"
            }"#,
        )
        .unwrap()
    }
}
