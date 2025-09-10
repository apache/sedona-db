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

use arrow_schema::SchemaRef;
use datafusion::datasource::{
    file_format::parquet::fetch_parquet_metadata,
    listing::PartitionedFile,
    physical_plan::{parquet::ParquetAccessPlan, FileMeta, FileOpenFuture, FileOpener},
};
use datafusion_common::Result;
use datafusion_physical_expr::PhysicalExpr;
use object_store::ObjectStore;
use parquet::file::{
    metadata::{ParquetMetaData, RowGroupMetaData},
    statistics::Statistics,
};
use sedona_expr::{spatial_filter::SpatialFilter, statistics::GeoStatistics};
use sedona_geometry::bounding_box::BoundingBox;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

use crate::metadata::GeoParquetMetadata;

/// Geo-aware [FileOpener] implementing file and row group pruning
///
/// Pruning happens (for Parquet) in the [FileOpener], so we implement
/// that here, too.
#[derive(Clone)]
pub struct GeoParquetFileOpener {
    inner: Arc<dyn FileOpener>,
    object_store: Arc<dyn ObjectStore>,
    metadata_size_hint: Option<usize>,
    predicate: Arc<dyn PhysicalExpr>,
    file_schema: SchemaRef,
}

impl GeoParquetFileOpener {
    /// Create a new file opener
    pub fn new(
        inner: Arc<dyn FileOpener>,
        object_store: Arc<dyn ObjectStore>,
        metadata_size_hint: Option<usize>,
        predicate: Arc<dyn PhysicalExpr>,
        file_schema: SchemaRef,
    ) -> Self {
        Self {
            inner,
            object_store,
            metadata_size_hint,
            predicate,
            file_schema,
        }
    }
}

impl FileOpener for GeoParquetFileOpener {
    fn open(&self, file_meta: FileMeta, _file: PartitionedFile) -> Result<FileOpenFuture> {
        let self_clone = self.clone();

        Ok(Box::pin(async move {
            let parquet_metadata = fetch_parquet_metadata(
                &self_clone.object_store,
                &file_meta.object_meta,
                self_clone.metadata_size_hint,
                None,
            )
            .await?;

            let mut access_plan = ParquetAccessPlan::new_all(parquet_metadata.num_row_groups());
            let spatial_filter = SpatialFilter::try_from_expr(&self_clone.predicate)?;

            if let Some(geoparquet_metadata) =
                GeoParquetMetadata::try_from_parquet_metadata(&parquet_metadata)?
            {
                filter_access_plan_using_geoparquet_file_metadata(
                    &self_clone.file_schema,
                    &mut access_plan,
                    &spatial_filter,
                    &geoparquet_metadata,
                )?;

                filter_access_plan_using_geoparquet_covering(
                    &self_clone.file_schema,
                    &mut access_plan,
                    &spatial_filter,
                    &geoparquet_metadata,
                    &parquet_metadata,
                )?;
            }

            // When we have built-in GEOMETRY/GEOGRAPHY types, we can filter the access plan
            // from the native GeoStatistics here.

            // We could also consider filtering using null_count here in the future (i.e.,
            // skip row groups that are all null)

            let file_meta = FileMeta {
                object_meta: file_meta.object_meta,
                range: file_meta.range,
                extensions: Some(Arc::new(access_plan)),
                metadata_size_hint: self_clone.metadata_size_hint,
            };

            self_clone.inner.open(file_meta, _file)?.await
        }))
    }
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
) -> Result<()> {
    let table_geo_stats = geoparquet_file_geo_stats(file_schema, metadata)?;
    if !spatial_filter.evaluate(&table_geo_stats) {
        for i in access_plan.row_group_indexes() {
            access_plan.skip(i);
        }
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
) -> Result<()> {
    let row_group_indices_to_scan = access_plan.row_group_indexes();

    // What we're about to do is a bit of work, so skip it if we can.
    if row_group_indices_to_scan.is_empty() {
        return Ok(());
    }

    // The GeoParquetMetadata refers to bbox covering columns as e.g. {"xmin": ["bbox", "xmin"]},
    // but we need flattened integer references to retrieve min/max statistics for each of these.
    let covering_specs = parse_column_coverings(file_schema, parquet_metadata, metadata)?;

    // Iterate through the row groups
    for i in row_group_indices_to_scan {
        // Generate row group statistics based on the covering statistics
        let row_group_geo_stats =
            row_group_covering_geo_stats(parquet_metadata.row_group(i), &covering_specs);

        // Evaluate predicate!
        if !spatial_filter.evaluate(&row_group_geo_stats) {
            access_plan.skip(i);
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
