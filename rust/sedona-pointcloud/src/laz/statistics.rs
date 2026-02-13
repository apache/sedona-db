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

use std::{collections::HashSet, io::Cursor, sync::Arc};

use arrow_array::{
    builder::PrimitiveBuilder,
    cast::AsArray,
    types::{Float64Type, UInt64Type},
    ArrayRef, BooleanArray, Float64Array, Int32Array, RecordBatch, UInt64Array,
};
use arrow_ipc::{reader::FileReader, writer::FileWriter};
use arrow_schema::{DataType, Field, Schema};
use datafusion_common::{arrow::compute::concat_batches, Column, DataFusionError, ScalarValue};
use datafusion_pruning::PruningStatistics;
use las::{raw::Point as RawPoint, Header, Point};
use object_store::{path::Path, ObjectMeta, ObjectStore, PutPayload};

use sedona_geometry::bounding_box::BoundingBox;

use crate::laz::{metadata::ChunkMeta, reader::record_decompressor};

/// Spatial statistics (extent) of LAZ chunks for pruning.
///
/// It wraps a `RecordBatch` with x, y, z min and max values and row count per chunk.
#[derive(Clone, Debug)]
pub struct LazStatistics {
    pub values: RecordBatch,
}

impl LazStatistics {
    /// Get the [BoundingBox] of a chunk by index.
    pub fn get_bbox(&self, index: usize) -> Option<BoundingBox> {
        if index >= self.values.num_rows() {
            None
        } else {
            let xmin = unsafe {
                self.values
                    .column(0)
                    .as_primitive::<Float64Type>()
                    .value_unchecked(index)
            };
            let xmax = unsafe {
                self.values
                    .column(1)
                    .as_primitive::<Float64Type>()
                    .value_unchecked(index)
            };
            let ymin = unsafe {
                self.values
                    .column(2)
                    .as_primitive::<Float64Type>()
                    .value_unchecked(index)
            };
            let ymax = unsafe {
                self.values
                    .column(3)
                    .as_primitive::<Float64Type>()
                    .value_unchecked(index)
            };
            let zmin = unsafe {
                self.values
                    .column(4)
                    .as_primitive::<Float64Type>()
                    .value_unchecked(index)
            };
            let zmax = unsafe {
                self.values
                    .column(5)
                    .as_primitive::<Float64Type>()
                    .value_unchecked(index)
            };

            let bbox =
                BoundingBox::xyzm((xmin, xmax), (ymin, ymax), Some((zmin, zmax).into()), None);

            Some(bbox)
        }
    }
}

impl PruningStatistics for LazStatistics {
    fn min_values(&self, column: &Column) -> Option<ArrayRef> {
        match column.name.as_str() {
            "x" => self.values.column_by_name("x_min").cloned(),
            "y" => self.values.column_by_name("y_min").cloned(),
            "z" => self.values.column_by_name("z_min").cloned(),
            _ => None,
        }
    }

    fn max_values(&self, column: &Column) -> Option<ArrayRef> {
        match column.name.as_str() {
            "x" => self.values.column_by_name("x_max").cloned(),
            "y" => self.values.column_by_name("y_max").cloned(),
            "z" => self.values.column_by_name("z_max").cloned(),
            _ => None,
        }
    }

    fn num_containers(&self) -> usize {
        self.values.num_rows()
    }

    fn null_counts(&self, column: &Column) -> Option<ArrayRef> {
        match column.name.as_str() {
            "x" | "y" | "z" => Some(Arc::new(Int32Array::from_value(0, self.values.num_rows()))),
            _ => None,
        }
    }

    fn row_counts(&self, _column: &Column) -> Option<ArrayRef> {
        self.values.column_by_name("row_counts").cloned()
    }

    fn contained(&self, _column: &Column, _values: &HashSet<ScalarValue>) -> Option<BooleanArray> {
        None
    }
}

pub struct LasStatisticsBuilder {
    x_min: PrimitiveBuilder<Float64Type>,
    x_max: PrimitiveBuilder<Float64Type>,
    y_min: PrimitiveBuilder<Float64Type>,
    y_max: PrimitiveBuilder<Float64Type>,
    z_min: PrimitiveBuilder<Float64Type>,
    z_max: PrimitiveBuilder<Float64Type>,
    row_counts: PrimitiveBuilder<UInt64Type>,
}

impl LasStatisticsBuilder {
    pub fn new_with_capacity(capacity: usize) -> Self {
        LasStatisticsBuilder {
            x_min: Float64Array::builder(capacity),
            x_max: Float64Array::builder(capacity),
            y_min: Float64Array::builder(capacity),
            y_max: Float64Array::builder(capacity),
            z_min: Float64Array::builder(capacity),
            z_max: Float64Array::builder(capacity),
            row_counts: UInt64Array::builder(capacity),
        }
    }

    pub fn add_values(&mut self, values: &[f64; 6], row_count: u64) {
        self.x_min.append_value(values[0]);
        self.x_max.append_value(values[1]);
        self.y_min.append_value(values[2]);
        self.y_max.append_value(values[3]);
        self.z_min.append_value(values[4]);
        self.z_max.append_value(values[5]);
        self.row_counts.append_value(row_count);
    }

    pub fn finish(mut self) -> LazStatistics {
        let schema = Schema::new([
            Arc::new(Field::new("x_min", DataType::Float64, false)),
            Arc::new(Field::new("x_max", DataType::Float64, false)),
            Arc::new(Field::new("y_min", DataType::Float64, false)),
            Arc::new(Field::new("y_max", DataType::Float64, false)),
            Arc::new(Field::new("z_min", DataType::Float64, false)),
            Arc::new(Field::new("z_max", DataType::Float64, false)),
            Arc::new(Field::new("row_counts", DataType::UInt64, false)),
        ]);

        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(self.x_min.finish()),
                Arc::new(self.x_max.finish()),
                Arc::new(self.y_min.finish()),
                Arc::new(self.y_max.finish()),
                Arc::new(self.z_min.finish()),
                Arc::new(self.z_max.finish()),
                Arc::new(self.row_counts.finish()),
            ],
        )
        .unwrap();

        LazStatistics { values: batch }
    }
}

/// Extract the [LazStatistics] from a LAZ file in an object store.
///
/// This will scan the entire file. To reuse the statistics, they can
/// optionally be persisted, which creates a sidecar file with a `.stats`
/// extension next to the original file like `path/to/file.laz.stats`.
pub async fn chunk_statistics(
    store: &dyn ObjectStore,
    object_meta: &ObjectMeta,
    chunk_table: &[ChunkMeta],
    header: &Header,
    persist: bool,
) -> Result<LazStatistics, DataFusionError> {
    let stats_path = Path::parse(format!("{}.stats", object_meta.location.as_ref()))?;

    match store.head(&stats_path).await {
        Ok(stats_object) => {
            if stats_object.last_modified < object_meta.last_modified {
                return Err(DataFusionError::Internal(
                    "Statistics are outdated".to_string(),
                ));
            }
            // read persisted statistics
            let bytes = store.get(&stats_path).await?.bytes().await?;
            let reader = FileReader::try_new_buffered(Cursor::new(bytes), None)?;
            let schema = reader.schema();

            let batches = reader.collect::<Result<Vec<_>, _>>()?;
            let values = concat_batches(&schema, &batches)?;

            assert_eq!(values.num_rows(), chunk_table.len());

            Ok(LazStatistics { values })
        }
        Err(object_store::Error::NotFound { path: _, source: _ }) => {
            // extract statistics
            let mut builder = LasStatisticsBuilder::new_with_capacity(chunk_table.len());

            for chunk_meta in chunk_table {
                let stats = extract_chunk_stats(store, object_meta, chunk_meta, header).await?;
                builder.add_values(&stats, chunk_meta.num_points);
            }

            let stats = builder.finish();

            if persist {
                // persist statistics
                let writer = Cursor::new(Vec::with_capacity(stats.values.get_array_memory_size()));
                let mut writer = FileWriter::try_new_buffered(writer, &stats.values.schema())?;
                writer.write(&stats.values)?;

                let bytes = writer.into_inner()?.into_inner().unwrap().into_inner();
                let payload = PutPayload::from_bytes(bytes.into());
                store.put(&stats_path, payload).await?;
            }

            Ok(stats)
        }
        Err(e) => Err(e.into()),
    }
}

async fn extract_chunk_stats(
    store: &dyn ObjectStore,
    object_meta: &ObjectMeta,
    chunk_meta: &ChunkMeta,
    header: &Header,
) -> Result<[f64; 6], DataFusionError> {
    // fetch chunk bytes
    let bytes = store
        .get_range(&object_meta.location, chunk_meta.byte_range.clone())
        .await?;

    // setup laz decompressor
    let mut decompressor =
        record_decompressor(header, bytes).map_err(|e| DataFusionError::External(Box::new(e)))?;

    // statistics
    let mut stats = [
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];

    let out = vec![0; header.point_format().len() as usize];
    let mut buffer = Cursor::new(out);

    for _ in 0..chunk_meta.num_points {
        buffer.set_position(0);
        decompressor.decompress_next(buffer.get_mut())?;

        let point = RawPoint::read_from(&mut buffer, header.point_format())
            .map(|raw_point| Point::new(raw_point, header.transforms()))
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        stats = [
            stats[0].min(point.x),
            stats[1].max(point.x),
            stats[2].min(point.y),
            stats[3].max(point.y),
            stats[4].min(point.z),
            stats[5].max(point.z),
        ];
    }

    Ok(stats)
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use arrow_array::{cast::AsArray, types::UInt64Type};
    use datafusion_common::Column;
    use datafusion_pruning::PruningStatistics;
    use las::{point::Format, Builder, Point, Writer};
    use object_store::{local::LocalFileSystem, path::Path, ObjectStore};
    use sedona_geometry::bounding_box::BoundingBox;

    use crate::{laz::metadata::LazMetadataReader, options::PointcloudOptions};

    #[tokio::test]
    async fn chunk_statistics() {
        let path = "tests/data/large.laz";

        // read with `LazMetadataReader`
        let store = LocalFileSystem::new();
        let location = Path::from_filesystem_path(path).unwrap();
        let object_meta = store.head(&location).await.unwrap();

        let metadata_reader = LazMetadataReader::new(&store, &object_meta);
        let metadata = metadata_reader.fetch_metadata().await.unwrap();
        assert!(metadata.statistics.is_none());

        let options = PointcloudOptions {
            collect_statistics: true,
            ..Default::default()
        };
        let metadata_reader = LazMetadataReader::new(&store, &object_meta).with_options(options);
        let metadata = metadata_reader.fetch_metadata().await.unwrap();
        let statistics = metadata.statistics.as_ref().unwrap();
        assert_eq!(statistics.num_containers(), 2);
        assert_eq!(
            statistics
                .row_counts(&Column::from_name(""))
                .unwrap()
                .as_primitive::<UInt64Type>()
                .value(0),
            50000
        );
        assert_eq!(
            statistics.get_bbox(0),
            Some(BoundingBox::xyzm(
                (0.5, 0.5),
                (0.5, 0.5),
                Some((0.5, 0.5).into()),
                None
            ))
        );
        assert_eq!(
            statistics.get_bbox(1),
            Some(BoundingBox::xyzm(
                (1.0, 1.0),
                (1.0, 1.0),
                Some((1.0, 1.0).into()),
                None
            ))
        );
    }

    #[tokio::test]
    async fn persist_statistics() {
        let tmpdir = tempfile::tempdir().unwrap();

        let tmp_path = tmpdir.path().join("tmp.laz");
        let tmp_file = File::create(&tmp_path).unwrap();

        // create laz file
        let mut builder = Builder::from((1, 4));
        builder.point_format = Format::new(0).unwrap();
        builder.point_format.is_compressed = true;
        let header = builder.into_header().unwrap();
        let mut writer = Writer::new(tmp_file, header).unwrap();
        let point = Point {
            x: 0.5,
            y: 0.5,
            z: 0.5,
            ..Default::default()
        };
        writer.write_point(point).unwrap();
        writer.close().unwrap();

        // read with `LazMetadataReader`
        let store = LocalFileSystem::new();
        let location = Path::from_filesystem_path(&tmp_path).unwrap();
        let object_meta = store.head(&location).await.unwrap();

        let options = PointcloudOptions {
            collect_statistics: true,
            persist_statistics: true,
            ..Default::default()
        };
        let metadata_reader = LazMetadataReader::new(&store, &object_meta).with_options(options);
        let metadata = metadata_reader.fetch_metadata().await.unwrap();

        assert!(tmp_path.with_extension("laz.stats").exists());
        assert_eq!(
            metadata.statistics.as_ref().unwrap().get_bbox(0),
            metadata_reader
                .fetch_metadata()
                .await
                .unwrap()
                .statistics
                .as_ref()
                .unwrap()
                .get_bbox(0)
        );
    }
}
