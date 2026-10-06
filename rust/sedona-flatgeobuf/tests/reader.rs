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

use arrow_array::{Array, Int32Array};
use datafusion::{
    datasource::listing::{FileRange, ListingTableUrl},
    prelude::{SessionConfig, SessionContext},
};
use datafusion_execution::object_store::ObjectStoreUrl;
use flatgeobuf::geozero::{ColumnValue, GeomProcessor, GeozeroGeometry, PropertyProcessor};
use flatgeobuf::{ColumnType, FgbCrs, FgbWriter, FgbWriterOptions, GeometryType};
use object_store::{local::LocalFileSystem, path::Path};
use sedona_datasource::{
    provider::external_table,
    spec::{ExternalFormatSpec, Object, OpenReaderArgs},
};
use sedona_flatgeobuf::FlatGeobufFormatSpec;
use std::{fs::File, sync::Arc};
use tempfile::TempDir;

struct Point(f64);
impl GeozeroGeometry for Point {
    fn process_geom<P: GeomProcessor>(&self, p: &mut P) -> flatgeobuf::geozero::error::Result<()> {
        p.point_begin(0)?;
        p.xy(self.0, self.0 + 1.0, 0)?;
        p.point_end(0)
    }
}
fn fixture(index: bool, count: usize) -> (TempDir, Object) {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("points.fgb");
    let mut w = FgbWriter::create_with_options(
        "points",
        GeometryType::Point,
        FgbWriterOptions {
            write_index: index,
            promote_to_multi: false,
            crs: FgbCrs {
                code: 3857,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();
    w.add_column("id", ColumnType::Int, |_, _| {});
    w.add_column("label", ColumnType::String, |_, _| {});
    for i in 0..count {
        w.add_feature_geom(Point(i as f64), |f| {
            f.property(0, "id", &ColumnValue::Int(i as i32)).unwrap();
            if i % 2 == 0 {
                f.property(1, "label", &ColumnValue::String(&format!("point {i}")))
                    .unwrap();
            }
        })
        .unwrap();
    }
    w.write(File::create(&path).unwrap()).unwrap();
    let obj = Object {
        store: Some(Arc::new(LocalFileSystem::new())),
        url: Some(ObjectStoreUrl::local_filesystem()),
        meta: Some(object_store::ObjectMeta {
            location: Path::from_filesystem_path(&path).unwrap(),
            size: std::fs::metadata(&path).unwrap().len(),
            last_modified: Default::default(),
            e_tag: None,
            version: None,
        }),
        range: None,
    };
    (dir, obj)
}
fn args(obj: Object, projection: Option<Vec<usize>>) -> OpenReaderArgs {
    OpenReaderArgs {
        src: obj,
        batch_size: Some(3),
        file_schema: None,
        file_projection: projection,
        filters: vec![],
    }
}

#[tokio::test]
async fn projected_batches_preserve_schema_nulls_and_geometry() {
    let (_d, obj) = fixture(true, 11);
    let spec = FlatGeobufFormatSpec::default();
    let schema = spec.infer_schema(&obj).await.unwrap();
    assert_eq!(
        schema.field(2).metadata()["ARROW:extension:name"],
        "geoarrow.wkb"
    );
    let mut reader = spec
        .open_reader(&args(obj, Some(vec![1, 0])))
        .await
        .unwrap();
    assert_eq!(reader.schema().fields().len(), 2);
    let batches = reader.by_ref().collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 11);
    assert!(batches.iter().all(|b| b.num_rows() <= 3));
    assert_eq!(
        batches
            .iter()
            .map(|b| b.column(0).null_count())
            .sum::<usize>(),
        5
    );
}
#[tokio::test]
async fn arbitrary_byte_partitions_equal_serial_output() {
    for indexed in [false, true] {
        let (_d, obj) = fixture(indexed, 17);
        let spec = FlatGeobufFormatSpec::default();
        let serial = spec
            .open_reader(&args(obj.clone(), None))
            .await
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let mut got = vec![];
        let size = obj.meta.as_ref().unwrap().size as i64;
        for i in 0..13 {
            let mut part = obj.clone();
            part.range = Some(FileRange {
                start: size * i / 13,
                end: size * (i + 1) / 13,
            });
            got.extend(
                spec.open_reader(&args(part, None))
                    .await
                    .unwrap()
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap(),
            );
        }
        let ids = |batches: &[arrow_array::RecordBatch]| {
            batches
                .iter()
                .flat_map(|b| {
                    b.column(0)
                        .as_any()
                        .downcast_ref::<Int32Array>()
                        .unwrap()
                        .values()
                        .to_vec()
                })
                .collect::<Vec<_>>()
        };
        let rows = |batches: &[arrow_array::RecordBatch]| {
            batches
                .iter()
                .flat_map(|b| {
                    (0..b.num_rows())
                        .map(|r| {
                            b.columns()
                                .iter()
                                .map(|c| {
                                    datafusion_common::ScalarValue::try_from_array(c, r).unwrap()
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(rows(&got), rows(&serial));
        assert_eq!(ids(&got), ids(&serial));
        assert_eq!(ids(&serial).len(), 17);
        for b in &got {
            assert_eq!(b.schema(), serial[0].schema());
        }
    }
}
#[tokio::test]
async fn empty_projection_keeps_rows_and_empty_file_keeps_schema() {
    let (_d, obj) = fixture(true, 7);
    let spec = FlatGeobufFormatSpec::default();
    let batches = spec
        .open_reader(&args(obj, Some(vec![])))
        .await
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 7);
    assert!(batches.iter().all(|b| b.num_columns() == 0));
    let (_d, obj) = fixture(false, 0);
    let r = spec.open_reader(&args(obj, None)).await.unwrap();
    assert_eq!(r.schema().fields().len(), 3);
    assert_eq!(r.count(), 0);
}
#[tokio::test]
async fn datafusion_parallel_scan_matches_serial_scan() {
    #[derive(Debug, Clone, Default)]
    struct RecordingSpec {
        inner: FlatGeobufFormatSpec,
        ranges: Arc<std::sync::Mutex<Vec<Option<FileRange>>>>,
    }
    #[async_trait::async_trait]
    impl ExternalFormatSpec for RecordingSpec {
        async fn infer_schema(
            &self,
            o: &Object,
        ) -> datafusion_common::Result<arrow_schema::Schema> {
            self.inner.infer_schema(o).await
        }
        async fn open_reader(
            &self,
            a: &OpenReaderArgs,
        ) -> datafusion_common::Result<Box<dyn arrow_array::RecordBatchReader + Send>> {
            self.ranges.lock().unwrap().push(a.src.range.clone());
            self.inner.open_reader(a).await
        }
        fn with_options(
            &self,
            _: &std::collections::HashMap<String, String>,
        ) -> datafusion_common::Result<Arc<dyn ExternalFormatSpec>> {
            Ok(Arc::new(self.clone()))
        }
        fn extension(&self) -> &str {
            "fgb"
        }
        fn supports_repartition(&self) -> sedona_datasource::spec::SupportsRepartition {
            sedona_datasource::spec::SupportsRepartition::ByRange
        }
    }

    let (_d, obj) = fixture(true, 25);
    let mut results = vec![];
    for partitions in [1, 8] {
        let cfg = SessionConfig::new()
            .with_target_partitions(partitions)
            .with_batch_size(4)
            .set_usize("datafusion.optimizer.repartition_file_min_size", 0);
        let ctx = SessionContext::new_with_config(cfg);
        let spec = RecordingSpec::default();
        let table = external_table(
            Arc::new(spec.clone()),
            &ctx,
            vec![ListingTableUrl::parse(obj.to_url_string().unwrap()).unwrap()],
            true,
            Some(vec![]),
        )
        .await
        .unwrap();
        ctx.register_table("points", table).unwrap();
        let batches = ctx
            .sql("SELECT id, label FROM points WHERE id >= 4 ORDER BY id")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        if partitions > 1 {
            assert!(
                spec.ranges
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|r| r.is_some())
                    .count()
                    > 1,
                "DataFusion must open multiple byte-range readers"
            );
        }
        results.push(
            batches
                .iter()
                .flat_map(|b| {
                    b.column(0)
                        .as_any()
                        .downcast_ref::<Int32Array>()
                        .unwrap()
                        .values()
                        .to_vec()
                })
                .collect::<Vec<_>>(),
        );
    }
    assert_eq!(results[0], (4..25).collect::<Vec<_>>());
    assert_eq!(results[0], results[1]);
}

fn raw_geometry_file(
    xy: Option<&[f64]>,
    z: Option<&[f64]>,
    m: Option<&[f64]>,
    has_z: bool,
    has_m: bool,
) -> (TempDir, Object) {
    let (dir, mut obj) = fixture(false, 0);
    let path = dir.path().join("points.fgb");
    let mut b = flatbuffers::FlatBufferBuilder::new();
    let h = flatgeobuf::Header::create(
        &mut b,
        &flatgeobuf::HeaderArgs {
            geometry_type: GeometryType::Point,
            has_z,
            has_m,
            features_count: 1,
            index_node_size: 0,
            ..Default::default()
        },
    );
    b.finish_size_prefixed(h, None);
    let mut out = b"fgb\x03fgb\0".to_vec();
    out.extend_from_slice(b.finished_data());
    b.reset();
    let geom = if let Some(coords) = xy {
        let xy = b.create_vector(coords);
        let z = z.map(|v| b.create_vector(v));
        let m = m.map(|v| b.create_vector(v));
        Some(flatgeobuf::Geometry::create(
            &mut b,
            &flatgeobuf::GeometryArgs {
                xy: Some(xy),
                z,
                m,
                ..Default::default()
            },
        ))
    } else {
        None
    };
    let f = flatgeobuf::Feature::create(
        &mut b,
        &flatgeobuf::FeatureArgs {
            geometry: geom,
            ..Default::default()
        },
    );
    b.finish_size_prefixed(f, None);
    out.extend_from_slice(b.finished_data());
    std::fs::write(path, &out).unwrap();
    obj.meta.as_mut().unwrap().size = out.len() as u64;
    (dir, obj)
}
#[tokio::test]
async fn point_dimensions_and_null_empty_geometry() {
    use arrow_array::BinaryArray;
    for (z, m, code) in [
        (false, false, 1u32),
        (true, false, 1001),
        (false, true, 2001),
        (true, true, 3001),
    ] {
        let (_d, obj) = raw_geometry_file(
            Some(&[1., 2.]),
            z.then_some(&[3.][..]),
            m.then_some(&[4.][..]),
            z,
            m,
        );
        let batches = FlatGeobufFormatSpec::default()
            .open_reader(&args(obj, None))
            .await
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let a = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap();
        let data = a.value(0);
        assert_eq!(u32::from_le_bytes(data[1..5].try_into().unwrap()), code);
        assert_eq!(data.len(), 5 + 8 * (2 + usize::from(z) + usize::from(m)));
        wkb::reader::read_wkb(data).unwrap();
    }
    let (_d, obj) = raw_geometry_file(None, None, None, false, false);
    let batch = FlatGeobufFormatSpec::default()
        .open_reader(&args(obj, None))
        .await
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    assert_eq!(batch.column(0).null_count(), 1);
    let (_d, obj) = raw_geometry_file(Some(&[]), None, None, false, false);
    let batch = FlatGeobufFormatSpec::default()
        .open_reader(&args(obj, None))
        .await
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    let a = batch
        .column(0)
        .as_any()
        .downcast_ref::<BinaryArray>()
        .unwrap();
    assert!(f64::from_le_bytes(a.value(0)[5..13].try_into().unwrap()).is_nan());
}
#[tokio::test]
async fn malformed_coordinates_return_error() {
    for (xy, z) in [(&[1.][..], None), (&[1., 2.][..], Some(&[][..]))] {
        let (_d, obj) = raw_geometry_file(Some(xy), z, None, z.is_some(), false);
        let mut r = FlatGeobufFormatSpec::default()
            .open_reader(&args(obj, None))
            .await
            .unwrap();
        assert!(r.next().unwrap().is_err());
        assert!(r.next().is_none());
    }
}
#[tokio::test]
async fn invalid_arguments_and_truncated_files_are_errors() {
    let (_d, obj) = fixture(true, 7);
    let spec = FlatGeobufFormatSpec::default();
    let mut a = args(obj.clone(), None);
    a.batch_size = Some(0);
    assert!(spec.open_reader(&a).await.is_err());
    assert!(
        spec.open_reader(&args(obj.clone(), Some(vec![99])))
            .await
            .is_err()
    );
    let mut ranged = obj.clone();
    ranged.range = Some(FileRange { start: -1, end: 9 });
    assert!(spec.open_reader(&args(ranged, None)).await.is_err());
    let path = url::Url::parse(&obj.to_url_string().unwrap())
        .unwrap()
        .to_file_path()
        .unwrap();
    let size = std::fs::metadata(&path).unwrap().len();
    File::options()
        .write(true)
        .open(path)
        .unwrap()
        .set_len(size - 1)
        .unwrap();
    assert!(spec.infer_schema(&obj).await.is_err());
    let mut remote = obj;
    remote.url = Some(ObjectStoreUrl::parse("https://example.com").unwrap());
    assert!(spec.infer_schema(&remote).await.is_err());
    assert!(
        spec.with_options(&[("unknown".into(), "true".into())].into())
            .is_err()
    );
}
#[tokio::test]
async fn cache_invalidates_after_file_replacement() {
    let (_d, obj) = fixture(true, 9);
    let spec = FlatGeobufFormatSpec::default();
    spec.infer_schema(&obj).await.unwrap();
    let (_other, replacement) = fixture(true, 2);
    let to = url::Url::parse(&obj.to_url_string().unwrap())
        .unwrap()
        .to_file_path()
        .unwrap();
    let from = url::Url::parse(&replacement.to_url_string().unwrap())
        .unwrap()
        .to_file_path()
        .unwrap();
    std::fs::copy(from, to).unwrap();
    let batches = spec
        .open_reader(&args(obj, None))
        .await
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 2);
}

#[tokio::test]
async fn all_attribute_types_preserve_values_and_missing_nulls() {
    use datafusion_common::ScalarValue;
    let (_d, mut obj) = fixture(false, 0);
    let cols = [
        ColumnType::Byte,
        ColumnType::UByte,
        ColumnType::Bool,
        ColumnType::Short,
        ColumnType::UShort,
        ColumnType::Int,
        ColumnType::UInt,
        ColumnType::Long,
        ColumnType::ULong,
        ColumnType::Float,
        ColumnType::Double,
        ColumnType::String,
        ColumnType::Json,
        ColumnType::DateTime,
        ColumnType::Binary,
    ];
    let vals = [
        ColumnValue::Byte(-2),
        ColumnValue::UByte(3),
        ColumnValue::Bool(true),
        ColumnValue::Short(-4),
        ColumnValue::UShort(5),
        ColumnValue::Int(-6),
        ColumnValue::UInt(7),
        ColumnValue::Long(-8),
        ColumnValue::ULong(9),
        ColumnValue::Float(1.5),
        ColumnValue::Double(2.5),
        ColumnValue::String("hi"),
        ColumnValue::Json("{}"),
        ColumnValue::DateTime("2026-10-06T00:00:00Z"),
        ColumnValue::Binary(&[0, 255]),
    ];
    let mut w = FgbWriter::create_with_options(
        "types",
        GeometryType::Point,
        FgbWriterOptions {
            write_index: false,
            promote_to_multi: false,
            ..Default::default()
        },
    )
    .unwrap();
    for (i, t) in cols.into_iter().enumerate() {
        w.add_column(&format!("c{i}"), t, |_, _| {});
    }
    w.add_feature_geom(Point(0.), |f| {
        for (i, v) in vals.iter().enumerate() {
            f.property(i, &format!("c{i}"), v).unwrap();
        }
    })
    .unwrap();
    w.add_feature_geom(Point(1.), |_| {}).unwrap();
    let path = url::Url::parse(&obj.to_url_string().unwrap())
        .unwrap()
        .to_file_path()
        .unwrap();
    w.write(File::create(&path).unwrap()).unwrap();
    obj.meta.as_mut().unwrap().size = std::fs::metadata(path).unwrap().len();
    let b = FlatGeobufFormatSpec::default()
        .open_reader(&args(obj, None))
        .await
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    let expected = [
        ScalarValue::Int8(Some(-2)),
        ScalarValue::UInt8(Some(3)),
        ScalarValue::Boolean(Some(true)),
        ScalarValue::Int16(Some(-4)),
        ScalarValue::UInt16(Some(5)),
        ScalarValue::Int32(Some(-6)),
        ScalarValue::UInt32(Some(7)),
        ScalarValue::Int64(Some(-8)),
        ScalarValue::UInt64(Some(9)),
        ScalarValue::Float32(Some(1.5)),
        ScalarValue::Float64(Some(2.5)),
        ScalarValue::Utf8(Some("hi".into())),
        ScalarValue::Utf8(Some("{}".into())),
        ScalarValue::Utf8(Some("2026-10-06T00:00:00Z".into())),
        ScalarValue::Binary(Some(vec![0, 255])),
    ];
    for (i, v) in expected.into_iter().enumerate() {
        assert_eq!(ScalarValue::try_from_array(b.column(i), 0).unwrap(), v);
        assert!(b.column(i).is_null(1));
    }
}
#[tokio::test]
async fn multiple_files_and_count_projection() {
    let (dir, obj) = fixture(true, 12);
    let file = dir.path().join("points.fgb");
    std::fs::copy(file, dir.path().join("other.fgb")).unwrap();
    let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(4));
    let table = external_table(
        Arc::new(FlatGeobufFormatSpec::default()),
        &ctx,
        vec![
            ListingTableUrl::parse(url::Url::from_directory_path(dir.path()).unwrap().as_str())
                .unwrap(),
        ],
        true,
        Some(vec![]),
    )
    .await
    .unwrap();
    ctx.register_table("points", table).unwrap();
    let out = ctx
        .sql("SELECT COUNT(*) AS n FROM points")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(
        out[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .unwrap()
            .value(0),
        24
    );
    let schema = FlatGeobufFormatSpec::default()
        .infer_schema(&obj)
        .await
        .unwrap();
    assert!(schema.field(2).metadata()["ARROW:extension:metadata"].contains("EPSG:3857"));
}

#[tokio::test]
async fn missing_required_property_is_an_error() {
    let (_d, obj) = fixture(false, 0);
    let mut w = FgbWriter::create_with_options(
        "required",
        GeometryType::Point,
        FgbWriterOptions {
            write_index: false,
            promote_to_multi: false,
            ..Default::default()
        },
    )
    .unwrap();
    w.add_column("required", ColumnType::Int, |_, c| {
        c.nullable = false;
    });
    w.add_feature_geom(Point(1.), |_| {}).unwrap();
    let path = url::Url::parse(&obj.to_url_string().unwrap())
        .unwrap()
        .to_file_path()
        .unwrap();
    w.write(File::create(path).unwrap()).unwrap();
    let mut r = FlatGeobufFormatSpec::default()
        .open_reader(&args(obj, None))
        .await
        .unwrap();
    assert!(r.next().unwrap().is_err());
}
#[tokio::test]
async fn malformed_property_bytes_return_error_without_panicking() {
    for properties in [&[0u8, 0, 1][..], &[99u8, 0][..]] {
        let (_d, obj) = fixture(false, 0);
        let path = url::Url::parse(&obj.to_url_string().unwrap())
            .unwrap()
            .to_file_path()
            .unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        let mut b = flatbuffers::FlatBufferBuilder::new();
        let xy = b.create_vector(&[1., 2.]);
        let geometry = flatgeobuf::Geometry::create(
            &mut b,
            &flatgeobuf::GeometryArgs {
                xy: Some(xy),
                ..Default::default()
            },
        );
        let props = b.create_vector(properties);
        let f = flatgeobuf::Feature::create(
            &mut b,
            &flatgeobuf::FeatureArgs {
                geometry: Some(geometry),
                properties: Some(props),
                ..Default::default()
            },
        );
        b.finish_size_prefixed(f, None);
        bytes.extend_from_slice(b.finished_data());
        std::fs::write(path, bytes).unwrap();
        let mut r = FlatGeobufFormatSpec::default()
            .open_reader(&args(obj, None))
            .await
            .unwrap();
        assert!(r.next().unwrap().is_err());
        assert!(r.next().is_none());
    }
}

#[tokio::test]
async fn feature_local_schema_is_explicitly_unsupported() {
    let (_d, obj) = fixture(false, 0);
    let path = url::Url::parse(&obj.to_url_string().unwrap())
        .unwrap()
        .to_file_path()
        .unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    let mut b = flatbuffers::FlatBufferBuilder::new();
    let xy = b.create_vector(&[1., 2.]);
    let geometry = flatgeobuf::Geometry::create(
        &mut b,
        &flatgeobuf::GeometryArgs {
            xy: Some(xy),
            ..Default::default()
        },
    );
    let name = b.create_string("local");
    let col = flatgeobuf::Column::create(
        &mut b,
        &flatgeobuf::ColumnArgs {
            name: Some(name),
            type_: ColumnType::Int,
            ..Default::default()
        },
    );
    let cols = b.create_vector(&[col]);
    let f = flatgeobuf::Feature::create(
        &mut b,
        &flatgeobuf::FeatureArgs {
            geometry: Some(geometry),
            columns: Some(cols),
            ..Default::default()
        },
    );
    b.finish_size_prefixed(f, None);
    bytes.extend_from_slice(b.finished_data());
    std::fs::write(path, bytes).unwrap();
    let mut r = FlatGeobufFormatSpec::default()
        .open_reader(&args(obj, Some(vec![2])))
        .await
        .unwrap();
    assert!(r.next().unwrap().is_err());
}
