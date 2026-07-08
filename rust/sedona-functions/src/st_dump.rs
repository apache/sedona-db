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
use arrow_array::{
    builder::{BinaryBuilder, NullBufferBuilder, OffsetBufferBuilder, UInt32Builder},
    ListArray, StructArray,
};
use arrow_schema::{DataType, Field, Fields};
use datafusion_common::{config::ConfigOptions, Result};
use datafusion_expr::{ColumnarValue, Volatility};
use geo_traits::{
    GeometryCollectionTrait, GeometryTrait, GeometryType, MultiLineStringTrait, MultiPointTrait,
    MultiPolygonTrait,
};
use sedona_common::sedona_internal_err;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_geometry::wkb_factory::WKB_MIN_PROBABLE_BYTES;
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOGRAPHY, WKB_GEOMETRY},
    matchers::ArgMatcher,
};
use std::{io::Write, sync::Arc};

use crate::executor::WkbExecutor;

/// ST_Dump() scalar UDF
///
/// Native implementation to get all the points of a geometry as MULTIPOINT
pub fn st_dump_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_dump",
        vec![
            Arc::new(STDump {
                matcher: ArgMatcher::new(vec![ArgMatcher::is_geometry()], WKB_GEOMETRY),
            }),
            Arc::new(STDump {
                matcher: ArgMatcher::new(vec![ArgMatcher::is_geography()], WKB_GEOGRAPHY),
            }),
        ],
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct STDump {
    matcher: ArgMatcher,
}

// A builder for a list of the structs
struct STDumpBuilder {
    path_array_builder: UInt32Builder,
    path_array_offsets_builder: OffsetBufferBuilder<i32>,
    geom_builder: BinaryBuilder,
    struct_offsets_builder: OffsetBufferBuilder<i32>,
    null_builder: NullBufferBuilder,
    parent_path: Vec<u32>,
    return_type: SedonaType,
}

impl STDumpBuilder {
    fn new(num_iter: usize, return_type: SedonaType) -> Self {
        let path_array_builder = UInt32Builder::with_capacity(num_iter);
        let path_array_offsets_builder = OffsetBufferBuilder::new(num_iter);
        let geom_builder =
            BinaryBuilder::with_capacity(num_iter, WKB_MIN_PROBABLE_BYTES * num_iter);
        let struct_offsets_builder = OffsetBufferBuilder::new(num_iter);
        let null_builder = NullBufferBuilder::new(num_iter);

        Self {
            path_array_builder,
            path_array_offsets_builder,
            geom_builder,
            struct_offsets_builder,
            null_builder,
            parent_path: Vec::new(), // Reusable buffer to avoid allocation per row
            return_type,
        }
    }

    // This appends both path and geom at once.
    fn append_single_struct(&mut self, cur_index: Option<u32>, wkb: &[u8]) -> Result<()> {
        self.path_array_builder.append_slice(&self.parent_path);
        if let Some(cur_index) = cur_index {
            self.path_array_builder.append_value(cur_index);
            self.path_array_offsets_builder
                .push_length(self.parent_path.len() + 1);
        } else {
            self.path_array_offsets_builder
                .push_length(self.parent_path.len());
        }

        self.geom_builder.write_all(wkb)?;
        self.geom_builder.append_value([]);

        Ok(())
    }

    fn append_structs(&mut self, wkb: &wkb::reader::Wkb<'_>) -> Result<i32> {
        match wkb.as_type() {
            GeometryType::Point(point) => {
                self.append_single_struct(None, point.buf())?;
                Ok(1)
            }
            GeometryType::LineString(line_string) => {
                self.append_single_struct(None, line_string.buf())?;
                Ok(1)
            }
            GeometryType::Polygon(polygon) => {
                self.append_single_struct(None, polygon.buf())?;
                Ok(1)
            }
            GeometryType::MultiPoint(multi_point) => {
                for (index, point) in multi_point.points().enumerate() {
                    self.append_single_struct(Some((index + 1) as _), point.buf())?;
                }
                Ok(multi_point.num_points() as _)
            }
            GeometryType::MultiLineString(multi_line_string) => {
                for (index, line_string) in multi_line_string.line_strings().enumerate() {
                    self.append_single_struct(Some((index + 1) as _), line_string.buf())?;
                }
                Ok(multi_line_string.num_line_strings() as _)
            }
            GeometryType::MultiPolygon(multi_polygon) => {
                for (index, polygon) in multi_polygon.polygons().enumerate() {
                    self.append_single_struct(Some((index + 1) as _), polygon.buf())?;
                }
                Ok(multi_polygon.num_polygons() as _)
            }
            GeometryType::GeometryCollection(geometry_collection) => {
                let mut num_geometries: i32 = 0;

                self.parent_path.push(0); // add an index for the next nested level

                for geometry in geometry_collection.geometries() {
                    // increment the index
                    if let Some(index) = self.parent_path.last_mut() {
                        *index += 1;
                    }
                    num_geometries += self.append_structs(geometry)?;
                }

                self.parent_path.truncate(self.parent_path.len() - 1); // clear the index before returning to the upper level

                Ok(num_geometries)
            }
            _ => sedona_internal_err!("Invalid geometry type"),
        }
    }

    fn append(&mut self, wkb: &wkb::reader::Wkb<'_>) -> Result<()> {
        self.parent_path.clear();

        let num_geometries = self.append_structs(wkb)?;
        self.null_builder.append(true);
        self.struct_offsets_builder
            .push_length(num_geometries as usize);
        Ok(())
    }

    fn append_null(&mut self) {
        self.path_array_offsets_builder.push_length(0);
        self.geom_builder.append_null();
        self.struct_offsets_builder.push_length(1);
        self.null_builder.append(false);
    }

    fn finish(mut self) -> Result<ListArray> {
        let path_array = Arc::new(self.path_array_builder.finish());
        let path_offsets = self.path_array_offsets_builder.finish();
        let geom_array = self.geom_builder.finish();

        let path_field = Arc::new(Field::new("item", DataType::UInt32, true));
        let path_list = ListArray::new(path_field, path_offsets, path_array, None);

        let SedonaType::Arrow(DataType::List(return_list_field)) = self.return_type else {
            return sedona_internal_err!("Unexpected return type in st_dump()");
        };

        let DataType::Struct(fields) = return_list_field.data_type() else {
            return sedona_internal_err!("Unexpected return type");
        };

        let struct_array = StructArray::try_new(
            fields.clone(),
            vec![Arc::new(path_list), Arc::new(geom_array)],
            None,
        )
        .unwrap();

        let struct_offsets = self.struct_offsets_builder.finish();
        let nulls = self.null_builder.finish();
        Ok(ListArray::new(
            return_list_field,
            struct_offsets,
            Arc::new(struct_array),
            nulls,
        ))
    }
}

impl SedonaScalarKernel for STDump {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        Ok(self
            .matcher
            .match_args(args)?
            .map(|output_type| geometry_dump_type(&output_type)))
    }

    fn invoke_batch_from_args(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
        return_type: &SedonaType,
        _num_rows: usize,
        _config_options: Option<&ConfigOptions>,
    ) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);

        let mut builder = STDumpBuilder::new(executor.num_iterations(), return_type.clone());
        executor.execute_wkb_void(|maybe_wkb| {
            if let Some(wkb) = maybe_wkb {
                builder.append(&wkb)?;
            } else {
                builder.append_null();
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()?))
    }

    fn invoke_batch(
        &self,
        _arg_types: &[SedonaType],
        _args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        sedona_internal_err!("invoke_batch() should not be called for st_dump()")
    }
}

fn geometry_dump_fields(geo_type: &SedonaType) -> Fields {
    let path = Field::new(
        "path",
        DataType::List(Field::new("item", DataType::UInt32, true).into()),
        true,
    );
    let geom = geo_type.to_storage_field("geom", true).unwrap();
    vec![path, geom].into()
}

fn geometry_dump_type(geo_type: &SedonaType) -> SedonaType {
    let fields = geometry_dump_fields(geo_type);
    let struct_type = DataType::Struct(fields);

    SedonaType::Arrow(DataType::List(Field::new("item", struct_type, true).into()))
}

#[cfg(test)]
mod tests {
    use arrow_array::{Array, ArrayRef, ListArray, StructArray, UInt32Array};
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_geometry::types::Edges;
    use sedona_schema::crs::lnglat;
    use sedona_testing::{
        compare::assert_array_equal, create::create_array, testers::ScalarUdfTester,
    };

    use super::*;

    #[test]
    fn udf_metadata() {
        let st_dump_udf: ScalarUDF = st_dump_udf().into();
        assert_eq!(st_dump_udf.name(), "st_dump");
        assert!(st_dump_udf.documentation().is_none());
    }

    #[rstest]
    fn udf(
        #[values(WKB_GEOMETRY, WKB_GEOGRAPHY, SedonaType::Wkb(Edges::Planar, lnglat()))]
        sedona_type: SedonaType,
    ) {
        let tester = ScalarUdfTester::new(st_dump_udf().into(), vec![sedona_type.clone()]);

        let input = create_array(
            &[
                Some("POINT (1 2)"),
                Some("LINESTRING (1 1, 2 2)"),
                Some("POLYGON ((1 1, 2 2, 2 1, 1 1))"),
                Some("MULTIPOINT (1 1, 2 2)"),
                Some("MULTILINESTRING ((1 1, 2 2), EMPTY, (3 3, 4 4))"),
                Some("MULTIPOLYGON (((1 1, 2 2, 2 1, 1 1)), EMPTY, ((3 3, 4 4, 4 3, 3 3)))"),
                Some("GEOMETRYCOLLECTION (POINT (1 2), MULTILINESTRING ((1 1, 2 2), EMPTY, (3 3, 4 4)), LINESTRING (1 1, 2 2))"),
                Some("GEOMETRYCOLLECTION (POINT (1 2), GEOMETRYCOLLECTION (MULTILINESTRING ((1 1, 2 2), EMPTY, (3 3, 4 4)), LINESTRING (1 1, 2 2)))"),
            ],
            &sedona_type,
        );
        let result = tester.invoke_array(input).unwrap();
        assert_dump_row(&result, 0, &[(&[], Some("POINT (1 2)"))], &sedona_type);
        assert_dump_row(
            &result,
            1,
            &[(&[], Some("LINESTRING (1 1, 2 2)"))],
            &sedona_type,
        );
        assert_dump_row(
            &result,
            2,
            &[(&[], Some("POLYGON ((1 1, 2 2, 2 1, 1 1))"))],
            &sedona_type,
        );
        assert_dump_row(
            &result,
            3,
            &[(&[1], Some("POINT (1 1)")), (&[2], Some("POINT (2 2)"))],
            &sedona_type,
        );

        assert_dump_row(
            &result,
            4,
            &[
                (&[1], Some("LINESTRING (1 1, 2 2)")),
                (&[2], Some("LINESTRING EMPTY")),
                (&[3], Some("LINESTRING (3 3, 4 4)")),
            ],
            &sedona_type,
        );
        assert_dump_row(
            &result,
            5,
            &[
                (&[1], Some("POLYGON ((1 1, 2 2, 2 1, 1 1))")),
                (&[2], Some("POLYGON EMPTY")),
                (&[3], Some("POLYGON ((3 3, 4 4, 4 3, 3 3)))")),
            ],
            &sedona_type,
        );
        assert_dump_row(
            &result,
            6,
            &[
                (&[1], Some("POINT (1 2)")),
                (&[2, 1], Some("LINESTRING (1 1, 2 2)")),
                (&[2, 2], Some("LINESTRING EMPTY")),
                (&[2, 3], Some("LINESTRING (3 3, 4 4)")),
                (&[3], Some("LINESTRING (1 1, 2 2)")),
            ],
            &sedona_type,
        );
        assert_dump_row(
            &result,
            7,
            &[
                (&[1], Some("POINT (1 2)")),
                (&[2, 1, 1], Some("LINESTRING (1 1, 2 2)")),
                (&[2, 1, 2], Some("LINESTRING EMPTY")),
                (&[2, 1, 3], Some("LINESTRING (3 3, 4 4)")),
                (&[2, 2], Some("LINESTRING (1 1, 2 2)")),
            ],
            &sedona_type,
        );

        let null_input = create_array(&[None], &sedona_type);
        let result = tester.invoke_array(null_input).unwrap();
        assert_dump_row_null(&result, 0);
    }

    fn assert_dump_row(
        result: &ArrayRef,
        row: usize,
        expected: &[(&[u32], Option<&str>)],
        sedona_type: &SedonaType,
    ) {
        let list_array = result
            .as_ref()
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("result should be a ListArray");
        assert!(
            !list_array.is_null(row),
            "row {row} should not be null in dump result"
        );
        let dumped = list_array.value(row);
        let dumped = dumped
            .as_ref()
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("list elements should be StructArray");
        assert_eq!(dumped.len(), expected.len());

        let path_array = dumped
            .column(0)
            .as_ref()
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("path should be a ListArray");
        assert_eq!(path_array.len(), expected.len());
        for (i, (expected_path, _)) in expected.iter().enumerate() {
            let path_array_value = path_array.value(i);
            let path_values = path_array_value
                .as_ref()
                .as_any()
                .downcast_ref::<UInt32Array>()
                .expect("path values should be UInt32Array");
            assert_eq!(
                path_values.len(),
                expected_path.len(),
                "unexpected path length at index {i}"
            );
            for (j, expected_value) in expected_path.iter().enumerate() {
                assert_eq!(
                    path_values.value(j),
                    *expected_value,
                    "unexpected path value at index {i}:{j}"
                );
            }
        }

        let expected_geom_values: Vec<Option<&str>> =
            expected.iter().map(|(_, geom)| *geom).collect();
        let expected_geom_array = create_array(&expected_geom_values, sedona_type);
        assert_array_equal(dumped.column(1), &expected_geom_array);
    }

    fn assert_dump_row_null(result: &ArrayRef, row: usize) {
        let list_array = result
            .as_ref()
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("result should be a ListArray");
        assert!(list_array.is_null(row), "row {row} should be null");
    }
}
