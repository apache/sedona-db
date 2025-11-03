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
    builder::{BinaryBuilder, ListBuilder, StructBuilder, UInt32Builder},
    ListArray,
};
use arrow_schema::{DataType, Field, Fields};
use datafusion_common::error::Result;
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use geo_traits::{
    GeometryCollectionTrait, GeometryTrait, GeometryType, MultiLineStringTrait, MultiPointTrait,
    MultiPolygonTrait,
};
use sedona_common::sedona_internal_err;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_geometry::wkb_factory::WKB_MIN_PROBABLE_BYTES;
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};
use std::sync::Arc;

use crate::executor::WkbExecutor;

/// ST_Dump() scalar UDF
///
/// Native implementation to get all the points of a geometry as MULTIPOINT
pub fn st_dump_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_dump",
        vec![Arc::new(STDump)],
        Volatility::Immutable,
        Some(st_dump_doc()),
    )
}

fn st_dump_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Extracts the components of a geometry.",
        "ST_Dump (geom: Geometry)",
    )
    .with_argument("geom", "geometry: Input geometry")
    .with_sql_example("SELECT ST_Dump(ST_GeomFromWKT('MULTIPOINT (0 1, 2 3, 4 5)'))")
    .build()
}

#[derive(Debug)]
struct STDump;

// This enum is solely for passing the subset of wkb geometry to STDumpStructBuilder.
// Maybe we can pass the underlying raw WKB bytes directly, but this just works for now.
enum SingleWkb<'a> {
    Point(&'a wkb::reader::Point<'a>),
    LineString(&'a wkb::reader::LineString<'a>),
    Polygon(&'a wkb::reader::Polygon<'a>),
}

// A builder for a single struct of { path: [u32], geom: POINT | LINESTRING | POLYGON }
struct STDumpStructBuilder<'a> {
    struct_builder: &'a mut StructBuilder,
}

// A builder for a list of the structs
struct STDumpBuilder {
    builder: ListBuilder<StructBuilder>,
}

impl<'a> STDumpStructBuilder<'a> {
    // This appends both path and geom at once.
    fn append(
        &mut self,
        parent_path: &[u32],
        cur_index: Option<u32>,
        wkb: SingleWkb<'_>,
    ) -> Result<()> {
        let path_builder = self
            .struct_builder
            .field_builder::<ListBuilder<UInt32Builder>>(0)
            .unwrap();

        let path_array_builder = path_builder.values();
        path_array_builder.append_slice(parent_path);
        if let Some(cur_index) = cur_index {
            path_array_builder.append_value(cur_index);
        }
        path_builder.append(true);

        let geom_builder = self
            .struct_builder
            .field_builder::<BinaryBuilder>(1)
            .unwrap();

        let write_result = match wkb {
            SingleWkb::Point(point) => {
                wkb::writer::write_point(geom_builder, &point, &Default::default())
            }
            SingleWkb::LineString(line_string) => {
                wkb::writer::write_line_string(geom_builder, &line_string, &Default::default())
            }
            SingleWkb::Polygon(polygon) => {
                wkb::writer::write_polygon(geom_builder, &polygon, &Default::default())
            }
        };
        if let Err(e) = write_result {
            return sedona_internal_err!("Failed to write WKB: {e}");
        }

        geom_builder.append_value([]);

        self.struct_builder.append(true);

        Ok(())
    }
}

impl STDumpBuilder {
    fn new(num_iter: usize) -> Self {
        let path_builder =
            ListBuilder::with_capacity(UInt32Builder::with_capacity(num_iter), num_iter);
        let geom_builder =
            BinaryBuilder::with_capacity(num_iter, WKB_MIN_PROBABLE_BYTES * num_iter);
        let struct_builder = StructBuilder::new(
            geometry_dump_fields(),
            vec![Box::new(path_builder), Box::new(geom_builder)],
        );
        let builder = ListBuilder::with_capacity(struct_builder, WKB_MIN_PROBABLE_BYTES * num_iter);

        Self { builder }
    }

    fn append(&mut self, is_valid: bool) {
        self.builder.append(is_valid);
    }

    fn append_null(&mut self) {
        self.builder.append_null();
    }

    fn struct_builder<'a>(&'a mut self) -> STDumpStructBuilder<'a> {
        STDumpStructBuilder {
            struct_builder: self.builder.values(),
        }
    }

    fn finish(&mut self) -> ListArray {
        self.builder.finish()
    }
}

impl SedonaScalarKernel for STDump {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(vec![ArgMatcher::is_geometry()], geometry_dump_type());
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);

        let mut builder = STDumpBuilder::new(executor.num_iterations());

        executor.execute_wkb_void(|maybe_wkb| {
            if let Some(wkb) = maybe_wkb {
                let mut struct_builder = builder.struct_builder();

                let mut cur_path: Vec<u32> = Vec::new();
                append_struct(&mut struct_builder, &wkb, &mut cur_path)?;

                builder.append(true);
            } else {
                builder.append_null();
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn append_struct(
    struct_builder: &mut STDumpStructBuilder<'_>,
    wkb: &wkb::reader::Wkb<'_>,
    parent_path: &mut [u32],
) -> Result<()> {
    match wkb.as_type() {
        GeometryType::Point(point) => {
            struct_builder.append(parent_path, None, SingleWkb::Point(point))?;
        }
        GeometryType::LineString(line_string) => {
            struct_builder.append(parent_path, None, SingleWkb::LineString(line_string))?;
        }
        GeometryType::Polygon(polygon) => {
            struct_builder.append(parent_path, None, SingleWkb::Polygon(polygon))?;
        }
        GeometryType::MultiPoint(multi_point) => {
            for (index, point) in multi_point.points().enumerate() {
                struct_builder.append(
                    parent_path,
                    Some((index + 1) as _),
                    SingleWkb::Point(&point),
                )?;
            }
        }
        GeometryType::MultiLineString(multi_line_string) => {
            for (index, line_string) in multi_line_string.line_strings().enumerate() {
                struct_builder.append(
                    parent_path,
                    Some((index + 1) as _),
                    SingleWkb::LineString(line_string),
                )?;
            }
        }
        GeometryType::MultiPolygon(multi_polygon) => {
            for (index, polygon) in multi_polygon.polygons().enumerate() {
                struct_builder.append(
                    parent_path,
                    Some((index + 1) as _),
                    SingleWkb::Polygon(polygon),
                )?;
            }
        }
        GeometryType::GeometryCollection(geometry_collection) => {
            for (index, geometry) in geometry_collection.geometries().enumerate() {
                let mut path = parent_path.to_vec();
                path.push((index + 1) as _);
                append_struct(struct_builder, geometry, &mut path)?;
            }
        }
        _ => return sedona_internal_err!("Invalid geometry type"),
    }

    Ok(())
}

fn geometry_dump_fields() -> Fields {
    let path = Field::new(
        "path",
        DataType::List(Field::new("item", DataType::UInt32, true).into()),
        true,
    );
    let geom = WKB_GEOMETRY.to_storage_field("geom", true).unwrap();
    vec![path, geom].into()
}

fn geometry_dump_type() -> SedonaType {
    let fields = geometry_dump_fields();
    let struct_type = DataType::Struct(fields);

    SedonaType::Arrow(DataType::List(Field::new("item", struct_type, true).into()))
}

#[cfg(test)]
mod tests {
    use arrow_array::{Array, ArrayRef, ListArray, StructArray, UInt32Array};
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::WKB_VIEW_GEOMETRY;
    use sedona_testing::{
        compare::assert_array_equal, create::create_array, testers::ScalarUdfTester,
    };

    use super::*;

    #[test]
    fn udf_metadata() {
        let st_dump_udf: ScalarUDF = st_dump_udf().into();
        assert_eq!(st_dump_udf.name(), "st_dump");
        assert!(st_dump_udf.documentation().is_some());
    }

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
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
        assert_dump_row(&result, 0, &[(&[], Some("POINT (1 2)"))]);
        assert_dump_row(&result, 1, &[(&[], Some("LINESTRING (1 1, 2 2)"))]);
        assert_dump_row(&result, 2, &[(&[], Some("POLYGON ((1 1, 2 2, 2 1, 1 1))"))]);
        assert_dump_row(
            &result,
            3,
            &[(&[1], Some("POINT (1 1)")), (&[2], Some("POINT (2 2)"))],
        );
        assert_dump_row(
            &result,
            4,
            &[
                (&[1], Some("LINESTRING (1 1, 2 2)")),
                (&[2], Some("LINESTRING EMPTY")),
                (&[3], Some("LINESTRING (3 3, 4 4)")),
            ],
        );
        assert_dump_row(
            &result,
            5,
            &[
                (&[1], Some("POLYGON ((1 1, 2 2, 2 1, 1 1))")),
                (&[2], Some("POLYGON EMPTY")),
                (&[3], Some("POLYGON ((3 3, 4 4, 4 3, 3 3)))")),
            ],
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
        );

        let null_input = create_array(&[None], &sedona_type);
        let result = tester.invoke_array(null_input).unwrap();
        assert_dump_row_null(&result, 0);
    }

    fn assert_dump_row(result: &ArrayRef, row: usize, expected: &[(&[u32], Option<&str>)]) {
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
        let expected_geom_array = create_array(&expected_geom_values, &WKB_GEOMETRY);
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
