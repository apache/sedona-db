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
    Array, ListArray,
    builder::{BinaryBuilder, NullBufferBuilder, OffsetBufferBuilder},
};
use arrow_schema::{DataType, FieldRef};
use datafusion_common::{Result, config::ConfigOptions};
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
use std::sync::Arc;

use crate::executor::WkbExecutor;

/// ST_Geometries() scalar UDF
///
/// Lists a geometry's direct members: the parts of a multi-geometry or the
/// members of a geometry collection, one level deep (a nested collection is
/// returned whole, unlike ST_Dump, which recurses), or the geometry itself
/// when it is not a collection.
pub fn st_geometries_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_geometries",
        vec![
            Arc::new(STGeometries {
                matcher: ArgMatcher::new(vec![ArgMatcher::is_geometry()], WKB_GEOMETRY),
            }),
            Arc::new(STGeometries {
                matcher: ArgMatcher::new(vec![ArgMatcher::is_geography()], WKB_GEOGRAPHY),
            }),
        ],
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct STGeometries {
    matcher: ArgMatcher,
}

/// Builds a list of WKB members per input row
struct STGeometriesBuilder {
    geom_builder: BinaryBuilder,
    offsets_builder: OffsetBufferBuilder<i32>,
    null_builder: NullBufferBuilder,
    item_field: FieldRef,
}

impl STGeometriesBuilder {
    fn new(num_iter: usize, data_capacity: usize, return_type: &SedonaType) -> Result<Self> {
        let SedonaType::Arrow(DataType::List(item_field)) = return_type else {
            return sedona_internal_err!("Unexpected return type in st_geometries()");
        };

        Ok(Self {
            geom_builder: BinaryBuilder::with_capacity(num_iter, data_capacity),
            offsets_builder: OffsetBufferBuilder::new(num_iter),
            null_builder: NullBufferBuilder::new(num_iter),
            item_field: item_field.clone(),
        })
    }

    fn append_member(&mut self, wkb: &[u8]) -> usize {
        self.geom_builder.append_value(wkb);
        1
    }

    fn append(&mut self, wkb: &wkb::reader::Wkb<'_>) -> Result<()> {
        // Every member of a multi-geometry or collection is itself a complete
        // WKB geometry, so its bytes are copied as they are.
        let num_members = match wkb.as_type() {
            GeometryType::Point(_) | GeometryType::LineString(_) | GeometryType::Polygon(_) => {
                self.append_member(wkb.buf())
            }
            GeometryType::MultiPoint(multi_point) => multi_point
                .points()
                .map(|point| self.append_member(point.buf()))
                .sum(),
            GeometryType::MultiLineString(multi_line_string) => multi_line_string
                .line_strings()
                .map(|line_string| self.append_member(line_string.buf()))
                .sum(),
            GeometryType::MultiPolygon(multi_polygon) => multi_polygon
                .polygons()
                .map(|polygon| self.append_member(polygon.buf()))
                .sum(),
            GeometryType::GeometryCollection(geometry_collection) => geometry_collection
                .geometries()
                .map(|geometry| self.append_member(geometry.buf()))
                .sum(),
            _ => return sedona_internal_err!("Invalid geometry type"),
        };

        self.offsets_builder.push_length(num_members);
        self.null_builder.append(true);
        Ok(())
    }

    fn append_null(&mut self) {
        self.offsets_builder.push_length(0);
        self.null_builder.append(false);
    }

    fn finish(mut self) -> ListArray {
        ListArray::new(
            self.item_field,
            self.offsets_builder.finish(),
            Arc::new(self.geom_builder.finish()),
            self.null_builder.finish(),
        )
    }
}

impl SedonaScalarKernel for STGeometries {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        // The list items have the input's spatial kind and CRS.
        self.matcher
            .match_args(args)?
            .map(|item_type| {
                let item_field = item_type.to_storage_field("item", true)?;
                Ok(SedonaType::Arrow(DataType::List(item_field.into())))
            })
            .transpose()
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
        let num_iter = executor.num_iterations();

        // The members are slices of the input's WKB, so the input's size is a
        // close estimate of theirs.
        let data_capacity = match &args[0] {
            ColumnarValue::Array(array) => array.to_data().get_slice_memory_size().ok(),
            ColumnarValue::Scalar(_) => None,
        }
        .unwrap_or(WKB_MIN_PROBABLE_BYTES * num_iter);

        let mut builder = STGeometriesBuilder::new(num_iter, data_capacity, return_type)?;
        executor.execute_wkb_void(|maybe_wkb| {
            if let Some(wkb) = maybe_wkb {
                builder.append(wkb)?;
            } else {
                builder.append_null();
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }

    fn invoke_batch(
        &self,
        _arg_types: &[SedonaType],
        _args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        sedona_internal_err!("invoke_batch() should not be called for st_geometries()")
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{Array, ArrayRef};
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_geometry::types::Edges;
    use sedona_schema::crs::lnglat;
    use sedona_schema::datatypes::{WKB_VIEW_GEOGRAPHY, WKB_VIEW_GEOMETRY};
    use sedona_testing::{
        compare::assert_array_equal, create::create_array, testers::ScalarUdfTester,
    };

    use super::*;

    #[test]
    fn udf_metadata() {
        let st_geometries_udf: ScalarUDF = st_geometries_udf().into();
        assert_eq!(st_geometries_udf.name(), "st_geometries");
        assert!(st_geometries_udf.documentation().is_none());
    }

    #[rstest]
    fn return_type_keeps_kind_and_crs(
        #[values(
            WKB_GEOMETRY,
            WKB_GEOGRAPHY,
            WKB_VIEW_GEOMETRY,
            WKB_VIEW_GEOGRAPHY,
            SedonaType::Wkb(Edges::Planar, lnglat())
        )]
        sedona_type: SedonaType,
    ) {
        let tester = ScalarUdfTester::new(st_geometries_udf().into(), vec![sedona_type.clone()]);
        let expected_item = output_item_type(&sedona_type)
            .to_storage_field("item", true)
            .unwrap();
        assert_eq!(
            tester.return_type().unwrap(),
            SedonaType::Arrow(DataType::List(expected_item.into()))
        );
    }

    #[rstest]
    fn udf(
        #[values(
            WKB_GEOMETRY,
            WKB_GEOGRAPHY,
            WKB_VIEW_GEOMETRY,
            WKB_VIEW_GEOGRAPHY,
            SedonaType::Wkb(Edges::Planar, lnglat())
        )]
        sedona_type: SedonaType,
    ) {
        let tester = ScalarUdfTester::new(st_geometries_udf().into(), vec![sedona_type.clone()]);

        let cases: Vec<(Option<&str>, Option<Vec<&str>>)> = vec![
            (Some("POINT (1 2)"), Some(vec!["POINT (1 2)"])),
            (Some("POINT EMPTY"), Some(vec!["POINT EMPTY"])),
            (
                Some("LINESTRING (1 1, 2 2)"),
                Some(vec!["LINESTRING (1 1, 2 2)"]),
            ),
            (
                Some("POLYGON ((1 1, 2 2, 2 1, 1 1))"),
                Some(vec!["POLYGON ((1 1, 2 2, 2 1, 1 1))"]),
            ),
            (
                Some("MULTIPOINT (0 1, 1 2)"),
                Some(vec!["POINT (0 1)", "POINT (1 2)"]),
            ),
            (Some("MULTIPOINT EMPTY"), Some(vec![])),
            (
                Some("MULTILINESTRING ((1 1, 2 2), EMPTY, (3 3, 4 4))"),
                Some(vec![
                    "LINESTRING (1 1, 2 2)",
                    "LINESTRING EMPTY",
                    "LINESTRING (3 3, 4 4)",
                ]),
            ),
            (
                Some("MULTIPOLYGON (((1 1, 2 2, 2 1, 1 1)), EMPTY, ((3 3, 4 4, 4 3, 3 3)))"),
                Some(vec![
                    "POLYGON ((1 1, 2 2, 2 1, 1 1))",
                    "POLYGON EMPTY",
                    "POLYGON ((3 3, 4 4, 4 3, 3 3))",
                ]),
            ),
            // A nested collection is a member, not flattened.
            (
                Some(
                    "GEOMETRYCOLLECTION (POINT (1 2), GEOMETRYCOLLECTION (MULTILINESTRING ((1 1, 2 2), EMPTY, (3 3, 4 4)), LINESTRING (1 1, 2 2)))",
                ),
                Some(vec![
                    "POINT (1 2)",
                    "GEOMETRYCOLLECTION (MULTILINESTRING ((1 1, 2 2), EMPTY, (3 3, 4 4)), LINESTRING (1 1, 2 2))",
                ]),
            ),
            (Some("GEOMETRYCOLLECTION EMPTY"), Some(vec![])),
            (
                Some("MULTIPOINT Z ((0 1 2), (1 2 3))"),
                Some(vec!["POINT Z (0 1 2)", "POINT Z (1 2 3)"]),
            ),
            (None, None),
        ];

        let input = create_array(
            &cases.iter().map(|(wkt, _)| *wkt).collect::<Vec<_>>(),
            &sedona_type,
        );
        let result = tester.invoke_array(input).unwrap();

        let item_type = output_item_type(&sedona_type);
        for (row, (_, expected)) in cases.iter().enumerate() {
            assert_members(&result, row, expected.as_deref(), &item_type);
        }
    }

    /// The members are WKB with the input's edges and CRS, whatever the
    /// input's storage.
    fn output_item_type(input: &SedonaType) -> SedonaType {
        match input {
            SedonaType::WkbView(edges, crs) => SedonaType::Wkb(*edges, crs.clone()),
            other => other.clone(),
        }
    }

    fn assert_members(
        result: &ArrayRef,
        row: usize,
        expected: Option<&[&str]>,
        sedona_type: &SedonaType,
    ) {
        let list_array = result
            .as_ref()
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("result should be a ListArray");
        let Some(expected) = expected else {
            assert!(list_array.is_null(row), "row {row} should be null");
            return;
        };
        assert!(!list_array.is_null(row), "row {row} should not be null");
        let expected: Vec<Option<&str>> = expected.iter().map(|wkt| Some(*wkt)).collect();
        assert_array_equal(
            &list_array.value(row),
            &create_array(&expected, sedona_type),
        );
    }
}
