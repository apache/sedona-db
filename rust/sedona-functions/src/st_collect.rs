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
use std::{sync::Arc, vec};

use crate::executor::WkbExecutor;
use arrow_array::ArrayRef;
use arrow_schema::{DataType, Field, FieldRef};
use datafusion_common::{
    cast::{as_binary_array, as_int64_array, as_string_array},
    error::{DataFusionError, Result},
    exec_err, HashSet, ScalarValue,
};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, Accumulator, ColumnarValue, Documentation, Volatility,
};
use geo_traits::Dimensions;
use sedona_common::sedona_internal_err;
use sedona_expr::aggregate_udf::{SedonaAccumulator, SedonaAggregateUDF};
use sedona_geometry::{
    types::{GeometryTypeAndDimensions, GeometryTypeId},
    wkb_factory::{
        write_wkb_geometrycollection_header, write_wkb_multilinestring_header,
        write_wkb_multipoint_header, write_wkb_multipolygon_header,
    },
};
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};

/// ST_Collect() aggregate UDF implementation
///
/// An implementation of envelope (bounding shape) calculation.
pub fn st_collect_udf() -> SedonaAggregateUDF {
    SedonaAggregateUDF::new(
        "st_collect",
        vec![Arc::new(STCollectAggr {})],
        Volatility::Immutable,
        Some(st_collect_doc()),
    )
}

fn st_collect_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Return the entire envelope boundary of all geometries in geom",
        "ST_Collect (geom: Geometry)",
    )
    .with_argument("geom", "geometry: Input geometry or geography")
    .with_sql_example("SELECT ST_Collect(ST_GeomFromWKT('MULTIPOINT (0 1, 10 11)'))")
    .build()
}

#[derive(Debug)]
struct STCollectAggr {}

impl SedonaAccumulator for STCollectAggr {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(vec![ArgMatcher::is_geometry_or_geography()], WKB_GEOMETRY);
        matcher.match_args(args)
    }

    fn accumulator(
        &self,
        args: &[SedonaType],
        output_type: &SedonaType,
    ) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(CollectionAccumulator::try_new(
            args[0].clone(),
            output_type.clone(),
        )?))
    }

    fn state_fields(&self, _args: &[SedonaType]) -> Result<Vec<FieldRef>> {
        Ok(vec![
            Arc::new(Field::new("unique_geometry_types", DataType::Utf8, false)),
            Arc::new(Field::new("unique_dimensions", DataType::Utf8, false)),
            Arc::new(Field::new("count", DataType::Int64, false)),
            Arc::new(WKB_GEOMETRY.to_storage_field("item", true)?),
        ])
    }
}

#[derive(Debug)]
struct CollectionAccumulator {
    input_type: SedonaType,
    unique_geometry_types: HashSet<GeometryTypeId>,
    unique_dimensions: HashSet<Dimensions>,
    count: i64,
    item: Option<Vec<u8>>,
}

const WKB_HEADER_SIZE: usize = 1 + 4 + 4;

impl CollectionAccumulator {
    pub fn try_new(input_type: SedonaType, _output_type: SedonaType) -> Result<Self> {
        // Write a dummy header with the correct number of bytes. We'll rewrite this later
        // when we know what type/dimension of geometrycollection we have based on the
        // items encountered.
        let mut item = Vec::new();
        write_wkb_geometrycollection_header(&mut item, Dimensions::Xy, 0)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        Ok(Self {
            input_type,
            unique_geometry_types: HashSet::new(),
            unique_dimensions: HashSet::new(),
            count: 0,
            item: Some(item),
        })
    }

    // Create a WKB result based on the current state of the accumulator.
    fn make_wkb_result(&mut self) -> Result<Option<Vec<u8>>> {
        if self.count == 0 {
            return Ok(None);
        }

        // Generate the correct header: collections of points become multipoint, ensure
        // dimensions are preserved if possible.
        let mut new_header = Vec::new();
        let count_usize = self.count.try_into().unwrap();

        if self.unique_dimensions.len() != 1 {
            return exec_err!("Can't ST_Collect() mixed dimension geometries");
        }

        let dimensions = *self.unique_dimensions.iter().next().unwrap();
        if self.unique_geometry_types.len() == 1 {
            match self.unique_geometry_types.iter().next().unwrap() {
                GeometryTypeId::Point => {
                    write_wkb_multipoint_header(&mut new_header, dimensions, count_usize)
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;
                }
                GeometryTypeId::LineString => {
                    write_wkb_multilinestring_header(&mut new_header, dimensions, count_usize)
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;
                }
                GeometryTypeId::Polygon => {
                    write_wkb_multipolygon_header(&mut new_header, dimensions, count_usize)
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;
                }
                _ => {
                    write_wkb_geometrycollection_header(&mut new_header, dimensions, count_usize)
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;
                }
            }
        } else {
            write_wkb_geometrycollection_header(&mut new_header, dimensions, count_usize)
                .map_err(|e| DataFusionError::External(Box::new(e)))?;
        }

        // Update the header bytes of the output and return it
        if let Some(mut out) = self.item.take() {
            out[0..WKB_HEADER_SIZE].copy_from_slice(&new_header);
            Ok(Some(out))
        } else {
            sedona_internal_err!("Unexpected internal state in ST_Collect()")
        }
    }
}

impl Accumulator for CollectionAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let item_ref = if let Some(item_ref) = self.item.as_mut() {
            item_ref
        } else {
            return sedona_internal_err!("Unexpected internal state in ST_Collect()");
        };

        let arg_types = [self.input_type.clone()];
        let args = [ColumnarValue::Array(values[0].clone())];
        let executor = WkbExecutor::new(&arg_types, &args);
        executor.execute_wkb_void(|maybe_item| {
            if let Some(item) = maybe_item {
                let type_and_dims = GeometryTypeAndDimensions::try_from_geom(&item)
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
                self.unique_geometry_types
                    .insert(type_and_dims.geometry_type());
                self.unique_dimensions.insert(type_and_dims.dimensions());
                self.count += 1;
                item_ref.extend_from_slice(item.buf());
            }
            Ok(())
        })?;
        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let wkb = self.make_wkb_result()?;
        Ok(ScalarValue::Binary(wkb))
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let geometry_types_value =
            serde_json::to_string(&self.unique_geometry_types.iter().collect::<Vec<_>>())
                .map_err(|e| DataFusionError::External(Box::new(e)))?;
        let dimensions_value = serde_json::to_string(
            &self
                .unique_dimensions
                .iter()
                .map(|dim| GeometryTypeAndDimensions::new(GeometryTypeId::Geometry, *dim))
                .collect::<Vec<_>>(),
        )
        .map_err(|e| DataFusionError::External(Box::new(e)))?;

        let serialized_geometry_types = ScalarValue::Utf8(Some(geometry_types_value));
        let serialized_dimensions = ScalarValue::Utf8(Some(dimensions_value));
        let serialized_count = ScalarValue::Int64(Some(self.count));
        let serialized_item = ScalarValue::Binary(self.item.take());

        Ok(vec![
            serialized_geometry_types,
            serialized_dimensions,
            serialized_count,
            serialized_item,
        ])
    }

    fn size(&self) -> usize {
        let item_capacity = self.item.as_ref().map(|e| e.capacity()).unwrap_or(0);
        size_of::<CollectionAccumulator>() + item_capacity
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        if states.len() != 4 {
            return sedona_internal_err!(
                "Unexpected number of state fields for st_collect() (expected 4, got {})",
                states.len()
            );
        }

        let item_ref = if let Some(item_ref) = self.item.as_mut() {
            item_ref
        } else {
            return sedona_internal_err!("Unexpected internal state in ST_Collect()");
        };

        let mut geometry_types_iter = as_string_array(&states[0])?.into_iter();
        let mut dimensions_iter = as_string_array(&states[1])?.into_iter();
        let mut count_iter = as_int64_array(&states[2])?.into_iter();
        let mut item_iter = as_binary_array(&states[3])?.into_iter();

        for _ in 0..geometry_types_iter.len() {
            match (
                geometry_types_iter.next(),
                dimensions_iter.next(),
                count_iter.next(),
                item_iter.next(),
            ) {
                (
                    Some(Some(serialized_geometry_types)),
                    Some(Some(serialized_dimensions)),
                    Some(Some(count)),
                    Some(Some(item)),
                ) => {
                    let geometry_types =
                        serde_json::from_str::<Vec<GeometryTypeId>>(serialized_geometry_types)
                            .map_err(|e| DataFusionError::External(Box::new(e)))?;
                    let dimensions = serde_json::from_str::<Vec<GeometryTypeAndDimensions>>(
                        serialized_dimensions,
                    )
                    .map_err(|e| DataFusionError::External(Box::new(e)))?
                    .into_iter()
                    .map(|item| item.dimensions());

                    self.unique_geometry_types.extend(geometry_types);
                    self.unique_dimensions.extend(dimensions);
                    self.count += count;
                    item_ref.extend_from_slice(&item[WKB_HEADER_SIZE..item.len()]);
                }
                _ => {
                    return sedona_internal_err!(
                        "unexpected nulls in st_collect() serialized state"
                    )
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use datafusion_expr::AggregateUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::WKB_VIEW_GEOMETRY;
    use sedona_testing::{compare::assert_scalar_equal_wkb_geometry, testers::AggregateUdfTester};

    use super::*;

    #[test]
    fn udf_metadata() {
        let udf: AggregateUDF = st_collect_udf().into();
        assert_eq!(udf.name(), "st_collect");
        assert!(udf.documentation().is_some());
    }

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester = AggregateUdfTester::new(st_collect_udf().into(), vec![sedona_type.clone()]);
        assert_eq!(tester.return_type().unwrap(), WKB_GEOMETRY);

        // Finite point input with nulls
        let batches = vec![
            vec![Some("POINT (0 1)"), None, Some("POINT (2 3)")],
            vec![Some("POINT (4 5)"), None, Some("POINT (6 7)")],
        ];
        assert_scalar_equal_wkb_geometry(
            &tester.aggregate_wkt(batches).unwrap(),
            Some("MULTIPOINT (0 1, 2 3, 4 5, 6 7)"),
        );

        // Finite linestring input with nulls
        let batches = vec![
            vec![Some("LINESTRING (0 1, 2 3)"), None],
            vec![Some("LINESTRING (4 5, 6 7)"), None],
        ];
        assert_scalar_equal_wkb_geometry(
            &tester.aggregate_wkt(batches).unwrap(),
            Some("MULTILINESTRING ((0 1, 2 3), (4 5, 6 7))"),
        );

        // Finite polygon input with nulls
        let batches = vec![
            vec![Some("POLYGON ((0 0, 1 0, 0 1, 0 0))"), None],
            vec![Some("POLYGON ((10 10, 11 10, 10 11, 10 10))"), None],
        ];
        assert_scalar_equal_wkb_geometry(
            &tester.aggregate_wkt(batches).unwrap(),
            Some("MULTIPOLYGON (((0 0, 1 0, 0 1, 0 0)), ((10 10, 11 10, 10 11, 10 10)))"),
        );

        // Mixed input
        let batches = vec![
            vec![Some("POINT (0 1)"), None],
            vec![Some("LINESTRING (4 5, 6 7)"), None],
        ];
        assert_scalar_equal_wkb_geometry(
            &tester.aggregate_wkt(batches).unwrap(),
            Some("GEOMETRYCOLLECTION (POINT (0 1), LINESTRING (4 5, 6 7))"),
        );

        // Empty input
        assert_scalar_equal_wkb_geometry(&tester.aggregate_wkt(vec![]).unwrap(), None);

        // Error for mixed dimensions
        let batches = vec![
            vec![Some("POINT (0 1)"), None],
            vec![Some("POINT Z (0 1 2)"), None],
        ];
        let err = tester.aggregate_wkt(batches).unwrap_err();
        assert_eq!(
            err.message(),
            "Can't ST_Collect() mixed dimension geometries"
        );
    }
}
