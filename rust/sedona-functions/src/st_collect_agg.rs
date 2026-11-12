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
use std::{io::Write, sync::Arc, vec};

use crate::executor::{WkbBytesExecutor, WkbExecutor};
use arrow_array::{Array, ArrayRef, BooleanArray, Int64Array, UInt32Array, builder::BinaryBuilder};
use arrow_schema::{DataType, Field, FieldRef};
use datafusion_common::{
    ScalarValue,
    cast::{as_binary_array, as_int64_array, as_uint32_array},
    error::{DataFusionError, Result},
    exec_err,
};
use datafusion_expr::{Accumulator, ColumnarValue, EmitTo, GroupsAccumulator, Volatility};
use geo_traits::Dimensions;
use sedona_common::sedona_internal_err;
use sedona_expr::{
    aggregate_udf::{SedonaAccumulator, SedonaAggregateUDF},
    item_crs::ItemCrsSedonaAccumulator,
};
use sedona_geometry::{
    types::{GeometryTypeAndDimensions, GeometryTypeAndDimensionsSet, GeometryTypeId},
    wkb_factory::{
        write_wkb_geometrycollection_header, write_wkb_multilinestring_header,
        write_wkb_multipoint_header, write_wkb_multipolygon_header,
    },
};
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOGRAPHY, WKB_GEOMETRY},
    matchers::ArgMatcher,
};
use wkb::reader::{Wkb, read_wkb};

/// ST_Collect_Agg() aggregate UDF implementation
///
/// An implementation of envelope (bounding shape) calculation.
pub fn st_collect_agg_udf() -> SedonaAggregateUDF {
    SedonaAggregateUDF::new(
        "st_collect_agg",
        ItemCrsSedonaAccumulator::wrap_impl(vec![
            Arc::new(STCollectAggr {
                is_geography: false,
            }),
            Arc::new(STCollectAggr { is_geography: true }),
        ]),
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct STCollectAggr {
    is_geography: bool,
}

impl SedonaAccumulator for STCollectAggr {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = match self.is_geography {
            true => ArgMatcher::new(vec![ArgMatcher::is_geography()], WKB_GEOGRAPHY),
            false => ArgMatcher::new(vec![ArgMatcher::is_geometry()], WKB_GEOMETRY),
        };
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

    fn groups_accumulator_supported(&self, _args: &[SedonaType]) -> bool {
        true
    }

    fn groups_accumulator(
        &self,
        args: &[SedonaType],
        _output_type: &SedonaType,
    ) -> Result<Box<dyn GroupsAccumulator>> {
        Ok(Box::new(CollectionGroupsAccumulator::new(args[0].clone())))
    }

    fn state_fields(&self, _args: &[SedonaType]) -> Result<Vec<FieldRef>> {
        Ok(vec![
            Arc::new(Field::new("types_and_dims", DataType::UInt32, false)),
            Arc::new(Field::new("count", DataType::Int64, false)),
            Arc::new(WKB_GEOMETRY.to_storage_field("item", true)?),
        ])
    }
}

#[derive(Debug)]
struct CollectionAccumulator {
    input_type: SedonaType,
    types_and_dims: GeometryTypeAndDimensionsSet,
    count: i64,
    item: Option<Vec<u8>>,
}

const WKB_HEADER_SIZE: usize = 1 + 4 + 4;

/// A placeholder collection header of the right size, written ahead of the items
/// until their types and dimensions are known
fn dummy_collection_header() -> Result<Vec<u8>> {
    let mut header = Vec::with_capacity(WKB_HEADER_SIZE);
    write_wkb_geometrycollection_header(&mut header, Dimensions::Xy, 0)
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
    Ok(header)
}

/// Write the header of a collection of `count` items with the given types and dimensions
///
/// Collections of points become multipoints (and likewise for linestrings and polygons).
/// The items must share their dimensions.
fn write_collection_header(
    buf: &mut impl Write,
    types_and_dims: &GeometryTypeAndDimensionsSet,
    count: i64,
) -> Result<()> {
    let count_usize = count.try_into().unwrap();

    let unique_dimensions = types_and_dims.dimensions();
    if unique_dimensions.len() != 1 {
        return exec_err!("Can't ST_Collect_Agg() mixed dimension geometries");
    }

    let dimensions = unique_dimensions[0];
    let unique_geometry_types = types_and_dims.geometry_types();
    let written = if unique_geometry_types.len() == 1 {
        match unique_geometry_types[0] {
            GeometryTypeId::Point => write_wkb_multipoint_header(buf, dimensions, count_usize),
            GeometryTypeId::LineString => {
                write_wkb_multilinestring_header(buf, dimensions, count_usize)
            }
            GeometryTypeId::Polygon => write_wkb_multipolygon_header(buf, dimensions, count_usize),
            _ => write_wkb_geometrycollection_header(buf, dimensions, count_usize),
        }
    } else {
        write_wkb_geometrycollection_header(buf, dimensions, count_usize)
    };
    written.map_err(|e| DataFusionError::External(Box::new(e)))
}

/// The types and dimensions of one item, which must have standard dimensions
fn item_types_and_dims(item: &Wkb) -> Result<GeometryTypeAndDimensions> {
    GeometryTypeAndDimensions::try_from_geom(item)
        .map_err(|e| DataFusionError::External(Box::new(e)))
}

impl CollectionAccumulator {
    pub fn try_new(input_type: SedonaType, _output_type: SedonaType) -> Result<Self> {
        // Write a dummy header with the correct number of bytes. We'll rewrite this later
        // when we know what type/dimension of geometrycollection we have based on the
        // items encountered.
        let item = dummy_collection_header()?;

        Ok(Self {
            input_type,
            types_and_dims: GeometryTypeAndDimensionsSet::new(),
            count: 0,
            item: Some(item),
        })
    }

    // Create a WKB result based on the current state of the accumulator.
    fn make_wkb_result(&mut self) -> Result<Option<Vec<u8>>> {
        if self.count == 0 {
            return Ok(None);
        }

        let mut new_header = Vec::new();
        write_collection_header(&mut new_header, &self.types_and_dims, self.count)?;

        // Update the header bytes of the output and return it
        if let Some(mut out) = self.item.take() {
            out[0..WKB_HEADER_SIZE].copy_from_slice(&new_header);
            Ok(Some(out))
        } else {
            sedona_internal_err!("Unexpected internal state in ST_Collect_Agg()")
        }
    }
}

impl Accumulator for CollectionAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let item_ref = if let Some(item_ref) = self.item.as_mut() {
            item_ref
        } else {
            return sedona_internal_err!("Unexpected internal state in ST_Collect_Agg()");
        };

        let arg_types = [self.input_type.clone()];
        let args = [ColumnarValue::Array(values[0].clone())];
        let executor = WkbExecutor::new(&arg_types, &args);
        executor.execute_wkb_void(|maybe_item| {
            if let Some(item) = maybe_item {
                let types_and_dims = item_types_and_dims(item)?;
                // insert() (not insert_or_ignore()) so geometries with
                // unknown dimensions fail loudly instead of being silently
                // dropped from the dimension check in make_wkb_result().
                self.types_and_dims
                    .insert(&types_and_dims)
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
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
        Ok(vec![
            ScalarValue::UInt32(Some(self.types_and_dims.bits())),
            ScalarValue::Int64(Some(self.count)),
            ScalarValue::Binary(self.item.take()),
        ])
    }

    fn size(&self) -> usize {
        let item_capacity = self.item.as_ref().map(|e| e.capacity()).unwrap_or(0);
        size_of::<CollectionAccumulator>() + item_capacity
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        if states.len() != 3 {
            return sedona_internal_err!(
                "Unexpected number of state fields for st_collect() (expected 3, got {})",
                states.len()
            );
        }

        let item_ref = if let Some(item_ref) = self.item.as_mut() {
            item_ref
        } else {
            return sedona_internal_err!("Unexpected internal state in ST_Collect_Agg()");
        };

        let mut bits_iter = as_uint32_array(&states[0])?.into_iter();
        let mut count_iter = as_int64_array(&states[1])?.into_iter();
        let mut item_iter = as_binary_array(&states[2])?.into_iter();

        for _ in 0..bits_iter.len() {
            match (bits_iter.next(), count_iter.next(), item_iter.next()) {
                (Some(Some(bits)), Some(Some(count)), Some(Some(item))) => {
                    self.types_and_dims
                        .merge(&GeometryTypeAndDimensionsSet::from_bits(bits));
                    self.count += count;
                    item_ref.extend_from_slice(&item[WKB_HEADER_SIZE..item.len()]);
                }
                _ => {
                    return sedona_internal_err!(
                        "unexpected nulls in st_collect() serialized state"
                    );
                }
            }
        }

        Ok(())
    }
}

/// Whether row `i` passes the optional filter of a [GroupsAccumulator] call
fn is_selected(opt_filter: Option<&BooleanArray>, i: usize) -> bool {
    opt_filter.is_none_or(|filter| filter.is_valid(i) && filter.value(i))
}

/// Parse an item as [WkbExecutor] does
fn parse_item(bytes: &[u8]) -> Result<Wkb<'_>> {
    read_wkb(bytes).map_err(|e| DataFusionError::External(Box::new(e)))
}

/// [GroupsAccumulator] for ST_Collect_Agg()
///
/// Each group keeps the WKB of its items without a collection header, along with
/// their types and dimensions and their count. The serialized state is the same as
/// that of [CollectionAccumulator].
#[derive(Debug)]
struct CollectionGroupsAccumulator {
    input_type: SedonaType,
    types_and_dims: Vec<GeometryTypeAndDimensionsSet>,
    counts: Vec<i64>,
    items: Vec<Vec<u8>>,
    /// The sum of the capacities of `items`, kept so that size() doesn't visit every group
    items_capacity: usize,
}

impl CollectionGroupsAccumulator {
    fn new(input_type: SedonaType) -> Self {
        Self {
            input_type,
            types_and_dims: Vec::new(),
            counts: Vec::new(),
            items: Vec::new(),
            items_capacity: 0,
        }
    }

    fn resize(&mut self, total_num_groups: usize) {
        self.types_and_dims
            .resize_with(total_num_groups, GeometryTypeAndDimensionsSet::new);
        self.counts.resize(total_num_groups, 0);
        self.items.resize_with(total_num_groups, Vec::new);
    }

    fn append_wkb(&mut self, group_index: usize, wkb: &[u8]) {
        let item = &mut self.items[group_index];
        let capacity_before = item.capacity();
        item.extend_from_slice(wkb);
        self.items_capacity += item.capacity() - capacity_before;
    }

    /// Remove the groups to emit, shifting the rest down
    fn take(
        &mut self,
        emit_to: EmitTo,
    ) -> (Vec<GeometryTypeAndDimensionsSet>, Vec<i64>, Vec<Vec<u8>>) {
        let items = emit_to.take_needed(&mut self.items);
        self.items_capacity -= items.iter().map(|item| item.capacity()).sum::<usize>();
        (
            emit_to.take_needed(&mut self.types_and_dims),
            emit_to.take_needed(&mut self.counts),
            items,
        )
    }
}

impl GroupsAccumulator for CollectionGroupsAccumulator {
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        self.resize(total_num_groups);

        let arg_types = [self.input_type.clone()];
        let args = [ColumnarValue::Array(values[0].clone())];
        // Iterate over the bytes so that rows the filter leaves out aren't parsed
        let executor = WkbBytesExecutor::new(&arg_types, &args);
        let mut i = 0;
        executor.execute_wkb_void(|maybe_bytes| {
            if let Some(bytes) = maybe_bytes
                && is_selected(opt_filter, i)
            {
                let item = parse_item(bytes)?;
                let group_index = group_indices[i];
                // insert() (not insert_or_ignore()), as in CollectionAccumulator
                self.types_and_dims[group_index]
                    .insert(&item_types_and_dims(&item)?)
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
                self.counts[group_index] += 1;
                self.append_wkb(group_index, item.buf());
            }
            i += 1;
            Ok(())
        })?;
        Ok(())
    }

    fn evaluate(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        let (types_and_dims, counts, items) = self.take(emit_to);
        let data_capacity = items.iter().map(|item| WKB_HEADER_SIZE + item.len()).sum();
        let mut builder = BinaryBuilder::with_capacity(items.len(), data_capacity);
        for ((types_and_dims, count), item) in types_and_dims.iter().zip(counts).zip(items) {
            if count == 0 {
                builder.append_null();
                continue;
            }
            write_collection_header(&mut builder, types_and_dims, count)?;
            builder.write_all(&item)?;
            builder.append_value([]);
        }
        Ok(Arc::new(builder.finish()))
    }

    fn state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        let (types_and_dims, counts, items) = self.take(emit_to);
        let header = dummy_collection_header()?;
        let data_capacity = items.iter().map(|item| header.len() + item.len()).sum();
        let mut builder = BinaryBuilder::with_capacity(items.len(), data_capacity);
        for item in items {
            builder.write_all(&header)?;
            builder.write_all(&item)?;
            builder.append_value([]);
        }
        Ok(vec![
            Arc::new(UInt32Array::from_iter_values(
                types_and_dims
                    .iter()
                    .map(|types_and_dims| types_and_dims.bits()),
            )),
            Arc::new(Int64Array::from(counts)),
            Arc::new(builder.finish()),
        ])
    }

    fn merge_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        total_num_groups: usize,
    ) -> Result<()> {
        if values.len() != 3 {
            return sedona_internal_err!(
                "Unexpected number of state fields for st_collect() (expected 3, got {})",
                values.len()
            );
        }
        self.resize(total_num_groups);

        let bits = as_uint32_array(&values[0])?;
        let counts = as_int64_array(&values[1])?;
        let items = as_binary_array(&values[2])?;
        for (i, &group_index) in group_indices.iter().enumerate() {
            if bits.is_null(i) || counts.is_null(i) || items.is_null(i) {
                return sedona_internal_err!("unexpected nulls in st_collect() serialized state");
            }
            self.types_and_dims[group_index]
                .merge(&GeometryTypeAndDimensionsSet::from_bits(bits.value(i)));
            self.counts[group_index] += counts.value(i);
            self.append_wkb(group_index, &items.value(i)[WKB_HEADER_SIZE..]);
        }

        Ok(())
    }

    fn convert_to_state(
        &self,
        values: &[ArrayRef],
        opt_filter: Option<&BooleanArray>,
    ) -> Result<Vec<ArrayRef>> {
        // The state of each row on its own: a collection of its item, or an empty one
        let header = dummy_collection_header()?;
        let mut bits = Vec::with_capacity(values[0].len());
        let mut counts = Vec::with_capacity(values[0].len());
        let mut builder = BinaryBuilder::with_capacity(values[0].len(), 0);

        let arg_types = [self.input_type.clone()];
        let args = [ColumnarValue::Array(values[0].clone())];
        let executor = WkbBytesExecutor::new(&arg_types, &args);
        let mut i = 0;
        executor.execute_wkb_void(|maybe_bytes| {
            let mut types_and_dims = GeometryTypeAndDimensionsSet::new();
            builder.write_all(&header)?;
            if let Some(bytes) = maybe_bytes
                && is_selected(opt_filter, i)
            {
                let item = parse_item(bytes)?;
                types_and_dims
                    .insert(&item_types_and_dims(&item)?)
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
                counts.push(1);
                builder.write_all(item.buf())?;
            } else {
                counts.push(0);
            }
            bits.push(types_and_dims.bits());
            builder.append_value([]);
            i += 1;
            Ok(())
        })?;

        Ok(vec![
            Arc::new(UInt32Array::from(bits)),
            Arc::new(Int64Array::from(counts)),
            Arc::new(builder.finish()),
        ])
    }

    fn size(&self) -> usize {
        size_of::<CollectionGroupsAccumulator>()
            + self.types_and_dims.capacity() * size_of::<GeometryTypeAndDimensionsSet>()
            + self.counts.capacity() * size_of::<i64>()
            + self.items.capacity() * size_of::<Vec<u8>>()
            + self.items_capacity
    }
}

#[cfg(test)]
mod test {
    use arrow_array::{BinaryArray, BinaryViewArray};
    use datafusion_expr::AggregateUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::{
        WKB_GEOGRAPHY_ITEM_CRS, WKB_GEOMETRY_ITEM_CRS, WKB_VIEW_GEOGRAPHY, WKB_VIEW_GEOMETRY,
    };
    use sedona_testing::{
        compare::{assert_array_equal, assert_scalar_equal, assert_scalar_equal_wkb_geometry},
        create::{create_array, create_array_item_crs, create_scalar, create_scalar_item_crs},
        testers::AggregateUdfTester,
    };

    use super::*;

    #[test]
    fn udf_metadata() {
        let udf: AggregateUDF = st_collect_agg_udf().into();
        assert_eq!(udf.name(), "st_collect_agg");
    }

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester =
            AggregateUdfTester::new(st_collect_agg_udf().into(), vec![sedona_type.clone()]);
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
            "Can't ST_Collect_Agg() mixed dimension geometries"
        );
    }

    #[rstest]
    fn udf_grouped_accumulate(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester =
            AggregateUdfTester::new(st_collect_agg_udf().into(), vec![sedona_type.clone()]);

        // Six rows in four groups: group 0 has a null among its items, group 1 takes an
        // item from each batch, and group 3 has only a null
        let group_indices = vec![0, 3, 1, 1, 0, 2];
        let batches = vec![
            create_array(
                &[Some("POINT (0 1)"), None, Some("POINT (2 3)")],
                &sedona_type,
            ),
            create_array(
                &[Some("LINESTRING (4 5, 6 7)"), None, Some("POINT (8 9)")],
                &sedona_type,
            ),
        ];

        let expected = create_array(
            &[
                Some("MULTIPOINT (0 1)"),
                Some("GEOMETRYCOLLECTION (POINT (2 3), LINESTRING (4 5, 6 7))"),
                Some("MULTIPOINT (8 9)"),
                None,
            ],
            &WKB_GEOMETRY,
        );
        for emit_sizes in [vec![], vec![1, 1, 1, 1], vec![3, 1]] {
            let result = tester
                .aggregate_groups(&batches, group_indices.clone(), None, emit_sizes)
                .unwrap();
            assert_array_equal(&result, &expected);
        }

        // Keep only the two rows of group 1
        let filter = vec![false, false, true, true, false, false];
        let expected = create_array(
            &[
                None,
                Some("GEOMETRYCOLLECTION (POINT (2 3), LINESTRING (4 5, 6 7))"),
                None,
                None,
            ],
            &WKB_GEOMETRY,
        );
        let result = tester
            .aggregate_groups(&batches, group_indices, Some(&filter), vec![])
            .unwrap();
        assert_array_equal(&result, &expected);
    }

    #[test]
    fn udf_grouped_mixed_dimensions() {
        let tester = AggregateUdfTester::new(st_collect_agg_udf().into(), vec![WKB_GEOMETRY]);
        let batches = vec![create_array(
            &[
                Some("POINT (0 1)"),
                Some("POINT Z (0 1 2)"),
                Some("POINT (2 3)"),
            ],
            &WKB_GEOMETRY,
        )];
        let err = tester
            .aggregate_groups(&batches, vec![0, 0, 1], None, vec![])
            .unwrap_err();
        assert_eq!(
            err.message(),
            "Can't ST_Collect_Agg() mixed dimension geometries"
        );
    }

    /// Merging the state that convert_to_state() makes of some rows gives the same
    /// collections as updating with those rows, with nulls and a filter
    #[rstest]
    fn groups_convert_to_state(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let values = vec![create_array(
            &[
                Some("POINT (0 1)"),
                None,
                Some("POINT (2 3)"),
                Some("LINESTRING (4 5, 6 7)"),
            ],
            &sedona_type,
        )];
        let filter = BooleanArray::from(vec![true, true, false, true]);
        let group_indices = [0, 0, 1, 1];
        let expected = create_array(
            &[
                Some("MULTIPOINT (0 1)"),
                Some("MULTILINESTRING ((4 5, 6 7))"),
            ],
            &WKB_GEOMETRY,
        );

        let converter = CollectionGroupsAccumulator::new(sedona_type.clone());
        let state = converter.convert_to_state(&values, Some(&filter)).unwrap();
        let mut merged = CollectionGroupsAccumulator::new(sedona_type.clone());
        merged.merge_batch(&state, &group_indices, 2).unwrap();
        assert_array_equal(&merged.evaluate(EmitTo::All).unwrap(), &expected);

        let mut updated = CollectionGroupsAccumulator::new(sedona_type);
        updated
            .update_batch(&values, &group_indices, Some(&filter), 2)
            .unwrap();
        assert_array_equal(&updated.evaluate(EmitTo::All).unwrap(), &expected);
    }

    /// Rows the filter leaves out are not parsed, as they are not when DataFusion
    /// filters the rows for CollectionAccumulator
    #[rstest]
    fn groups_filter_skips_unparsable_rows(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let point = create_array(&[Some("POINT (1 2)")], &WKB_GEOMETRY);
        let point = as_binary_array(&point).unwrap().value(0);
        let bad: &[u8] = &[0x01, 0xff, 0xff, 0xff, 0xff];
        let rows = vec![Some(point), Some(bad), Some(bad)];
        let values: Vec<ArrayRef> = match sedona_type.storage_type() {
            DataType::BinaryView => vec![Arc::new(BinaryViewArray::from(rows))],
            _ => vec![Arc::new(BinaryArray::from(rows))],
        };
        let filter = BooleanArray::from(vec![Some(true), Some(false), None]);
        let group_indices = [0, 0, 0];
        let expected = create_array(&[Some("MULTIPOINT (1 2)")], &WKB_GEOMETRY);

        let mut updated = CollectionGroupsAccumulator::new(sedona_type.clone());
        updated
            .update_batch(&values, &group_indices, Some(&filter), 1)
            .unwrap();
        assert_array_equal(&updated.evaluate(EmitTo::All).unwrap(), &expected);

        let converter = CollectionGroupsAccumulator::new(sedona_type.clone());
        let state = converter.convert_to_state(&values, Some(&filter)).unwrap();
        let mut merged = CollectionGroupsAccumulator::new(sedona_type);
        merged.merge_batch(&state, &group_indices, 1).unwrap();
        assert_array_equal(&merged.evaluate(EmitTo::All).unwrap(), &expected);
    }

    /// The groups accumulator's state is that of CollectionAccumulator
    #[test]
    fn groups_state_merges_into_accumulator() {
        let values = vec![create_array(
            &[Some("POINT (0 1)"), Some("POINT (2 3)")],
            &WKB_GEOMETRY,
        )];
        let mut groups = CollectionGroupsAccumulator::new(WKB_GEOMETRY);
        groups.update_batch(&values, &[0, 0], None, 1).unwrap();
        let state = groups.state(EmitTo::All).unwrap();

        let mut acc = CollectionAccumulator::try_new(WKB_GEOMETRY, WKB_GEOMETRY).unwrap();
        acc.merge_batch(&state).unwrap();
        assert_scalar_equal_wkb_geometry(&acc.evaluate().unwrap(), Some("MULTIPOINT (0 1, 2 3)"));

        let mut acc = CollectionAccumulator::try_new(WKB_GEOMETRY, WKB_GEOMETRY).unwrap();
        acc.update_batch(&values).unwrap();
        let state: Vec<ArrayRef> = acc
            .state()
            .unwrap()
            .iter()
            .map(|s| s.to_array().unwrap())
            .collect();
        let mut groups = CollectionGroupsAccumulator::new(WKB_GEOMETRY);
        groups.merge_batch(&state, &[0], 1).unwrap();
        assert_array_equal(
            &groups.evaluate(EmitTo::All).unwrap(),
            &create_array(&[Some("MULTIPOINT (0 1, 2 3)")], &WKB_GEOMETRY),
        );
    }

    /// size() counts the WKB the groups hold until they are emitted
    #[test]
    fn groups_accumulator_size_counts_items() {
        let mut groups = CollectionGroupsAccumulator::new(WKB_GEOMETRY);
        let empty_size = groups.size();

        let values = vec![create_array(
            &[Some("LINESTRING (0 0, 1 1, 2 2, 3 3)")],
            &WKB_GEOMETRY,
        )];
        groups.update_batch(&values, &[0], None, 1).unwrap();
        // 9 header bytes and four XY coordinates of 16 bytes
        assert!(groups.size() >= empty_size + 9 + 4 * 16);

        groups.evaluate(EmitTo::All).unwrap();
        assert_eq!(groups.size(), empty_size);
    }

    #[rstest]
    fn udf_geog(#[values(WKB_GEOGRAPHY, WKB_VIEW_GEOGRAPHY)] sedona_type: SedonaType) {
        let tester =
            AggregateUdfTester::new(st_collect_agg_udf().into(), vec![sedona_type.clone()]);
        assert_eq!(tester.return_type().unwrap(), WKB_GEOGRAPHY);
    }

    #[rstest]
    fn udf_invoke_item_crs(
        #[values(WKB_GEOMETRY_ITEM_CRS.clone(), WKB_GEOGRAPHY_ITEM_CRS.clone())]
        sedona_type: SedonaType,
    ) {
        let tester =
            AggregateUdfTester::new(st_collect_agg_udf().into(), vec![sedona_type.clone()]);
        assert_eq!(tester.return_type().unwrap(), sedona_type.clone());

        let batch0 = create_array(
            &[Some("POINT (0 1)"), None, Some("POINT (2 3)")],
            &sedona_type,
        );
        let batch1 = create_array(
            &[Some("POINT (4 5)"), None, Some("POINT (6 7)")],
            &sedona_type,
        );

        let batches = vec![batch0, batch1];
        let expected = create_scalar(Some("MULTIPOINT (0 1, 2 3, 4 5, 6 7)"), &sedona_type);

        assert_scalar_equal(&tester.aggregate(&batches).unwrap(), &expected);
    }

    #[rstest]
    fn udf_invoke_item_crs_idential_crs() {
        let sedona_type = WKB_GEOMETRY_ITEM_CRS.clone();
        let tester =
            AggregateUdfTester::new(st_collect_agg_udf().into(), vec![sedona_type.clone()]);
        assert_eq!(tester.return_type().unwrap(), sedona_type.clone());

        let batch0 = create_array_item_crs(
            &[Some("POINT (0 1)"), None, Some("POINT (2 3)")],
            [Some("EPSG:4326"), None, Some("EPSG:4326")],
            &WKB_GEOMETRY,
        );
        let batch1 = create_array_item_crs(
            &[Some("POINT (4 5)"), None, Some("POINT (6 7)")],
            [Some("EPSG:4326"), None, Some("EPSG:4326")],
            &WKB_GEOMETRY,
        );

        let expected = create_scalar_item_crs(
            Some("MULTIPOINT (0 1, 2 3, 4 5, 6 7)"),
            Some("EPSG:4326"),
            &WKB_GEOMETRY,
        );

        assert_scalar_equal(
            &tester
                .aggregate(&vec![batch0.clone(), batch1.clone()])
                .unwrap(),
            &expected,
        );
    }

    #[rstest]
    fn udf_invoke_item_crs_multiple_compatible_crs() {
        let sedona_type = WKB_GEOMETRY_ITEM_CRS.clone();
        let tester =
            AggregateUdfTester::new(st_collect_agg_udf().into(), vec![sedona_type.clone()]);
        assert_eq!(tester.return_type().unwrap(), sedona_type.clone());

        let batch0 = create_array_item_crs(
            &[Some("POINT (0 1)"), None, Some("POINT (2 3)")],
            [Some("OGC:CRS84"), None, Some("EPSG:4326")],
            &WKB_GEOMETRY,
        );
        let batch1 = create_array_item_crs(
            &[Some("POINT (4 5)"), None, Some("POINT (6 7)")],
            [Some("EPSG:4326"), None, Some("OGC:CRS84")],
            &WKB_GEOMETRY,
        );

        let expected = create_scalar_item_crs(
            Some("MULTIPOINT (0 1, 2 3, 4 5, 6 7)"),
            Some("OGC:CRS84"),
            &WKB_GEOMETRY,
        );

        assert_scalar_equal(
            &tester
                .aggregate(&vec![batch0.clone(), batch1.clone()])
                .unwrap(),
            &expected,
        );
    }

    #[rstest]
    fn udf_invoke_item_crs_incompatible_crs() {
        let sedona_type = WKB_GEOMETRY_ITEM_CRS.clone();
        let tester =
            AggregateUdfTester::new(st_collect_agg_udf().into(), vec![sedona_type.clone()]);
        assert_eq!(tester.return_type().unwrap(), sedona_type.clone());

        let batch0 = create_array_item_crs(
            &[Some("POINT (0 1)"), None, Some("POINT (2 3)")],
            [Some("OGC:CRS84"), None, Some("EPSG:4326")],
            &WKB_GEOMETRY,
        );

        // We should error if we see incompatible CRSes between batches
        let batch1_other_crs = create_array_item_crs(
            &[Some("POINT (4 5)"), None, Some("POINT (6 7)")],
            [Some("EPSG:3857"), None, Some("EPSG:3857")],
            &WKB_GEOMETRY,
        );
        let err = tester
            .aggregate(&vec![batch0.clone(), batch1_other_crs.clone()])
            .unwrap_err();
        assert_eq!(
            err.message(),
            "CRS values not equal: ogc:crs84 vs epsg:3857"
        );

        // We should error if we see incompatible CRSes between batches (None
        // should be considered an incompatible CRS)
        let batch1_other_crs = create_array_item_crs(
            &[Some("POINT (4 5)"), None, Some("POINT (6 7)")],
            [None, None, None],
            &WKB_GEOMETRY,
        );
        let err = tester
            .aggregate(&vec![batch0.clone(), batch1_other_crs.clone()])
            .unwrap_err();
        assert_eq!(err.message(), "CRS values not equal: ogc:crs84 vs None");

        // Or if we see incompatible CRSes in a single batch
        let batch0_incompatible_crses = create_array_item_crs(
            &[Some("POINT (4 5)"), None, Some("POINT (6 7)")],
            [Some("OGC:CRS84"), None, Some("EPSG:3857")],
            &WKB_GEOMETRY,
        );

        let err = tester
            .aggregate(&vec![batch0_incompatible_crses.clone()])
            .unwrap_err();
        assert_eq!(
            err.message(),
            "CRS values not equal: ogc:crs84 vs epsg:3857"
        );
    }

    /// A serialized state round-trips: emitting state and merging it back into
    /// a fresh accumulator reconstructs the same type/dimension set and count.
    #[test]
    fn state_round_trips() {
        let mut acc = CollectionAccumulator::try_new(WKB_GEOMETRY, WKB_GEOMETRY).unwrap();
        for td in [
            GeometryTypeAndDimensions::new(GeometryTypeId::Point, Dimensions::Xy),
            GeometryTypeAndDimensions::new(GeometryTypeId::LineString, Dimensions::Xyz),
        ] {
            acc.types_and_dims.insert(&td).unwrap();
        }
        acc.count = 3;
        let expected_set = acc.types_and_dims.clone();

        let state = acc.state().unwrap();
        let arrays: Vec<ArrayRef> = state.iter().map(|s| s.to_array().unwrap()).collect();

        let mut merged = CollectionAccumulator::try_new(WKB_GEOMETRY, WKB_GEOMETRY).unwrap();
        merged.merge_batch(&arrays).unwrap();

        assert_eq!(merged.types_and_dims, expected_set);
        assert_eq!(merged.count, 3);
    }

    /// Regression test for memory accounting: the accumulator must report an
    /// exact size with no unaccounted per-group heap. The HashSets this
    /// replaces allocated ~100 heap bytes per non-empty group that size()
    /// reported as zero, hiding gigabytes of aggregate state from the memory
    /// pool on large aggregations (no spill, unaccounted anon growth).
    #[test]
    fn accumulator_size_is_exact() {
        let mut acc = CollectionAccumulator::try_new(WKB_GEOMETRY, WKB_GEOMETRY).unwrap();
        let empty_size = acc.size();

        acc.types_and_dims
            .insert(&GeometryTypeAndDimensions::new(
                GeometryTypeId::Point,
                Dimensions::Xy,
            ))
            .unwrap();

        // Inserting into the inline bitset allocates nothing.
        assert_eq!(acc.size(), empty_size);
        assert_eq!(
            acc.size(),
            size_of::<CollectionAccumulator>() + acc.item.as_ref().unwrap().capacity()
        );
    }
}
