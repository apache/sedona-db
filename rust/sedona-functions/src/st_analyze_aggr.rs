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
use std::vec;

use std::{mem::size_of_val, sync::Arc};

use arrow_array::builder::Float64Builder;
use arrow_array::builder::Int64Builder;
use arrow_array::StructArray;
use arrow_array::{Array, ArrayRef, Float64Array, Int64Array};
use arrow_schema::{DataType, Field, FieldRef};
use datafusion_common::{
    cast::as_binary_array,
    error::{DataFusionError, Result},
    ScalarValue,
};
use datafusion_expr::{scalar_doc_sections::DOC_SECTION_OTHER, Documentation, Volatility};
use datafusion_expr::{Accumulator, ColumnarValue};
use sedona_expr::aggregate_udf::SedonaAccumulatorRef;
use sedona_expr::aggregate_udf::SedonaAggregateUDF;
use sedona_expr::{aggregate_udf::SedonaAccumulator, statistics::GeoStatistics};
use sedona_geometry::analyze::GeometryAnalysis;
use sedona_geometry::interval::IntervalTrait;
use sedona_geometry::types::GeometryTypeAndDimensions;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};
use wkb::reader::Wkb;

use crate::executor::WkbExecutor;

/// ST_Analyze_Aggr() aggregate UDF implementation
///
/// This function computes comprehensive statistics for a collection of geometries.
/// It returns a struct containing various metrics such as count, min/max coordinates,
/// mean size, and geometry type counts.
pub fn st_analyze_aggr_udf() -> SedonaAggregateUDF {
    SedonaAggregateUDF::new(
        "st_analyze_aggr",
        vec![Arc::new(STAnalyzeAggr {})],
        Volatility::Immutable,
        Some(st_analyze_aggr_doc()),
    )
}

fn st_analyze_aggr_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Return Return the statistics of geometries for geom.",
        "ST_Analyze_Aggr (A: Geometry)",
    )
    .with_argument("geom", "geometry: Input geometry or geography")
    .with_sql_example("
        SELECT ST_Analyze_Aggr(ST_GeomFromText('MULTIPOINT(1.1 101.1,2.1 102.1,3.1 103.1,4.1 104.1,5.1 105.1,6.1 106.1,7.1 107.1,8.1 108.1,9.1 109.1,10.1 110.1)'))")
    .build()
}
/// ST_Analyze_Aggr() implementation
pub fn st_analyze_aggr_impl() -> SedonaAccumulatorRef {
    Arc::new(STAnalyzeAggr {})
}

#[derive(Debug)]
struct STAnalyzeAggr {}

impl SedonaAccumulator for STAnalyzeAggr {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let output_fields = Self::output_fields();

        let r_type = SedonaType::Arrow(DataType::Struct(output_fields.into()));
        let matcher = ArgMatcher::new(vec![ArgMatcher::is_geometry()], r_type);
        matcher.match_args(args)
    }

    fn accumulator(
        &self,
        args: &[SedonaType],
        output_type: &SedonaType,
    ) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(AnalyzeAccumulator::new(
            args[0].clone(),
            output_type.clone(),
        )))
    }

    fn state_fields(&self, _args: &[SedonaType]) -> Result<Vec<FieldRef>> {
        Ok(vec![Arc::new(Field::new(
            "analyze",
            DataType::Binary,
            true,
        ))])
    }
}

impl STAnalyzeAggr {
    fn create_float64_array(value: Option<f64>) -> Float64Array {
        match value {
            Some(v) => Float64Array::from(vec![v]),
            None => {
                let mut builder = Float64Builder::new();
                builder.append_null();
                builder.finish()
            }
        }
    }

    fn create_int64_array(value: Option<i64>) -> Int64Array {
        match value {
            Some(v) => Int64Array::from(vec![v]),
            None => {
                let mut builder = Int64Builder::new();
                builder.append_null();
                builder.finish()
            }
        }
    }

    fn output_fields() -> Vec<Field> {
        let output_fields = vec![
            Field::new("count", DataType::Int64, true),
            Field::new("minx", DataType::Float64, true),
            Field::new("miny", DataType::Float64, true),
            Field::new("maxx", DataType::Float64, true),
            Field::new("maxy", DataType::Float64, true),
            Field::new("mean_size_in_bytes", DataType::Int64, true),
            Field::new("mean_points_per_geometry", DataType::Float64, true),
            Field::new("puntal_count", DataType::Int64, true),
            Field::new("lineal_count", DataType::Int64, true),
            Field::new("polygonal_count", DataType::Int64, true),
            Field::new("geometrycollection_count", DataType::Int64, true),
            Field::new("mean_envelope_width", DataType::Float64, true),
            Field::new("mean_envelope_height", DataType::Float64, true),
            Field::new("mean_envelope_area", DataType::Float64, true),
        ];
        output_fields
    }

    fn output_arrays(stats: GeoStatistics) -> Vec<(FieldRef, ArrayRef)> {
        // Get total geometries count
        let total_geometries = stats.total_geometries().unwrap_or(0);

        // Handle bounding box values for empty collections
        let (min_x, min_y, max_x, max_y) = if let Some(bbox) = stats.bbox() {
            // Get actual values from the bbox
            (
                Some(bbox.x().lo()),
                Some(bbox.y().lo()),
                Some(bbox.x().hi()),
                Some(bbox.y().hi()),
            )
        } else if total_geometries == 0 {
            // When no geometries, use null values for bounds
            (None, None, None, None)
        } else {
            // This case shouldn't happen but default to zeros
            (Some(0.0), Some(0.0), Some(0.0), Some(0.0))
        };

        // Calculate means - use None for empty collections
        let mean_size_bytes = if total_geometries > 0 {
            Some(stats.total_size_bytes().unwrap_or(0) / total_geometries)
        } else {
            None
        };

        let mean_points = if total_geometries > 0 {
            Some(stats.total_points().unwrap_or(0) as f64 / total_geometries as f64)
        } else {
            None
        };

        // Calculate mean envelope dimensions
        let mean_envelope_width = if total_geometries > 0 {
            Some(stats.total_envelope_width().unwrap_or(0.0) / total_geometries as f64)
        } else {
            None
        };

        let mean_envelope_height = if total_geometries > 0 {
            Some(stats.total_envelope_height().unwrap_or(0.0) / total_geometries as f64)
        } else {
            None
        };

        let mean_envelope_area = if total_geometries > 0 {
            mean_envelope_width
                .zip(mean_envelope_height)
                .map(|(w, h)| w * h)
        } else {
            None
        };

        // Define output fields
        let fields = STAnalyzeAggr::output_fields();

        // Create arrays with proper null handling
        let values = vec![
            Arc::new(Int64Array::from(vec![total_geometries])) as ArrayRef,
            Arc::new(Self::create_float64_array(min_x)) as ArrayRef,
            Arc::new(Self::create_float64_array(min_y)) as ArrayRef,
            Arc::new(Self::create_float64_array(max_x)) as ArrayRef,
            Arc::new(Self::create_float64_array(max_y)) as ArrayRef,
            Arc::new(Self::create_int64_array(mean_size_bytes)) as ArrayRef,
            Arc::new(Self::create_float64_array(mean_points)) as ArrayRef,
            Arc::new(Int64Array::from(vec![stats.puntal_count().unwrap_or(0)])) as ArrayRef,
            Arc::new(Int64Array::from(vec![stats.lineal_count().unwrap_or(0)])) as ArrayRef,
            Arc::new(Int64Array::from(vec![stats.polygonal_count().unwrap_or(0)])) as ArrayRef,
            Arc::new(Int64Array::from(vec![stats
                .collection_count()
                .unwrap_or(0)])) as ArrayRef,
            Arc::new(Self::create_float64_array(mean_envelope_width)) as ArrayRef,
            Arc::new(Self::create_float64_array(mean_envelope_height)) as ArrayRef,
            Arc::new(Self::create_float64_array(mean_envelope_area)) as ArrayRef,
        ];

        // Pair fields with values
        fields.into_iter().map(Arc::new).zip(values).collect()
    }
}

#[derive(Debug)]
pub struct AnalyzeAccumulator {
    input_type: SedonaType,
    _output_type: SedonaType,
    stats: GeoStatistics,
}

impl AnalyzeAccumulator {
    pub fn new(input_type: SedonaType, output_type: SedonaType) -> Self {
        Self {
            input_type,
            _output_type: output_type,
            stats: GeoStatistics::empty(),
        }
    }

    pub fn update_statistics(&mut self, geom: &Wkb, size_bytes: usize) -> Result<()> {
        // Get geometry analysis information
        let analysis = sedona_geometry::analyze::analyze_geometry(geom)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        // Start with a clone of the current stats
        let mut stats = self.stats.clone();

        // Update each component of the statistics
        stats = self.update_basic_counts(stats, size_bytes);
        stats = self.update_geometry_type_counts(stats, &analysis);
        stats = self.update_point_count(stats, analysis.point_count);
        stats = self.update_envelope_info(stats, &analysis);
        stats = self.update_geometry_types(stats, analysis.geometry_type);

        // Assign the updated stats back to self.stats
        self.stats = stats;

        Ok(())
    }

    pub fn finish(self) -> GeoStatistics {
        self.stats
    }

    // Update basic counts (total geometries and size)
    fn update_basic_counts(&self, stats: GeoStatistics, size_bytes: usize) -> GeoStatistics {
        let total_geometries = stats.total_geometries().unwrap_or(0) + 1;
        stats
            .clone()
            .with_total_geometries(total_geometries)
            .with_total_size_bytes(stats.total_size_bytes().unwrap_or(0) + size_bytes as i64)
    }

    // Update geometry type counts
    fn update_geometry_type_counts(
        &self,
        stats: GeoStatistics,
        analysis: &GeometryAnalysis,
    ) -> GeoStatistics {
        // Add the counts from analysis to existing stats
        let puntal = stats.puntal_count().unwrap_or(0) + analysis.puntal_count;
        let lineal = stats.lineal_count().unwrap_or(0) + analysis.lineal_count;
        let polygonal = stats.polygonal_count().unwrap_or(0) + analysis.polygonal_count;
        let collection = stats.collection_count().unwrap_or(0) + analysis.collection_count;

        stats
            .with_puntal_count(puntal)
            .with_lineal_count(lineal)
            .with_polygonal_count(polygonal)
            .with_collection_count(collection)
    }

    // Update point count statistics
    fn update_point_count(&self, stats: GeoStatistics, point_count: i64) -> GeoStatistics {
        let total_points = stats.total_points().unwrap_or(0) + point_count;
        stats.with_total_points(total_points)
    }

    // Update envelope dimensions and bounding box
    fn update_envelope_info(
        &self,
        stats: GeoStatistics,
        analysis: &GeometryAnalysis,
    ) -> GeoStatistics {
        // The bbox is directly available on analysis, not wrapped in an Option
        let bbox = &analysis.bbox;

        // Calculate envelope width and height from the bbox
        let envelope_width = if bbox.x().is_empty() {
            0.0
        } else {
            bbox.x().width()
        };
        let envelope_height = if bbox.y().is_empty() {
            0.0
        } else {
            bbox.y().width()
        };

        // Update envelope dimensions
        let total_width = stats.total_envelope_width().unwrap_or(0.0) + envelope_width;
        let total_height = stats.total_envelope_height().unwrap_or(0.0) + envelope_height;

        let stats = stats
            .with_total_envelope_width(total_width)
            .with_total_envelope_height(total_height);

        // Update bounding box
        let existing_bbox = stats.bbox();
        if let Some(current_bbox) = existing_bbox {
            let mut updated_bbox = bbox.clone();
            updated_bbox.update_box(current_bbox);
            stats.with_bbox(Some(updated_bbox))
        } else {
            stats.with_bbox(Some(bbox.clone()))
        }
    }

    // Update geometry types
    fn update_geometry_types(
        &self,
        stats: GeoStatistics,
        geometry_type: GeometryTypeAndDimensions,
    ) -> GeoStatistics {
        let current_types = stats.geometry_types();
        let types = if let Some(existing_types) = current_types {
            let mut new_types = existing_types.clone();
            new_types.insert(geometry_type);
            Some(new_types)
        } else {
            Some(std::collections::HashSet::from([geometry_type]))
        };

        if let Some(type_set) = &types {
            let type_vec: Vec<GeometryTypeAndDimensions> = type_set.iter().cloned().collect();
            stats.with_geometry_types(Some(&type_vec))
        } else {
            stats.with_geometry_types(None)
        }
    }

    fn execute_update(&mut self, executor: WkbExecutor) -> Result<()> {
        executor.execute_wkb_void(|maybe_item| {
            if let Some(item) = maybe_item {
                self.update_statistics(&item, item.buf().len())?;
            }
            Ok(())
        })?;
        Ok(())
    }
}

impl Accumulator for AnalyzeAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.is_empty() {
            return Err(DataFusionError::Internal(
                "No input arrays provided to accumulator".to_string(),
            ));
        }
        let arg_types = [self.input_type.clone()];
        let arg_values = [ColumnarValue::Array(values[0].clone())];
        let executor = WkbExecutor::new(&arg_types, &arg_values);
        self.execute_update(executor)?;
        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let field_array_pairs = STAnalyzeAggr::output_arrays(self.stats.clone());

        // Create the struct array with field values
        let struct_array = StructArray::from(field_array_pairs);

        // Return the ScalarValue::Struct with Arc<StructArray>
        Ok(ScalarValue::Struct(Arc::new(struct_array)))
    }

    fn size(&self) -> usize {
        let base_size = size_of_val(self);

        // Add approximate size for bbox if present
        let bbox_size = match self.stats.bbox() {
            Some(bbox) => size_of_val(bbox),
            None => 0,
        };

        // Add approximate size for geometry types if present
        let types_size = match self.stats.geometry_types() {
            Some(types) => {
                let elem_size = size_of::<GeometryTypeAndDimensions>();
                let capacity = types.capacity();
                capacity * elem_size
            }
            None => 0,
        };

        base_size + bbox_size + types_size
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        if self.stats.total_geometries().unwrap_or(0) == 0 {
            // Return null if no data was processed
            return Ok(vec![ScalarValue::Binary(None)]);
        }

        // Serialize the statistics to JSON
        let scalar = self.stats.to_scalar_value()?;
        Ok(vec![scalar])
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        // Check input length (expecting 1 state field)
        if states.is_empty() {
            return Err(DataFusionError::Internal(
                "No input arrays provided to accumulator in merge_batch".to_string(),
            ));
        }

        let array = &states[0];
        let binary_array = as_binary_array(&array)?;

        for i in 0..binary_array.len() {
            if binary_array.is_null(i) {
                continue;
            }

            let serialized = binary_array.value(i);
            let other_stats: GeoStatistics = serde_json::from_slice(serialized).map_err(|e| {
                DataFusionError::Internal(format!("Failed to deserialize stats: {e}"))
            })?;

            // Use the merge method to combine statistics
            self.stats.merge(&other_stats);
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use arrow_array::RecordBatch;
    use arrow_json::ArrayWriter;
    use arrow_schema::Schema;
    use rstest::rstest;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::testers::AggregateUdfTester;
    use serde_json::Value;

    fn actual_result_json(struct_array: Arc<StructArray>) -> Value {
        // Create a schema that matches the struct array fields
        let fields = struct_array.fields().to_vec();
        let schema = Schema::new(fields);

        // Create a record batch with just the struct array converted to a column
        let columns: Vec<ArrayRef> = struct_array.columns().to_vec();
        let record_batch = RecordBatch::try_new(Arc::new(schema), columns).unwrap();

        // Serialize to JSON using Arrow's JSON writer
        let buf = Vec::new();
        let mut writer = ArrayWriter::new(buf);
        writer.write_batches(&[&record_batch]).unwrap();
        writer.finish().unwrap();

        // Get the JSON string
        let json_str = String::from_utf8(writer.into_inner()).unwrap();
        let json_value: Value = serde_json::from_str(&json_str).unwrap();

        // Get the first (and only) row from the JSON array
        json_value[0].clone()
    }

    #[rstest]
    fn basic_analyze_cases(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let mut udaf = st_analyze_aggr_udf();
        udaf.add_kernel(st_analyze_aggr_impl());

        let tester = AggregateUdfTester::new(udaf.into(), vec![sedona_type.clone()]);

        // Basic point analysis
        let result = tester
            .aggregate_wkt(vec![vec![Some("POINT(0 0)"), Some("POINT(1 1)")]])
            .unwrap();

        assert!(matches!(result, ScalarValue::Struct(_)));

        if let ScalarValue::Struct(struct_array) = result {
            let actual_json = actual_result_json(struct_array);

            // Define the expected values
            let expected = serde_json::json!({
                "count": 2,
                "minx": 0.0,
                "miny": 0.0,
                "maxx": 1.0,
                "maxy": 1.0,
                "mean_size_in_bytes": 21,
                "mean_points_per_geometry": 1.0,
                "puntal_count": 2,
                "lineal_count": 0,
                "polygonal_count": 0,
                "geometrycollection_count": 0,
                "mean_envelope_width": 0.0,
                "mean_envelope_height": 0.0,
                "mean_envelope_area": 0.0,
            });

            // Single assertion to compare the entire structure
            assert_eq!(actual_json, expected);
        }
    }

    #[rstest]
    fn analyze_linestring(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let mut udaf = st_analyze_aggr_udf();
        udaf.add_kernel(st_analyze_aggr_impl());

        let tester = AggregateUdfTester::new(udaf.into(), vec![sedona_type.clone()]);

        // Batch with linestrings
        let result = tester
            .aggregate_wkt(vec![vec![
                Some("LINESTRING(0 0, 1 1, 2 2)"),
                Some("LINESTRING(0 0, 0 1, 1 1)"),
            ]])
            .unwrap();
        assert!(matches!(result, ScalarValue::Struct(_)));

        if let ScalarValue::Struct(struct_array) = result {
            let actual_json = actual_result_json(struct_array);

            let expected = serde_json::json!({
                "count": 2,
                "minx": 0.0,
                "miny": 0.0,
                "maxx": 2.0,
                "maxy": 2.0,
                "mean_size_in_bytes": 57, // Approximate size
                "mean_points_per_geometry": 3.0,
                "puntal_count": 0,
                "lineal_count": 2,
                "polygonal_count": 0,
                "geometrycollection_count": 0,
                "mean_envelope_width": 1.5,
                "mean_envelope_height": 1.5,
                "mean_envelope_area": 2.25,
            });

            assert_eq!(actual_json, expected);
        }
    }

    #[rstest]
    fn analyze_polygon(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let mut udaf = st_analyze_aggr_udf();
        udaf.add_kernel(st_analyze_aggr_impl());

        let tester = AggregateUdfTester::new(udaf.into(), vec![sedona_type.clone()]);

        // Batch with polygons
        let result = tester
            .aggregate_wkt(vec![vec![
                Some("POLYGON((0 0, 0 3, 3 3, 3 0, 0 0))"),
                Some("POLYGON((1 1, 1 2, 2 2, 2 1, 1 1))"),
            ]])
            .unwrap();
        assert!(matches!(result, ScalarValue::Struct(_)));

        if let ScalarValue::Struct(struct_array) = result {
            let actual_json = actual_result_json(struct_array);

            let expected = serde_json::json!({
                "count": 2,
                "minx": 0.0,
                "miny": 0.0,
                "maxx": 3.0,
                "maxy": 3.0,
                "mean_size_in_bytes": 93, // Approximate size
                "mean_points_per_geometry": 5.0,
                "puntal_count": 0,
                "lineal_count": 0,
                "polygonal_count": 2,
                "geometrycollection_count": 0,
                "mean_envelope_width": 2.0,
                "mean_envelope_height": 2.0,
                "mean_envelope_area": 4.0,
            });

            assert_eq!(actual_json, expected);
        }
    }

    #[rstest]
    fn analyze_mixed_geometries(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let mut udaf = st_analyze_aggr_udf();
        udaf.add_kernel(st_analyze_aggr_impl());

        let tester = AggregateUdfTester::new(udaf.into(), vec![sedona_type.clone()]);

        // Batch with mixed geometry types
        let result = tester
            .aggregate_wkt(vec![vec![
                Some("POINT(0 0)"),
                Some("LINESTRING(0 0, 1 1)"),
                Some("POLYGON((0 0, 0 2, 2 2, 2 0, 0 0))"),
                Some("MULTIPOINT((1 1), (2 2))"),
            ]])
            .unwrap();
        assert!(matches!(result, ScalarValue::Struct(_)));

        if let ScalarValue::Struct(struct_array) = result {
            let actual_json = actual_result_json(struct_array);

            let expected = serde_json::json!({
                "count": 4,
                "minx": 0.0,
                "miny": 0.0,
                "maxx": 2.0,
                "maxy": 2.0,
                "mean_size_in_bytes": 51, // Approximate size
                "mean_points_per_geometry": 2.5,
                "puntal_count": 2,
                "lineal_count": 1,
                "polygonal_count": 1,
                "geometrycollection_count": 0,
                "mean_envelope_width": 1.0,
                "mean_envelope_height": 1.0,
                "mean_envelope_area": 1.0,
            });

            assert_eq!(actual_json, expected);
        }
    }

    #[rstest]
    fn analyze_empty_input(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let mut udaf = st_analyze_aggr_udf();
        udaf.add_kernel(st_analyze_aggr_impl());

        let tester = AggregateUdfTester::new(udaf.into(), vec![sedona_type.clone()]);

        // Empty batch
        let result = tester.aggregate_wkt(vec![vec![None, None]]).unwrap();
        assert!(matches!(result, ScalarValue::Struct(_)));

        if let ScalarValue::Struct(struct_array) = result {
            let actual_json = actual_result_json(struct_array);

            let expected = serde_json::json!({
                // Only count should be 0, all other values should be null
                "count": 0,
                "geometrycollection_count": 0,
                "lineal_count": 0,
                "maxx": null,
                "maxy": null,
                "minx": null,
                "miny": null,
                "polygonal_count": 0,
                "puntal_count": 0,
            });

            assert_eq!(actual_json, expected);
        }
    }
}
