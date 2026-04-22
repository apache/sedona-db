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
use core::fmt;
use std::{mem::transmute, sync::Arc};

use arrow::compute::{concat as arrow_concat, interleave as arrow_interleave};
use arrow_array::{Array, ArrayRef, Float64Array, RecordBatch};
use arrow_schema::DataType;
use datafusion_common::{utils::proxy::VecAllocExt, JoinSide, Result, ScalarValue};
use datafusion_expr::ColumnarValue;
use datafusion_physical_expr::PhysicalExpr;
use float_next_after::NextAfter;
use geo_index::rtree::util::f64_box_to_f32;
use geo_types::{coord, Rect};
use sedona_functions::executor::IterGeo;
use sedona_geo_generic_alg::BoundingRect;
use sedona_schema::datatypes::SedonaType;
use wkb::reader::Wkb;

use sedona_common::sedona_internal_err;

use crate::{
    spatial_predicate::{DistancePredicate, KNNPredicate, RelationPredicate, SpatialPredicate},
    utils::arrow_utils::get_array_memory_size,
};

/// Factory for [EvaluatedGeometryArray] instances given an evaluated geometry column
///
/// Allows join providers to customize the eagerly evaluated representation of geometry.
/// This is currently limited to a concrete internal representation but may expand to
/// support non-Cartesian bounding or non-WKB backed geometry arrays. This also may
/// expand to support more compact or efficient serialization/deserialization of the
/// evaluated array when spilling.
pub trait EvaluatedGeometryArrayFactory: fmt::Debug + Send + Sync {
    /// Create a new [EvaluatedGeometryArray]
    fn try_new_evaluated_array(
        &self,
        geometry_array: ArrayRef,
        sedona_type: &SedonaType,
    ) -> Result<EvaluatedGeometryArray>;
}

/// Default EvaluatedGeometryArray factory
///
/// This factory constructs an array
#[derive(Debug)]
pub(crate) struct DefaultGeometryArrayFactory;

impl EvaluatedGeometryArrayFactory for DefaultGeometryArrayFactory {
    fn try_new_evaluated_array(
        &self,
        geometry_array: ArrayRef,
        sedona_type: &SedonaType,
    ) -> Result<EvaluatedGeometryArray> {
        EvaluatedGeometryArray::try_new(geometry_array, sedona_type)
    }
}

/// Operand evaluator is for evaluating the operands of a spatial predicate. It can be a distance
/// operand evaluator or a relation operand evaluator.
pub(crate) trait OperandEvaluator: fmt::Debug + Send + Sync {
    /// Evaluate the spatial predicate operand on the build side.
    fn evaluate_build(&self, batch: &RecordBatch) -> Result<EvaluatedGeometryArray>;

    /// Evaluate the spatial predicate operand on the probe side.
    fn evaluate_probe(&self, batch: &RecordBatch) -> Result<EvaluatedGeometryArray>;

    /// Get the expression for the build side.
    fn build_side_expr(&self) -> Result<Arc<dyn PhysicalExpr>>;

    /// Get the expression for the probe side.
    fn probe_side_expr(&self) -> Result<Arc<dyn PhysicalExpr>>;
}

/// Create a spatial predicate evaluator for the spatial predicate.
pub(crate) fn create_operand_evaluator(
    predicate: &SpatialPredicate,
    evaluated_array_factory: Arc<dyn EvaluatedGeometryArrayFactory>,
) -> Arc<dyn OperandEvaluator> {
    match predicate {
        SpatialPredicate::Distance(predicate) => Arc::new(DistanceOperandEvaluator::new(
            predicate.clone(),
            evaluated_array_factory,
        )),
        SpatialPredicate::Relation(predicate) => Arc::new(RelationOperandEvaluator::new(
            predicate.clone(),
            evaluated_array_factory,
        )),
        SpatialPredicate::KNearestNeighbors(predicate) => Arc::new(KNNOperandEvaluator::new(
            predicate.clone(),
            evaluated_array_factory,
        )),
    }
}

/// Result of evaluating a geometry batch.
pub struct EvaluatedGeometryArray {
    sedona_type: SedonaType,
    /// The array of geometries produced by evaluating the geometry expression.
    geometry_array: ArrayRef,
    /// The rects of the geometries in the geometry array. The length of this array is equal to the number of geometries.
    /// The rects will be None for empty or null geometries.
    rects: Vec<Option<Rect<f32>>>,
    /// The distance value produced by evaluating the distance expression.
    distance: Option<ColumnarValue>,
    /// WKBs of the geometries in `geometry_array`. The wkb values reference buffers inside the geometry array,
    /// but we'll only allow accessing Wkb<'a> where 'a is the lifetime of the GeometryBatchResult to make
    /// the interfaces safe. The buffers in `geometry_array` are allocated on the heap and won't be moved when
    /// the GeometryBatchResult is moved, so we don't need to worry about pinning.
    wkbs: Vec<Option<Wkb<'static>>>,
}

impl EvaluatedGeometryArray {
    /// Create a new EvaluatedGeometryArray and compute item bounds
    pub fn try_new(geometry_array: ArrayRef, sedona_type: &SedonaType) -> Result<Self> {
        let num_rows = geometry_array.len();
        let mut rect_vec = Vec::with_capacity(num_rows);
        let mut wkbs = Vec::with_capacity(num_rows);
        geometry_array.iter_as_wkb(sedona_type, num_rows, |wkb_opt| {
            let rect_opt = if let Some(wkb) = &wkb_opt {
                if let Some(rect) = wkb.bounding_rect() {
                    let min = rect.min();
                    let max = rect.max();
                    // f64_box_to_f32 will ensure the resulting `f32` box is no smaller than the `f64` box.
                    let (min_x, min_y, max_x, max_y) = f64_box_to_f32(min.x, min.y, max.x, max.y);
                    let rect = Rect::new(coord!(x: min_x, y: min_y), coord!(x: max_x, y: max_y));
                    Some(rect)
                } else {
                    None
                }
            } else {
                None
            };

            rect_vec.push(rect_opt);

            // Safety: The wkbs must reference buffers inside the `geometry_array`. Since the `geometry_array` and
            // `wkbs` are both owned by the `EvaluatedGeometryArray`, so they have the same lifetime. We'll never
            // have a situation where the `EvaluatedGeometryArray` is dropped while the `wkbs` are still in use
            // (guaranteed by the scope of the `wkbs` field and lifetime signature of the `wkbs` method).
            wkbs.push(wkb_opt.map(|wkb| unsafe { transmute(wkb) }));
            Ok(())
        })?;

        Ok(Self {
            sedona_type: sedona_type.clone(),
            geometry_array,
            rects: rect_vec,
            distance: None,
            wkbs,
        })
    }

    /// Create a new EvaluatedGeometryArray with precomputed item bounds
    pub fn try_new_with_rects(
        geometry_array: ArrayRef,
        rect_vec: Vec<Option<Rect<f32>>>,
        sedona_type: &SedonaType,
    ) -> Result<Self> {
        // Safety: The wkbs must reference buffers inside the `geometry_array`. Since the `geometry_array` and
        // `wkbs` are both owned by the `EvaluatedGeometryArray`, so they have the same lifetime. We'll never
        // have a situation where the `EvaluatedGeometryArray` is dropped while the `wkbs` are still in use
        // (guaranteed by the scope of the `wkbs` field and lifetime signature of the `wkbs` method).
        let num_rows = geometry_array.len();
        let mut wkbs = Vec::with_capacity(num_rows);
        geometry_array.iter_as_wkb(sedona_type, num_rows, |wkb_opt| {
            wkbs.push(wkb_opt.map(|wkb| unsafe { transmute(wkb) }));
            Ok(())
        })?;

        if num_rows != rect_vec.len() {
            return sedona_internal_err!("Expected rect_vec to have same length as geometry array");
        }

        Ok(Self {
            sedona_type: sedona_type.clone(),
            geometry_array,
            rects: rect_vec,
            distance: None,
            wkbs,
        })
    }

    /// Set evaluated distance
    pub fn with_distance(mut self, distance: Option<ColumnarValue>) -> Self {
        self.distance = distance;
        self
    }

    /// Build a new `EvaluatedGeometryArray` by concatenating the provided source arrays in order.
    ///
    /// The rectangles are appended in source order without recomputation, while the WKB references
    /// are rebuilt from the concatenated Arrow array since their lifetimes are tied to it.
    pub fn concat(geom_arrays: &[&EvaluatedGeometryArray]) -> Result<Self> {
        if geom_arrays.is_empty() {
            return sedona_internal_err!("concat requires at least one geometry array");
        }

        let sedona_type = &geom_arrays[0].sedona_type;
        for geom in geom_arrays.iter().skip(1) {
            if geom.sedona_type != *sedona_type {
                return sedona_internal_err!(
                    "concat requires all geometry arrays to have the same sedona type"
                );
            }
        }

        let value_refs: Vec<&dyn Array> = geom_arrays
            .iter()
            .map(|g| g.geometry_array.as_ref())
            .collect();
        let geometry_array = arrow_concat(&value_refs)?;

        let total_len: usize = geom_arrays.iter().map(|g| g.rects.len()).sum();
        let mut rects = Vec::with_capacity(total_len);
        for geom in geom_arrays {
            rects.extend(geom.rects.iter().cloned());
        }

        let mut out = Self::try_new_with_rects(geometry_array, rects, sedona_type)?;
        out.distance = Self::concat_distance(geom_arrays)?;

        Ok(out)
    }

    /// Merge optional distance metadata across source geometry arrays using the provided
    /// array-combining function.
    fn merge_distance<F>(
        geom_arrays: &[&EvaluatedGeometryArray],
        build_array: F,
    ) -> Result<Option<ColumnarValue>>
    where
        F: FnOnce(&[&dyn Array]) -> Result<ArrayRef>,
    {
        let mut first_value: Option<&ColumnarValue> = None;
        let mut needs_array = false;
        let mut all_null = true;
        let mut first_scalar: Option<&ScalarValue> = None;

        for geom in geom_arrays {
            match &geom.distance {
                Some(value) => {
                    if first_value.is_none() {
                        first_value = Some(value);
                    }
                    match value {
                        ColumnarValue::Array(array) => {
                            needs_array = true;
                            if all_null && array.logical_null_count() != array.len() {
                                all_null = false;
                            }
                        }
                        ColumnarValue::Scalar(scalar) => {
                            if let Some(first) = first_scalar {
                                if first != scalar {
                                    needs_array = true;
                                }
                            } else {
                                first_scalar = Some(scalar);
                            }
                            if !scalar.is_null() {
                                all_null = false;
                            }
                        }
                    }
                }
                None => {
                    if first_value.is_some() && !all_null {
                        return sedona_internal_err!(
                            "Inconsistent distance metadata across batches"
                        );
                    }
                }
            }
        }

        if all_null {
            return Ok(None);
        }

        let Some(distance_value) = first_value else {
            return Ok(None);
        };

        if !needs_array {
            if let ColumnarValue::Scalar(value) = distance_value {
                return Ok(Some(ColumnarValue::Scalar(value.clone())));
            }
        }

        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(geom_arrays.len());
        for geom in geom_arrays {
            match &geom.distance {
                Some(ColumnarValue::Array(array)) => arrays.push(array.clone()),
                Some(ColumnarValue::Scalar(value)) => {
                    arrays.push(value.to_array_of_size(geom.geometry_array.len())?);
                }
                None => {
                    return sedona_internal_err!("Inconsistent distance metadata across batches");
                }
            }
        }

        let array_refs: Vec<&dyn Array> = arrays.iter().map(|a| a.as_ref()).collect();
        let array = build_array(&array_refs)?;
        Ok(Some(ColumnarValue::Array(array)))
    }

    /// Concatenate the optional distance metadata across source geometry arrays.
    pub(crate) fn concat_distance(
        geom_arrays: &[&EvaluatedGeometryArray],
    ) -> Result<Option<ColumnarValue>> {
        Self::merge_distance(geom_arrays, |arrays| Ok(arrow_concat(arrays)?))
    }

    /// Build a new `EvaluatedGeometryArray` by interleaving rows from the provided
    /// source arrays according to `indices`. Each `(batch_idx, row_idx)` pair
    /// identifies a source array and row.
    ///
    /// The rectangles are gathered directly from the source arrays (no
    /// recomputation), while the WKB references are recomputed from the
    /// interleaved Arrow array since their lifetimes are tied to it.
    pub fn interleave(
        geom_arrays: &[&EvaluatedGeometryArray],
        indices: &[(usize, usize)],
    ) -> Result<Self> {
        if geom_arrays.is_empty() {
            return sedona_internal_err!("interleave requires at least one geometry array");
        }

        let sedona_type = &geom_arrays[0].sedona_type;

        // Interleave the Arrow geometry arrays.
        let value_refs: Vec<&dyn Array> = geom_arrays
            .iter()
            .map(|g| g.geometry_array.as_ref())
            .collect();
        let geometry_array = arrow_interleave(&value_refs, indices)?;

        // Gather rects by index — no recomputation.
        let rects: Vec<Option<Rect<f32>>> = indices
            .iter()
            .map(|&(batch_idx, row_idx)| geom_arrays[batch_idx].rects[row_idx])
            .collect();

        let mut out = Self::try_new_with_rects(geometry_array, rects, sedona_type)?;

        // Interleave distance columns.
        out.distance = Self::interleave_distance(geom_arrays, indices)?;

        Ok(out)
    }

    /// Interleave the optional distance metadata across source geometry arrays.
    pub(crate) fn interleave_distance(
        geom_arrays: &[&EvaluatedGeometryArray],
        indices: &[(usize, usize)],
    ) -> Result<Option<ColumnarValue>> {
        Self::merge_distance(geom_arrays, |arrays| Ok(arrow_interleave(arrays, indices)?))
    }

    /// Type of geometry_array
    pub fn sedona_type(&self) -> &SedonaType {
        &self.sedona_type
    }

    /// Evaluated array of geometries
    pub fn geometry_array(&self) -> &ArrayRef {
        &self.geometry_array
    }

    /// Bounding rectangles of each element in geometry_array
    pub fn rects(&self) -> &[Option<Rect<f32>>] {
        &self.rects
    }

    /// Evaluated array of distances
    pub fn distance(&self) -> &Option<ColumnarValue> {
        &self.distance
    }

    /// Get the distance value for a specific row, if distances are present.
    pub fn distance_at(&self, row_idx: usize) -> Result<Option<f64>> {
        match &self.distance {
            Some(cv) => distance_value_at(cv, row_idx),
            None => Ok(None),
        }
    }

    /// Get the WKBs of the geometries in the geometry array.
    pub fn wkbs(&self) -> &Vec<Option<Wkb<'_>>> {
        // The returned WKBs are guaranteed to be valid for the lifetime of the GeometryBatchResult,
        // because the WKBs reference buffers inside `geometry_array`, which is guaranteed to be valid
        // for the lifetime of the GeometryBatchResult. We shorten the lifetime of the WKBs from 'static
        // to '_, so that the caller can use the WKBs without worrying about the lifetime.
        &self.wkbs
    }

    /// Get a single WKB
    pub fn wkb(&self, idx: usize) -> Option<&Wkb<'_>> {
        self.wkbs[idx].as_ref()
    }

    pub fn in_mem_size(&self) -> Result<usize> {
        let geom_array_size = get_array_memory_size(&self.geometry_array)?;

        let distance_in_mem_size = match &self.distance {
            Some(ColumnarValue::Array(array)) => get_array_memory_size(array)?,
            _ => 8,
        };

        // Note: this is not an accurate, because wkbs has inner Vecs. However, the size of inner vecs
        // should be small, so the inaccuracy does not matter too much.
        let wkb_vec_size = self.wkbs.allocated_size();

        Ok(geom_array_size + self.rects.allocated_size() + distance_in_mem_size + wkb_vec_size)
    }
}

/// Evaluator for a relation predicate.
#[derive(Debug)]
struct RelationOperandEvaluator {
    inner: RelationPredicate,
    evaluated_array_factory: Arc<dyn EvaluatedGeometryArrayFactory>,
}

impl RelationOperandEvaluator {
    pub fn new(
        inner: RelationPredicate,
        evaluated_array_factory: Arc<dyn EvaluatedGeometryArrayFactory>,
    ) -> Self {
        Self {
            inner,
            evaluated_array_factory,
        }
    }
}

/// Evaluator for a distance predicate.
#[derive(Debug)]
struct DistanceOperandEvaluator {
    inner: DistancePredicate,
    evaluated_array_factory: Arc<dyn EvaluatedGeometryArrayFactory>,
}

impl DistanceOperandEvaluator {
    pub fn new(
        inner: DistancePredicate,
        evaluated_array_factory: Arc<dyn EvaluatedGeometryArrayFactory>,
    ) -> Self {
        Self {
            inner,
            evaluated_array_factory,
        }
    }
}

fn evaluate_with_rects(
    batch: &RecordBatch,
    geom_expr: &Arc<dyn PhysicalExpr>,
    evaluated_array_factory: &dyn EvaluatedGeometryArrayFactory,
) -> Result<EvaluatedGeometryArray> {
    let geometry_columnar_value = geom_expr.evaluate(batch)?;
    let num_rows = batch.num_rows();
    let geometry_array = geometry_columnar_value.to_array(num_rows)?;
    let sedona_type =
        SedonaType::from_storage_field(geom_expr.return_field(&batch.schema())?.as_ref())?;
    evaluated_array_factory.try_new_evaluated_array(geometry_array, &sedona_type)
}

impl DistanceOperandEvaluator {
    fn evaluate_with_rects(
        &self,
        batch: &RecordBatch,
        geom_expr: &Arc<dyn PhysicalExpr>,
        side: JoinSide,
    ) -> Result<EvaluatedGeometryArray> {
        let mut result =
            evaluate_with_rects(batch, geom_expr, self.evaluated_array_factory.as_ref())?;

        let should_expand = match side {
            JoinSide::Left => self.inner.distance_side == JoinSide::Left,
            JoinSide::Right => self.inner.distance_side != JoinSide::Left,
            JoinSide::None => unreachable!(),
        };

        if !should_expand {
            return Ok(result);
        }

        // Expand the vec by distance
        let distance_columnar_value = self.inner.distance.evaluate(batch)?;
        // No timezone conversion needed for distance; pass None as cast_options explicitly.
        let distance_columnar_value = distance_columnar_value.cast_to(&DataType::Float64, None)?;
        match &distance_columnar_value {
            ColumnarValue::Scalar(ScalarValue::Float64(Some(distance))) => {
                result.rects.iter_mut().for_each(|rect_opt| {
                    if let Some(rect) = rect_opt {
                        expand_rect_in_place(rect, *distance);
                    };
                });
            }
            ColumnarValue::Scalar(ScalarValue::Float64(None)) => {
                // Distance expression evaluates to NULL, the resulting distance should be NULL as well.
                result.rects.clear();
            }
            ColumnarValue::Array(array) => {
                if let Some(array) = array.as_any().downcast_ref::<Float64Array>() {
                    for (geom_idx, rect_opt) in result.rects.iter_mut().enumerate() {
                        if !array.is_null(geom_idx) {
                            let dist = array.value(geom_idx);
                            if let Some(rect) = rect_opt {
                                expand_rect_in_place(rect, dist);
                            };
                        }
                    }
                } else {
                    return sedona_internal_err!("Distance columnar value is not a Float64Array");
                }
            }
            _ => {
                return sedona_internal_err!("Distance columnar value is not a Float64");
            }
        }

        result.distance = Some(distance_columnar_value);
        Ok(result)
    }
}

pub(crate) fn distance_value_at(
    distance_columnar_value: &ColumnarValue,
    i: usize,
) -> Result<Option<f64>> {
    match distance_columnar_value {
        ColumnarValue::Scalar(ScalarValue::Float64(dist_opt)) => Ok(*dist_opt),
        ColumnarValue::Array(array) => {
            if let Some(array) = array.as_any().downcast_ref::<Float64Array>() {
                if array.is_null(i) {
                    Ok(None)
                } else {
                    Ok(Some(array.value(i)))
                }
            } else {
                sedona_internal_err!("Distance columnar value is not a Float64Array")
            }
        }
        _ => sedona_internal_err!("Distance columnar value is not a Float64"),
    }
}

fn expand_rect_in_place(rect: &mut Rect<f32>, distance: f64) {
    let mut min = rect.min();
    let mut max = rect.max();
    let mut distance_f32 = distance as f32;
    // distance_f32 may be smaller than the original f64 value due to loss of precision.
    // We need to expand the rect using next_after to ensure that the rect expansion
    // is always inclusive, otherwise we may miss some query results.
    if (distance_f32 as f64) < distance {
        distance_f32 = distance_f32.next_after(f32::INFINITY);
    }
    min.x -= distance_f32;
    min.y -= distance_f32;
    max.x += distance_f32;
    max.y += distance_f32;
    rect.set_min(min);
    rect.set_max(max);
}

impl OperandEvaluator for DistanceOperandEvaluator {
    fn evaluate_build(&self, batch: &RecordBatch) -> Result<EvaluatedGeometryArray> {
        let geom_expr = self.build_side_expr()?;
        self.evaluate_with_rects(batch, &geom_expr, JoinSide::Left)
    }

    fn evaluate_probe(&self, batch: &RecordBatch) -> Result<EvaluatedGeometryArray> {
        let geom_expr = self.probe_side_expr()?;
        self.evaluate_with_rects(batch, &geom_expr, JoinSide::Right)
    }

    fn build_side_expr(&self) -> Result<Arc<dyn PhysicalExpr>> {
        Ok(Arc::clone(&self.inner.left))
    }

    fn probe_side_expr(&self) -> Result<Arc<dyn PhysicalExpr>> {
        Ok(Arc::clone(&self.inner.right))
    }
}

impl OperandEvaluator for RelationOperandEvaluator {
    fn evaluate_build(&self, batch: &RecordBatch) -> Result<EvaluatedGeometryArray> {
        let geom_expr = self.build_side_expr()?;
        evaluate_with_rects(batch, &geom_expr, self.evaluated_array_factory.as_ref())
    }

    fn evaluate_probe(&self, batch: &RecordBatch) -> Result<EvaluatedGeometryArray> {
        let geom_expr = self.probe_side_expr()?;
        evaluate_with_rects(batch, &geom_expr, self.evaluated_array_factory.as_ref())
    }

    fn build_side_expr(&self) -> Result<Arc<dyn PhysicalExpr>> {
        Ok(Arc::clone(&self.inner.left))
    }

    fn probe_side_expr(&self) -> Result<Arc<dyn PhysicalExpr>> {
        Ok(Arc::clone(&self.inner.right))
    }
}

/// KNN operand evaluator for evaluating the KNN predicate.
#[derive(Debug)]
struct KNNOperandEvaluator {
    inner: KNNPredicate,
    evaluated_array_factory: Arc<dyn EvaluatedGeometryArrayFactory>,
}

impl KNNOperandEvaluator {
    fn new(
        inner: KNNPredicate,
        evaluated_array_factory: Arc<dyn EvaluatedGeometryArrayFactory>,
    ) -> Self {
        Self {
            inner,
            evaluated_array_factory,
        }
    }
}

impl OperandEvaluator for KNNOperandEvaluator {
    fn evaluate_build(&self, batch: &RecordBatch) -> Result<EvaluatedGeometryArray> {
        let geom_expr = self.build_side_expr()?;
        evaluate_with_rects(batch, &geom_expr, self.evaluated_array_factory.as_ref())
    }

    fn evaluate_probe(&self, batch: &RecordBatch) -> Result<EvaluatedGeometryArray> {
        let geom_expr = self.probe_side_expr()?;
        evaluate_with_rects(batch, &geom_expr, self.evaluated_array_factory.as_ref())
    }

    fn build_side_expr(&self) -> Result<Arc<dyn PhysicalExpr>> {
        // For KNN, the right side (objects/candidates) is the build side
        Ok(Arc::clone(&self.inner.right))
    }

    fn probe_side_expr(&self) -> Result<Arc<dyn PhysicalExpr>> {
        // For KNN, the left side (queries) is the probe side
        Ok(Arc::clone(&self.inner.left))
    }
}

#[cfg(test)]
mod test {
    use arrow_array::BinaryArray;
    use sedona_geometry::wkb_factory::wkb_point;
    use sedona_schema::datatypes::WKB_GEOMETRY;

    use super::*;

    fn make_geom_array_with_distance(
        wkbs: Vec<Vec<u8>>,
        distance: Option<ColumnarValue>,
    ) -> Result<EvaluatedGeometryArray> {
        let geom_array: ArrayRef = Arc::new(BinaryArray::from(
            wkbs.iter()
                .map(|wkb| Some(wkb.as_slice()))
                .collect::<Vec<_>>(),
        ));
        let mut geom = EvaluatedGeometryArray::try_new(geom_array, &WKB_GEOMETRY)?;
        geom.distance = distance;
        Ok(geom)
    }

    #[test]
    fn concat_empty_input_errors() {
        let result = EvaluatedGeometryArray::concat(&[]);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e
                .to_string()
                .contains("concat requires at least one geometry array"));
        }
    }

    #[test]
    fn concat_preserves_order_and_distance() -> Result<()> {
        use arrow_array::Float64Array;

        let wkbs1 = vec![
            wkb_point((10.0, 10.0)).unwrap(),
            wkb_point((20.0, 20.0)).unwrap(),
        ];
        let wkbs2 = vec![wkb_point((30.0, 30.0)).unwrap()];

        let array: ArrayRef = Arc::new(Float64Array::from(vec![10.0]));
        let geom1 = make_geom_array_with_distance(
            wkbs1,
            Some(ColumnarValue::Scalar(ScalarValue::Float64(Some(5.0)))),
        )?;
        let geom2 = make_geom_array_with_distance(wkbs2, Some(ColumnarValue::Array(array)))?;

        let geom_arrays = vec![&geom1, &geom2];
        let result = EvaluatedGeometryArray::concat(&geom_arrays)?;

        assert_eq!(result.geometry_array().len(), 3);
        assert_eq!(result.rects().len(), 3);
        assert_eq!(result.rects()[0].unwrap().min().x, 10.0);
        assert_eq!(result.rects()[1].unwrap().min().x, 20.0);
        assert_eq!(result.rects()[2].unwrap().min().x, 30.0);

        let distance = result.distance().as_ref().unwrap();
        match distance {
            ColumnarValue::Array(array) => {
                let float_array = array.as_any().downcast_ref::<Float64Array>().unwrap();
                assert_eq!(float_array.len(), 3);
                assert_eq!(float_array.value(0), 5.0);
                assert_eq!(float_array.value(1), 5.0);
                assert_eq!(float_array.value(2), 10.0);
            }
            _ => panic!("expected concatenated distance array"),
        }

        Ok(())
    }

    #[test]
    fn concat_uniform_scalar_distance_stays_scalar() -> Result<()> {
        let wkbs1 = vec![wkb_point((10.0, 10.0)).unwrap()];
        let wkbs2 = vec![wkb_point((20.0, 20.0)).unwrap()];

        let scalar = ScalarValue::Float64(Some(7.5));
        let geom1 =
            make_geom_array_with_distance(wkbs1, Some(ColumnarValue::Scalar(scalar.clone())))?;
        let geom2 =
            make_geom_array_with_distance(wkbs2, Some(ColumnarValue::Scalar(scalar.clone())))?;

        let geom_arrays = vec![&geom1, &geom2];
        let result = EvaluatedGeometryArray::concat(&geom_arrays)?;

        assert_eq!(result.geometry_array().len(), 2);
        assert!(matches!(result.distance(), Some(ColumnarValue::Scalar(_))));
        if let Some(ColumnarValue::Scalar(value)) = result.distance() {
            assert_eq!(*value, scalar);
        }
        Ok(())
    }

    #[test]
    fn concat_inconsistent_distance_metadata_errors() -> Result<()> {
        let wkbs1 = vec![wkb_point((10.0, 10.0)).unwrap()];
        let wkbs2 = vec![wkb_point((20.0, 20.0)).unwrap()];

        let geom1 = make_geom_array_with_distance(
            wkbs1,
            Some(ColumnarValue::Scalar(ScalarValue::Float64(Some(5.0)))),
        )?;
        let geom2 = make_geom_array_with_distance(wkbs2, None)?;

        let geom_arrays = vec![&geom1, &geom2];
        let result = EvaluatedGeometryArray::concat(&geom_arrays);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Inconsistent distance metadata"));
        }
        Ok(())
    }

    #[test]
    fn interleave_distance_none() -> Result<()> {
        let wkbs1 = vec![
            wkb_point((10.0, 10.0)).unwrap(),
            wkb_point((20.0, 20.0)).unwrap(),
        ];
        let wkbs2 = vec![wkb_point((30.0, 30.0)).unwrap()];

        let geom1 = make_geom_array_with_distance(wkbs1, None)?;
        let geom2 = make_geom_array_with_distance(wkbs2, None)?;

        let geom_arrays = vec![&geom1, &geom2];
        let assignments = vec![(0, 0), (1, 0), (0, 1)];

        let result = EvaluatedGeometryArray::interleave_distance(&geom_arrays, &assignments)?;
        assert!(result.is_none());
        Ok(())
    }

    #[test]
    fn interleave_distance_uniform_scalar() -> Result<()> {
        let wkbs1 = vec![
            wkb_point((10.0, 10.0)).unwrap(),
            wkb_point((20.0, 20.0)).unwrap(),
        ];
        let wkbs2 = vec![wkb_point((30.0, 30.0)).unwrap()];

        let scalar = ScalarValue::Float64(Some(5.0));
        let geom1 =
            make_geom_array_with_distance(wkbs1, Some(ColumnarValue::Scalar(scalar.clone())))?;
        let geom2 =
            make_geom_array_with_distance(wkbs2, Some(ColumnarValue::Scalar(scalar.clone())))?;

        let geom_arrays = vec![&geom1, &geom2];
        let assignments = vec![(0, 0), (1, 0), (0, 1)];

        let result = EvaluatedGeometryArray::interleave_distance(&geom_arrays, &assignments)?;
        assert!(matches!(result, Some(ColumnarValue::Scalar(_))));
        if let Some(ColumnarValue::Scalar(value)) = result {
            assert_eq!(value, ScalarValue::Float64(Some(5.0)));
        }
        Ok(())
    }

    #[test]
    fn interleave_distance_different_scalars() -> Result<()> {
        use arrow_array::Float64Array;

        let wkbs1 = vec![
            wkb_point((10.0, 10.0)).unwrap(),
            wkb_point((20.0, 20.0)).unwrap(),
        ];
        let wkbs2 = vec![wkb_point((30.0, 30.0)).unwrap()];

        let scalar1 = ScalarValue::Float64(Some(5.0));
        let scalar2 = ScalarValue::Float64(Some(10.0));
        let geom1 = make_geom_array_with_distance(wkbs1, Some(ColumnarValue::Scalar(scalar1)))?;
        let geom2 = make_geom_array_with_distance(wkbs2, Some(ColumnarValue::Scalar(scalar2)))?;

        let geom_arrays = vec![&geom1, &geom2];
        let assignments = vec![(0, 0), (1, 0), (0, 1)];

        let result = EvaluatedGeometryArray::interleave_distance(&geom_arrays, &assignments)?;
        assert!(matches!(result, Some(ColumnarValue::Array(_))));
        if let Some(ColumnarValue::Array(array)) = result {
            let float_array = array.as_any().downcast_ref::<Float64Array>().unwrap();
            assert_eq!(float_array.len(), 3);
            assert_eq!(float_array.value(0), 5.0);
            assert_eq!(float_array.value(1), 10.0);
            assert_eq!(float_array.value(2), 5.0);
        }
        Ok(())
    }

    #[test]
    fn interleave_distance_arrays() -> Result<()> {
        use arrow_array::Float64Array;

        let wkbs1 = vec![
            wkb_point((10.0, 10.0)).unwrap(),
            wkb_point((20.0, 20.0)).unwrap(),
        ];
        let wkbs2 = vec![wkb_point((30.0, 30.0)).unwrap()];

        let array1: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 2.0]));
        let array2: ArrayRef = Arc::new(Float64Array::from(vec![3.0]));
        let geom1 = make_geom_array_with_distance(wkbs1, Some(ColumnarValue::Array(array1)))?;
        let geom2 = make_geom_array_with_distance(wkbs2, Some(ColumnarValue::Array(array2)))?;

        let geom_arrays = vec![&geom1, &geom2];
        let assignments = vec![(0, 0), (1, 0), (0, 1)];

        let result = EvaluatedGeometryArray::interleave_distance(&geom_arrays, &assignments)?;
        assert!(matches!(result, Some(ColumnarValue::Array(_))));
        if let Some(ColumnarValue::Array(array)) = result {
            let float_array = array.as_any().downcast_ref::<Float64Array>().unwrap();
            assert_eq!(float_array.len(), 3);
            assert_eq!(float_array.value(0), 1.0);
            assert_eq!(float_array.value(1), 3.0);
            assert_eq!(float_array.value(2), 2.0);
        }
        Ok(())
    }

    #[test]
    fn interleave_distance_mixed_scalar_and_array() -> Result<()> {
        use arrow_array::Float64Array;

        let wkbs1 = vec![
            wkb_point((10.0, 10.0)).unwrap(),
            wkb_point((20.0, 20.0)).unwrap(),
        ];
        let wkbs2 = vec![wkb_point((30.0, 30.0)).unwrap()];

        let scalar = ScalarValue::Float64(Some(5.0));
        let array: ArrayRef = Arc::new(Float64Array::from(vec![10.0]));
        let geom1 = make_geom_array_with_distance(wkbs1, Some(ColumnarValue::Scalar(scalar)))?;
        let geom2 = make_geom_array_with_distance(wkbs2, Some(ColumnarValue::Array(array)))?;

        let geom_arrays = vec![&geom1, &geom2];
        let assignments = vec![(0, 0), (1, 0), (0, 1)];

        let result = EvaluatedGeometryArray::interleave_distance(&geom_arrays, &assignments)?;
        assert!(matches!(result, Some(ColumnarValue::Array(_))));
        if let Some(ColumnarValue::Array(array)) = result {
            let float_array = array.as_any().downcast_ref::<Float64Array>().unwrap();
            assert_eq!(float_array.len(), 3);
            assert_eq!(float_array.value(0), 5.0);
            assert_eq!(float_array.value(1), 10.0);
            assert_eq!(float_array.value(2), 5.0);
        }
        Ok(())
    }

    #[test]
    fn interleave_distance_inconsistent_metadata() -> Result<()> {
        let wkbs1 = vec![wkb_point((10.0, 10.0)).unwrap()];
        let wkbs2 = vec![wkb_point((20.0, 20.0)).unwrap()];

        let scalar = ScalarValue::Float64(Some(5.0));
        let geom1 = make_geom_array_with_distance(wkbs1, Some(ColumnarValue::Scalar(scalar)))?;
        let geom2 = make_geom_array_with_distance(wkbs2, None)?;

        let geom_arrays = vec![&geom1, &geom2];
        let assignments = vec![(0, 0), (1, 0)];

        let result = EvaluatedGeometryArray::interleave_distance(&geom_arrays, &assignments);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Inconsistent distance metadata"));
        }
        Ok(())
    }

    #[test]
    fn interleave_distance_mixed_none_and_null() -> Result<()> {
        use arrow_array::Float64Array;

        let wkbs1 = vec![wkb_point((10.0, 10.0)).unwrap()];
        let wkbs2 = vec![wkb_point((20.0, 20.0)).unwrap()];
        let wkbs3 = vec![wkb_point((30.0, 30.0)).unwrap()];

        let null_array = Arc::new(Float64Array::new_null(1));
        let ega1 = make_geom_array_with_distance(wkbs1, Some(ColumnarValue::Array(null_array)))?;

        let null_scalar = ScalarValue::Float64(None);
        let ega2 = make_geom_array_with_distance(wkbs2, Some(ColumnarValue::Scalar(null_scalar)))?;

        let ega3 = make_geom_array_with_distance(wkbs3, None)?;

        let vec_ega = vec![&ega1, &ega2, &ega3];
        let assignments = vec![(0, 0), (1, 0), (2, 0)];

        let result = EvaluatedGeometryArray::interleave_distance(&vec_ega, &assignments)?;
        assert!(result.is_none());
        Ok(())
    }
}
