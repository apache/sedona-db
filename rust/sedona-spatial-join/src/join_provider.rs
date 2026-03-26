use arrow_array::ArrayRef;
use arrow_schema::SchemaRef;
use datafusion_common::Result;
use datafusion_expr::JoinType;
use sedona_common::SpatialJoinOptions;
use sedona_schema::datatypes::SedonaType;

use crate::{
    index::{
        spatial_index_builder::{SpatialIndexBuilder, SpatialJoinBuildMetrics},
        DefaultSpatialIndexBuilder,
    },
    operand_evaluator::EvaluatedGeometryArray,
    SpatialPredicate,
};

pub(crate) trait SpatialJoinProvider: std::fmt::Debug + Send + Sync {
    fn try_new_spatial_index_builder(
        &self,
        schema: SchemaRef,
        spatial_predicate: SpatialPredicate,
        options: SpatialJoinOptions,
        join_type: JoinType,
        probe_threads_count: usize,
        metrics: SpatialJoinBuildMetrics,
    ) -> Result<Box<dyn SpatialIndexBuilder>>;
    fn try_new_evaluated_array(
        &self,
        geometry_array: ArrayRef,
        sedona_type: &SedonaType,
    ) -> Result<EvaluatedGeometryArray>;
}

#[derive(Debug)]
pub(crate) struct DefaultSpatialJoinEvaluator;

impl SpatialJoinProvider for DefaultSpatialJoinEvaluator {
    fn try_new_spatial_index_builder(
        &self,
        schema: SchemaRef,
        spatial_predicate: SpatialPredicate,
        options: SpatialJoinOptions,
        join_type: JoinType,
        probe_threads_count: usize,
        metrics: SpatialJoinBuildMetrics,
    ) -> Result<Box<dyn SpatialIndexBuilder>> {
        let builder = DefaultSpatialIndexBuilder::new(
            schema,
            spatial_predicate,
            options,
            join_type,
            probe_threads_count,
            metrics,
        )?;
        Ok(Box::new(builder))
    }

    fn try_new_evaluated_array(
        &self,
        geometry_array: ArrayRef,
        sedona_type: &SedonaType,
    ) -> Result<EvaluatedGeometryArray> {
        EvaluatedGeometryArray::try_new(geometry_array, sedona_type)
    }
}
