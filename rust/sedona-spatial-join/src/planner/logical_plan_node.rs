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

use std::cmp::Ordering;
use std::fmt;
use std::sync::Arc;

use datafusion_common::{DFSchemaRef, NullEquality, Result};
use datafusion_expr::logical_plan::UserDefinedLogicalNodeCore;
use datafusion_expr::{Expr, JoinConstraint, JoinType, LogicalPlan};
use sedona_common::sedona_internal_err;

/// Logical extension node used as a planning hook for spatial joins.
///
/// Carries a join's inputs and filter expression so the physical planner can recognize and plan
/// a `SpatialJoinExec`.
#[derive(PartialEq, Eq, Hash)]
pub(crate) struct SpatialJoinPlanNode {
    pub left: LogicalPlan,
    pub right: LogicalPlan,
    pub join_type: JoinType,
    pub filter: Expr,
    pub schema: DFSchemaRef,
    pub join_constraint: JoinConstraint,
    pub null_equality: NullEquality,
}

// Manual implementation needed because of `schema` field. Comparison excludes this field.
// See https://github.com/apache/datafusion/blob/52.1.0/datafusion/expr/src/logical_plan/plan.rs#L3886
impl PartialOrd for SpatialJoinPlanNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        #[derive(PartialEq, PartialOrd)]
        struct ComparableJoin<'a> {
            pub left: &'a LogicalPlan,
            pub right: &'a LogicalPlan,
            pub filter: &'a Expr,
            pub join_type: &'a JoinType,
            pub join_constraint: &'a JoinConstraint,
            pub null_equality: &'a NullEquality,
        }
        let comparable_self = ComparableJoin {
            left: &self.left,
            right: &self.right,
            filter: &self.filter,
            join_type: &self.join_type,
            join_constraint: &self.join_constraint,
            null_equality: &self.null_equality,
        };
        let comparable_other = ComparableJoin {
            left: &other.left,
            right: &other.right,
            filter: &other.filter,
            join_type: &other.join_type,
            join_constraint: &other.join_constraint,
            null_equality: &other.null_equality,
        };
        comparable_self
            .partial_cmp(&comparable_other)
            // TODO (https://github.com/apache/datafusion/issues/17477) avoid recomparing all fields
            .filter(|cmp| *cmp != Ordering::Equal || self == other)
    }
}

impl fmt::Debug for SpatialJoinPlanNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        UserDefinedLogicalNodeCore::fmt_for_explain(self, f)
    }
}

impl UserDefinedLogicalNodeCore for SpatialJoinPlanNode {
    fn name(&self) -> &str {
        "SpatialJoin"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.left, &self.right]
    }

    fn schema(&self) -> &DFSchemaRef {
        &self.schema
    }

    fn expressions(&self) -> Vec<Expr> {
        vec![self.filter.clone()]
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "SpatialJoin: join_type={:?}, filter={}",
            self.join_type, self.filter
        )
    }

    fn necessary_children_exprs(&self, _output_columns: &[usize]) -> Option<Vec<Vec<usize>>> {
        // Request all columns from both children. This ensures the optimizer
        // recurses into children while preserving all columns needed by the
        // join filter and output schema.
        let left_indices: Vec<usize> = (0..self.left.schema().fields().len()).collect();
        let right_indices: Vec<usize> = (0..self.right.schema().fields().len()).collect();
        Some(vec![left_indices, right_indices])
    }

    fn with_exprs_and_inputs(
        &self,
        mut exprs: Vec<Expr>,
        mut inputs: Vec<LogicalPlan>,
    ) -> Result<Self> {
        if exprs.len() != 1 {
            return sedona_internal_err!("SpatialJoinPlanNode expects 1 expr");
        }
        if inputs.len() != 2 {
            return sedona_internal_err!("SpatialJoinPlanNode expects 2 inputs");
        }
        Ok(Self {
            left: inputs.swap_remove(0),
            right: inputs.swap_remove(0),
            join_type: self.join_type,
            filter: exprs.swap_remove(0),
            schema: Arc::clone(&self.schema),
            join_constraint: self.join_constraint,
            null_equality: self.null_equality,
        })
    }
}
