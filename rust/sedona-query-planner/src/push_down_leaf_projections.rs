// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Compatibility version of DataFusion's `PushDownLeafProjections` rule.
//!
//! DataFusion PR #22620 made `Unnest` a semantic barrier for leaf projection
//! pushdown. DataFusion 54.1 predates that fix and can push a field access on
//! the post-unnest value below `Unnest`, where the same-named input still has
//! list type. Remove this module after upgrading to a DataFusion release that
//! contains <https://github.com/apache/datafusion/pull/22620>.
//! Removal is tracked by <https://github.com/apache/sedona-db/issues/1232>.

use datafusion_common::{tree_node::Transformed, Result};
use datafusion_expr::LogicalPlan;
use datafusion_optimizer::{
    extract_leaf_expressions::PushDownLeafProjections as DataFusionPushDownLeafProjections,
    ApplyOrder, OptimizerConfig, OptimizerRule,
};

/// The post-#22620 leaf projection rule, backported for DataFusion 54.1.
/// See <https://github.com/apache/sedona-db/issues/1232>.
#[derive(Debug, Default)]
pub struct PushDownLeafProjections {
    inner: DataFusionPushDownLeafProjections,
}

impl PushDownLeafProjections {
    pub fn new() -> Self {
        Self::default()
    }
}

impl OptimizerRule for PushDownLeafProjections {
    fn name(&self) -> &str {
        // Deliberately identical: registration replaces DataFusion's rule by name.
        "push_down_leaf_projections"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::TopDown)
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        // PR #22620 prevents leaf expressions from crossing Unnest because its
        // output may have the same column name as its input but a different
        // value and type. Since DataFusion's pushdown helpers are private, stop
        // delegation for any projection whose pushdown path could reach Unnest.
        if contains_unnest(&plan) {
            return Ok(Transformed::no(plan));
        }

        self.inner.rewrite(plan, config)
    }
}

fn contains_unnest(plan: &LogicalPlan) -> bool {
    matches!(plan, LogicalPlan::Unnest(_)) || plan.inputs().into_iter().any(contains_unnest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_datafusion_rule_name() {
        assert_eq!(
            PushDownLeafProjections::new().name(),
            "push_down_leaf_projections"
        );
    }
}
