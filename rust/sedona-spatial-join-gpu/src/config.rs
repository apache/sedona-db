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
use datafusion::logical_expr::JoinType;
use datafusion_physical_plan::joins::utils::JoinFilter;

#[derive(Debug, Clone)]
pub struct GpuSpatialJoinConfig {
    /// Join type (Inner, Left, Right, Full)
    pub join_type: JoinType,

    /// Left geometry column information
    pub left_geom_column: GeometryColumnInfo,

    /// Right geometry column information
    pub right_geom_column: GeometryColumnInfo,

    /// Spatial predicate for the join
    pub predicate: GpuSpatialPredicate,

    /// GPU device ID to use
    pub device_id: i32,

    /// Batch size for GPU processing
    pub batch_size: usize,

    /// Additional join filters (from WHERE clause)
    pub additional_filters: Option<JoinFilter>,

    /// Maximum GPU memory to use (bytes, None = unlimited)
    pub max_memory: Option<usize>,

    /// Fall back to CPU if GPU fails
    pub fallback_to_cpu: bool,
}

#[derive(Debug, Clone)]
pub struct GeometryColumnInfo {
    /// Column name
    pub name: String,

    /// Column index in schema
    pub index: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum GpuSpatialPredicate {
    /// Relation predicate (Intersects, Contains, etc.)
    Relation(sedona_libgpuspatial::SpatialPredicate),
    // Future extensions: Distance, KNN
}

impl Default for GpuSpatialJoinConfig {
    fn default() -> Self {
        Self {
            join_type: JoinType::Inner,
            left_geom_column: GeometryColumnInfo {
                name: "geometry".to_string(),
                index: 0,
            },
            right_geom_column: GeometryColumnInfo {
                name: "geometry".to_string(),
                index: 0,
            },
            predicate: GpuSpatialPredicate::Relation(
                sedona_libgpuspatial::SpatialPredicate::Intersects,
            ),
            device_id: 0,
            batch_size: 8192,
            additional_filters: None,
            max_memory: None,
            fallback_to_cpu: true,
        }
    }
}
