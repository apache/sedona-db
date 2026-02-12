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

use std::os::raw::c_uint;

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum GpuSpatialRelationPredicate {
    Equals,
    Disjoint,
    Touches,
    Contains,
    Covers,
    Intersects,
    Within,
    CoveredBy,
}

impl GpuSpatialRelationPredicate {
    /// Internal helper to convert the Rust enum to the C-compatible integer.
    pub(crate) fn as_c_uint(self) -> c_uint {
        match self {
            Self::Equals => 0,
            Self::Disjoint => 1,
            Self::Touches => 2,
            Self::Contains => 3,
            Self::Covers => 4,
            Self::Intersects => 5,
            Self::Within => 6,
            Self::CoveredBy => 7,
        }
    }
}
impl std::fmt::Display for GpuSpatialRelationPredicate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuSpatialRelationPredicate::Equals => write!(f, "equals"),
            GpuSpatialRelationPredicate::Disjoint => write!(f, "disjoint"),
            GpuSpatialRelationPredicate::Touches => write!(f, "touches"),
            GpuSpatialRelationPredicate::Contains => write!(f, "contains"),
            GpuSpatialRelationPredicate::Covers => write!(f, "covers"),
            GpuSpatialRelationPredicate::Intersects => write!(f, "intersects"),
            GpuSpatialRelationPredicate::Within => write!(f, "within"),
            GpuSpatialRelationPredicate::CoveredBy => write!(f, "coveredby"),
        }
    }
}
