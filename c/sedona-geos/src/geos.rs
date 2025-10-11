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
use geos::{GResult, Geom, Geometry};

#[derive(Debug, Default)]
pub struct GeosPredicate<Op> {
    _op: Op,
}

pub trait BinaryPredicate: std::fmt::Debug + Default {
    fn evaluate(lhs: &Geometry, rhs: &Geometry) -> GResult<bool>;
}

/// Check for topological equality
#[derive(Debug, Default)]
pub struct Equals {}
impl BinaryPredicate for Equals {
    fn evaluate(lhs: &Geometry, rhs: &Geometry) -> GResult<bool> {
        lhs.equals(rhs)
    }
}

/// Check if two geometries intersect
#[derive(Debug, Default)]
pub struct Intersects {}
impl BinaryPredicate for Intersects {
    fn evaluate(lhs: &Geometry, rhs: &Geometry) -> GResult<bool> {
        lhs.intersects(rhs)
    }
}

/// Check if two geometries do not intersect
#[derive(Debug, Default)]
pub struct Disjoint {}
impl BinaryPredicate for Disjoint {
    fn evaluate(lhs: &Geometry, rhs: &Geometry) -> GResult<bool> {
        lhs.disjoint(rhs)
    }
}

/// Check if the left geometry contains the right
#[derive(Debug, Default)]
pub struct Contains {}
impl BinaryPredicate for Contains {
    fn evaluate(lhs: &Geometry, rhs: &Geometry) -> GResult<bool> {
        lhs.contains(rhs)
    }
}

/// Check if the right geometry contains the left
#[derive(Debug, Default)]
pub struct Within {}
impl BinaryPredicate for Within {
    fn evaluate(lhs: &Geometry, rhs: &Geometry) -> GResult<bool> {
        lhs.within(rhs)
    }
}

/// Check if the left geometry covers the right geometry
#[derive(Debug, Default)]
pub struct Covers {}
impl BinaryPredicate for Covers {
    fn evaluate(lhs: &Geometry, rhs: &Geometry) -> GResult<bool> {
        lhs.covers(rhs)
    }
}

/// Check if the right geometry covers the left geometry
#[derive(Debug, Default)]
pub struct CoveredBy {}
impl BinaryPredicate for CoveredBy {
    fn evaluate(lhs: &Geometry, rhs: &Geometry) -> GResult<bool> {
        lhs.covered_by(rhs)
    }
}

/// Check if the geometries touch but do not otherwise intersect
#[derive(Debug, Default)]
pub struct Touches {}
impl BinaryPredicate for Touches {
    fn evaluate(lhs: &Geometry, rhs: &Geometry) -> GResult<bool> {
        lhs.touches(rhs)
    }
}

/// Check if the geometries cross
#[derive(Debug, Default)]
pub struct Crosses {}
impl BinaryPredicate for Crosses {
    fn evaluate(lhs: &Geometry, rhs: &Geometry) -> GResult<bool> {
        lhs.crosses(rhs)
    }
}

/// Check if the geometries overlap
#[derive(Debug, Default)]
pub struct Overlaps {}
impl BinaryPredicate for Overlaps {
    fn evaluate(lhs: &Geometry, rhs: &Geometry) -> GResult<bool> {
        lhs.overlaps(rhs)
    }
}
