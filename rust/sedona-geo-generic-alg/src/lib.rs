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
//! Crate root for generic geometry algorithms ported from `geo`.
//!
//! Substantial portions of this crate (algorithm modules, trait patterns, and API surface) are
//! ported (and contain copied code) from the `geo` crate at commit
//! `5d667f844716a3d0a17aa60bc0a58528cb5808c3`:
//! <https://github.com/georust/geo/tree/5d667f844716a3d0a17aa60bc0a58528cb5808c3/geo/src>.
//! The original upstream project is dual-licensed under Apache-2.0 or MIT; the copied/ported code
//! here is used under the Apache-2.0 license consistent with this repository.
//! This top-level file orchestrates module exposure and numeric traits mirroring upstream design.
pub use crate::algorithm::*;
use std::cmp::Ordering;

pub use geo_types::{coord, line_string, point, polygon, wkt, CoordFloat, CoordNum};

pub mod geometry;
pub use geometry::*;

/// This module includes all the functions of geometric calculations
pub mod algorithm;
mod utils;
use crate::kernels::{RobustKernel, SimpleKernel};

#[cfg(test)]
#[macro_use]
extern crate approx;

#[cfg(test)]
#[macro_use]
extern crate log;

/// A prelude which re-exports the traits for manipulating objects in this
/// crate. Typically imported with `use geo::prelude::*`.
pub mod prelude {
    pub use crate::algorithm::*;
}

/// A common numeric trait used for geo algorithms
///
/// Different numeric types have different tradeoffs. `geo` strives to utilize generics to allow
/// users to choose their numeric types. If you are writing a function which you'd like to be
/// generic over all the numeric types supported by geo, you probably want to constrain
/// your function input to `GeoFloat`. For methods which work for integers, and not just floating
/// point, see [`GeoNum`].
pub trait GeoFloat:
    GeoNum + num_traits::Float + num_traits::Signed + num_traits::Bounded + float_next_after::NextAfter
{
}
impl<T> GeoFloat for T where
    T: GeoNum
        + num_traits::Float
        + num_traits::Signed
        + num_traits::Bounded
        + float_next_after::NextAfter
{
}

/// A trait for methods which work for both integers **and** floating point
pub trait GeoNum: CoordNum {
    type Ker: Kernel<Self>;

    /// Return the ordering between self and other.
    ///
    /// For integers, this should behave just like [`Ord`].
    ///
    /// For floating point numbers, unlike the standard partial comparison between floating point numbers, this comparison
    /// always produces an ordering.
    ///
    /// See [f64::total_cmp](https://doc.rust-lang.org/src/core/num/f64.rs.html#1432) for details.
    fn total_cmp(&self, other: &Self) -> Ordering;
}

macro_rules! impl_geo_num_for_float {
    ($t: ident) => {
        impl GeoNum for $t {
            type Ker = RobustKernel;
            fn total_cmp(&self, other: &Self) -> Ordering {
                self.total_cmp(other)
            }
        }
    };
}
macro_rules! impl_geo_num_for_int {
    ($t: ident) => {
        impl GeoNum for $t {
            type Ker = SimpleKernel;
            fn total_cmp(&self, other: &Self) -> Ordering {
                self.cmp(other)
            }
        }
    };
}

// This is the list of primitives that we support.
impl_geo_num_for_float!(f32);
impl_geo_num_for_float!(f64);
impl_geo_num_for_int!(i16);
impl_geo_num_for_int!(i32);
impl_geo_num_for_int!(i64);
impl_geo_num_for_int!(i128);
impl_geo_num_for_int!(isize);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_ord_float() {
        assert_eq!(GeoNum::total_cmp(&3.0f64, &2.0f64), Ordering::Greater);
        assert_eq!(GeoNum::total_cmp(&2.0f64, &2.0f64), Ordering::Equal);
        assert_eq!(GeoNum::total_cmp(&1.0f64, &2.0f64), Ordering::Less);
        assert_eq!(GeoNum::total_cmp(&1.0f64, &f64::NAN), Ordering::Less);
        assert_eq!(GeoNum::total_cmp(&f64::NAN, &f64::NAN), Ordering::Equal);
        assert_eq!(GeoNum::total_cmp(&f64::INFINITY, &f64::NAN), Ordering::Less);
    }

    #[test]
    fn total_ord_int() {
        assert_eq!(GeoNum::total_cmp(&3i32, &2i32), Ordering::Greater);
        assert_eq!(GeoNum::total_cmp(&2i32, &2i32), Ordering::Equal);
        assert_eq!(GeoNum::total_cmp(&1i32, &2i32), Ordering::Less);
    }

    #[test]
    fn numeric_types() {
        let _n_i16 = Point::new(1i16, 2i16);
        let _n_i32 = Point::new(1i32, 2i32);
        let _n_i64 = Point::new(1i64, 2i64);
        let _n_i128 = Point::new(1i128, 2i128);
        let _n_isize = Point::new(1isize, 2isize);
        let _n_f32 = Point::new(1.0f32, 2.0f32);
        let _n_f64 = Point::new(1.0f64, 2.0f64);
    }
}
