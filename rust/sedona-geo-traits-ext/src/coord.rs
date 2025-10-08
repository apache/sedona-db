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
//! Extend CoordTrait traits for the `geo-traits` crate

use geo_traits::{CoordTrait, UnimplementedCoord};
use geo_types::{Coord, CoordNum};

use crate::{CoordTag, GeoTraitExtWithTypeTag};

/// Extension methods that bridge [`CoordTrait`] with concrete `geo-types` helpers.
pub trait CoordTraitExt: CoordTrait + GeoTraitExtWithTypeTag<Tag = CoordTag>
where
    <Self as CoordTrait>::T: CoordNum,
{
    #[inline]
    /// Converts this coordinate into the concrete [`geo_types::Coord`].
    fn geo_coord(&self) -> Coord<Self::T> {
        Coord {
            x: self.x(),
            y: self.y(),
        }
    }
}

impl<T> CoordTraitExt for Coord<T>
where
    T: CoordNum,
{
    fn geo_coord(&self) -> Coord<T> {
        *self
    }
}

impl<T: CoordNum> GeoTraitExtWithTypeTag for Coord<T> {
    type Tag = CoordTag;
}

impl<T> CoordTraitExt for &Coord<T>
where
    T: CoordNum,
{
    fn geo_coord(&self) -> Coord<T> {
        **self
    }
}

impl<T: CoordNum> GeoTraitExtWithTypeTag for &Coord<T> {
    type Tag = CoordTag;
}

impl<T> CoordTraitExt for UnimplementedCoord<T> where T: CoordNum {}

impl<T: CoordNum> GeoTraitExtWithTypeTag for UnimplementedCoord<T> {
    type Tag = CoordTag;
}
