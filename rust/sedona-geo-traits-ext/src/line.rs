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
// Extend LineTrait traits for the `geo-traits` crate

use geo_traits::{GeometryTrait, LineTrait, UnimplementedLine};
use geo_types::{CoordNum, Line};

use crate::{CoordTraitExt, GeoTraitExtWithTypeTag, LineTag};

/// Extra helpers for [`LineTrait`] implementers that mirror `geo-types` APIs.
pub trait LineTraitExt: LineTrait + GeoTraitExtWithTypeTag<Tag = LineTag>
where
    <Self as GeometryTrait>::T: CoordNum,
{
    type CoordTypeExt<'a>: 'a + CoordTraitExt<T = <Self as GeometryTrait>::T>
    where
        Self: 'a;

    /// Returns the start coordinate as an extension trait instance.
    fn start_ext(&self) -> Self::CoordTypeExt<'_>;
    /// Returns the end coordinate as an extension trait instance.
    fn end_ext(&self) -> Self::CoordTypeExt<'_>;
    /// Returns both start and end coordinates in a fixed-size array.
    fn coords_ext(&self) -> [Self::CoordTypeExt<'_>; 2];

    #[inline]
    /// Returns the start coordinate converted to [`geo_types::Coord`].
    fn start_coord(&self) -> geo_types::Coord<<Self as GeometryTrait>::T> {
        self.start_ext().geo_coord()
    }

    #[inline]
    /// Returns the end coordinate converted to [`geo_types::Coord`].
    fn end_coord(&self) -> geo_types::Coord<<Self as GeometryTrait>::T> {
        self.end_ext().geo_coord()
    }

    #[inline]
    /// Returns the line converted to a [`geo_types::Line`].
    fn geo_line(&self) -> Line<<Self as GeometryTrait>::T> {
        Line::new(self.start_coord(), self.end_coord())
    }
}

#[macro_export]
/// Forwards [`LineTraitExt`] methods to an underlying [`LineTrait`] implementation.
macro_rules! forward_line_trait_ext_funcs {
    () => {
        type CoordTypeExt<'__l_inner>
            = <Self as LineTrait>::CoordType<'__l_inner>
        where
            Self: '__l_inner;

        #[inline]
        fn start_ext(&self) -> Self::CoordTypeExt<'_> {
            <Self as LineTrait>::start(self)
        }

        #[inline]
        fn end_ext(&self) -> Self::CoordTypeExt<'_> {
            <Self as LineTrait>::end(self)
        }

        #[inline]
        fn coords_ext(&self) -> [Self::CoordTypeExt<'_>; 2] {
            [self.start_ext(), self.end_ext()]
        }
    };
}

impl<T> LineTraitExt for Line<T>
where
    T: CoordNum,
{
    forward_line_trait_ext_funcs!();

    fn start_coord(&self) -> geo_types::Coord<T> {
        self.start
    }

    fn end_coord(&self) -> geo_types::Coord<T> {
        self.end
    }

    fn geo_line(&self) -> geo_types::Line<T> {
        *self
    }
}

impl<T: CoordNum> GeoTraitExtWithTypeTag for Line<T> {
    type Tag = LineTag;
}

impl<T> LineTraitExt for &Line<T>
where
    T: CoordNum,
{
    forward_line_trait_ext_funcs!();

    fn start_coord(&self) -> geo_types::Coord<T> {
        self.start
    }

    fn end_coord(&self) -> geo_types::Coord<T> {
        self.end
    }

    fn geo_line(&self) -> geo_types::Line<T> {
        **self
    }
}

impl<T: CoordNum> GeoTraitExtWithTypeTag for &Line<T> {
    type Tag = LineTag;
}

impl<T> LineTraitExt for UnimplementedLine<T>
where
    T: CoordNum,
{
    forward_line_trait_ext_funcs!();
}

impl<T: CoordNum> GeoTraitExtWithTypeTag for UnimplementedLine<T> {
    type Tag = LineTag;
}
