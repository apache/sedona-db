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
// Extend MultiLineStringTrait traits for the `geo-traits` crate

use geo_traits::{GeometryTrait, MultiLineStringTrait, UnimplementedMultiLineString};
use geo_types::{CoordNum, MultiLineString};

use crate::{GeoTraitExtWithTypeTag, LineStringTraitExt, MultiLineStringTag};

/// Extension trait that layers additional ergonomics on
/// [`geo_traits::MultiLineStringTrait`].
///
/// implementers gain access to extension-aware iterators and helper methods
/// that mirror the behavior of `geo-types::MultiLineString`, while still being
/// consumable through the trait abstractions provided by `geo-traits`.
pub trait MultiLineStringTraitExt:
    MultiLineStringTrait + GeoTraitExtWithTypeTag<Tag = MultiLineStringTag>
where
    <Self as GeometryTrait>::T: CoordNum,
{
    /// Extension-friendly line string type returned by accessor methods.
    type LineStringTypeExt<'a>: 'a + LineStringTraitExt<T = <Self as GeometryTrait>::T>
    where
        Self: 'a;

    /// Returns the line string at index `i` with the extension trait applied.
    ///
    /// This is analogous to [`geo_traits::MultiLineStringTrait::line_string`]
    /// but ensures the result implements [`LineStringTraitExt`].
    fn line_string_ext(&self, i: usize) -> Option<Self::LineStringTypeExt<'_>>;

    /// Returns a line string by index without bounds checking.
    ///
    /// # Safety
    /// The caller must ensure that `i` is a valid index less than the number of line strings.
    /// Otherwise, this function may cause undefined behavior.
    unsafe fn line_string_unchecked_ext(&self, i: usize) -> Self::LineStringTypeExt<'_>;

    /// Iterates over all line strings with extension-aware wrappers applied.
    fn line_strings_ext(&self) -> impl Iterator<Item = Self::LineStringTypeExt<'_>>;

    /// Returns `true` when the multi line string is empty or every component is closed.
    #[inline]
    fn is_closed(&self) -> bool {
        // Note: Unlike JTS et al, we consider an empty MultiLineString as closed.
        self.line_strings_ext().all(|ls| ls.is_closed())
    }
}

#[macro_export]
/// Forwards [`MultiLineStringTraitExt`] methods to the underlying
/// [`geo_traits::MultiLineStringTrait`] implementation while keeping the
/// extension trait wrappers intact.
macro_rules! forward_multi_line_string_trait_ext_funcs {
    () => {
        type LineStringTypeExt<'__l_inner>
            = <Self as MultiLineStringTrait>::InnerLineStringType<'__l_inner>
        where
            Self: '__l_inner;

        #[inline]
        fn line_string_ext(&self, i: usize) -> Option<Self::LineStringTypeExt<'_>> {
            <Self as MultiLineStringTrait>::line_string(self, i)
        }

        #[inline]
        unsafe fn line_string_unchecked_ext(&self, i: usize) -> Self::LineStringTypeExt<'_> {
            <Self as MultiLineStringTrait>::line_string_unchecked(self, i)
        }

        #[inline]
        fn line_strings_ext(&self) -> impl Iterator<Item = Self::LineStringTypeExt<'_>> {
            <Self as MultiLineStringTrait>::line_strings(self)
        }
    };
}

impl<T> MultiLineStringTraitExt for MultiLineString<T>
where
    T: CoordNum,
{
    forward_multi_line_string_trait_ext_funcs!();
}

impl<T: CoordNum> GeoTraitExtWithTypeTag for MultiLineString<T> {
    type Tag = MultiLineStringTag;
}

impl<T> MultiLineStringTraitExt for &MultiLineString<T>
where
    T: CoordNum,
{
    forward_multi_line_string_trait_ext_funcs!();
}

impl<T: CoordNum> GeoTraitExtWithTypeTag for &MultiLineString<T> {
    type Tag = MultiLineStringTag;
}

impl<T> MultiLineStringTraitExt for UnimplementedMultiLineString<T>
where
    T: CoordNum,
{
    forward_multi_line_string_trait_ext_funcs!();
}

impl<T: CoordNum> GeoTraitExtWithTypeTag for UnimplementedMultiLineString<T> {
    type Tag = MultiLineStringTag;
}
