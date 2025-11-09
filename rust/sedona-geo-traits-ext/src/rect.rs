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
// Extend RectTrait traits for the `geo-traits` crate

use geo_traits::{CoordTrait, GeometryTrait, RectTrait, UnimplementedRect};
use geo_types::{coord, Coord, CoordFloat, CoordNum, Line, LineString, Polygon, Rect};
use num_traits::One;

use crate::{CoordTraitExt, GeoTraitExtWithTypeTag, RectTag};

static RECT_INVALID_BOUNDS_ERROR: &str = "Failed to create Rect: 'min' coordinate's x/y value must be smaller or equal to the 'max' x/y value";

/// Extension trait that augments [`geo_traits::RectTrait`] with additional
/// helpers for working with axis-aligned bounding boxes.
pub trait RectTraitExt: RectTrait + GeoTraitExtWithTypeTag<Tag = RectTag>
where
    <Self as GeometryTrait>::T: CoordNum,
{
    /// Extension-aware coordinate type returned from accessors.
    type CoordTypeExt<'a>: 'a + CoordTraitExt<T = <Self as GeometryTrait>::T>
    where
        Self: 'a;

    /// Returns the minimum corner using the extension trait wrapper.
    fn min_ext(&self) -> Self::CoordTypeExt<'_>;

    /// Returns the maximum corner using the extension trait wrapper.
    fn max_ext(&self) -> Self::CoordTypeExt<'_>;

    #[inline]
    /// Returns the minimum corner as a `geo-types::Coord`.
    fn min_coord(&self) -> Coord<<Self as GeometryTrait>::T> {
        self.min_ext().geo_coord()
    }

    #[inline]
    /// Returns the maximum corner as a `geo-types::Coord`.
    fn max_coord(&self) -> Coord<<Self as GeometryTrait>::T> {
        self.max_ext().geo_coord()
    }

    #[inline]
    /// Constructs a [`geo_types::Rect`] from the extension trait accessors.
    fn geo_rect(&self) -> Rect<<Self as GeometryTrait>::T> {
        Rect::new(self.min_coord(), self.max_coord())
    }

    #[inline]
    /// Returns the width of the rectangle.
    fn width(&self) -> <Self as GeometryTrait>::T {
        self.max().x() - self.min().x()
    }

    #[inline]
    /// Returns the height of the rectangle.
    fn height(&self) -> <Self as GeometryTrait>::T {
        self.max().y() - self.min().y()
    }

    /// Converts the rectangle into a polygon with four corners.
    fn to_polygon(&self) -> Polygon<<Self as GeometryTrait>::T>
    where
        <Self as GeometryTrait>::T: Clone,
    {
        let min_coord = self.min_coord();
        let max_coord = self.max_coord();

        let min_x = min_coord.x;
        let min_y = min_coord.y;
        let max_x = max_coord.x;
        let max_y = max_coord.y;

        let line_string = LineString::new(vec![
            Coord { x: min_x, y: min_y },
            Coord { x: min_x, y: max_y },
            Coord { x: max_x, y: max_y },
            Coord { x: max_x, y: min_y },
            Coord { x: min_x, y: min_y },
        ]);

        Polygon::new(line_string, vec![])
    }

    /// Returns the four outer edges as line segments.
    fn to_lines(&self) -> [Line<<Self as GeometryTrait>::T>; 4] {
        let min_coord = self.min_coord();
        let max_coord = self.max_coord();
        [
            Line::new(
                coord! {
                    x: max_coord.x,
                    y: min_coord.y,
                },
                coord! {
                    x: max_coord.x,
                    y: max_coord.y,
                },
            ),
            Line::new(
                coord! {
                    x: max_coord.x,
                    y: max_coord.y,
                },
                coord! {
                    x: min_coord.x,
                    y: max_coord.y,
                },
            ),
            Line::new(
                coord! {
                    x: min_coord.x,
                    y: max_coord.y,
                },
                coord! {
                    x: min_coord.x,
                    y: min_coord.y,
                },
            ),
            Line::new(
                coord! {
                    x: min_coord.x,
                    y: min_coord.y,
                },
                coord! {
                    x: max_coord.x,
                    y: min_coord.y,
                },
            ),
        ]
    }

    /// Converts the rectangle into a closed line string in counter-clockwise order.
    fn to_line_string(&self) -> LineString<<Self as GeometryTrait>::T>
    where
        <Self as GeometryTrait>::T: Clone,
    {
        let min_coord = self.min_coord();
        let max_coord = self.max_coord();

        let min_x = min_coord.x;
        let min_y = min_coord.y;
        let max_x = max_coord.x;
        let max_y = max_coord.y;

        LineString::new(vec![
            Coord { x: min_x, y: min_y },
            Coord { x: min_x, y: max_y },
            Coord { x: max_x, y: max_y },
            Coord { x: max_x, y: min_y },
            Coord { x: min_x, y: min_y },
        ])
    }

    #[inline]
    /// Returns `true` if the rectangle has non-decreasing bounds.
    fn has_valid_bounds(&self) -> bool {
        let min_coord = self.min_coord();
        let max_coord = self.max_coord();
        min_coord.x <= max_coord.x && min_coord.y <= max_coord.y
    }

    #[inline]
    /// Panics when the rectangle bounds are invalid.
    fn assert_valid_bounds(&self) {
        if !self.has_valid_bounds() {
            panic!("{}", RECT_INVALID_BOUNDS_ERROR);
        }
    }

    #[inline]
    /// Returns `true` if `coord` lies inside or on the rectangle boundary.
    fn contains_point(&self, coord: &Coord<<Self as GeometryTrait>::T>) -> bool
    where
        <Self as GeometryTrait>::T: PartialOrd,
    {
        let min_coord = self.min_coord();
        let max_coord = self.max_coord();

        let min_x = min_coord.x;
        let min_y = min_coord.y;
        let max_x = max_coord.x;
        let max_y = max_coord.y;

        (min_x <= coord.x && coord.x <= max_x) && (min_y <= coord.y && coord.y <= max_y)
    }

    #[inline]
    /// Returns `true` if `rect` is fully contained within `self`.
    fn contains_rect(&self, rect: &Self) -> bool
    where
        <Self as GeometryTrait>::T: PartialOrd,
    {
        let self_min = self.min_coord();
        let self_max = self.max_coord();
        let other_min = rect.min_coord();
        let other_max = rect.max_coord();

        let self_min_x = self_min.x;
        let self_min_y = self_min.y;
        let self_max_x = self_max.x;
        let self_max_y = self_max.y;

        let other_min_x = other_min.x;
        let other_min_y = other_min.y;
        let other_max_x = other_max.x;
        let other_max_y = other_max.y;

        (self_min_x <= other_min_x && other_max_x <= self_max_x)
            && (self_min_y <= other_min_y && other_max_y <= self_max_y)
    }

    #[inline]
    /// Returns the rectangle centroid as a coordinate.
    fn center(&self) -> Coord<<Self as GeometryTrait>::T>
    where
        <Self as GeometryTrait>::T: CoordFloat,
    {
        let two = <Self as GeometryTrait>::T::one() + <Self as GeometryTrait>::T::one();
        coord! {
            x: (self.max_coord().x + self.min_coord().x) / two,
            y: (self.max_coord().y + self.min_coord().y) / two,
        }
    }
}

#[macro_export]
/// Forwards [`RectTraitExt`] methods to the underlying
/// [`geo_traits::RectTrait`] implementation while keeping the extension trait
/// wrappers intact.
macro_rules! forward_rect_trait_ext_funcs {
    () => {
        type CoordTypeExt<'__l_inner>
            = <Self as RectTrait>::CoordType<'__l_inner>
        where
            Self: '__l_inner;

        fn min_ext(&self) -> Self::CoordTypeExt<'_> {
            <Self as RectTrait>::min(self)
        }

        fn max_ext(&self) -> Self::CoordTypeExt<'_> {
            <Self as RectTrait>::max(self)
        }
    };
}

impl<T> RectTraitExt for Rect<T>
where
    T: CoordNum,
{
    forward_rect_trait_ext_funcs!();

    fn min_coord(&self) -> Coord<T> {
        Rect::min(*self)
    }

    fn max_coord(&self) -> Coord<T> {
        Rect::max(*self)
    }

    fn geo_rect(&self) -> Rect<T> {
        *self
    }

    fn to_lines(&self) -> [Line<<Self as GeometryTrait>::T>; 4] {
        self.to_lines()
    }
}

impl<T: CoordNum> GeoTraitExtWithTypeTag for Rect<T> {
    type Tag = RectTag;
}

impl<T> RectTraitExt for &Rect<T>
where
    T: CoordNum,
{
    forward_rect_trait_ext_funcs!();

    fn min_coord(&self) -> Coord<T> {
        Rect::min(**self)
    }

    fn max_coord(&self) -> Coord<T> {
        Rect::max(**self)
    }

    fn geo_rect(&self) -> Rect<T> {
        **self
    }

    fn to_polygon(&self) -> Polygon<<Self as GeometryTrait>::T>
    where
        <Self as GeometryTrait>::T: Clone,
    {
        (*self).to_polygon()
    }

    fn to_lines(&self) -> [Line<<Self as GeometryTrait>::T>; 4] {
        (*self).to_lines()
    }
}

impl<T: CoordNum> GeoTraitExtWithTypeTag for &Rect<T> {
    type Tag = RectTag;
}

impl<T> RectTraitExt for UnimplementedRect<T>
where
    T: CoordNum,
{
    forward_rect_trait_ext_funcs!();
}

impl<T: CoordNum> GeoTraitExtWithTypeTag for UnimplementedRect<T> {
    type Tag = RectTag;
}
