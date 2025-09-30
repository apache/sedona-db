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
use crate::{Coord, CoordNum, MapCoords};

/// Convert (infalliby) the type of a geometry’s coordinate value.
///
/// # Examples
///
/// ```rust
/// use geo_generic_alg::{Convert, LineString, line_string};
///
/// let line_string_32: LineString<f32> = line_string![
///     (x: 5., y: 10.),
///     (x: 3., y: 1.),
///     (x: 8., y: 9.),
/// ];
///
/// let line_string_64: LineString<f64> = line_string_32.convert();
/// ```
///
pub trait Convert<T, U> {
    type Output;

    fn convert(&self) -> Self::Output;
}
impl<G, T: CoordNum, U: CoordNum> Convert<T, U> for G
where
    G: MapCoords<T, U>,
    U: From<T>,
{
    type Output = <Self as MapCoords<T, U>>::Output;

    fn convert(&self) -> Self::Output {
        self.map_coords(|Coord { x, y }| Coord {
            x: x.into(),
            y: y.into(),
        })
    }
}

/// Convert (fallibly) the type of a geometry’s coordinate value.
///
/// # Examples
///
/// ```rust
/// use geo_generic_alg::{TryConvert, LineString, line_string};
///
/// let line_string_64: LineString<i64> = line_string![
///     (x: 5, y: 10),
///     (x: 3, y: 1),
///     (x: 8, y: 9),
/// ];
///
/// let line_string_32: Result<LineString<i32>, _> = line_string_64.try_convert();
/// ```
///
pub trait TryConvert<T, U> {
    type Output;

    fn try_convert(&self) -> Self::Output;
}
impl<G, T: CoordNum, U: CoordNum> TryConvert<T, U> for G
where
    G: MapCoords<T, U>,
    U: TryFrom<T>,
{
    type Output = Result<<Self as MapCoords<T, U>>::Output, <U as TryFrom<T>>::Error>;

    fn try_convert(&self) -> Self::Output {
        self.try_map_coords(|Coord { x, y }| {
            Ok(Coord {
                x: x.try_into()?,
                y: y.try_into()?,
            })
        })
    }
}
