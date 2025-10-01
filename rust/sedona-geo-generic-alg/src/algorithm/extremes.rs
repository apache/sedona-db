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
use crate::CoordsIter;
use crate::{Coord, CoordNum};

/// Find the extreme coordinates and indices of a geometry.
///
/// # Examples
///
/// ```
/// use sedona_geo_generic_alg::extremes::Extremes;
/// use sedona_geo_generic_alg::polygon;
///
/// // a diamond shape
/// let polygon = polygon![
///     (x: 1.0, y: 0.0),
///     (x: 2.0, y: 1.0),
///     (x: 1.0, y: 2.0),
///     (x: 0.0, y: 1.0),
///     (x: 1.0, y: 0.0),
/// ];
///
/// let extremes = polygon.extremes().unwrap();
///
/// assert_eq!(extremes.y_max.index, 2);
/// assert_eq!(extremes.y_max.coord.x, 1.);
/// assert_eq!(extremes.y_max.coord.y, 2.);
/// ```
pub trait Extremes<'a, T: CoordNum> {
    fn extremes(&'a self) -> Option<Outcome<T>>;
}

#[derive(Debug, PartialEq, Eq)]
pub struct Extreme<T: CoordNum> {
    pub index: usize,
    pub coord: Coord<T>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Outcome<T: CoordNum> {
    pub x_min: Extreme<T>,
    pub y_min: Extreme<T>,
    pub x_max: Extreme<T>,
    pub y_max: Extreme<T>,
}

impl<'a, T, G> Extremes<'a, T> for G
where
    G: CoordsIter<Scalar = T>,
    T: CoordNum,
{
    fn extremes(&'a self) -> Option<Outcome<T>> {
        let mut iter = self.exterior_coords_iter().enumerate();

        let mut outcome = iter.next().map(|(index, coord)| Outcome {
            x_min: Extreme { index, coord },
            y_min: Extreme { index, coord },
            x_max: Extreme { index, coord },
            y_max: Extreme { index, coord },
        })?;

        for (index, coord) in iter {
            if coord.x < outcome.x_min.coord.x {
                outcome.x_min = Extreme { coord, index };
            }

            if coord.y < outcome.y_min.coord.y {
                outcome.y_min = Extreme { coord, index };
            }

            if coord.x > outcome.x_max.coord.x {
                outcome.x_max = Extreme { coord, index };
            }

            if coord.y > outcome.y_max.coord.y {
                outcome.y_max = Extreme { coord, index };
            }
        }

        Some(outcome)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{coord, polygon, MultiPoint};

    #[test]
    fn polygon() {
        // a diamond shape
        let polygon = polygon![
            (x: 1.0, y: 0.0),
            (x: 2.0, y: 1.0),
            (x: 1.0, y: 2.0),
            (x: 0.0, y: 1.0),
            (x: 1.0, y: 0.0),
        ];

        let actual = polygon.extremes();

        assert_eq!(
            Some(Outcome {
                x_min: Extreme {
                    index: 3,
                    coord: coord! { x: 0.0, y: 1.0 }
                },
                y_min: Extreme {
                    index: 0,
                    coord: coord! { x: 1.0, y: 0.0 }
                },
                x_max: Extreme {
                    index: 1,
                    coord: coord! { x: 2.0, y: 1.0 }
                },
                y_max: Extreme {
                    index: 2,
                    coord: coord! { x: 1.0, y: 2.0 }
                }
            }),
            actual
        );
    }

    #[test]
    fn empty() {
        let multi_point: MultiPoint<f32> = MultiPoint::new(vec![]);

        let actual = multi_point.extremes();

        assert!(actual.is_none());
    }
}
