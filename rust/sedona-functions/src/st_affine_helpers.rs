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
use arrow_array::types::Float64Type;
use arrow_array::Array;
use arrow_array::PrimitiveArray;
use datafusion_common::cast::as_float64_array;
use datafusion_common::error::Result;
use datafusion_common::ScalarValue;
use datafusion_expr::ColumnarValue;
use sedona_common::sedona_internal_err;
use sedona_geometry::transform::CrsTransform;
use std::sync::Arc;

#[derive(Clone, Copy)]
enum FloatArg<'a> {
    Array(&'a PrimitiveArray<Float64Type>),
    Scalar(Option<f64>),
}

impl<'a> FloatArg<'a> {
    fn is_null(&self, i: usize) -> bool {
        match self {
            FloatArg::Array(values) => values.is_null(i),
            FloatArg::Scalar(value) => value.is_none(),
        }
    }

    fn value(&self, i: usize) -> f64 {
        match self {
            FloatArg::Array(values) => values.value(i),
            FloatArg::Scalar(Some(value)) => *value,
            FloatArg::Scalar(None) => 0.0,
        }
    }

    fn no_nulls(&self) -> bool {
        match self {
            FloatArg::Array(values) => values.null_count() == 0,
            FloatArg::Scalar(value) => value.is_some(),
        }
    }
}

fn float_arg_from_columnar<'a>(arg: &'a ColumnarValue) -> Result<FloatArg<'a>> {
    match arg {
        ColumnarValue::Array(array) => Ok(FloatArg::Array(as_float64_array(array)?)),
        ColumnarValue::Scalar(ScalarValue::Float64(value)) => Ok(FloatArg::Scalar(*value)),
        ColumnarValue::Scalar(ScalarValue::Null) => Ok(FloatArg::Scalar(None)),
        _ => sedona_internal_err!("Invalid scalar type for affine argument"),
    }
}

pub(crate) struct DAffine2Iterator<'a> {
    index: usize,
    a: FloatArg<'a>,
    b: FloatArg<'a>,
    d: FloatArg<'a>,
    e: FloatArg<'a>,
    x_offset: FloatArg<'a>,
    y_offset: FloatArg<'a>,
    no_null: bool,
}

impl<'a> DAffine2Iterator<'a> {
    pub(crate) fn new(array_args: &'a [ColumnarValue]) -> Result<Self> {
        if array_args.len() != 6 {
            return sedona_internal_err!("Invalid number of arguments are passed");
        }

        let a = float_arg_from_columnar(&array_args[0])?;
        let b = float_arg_from_columnar(&array_args[1])?;
        let d = float_arg_from_columnar(&array_args[2])?;
        let e = float_arg_from_columnar(&array_args[3])?;
        let x_offset = float_arg_from_columnar(&array_args[4])?;
        let y_offset = float_arg_from_columnar(&array_args[5])?;
        let no_null = a.no_nulls()
            && b.no_nulls()
            && d.no_nulls()
            && e.no_nulls()
            && x_offset.no_nulls()
            && y_offset.no_nulls();

        Ok(Self {
            index: 0,
            a,
            b,
            d,
            e,
            x_offset,
            y_offset,
            no_null,
        })
    }

    fn is_null(&self, i: usize) -> bool {
        if self.no_null {
            return false;
        }

        self.a.is_null(i)
            || self.b.is_null(i)
            || self.d.is_null(i)
            || self.e.is_null(i)
            || self.x_offset.is_null(i)
            || self.y_offset.is_null(i)
    }
}

impl<'a> Iterator for DAffine2Iterator<'a> {
    // As this needs to distinguish NULL, next() returns Some(Some(value))
    type Item = Option<glam::DAffine2>;

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.index;
        self.index += 1;

        if self.is_null(i) {
            return Some(None);
        }

        Some(Some(glam::DAffine2 {
            matrix2: glam::DMat2 {
                x_axis: glam::DVec2 {
                    x: self.a.value(i),
                    y: self.b.value(i),
                },
                y_axis: glam::DVec2 {
                    x: self.d.value(i),
                    y: self.e.value(i),
                },
            },
            translation: glam::DVec2 {
                x: self.x_offset.value(i),
                y: self.y_offset.value(i),
            },
        }))
    }
}

pub(crate) struct DAffine3Iterator<'a> {
    index: usize,
    a: FloatArg<'a>,
    b: FloatArg<'a>,
    c: FloatArg<'a>,
    d: FloatArg<'a>,
    e: FloatArg<'a>,
    f: FloatArg<'a>,
    g: FloatArg<'a>,
    h: FloatArg<'a>,
    i: FloatArg<'a>,
    x_offset: FloatArg<'a>,
    y_offset: FloatArg<'a>,
    z_offset: FloatArg<'a>,
    no_null: bool,
}

impl<'a> DAffine3Iterator<'a> {
    pub(crate) fn new(array_args: &'a [ColumnarValue]) -> Result<Self> {
        if array_args.len() != 12 {
            return sedona_internal_err!("Invalid number of arguments are passed");
        }

        let a = float_arg_from_columnar(&array_args[0])?;
        let b = float_arg_from_columnar(&array_args[1])?;
        let c = float_arg_from_columnar(&array_args[2])?;
        let d = float_arg_from_columnar(&array_args[3])?;
        let e = float_arg_from_columnar(&array_args[4])?;
        let f = float_arg_from_columnar(&array_args[5])?;
        let g = float_arg_from_columnar(&array_args[6])?;
        let h = float_arg_from_columnar(&array_args[7])?;
        let i = float_arg_from_columnar(&array_args[8])?;
        let x_offset = float_arg_from_columnar(&array_args[9])?;
        let y_offset = float_arg_from_columnar(&array_args[10])?;
        let z_offset = float_arg_from_columnar(&array_args[11])?;

        let no_null = a.no_nulls()
            && b.no_nulls()
            && c.no_nulls()
            && d.no_nulls()
            && e.no_nulls()
            && f.no_nulls()
            && g.no_nulls()
            && h.no_nulls()
            && i.no_nulls()
            && x_offset.no_nulls()
            && y_offset.no_nulls()
            && z_offset.no_nulls();

        Ok(Self {
            index: 0,
            a,
            b,
            c,
            d,
            e,
            f,
            g,
            h,
            i,
            x_offset,
            y_offset,
            z_offset,
            no_null,
        })
    }

    fn is_null(&self, i: usize) -> bool {
        if self.no_null {
            return false;
        }

        self.a.is_null(i)
            || self.b.is_null(i)
            || self.c.is_null(i)
            || self.d.is_null(i)
            || self.e.is_null(i)
            || self.f.is_null(i)
            || self.g.is_null(i)
            || self.h.is_null(i)
            || self.i.is_null(i)
            || self.x_offset.is_null(i)
            || self.y_offset.is_null(i)
            || self.z_offset.is_null(i)
    }
}

impl<'a> Iterator for DAffine3Iterator<'a> {
    // As this needs to distinguish NULL, next() returns Some(Some(value))
    type Item = Option<glam::DAffine3>;

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.index;
        self.index += 1;

        if self.is_null(i) {
            return Some(None);
        }

        Some(Some(glam::DAffine3 {
            matrix3: glam::DMat3 {
                x_axis: glam::DVec3 {
                    x: self.a.value(i),
                    y: self.b.value(i),
                    z: self.c.value(i),
                },
                y_axis: glam::DVec3 {
                    x: self.d.value(i),
                    y: self.e.value(i),
                    z: self.f.value(i),
                },
                z_axis: glam::DVec3 {
                    x: self.g.value(i),
                    y: self.h.value(i),
                    z: self.i.value(i),
                },
            },
            translation: glam::DVec3 {
                x: self.x_offset.value(i),
                y: self.y_offset.value(i),
                z: self.z_offset.value(i),
            },
        }))
    }
}

pub(crate) struct DAffine2ScaleIterator<'a> {
    index: usize,
    x_scale: &'a PrimitiveArray<Float64Type>,
    y_scale: &'a PrimitiveArray<Float64Type>,
    no_null: bool,
}

impl<'a> DAffine2ScaleIterator<'a> {
    pub(crate) fn new(array_args: &'a [Arc<dyn Array>]) -> Result<Self> {
        if array_args.len() != 2 {
            return sedona_internal_err!("Invalid number of arguments are passed");
        }

        let x_scale = as_float64_array(&array_args[0])?;
        let y_scale = as_float64_array(&array_args[1])?;

        Ok(Self {
            index: 0,
            x_scale,
            y_scale,
            no_null: x_scale.null_count() == 0 && y_scale.null_count() == 0,
        })
    }

    fn is_null(&self, i: usize) -> bool {
        if self.no_null {
            return false;
        }

        self.x_scale.is_null(i) || self.y_scale.is_null(i)
    }
}

impl<'a> Iterator for DAffine2ScaleIterator<'a> {
    // As this needs to distinguish NULL, next() returns Some(Some(value))
    type Item = Option<glam::DAffine2>;

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.index;
        self.index += 1;

        if self.is_null(i) {
            return Some(None);
        }

        let scale = glam::DVec2::new(self.x_scale.value(i), self.y_scale.value(i));
        Some(Some(glam::DAffine2::from_scale(scale)))
    }
}

pub(crate) struct DAffine3ScaleIterator<'a> {
    index: usize,
    x_scale: &'a PrimitiveArray<Float64Type>,
    y_scale: &'a PrimitiveArray<Float64Type>,
    z_scale: &'a PrimitiveArray<Float64Type>,
    no_null: bool,
}

impl<'a> DAffine3ScaleIterator<'a> {
    pub(crate) fn new(array_args: &'a [Arc<dyn Array>]) -> Result<Self> {
        if array_args.len() != 3 {
            return sedona_internal_err!("Invalid number of arguments are passed");
        }

        let x_scale = as_float64_array(&array_args[0])?;
        let y_scale = as_float64_array(&array_args[1])?;
        let z_scale = as_float64_array(&array_args[2])?;

        Ok(Self {
            index: 0,
            x_scale,
            y_scale,
            z_scale,
            no_null: x_scale.null_count() == 0
                && y_scale.null_count() == 0
                && z_scale.null_count() == 0,
        })
    }

    fn is_null(&self, i: usize) -> bool {
        if self.no_null {
            return false;
        }

        self.x_scale.is_null(i) || self.y_scale.is_null(i) || self.z_scale.is_null(i)
    }
}

impl<'a> Iterator for DAffine3ScaleIterator<'a> {
    // As this needs to distinguish NULL, next() returns Some(Some(value))
    type Item = Option<glam::DAffine3>;

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.index;
        self.index += 1;

        if self.is_null(i) {
            return Some(None);
        }

        let scale = glam::DVec3::new(
            self.x_scale.value(i),
            self.y_scale.value(i),
            self.z_scale.value(i),
        );
        Some(Some(glam::DAffine3::from_scale(scale)))
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum RotateAxis {
    X,
    Y,
    Z,
}

pub(crate) struct DAffineRotateIterator<'a> {
    index: usize,
    angle: &'a PrimitiveArray<Float64Type>,
    axis: RotateAxis,
    no_null: bool,
}

impl<'a> DAffineRotateIterator<'a> {
    pub(crate) fn new(angle: &'a Arc<dyn Array>, axis: RotateAxis) -> Result<Self> {
        let angle = as_float64_array(angle)?;
        Ok(Self {
            index: 0,
            angle,
            axis,
            no_null: angle.null_count() == 0,
        })
    }

    fn is_null(&self, i: usize) -> bool {
        if self.no_null {
            return false;
        }

        self.angle.is_null(i)
    }
}

impl<'a> Iterator for DAffineRotateIterator<'a> {
    // As this needs to distinguish NULL, next() returns Some(Some(value))
    type Item = Option<glam::DAffine3>;

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.index;
        self.index += 1;

        if self.is_null(i) {
            return Some(None);
        }

        match self.axis {
            RotateAxis::X => Some(Some(glam::DAffine3::from_rotation_x(self.angle.value(i)))),
            RotateAxis::Y => Some(Some(glam::DAffine3::from_rotation_y(self.angle.value(i)))),
            RotateAxis::Z => Some(Some(glam::DAffine3::from_rotation_z(self.angle.value(i)))),
        }
    }
}

pub(crate) enum DAffineIterator<'a> {
    DAffine2(DAffine2Iterator),
    DAffine3(DAffine3Iterator),
    DAffine2Scale(DAffine2ScaleIterator<'a>),
    DAffine3Scale(DAffine3ScaleIterator<'a>),
    DAffineRotate(DAffineRotateIterator<'a>),
    // Short-circuit when any scalar affine argument is NULL; avoids holding args and always yields NULL.
    Null,
}

impl<'a> DAffineIterator<'a> {
    pub(crate) fn new_2d(array_args: &'a [ColumnarValue]) -> Result<Self> {
        if has_null_scalar(array_args) {
            return Ok(Self::Null);
        }
        Ok(Self::DAffine2(DAffine2Iterator::new(array_args)?))
    }

    pub(crate) fn new_3d(array_args: &'a [ColumnarValue]) -> Result<Self> {
        if has_null_scalar(array_args) {
            return Ok(Self::Null);
        }
        Ok(Self::DAffine3(DAffine3Iterator::new(array_args)?))
    }

    pub(crate) fn from_scale_2d(array_args: &'a [Arc<dyn Array>]) -> Result<Self> {
        Ok(Self::DAffine2Scale(DAffine2ScaleIterator::new(array_args)?))
    }

    pub(crate) fn from_scale_3d(array_args: &'a [Arc<dyn Array>]) -> Result<Self> {
        Ok(Self::DAffine3Scale(DAffine3ScaleIterator::new(array_args)?))
    }

    pub(crate) fn from_angle(angle: &'a Arc<dyn Array>, axis: RotateAxis) -> Result<Self> {
        Ok(Self::DAffineRotate(DAffineRotateIterator::new(
            angle, axis,
        )?))
    }
}

#[derive(Debug, PartialEq)]
pub(crate) enum DAffine {
    DAffine2(glam::DAffine2),
    DAffine3(glam::DAffine3),
}

impl CrsTransform for DAffine {
    fn transform_coord_3d(
        &self,
        coord: &mut (f64, f64, f64),
    ) -> std::result::Result<(), sedona_geometry::error::SedonaGeometryError> {
        match self {
            DAffine::DAffine2(daffine2) => {
                let transformed = daffine2.transform_point2(glam::DVec2 {
                    x: coord.0,
                    y: coord.1,
                });
                coord.0 = transformed.x;
                coord.1 = transformed.y;
            }
            DAffine::DAffine3(daffine3) => {
                let transformed = daffine3.transform_point3(glam::DVec3 {
                    x: coord.0,
                    y: coord.1,
                    z: coord.2,
                });
                coord.0 = transformed.x;
                coord.1 = transformed.y;
                coord.2 = transformed.z;
            }
        }

        Ok(())
    }

    fn transform_coord(
        &self,
        coord: &mut (f64, f64),
    ) -> std::result::Result<(), sedona_geometry::error::SedonaGeometryError> {
        match self {
            DAffine::DAffine2(daffine2) => {
                let transformed = daffine2.transform_point2(glam::DVec2 {
                    x: coord.0,
                    y: coord.1,
                });
                coord.0 = transformed.x;
                coord.1 = transformed.y;
            }
            DAffine::DAffine3(daffine3) => {
                let transformed = daffine3.transform_point3(glam::DVec3 {
                    x: coord.0,
                    y: coord.1,
                    z: 0.0,
                });
                coord.0 = transformed.x;
                coord.1 = transformed.y;
            }
        }

        Ok(())
    }
}

impl<'a> Iterator for DAffineIterator<'a> {
    type Item = Option<DAffine>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            DAffineIterator::DAffine2(daffine2_iterator) => match daffine2_iterator.next() {
                Some(Some(a)) => Some(Some(DAffine::DAffine2(a))),
                Some(None) => Some(None),
                None => None,
            },
            DAffineIterator::DAffine3(daffine3_iterator) => match daffine3_iterator.next() {
                Some(Some(a)) => Some(Some(DAffine::DAffine3(a))),
                Some(None) => Some(None),
                None => None,
            },
            DAffineIterator::DAffine2Scale(daffine2_scale_iterator) => {
                match daffine2_scale_iterator.next() {
                    Some(Some(a)) => Some(Some(DAffine::DAffine2(a))),
                    Some(None) => Some(None),
                    None => None,
                }
            }
            DAffineIterator::DAffine3Scale(daffine3_scale_iterator) => {
                match daffine3_scale_iterator.next() {
                    Some(Some(a)) => Some(Some(DAffine::DAffine3(a))),
                    Some(None) => Some(None),
                    None => None,
                }
            }
            DAffineIterator::DAffineRotate(daffine_rotate_iterator) => {
                match daffine_rotate_iterator.next() {
                    Some(Some(a)) => Some(Some(DAffine::DAffine3(a))),
                    Some(None) => Some(None),
                    None => None,
                }
            }
            DAffineIterator::Null => Some(None),
        }
    }
}

fn has_null_scalar(args: &[ColumnarValue]) -> bool {
    args.iter()
        .any(|arg| matches!(arg, ColumnarValue::Scalar(ScalarValue::Null)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Array;
    use arrow_array::Float64Array;
    use std::sync::Arc;

    fn float_array(values: Vec<Option<f64>>) -> Arc<dyn Array> {
        Arc::new(Float64Array::from(values)) as Arc<dyn Array>
    }

    fn float_columnar(values: Vec<Option<f64>>) -> ColumnarValue {
        ColumnarValue::Array(float_array(values))
    }

    #[test]
    fn daffine2_iterator_handles_nulls() {
        let args = vec![
            float_columnar(vec![Some(1.0), Some(10.0)]),
            float_columnar(vec![Some(2.0), Some(20.0)]),
            float_columnar(vec![Some(3.0), Some(30.0)]),
            float_columnar(vec![Some(4.0), None]),
            float_columnar(vec![Some(5.0), Some(50.0)]),
            float_columnar(vec![Some(6.0), Some(60.0)]),
        ];

        let mut iter = DAffine2Iterator::new(&args).unwrap();

        let expected_first = glam::DAffine2 {
            matrix2: glam::DMat2 {
                x_axis: glam::DVec2 { x: 1.0, y: 2.0 },
                y_axis: glam::DVec2 { x: 3.0, y: 4.0 },
            },
            translation: glam::DVec2 { x: 5.0, y: 6.0 },
        };
        assert_eq!(iter.next(), Some(Some(expected_first)));

        // The second case contains NULL, so the result is NULL
        assert_eq!(iter.next(), Some(None));
    }

    #[test]
    fn daffine3_iterator_values() {
        let values = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let args = values
            .iter()
            .map(|value| float_columnar(vec![Some(*value)]))
            .collect::<Vec<_>>();

        let mut iter = DAffine3Iterator::new(&args).unwrap();
        let expected = glam::DAffine3 {
            matrix3: glam::DMat3::from_cols(
                glam::DVec3::new(1.0, 2.0, 3.0),
                glam::DVec3::new(4.0, 5.0, 6.0),
                glam::DVec3::new(7.0, 8.0, 9.0),
            ),
            translation: glam::DVec3::new(10.0, 11.0, 12.0),
        };

        assert_eq!(iter.next(), Some(Some(expected)));
    }

    #[test]
    fn daffine_iterator_from_scale() {
        let scale_args = vec![
            float_array(vec![Some(2.0), None]),
            float_array(vec![Some(3.0), Some(4.0)]),
        ];
        let mut iter = DAffineIterator::from_scale_2d(&scale_args).unwrap();

        let expected_scale =
            DAffine::DAffine2(glam::DAffine2::from_scale(glam::DVec2::new(2.0, 3.0)));
        assert_eq!(iter.next(), Some(Some(expected_scale)));

        // The second case contains NULL, so the result is NULL
        assert_eq!(iter.next(), Some(None));
    }

    #[test]
    fn daffine_iterator_from_rotate() {
        let angle = float_array(vec![Some(0.25), None]);
        let mut iter = DAffineIterator::from_angle(&angle, RotateAxis::X).unwrap();
        let expected_rotate = DAffine::DAffine3(glam::DAffine3::from_rotation_x(0.25));
        assert_eq!(iter.next(), Some(Some(expected_rotate)));

        // The second case contains NULL, so the result is NULL
        assert_eq!(iter.next(), Some(None));
    }

    #[test]
    fn daffine_crs_transform_changes_coords() {
        let mut coord_2d = (1.0, 2.0);
        let affine_2d = DAffine::DAffine2(glam::DAffine2::from_scale(glam::DVec2::new(2.0, 3.0)));
        affine_2d.transform_coord(&mut coord_2d).unwrap();
        assert_eq!(coord_2d, (2.0, 6.0));

        let mut coord_3d = (1.0, 2.0, 3.0);
        let affine_3d =
            DAffine::DAffine3(glam::DAffine3::from_scale(glam::DVec3::new(2.0, 3.0, 4.0)));
        affine_3d.transform_coord_3d(&mut coord_3d).unwrap();
        assert_eq!(coord_3d, (2.0, 6.0, 12.0));
    }
}
