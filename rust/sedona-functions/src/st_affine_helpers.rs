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
use datafusion_common::internal_err;
use std::sync::Arc;

pub(crate) struct DAffine2Iterator<'a> {
    pub(crate) index: usize,
    pub(crate) a: &'a PrimitiveArray<Float64Type>,
    pub(crate) b: &'a PrimitiveArray<Float64Type>,
    pub(crate) d: &'a PrimitiveArray<Float64Type>,
    pub(crate) e: &'a PrimitiveArray<Float64Type>,
    pub(crate) x_offset: &'a PrimitiveArray<Float64Type>,
    pub(crate) y_offset: &'a PrimitiveArray<Float64Type>,
}

impl<'a> DAffine2Iterator<'a> {
    pub(crate) fn new(array_args: &'a [Arc<dyn Array>]) -> Result<Self> {
        if array_args.len() != 6 {
            return internal_err!("Invalid number of arguments are passed");
        }

        let a = as_float64_array(&array_args[0])?;
        let b = as_float64_array(&array_args[1])?;
        let d = as_float64_array(&array_args[2])?;
        let e = as_float64_array(&array_args[3])?;
        let x_offset = as_float64_array(&array_args[4])?;
        let y_offset = as_float64_array(&array_args[5])?;

        Ok(Self {
            index: 0,
            a,
            b,
            d,
            e,
            x_offset,
            y_offset,
        })
    }
}

impl<'a> Iterator for DAffine2Iterator<'a> {
    type Item = glam::DAffine2;

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.index;
        self.index += 1;
        Some(glam::DAffine2 {
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
        })
    }
}

pub(crate) struct DAffine3Iterator<'a> {
    pub(crate) index: usize,
    pub(crate) a: &'a PrimitiveArray<Float64Type>,
    pub(crate) b: &'a PrimitiveArray<Float64Type>,
    pub(crate) c: &'a PrimitiveArray<Float64Type>,
    pub(crate) d: &'a PrimitiveArray<Float64Type>,
    pub(crate) e: &'a PrimitiveArray<Float64Type>,
    pub(crate) f: &'a PrimitiveArray<Float64Type>,
    pub(crate) g: &'a PrimitiveArray<Float64Type>,
    pub(crate) h: &'a PrimitiveArray<Float64Type>,
    pub(crate) i: &'a PrimitiveArray<Float64Type>,
    pub(crate) x_offset: &'a PrimitiveArray<Float64Type>,
    pub(crate) y_offset: &'a PrimitiveArray<Float64Type>,
    pub(crate) z_offset: &'a PrimitiveArray<Float64Type>,
}

impl<'a> DAffine3Iterator<'a> {
    pub(crate) fn new(array_args: &'a [Arc<dyn Array>]) -> Result<Self> {
        if array_args.len() != 12 {
            return internal_err!("Invalid number of arguments are passed");
        }

        let a = as_float64_array(&array_args[0])?;
        let b = as_float64_array(&array_args[1])?;
        let c = as_float64_array(&array_args[2])?;
        let d = as_float64_array(&array_args[3])?;
        let e = as_float64_array(&array_args[4])?;
        let f = as_float64_array(&array_args[5])?;
        let g = as_float64_array(&array_args[6])?;
        let h = as_float64_array(&array_args[7])?;
        let i = as_float64_array(&array_args[8])?;
        let x_offset = as_float64_array(&array_args[9])?;
        let y_offset = as_float64_array(&array_args[10])?;
        let z_offset = as_float64_array(&array_args[11])?;

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
        })
    }
}

impl<'a> Iterator for DAffine3Iterator<'a> {
    type Item = glam::DAffine3;

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.index;
        self.index += 1;
        Some(glam::DAffine3 {
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
        })
    }
}

pub(crate) struct DAffine2ScaleIterator<'a> {
    pub(crate) index: usize,
    pub(crate) x_scale: &'a PrimitiveArray<Float64Type>,
    pub(crate) y_scale: &'a PrimitiveArray<Float64Type>,
}

impl<'a> DAffine2ScaleIterator<'a> {
    pub(crate) fn new(array_args: &'a [Arc<dyn Array>]) -> Result<Self> {
        if array_args.len() != 2 {
            return internal_err!("Invalid number of arguments are passed");
        }

        let x_scale = as_float64_array(&array_args[0])?;
        let y_scale = as_float64_array(&array_args[1])?;

        Ok(Self {
            index: 0,
            x_scale,
            y_scale,
        })
    }
}

impl<'a> Iterator for DAffine2ScaleIterator<'a> {
    type Item = glam::DAffine2;

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.index;
        self.index += 1;
        let scale = glam::DVec2::new(self.x_scale.value(i), self.y_scale.value(i));
        Some(glam::DAffine2::from_scale(scale))
    }
}

pub(crate) struct DAffine3ScaleIterator<'a> {
    pub(crate) index: usize,
    pub(crate) x_scale: &'a PrimitiveArray<Float64Type>,
    pub(crate) y_scale: &'a PrimitiveArray<Float64Type>,
    pub(crate) z_scale: &'a PrimitiveArray<Float64Type>,
}

impl<'a> DAffine3ScaleIterator<'a> {
    pub(crate) fn new(array_args: &'a [Arc<dyn Array>]) -> Result<Self> {
        if array_args.len() != 3 {
            return internal_err!("Invalid number of arguments are passed");
        }

        let x_scale = as_float64_array(&array_args[0])?;
        let y_scale = as_float64_array(&array_args[1])?;
        let z_scale = as_float64_array(&array_args[2])?;

        Ok(Self {
            index: 0,
            x_scale,
            y_scale,
            z_scale,
        })
    }
}

impl<'a> Iterator for DAffine3ScaleIterator<'a> {
    type Item = glam::DAffine3;

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.index;
        self.index += 1;
        let scale = glam::DVec3::new(
            self.x_scale.value(i),
            self.y_scale.value(i),
            self.z_scale.value(i),
        );
        Some(glam::DAffine3::from_scale(scale))
    }
}

pub(crate) struct DAffineRotateIterator<'a> {
    pub(crate) index: usize,
    pub(crate) angle: &'a PrimitiveArray<Float64Type>,
}

impl<'a> DAffineRotateIterator<'a> {
    pub(crate) fn new(angle: &'a Arc<dyn Array>) -> Result<Self> {
        let angle = as_float64_array(angle)?;
        Ok(Self { index: 0, angle })
    }
}

impl<'a> Iterator for DAffineRotateIterator<'a> {
    type Item = glam::DAffine2;

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.index;
        self.index += 1;
        Some(glam::DAffine2::from_angle(self.angle.value(i)))
    }
}

pub(crate) enum DAffineIterator<'a> {
    DAffine2(DAffine2Iterator<'a>),
    DAffine3(DAffine3Iterator<'a>),
    DAffine2Scale(DAffine2ScaleIterator<'a>),
    DAffine3Scale(DAffine3ScaleIterator<'a>),
    DAffineRotate(DAffineRotateIterator<'a>),
}

impl<'a> DAffineIterator<'a> {
    pub(crate) fn new_2d(array_args: &'a [Arc<dyn Array>]) -> Result<Self> {
        Ok(Self::DAffine2(DAffine2Iterator::new(array_args)?))
    }

    pub(crate) fn new_3d(array_args: &'a [Arc<dyn Array>]) -> Result<Self> {
        Ok(Self::DAffine3(DAffine3Iterator::new(array_args)?))
    }

    pub(crate) fn from_scale_2d(array_args: &'a [Arc<dyn Array>]) -> Result<Self> {
        Ok(Self::DAffine2Scale(DAffine2ScaleIterator::new(array_args)?))
    }

    pub(crate) fn from_scale_3d(array_args: &'a [Arc<dyn Array>]) -> Result<Self> {
        Ok(Self::DAffine3Scale(DAffine3ScaleIterator::new(array_args)?))
    }

    pub(crate) fn from_angle(angle: &'a Arc<dyn Array>) -> Result<Self> {
        Ok(Self::DAffineRotate(DAffineRotateIterator::new(angle)?))
    }
}

pub(crate) enum DAffine {
    DAffine2(glam::DAffine2),
    DAffine3(glam::DAffine3),
}

impl DAffine {
    pub(crate) fn transform_point3(&self, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        match self {
            DAffine::DAffine2(daffine2) => {
                let transformed = daffine2.transform_point2(glam::DVec2 { x, y });
                (transformed.x, transformed.y, z)
            }
            DAffine::DAffine3(daffine3) => {
                let transformed = daffine3.transform_point3(glam::DVec3 { x, y, z });
                (transformed.x, transformed.y, transformed.z)
            }
        }
    }

    pub(crate) fn transform_point2(&self, x: f64, y: f64) -> (f64, f64) {
        match self {
            DAffine::DAffine2(daffine2) => {
                let transformed = daffine2.transform_point2(glam::DVec2 { x, y });
                (transformed.x, transformed.y)
            }
            DAffine::DAffine3(daffine3) => {
                let transformed = daffine3.transform_point3(glam::DVec3 { x, y, z: 0.0 });
                (transformed.x, transformed.y)
            }
        }
    }
}

impl<'a> Iterator for DAffineIterator<'a> {
    type Item = DAffine;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            DAffineIterator::DAffine2(daffine2_iterator) => {
                daffine2_iterator.next().map(DAffine::DAffine2)
            }
            DAffineIterator::DAffine3(daffine3_iterator) => {
                daffine3_iterator.next().map(DAffine::DAffine3)
            }
            DAffineIterator::DAffine2Scale(daffine2_scale_iterator) => {
                daffine2_scale_iterator.next().map(DAffine::DAffine2)
            }
            DAffineIterator::DAffine3Scale(daffine3_scale_iterator) => {
                daffine3_scale_iterator.next().map(DAffine::DAffine3)
            }
            DAffineIterator::DAffineRotate(daffine_rotate_iterator) => {
                daffine_rotate_iterator.next().map(DAffine::DAffine2)
            }
        }
    }
}
