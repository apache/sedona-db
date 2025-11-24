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
use std::{fmt::Debug, sync::Arc, vec};

use arrow_array::{ArrayRef, Float64Array};
use arrow_schema::DataType;

use datafusion_common::{Result, ScalarValue};
use datafusion_expr::{AggregateUDF, ScalarUDF};
use geo_types::Rect;
use rand::{distributions::Uniform, rngs::StdRng, Rng, SeedableRng};

use sedona_common::sedona_internal_err;
use sedona_geometry::types::GeometryTypeId;
use sedona_schema::datatypes::{SedonaType, RASTER, WKB_GEOMETRY};
use sedona_schema::raster::BandDataType;

use crate::{
    datagen::RandomPartitionedDataBuilder,
    rasters::generate_tiled_rasters,
    testers::{AggregateUdfTester, ScalarUdfTester},
};

/// The default number of rows per batch (the same as the DataFusion default)
pub const ROWS_PER_BATCH: usize = 8192;

/// The number of rows per batch to use for tiny size benchmarks
pub const ROWS_PER_BATCH_TINY: usize = 1024;

/// The default number of batches to use for small size benchmarks
///
/// This was chosen to ensure that most benchmarks run nicely with criterion
/// defaults (target 5s, 100 samples).
pub const NUM_BATCHES_SMALL: usize = 16;

/// The default number of batches to use for tiny size benchmarks
///
/// Just one batch for testing that benchmarks actually run.
pub const NUM_BATCHES_TINY: usize = 1;

#[cfg(feature = "criterion")]
pub mod benchmark {
    use super::*;
    use criterion::Criterion;
    use sedona_expr::function_set::FunctionSet;

    /// Benchmark a [ScalarUDF] using [Criterion]
    ///
    /// When built with the criterion feature, provides utilities for running a
    /// basic benchmark on a [ScalarUDF] given [BenchmarkArgs]. This
    /// basic benchmark currently has a hard-coded data size of 16 batches by
    /// 8192 rows (==131,072 rows), which was chosen to ensure that most benchmarks
    /// run nicely with criterion defaults (target 5s, 100 samples).
    pub fn scalar(
        c: &mut Criterion,
        functions: &FunctionSet,
        lib: &str,
        name: &str,
        config: impl Into<BenchmarkArgs>,
    ) {
        let not_found_err = format!("{name} was not found in function set");
        let udf: ScalarUDF = functions
            .scalar_udf(name)
            .expect(&not_found_err)
            .clone()
            .into();
        let data = config
            .into()
            .build_data(
                Config::default().num_batches(),
                Config::default().rows_per_batch(),
            )
            .unwrap();
        c.bench_function(&data.make_label(lib, name), |b| {
            b.iter(|| data.invoke_scalar(&udf).unwrap())
        });
    }

    /// Benchmark a [AggregateUDF] using [Criterion]
    ///
    /// When built with the criterion feature, provides utilities for running a
    /// basic benchmark on a [AggregateUDF] given [BenchmarkArgs]. This
    /// shares a the default batch configuration with [scalar]. Because
    /// aggregate functions can be invoked with varying combinations of
    /// accumulation and merging of states, they should also be benchmarked
    /// at a higher level. This benchmark primarily checks the accumulator.
    pub fn aggregate(
        c: &mut Criterion,
        functions: &FunctionSet,
        lib: &str,
        name: &str,
        config: impl Into<BenchmarkArgs>,
    ) {
        let not_found_err = format!("{name} was not found in function set");
        let udf: AggregateUDF = functions
            .aggregate_udf(name)
            .expect(&not_found_err)
            .clone()
            .into();
        let data = config
            .into()
            .build_data(
                Config::default().num_batches(),
                Config::default().rows_per_batch(),
            )
            .unwrap();
        c.bench_function(&data.make_label(lib, name), |b| {
            b.iter(|| data.invoke_aggregate(&udf).unwrap())
        });
    }

    pub enum Config {
        Tiny,
        Small,
    }

    impl Default for Config {
        fn default() -> Self {
            #[cfg(debug_assertions)]
            return Self::Tiny;

            #[cfg(not(debug_assertions))]
            return Self::Small;
        }
    }

    impl Config {
        fn num_batches(&self) -> usize {
            match self {
                Config::Tiny => NUM_BATCHES_TINY,
                Config::Small => NUM_BATCHES_SMALL,
            }
        }

        fn rows_per_batch(&self) -> usize {
            match self {
                Config::Tiny => ROWS_PER_BATCH_TINY,
                Config::Small => ROWS_PER_BATCH,
            }
        }
    }
}

/// Specification for benchmark arguments
///
/// This provides a concise definition of function input based on a
/// combination of scalar/array arguments each specified by a [BenchmarkArgSpec].
#[derive(Debug, Clone)]
pub enum BenchmarkArgs {
    /// Invoke a unary function with array input
    Array(BenchmarkArgSpec),
    /// Invoke a binary function with scalar and array input
    ScalarArray(BenchmarkArgSpec, BenchmarkArgSpec),
    /// Invoke a binary function with array and scalar input
    ArrayScalar(BenchmarkArgSpec, BenchmarkArgSpec),
    /// Invoke a binary function with two arrays
    ArrayArray(BenchmarkArgSpec, BenchmarkArgSpec),
    /// Invoke a function with an array and two scalar inputs
    ArrayScalarScalar(BenchmarkArgSpec, BenchmarkArgSpec, BenchmarkArgSpec),
    /// Invoke a ternary function with two arrays and a scalar
    ArrayArrayScalar(BenchmarkArgSpec, BenchmarkArgSpec, BenchmarkArgSpec),
    /// Invoke a ternary function with three arrays
    ArrayArrayArray(BenchmarkArgSpec, BenchmarkArgSpec, BenchmarkArgSpec),
    /// Invoke a quaternary function with four arrays
    ArrayArrayArrayArray(
        BenchmarkArgSpec,
        BenchmarkArgSpec,
        BenchmarkArgSpec,
        BenchmarkArgSpec,
    ),
}

impl From<BenchmarkArgSpec> for BenchmarkArgs {
    fn from(value: BenchmarkArgSpec) -> Self {
        BenchmarkArgs::Array(value)
    }
}

impl BenchmarkArgs {
    /// Calculate the [SedonaType]s of the input arguments
    fn sedona_types(&self) -> Vec<SedonaType> {
        self.specs().iter().map(|col| col.sedona_type()).collect()
    }

    /// Build [BenchmarkData] with the specified number of batches
    pub fn build_data(&self, num_batches: usize, rows_per_batch: usize) -> Result<BenchmarkData> {
        let array_configs = match self {
            BenchmarkArgs::Array(_)
            | BenchmarkArgs::ArrayArray(_, _)
            | BenchmarkArgs::ArrayArrayScalar(_, _, _)
            | BenchmarkArgs::ArrayArrayArray(_, _, _)
            | BenchmarkArgs::ArrayArrayArrayArray(_, _, _, _) => self.specs(),
            BenchmarkArgs::ScalarArray(_, col)
            | BenchmarkArgs::ArrayScalar(col, _)
            | BenchmarkArgs::ArrayScalarScalar(col, _, _) => {
                vec![col.clone()]
            }
        };
        let scalar_configs = match self {
            BenchmarkArgs::ScalarArray(col, _)
            | BenchmarkArgs::ArrayScalar(_, col)
            | BenchmarkArgs::ArrayArrayScalar(_, _, col) => {
                vec![col.clone()]
            }
            BenchmarkArgs::ArrayScalarScalar(_, col0, col1) => {
                vec![col0.clone(), col1.clone()]
            }
            _ => vec![],
        };

        let arrays = array_configs
            .iter()
            .enumerate()
            .map(|(i, col)| col.build_arrays(i, num_batches, rows_per_batch))
            .collect::<Result<Vec<_>>>()?;

        let scalars = scalar_configs
            .iter()
            .enumerate()
            .map(|(i, col)| col.build_scalar(i))
            .collect::<Result<Vec<_>>>()?;

        Ok(BenchmarkData {
            config: self.clone(),
            num_batches,
            arrays,
            scalars,
        })
    }

    fn specs(&self) -> Vec<BenchmarkArgSpec> {
        match self {
            BenchmarkArgs::Array(col) => vec![col.clone()],
            BenchmarkArgs::ScalarArray(col0, col1)
            | BenchmarkArgs::ArrayScalar(col0, col1)
            | BenchmarkArgs::ArrayArray(col0, col1) => {
                vec![col0.clone(), col1.clone()]
            }
            BenchmarkArgs::ArrayScalarScalar(col0, col1, col2)
            | BenchmarkArgs::ArrayArrayScalar(col0, col1, col2)
            | BenchmarkArgs::ArrayArrayArray(col0, col1, col2) => {
                vec![col0.clone(), col1.clone(), col2.clone()]
            }
            BenchmarkArgs::ArrayArrayArrayArray(col0, col1, col2, col3) => {
                vec![col0.clone(), col1.clone(), col2.clone(), col3.clone()]
            }
        }
    }
}

/// Specification of a single argument to a function
///
/// Geometries are generated using the [RandomPartitionedDataBuilder], which offers
/// more specific options for generating random geometries.
#[derive(Clone)]
pub enum BenchmarkArgSpec {
    /// Randomly generated point input
    Point,
    /// Randomly generated linestring input with a specified number of vertices
    LineString(usize),
    /// Randomly generated polygon input with a specified number of vertices
    Polygon(usize),
    /// Randomly generated linestring input with a specified number of vertices
    MultiPoint(usize),
    /// Randomly generated floating point input with a given range of values
    Float64(f64, f64),
    /// A transformation of any of the above based on a [ScalarUDF] accepting
    /// a single argument
    Transformed(Box<BenchmarkArgSpec>, ScalarUDF),
    /// A string that will be a constant
    String(String),
    /// Randomly generated raster input with a specified width, height
    Raster(usize, usize),
}

// Custom implementation of Debug because otherwise the output of Transformed()
// is excessively verbose
impl Debug for BenchmarkArgSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Point => write!(f, "Point"),
            Self::LineString(arg0) => f.debug_tuple("LineString").field(arg0).finish(),
            Self::Polygon(arg0) => f.debug_tuple("Polygon").field(arg0).finish(),
            Self::MultiPoint(arg0) => f.debug_tuple("MultiPoint").field(arg0).finish(),
            Self::Float64(arg0, arg1) => f.debug_tuple("Float64").field(arg0).field(arg1).finish(),
            Self::Transformed(inner, t) => write!(f, "{}({:?})", t.name(), inner),
            Self::String(s) => write!(f, "String({s})"),
            Self::Raster(w, h) => f.debug_tuple("Raster").field(w).field(h).finish(),
        }
    }
}

impl BenchmarkArgSpec {
    /// The [SedonaType] of this argument
    pub fn sedona_type(&self) -> SedonaType {
        match self {
            BenchmarkArgSpec::Point
            | BenchmarkArgSpec::Polygon(_)
            | BenchmarkArgSpec::LineString(_)
            | BenchmarkArgSpec::MultiPoint(_) => WKB_GEOMETRY,
            BenchmarkArgSpec::Float64(_, _) => SedonaType::Arrow(DataType::Float64),
            BenchmarkArgSpec::Transformed(inner, t) => {
                let tester = ScalarUdfTester::new(t.clone(), vec![inner.sedona_type()]);
                tester.return_type().unwrap()
            }
            BenchmarkArgSpec::String(_) => SedonaType::Arrow(DataType::Utf8),
            BenchmarkArgSpec::Raster(_, _) => RASTER,
        }
    }

    /// Build a [ScalarValue] for this argument
    ///
    /// This currently builds the same non-null scalar for each unique value
    /// of i (the argument number).
    pub fn build_scalar(&self, i: usize) -> Result<ScalarValue> {
        let array = self.build_arrays(i, 1, 1)?;
        ScalarValue::try_from_array(&array[0], 0)
    }

    /// Build a column of num_batches arrays
    ///
    /// This currently builds the same column for each unique value of i (the argument
    /// number). The batch size is currently fixed to 8192 (the DataFusion default).
    pub fn build_arrays(
        &self,
        i: usize,
        num_batches: usize,
        rows_per_batch: usize,
    ) -> Result<Vec<ArrayRef>> {
        match self {
            BenchmarkArgSpec::Point => {
                self.build_geometry(i, GeometryTypeId::Point, num_batches, 1, 1, rows_per_batch)
            }
            BenchmarkArgSpec::LineString(vertex_count) => self.build_geometry(
                i,
                GeometryTypeId::LineString,
                num_batches,
                *vertex_count,
                1,
                rows_per_batch,
            ),
            BenchmarkArgSpec::Polygon(vertex_count) => self.build_geometry(
                i,
                GeometryTypeId::Polygon,
                num_batches,
                *vertex_count,
                1,
                rows_per_batch,
            ),
            BenchmarkArgSpec::MultiPoint(part_count) => self.build_geometry(
                i,
                GeometryTypeId::MultiPoint,
                num_batches,
                1,
                *part_count,
                rows_per_batch,
            ),
            BenchmarkArgSpec::Float64(lo, hi) => {
                let mut rng = self.rng(i);
                let dist = Uniform::new(lo, hi);
                (0..num_batches)
                    .map(|_| -> Result<ArrayRef> {
                        let float64_array: Float64Array =
                            (0..rows_per_batch).map(|_| rng.sample(dist)).collect();
                        Ok(Arc::new(float64_array))
                    })
                    .collect()
            }
            BenchmarkArgSpec::Transformed(inner, t) => {
                let inner_type = inner.sedona_type();
                let inner_arrays = inner.build_arrays(i, num_batches, rows_per_batch)?;
                let tester = ScalarUdfTester::new(t.clone(), vec![inner_type]);
                inner_arrays
                    .into_iter()
                    .map(|array| tester.invoke_array(array))
                    .collect::<Result<Vec<_>>>()
            }
            BenchmarkArgSpec::String(s) => {
                let string_array = (0..num_batches)
                    .map(|_| {
                        let array = arrow_array::StringArray::from_iter_values(
                            std::iter::repeat_n(s, rows_per_batch),
                        );
                        Ok(Arc::new(array) as ArrayRef)
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(string_array)
            }
            BenchmarkArgSpec::Raster(width, height) => {
                let mut arrays = vec![];
                for _ in 0..num_batches {
                    let tile_size = (*width, *height);
                    let tile_count = (rows_per_batch, 1);
                    let raster = generate_tiled_rasters(
                        tile_size,
                        tile_count,
                        BandDataType::UInt8,
                        Some(43),
                    )?;
                    arrays.push(Arc::new(raster) as ArrayRef);
                }
                Ok(arrays)
            }
        }
    }

    fn build_geometry(
        &self,
        i: usize,
        geom_type: GeometryTypeId,
        num_batches: usize,
        vertex_count: usize,
        num_parts_count: usize,
        rows_per_batch: usize,
    ) -> Result<Vec<ArrayRef>> {
        let builder = RandomPartitionedDataBuilder::new()
            .num_partitions(1)
            .rows_per_batch(rows_per_batch)
            .batches_per_partition(num_batches)
            // Use a random geometry range that is also not unrealistic for geography
            .bounds(Rect::new((-10.0, -10.0), (10.0, 10.0)))
            .size_range((0.1, 2.0))
            .vertices_per_linestring_range((vertex_count, vertex_count))
            .num_parts_range((num_parts_count, num_parts_count))
            .geometry_type(geom_type)
            // Currently just use WKB_GEOMETRY (we can generate a view type with
            // Transformed)
            .sedona_type(WKB_GEOMETRY);

        builder
            .partition_reader(self.rng(i), 0)
            .map(|batch| -> Result<ArrayRef> { Ok(batch?.column(2).clone()) })
            .collect()
    }

    fn rng(&self, i: usize) -> impl Rng {
        StdRng::seed_from_u64(42 + i as u64)
    }
}

/// Fully resolved data ready for running a benchmark
///
/// This struct contains the fully built data (such that benchmarks do not
/// measure the time required to build the data) and has methods for invoking
/// functions on it.
pub struct BenchmarkData {
    config: BenchmarkArgs,
    num_batches: usize,
    arrays: Vec<Vec<ArrayRef>>,
    scalars: Vec<ScalarValue>,
}

impl BenchmarkData {
    /// Create a label based on the library, function name, and configuration
    pub fn make_label(&self, lib: &str, name: &str) -> String {
        format!("{lib}-{name}-{:?}", self.config)
    }

    /// Invoke a scalar function on this data
    pub fn invoke_scalar(&self, udf: &ScalarUDF) -> Result<()> {
        let tester = ScalarUdfTester::new(udf.clone(), self.config.sedona_types().clone());

        match self.config {
            BenchmarkArgs::Array(_) => {
                for i in 0..self.num_batches {
                    tester.invoke_array(self.arrays[0][i].clone())?;
                }
            }
            BenchmarkArgs::ScalarArray(_, _) => {
                let scalar = &self.scalars[0];
                for i in 0..self.num_batches {
                    tester.invoke_scalar_array(scalar.clone(), self.arrays[0][i].clone())?;
                }
            }
            BenchmarkArgs::ArrayScalar(_, _) => {
                let scalar = &self.scalars[0];
                for i in 0..self.num_batches {
                    tester.invoke_array_scalar(self.arrays[0][i].clone(), scalar.clone())?;
                }
            }
            BenchmarkArgs::ArrayArray(_, _) => {
                for i in 0..self.num_batches {
                    tester
                        .invoke_array_array(self.arrays[0][i].clone(), self.arrays[1][i].clone())?;
                }
            }
            BenchmarkArgs::ArrayScalarScalar(_, _, _) => {
                let scalar0 = &self.scalars[0];
                let scalar1 = &self.scalars[1];
                for i in 0..self.num_batches {
                    tester.invoke_array_scalar_scalar(
                        self.arrays[0][i].clone(),
                        scalar0.clone(),
                        scalar1.clone(),
                    )?;
                }
            }
            BenchmarkArgs::ArrayArrayScalar(_, _, _) => {
                for i in 0..self.num_batches {
                    tester.invoke_array_array_scalar(
                        self.arrays[0][i].clone(),
                        self.arrays[1][i].clone(),
                        self.scalars[0].clone(),
                    )?;
                }
            }
            BenchmarkArgs::ArrayArrayArray(_, _, _) => {
                for i in 0..self.num_batches {
                    tester.invoke_arrays(vec![
                        self.arrays[0][i].clone(),
                        self.arrays[1][i].clone(),
                        self.arrays[2][i].clone(),
                    ])?;
                }
            }
            BenchmarkArgs::ArrayArrayArrayArray(_, _, _, _) => {
                for i in 0..self.num_batches {
                    tester.invoke_arrays(vec![
                        self.arrays[0][i].clone(),
                        self.arrays[1][i].clone(),
                        self.arrays[2][i].clone(),
                        self.arrays[3][i].clone(),
                    ])?;
                }
            }
        }

        Ok(())
    }

    /// Invoke an aggregate function on this data
    pub fn invoke_aggregate(&self, udf: &AggregateUDF) -> Result<ScalarValue> {
        if !matches!(self.config, BenchmarkArgs::Array(_)) {
            return sedona_internal_err!(
                "invoke_aggregate() not implemented for {:?}",
                self.config
            );
        }

        let tester = AggregateUdfTester::new(udf.clone(), self.config.sedona_types().clone());
        tester.aggregate(&self.arrays[0])
    }
}

#[cfg(test)]
mod test {
    use arrow_array::{Array, StructArray};
    use datafusion_common::cast::as_binary_array;
    use datafusion_expr::{ColumnarValue, SimpleScalarUDF};
    use geo_traits::Dimensions;
    use rstest::rstest;
    use sedona_geometry::{analyze::analyze_geometry, types::GeometryTypeAndDimensions};

    use super::*;

    #[test]
    fn arg_spec_scalar() {
        let spec = BenchmarkArgSpec::Point;
        assert_eq!(spec.sedona_type(), WKB_GEOMETRY);

        let scalar = spec.build_scalar(0).unwrap();

        // Make sure this is deterministic
        assert_eq!(spec.build_scalar(0).unwrap(), scalar);

        // Make sure we generate different scalars for different columns
        assert_ne!(spec.build_scalar(1).unwrap(), scalar);

        if let ScalarValue::Binary(Some(wkb_bytes)) = scalar {
            let wkb = wkb::reader::read_wkb(&wkb_bytes).unwrap();
            let analysis = analyze_geometry(&wkb).unwrap();
            assert_eq!(analysis.point_count, 1);
            assert_eq!(
                analysis.geometry_type,
                GeometryTypeAndDimensions::new(GeometryTypeId::Point, Dimensions::Xy)
            )
        } else {
            unreachable!("Unexpected scalar output {scalar}")
        }
    }

    #[rstest]
    fn arg_spec_geometry(
        #[values(
            (BenchmarkArgSpec::Point, GeometryTypeId::Point, 1),
            (BenchmarkArgSpec::LineString(10), GeometryTypeId::LineString, 10),
            (BenchmarkArgSpec::Polygon(10), GeometryTypeId::Polygon, 11),
            (BenchmarkArgSpec::MultiPoint(10), GeometryTypeId::MultiPoint, 10),
        )]
        config: (BenchmarkArgSpec, GeometryTypeId, i64),
    ) {
        let (spec, geometry_type, point_count) = config;
        assert_eq!(spec.sedona_type(), WKB_GEOMETRY);

        let arrays = spec.build_arrays(0, 2, ROWS_PER_BATCH).unwrap();
        assert_eq!(arrays.len(), 2);

        // Make sure this is deterministic
        assert_eq!(spec.build_arrays(0, 2, ROWS_PER_BATCH).unwrap(), arrays);

        // Make sure we generate different arrays for different argument numbers
        assert_ne!(spec.build_arrays(1, 2, ROWS_PER_BATCH).unwrap(), arrays);

        for array in arrays {
            assert_eq!(array.data_type(), WKB_GEOMETRY.storage_type());
            assert_eq!(array.len(), ROWS_PER_BATCH);

            let binary_array = as_binary_array(&array).unwrap();
            assert_eq!(binary_array.null_count(), 0);

            for wkb_bytes in binary_array {
                let wkb = wkb::reader::read_wkb(wkb_bytes.unwrap()).unwrap();
                let analysis = analyze_geometry(&wkb).unwrap();
                assert_eq!(analysis.point_count, point_count);
                assert_eq!(
                    analysis.geometry_type,
                    GeometryTypeAndDimensions::new(geometry_type, Dimensions::Xy)
                )
            }
        }
    }

    #[test]
    fn arg_spec_float() {
        let spec = BenchmarkArgSpec::Float64(1.0, 2.0);
        assert_eq!(spec.sedona_type(), SedonaType::Arrow(DataType::Float64));

        let arrays = spec.build_arrays(0, 2, ROWS_PER_BATCH).unwrap();
        assert_eq!(arrays.len(), 2);

        // Make sure this is deterministic
        assert_eq!(spec.build_arrays(0, 2, ROWS_PER_BATCH).unwrap(), arrays);

        // Make sure we generate different arrays for different argument numbers
        assert_ne!(spec.build_arrays(1, 2, ROWS_PER_BATCH).unwrap(), arrays);

        for array in arrays {
            assert_eq!(array.data_type(), &DataType::Float64);
            assert_eq!(array.len(), ROWS_PER_BATCH);
            assert_eq!(array.null_count(), 0);
        }
    }

    #[test]
    fn arg_spec_transformed() {
        let udf = SimpleScalarUDF::new(
            "float32",
            vec![DataType::Float64],
            DataType::Float32,
            datafusion_expr::Volatility::Immutable,
            Arc::new(|args| -> Result<ColumnarValue> { args[0].cast_to(&DataType::Float32, None) }),
        );

        let spec =
            BenchmarkArgSpec::Transformed(BenchmarkArgSpec::Float64(1.0, 2.0).into(), udf.into());
        assert_eq!(spec.sedona_type(), SedonaType::Arrow(DataType::Float32));

        assert_eq!(format!("{spec:?}"), "float32(Float64(1.0, 2.0))");
        let arrays = spec.build_arrays(0, 2, ROWS_PER_BATCH).unwrap();
        assert_eq!(arrays.len(), 2);

        // Make sure this is deterministic
        assert_eq!(spec.build_arrays(0, 2, ROWS_PER_BATCH).unwrap(), arrays);

        // Make sure we generate different arrays for different argument numbers
        assert_ne!(spec.build_arrays(1, 2, ROWS_PER_BATCH).unwrap(), arrays);

        for array in arrays {
            assert_eq!(array.data_type(), &DataType::Float32);
            assert_eq!(array.len(), ROWS_PER_BATCH);
            assert_eq!(array.null_count(), 0);
        }
    }

    #[test]
    fn args_array() {
        let spec = BenchmarkArgs::Array(BenchmarkArgSpec::Point);
        assert_eq!(spec.sedona_types(), [WKB_GEOMETRY]);

        let data = spec.build_data(2, ROWS_PER_BATCH).unwrap();
        assert_eq!(data.num_batches, 2);
        assert_eq!(data.arrays.len(), 1);
        assert_eq!(data.scalars.len(), 0);

        assert_eq!(data.arrays[0].len(), 2);
        assert_eq!(WKB_GEOMETRY.storage_type(), data.arrays[0][0].data_type());
    }

    #[test]
    fn args_array_scalar() {
        let spec = BenchmarkArgs::ArrayScalar(
            BenchmarkArgSpec::Point,
            BenchmarkArgSpec::Float64(1.0, 2.0),
        );
        assert_eq!(
            spec.sedona_types(),
            [WKB_GEOMETRY, SedonaType::Arrow(DataType::Float64)]
        );

        let data = spec.build_data(2, ROWS_PER_BATCH).unwrap();
        assert_eq!(data.num_batches, 2);

        assert_eq!(data.arrays.len(), 1);
        assert_eq!(data.arrays[0].len(), 2);
        assert_eq!(WKB_GEOMETRY.storage_type(), data.arrays[0][0].data_type());

        assert_eq!(data.scalars.len(), 1);
        assert_eq!(data.scalars[0].data_type(), DataType::Float64);
    }

    #[test]
    fn args_scalar_array() {
        let spec = BenchmarkArgs::ScalarArray(
            BenchmarkArgSpec::Point,
            BenchmarkArgSpec::Float64(1.0, 2.0),
        );
        assert_eq!(
            spec.sedona_types(),
            [WKB_GEOMETRY, SedonaType::Arrow(DataType::Float64)]
        );

        let data = spec.build_data(2, ROWS_PER_BATCH).unwrap();
        assert_eq!(data.num_batches, 2);

        assert_eq!(data.scalars.len(), 1);
        assert_eq!(WKB_GEOMETRY.storage_type(), &data.scalars[0].data_type());

        assert_eq!(data.arrays.len(), 1);
        assert_eq!(data.arrays[0].len(), 2);
        assert_eq!(data.arrays[0][0].data_type(), &DataType::Float64);
    }

    #[test]
    fn args_array_array() {
        let spec =
            BenchmarkArgs::ArrayArray(BenchmarkArgSpec::Point, BenchmarkArgSpec::Float64(1.0, 2.0));
        assert_eq!(
            spec.sedona_types(),
            [WKB_GEOMETRY, SedonaType::Arrow(DataType::Float64)]
        );

        let data = spec.build_data(2, ROWS_PER_BATCH).unwrap();
        assert_eq!(data.num_batches, 2);
        assert_eq!(data.arrays.len(), 2);
        assert_eq!(data.scalars.len(), 0);

        assert_eq!(data.arrays[0].len(), 2);
        assert_eq!(WKB_GEOMETRY.storage_type(), data.arrays[0][0].data_type());

        assert_eq!(data.arrays[1].len(), 2);
        assert_eq!(data.arrays[1][0].data_type(), &DataType::Float64);
    }

    #[test]
    fn args_array_scalar_scalar() {
        let spec = BenchmarkArgs::ArrayScalarScalar(
            BenchmarkArgSpec::Point,
            BenchmarkArgSpec::Float64(1.0, 2.0),
            BenchmarkArgSpec::String("test".to_string()),
        );
        assert_eq!(
            spec.sedona_types(),
            [
                WKB_GEOMETRY,
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Utf8)
            ]
        );

        let data = spec.build_data(2, ROWS_PER_BATCH).unwrap();
        assert_eq!(data.num_batches, 2);
        assert_eq!(data.arrays.len(), 1);
        assert_eq!(data.scalars.len(), 2);
        assert_eq!(data.arrays[0].len(), 2);
        assert_eq!(WKB_GEOMETRY.storage_type(), data.arrays[0][0].data_type());
        assert_eq!(data.scalars[0].data_type(), DataType::Float64);
        assert_eq!(data.scalars[1].data_type(), DataType::Utf8);
    }

    #[test]
    fn args_array_array_scalar() {
        let spec = BenchmarkArgs::ArrayArrayScalar(
            BenchmarkArgSpec::Point,
            BenchmarkArgSpec::Point,
            BenchmarkArgSpec::Float64(1.0, 2.0),
        );
        assert_eq!(
            spec.sedona_types(),
            [
                WKB_GEOMETRY,
                WKB_GEOMETRY,
                SedonaType::Arrow(DataType::Float64)
            ]
        );

        let data = spec.build_data(2, ROWS_PER_BATCH).unwrap();
        assert_eq!(data.num_batches, 2);
        assert_eq!(data.arrays.len(), 3);
        assert_eq!(data.scalars.len(), 1);
        assert_eq!(data.arrays[0].len(), 2);
        assert_eq!(WKB_GEOMETRY.storage_type(), data.arrays[0][0].data_type());
        assert_eq!(data.arrays[1].len(), 2);
        assert_eq!(WKB_GEOMETRY.storage_type(), data.arrays[1][0].data_type());

        assert_eq!(data.scalars[0].data_type(), DataType::Float64);
    }

    #[test]
    fn args_array_array_array() {
        let spec = BenchmarkArgs::ArrayArrayArray(
            BenchmarkArgSpec::Point,
            BenchmarkArgSpec::Point,
            BenchmarkArgSpec::Float64(1.0, 2.0),
        );
        assert_eq!(
            spec.sedona_types(),
            [
                WKB_GEOMETRY,
                WKB_GEOMETRY,
                SedonaType::Arrow(DataType::Float64)
            ]
        );

        let data = spec.build_data(2, ROWS_PER_BATCH).unwrap();
        assert_eq!(data.num_batches, 2);
        assert_eq!(data.arrays.len(), 3);
        assert_eq!(data.scalars.len(), 0);
        assert_eq!(data.arrays[0].len(), 2);
        assert_eq!(WKB_GEOMETRY.storage_type(), data.arrays[0][0].data_type());
        assert_eq!(data.arrays[1].len(), 2);
        assert_eq!(WKB_GEOMETRY.storage_type(), data.arrays[1][0].data_type());
        assert_eq!(data.arrays[2].len(), 2);
        assert_eq!(data.arrays[2][0].data_type(), &DataType::Float64);
    }

    #[test]
    fn args_array_array_array_array() {
        let spec = BenchmarkArgs::ArrayArrayArrayArray(
            BenchmarkArgSpec::Float64(1.0, 2.0),
            BenchmarkArgSpec::Float64(3.0, 4.0),
            BenchmarkArgSpec::Float64(5.0, 6.0),
            BenchmarkArgSpec::Float64(7.0, 8.0),
        );
        assert_eq!(
            spec.sedona_types(),
            [
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64)
            ]
        );

        let data = spec.build_data(2, ROWS_PER_BATCH).unwrap();
        assert_eq!(data.num_batches, 2);
        assert_eq!(data.arrays.len(), 4);
        assert_eq!(data.scalars.len(), 0);
        assert_eq!(data.arrays[0].len(), 2);
        assert_eq!(data.arrays[0][0].data_type(), &DataType::Float64);
        assert_eq!(data.arrays[1].len(), 2);
        assert_eq!(data.arrays[1][0].data_type(), &DataType::Float64);
        assert_eq!(data.arrays[2].len(), 2);
        assert_eq!(data.arrays[2][0].data_type(), &DataType::Float64);
        assert_eq!(data.arrays[3].len(), 2);
        assert_eq!(data.arrays[3][0].data_type(), &DataType::Float64);
    }

    #[test]
    fn arg_spec_raster() {
        use sedona_raster::array::RasterStructArray;
        use sedona_raster::traits::RasterRef;

        let spec = BenchmarkArgSpec::Raster(10, 5);
        assert_eq!(spec.sedona_type(), RASTER);
        let data = spec.build_arrays(0, 2, ROWS_PER_BATCH).unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(data[0].data_type(), RASTER.storage_type());

        let raster_array = data[0].as_any().downcast_ref::<StructArray>().unwrap();
        let rasters = RasterStructArray::new(raster_array);
        assert_eq!(rasters.len(), ROWS_PER_BATCH);
        let raster = rasters.get(0).unwrap();
        let metadata = raster.metadata();
        assert_eq!(metadata.width(), 10);
        assert_eq!(metadata.height(), 5);
    }
}
