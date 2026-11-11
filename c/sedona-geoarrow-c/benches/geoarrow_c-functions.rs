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
use arrow_schema::DataType;
use datafusion_common::Result;
use datafusion_expr::ColumnarValue;
use sedona_expr::scalar_udf::SedonaScalarKernel;
use sedona_functions::executor::WkbExecutor;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};
use sedona_testing::benchmark_util::BenchmarkArgs;

// Keep the comparison implementation out of the public library API.
#[allow(dead_code)]
#[path = "../src/error.rs"]
mod error;
#[allow(dead_code, unused_imports)]
#[path = "../src/geoarrow_c.rs"]
mod geoarrow_c;
#[allow(dead_code)]
#[path = "../src/geoarrow_c_bindgen.rs"]
mod geoarrow_c_bindgen;

use criterion::{Criterion, criterion_group, criterion_main};
use datafusion_expr::ScalarUDF;
use geoarrow_c::{ArrayReader, ArrayWriter, CError};
use sedona_testing::benchmark_util::{BenchmarkArgSpec::*, benchmark};

/// The previous whole-array visitor path, using the same upstream snapshot and
/// output writer as the geometry-view kernels.
#[derive(Debug)]
struct VisitorAsText;

impl SedonaScalarKernel for VisitorAsText {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        ArgMatcher::new(
            vec![ArgMatcher::is_geometry_or_geography()],
            SedonaType::Arrow(DataType::Utf8),
        )
        .match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);
        let array = args[0].to_array(executor.num_iterations())?;
        let mut reader = ArrayReader::try_new(&arg_types[0])?;
        let mut writer = ArrayWriter::try_new(&SedonaType::Arrow(DataType::Utf8))?;
        reader.visit(&array, writer.visitor_mut(), &mut CError::new())?;
        executor.finish(writer.finish()?)
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut f = sedona_functions::register::default_function_set();
    for (name, kernels) in sedona_geoarrow_c::register::scalar_kernels() {
        f.add_scalar_udf_impl(name, kernels).unwrap();
    }

    let mut visitor_functions = sedona_functions::register::default_function_set();
    visitor_functions
        .add_scalar_udf_impl("st_astext", VisitorAsText)
        .unwrap();
    for geometry in [Point, LineString(10), LineString(100), Polygon(100)] {
        benchmark::scalar(
            c,
            &visitor_functions,
            "geoarrow_c_visitor",
            "st_astext",
            geometry.clone(),
        );
        benchmark::scalar(c, &f, "geoarrow_c_geometry", "st_astext", geometry.clone());
        benchmark::scalar(
            c,
            &f,
            "geoarrow_c_precision",
            "st_astext",
            BenchmarkArgs::ArrayScalar(geometry.clone(), Int64(6, 7)),
        );
        benchmark::scalar(
            c,
            &f,
            "geoarrow_c_precision",
            "st_astext",
            BenchmarkArgs::ArrayArray(geometry, Int64(0, 17)),
        );
    }

    let st_asbinary: ScalarUDF = f.scalar_udf("st_asbinary").unwrap().clone().into();
    let st_astext: ScalarUDF = f.scalar_udf("st_astext").unwrap().clone().into();

    benchmark::scalar(
        c,
        &f,
        "geoarrow_c",
        "st_geomfromwkb",
        Transformed(Point.into(), st_asbinary.clone()),
    );
    benchmark::scalar(
        c,
        &f,
        "geoarrow_c",
        "st_geomfromwkt",
        Transformed(Point.into(), st_astext.clone()),
    );
    benchmark::scalar(
        c,
        &f,
        "geoarrow_c",
        "st_geomfromwkt",
        Transformed(LineString(10).into(), st_astext.clone()),
    );
}

/// Compare the default Rust WKT kernels used before geoarrow-c registration
/// with the replacement kernels. Both implementations receive the same data;
/// text input is generated once with the previous writer, outside the timings.
fn non_precision_benchmark(c: &mut Criterion) {
    let previous = sedona_functions::register::default_function_set();
    let mut current = sedona_functions::register::default_function_set();
    for (name, kernels) in sedona_geoarrow_c::register::scalar_kernels() {
        current.add_scalar_udf_impl(name, kernels).unwrap();
    }
    let previous_writer: ScalarUDF = previous.scalar_udf("st_astext").unwrap().clone().into();
    let mut group = c.benchmark_group("wkt_non_precision");
    // One standard-size batch keeps larger geometries practical for repeated runs.
    group.throughput(criterion::Throughput::Elements(8192));
    for geometry in [Point, LineString(10), LineString(100), Polygon(100)] {
        for (operation, function, input) in [
            ("write", "st_astext", geometry.clone()),
            (
                "read",
                "st_geomfromwkt",
                Transformed(geometry.clone().into(), previous_writer.clone()),
            ),
        ] {
            let data = BenchmarkArgs::Array(input).build_data(1, 8192).unwrap();
            for (implementation, functions) in
                [("previous_rust", &previous), ("geoarrow_c", &current)]
            {
                let udf: ScalarUDF = functions.scalar_udf(function).unwrap().clone().into();
                let id = criterion::BenchmarkId::new(
                    format!("{operation}/{geometry:?}"),
                    implementation,
                );
                group.bench_with_input(id, &udf, |b, udf| {
                    b.iter(|| data.invoke_scalar(udf).unwrap());
                });
            }
        }
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark, non_precision_benchmark);
criterion_main!(benches);
