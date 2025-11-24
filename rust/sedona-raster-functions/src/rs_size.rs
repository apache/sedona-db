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
use std::{sync::Arc, vec};

use crate::executor::RasterExecutor;
use arrow_array::builder::UInt64Builder;
use arrow_schema::DataType;
use datafusion_common::error::Result;
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::traits::RasterRef;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

/// RS_Width() scalar UDF implementation
///
/// Extract the width of the raster
pub fn rs_width_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_width",
        vec![Arc::new(RsSize {
            size_type: SizeType::Width,
        })],
        Volatility::Immutable,
        Some(rs_width_doc()),
    )
}

/// RS_Height() scalar UDF documentation
///
/// Extract the height of the raster
pub fn rs_height_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_height",
        vec![Arc::new(RsSize {
            size_type: SizeType::Height,
        })],
        Volatility::Immutable,
        Some(rs_height_doc()),
    )
}

fn rs_width_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Return the width component of a raster".to_string(),
        "RS_Width(raster: Raster)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_sql_example("SELECT RS_Width(RS_Example())".to_string())
    .build()
}

fn rs_height_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Return the height component of a raster".to_string(),
        "RS_Height(raster: Raster)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_sql_example("SELECT RS_Height(RS_Example())".to_string())
    .build()
}

#[derive(Debug, Clone)]
enum SizeType {
    Width,
    Height,
}

#[derive(Debug)]
struct RsSize {
    size_type: SizeType,
}

impl SedonaScalarKernel for RsSize {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_raster()],
            SedonaType::Arrow(DataType::UInt64),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let mut builder = UInt64Builder::with_capacity(executor.num_iterations());

        executor.execute_raster_void(|_i, raster_opt| {
            match raster_opt {
                None => builder.append_null(),
                Some(raster) => match self.size_type {
                    SizeType::Width => {
                        let width = raster.metadata().width();
                        builder.append_value(width);
                    }
                    SizeType::Height => {
                        let height = raster.metadata().height();
                        builder.append_value(height);
                    }
                },
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::UInt64Array;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::RASTER;
    use sedona_testing::compare::assert_array_equal;
    use sedona_testing::rasters::generate_test_rasters;
    use sedona_testing::testers::ScalarUdfTester;

    #[test]
    fn udf_size() {
        let udf: ScalarUDF = rs_width_udf().into();
        assert_eq!(udf.name(), "rs_width");
        assert!(udf.documentation().is_some());

        let udf: ScalarUDF = rs_height_udf().into();
        assert_eq!(udf.name(), "rs_height");
        assert!(udf.documentation().is_some());
    }

    #[rstest]
    fn udf_invoke(#[values(SizeType::Width, SizeType::Height)] st: SizeType) {
        let udf = match st {
            SizeType::Height => rs_height_udf(),
            SizeType::Width => rs_width_udf(),
        };
        let tester = ScalarUdfTester::new(udf.into(), vec![RASTER]);

        let rasters = generate_test_rasters(3, Some(1)).unwrap();
        let expected_values = match st {
            SizeType::Height => vec![Some(2), None, Some(4)],
            SizeType::Width => vec![Some(1), None, Some(3)],
        };
        let expected: Arc<dyn arrow_array::Array> = Arc::new(UInt64Array::from(expected_values));

        // Check scalars
        let result = tester.invoke_array(Arc::new(rasters)).unwrap();
        assert_array_equal(&result, &expected);
    }
}
