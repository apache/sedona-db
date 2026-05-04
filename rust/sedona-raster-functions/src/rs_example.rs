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
use datafusion_common::error::Result;
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::builder::RasterBuilder;
use sedona_schema::{
    crs::lnglat, datatypes::SedonaType, matchers::ArgMatcher, raster::BandDataType,
};

/// RS_Example() scalar UDF implementation
///
/// Creates a simple concrete example for testing purposes
/// May expand with additional optional parameters in the future
pub fn rs_example_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_example",
        vec![Arc::new(RsExample {})],
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct RsExample {}

impl SedonaScalarKernel for RsExample {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(vec![], SedonaType::Raster);

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let mut builder = RasterBuilder::new(1);

        let width: u64 = 64;
        let height: u64 = 32;
        let crs = lnglat().unwrap().to_crs_string();
        builder.start_raster_2d(width, height, 43.08, 79.07, 2.0, 2.0, 1.0, 1.0, Some(&crs))?;
        let nodata_value = 127u8;
        for band_id in 1..=3 {
            builder.start_band_2d(BandDataType::UInt8, Some(&[nodata_value]))?;

            let mut band_data = vec![band_id as u8; (width * height) as usize];
            band_data[0] = nodata_value; // set the top corner to nodata

            builder.band_data_writer().append_value(&band_data);
            builder.finish_band()?;
        }
        builder.finish_raster()?;

        executor.finish(Arc::new(builder.finish()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_common::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use sedona_raster::array::RasterStructArray;
    use sedona_raster::traits::RasterRef;

    #[test]
    fn udf_size() {
        let udf: ScalarUDF = rs_example_udf().into();
        assert_eq!(udf.name(), "rs_example");
    }

    #[test]
    fn udf_invoke() {
        let kernel = RsExample {};
        let args = [];
        let arg_types = vec![];

        let result = kernel.invoke_batch(&arg_types, &args).unwrap();
        if let ColumnarValue::Scalar(ScalarValue::Struct(arc_struct)) = result {
            let raster_array = RasterStructArray::new(arc_struct.as_ref());

            assert_eq!(raster_array.len(), 1);
            let raster = raster_array.get(0).unwrap();
            assert_eq!(raster.width().unwrap(), 64);
            assert_eq!(raster.height().unwrap(), 32);

            let band = raster.band(0).unwrap();
            assert_eq!(band.data_type(), BandDataType::UInt8);
            assert_eq!(band.nodata(), Some(&[127u8][..]));
            assert!(band.outdb_uri().is_none());
        } else {
            panic!("Expected scalar struct result");
        }
    }
}
