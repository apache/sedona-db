use std::sync::Arc;

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

use datafusion_common::Result;
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_schema::datatypes::SedonaType;

/// SD_Order() scalar UDF implementation
///
/// This function is invoked to obtain a proxy array whose order may be used
/// to sort based on the value. The default implementation returns the value
/// and a utility is provided to order geometry and/or geographies based on
/// the first coordinate. More sophisticated sorting (e.g., XZ2) may be added
/// in the future.
pub fn sd_order_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "sd_order",
        vec![Arc::new(SDOrderDefault {})],
        Volatility::Immutable,
        Some(sd_order_doc()),
    )
}

fn sd_order_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Return an arbitrary value that may be used to sort the input.",
        "SD_Order (value: Any)",
    )
    .with_argument("value", "Any: An arbitrary value")
    .with_sql_example("SELECT SD_Order(ST_Point(1.0, 2.0, '{}'))")
    .build()
}

/// Default implementation that returns its input (i.e., by default, just
/// do whatever DataFusion would have done with the value)
#[derive(Debug)]
struct SDOrderDefault {}

impl SedonaScalarKernel for SDOrderDefault {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        if args.len() != 1 {
            return Ok(None);
        }

        Ok(Some(args[0].clone()))
    }

    fn invoke_batch(
        &self,
        _arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        Ok(args[0].clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_expr::ScalarUDF;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = sd_order_udf().into();
        assert_eq!(udf.name(), "sd_order");
        assert!(udf.documentation().is_some())
    }
}
