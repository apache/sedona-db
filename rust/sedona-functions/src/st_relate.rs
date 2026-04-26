use arrow_array::builder::StringBuilder;
use arrow_schema::DataType;
use datafusion_common::{error::Result, DataFusionError};
use datafusion_expr::{ColumnarValue, Volatility};

use geos::{Geom, Geometry as GeosGeometry};

use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{SedonaScalarKernel, SedonaScalarUDF},
};
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};
use std::sync::Arc;
use crate::executor::WkbExecutor;

/// ST_Relate() scalar UDF
///
/// Returns a 9-character text string representing the DE-9IM relationship matrix
pub fn st_relate_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_relate",
        ItemCrsKernel::wrap_impl(vec![Arc::new(STRelate)]),
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct STRelate;

impl SedonaScalarKernel for STRelate {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry(), ArgMatcher::is_geometry()],
            SedonaType::Arrow(DataType::Utf8),
        );
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let mut executor = WkbExecutor::new(arg_types, args);
        let mut builder = StringBuilder::with_capacity(
            executor.num_iterations(),
            9 * executor.num_iterations(),
        );

        executor.execute(|args| {
            match (args[0], args[1]) {
                (Some(wkb1), Some(wkb2)) => {
                    // Convert Sedona WKB to GEOS Geometry
                    let g1 = GeosGeometry::try_from(wkb1)
                        .map_err(|e| DataFusionError::Execution(format!("GEOS conversion error: {}", e)))?;
                    let g2 = GeosGeometry::try_from(wkb2)
                        .map_err(|e| DataFusionError::Execution(format!("GEOS conversion error: {}", e)))?;

                    let matrix = g1.relate(&g2)
                        .map_err(|e| DataFusionError::Execution(format!("ST_Relate error: {}", e)))?;

                    builder.append_value(matrix);
                }
                _ => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}