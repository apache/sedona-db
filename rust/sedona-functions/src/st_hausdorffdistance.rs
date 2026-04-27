use arrow_array::builder::Float64Builder;
use arrow_schema::DataType;
use datafusion_common::{error::Result, DataFusionError};
use datafusion_expr::{ColumnarValue, Volatility};

use ::geos::Geom;

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

pub fn st_hausdorff_distance_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_hausdorff_distance",
        ItemCrsKernel::wrap_impl(vec![Arc::new(STHausdorffDistance)]),
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct STHausdorffDistance;

impl SedonaScalarKernel for STHausdorffDistance {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry(), ArgMatcher::is_geometry()],
            SedonaType::Arrow(DataType::Float64),
        );
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let mut executor = WkbExecutor::new(arg_types, args);
        let mut builder = Float64Builder::with_capacity(executor.num_iterations());


        executor.execute_wkb_void(|maybe_g1, maybe_g2| {
            match (maybe_g1, maybe_g2) {
                (Some(wkb1), Some(wkb2)) => {
                    
                    let g1 = ::geos::Geometry::try_from(wkb1)
                        .map_err(|e| DataFusionError::Execution(format!("GEOS conversion error: {}", e)))?;
                    let g2 = ::geos::Geometry::try_from(wkb2)
                        .map_err(|e| DataFusionError::Execution(format!("GEOS conversion error: {}", e)))?;

                    let dist = g1.hausdorff_distance(&g2)
                        .map_err(|e| DataFusionError::Execution(format!("ST_HausdorffDistance error: {}", e)))?;

                    builder.append_value(dist);
                }
                _ => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}