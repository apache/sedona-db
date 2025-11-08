use std::sync::Arc;

use arrow_array::builder::BinaryBuilder;
use datafusion_common::{error::Result, DataFusionError};
use datafusion_expr::ColumnarValue;
use geos::Geom;
use sedona_expr::scalar_udf::{ArgMatcher, ScalarKernelRef, SedonaScalarKernel};
use sedona_geometry::wkb_factory::WKB_MIN_PROBABLE_BYTES;
use sedona_schema::datatypes::{SedonaType, WKB_GEOMETRY};

use crate::executor::GeosExecutor;

/// ST_ConcaveHull(geometry, ratio) implementation using the geos crate
pub fn st_concavehull_impl() -> ScalarKernelRef {
    Arc::new(STConcaveHull {})
}

#[derive(Debug)]
struct STConcaveHull {}

impl SedonaScalarKernel for STConcaveHull {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        // Expect 2 arguments: geometry and numeric (ratio)
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry(), ArgMatcher::is_numeric()],
            WKB_GEOMETRY,
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        // Second argument is the ratio (float)
        let ratio = GeosExecutor::get_f64_scalar(&args[1])?;

        // Only the first argument (geometry) is processed as WKB
        let executor = GeosExecutor::new(&arg_types[0..1], &args[0..1]);
        let mut builder = BinaryBuilder::with_capacity(
            executor.num_iterations(),
            WKB_MIN_PROBABLE_BYTES * executor.num_iterations(),
        );

        executor.execute_wkb_void(|maybe_geom| {
            match maybe_geom {
                Some(geom) => {
                    invoke_concave_hull(&geom, ratio, &mut builder)?;
                    builder.append_value([]);
                }
                _ => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_concave_hull(
    geos_geom: &geos::Geometry,
    ratio: f64,
    writer: &mut impl std::io::Write,
) -> Result<()> {
    // Compute the concave hull using GEOS
    let geometry = geos_geom
        .concave_hull(ratio, false) // false = do not allow holes
        .map_err(|e| DataFusionError::Execution(format!("Failed to calculate concave hull: {e}")))?;

    // Convert result back to WKB bytes
    let wkb = geometry
        .to_wkb()
        .map_err(|e| DataFusionError::Execution(format!("Failed to serialize hull: {e}")))?;

    writer.write_all(wkb.as_ref())?;
    Ok(())
}
