use std::{fmt::Debug, sync::Arc};

use arrow_array::builder::UInt64Builder;
use arrow_schema::DataType;
use datafusion_common::{DataFusionError, Result};
use datafusion_expr::ColumnarValue;
use sedona_expr::scalar_udf::SedonaScalarKernel;
use sedona_functions::executor::WkbBytesExecutor;
use sedona_geometry::{transform::CrsEngine, wkb_header::WkbHeader};
use sedona_schema::{crs::lnglat, datatypes::SedonaType, matchers::ArgMatcher};

use crate::st_transform::with_global_proj_engine;

pub struct OrderLngLat<F> {
    order_fn: F,
}

impl<F: Fn((f64, f64)) -> u64> OrderLngLat<F> {
    pub fn new(order_fn: F) -> Self {
        Self { order_fn }
    }
}

impl<F: Fn((f64, f64)) -> u64> Debug for OrderLngLat<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrderLngLat").finish()
    }
}

impl<F: Fn((f64, f64)) -> u64> SedonaScalarKernel for OrderLngLat<F> {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry_or_geography()],
            SedonaType::Arrow(DataType::UInt64),
        );
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let maybe_src_crs = match &arg_types[0] {
            SedonaType::Wkb(_, maybe_crs) | SedonaType::WkbView(_, maybe_crs)
                if maybe_crs != &lnglat() =>
            {
                match maybe_crs {
                    Some(crs) => Some(crs.to_json()),
                    None => Some("OGC:CRS84".to_string()),
                }
            }
            _ => None,
        };

        let executor = WkbBytesExecutor::new(arg_types, args);
        let mut builder = UInt64Builder::with_capacity(executor.num_iterations());

        if let Some(src_crs) = maybe_src_crs {
            with_global_proj_engine(|engine| {
                let to_lnglat = engine
                    .get_transform_crs_to_crs(&src_crs, "OGC:CRS84", None, "")
                    .map_err(|e| DataFusionError::Execution(format!("{e}")))?;

                executor.execute_wkb_void(|maybe_wkb| {
                    match maybe_wkb {
                        Some(wkb_bytes) => {
                            let header = WkbHeader::try_new(wkb_bytes)
                                .map_err(|e| DataFusionError::Execution(format!("{e}")))?;
                            let mut first_xy = header.first_xy();
                            to_lnglat
                                .transform_coord(&mut first_xy)
                                .map_err(|e| DataFusionError::Execution(format!("{e}")))?;
                            let order = (self.order_fn)(first_xy);
                            builder.append_value(order);
                        }
                        None => builder.append_null(),
                    }

                    Ok(())
                })?;

                todo!()
            })?;
        } else {
            executor.execute_wkb_void(|maybe_wkb| {
                match maybe_wkb {
                    Some(wkb_bytes) => {
                        let header = WkbHeader::try_new(wkb_bytes)
                            .map_err(|e| DataFusionError::Execution(format!("{e}")))?;
                        let first_xy = header.first_xy();
                        let order = (self.order_fn)(first_xy);
                        builder.append_value(order);
                    }
                    None => builder.append_null(),
                }

                Ok(())
            })?;
        }

        executor.finish(Arc::new(builder.finish()))
    }
}
