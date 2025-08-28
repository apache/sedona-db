use std::sync::Arc;

use crate::executor::WkbExecutor;
use arrow_array::builder::StringBuilder;
use arrow_schema::DataType;
use datafusion_common::error::Result;
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use geo_traits::GeometryTrait;
use sedona_common::sedona_internal_err;
use sedona_expr::scalar_udf::{ArgMatcher, SedonaScalarKernel, SedonaScalarUDF};
use sedona_schema::datatypes::SedonaType;
use wkb::reader::Wkb;

pub fn st_geometry_type_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_geometrytype",
        vec![Arc::new(STGeometryType {})],
        Volatility::Immutable,
        Some(st_geometry_type_doc()),
    )
}

fn st_geometry_type_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Return the type of a geometry",
        "ST_GeometryType (A: Geometry)",
    )
    .with_argument("geom", "geometry: Input geometry")
    .with_sql_example("SELECT ST_GeometryType(ST_GeomFromWKT('POLYGON ((0 0, 1 0, 0 1, 0 0))'))")
    .build()
}

#[derive(Debug)]
struct STGeometryType {}

impl SedonaScalarKernel for STGeometryType {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(vec![ArgMatcher::is_geometry()], DataType::Utf8.try_into()?);

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);
        let min_output_size = "POINT".len() * executor.num_iterations();
        let mut builder = StringBuilder::with_capacity(executor.num_iterations(), min_output_size);

        // We can do quite a lot better than this with some vectorized WKB processing,
        // but for now we just do a slow iteration
        executor.execute_wkb_void(|maybe_item| {
            match maybe_item {
                Some(item) => {
                    builder.append_option(invoke_scalar(&item)?);
                }
                None => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(item: &Wkb) -> Result<Option<String>> {
    match item.as_type() {
        geo_traits::GeometryType::Point(_) => Ok(Some("ST_POINT".to_string())),
        geo_traits::GeometryType::LineString(_) => Ok(Some("ST_LINESTRING".to_string())),
        geo_traits::GeometryType::Polygon(_) => Ok(Some("ST_POLYGON".to_string())),
        geo_traits::GeometryType::MultiPoint(_) => Ok(Some("ST_MULTIPOINT".to_string())),
        geo_traits::GeometryType::MultiLineString(_) => Ok(Some("ST_MULTILINESTRING".to_string())),
        geo_traits::GeometryType::MultiPolygon(_) => Ok(Some("ST_MULTIPOLYGON".to_string())),
        geo_traits::GeometryType::GeometryCollection(_) => {
            Ok(Some("ST_GEOMETRYCOLLECTION".to_string()))
        }

        // Other geometry types in geo that we should not get here: Rect, Triangle, Line
        _ => sedona_internal_err!("unexpected geometry type"),
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{create_array, ArrayRef};
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = st_geometry_type_udf().into();
        assert_eq!(udf.name(), "st_geometrytype");
        assert!(udf.documentation().is_some())
    }

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        use datafusion_common::ScalarValue;

        let udf: ScalarUDF = st_geometry_type_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![sedona_type]);
        tester.assert_return_type(DataType::Utf8);

        let result = tester
            .invoke_scalar("POLYGON ((0 0, 1 0, 0 1, 0 0))")
            .unwrap();
        tester.assert_scalar_result_equals(result, "ST_POLYGON");

        let result = tester.invoke_scalar(ScalarValue::Null).unwrap();
        assert!(result.is_null());

        let input_wkt = vec![
            None,
            Some("POINT (1 2)"),
            Some("POLYGON ((0 0, 1 0, 0 1, 0 0))"),
            Some("LINESTRING (0 0, 1 0, 0 1)"),
            Some("MULTIPOINT ((0 1), (2 3))"),
            Some("MULTILINESTRING ((0 1, 2 3))"),
            Some("MULTIPOLYGON (((0 0, 0 1, 1 0, 0 0)))"),
            Some("GEOMETRYCOLLECTION (POINT (0 1))"),
        ];
        let expected: ArrayRef = create_array!(
            Utf8,
            [
                None,
                Some("ST_POINT"),
                Some("ST_POLYGON"),
                Some("ST_LINESTRING"),
                Some("ST_MULTIPOINT"),
                Some("ST_MULTILINESTRING"),
                Some("ST_MULTIPOLYGON"),
                Some("ST_GEOMETRYCOLLECTION")
            ]
        );
        assert_eq!(&tester.invoke_wkb_array(input_wkt).unwrap(), &expected);
    }
}
