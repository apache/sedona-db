use crate::executor::WkbExecutor;
use arrow_array::builder::BinaryBuilder;
use arrow_schema::DataType;
use datafusion_expr::scalar_doc_sections::DOC_SECTION_OTHER;
use datafusion_expr::{ColumnarValue, Documentation, Volatility};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_geometry::wkb_factory::WKB_MIN_PROBABLE_BYTES;
use sedona_schema::datatypes::SedonaType;
use sedona_schema::matchers::ArgMatcher;
use sedona_serde::serialize::serialize;
use std::sync::Arc;

#[derive(Debug)]
struct STGeomToSedonaSpark {}

impl SedonaScalarKernel for STGeomToSedonaSpark {
    fn return_type(&self, args: &[SedonaType]) -> datafusion_common::Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry()],
            SedonaType::Arrow(DataType::BinaryView),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> datafusion_common::Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);
        let mut builder = BinaryBuilder::with_capacity(
            executor.num_iterations(),
            WKB_MIN_PROBABLE_BYTES * executor.num_iterations(),
        );

        let crs_value = match &arg_types[0] {
            SedonaType::Wkb(_, crs) => {
                match crs {
                    Some(_crs) => {
                        let crs_id = _crs.srid()?;

                        match crs_id {
                            Some(srid) => Ok(Some(srid)),
                            None => Err(datafusion_common::DataFusionError::Internal(
                                "ST_GeomToSedonaSpark: Unsupported CRS without SRID".to_string(),
                            )),
                        }
                    }
                    None => Ok(None),
                }
                //
            }
            _ => Err(datafusion_common::DataFusionError::Internal(
                "ST_GeomToSedonaSpark: Unsupported geometry type".to_string(),
            )),
        }?;

        executor.execute_wkb_void(|maybe_item| {
            match maybe_item {
                Some(item) => {
                    serialize(&item, &mut builder, crs_value)?;
                    builder.append_value([]);
                }
                None => builder.append_null(),
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

pub fn st_geomtosedona_udf() -> SedonaScalarUDF {
    let kernel = Arc::new(STGeomToSedonaSpark {});

    SedonaScalarUDF::new(
        "st_geomtosedonaspark",
        vec![kernel],
        Volatility::Immutable,
        Some(doc()),
    )
}

fn doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Internal only, it's function used in the vectorized UDFs to translate WKB to Sedona Spark binary format",
        "ST_GeomToSedonaSpark (geom: Geometry, crs: string)",
    )
        .with_argument("geom", "wkb geometry")
        .with_argument("crs", "crs: coordinate reference system")
        .with_sql_example("SELECT ST_GeomToSedonaSpark(geom, 'EPSG:4326')")
        .build()
}

#[cfg(test)]
mod tests {
    use crate::st_to_sedona_spark::st_geomtosedona_udf;
    use datafusion_common::ScalarValue;
    use rstest::rstest;
    use sedona_schema::crs::{deserialize_crs, lnglat};
    use sedona_schema::datatypes::{Edges, SedonaType};
    use sedona_testing::create::create_scalar;
    use sedona_testing::testers::ScalarUdfTester;

    const POINT_WKT: &str = "POINT (1 1)";
    const LINESTRING_WKT: &str = "LINESTRING (0 0, 1 1, 2 2)";
    const MULTILINESTRING_WKT: &str = "MULTILINESTRING ((1 1, 2 2), (4 5, 6 7))";
    const MULTIPOINT_WKT: &str = "MULTIPOINT ((1 1), (2 2), (4 5))";
    const POLYGON_WKT: &str = "POLYGON (
        (1 1, 10 1, 10 10, 1 10, 1 1),
        (2 2, 4 2, 4 4, 2 4, 2 2),
        (6 6, 8 6, 8 8, 6 8, 6 6)
    )";
    const MULTIPOLYGON_WKT: &str = "MULTIPOLYGON (
        (
            (1 1, 10 1, 10 10, 1 10, 1 1),
            (2 2, 4 2, 4 4, 2 4, 2 2),
            (6 6, 8 6, 8 8, 6 8, 6 6)
        ),
         (
            (12 1, 20 1, 20 9, 12 9, 12 1),
            (13 2, 15 2, 15 4, 13 4, 13 2),
            (17 5, 19 5, 19 7, 17 7, 17 5)
         )
     )";
    const GEOMETRYCOLLECTION_WKT: &str = "GEOMETRYCOLLECTION (
        POINT (4 6),
        LINESTRING (4 6,7 10),
        POLYGON((4 6,7 10,4 10,4 6))
    )";

    const COMPLEX_GEOMETRYCOLLECTION_WKT: &str = "GEOMETRYCOLLECTION(
        POINT(4 6),
        LINESTRING(4 6,7 10),
        POLYGON((4 6,7 10,4 10,4 6)),
        MULTIPOINT((1 2),(3 4))
    )";
    const NESTED_GEOMETRYCOLLECTION_WKT: &str = "GEOMETRYCOLLECTION (
        POINT (1 1),
        GEOMETRYCOLLECTION (
            LINESTRING (0 0, 1 1),
            POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))
        )
     )";

    const FLOATING_POLYGON_WKT: &str = "POLYGON (
        (
            12.345678901234 45.678901234567,
            23.456789012345 67.890123456789,
            34.567890123456 56.789012345678,
            45.678901234567 34.567890123456,
            29.876543210987 22.345678901234,
            12.345678901234 45.678901234567
        ),
        (
            25.123456789012 45.987654321098,
            30.987654321098 50.123456789012,
            35.456789012345 45.456789012345,
            30.234567890123 40.987654321098,
            25.123456789012 45.987654321098
        )
    )";

    fn get_tester() -> ScalarUdfTester {
        ScalarUdfTester::new(
            st_geomtosedona_udf().into(),
            vec![SedonaType::Wkb(Edges::Planar, None)],
        )
    }

    fn fixture_to_bytes(wkb: &str) -> Vec<u8> {
        wkb.split("\n")
            .filter(|line| !line.starts_with("//") && !line.is_empty())
            .flat_map(|s| s.split_whitespace())
            .map(|num| num.parse::<u8>().expect("invalid byte"))
            .collect::<Vec<u8>>()
    }

    #[rstest]
    fn test_geometries_serialization(
        #[values(
        (POINT_WKT, include_str!("fixtures/point.sedona")),
        (LINESTRING_WKT, include_str!("fixtures/linestring.sedona")),
        (MULTILINESTRING_WKT, include_str!("fixtures/multilinestring.sedona")),
        (MULTIPOINT_WKT, include_str!("fixtures/multipoint.sedona")),
        (POLYGON_WKT, include_str!("fixtures/polygon.sedona")),
        (MULTIPOLYGON_WKT, include_str!("fixtures/multipolygon.sedona")),
        (GEOMETRYCOLLECTION_WKT, include_str!("fixtures/geometrycollection.sedona")),
        (COMPLEX_GEOMETRYCOLLECTION_WKT, include_str!("fixtures/geometrycollectioncomplex.sedona")),
        (NESTED_GEOMETRYCOLLECTION_WKT, include_str!("fixtures/nested_geometry_collection.sedona")),
        ("POINT EMPTY", include_str!("fixtures/empty_point.sedona")),
        ("LINESTRING EMPTY", include_str!("fixtures/empty_linestring.sedona")),
        ("POLYGON EMPTY", include_str!("fixtures/empty_polygon.sedona")),
        ("MULTIPOINT EMPTY", include_str!("fixtures/multipoint_empty.sedona")),
        ("MULTIPOLYGON EMPTY", include_str!("fixtures/empty_multipolygon.sedona")),
        ("MULTILINESTRING EMPTY", include_str!("fixtures/empty_multilinestring.sedona")),
        ("GEOMETRYCOLLECTION EMPTY", include_str!("fixtures/empty_geometry_collection.sedona")),
        (FLOATING_POLYGON_WKT, include_str!("fixtures/point_float_coords.sedona"))
    )]
        value: (&str, &str),
    ) {
        let tester = get_tester();

        let (input_wkt, fixture) = value;

        let geometry = create_scalar(Some(input_wkt), &SedonaType::Wkb(Edges::Planar, None));

        let result = tester.invoke_scalar(geometry).unwrap();

        let binary_geometry = fixture_to_bytes(fixture);

        assert_eq!(result, ScalarValue::Binary(Some(binary_geometry)));
    }

    #[test]
    fn test_serialization_with_crs() {
        let crs = deserialize_crs("EPSG:4326").unwrap(); // to ensure Crs can be deserialized to provide

        let tester = ScalarUdfTester::new(
            st_geomtosedona_udf().into(),
            vec![SedonaType::Wkb(Edges::Planar, crs.clone())],
        );

        let geometry = create_scalar(Some(POINT_WKT), &SedonaType::Wkb(Edges::Planar, crs));

        let result = tester.invoke_scalar(geometry).unwrap();

        let expected_fixture = include_str!("fixtures/crs_point.sedona");
        let binary_geometry = fixture_to_bytes(expected_fixture);

        assert_eq!(result, ScalarValue::Binary(Some(binary_geometry)));
    }
}
