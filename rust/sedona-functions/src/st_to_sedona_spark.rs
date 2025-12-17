use std::sync::Arc;
use arrow_schema::DataType;
use datafusion_expr::scalar_doc_sections::DOC_SECTION_OTHER;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_schema::datatypes::{SedonaType, WKB_GEOMETRY};
use datafusion_expr::{ColumnarValue, Documentation, Volatility};
use sedona_schema::matchers::ArgMatcher;

#[derive(Debug)]
struct STGeomToSedonaSpark {
    // out_type: SedonaType,
}

impl SedonaScalarKernel for STGeomToSedonaSpark {
    fn return_type(&self, args: &[SedonaType]) -> datafusion_common::Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry()],
            SedonaType::Arrow(DataType::BinaryView),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(&self, arg_types: &[SedonaType], args: &[ColumnarValue]) -> datafusion_common::Result<ColumnarValue> {
        todo!()
    }
}

pub fn st_geomtosedona_udf() -> SedonaScalarUDF {
    let kernel = Arc::new(STGeomToSedonaSpark {
        // out_type: WKB_GEOMETRY,
    });

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
    use arrow_schema::DataType;
    use sedona_schema::datatypes::{Edges, SedonaType};
    use sedona_testing::testers::ScalarUdfTester;
    use crate::st_from_sedona_spark::st_geomfromsedona_udf;
    use crate::st_to_sedona_spark::st_geomtosedona_udf;

    fn get_tester() -> ScalarUdfTester {
        ScalarUdfTester::new(
            st_geomtosedona_udf().into(),
            vec![
                SedonaType::Wkb(Edges::Planar, None),
                SedonaType::Arrow(DataType::Utf8),
            ],
        )
    }
}