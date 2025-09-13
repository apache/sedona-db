use std::sync::Arc;

use datafusion::{
    config::TableParquetOptions,
    datasource::{
        file_format::parquet::ParquetSink, physical_plan::FileSinkConfig, sink::DataSinkExec,
    },
};
use datafusion_common::{exec_datafusion_err, exec_err, not_impl_err, Result};
use datafusion_expr::dml::InsertOp;
use datafusion_physical_expr::LexRequirement;
use datafusion_physical_plan::ExecutionPlan;
use sedona_common::sedona_internal_err;
use sedona_schema::{
    crs::lnglat,
    datatypes::{Edges, SedonaType},
    schema::SedonaSchema,
};

use crate::{
    metadata::{GeoParquetColumnMetadata, GeoParquetMetadata},
    options::{GeoParquetVersion, TableGeoParquetOptions},
};

pub fn create_geoparquet_writer_physical_plan(
    input: Arc<dyn ExecutionPlan>,
    conf: FileSinkConfig,
    order_requirements: Option<LexRequirement>,
    options: &TableGeoParquetOptions,
) -> Result<Arc<dyn ExecutionPlan>> {
    if conf.insert_op != InsertOp::Append {
        return not_impl_err!("Overwrites are not implemented yet for Parquet");
    }

    // If there is no geometry, just use the inner implementation
    let output_geometry_column_indices = conf.output_schema().geometry_column_indices()?;
    if output_geometry_column_indices.is_empty() {
        return create_inner_writer(input, conf, order_requirements, options.inner.clone());
    }

    // We have geometry and/or geography! Collect the GeoParquetMetadata we'll need to write
    let mut metadata = GeoParquetMetadata::default();

    // Check the version
    match options.geoparquet_version {
        GeoParquetVersion::V1_0 => {
            metadata.version = "1.0.0".to_string();
        }
        _ => {
            return not_impl_err!(
                "GeoParquetVersion {:?} is not yet supported",
                options.geoparquet_version
            );
        }
    };

    let field_names = conf
        .output_schema()
        .fields()
        .iter()
        .map(|f| f.name())
        .collect::<Vec<_>>();

    // Apply primary column
    if let Some(output_geometry_primary) = conf.output_schema().primary_geometry_column_index()? {
        metadata.primary_column = field_names[output_geometry_primary].clone();
    }

    // Apply all columns
    for i in output_geometry_column_indices {
        let f = conf.output_schema().field(i);
        let sedona_type = SedonaType::from_storage_field(f)?;
        let mut column_metadata = GeoParquetColumnMetadata::default();

        let (edge_type, crs) = match sedona_type {
            SedonaType::Wkb(edge_type, crs) | SedonaType::WkbView(edge_type, crs) => {
                (edge_type, crs)
            }
            _ => return sedona_internal_err!("Unexpected type: {sedona_type}"),
        };

        // Assign edge type if needed
        match edge_type {
            Edges::Planar => {}
            Edges::Spherical => {
                column_metadata.edges = Some("spherical".to_string());
            }
        }

        // Assign crs
        if crs == lnglat() {
            // Do nothing, lnglat is the meaning of an omitted CRS
        } else if let Some(crs) = crs {
            column_metadata.crs = Some(crs.to_json().parse().map_err(|e| {
                exec_datafusion_err!("Failed to parse CRS for column '{}' {e}", f.name())
            })?);
        } else {
            return exec_err!(
                "Can't write GeoParquet from null CRS\nUse ST_SetSRID({}, ...) to assign it one",
                f.name()
            );
        }

        // Add to metadata
        metadata
            .columns
            .insert(f.name().to_string(), column_metadata);
    }

    // Apply to the Parquet options
    let mut parquet_options = options.inner.clone();
    parquet_options.key_value_metadata.insert(
        "geo".to_string(),
        Some(
            serde_json::to_string(&metadata).map_err(|e| {
                exec_datafusion_err!("Failed to serialize GeoParquet metadata: {e}")
            })?,
        ),
    );

    // Create the sink
    let sink = Arc::new(ParquetSink::new(conf, parquet_options));
    Ok(Arc::new(DataSinkExec::new(input, sink, order_requirements)) as _)
}

fn create_inner_writer(
    input: Arc<dyn ExecutionPlan>,
    conf: FileSinkConfig,
    order_requirements: Option<LexRequirement>,
    options: TableParquetOptions,
) -> Result<Arc<dyn ExecutionPlan>> {
    // Create the sink
    let sink = Arc::new(ParquetSink::new(conf, options));
    Ok(Arc::new(DataSinkExec::new(input, sink, order_requirements)) as _)
}

#[cfg(test)]
mod test {
    use std::iter::zip;

    use datafusion::datasource::file_format::format_as_file_type;
    use datafusion::prelude::DataFrame;
    use datafusion::{
        execution::SessionStateBuilder,
        prelude::{col, SessionContext},
    };
    use datafusion_expr::LogicalPlanBuilder;
    use sedona_testing::data::test_geoparquet;
    use tempfile::tempdir;

    use crate::format::GeoParquetFormatFactory;

    use super::*;

    fn setup_context() -> SessionContext {
        let mut state = SessionStateBuilder::new().build();
        state
            .register_file_format(Arc::new(GeoParquetFormatFactory::new()), true)
            .unwrap();
        SessionContext::new_with_state(state).enable_url_table()
    }

    async fn test_dataframe_roundtrip(ctx: SessionContext, df: DataFrame) {
        // It's a bit verbose to trigger this without helpers
        let format = GeoParquetFormatFactory::new();
        let file_type = format_as_file_type(Arc::new(format));
        let tmpdir = tempdir().unwrap();

        let df_batches = df.clone().collect().await.unwrap();

        let tmp_parquet = tmpdir.path().join("foofy_spatial.parquet");

        let plan = LogicalPlanBuilder::copy_to(
            df.into_unoptimized_plan(),
            tmp_parquet.to_string_lossy().into(),
            file_type,
            Default::default(),
            vec![],
        )
        .unwrap()
        .build()
        .unwrap();

        DataFrame::new(ctx.state(), plan).collect().await.unwrap();

        let df_parquet_batches = ctx
            .table(tmp_parquet.to_string_lossy().to_string())
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(df_parquet_batches.len(), df_batches.len());

        // Check types, since the schema may not compare byte-for-byte equal (CRSes)
        let df_parquet_sedona_types = df_parquet_batches[0]
            .schema()
            .sedona_types()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        let df_sedona_types = df_batches[0]
            .schema()
            .sedona_types()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(df_parquet_sedona_types, df_sedona_types);

        // Check batches without metadata
        for (df_parquet_batch, df_batch) in zip(df_parquet_batches, df_batches) {
            assert_eq!(df_parquet_batch.columns(), df_batch.columns())
        }
    }

    #[tokio::test]
    async fn writer_without_spatial() {
        let example = test_geoparquet("example", "geometry").unwrap();
        let ctx = setup_context();

        // Deselect all geometry columns
        let df = ctx
            .table(&example)
            .await
            .unwrap()
            .select(vec![col("wkt")])
            .unwrap();

        test_dataframe_roundtrip(ctx, df).await;
    }

    #[tokio::test]
    async fn writer_with_geometry() {
        let example = test_geoparquet("example", "geometry").unwrap();
        let ctx = setup_context();
        let df = ctx.table(&example).await.unwrap();

        test_dataframe_roundtrip(ctx, df).await;
    }

    #[tokio::test]
    async fn writer_with_geography() {
        let example = test_geoparquet("natural-earth", "countries-geography").unwrap();
        let ctx = setup_context();
        let df = ctx.table(&example).await.unwrap();

        test_dataframe_roundtrip(ctx, df).await;
    }
}
