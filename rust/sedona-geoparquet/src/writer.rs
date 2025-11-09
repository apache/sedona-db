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

use std::{collections::HashMap, sync::Arc};

use arrow_array::{
    builder::{Float32Builder, NullBufferBuilder},
    ArrayRef, StructArray,
};
use arrow_schema::{DataType, Field, Fields};
use datafusion::{
    config::TableParquetOptions,
    datasource::{
        file_format::parquet::ParquetSink, physical_plan::FileSinkConfig, sink::DataSinkExec,
    },
};
use datafusion_common::{
    config::ConfigOptions, exec_datafusion_err, exec_err, not_impl_err, DataFusionError, Result,
};
use datafusion_expr::{dml::InsertOp, ColumnarValue, ScalarUDF, Volatility};
use datafusion_physical_expr::{
    expressions::Column, LexRequirement, PhysicalExpr, ScalarFunctionExpr,
};
use datafusion_physical_plan::{projection::ProjectionExec, ExecutionPlan};
use float_next_after::NextAfter;
use geo_traits::GeometryTrait;
use sedona_common::sedona_internal_err;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_functions::executor::WkbExecutor;
use sedona_geometry::{
    bounds::geo_traits_update_xy_bounds,
    interval::{Interval, IntervalTrait},
};
use sedona_schema::{
    crs::lnglat,
    datatypes::{Edges, SedonaType},
    matchers::ArgMatcher,
    schema::SedonaSchema,
};

use crate::{
    metadata::{GeoParquetColumnMetadata, GeoParquetCovering, GeoParquetMetadata},
    options::{GeoParquetVersion, TableGeoParquetOptions},
};

pub fn create_geoparquet_writer_physical_plan(
    mut input: Arc<dyn ExecutionPlan>,
    mut conf: FileSinkConfig,
    order_requirements: Option<LexRequirement>,
    options: &TableGeoParquetOptions,
) -> Result<Arc<dyn ExecutionPlan>> {
    if conf.insert_op != InsertOp::Append {
        return not_impl_err!("Overwrites are not implemented yet for Parquet");
    }

    // If there is no geometry, just use the inner implementation
    let mut output_geometry_column_indices = conf.output_schema().geometry_column_indices()?;
    if output_geometry_column_indices.is_empty() {
        return create_inner_writer(input, conf, order_requirements, options.inner.clone());
    }

    // We have geometry and/or geography! Collect the GeoParquetMetadata we'll need to write
    let mut metadata = GeoParquetMetadata::default();
    let mut bbox_columns = HashMap::new();

    // Check the version
    match options.geoparquet_version {
        GeoParquetVersion::V1_0 => {
            metadata.version = "1.0.0".to_string();
        }
        GeoParquetVersion::V1_1 => {
            metadata.version = "1.1.0".to_string();
            (input, bbox_columns) = project_bboxes(input, options.overwrite_bbox_columns)?;
            conf.output_schema = input.schema();
            output_geometry_column_indices = input.schema().geometry_column_indices()?;
        }
        _ => {
            return not_impl_err!(
                "GeoParquetVersion {:?} is not yet supported",
                options.geoparquet_version
            );
        }
    }

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

        // Add bbox column info, if we added one in project_bboxes()
        if let Some(bbox_column_name) = bbox_columns.get(f.name()) {
            column_metadata
                .covering
                .replace(GeoParquetCovering::bbox_struct_xy(bbox_column_name));
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

/// Create a regular Parquet writer like DataFusion would otherwise do.
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

/// Create a projection that inserts a bbox column for every geometry column
///
/// This implements creating the GeoParquet 1.1 bounding box columns,
/// returning a map from the name of the geometry column to the name of the
/// bounding box column it created. This does not currently create such
/// a column for any geography input.
///
/// The inserted bounding box columns always directly precede their
/// corresponding geometry column and are named a follows:
///
/// - For a column named "geometry", the bbox column is named "bbox". This
///   reflects what pretty much everybody is already naming their columns
///   today.
/// - For any other column, the bbox column is named "{col_name}_bbox".
///
/// If a bbox column name already exists in the schema, we replace it.
/// In the context of writing a file and all that goes with it, the time it
/// takes to recompute the bounding box is not important; because writing
/// GeoParquet 1.1 is opt-in, if somebody *did* have a column with "bbox" or
/// "some_col_bbox", it is unlikely that replacing it would have unintended
/// consequences.
fn project_bboxes(
    input: Arc<dyn ExecutionPlan>,
    overwrite_bbox_columns: bool,
) -> Result<(Arc<dyn ExecutionPlan>, HashMap<String, String>)> {
    let input_schema = input.schema();
    let matcher = ArgMatcher::is_geometry();
    let bbox_udf: Arc<ScalarUDF> = Arc::new(geoparquet_bbox_udf().into());
    let bbox_udf_name = bbox_udf.name();

    // Calculate and keep track of the expression, name pairs for the bounding box
    // columns we are about to (potentially) create.
    let mut bbox_exprs = HashMap::<usize, (Arc<dyn PhysicalExpr>, String)>::new();
    let mut bbox_column_names = HashMap::new();
    for (i, f) in input.schema().fields().iter().enumerate() {
        let column = Arc::new(Column::new(f.name(), i));

        // If this is a geometry column (not geography), compute the
        // expression that is a function call to our bbox column creator
        if matcher.match_type(&SedonaType::from_storage_field(
            column.return_field(&input_schema)?.as_ref(),
        )?) {
            let bbox_field_name = bbox_column_name(f.name());
            // TODO: Pipe actual ConfigOptions from session instead of using defaults
            // See: https://github.com/apache/sedona-db/issues/248
            let expr = Arc::new(ScalarFunctionExpr::new(
                bbox_udf_name,
                bbox_udf.clone(),
                vec![column],
                Arc::new(Field::new("", bbox_type(), true)),
                Arc::new(ConfigOptions::default()),
            ));

            bbox_exprs.insert(i, (expr, bbox_field_name.clone()));
            bbox_column_names.insert(bbox_field_name, f.name().clone());
        }
    }

    // If we don't need to create any bbox columns, don't add an additional
    // projection at the end of the input plan
    if bbox_exprs.is_empty() {
        return Ok((input, HashMap::new()));
    }

    // Create the projection expressions
    let mut exprs = Vec::new();
    for (i, f) in input.schema().fields().iter().enumerate() {
        // Skip any column with the same name as a bbox column, since we are
        // about to replace it with the recomputed bbox.
        if bbox_column_names.contains_key(f.name()) {
            if overwrite_bbox_columns {
                continue;
            } else {
                return exec_err!(
                    "Can't overwrite GeoParquet 1.1 bbox column '{}'.
Use overwrite_bbox_columns = True if this is what was intended.",
                    f.name()
                );
            }
        }

        // If this is a column with a bbox, insert the bbox expression now
        if let Some((expr, expr_name)) = bbox_exprs.remove(&i) {
            exprs.push((expr, expr_name));
        }

        // Insert the column (whether it does or does not have geometry)
        let column = Arc::new(Column::new(f.name(), i));
        exprs.push((column, f.name().clone()));
    }

    // Create the projection
    let exec = ProjectionExec::try_new(exprs, input)?;

    // Flip the bbox_column_names into the form our caller needs it
    let bbox_column_names_by_field = bbox_column_names.drain().map(|(k, v)| (v, k)).collect();

    Ok((Arc::new(exec), bbox_column_names_by_field))
}

fn geoparquet_bbox_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "geoparquet_bbox",
        vec![Arc::new(GeoParquetBbox {})],
        Volatility::Immutable,
        None,
    )
}

fn bbox_column_name(geometry_column_name: &str) -> String {
    if geometry_column_name == "geometry" {
        "bbox".to_string()
    } else {
        format!("{geometry_column_name}_bbox")
    }
}

#[derive(Debug)]
struct GeoParquetBbox {}

impl SedonaScalarKernel for GeoParquetBbox {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry()],
            SedonaType::Arrow(bbox_type()),
        );
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);

        // Initialize the builders. We use Float32 to minimize the impact
        // on the file size.
        let mut nulls = NullBufferBuilder::new(executor.num_iterations());
        let mut builders = [
            Float32Builder::with_capacity(executor.num_iterations()),
            Float32Builder::with_capacity(executor.num_iterations()),
            Float32Builder::with_capacity(executor.num_iterations()),
            Float32Builder::with_capacity(executor.num_iterations()),
        ];

        executor.execute_wkb_void(|maybe_item| {
            match maybe_item {
                Some(item) => {
                    nulls.append(true);
                    append_float_bbox(&item, &mut builders)?;
                }
                None => {
                    // If we have a null, we set the outer validity bitmap to null
                    // (i.e., "the bounding box is null") but also the inner bitmap
                    // to null to ensure the value is not counted for the purposes
                    // of computing statistics for the nested column.
                    nulls.append(false);
                    for builder in &mut builders {
                        builder.append_null();
                    }
                }
            }
            Ok(())
        })?;

        let out_array = StructArray::try_new(
            bbox_fields(),
            builders
                .iter_mut()
                .map(|builder| -> ArrayRef { Arc::new(builder.finish()) })
                .collect(),
            nulls.finish(),
        )?;

        executor.finish(Arc::new(out_array))
    }
}

fn bbox_type() -> DataType {
    DataType::Struct(bbox_fields())
}

fn bbox_fields() -> Fields {
    vec![
        Field::new("xmin", DataType::Float32, true),
        Field::new("ymin", DataType::Float32, true),
        Field::new("xmax", DataType::Float32, true),
        Field::new("ymax", DataType::Float32, true),
    ]
    .into()
}

// Calculates a bounding box and appends the float32-rounded version to
// a set of builders, ensuring the float bounds always include the double
// bounds.
fn append_float_bbox(
    wkb: impl GeometryTrait<T = f64>,
    builders: &mut [Float32Builder],
) -> Result<()> {
    let mut x = Interval::empty();
    let mut y = Interval::empty();
    geo_traits_update_xy_bounds(wkb, &mut x, &mut y)
        .map_err(|e| DataFusionError::External(e.into()))?;

    // If we have an empty, append null values to the individual min/max
    // columns to ensure their values aren't considered in the Parquet
    // statistics.
    if x.is_empty() || y.is_empty() {
        for builder in builders {
            builder.append_null();
        }
    } else {
        builders[0].append_value((x.lo() as f32).next_after(-f32::INFINITY));
        builders[1].append_value((y.lo() as f32).next_after(-f32::INFINITY));
        builders[2].append_value((x.hi() as f32).next_after(f32::INFINITY));
        builders[3].append_value((y.hi() as f32).next_after(f32::INFINITY));
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use std::iter::zip;

    use arrow_array::{create_array, Array, RecordBatch};
    use datafusion::datasource::file_format::format_as_file_type;
    use datafusion::prelude::DataFrame;
    use datafusion::{
        execution::SessionStateBuilder,
        prelude::{col, lit, SessionContext},
    };
    use datafusion_common::cast::{as_float32_array, as_struct_array};
    use datafusion_common::ScalarValue;
    use datafusion_expr::{Expr, LogicalPlanBuilder};
    use sedona_schema::datatypes::WKB_GEOMETRY;
    use sedona_testing::create::create_array;
    use sedona_testing::data::test_geoparquet;
    use sedona_testing::testers::ScalarUdfTester;
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

    async fn test_dataframe_roundtrip(ctx: SessionContext, src: DataFrame) {
        let df_batches = src.clone().collect().await.unwrap();
        test_write_dataframe(
            ctx,
            src,
            df_batches,
            TableGeoParquetOptions::default(),
            vec![],
        )
        .await
        .unwrap()
    }

    async fn test_write_dataframe(
        ctx: SessionContext,
        src: DataFrame,
        expected_batches: Vec<RecordBatch>,
        options: TableGeoParquetOptions,
        partition_by: Vec<String>,
    ) -> Result<()> {
        // It's a bit verbose to trigger this without helpers
        let format = GeoParquetFormatFactory::new_with_options(options);
        let file_type = format_as_file_type(Arc::new(format));
        let tmpdir = tempdir().unwrap();

        let tmp_parquet = tmpdir.path().join("foofy_spatial.parquet");

        let plan = LogicalPlanBuilder::copy_to(
            src.into_unoptimized_plan(),
            tmp_parquet.to_string_lossy().into(),
            file_type,
            Default::default(),
            partition_by,
        )
        .unwrap()
        .build()
        .unwrap();

        DataFrame::new(ctx.state(), plan).collect().await?;

        let df_parquet_batches = ctx
            .table(tmp_parquet.to_string_lossy().to_string())
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(df_parquet_batches.len(), expected_batches.len());

        // Check column names
        let df_parquet_names = df_parquet_batches[0]
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect::<Vec<_>>();
        let expected_names = expected_batches[0]
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect::<Vec<_>>();
        assert_eq!(df_parquet_names, expected_names);

        // Check types, since the schema may not compare byte-for-byte equal (CRSes)
        let df_parquet_sedona_types = df_parquet_batches[0]
            .schema()
            .sedona_types()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        let df_sedona_types = expected_batches[0]
            .schema()
            .sedona_types()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(df_parquet_sedona_types, df_sedona_types);

        // Check batches without metadata
        for (df_parquet_batch, df_batch) in zip(df_parquet_batches, expected_batches) {
            assert_eq!(df_parquet_batch.columns(), df_batch.columns())
        }

        Ok(())
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

    #[tokio::test]
    async fn geoparquet_1_1_basic() {
        let example = test_geoparquet("example", "geometry").unwrap();
        let ctx = setup_context();
        let df = ctx
            .table(&example)
            .await
            .unwrap()
            // DataFusion internals lose the nullability we assigned to the bbox
            // and without this line the test fails.
            .filter(Expr::IsNotNull(col("geometry").into()))
            .unwrap();

        let mut options = TableGeoParquetOptions::new();
        options.geoparquet_version = GeoParquetVersion::V1_1;

        let bbox_udf: ScalarUDF = geoparquet_bbox_udf().into();

        let df_batches_with_bbox = df
            .clone()
            .select(vec![
                col("wkt"),
                bbox_udf.call(vec![col("geometry")]).alias("bbox"),
                col("geometry"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        test_write_dataframe(ctx, df, df_batches_with_bbox, options, vec![])
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn geoparquet_1_1_multiple_columns() {
        let example = test_geoparquet("example", "geometry").unwrap();
        let ctx = setup_context();

        // Include >1 geometry and sprinkle in some non-geometry columns
        let df = ctx
            .table(&example)
            .await
            .unwrap()
            // DataFusion internals lose the nullability we assigned to the bbox
            // and without this line the test fails.
            .filter(Expr::IsNotNull(col("geometry").into()))
            .unwrap()
            .select(vec![
                col("wkt"),
                col("geometry").alias("geom"),
                col("wkt").alias("wkt2"),
                col("geometry"),
                col("wkt").alias("wkt3"),
            ])
            .unwrap();

        let mut options = TableGeoParquetOptions::new();
        options.geoparquet_version = GeoParquetVersion::V1_1;

        let bbox_udf: ScalarUDF = geoparquet_bbox_udf().into();

        let df_batches_with_bbox = df
            .clone()
            .select(vec![
                col("wkt"),
                bbox_udf.call(vec![col("geom")]).alias("geom_bbox"),
                col("geom"),
                col("wkt2"),
                bbox_udf.call(vec![col("geometry")]).alias("bbox"),
                col("geometry"),
                col("wkt3"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        test_write_dataframe(ctx, df, df_batches_with_bbox, options, vec![])
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn geoparquet_1_1_overwrite_existing_bbox() {
        let example = test_geoparquet("example", "geometry").unwrap();
        let ctx = setup_context();

        // Test writing a DataFrame that already has a column named "bbox".
        // Writing this using GeoParquet 1.1 will overwrite the column.
        let df = ctx
            .table(&example)
            .await
            .unwrap()
            // DataFusion internals lose the nullability we assigned to the bbox
            // and without this line the test fails.
            .filter(Expr::IsNotNull(col("geometry").into()))
            .unwrap()
            .select(vec![
                lit("this is definitely not a bbox").alias("bbox"),
                col("geometry"),
            ])
            .unwrap();

        let mut options = TableGeoParquetOptions::new();
        options.geoparquet_version = GeoParquetVersion::V1_1;

        let bbox_udf: ScalarUDF = geoparquet_bbox_udf().into();

        let df_batches_with_bbox = df
            .clone()
            .select(vec![
                bbox_udf.call(vec![col("geometry")]).alias("bbox"),
                col("geometry"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        // Without setting overwrite_bbox_columns = true, this should error
        let err = test_write_dataframe(
            ctx.clone(),
            df.clone(),
            df_batches_with_bbox.clone(),
            options.clone(),
            vec!["part".into()],
        )
        .await
        .unwrap_err();
        assert!(err
            .message()
            .starts_with("Can't overwrite GeoParquet 1.1 bbox column 'bbox'"));

        options.overwrite_bbox_columns = true;
        test_write_dataframe(ctx, df, df_batches_with_bbox, options, vec![])
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn geoparquet_1_1_with_partition() {
        let example = test_geoparquet("example", "geometry").unwrap();
        let ctx = setup_context();
        let df = ctx
            .table(&example)
            .await
            .unwrap()
            // DataFusion internals loose the nullability we assigned to the bbox
            // and without this line the test fails.
            .filter(Expr::IsNotNull(col("geometry").into()))
            .unwrap()
            .select(vec![
                lit("some_partition").alias("part"),
                col("wkt"),
                col("geometry"),
            ])
            .unwrap();

        let mut options = TableGeoParquetOptions::new();
        options.geoparquet_version = GeoParquetVersion::V1_1;

        let bbox_udf: ScalarUDF = geoparquet_bbox_udf().into();

        let df_batches_with_bbox = df
            .clone()
            .select(vec![
                col("wkt"),
                bbox_udf.call(vec![col("geometry")]).alias("bbox"),
                col("geometry"),
                lit(ScalarValue::Dictionary(
                    DataType::UInt16.into(),
                    ScalarValue::Utf8(Some("some_partition".to_string())).into(),
                ))
                .alias("part"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        test_write_dataframe(ctx, df, df_batches_with_bbox, options, vec!["part".into()])
            .await
            .unwrap();
    }

    #[test]
    fn float_bbox() {
        let tester = ScalarUdfTester::new(geoparquet_bbox_udf().into(), vec![WKB_GEOMETRY]);
        assert_eq!(
            tester.return_type().unwrap(),
            SedonaType::Arrow(bbox_type())
        );

        let array = create_array(
            &[
                Some("POINT (0 1)"),
                Some("POINT (2 3)"),
                Some("LINESTRING (4 5, 6 7)"),
            ],
            &WKB_GEOMETRY,
        );

        let result = tester.invoke_array(array).unwrap();
        assert_eq!(result.len(), 3);

        let expected_cols_f64 = [
            create_array!(Float64, [Some(0.0), Some(2.0), Some(4.0)]),
            create_array!(Float64, [Some(1.0), Some(3.0), Some(5.0)]),
            create_array!(Float64, [Some(0.0), Some(2.0), Some(6.0)]),
            create_array!(Float64, [Some(1.0), Some(3.0), Some(7.0)]),
        ];

        let result_struct = as_struct_array(&result).unwrap();
        let actual_cols = result_struct
            .columns()
            .iter()
            .map(|col| as_float32_array(col).unwrap())
            .collect::<Vec<_>>();
        for i in 0..result.len() {
            let actual = actual_cols
                .iter()
                .map(|a| a.value(i) as f64)
                .collect::<Vec<_>>();
            let expected = expected_cols_f64
                .iter()
                .map(|e| e.value(i))
                .collect::<Vec<_>>();

            // These values aren't equal (the actual values were float32 values that
            // had been rounded down); however, they should "contain" the expected box)
            assert!(actual[0] <= expected[0]);
            assert!(actual[1] <= expected[1]);
            assert!(actual[2] >= expected[2]);
            assert!(actual[3] >= expected[3]);
        }
    }

    #[test]
    fn float_bbox_null() {
        let tester = ScalarUdfTester::new(geoparquet_bbox_udf().into(), vec![WKB_GEOMETRY]);

        let null_result = tester.invoke_scalar(ScalarValue::Null).unwrap();
        assert!(null_result.is_null());
        if let ScalarValue::Struct(s) = null_result {
            let actual_cols = s
                .columns()
                .iter()
                .map(|col| as_float32_array(col).unwrap())
                .collect::<Vec<_>>();
            assert!(actual_cols[0].is_null(0));
            assert!(actual_cols[1].is_null(0));
            assert!(actual_cols[2].is_null(0));
            assert!(actual_cols[3].is_null(0));
        } else {
            panic!("Expected struct")
        }
    }
}
