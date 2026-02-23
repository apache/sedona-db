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

use std::sync::Arc;

use arrow_array::{RecordBatch, RecordBatchIterator, RecordBatchReader};
use arrow_schema::{ArrowError, Schema};
use datafusion::{
    catalog::Session,
    physical_expr::ScalarFunctionExpr,
    physical_plan::{expressions::Column, PhysicalExpr},
};
use datafusion_common::Result;
use datafusion_expr::ReturnFieldArgs;
use sedona_common::sedona_internal_datafusion_err;

pub fn projected_record_batch_reader(
    reader: Box<dyn RecordBatchReader + Send>,
    projection: Vec<(Arc<dyn PhysicalExpr>, String)>,
) -> Result<Box<dyn RecordBatchReader + Send>> {
    let existing_schema = reader.schema();
    let new_fields = projection
        .iter()
        .map(|expr| {
            Ok(expr
                .0
                .return_field(&existing_schema)?
                .as_ref()
                .clone()
                .with_name(expr.1.clone()))
        })
        .collect::<Result<Vec<_>>>()?;

    let new_schema = Arc::new(Schema::new_with_metadata(
        new_fields,
        existing_schema.metadata().clone(),
    ));
    let iter_schema = new_schema.clone();
    let reader_iter = reader.map(move |maybe_batch| {
        let batch = maybe_batch?;
        let new_columns = projection
            .iter()
            .map(|expr| expr.0.evaluate(&batch)?.to_array(batch.num_rows()))
            .collect::<Result<Vec<_>>>()
            .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
        RecordBatch::try_new(iter_schema.clone(), new_columns)
    });

    Ok(Box::new(RecordBatchIterator::new(reader_iter, new_schema)))
}

pub fn simplify_record_batch_reader(
    ctx: &dyn Session,
    reader: Box<dyn RecordBatchReader + Send>,
) -> Result<Box<dyn RecordBatchReader + Send>> {
    let existing_schema = reader.schema();
    let config_options = Arc::new(ctx.config_options().clone());
    let udf = ctx
        .scalar_functions()
        .get("sd_simplifystorage")
        .ok_or_else(|| sedona_internal_datafusion_err!("Expected sd_simplifystorage UDF"))?;
    let projection = existing_schema
        .fields()
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let arg_field = Arc::new(f.clone());
            let col_expr = Column::new(arg_field.name(), i);
            let return_field = udf.return_field_from_args(ReturnFieldArgs {
                arg_fields: std::slice::from_ref(&arg_field),
                scalar_arguments: &[None],
            })?;
            let expr = Arc::new(ScalarFunctionExpr::new(
                udf.name(),
                Arc::clone(udf),
                vec![Arc::new(col_expr)],
                return_field,
                config_options.clone(),
            ));
            Ok((expr as Arc<dyn PhysicalExpr>, arg_field.name().clone()))
        })
        .collect::<Result<Vec<_>>>()?;

    projected_record_batch_reader(reader, projection)
}
