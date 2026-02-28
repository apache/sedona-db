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

use std::{collections::HashMap, iter::zip, ptr::swap_nonoverlapping, sync::Arc};

use arrow_array::{
    ffi_stream::FFI_ArrowArrayStream, RecordBatch, RecordBatchIterator, RecordBatchReader,
};
use arrow_schema::{ArrowError, Schema};
use savvy::{savvy, savvy_err};
use sedona_schema::{
    crs::deserialize_crs,
    datatypes::{Edges, SedonaType},
};

use crate::ffi::import_array_stream;


#[savvy]
fn apply_crses_to_sf_stream(
    stream_in_xptr: savvy::Sexp,
    geometry_column_names: savvy::StringSexp,
    geometry_column_crses: savvy::StringSexp,
    stream_out_xptr: savvy::Sexp,
) -> savvy::Result<()> {
    let reader_in = Box::new(import_array_stream(stream_in_xptr)?);
    let reader_out = apply_crses_to_sf_stream_impl(
        reader_in,
        geometry_column_names.iter().collect(),
        geometry_column_crses.iter().collect(),
    )?;

    let out_void = unsafe { savvy_ffi::R_ExternalPtrAddr(stream_out_xptr.0) };
    if out_void.is_null() {
        return Err(savvy_err!(
            "external pointer to null in apply_crses_to_sf_stream()"
        ));
    }

    let mut ffi_stream = FFI_ArrowArrayStream::new(reader_out);
    let ffi_out = out_void as *mut FFI_ArrowArrayStream;
    unsafe { swap_nonoverlapping(&mut ffi_stream, ffi_out, 1) };
    Ok(())
}

fn apply_crses_to_sf_stream_impl(
    stream: Box<dyn RecordBatchReader + Send>,
    geometry_column_names: Vec<&str>,
    geometry_column_crses: Vec<&str>,
) -> savvy::Result<Box<dyn RecordBatchReader + Send>> {
    let crs_lookup =
        zip(geometry_column_names, geometry_column_crses).collect::<HashMap<&str, &str>>();
    let schema = stream.schema();

    let new_fields = schema
        .fields()
        .iter()
        .map(|f| {
            if f.extension_type_name() == Some("ogc.wkb") {
                let crs = crs_lookup.get(f.name().as_str()).unwrap_or(&"");
                let sedona_type = SedonaType::Wkb(Edges::Planar, deserialize_crs(crs)?);
                Ok(sedona_type.to_storage_field(f.name(), f.is_nullable())?)
            } else {
                Ok(f.as_ref().clone())
            }
        })
        .collect::<Result<Vec<_>, savvy::Error>>()?;

    let new_schema = Arc::new(Schema::new(new_fields).with_metadata(schema.metadata().clone()));
    let schema_iter = new_schema.clone();
    let iter = stream.map(move |maybe_batch| -> Result<RecordBatch, ArrowError> {
        let batch = maybe_batch?;
        RecordBatch::try_new(schema_iter.clone(), batch.columns().to_vec())
    });

    Ok(Box::new(RecordBatchIterator::new(iter, new_schema)))
}
