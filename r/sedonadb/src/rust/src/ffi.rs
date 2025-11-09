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

use arrow_array::{
    ffi::FFI_ArrowSchema,
    ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream},
};
use arrow_schema::Schema;
use datafusion::catalog::TableProvider;
use datafusion_expr::ScalarUDF;
use datafusion_ffi::{
    table_provider::{FFI_TableProvider, ForeignTableProvider},
    udf::{FFI_ScalarUDF, ForeignScalarUDF},
};
use savvy::{savvy_err, IntoExtPtrSexp};

pub fn import_schema(mut xptr: savvy::Sexp) -> savvy::Result<Schema> {
    let ffi_schema: &FFI_ArrowSchema = import_xptr(&mut xptr, "nanoarrow_schema")?;
    let schema = Schema::try_from(ffi_schema)?;
    Ok(schema)
}

pub fn import_array_stream(mut xptr: savvy::Sexp) -> savvy::Result<ArrowArrayStreamReader> {
    let ffi_stream: &mut FFI_ArrowArrayStream = import_xptr(&mut xptr, "nanoarrow_array_stream")?;
    let reader = unsafe { ArrowArrayStreamReader::from_raw(ffi_stream as _)? };
    Ok(reader)
}

pub fn import_table_provider(
    mut provider_xptr: savvy::Sexp,
) -> savvy::Result<Arc<dyn TableProvider>> {
    let ffi_provider: &FFI_TableProvider =
        import_xptr(&mut provider_xptr, "datafusion_table_provider")?;
    let provider_impl = ForeignTableProvider::from(ffi_provider);
    Ok(Arc::new(provider_impl))
}

pub fn import_scalar_udf(mut scalar_udf_xptr: savvy::Sexp) -> savvy::Result<ScalarUDF> {
    let ffi_scalar_udf_ref: &FFI_ScalarUDF =
        import_xptr(&mut scalar_udf_xptr, "datafusion_scalar_udf")?;
    let scalar_udf_impl = ForeignScalarUDF::try_from(ffi_scalar_udf_ref)?;
    Ok(scalar_udf_impl.into())
}

fn import_xptr<'a, T>(xptr: &'a mut savvy::Sexp, cls: &str) -> savvy::Result<&'a mut T> {
    if !xptr.is_external_pointer() {
        return Err(savvy_err!(
            "Expected external pointer with class {cls} but got a different R object"
        ));
    }

    if !xptr
        .get_class()
        .map(|classes| classes.contains(&cls))
        .unwrap_or(false)
    {
        return Err(savvy_err!(
            "Expected external pointer of class {cls} but got external pointer with classes {:?}",
            xptr.get_class()
        ));
    }

    let typed_ptr = unsafe { savvy_ffi::R_ExternalPtrAddr(xptr.0) as *mut T };
    if let Some(type_ref) = unsafe { typed_ptr.as_mut() } {
        Ok(type_ref)
    } else {
        Err(savvy_err!("external pointer with class {cls} is null"))
    }
}

#[repr(C)]
pub struct FFIScalarUdfR(pub FFI_ScalarUDF);
impl IntoExtPtrSexp for FFIScalarUdfR {}

#[repr(C)]
pub struct FFITableProviderR(pub FFI_TableProvider);
impl IntoExtPtrSexp for FFITableProviderR {}
