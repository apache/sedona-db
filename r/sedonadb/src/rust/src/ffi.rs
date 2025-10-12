use arrow_array::{ffi::FFI_ArrowSchema, ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream}};
use arrow_schema::Schema;
use datafusion_expr::ScalarUDF;
use datafusion_ffi::udf::{FFI_ScalarUDF, ForeignScalarUDF};
use savvy::savvy_err;

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

pub fn import_scalar_udf(mut scalar_udf_xptr: savvy::Sexp) -> savvy::Result<ScalarUDF> {
    let ffi_scalar_udf_ref: &FFI_ScalarUDF = import_xptr(&mut scalar_udf_xptr, "datafusion_scalar_udf")?;
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
