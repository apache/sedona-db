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

#include <Rinternals.h>

#include <R_ext/Parse.h>
#include <stdint.h>

#include "rust/api.h"

static uintptr_t TAGGED_POINTER_MASK = (uintptr_t)1;

SEXP handle_result(SEXP res_) {
  uintptr_t res = (uintptr_t)res_;

  // An error is indicated by tag.
  if ((res & TAGGED_POINTER_MASK) == 1) {
    // Remove tag
    SEXP res_aligned = (SEXP)(res & ~TAGGED_POINTER_MASK);

    // Currently, there are two types of error cases:
    //
    //   1. Error from Rust code
    //   2. Error from R's C API, which is caught by R_UnwindProtect()
    //
    if (TYPEOF(res_aligned) == CHARSXP) {
      // In case 1, the result is an error message that can be passed to
      // Rf_errorcall() directly.
      Rf_errorcall(R_NilValue, "%s", CHAR(res_aligned));
    } else {
      // In case 2, the result is the token to restart the
      // cleanup process on R's side.
      R_ContinueUnwind(res_aligned);
    }
  }

  return (SEXP)res;
}

SEXP savvy_init_r_runtime__impl(DllInfo *c_arg___dll_info) {
  SEXP res = savvy_init_r_runtime__ffi(c_arg___dll_info);
  return handle_result(res);
}

SEXP savvy_init_r_runtime_interrupts__impl(SEXP c_arg__interrupts_call,
                                           SEXP c_arg__pkg_env) {
  SEXP res = savvy_init_r_runtime_interrupts__ffi(c_arg__interrupts_call,
                                                  c_arg__pkg_env);
  return handle_result(res);
}

SEXP savvy_int_times_int__impl(SEXP c_arg__x, SEXP c_arg__y) {
  SEXP res = savvy_int_times_int__ffi(c_arg__x, c_arg__y);
  return handle_result(res);
}

SEXP savvy_sedonadb_adbc_init_func__impl(void) {
  SEXP res = savvy_sedonadb_adbc_init_func__ffi();
  return handle_result(res);
}

SEXP savvy_to_upper__impl(SEXP c_arg__x) {
  SEXP res = savvy_to_upper__ffi(c_arg__x);
  return handle_result(res);
}

SEXP savvy_InternalContext_data_frame_from_array_stream__impl(
    SEXP self__, SEXP c_arg__stream_xptr, SEXP c_arg__collect_now) {
  SEXP res = savvy_InternalContext_data_frame_from_array_stream__ffi(
      self__, c_arg__stream_xptr, c_arg__collect_now);
  return handle_result(res);
}

SEXP savvy_InternalContext_deregister_table__impl(SEXP self__,
                                                  SEXP c_arg__table_ref) {
  SEXP res =
      savvy_InternalContext_deregister_table__ffi(self__, c_arg__table_ref);
  return handle_result(res);
}

SEXP savvy_InternalContext_new__impl(void) {
  SEXP res = savvy_InternalContext_new__ffi();
  return handle_result(res);
}

SEXP savvy_InternalContext_read_parquet__impl(SEXP self__, SEXP c_arg__paths) {
  SEXP res = savvy_InternalContext_read_parquet__ffi(self__, c_arg__paths);
  return handle_result(res);
}

SEXP savvy_InternalContext_sql__impl(SEXP self__, SEXP c_arg__query) {
  SEXP res = savvy_InternalContext_sql__ffi(self__, c_arg__query);
  return handle_result(res);
}

SEXP savvy_InternalContext_view__impl(SEXP self__, SEXP c_arg__table_ref) {
  SEXP res = savvy_InternalContext_view__ffi(self__, c_arg__table_ref);
  return handle_result(res);
}

SEXP savvy_InternalDataFrame_collect__impl(SEXP self__, SEXP c_arg__out) {
  SEXP res = savvy_InternalDataFrame_collect__ffi(self__, c_arg__out);
  return handle_result(res);
}

SEXP savvy_InternalDataFrame_compute__impl(SEXP self__, SEXP c_arg__ctx) {
  SEXP res = savvy_InternalDataFrame_compute__ffi(self__, c_arg__ctx);
  return handle_result(res);
}

SEXP savvy_InternalDataFrame_count__impl(SEXP self__) {
  SEXP res = savvy_InternalDataFrame_count__ffi(self__);
  return handle_result(res);
}

SEXP savvy_InternalDataFrame_limit__impl(SEXP self__, SEXP c_arg__n) {
  SEXP res = savvy_InternalDataFrame_limit__ffi(self__, c_arg__n);
  return handle_result(res);
}

SEXP savvy_InternalDataFrame_primary_geometry_column_index__impl(SEXP self__) {
  SEXP res = savvy_InternalDataFrame_primary_geometry_column_index__ffi(self__);
  return handle_result(res);
}

SEXP savvy_InternalDataFrame_show__impl(SEXP self__, SEXP c_arg__ctx,
                                        SEXP c_arg__width_chars,
                                        SEXP c_arg__ascii, SEXP c_arg__limit) {
  SEXP res = savvy_InternalDataFrame_show__ffi(
      self__, c_arg__ctx, c_arg__width_chars, c_arg__ascii, c_arg__limit);
  return handle_result(res);
}

SEXP savvy_InternalDataFrame_to_arrow_schema__impl(SEXP self__,
                                                   SEXP c_arg__out) {
  SEXP res = savvy_InternalDataFrame_to_arrow_schema__ffi(self__, c_arg__out);
  return handle_result(res);
}

SEXP savvy_InternalDataFrame_to_arrow_stream__impl(SEXP self__,
                                                   SEXP c_arg__out) {
  SEXP res = savvy_InternalDataFrame_to_arrow_stream__ffi(self__, c_arg__out);
  return handle_result(res);
}

SEXP savvy_InternalDataFrame_to_view__impl(SEXP self__, SEXP c_arg__ctx,
                                           SEXP c_arg__table_ref,
                                           SEXP c_arg__overwrite) {
  SEXP res = savvy_InternalDataFrame_to_view__ffi(
      self__, c_arg__ctx, c_arg__table_ref, c_arg__overwrite);
  return handle_result(res);
}

SEXP savvy_Person_associated_function__impl(void) {
  SEXP res = savvy_Person_associated_function__ffi();
  return handle_result(res);
}

SEXP savvy_Person_name__impl(SEXP self__) {
  SEXP res = savvy_Person_name__ffi(self__);
  return handle_result(res);
}

SEXP savvy_Person_new__impl(void) {
  SEXP res = savvy_Person_new__ffi();
  return handle_result(res);
}

SEXP savvy_Person_set_name__impl(SEXP self__, SEXP c_arg__name) {
  SEXP res = savvy_Person_set_name__ffi(self__, c_arg__name);
  return handle_result(res);
}

static const R_CallMethodDef CallEntries[] = {
    {"savvy_init_r_runtime_interrupts__impl",
     (DL_FUNC)&savvy_init_r_runtime_interrupts__impl, 2},
    {"savvy_int_times_int__impl", (DL_FUNC)&savvy_int_times_int__impl, 2},
    {"savvy_sedonadb_adbc_init_func__impl",
     (DL_FUNC)&savvy_sedonadb_adbc_init_func__impl, 0},
    {"savvy_to_upper__impl", (DL_FUNC)&savvy_to_upper__impl, 1},
    {"savvy_InternalContext_data_frame_from_array_stream__impl",
     (DL_FUNC)&savvy_InternalContext_data_frame_from_array_stream__impl, 3},
    {"savvy_InternalContext_deregister_table__impl",
     (DL_FUNC)&savvy_InternalContext_deregister_table__impl, 2},
    {"savvy_InternalContext_new__impl",
     (DL_FUNC)&savvy_InternalContext_new__impl, 0},
    {"savvy_InternalContext_read_parquet__impl",
     (DL_FUNC)&savvy_InternalContext_read_parquet__impl, 2},
    {"savvy_InternalContext_sql__impl",
     (DL_FUNC)&savvy_InternalContext_sql__impl, 2},
    {"savvy_InternalContext_view__impl",
     (DL_FUNC)&savvy_InternalContext_view__impl, 2},
    {"savvy_InternalDataFrame_collect__impl",
     (DL_FUNC)&savvy_InternalDataFrame_collect__impl, 2},
    {"savvy_InternalDataFrame_compute__impl",
     (DL_FUNC)&savvy_InternalDataFrame_compute__impl, 2},
    {"savvy_InternalDataFrame_count__impl",
     (DL_FUNC)&savvy_InternalDataFrame_count__impl, 1},
    {"savvy_InternalDataFrame_limit__impl",
     (DL_FUNC)&savvy_InternalDataFrame_limit__impl, 2},
    {"savvy_InternalDataFrame_primary_geometry_column_index__impl",
     (DL_FUNC)&savvy_InternalDataFrame_primary_geometry_column_index__impl, 1},
    {"savvy_InternalDataFrame_show__impl",
     (DL_FUNC)&savvy_InternalDataFrame_show__impl, 5},
    {"savvy_InternalDataFrame_to_arrow_schema__impl",
     (DL_FUNC)&savvy_InternalDataFrame_to_arrow_schema__impl, 2},
    {"savvy_InternalDataFrame_to_arrow_stream__impl",
     (DL_FUNC)&savvy_InternalDataFrame_to_arrow_stream__impl, 2},
    {"savvy_InternalDataFrame_to_view__impl",
     (DL_FUNC)&savvy_InternalDataFrame_to_view__impl, 4},
    {"savvy_Person_associated_function__impl",
     (DL_FUNC)&savvy_Person_associated_function__impl, 0},
    {"savvy_Person_name__impl", (DL_FUNC)&savvy_Person_name__impl, 1},
    {"savvy_Person_new__impl", (DL_FUNC)&savvy_Person_new__impl, 0},
    {"savvy_Person_set_name__impl", (DL_FUNC)&savvy_Person_set_name__impl, 2},
    {NULL, NULL, 0}};

void R_init_sedonadb(DllInfo *dll) {
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);

  // Functions for initialization, if any.
  savvy_init_r_runtime__impl(dll);
}
