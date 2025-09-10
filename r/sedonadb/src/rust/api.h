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

SEXP savvy_init_r_runtime__ffi(DllInfo *c_arg___dll_info);
SEXP savvy_init_r_runtime_interrupts__ffi(SEXP c_arg__interrupts_call,
                                          SEXP c_arg__pkg_env);
SEXP savvy_sedonadb_adbc_init_func__ffi(void);

// methods and associated functions for InternalContext
SEXP savvy_InternalContext_data_frame_from_array_stream__ffi(
    SEXP self__, SEXP c_arg__stream_xptr, SEXP c_arg__collect_now);
SEXP savvy_InternalContext_deregister_table__ffi(SEXP self__,
                                                 SEXP c_arg__table_ref);
SEXP savvy_InternalContext_new__ffi(void);
SEXP savvy_InternalContext_read_parquet__ffi(SEXP self__, SEXP c_arg__paths);
SEXP savvy_InternalContext_sql__ffi(SEXP self__, SEXP c_arg__query);
SEXP savvy_InternalContext_view__ffi(SEXP self__, SEXP c_arg__table_ref);

// methods and associated functions for InternalDataFrame
SEXP savvy_InternalDataFrame_collect__ffi(SEXP self__, SEXP c_arg__out);
SEXP savvy_InternalDataFrame_compute__ffi(SEXP self__, SEXP c_arg__ctx);
SEXP savvy_InternalDataFrame_count__ffi(SEXP self__);
SEXP savvy_InternalDataFrame_limit__ffi(SEXP self__, SEXP c_arg__n);
SEXP savvy_InternalDataFrame_primary_geometry_column_index__ffi(SEXP self__);
SEXP savvy_InternalDataFrame_show__ffi(SEXP self__, SEXP c_arg__ctx,
                                       SEXP c_arg__width_chars,
                                       SEXP c_arg__ascii, SEXP c_arg__limit);
SEXP savvy_InternalDataFrame_to_arrow_schema__ffi(SEXP self__, SEXP c_arg__out);
SEXP savvy_InternalDataFrame_to_arrow_stream__ffi(SEXP self__, SEXP c_arg__out);
SEXP savvy_InternalDataFrame_to_view__ffi(SEXP self__, SEXP c_arg__ctx,
                                          SEXP c_arg__table_ref,
                                          SEXP c_arg__overwrite);
