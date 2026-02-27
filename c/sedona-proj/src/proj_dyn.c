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

#include "proj_dyn.h"

// Original source:
// https://github.com/apache/sedona/blob/670bb4c4a6fea49f0b0159ebdf2a92f00d3ed07a/python/src/geos_c_dyn.c

#if defined(_WIN32)
#define TARGETING_WINDOWS
#include <tchar.h>
#include <windows.h>
#else
#include <dlfcn.h>
#include <errno.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef TARGETING_WINDOWS
static void win32_get_last_error(char* err_msg, int len) {
  wchar_t info[256];
  unsigned int error_code = GetLastError();
  int info_length = FormatMessageW(
      FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, /* flags */
      NULL,                                                       /* message source*/
      error_code,                                /* the message (error) ID */
      MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), /* default language */
      info,                                      /* the buffer */
      sizeof(info) / sizeof(wchar_t),            /* size in wchars */
      NULL);
  int num_bytes =
      WideCharToMultiByte(CP_UTF8, 0, info, info_length, err_msg, len, NULL, NULL);
  num_bytes = (num_bytes < (len - 1)) ? num_bytes : (len - 1);
  err_msg[num_bytes] = '\0';
}
#endif

static void* try_load_proj_symbol(void* handle, const char* func_name) {
#ifndef TARGETING_WINDOWS
  return dlsym(handle, func_name);
#else
  return GetProcAddress((HMODULE)handle, func_name);
#endif
}

static int load_proj_symbol(void* handle, const char* func_name, void** p_func,
                            char* err_msg, int len) {
  void* func = try_load_proj_symbol(handle, func_name);
  if (func == NULL) {
#ifndef TARGETING_WINDOWS
    snprintf(err_msg, len, "%s", dlerror());
#else
    win32_get_last_error(err_msg, len);
#endif
    return -1;
  }
  *p_func = func;
  return 0;
}

#define LOAD_PROJ_FUNCTION(api, func)                                               \
  if (load_proj_symbol(handle, #func, (void**)(&(api)->func), err_msg, len) != 0) { \
    proj_dyn_release_api(api);                                                      \
    return -1;                                                                      \
  }

static void proj_dyn_release_api(struct ProjApi* api) {
#ifdef TARGETING_WINDOWS
  FreeLibrary((HMODULE)api->private_data);
#else
  dlclose(api->private_data);
#endif
  memset(api, 0, sizeof(struct ProjApi));
}

static int load_proj_from_handle(struct ProjApi* api, void* handle, char* err_msg,
                                 int len) {
  LOAD_PROJ_FUNCTION(api, proj_area_create);
  LOAD_PROJ_FUNCTION(api, proj_area_destroy);
  LOAD_PROJ_FUNCTION(api, proj_area_set_bbox);
  LOAD_PROJ_FUNCTION(api, proj_context_create);
  LOAD_PROJ_FUNCTION(api, proj_context_destroy);
  LOAD_PROJ_FUNCTION(api, proj_context_errno_string);
  LOAD_PROJ_FUNCTION(api, proj_context_errno);
  LOAD_PROJ_FUNCTION(api, proj_context_set_database_path);
  LOAD_PROJ_FUNCTION(api, proj_context_set_search_paths);
  LOAD_PROJ_FUNCTION(api, proj_create_crs_to_crs_from_pj);
  LOAD_PROJ_FUNCTION(api, proj_create);
  LOAD_PROJ_FUNCTION(api, proj_cs_get_axis_count);
  LOAD_PROJ_FUNCTION(api, proj_destroy);
  LOAD_PROJ_FUNCTION(api, proj_errno_reset);
  LOAD_PROJ_FUNCTION(api, proj_errno);
  LOAD_PROJ_FUNCTION(api, proj_info);
  LOAD_PROJ_FUNCTION(api, proj_log_level);
  LOAD_PROJ_FUNCTION(api, proj_normalize_for_visualization);
  LOAD_PROJ_FUNCTION(api, proj_trans);
  LOAD_PROJ_FUNCTION(api, proj_trans_array);
  LOAD_PROJ_FUNCTION(api, proj_as_projjson);

  api->release = &proj_dyn_release_api;
  api->private_data = handle;

  return 0;
}

#undef LOAD_PROJ_FUNCTION

int proj_dyn_api_init(struct ProjApi* api, const char* shared_object_path, char* err_msg,
                      int len) {
#ifndef TARGETING_WINDOWS
  void* handle = dlopen(shared_object_path, RTLD_LOCAL | RTLD_NOW);
  if (handle == NULL) {
    snprintf(err_msg, len, "%s", dlerror());
    return -1;
  }
#else
  int num_chars = MultiByteToWideChar(CP_UTF8, 0, shared_object_path, -1, NULL, 0);
  wchar_t* wpath = calloc(num_chars, sizeof(wchar_t));
  if (wpath == NULL) {
    snprintf(err_msg, len, "%s", "Cannot allocate memory for wpath");
    return -1;
  }
  MultiByteToWideChar(CP_UTF8, 0, shared_object_path, -1, wpath, num_chars);
  HMODULE module = LoadLibraryW(wpath);
  free(wpath);
  if (module == NULL) {
    win32_get_last_error(err_msg, len);
    return -1;
  }
  void* handle = module;
#endif
  return load_proj_from_handle(api, handle, err_msg, len);
}
