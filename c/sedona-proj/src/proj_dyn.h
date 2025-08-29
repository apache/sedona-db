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

#include <stddef.h>

#ifndef PROJ_DYN_H_INCLUDED
#define PROJ_DYN_H_INCLUDED

// Type definitions copied from proj.h
struct pj_ctx;
typedef struct pj_ctx PJ_CONTEXT;

struct PJ_AREA;
typedef struct PJ_AREA PJ_AREA;

struct PJconsts;
typedef struct PJconsts PJ;

typedef struct {
  double x, y, z, t;
} PJ_XYZT;

typedef union {
  double v[4];
  PJ_XYZT xyzt;
} PJ_COORD;

typedef enum { PJ_FWD = 1, PJ_IDENT = 0, PJ_INV = -1 } PJ_DIRECTION;

typedef enum PJ_LOG_LEVEL {
  PJ_LOG_NONE = 0,
  PJ_LOG_ERROR = 1,
  PJ_LOG_DEBUG = 2,
  PJ_LOG_TRACE = 3,
  PJ_LOG_TELL = 4,
} PJ_LOG_LEVEL;

typedef struct {
  int major;
  int minor;
  int patch;
  const char* release;
  const char* version;
  const char* searchpath;
} PJ_INFO;

struct ProjApi {
  PJ_AREA* (*proj_area_create)(void);
  void (*proj_area_destroy)(PJ_AREA* area);
  int (*proj_area_set_bbox)(PJ_AREA* area, double west_lon_degree,
                            double south_lat_degree, double east_lon_degree,
                            double north_lat_degree);
  PJ_CONTEXT* (*proj_context_create)(void);
  void (*proj_context_destroy)(PJ_CONTEXT* ctx);
  int (*proj_context_errno)(PJ_CONTEXT* ctx);
  const char* (*proj_context_errno_string)(PJ_CONTEXT* ctx, int err);
  int (*proj_context_set_database_path)(PJ_CONTEXT* ctx, const char* dbPath,
                                        const char* const* auxDbPaths,
                                        const char* const* options);
  void (*proj_context_set_search_paths)(PJ_CONTEXT* ctx, int count_paths,
                                        const char* const* paths);
  PJ* (*proj_create)(PJ_CONTEXT* ctx, const char* definition);
  PJ* (*proj_create_crs_to_crs_from_pj)(PJ_CONTEXT* ctx, PJ* source_crs, PJ* target_crs,
                                        PJ_AREA* area, const char* const* options);
  int (*proj_cs_get_axis_count)(PJ_CONTEXT* ctx, const PJ* cs);
  void (*proj_destroy)(PJ* P);
  int (*proj_errno)(const PJ* P);
  void (*proj_errno_reset)(PJ* P);
  PJ_INFO (*proj_info)(void);
  PJ_LOG_LEVEL (*proj_log_level)(PJ_CONTEXT* ctx, PJ_LOG_LEVEL level);
  PJ* (*proj_normalize_for_visualization)(PJ_CONTEXT* ctx, const PJ* obj);
  PJ_COORD (*proj_trans)(PJ* P, PJ_DIRECTION direction, PJ_COORD coord);
  PJ_COORD (*proj_trans_array)(PJ* P, PJ_DIRECTION direction, size_t n, PJ_COORD* coord);
  void (*release)(struct ProjApi*);
  void* private_data;
};

int proj_dyn_api_init(struct ProjApi* api, const char* shared_object_path, char* err_msg,
                      int len);

#endif
