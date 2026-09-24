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

// Reference ABI for header interoperability tests, copied from:
// https://github.com/apache/arrow/blob/1f789ffd70f96907698e93ef99f4e034caf0788b/cpp/src/arrow/c/abi.h
// Declarations and guards are unchanged; explanatory comments are omitted.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef ARROW_C_DATA_INTERFACE
#  define ARROW_C_DATA_INTERFACE

#  define ARROW_FLAG_DICTIONARY_ORDERED 1
#  define ARROW_FLAG_NULLABLE 2
#  define ARROW_FLAG_MAP_KEYS_SORTED 4

struct ArrowSchema {

  const char* format;
  const char* name;
  const char* metadata;
  int64_t flags;
  int64_t n_children;
  struct ArrowSchema** children;
  struct ArrowSchema* dictionary;

  void (*release)(struct ArrowSchema*);

  void* private_data;
};

struct ArrowArray {

  int64_t length;
  int64_t null_count;
  int64_t offset;
  int64_t n_buffers;
  int64_t n_children;
  const void** buffers;
  struct ArrowArray** children;
  struct ArrowArray* dictionary;

  void (*release)(struct ArrowArray*);

  void* private_data;
};

#  define ARROW_STATISTICS_KEY_AVERAGE_BYTE_WIDTH_EXACT "ARROW:average_byte_width:exact"
#  define ARROW_STATISTICS_KEY_AVERAGE_BYTE_WIDTH_APPROXIMATE \
    "ARROW:average_byte_width:approximate"
#  define ARROW_STATISTICS_KEY_DISTINCT_COUNT_EXACT "ARROW:distinct_count:exact"
#  define ARROW_STATISTICS_KEY_DISTINCT_COUNT_APPROXIMATE \
    "ARROW:distinct_count:approximate"
#  define ARROW_STATISTICS_KEY_MAX_BYTE_WIDTH_EXACT "ARROW:max_byte_width:exact"
#  define ARROW_STATISTICS_KEY_MAX_BYTE_WIDTH_APPROXIMATE \
    "ARROW:max_byte_width:approximate"
#  define ARROW_STATISTICS_KEY_MAX_VALUE_EXACT "ARROW:max_value:exact"
#  define ARROW_STATISTICS_KEY_MAX_VALUE_APPROXIMATE "ARROW:max_value:approximate"
#  define ARROW_STATISTICS_KEY_MIN_VALUE_EXACT "ARROW:min_value:exact"
#  define ARROW_STATISTICS_KEY_MIN_VALUE_APPROXIMATE "ARROW:min_value:approximate"
#  define ARROW_STATISTICS_KEY_NULL_COUNT_EXACT "ARROW:null_count:exact"
#  define ARROW_STATISTICS_KEY_NULL_COUNT_APPROXIMATE "ARROW:null_count:approximate"
#  define ARROW_STATISTICS_KEY_ROW_COUNT_EXACT "ARROW:row_count:exact"
#  define ARROW_STATISTICS_KEY_ROW_COUNT_APPROXIMATE "ARROW:row_count:approximate"

#endif

#ifndef ARROW_C_DEVICE_DATA_INTERFACE
#  define ARROW_C_DEVICE_DATA_INTERFACE

typedef int32_t ArrowDeviceType;

#  define ARROW_DEVICE_CPU 1

#  define ARROW_DEVICE_CUDA 2

#  define ARROW_DEVICE_CUDA_HOST 3

#  define ARROW_DEVICE_OPENCL 4

#  define ARROW_DEVICE_VULKAN 7

#  define ARROW_DEVICE_METAL 8

#  define ARROW_DEVICE_VPI 9

#  define ARROW_DEVICE_ROCM 10

#  define ARROW_DEVICE_ROCM_HOST 11

#  define ARROW_DEVICE_EXT_DEV 12

#  define ARROW_DEVICE_CUDA_MANAGED 13

#  define ARROW_DEVICE_ONEAPI 14

#  define ARROW_DEVICE_WEBGPU 15

#  define ARROW_DEVICE_HEXAGON 16

struct ArrowDeviceArray {

  struct ArrowArray array;

  int64_t device_id;

  ArrowDeviceType device_type;

  void* sync_event;

  int64_t reserved[3];
};

#endif

#ifndef ARROW_C_STREAM_INTERFACE
#  define ARROW_C_STREAM_INTERFACE

struct ArrowArrayStream {

  int (*get_schema)(struct ArrowArrayStream*, struct ArrowSchema* out);

  int (*get_next)(struct ArrowArrayStream*, struct ArrowArray* out);

  const char* (*get_last_error)(struct ArrowArrayStream*);

  void (*release)(struct ArrowArrayStream*);

  void* private_data;
};

#endif

#ifndef ARROW_C_DEVICE_STREAM_INTERFACE
#  define ARROW_C_DEVICE_STREAM_INTERFACE

struct ArrowDeviceArrayStream {

  ArrowDeviceType device_type;

  int (*get_schema)(struct ArrowDeviceArrayStream* self, struct ArrowSchema* out);

  int (*get_next)(struct ArrowDeviceArrayStream* self, struct ArrowDeviceArray* out);

  const char* (*get_last_error)(struct ArrowDeviceArrayStream* self);

  void (*release)(struct ArrowDeviceArrayStream* self);

  void* private_data;
};

#endif

#ifndef ARROW_C_ASYNC_STREAM_INTERFACE
#  define ARROW_C_ASYNC_STREAM_INTERFACE

struct ArrowAsyncTask {

  int (*extract_data)(struct ArrowAsyncTask* self, struct ArrowDeviceArray* out);

  void* private_data;
};

struct ArrowAsyncProducer {

  ArrowDeviceType device_type;

  void (*request)(struct ArrowAsyncProducer* self, int64_t n);

  void (*cancel)(struct ArrowAsyncProducer* self);

  const char* additional_metadata;

  void* private_data;
};

struct ArrowAsyncDeviceStreamHandler {

  int (*on_schema)(struct ArrowAsyncDeviceStreamHandler* self,
                   struct ArrowSchema* stream_schema);

  int (*on_next_task)(struct ArrowAsyncDeviceStreamHandler* self,
                      struct ArrowAsyncTask* task, const char* metadata);

  void (*on_error)(struct ArrowAsyncDeviceStreamHandler* self, int code,
                   const char* message, const char* metadata);

  void (*release)(struct ArrowAsyncDeviceStreamHandler* self);

  struct ArrowAsyncProducer* producer;

  void* private_data;
};

#endif

#ifdef __cplusplus
}
#endif
