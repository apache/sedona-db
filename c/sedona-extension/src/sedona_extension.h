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

#ifndef SEDONA_EXTENSION_H
#define SEDONA_EXTENSION_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Extra guard for versions of Arrow without the canonical guard
#ifndef ARROW_FLAG_DICTIONARY_ORDERED

#ifndef ARROW_C_DATA_INTERFACE
#define ARROW_C_DATA_INTERFACE

#define ARROW_FLAG_DICTIONARY_ORDERED 1
#define ARROW_FLAG_NULLABLE 2
#define ARROW_FLAG_MAP_KEYS_SORTED 4

struct ArrowSchema {
  // Array type description
  const char* format;
  const char* name;
  const char* metadata;
  int64_t flags;
  int64_t n_children;
  struct ArrowSchema** children;
  struct ArrowSchema* dictionary;

  // Release callback
  void (*release)(struct ArrowSchema*);
  // Opaque producer-specific data
  void* private_data;
};

struct ArrowArray {
  // Array data description
  int64_t length;
  int64_t null_count;
  int64_t offset;
  int64_t n_buffers;
  int64_t n_children;
  const void** buffers;
  struct ArrowArray** children;
  struct ArrowArray* dictionary;

  // Release callback
  void (*release)(struct ArrowArray*);
  // Opaque producer-specific data
  void* private_data;
};

#endif  // ARROW_C_DATA_INTERFACE

#ifndef ARROW_C_STREAM_INTERFACE
#define ARROW_C_STREAM_INTERFACE

struct ArrowArrayStream {
  // Callback to get the stream type
  // (will be the same for all arrays in the stream).
  //
  // Return value: 0 if successful, an `errno`-compatible error code otherwise.
  //
  // If successful, the ArrowSchema must be released independently from the stream.
  int (*get_schema)(struct ArrowArrayStream*, struct ArrowSchema* out);

  // Callback to get the next array
  // (if no error and the array is released, the stream has ended)
  //
  // Return value: 0 if successful, an `errno`-compatible error code otherwise.
  //
  // If successful, the ArrowArray must be released independently from the stream.
  int (*get_next)(struct ArrowArrayStream*, struct ArrowArray* out);

  // Callback to get optional detailed error information.
  // This must only be called if the last stream operation failed
  // with a non-0 return code.
  //
  // Return value: pointer to a null-terminated character array describing
  // the last error, or NULL if no description is available.
  //
  // The returned pointer is only valid until the next operation on this stream
  // (including release).
  const char* (*get_last_error)(struct ArrowArrayStream*);

  // Release callback: release the stream's own resources.
  // Note that arrays returned by `get_next` must be individually released.
  void (*release)(struct ArrowArrayStream*);

  // Opaque producer-specific data
  void* private_data;
};

#endif  // ARROW_C_STREAM_INTERFACE
#endif  // ARROW_FLAG_DICTIONARY_ORDERED

/// \brief Simple ABI-stable scalar function implementation
///
/// This object is not thread safe: callers must take care to serialize
/// access to methods if an instance is shared across threads. In general,
/// constructing and initializing this structure should be sufficiently
/// cheap that it shouldn't need to be shared in this way.
///
/// Briefly, the SedonaCScalarKernelImpl is typically the stack-allocated
/// structure that is not thread safe and the SedonaCScalarKernel is the
/// value that lives in a registry (whose job it is to initialize implementations
/// on each stack that needs one).
struct SedonaCScalarKernelImpl {
  /// \brief Initialize the state of this instance and calculate a return type
  ///
  /// The init callback either computes a return ArrowSchema or initializes the
  /// return ArrowSchema to an explicitly released value to indicate that this
  /// implementation does not apply to the arguments passed. An implementation
  /// that does not apply to the arguments passed is not necessarily an error
  /// (there may be another implementation prepared to handle such a case).
  ///
  /// \param arg_types Argument types
  /// \param scalar_args An optional array of scalar arguments. The entire
  /// array may be null to indicate that none of the arguments are scalars, or
  /// individual items in the array may be NULL to indicate that a particular
  /// argument is not a scalar. Any non-NULL arrays must be of length 1.
  /// Implementations MAY take ownership over the elements of scalar_args but
  /// are not required to do so (i.e., caller must check if these elements were
  /// released, and must release them if needed).
  /// \param n_args Number of elements in the arg_types and/or scalar_args arrays.
  /// \param out Will be populated with the return type on success, or initialized
  /// to a released value if this implementation does not apply to the arguments
  /// passed.
  ///
  /// \return An errno-compatible error code, or zero on success.
  int (*init)(struct SedonaCScalarKernelImpl* self,
              const struct ArrowSchema* const* arg_types,
              struct ArrowArray* const* scalar_args, int64_t n_args,
              struct ArrowSchema* out);

  /// \brief Execute a single batch
  ///
  /// \param args Input arguments. Input must be length one (e.g., a scalar)
  /// or the size of the batch. Implementations must handle scalar or array
  /// inputs.
  /// \param n_args The number of pointers in args
  /// \param out Will be populated with the result on success.
  int (*execute)(struct SedonaCScalarKernelImpl* self, struct ArrowArray* const* args,
                 int64_t n_args, int64_t n_rows, struct ArrowArray* out);

  /// \brief Get the last error message
  ///
  /// The result is valid until the next call to a UDF method.
  const char* (*get_last_error)(struct SedonaCScalarKernelImpl* self);

  /// \brief Release this instance
  ///
  /// Implementations of this callback must set self->release to NULL.
  void (*release)(struct SedonaCScalarKernelImpl* self);

  /// \brief Opaque implementation-specific data
  void* private_data;
};

/// \brief Scalar function/kernel initializer
///
/// Usually a SedonaCScalarKernelImpl will be used to execute a single batch
/// (although it may be reused if a caller can serialize callback use). This
/// structure is a factory object that initializes such objects. The
/// SedonaCScalarKernel is designed to be thread-safe and live in a registry.
struct SedonaCScalarKernel {
  /// \brief Function name
  ///
  /// Optional function name. This is used to register the kernel with the
  /// appropriate function when passing this kernel across a boundary.
  const char* (*function_name)(const struct SedonaCScalarKernel* self);

  /// \brief Initialize a new implementation struct
  ///
  /// This callback is thread safe and may be called concurrently from any
  /// thread at any time (as long as this object is valid).
  void (*new_impl)(const struct SedonaCScalarKernel* self,
                   struct SedonaCScalarKernelImpl* out);

  /// \brief Release this instance
  ///
  /// Implementations of this callback must set self->release to NULL.
  void (*release)(struct SedonaCScalarKernel* self);

  /// \brief Opaque implementation-specific data
  void* private_data;
};

struct SedonaCError {
  const char* err;

  uint32_t err_len;

  uint32_t reserved;

  /// \brief Release this instance
  ///
  /// Implementations of this callback must set self->release to NULL.
  void (*release)(struct SedonaCError* self);
};

struct SedonaCExpr {
  // Get a property of this expression (e.g., serialize, extract bbox)
  int (*get_property_schema)(const struct SedonaCExpr* self, const char* property,
                             struct SedonaCError* err);
  int (*get_property)(const struct SedonaCExpr* self, const char* property,
                      const char* args, struct ArrowArray* out, struct SedonaCError* err);

  void* reserved;

  /// \brief Release this instance
  ///
  /// Implementations of this callback must set self->release to NULL.
  void (*release)(struct SedonaCExecutionPlan* self);

  /// \brief Opaque implementation-specific data
  void* private_data;
};

struct SedonaCExecutionPlan;

/// \brief Arguments for execution plan and table provider operations
///
/// This structure is passed to methods that need JSON-serialized arguments,
/// optional execution plans, and/or expressions.
struct SedonaCExecutionPlanArgs {
  /// JSON-serialized arguments
  const uint8_t* args;
  size_t args_len;
  /// Optional array of execution plans
  const struct SedonaCExecutionPlan** exec_plans;
  size_t num_exec_plans;
  /// Optional array of expressions
  const struct SedonaCExpr** exprs;
  size_t num_exprs;
  void* reserved;
};

struct SedonaCExecutionPlan {
  void (*get_schema)(const struct SedonaCExecutionPlan* self, struct ArrowSchema* out);

  // Extract some serializable property from this plan (e.g., plan properties)
  int (*get_property_schema)(const struct SedonaCExecutionPlan* self,
                             const char* property, struct ArrowSchema* out,
                             struct SedonaCError* err);
  int (*get_property)(const struct SedonaCExecutionPlan* self, const char* property,
                      struct SedonaCExecutionPlanArgs* args, struct ArrowArray* out,
                      struct SedonaCError* err);

  // Clone this plan based on some new information (e.g., try pushdown filters)
  int (*with_property)(const struct SedonaCExecutionPlan* self, const char* property,
                       struct SedonaCExecutionPlanArgs* args,
                       struct SedonaCExecutionPlan* out, struct SedonaCError* err);

  // Resolve a synchronous stream for one partition from this plan
  int (*execute)(const struct SedonaCExecutionPlan* self,
                 struct SedonaCExecutionPlanArgs* args, struct ArrowArrayStream* out,
                 struct SedonaCError* err);

  // Future implementation with async streams (don't implement now)
  int (*execute_async)(const struct SedonaCExecutionPlan* self,
                       struct SedonaCExecutionPlanArgs* args, void* out,
                       struct SedonaCError* err);

  // Reserved for future use
  void* reserved;

  /// \brief Release this instance
  ///
  /// Implementations of this callback must set self->release to NULL.
  void (*release)(struct SedonaCExecutionPlan* self);

  /// \brief Opaque implementation-specific data
  void* private_data;
};

/// \brief ABI-stable table provider interface
///
/// This provides a minimal interface for importing a table provider
/// across an FFI boundary in a version-agnostic manner.
struct SedonaCTableProvider {
  /// Get the schema of this table provider
  void (*get_schema)(const struct SedonaCTableProvider* self, struct ArrowSchema* out);

  // Extract some serializable property from this table provider
  int (*get_property_schema)(const struct SedonaCTableProvider* self,
                             const char* property, struct ArrowSchema* out,
                             struct SedonaCError* err);
  int (*get_property)(const struct SedonaCTableProvider* self, const char* property,
                      struct SedonaCExecutionPlanArgs* args, struct ArrowArray* out,
                      struct SedonaCError* err);

  /// Perform a scan operation and return an execution plan
  ///
  /// The args parameter contains JSON-serialized scan arguments.
  /// Returns an execution plan that can be used to read the data.
  int (*scan)(const struct SedonaCTableProvider* self,
              struct SedonaCExecutionPlanArgs* args, struct SedonaCExecutionPlan* out,
              struct SedonaCError* err);

  /// Perform an insert operation
  ///
  /// The args parameter contains JSON-serialized insert arguments.
  /// The exec_plans field should contain the plan providing rows to insert.
  /// Returns an execution plan that performs the insert.
  int (*insert)(const struct SedonaCTableProvider* self,
                struct SedonaCExecutionPlanArgs* args, struct SedonaCExecutionPlan* out,
                struct SedonaCError* err);

  /// Perform an update operation
  ///
  /// The args parameter contains JSON-serialized update arguments
  /// (filters, column assignments, etc.).
  /// Returns an execution plan that performs the update.
  int (*update)(const struct SedonaCTableProvider* self,
                struct SedonaCExecutionPlanArgs* args, struct SedonaCExecutionPlan* out,
                struct SedonaCError* err);

  /// Perform a delete operation
  ///
  /// The args parameter contains JSON-serialized delete arguments
  /// (filters, etc.).
  /// Returns an execution plan that performs the delete.
  int (*delete_rows)(const struct SedonaCTableProvider* self,
                     struct SedonaCExecutionPlanArgs* args,
                     struct SedonaCExecutionPlan* out, struct SedonaCError* err);

  void* reserved;

  /// \brief Release this instance
  ///
  /// Implementations of this callback must set self->release to NULL.
  void (*release)(struct SedonaCTableProvider* self);

  /// \brief Opaque implementation-specific data
  void* private_data;
};

#ifdef __cplusplus
}
#endif

#endif
