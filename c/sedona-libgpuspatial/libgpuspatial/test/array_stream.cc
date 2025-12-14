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
#include <string>
#include <vector>

#include "array_stream.hpp"

#include "nanoarrow/nanoarrow.hpp"
#include "nanoarrow/nanoarrow_ipc.hpp"

namespace gpuspatial {

void ArrayStreamFromWKT(const std::vector<std::vector<std::string>>& batches,
                        enum GeoArrowType type, struct ArrowArrayStream* out) {
  nanoarrow::UniqueSchema schema;
  geoarrow::GeometryDataType::Make(type).InitSchema(schema.get());

  std::vector<nanoarrow::UniqueArray> arrays;
  for (const auto& batch : batches) {
    nanoarrow::UniqueArray array;
    testing::MakeWKBArrayFromWKT(batch, array.get());
    arrays.push_back(std::move(array));
  }

  nanoarrow::VectorArrayStream(schema.get(), std::move(arrays)).ToArrayStream(out);
}

/// \brief An ArrowArrayStream wrapper that plucks a specific column
class ColumnArrayStream {
 public:
  ColumnArrayStream(nanoarrow::UniqueArrayStream inner, std::string column_name)
      : inner_(std::move(inner)), column_name_(std::move(column_name)) {}

  void ToArrayStream(struct ArrowArrayStream* out) {
    ColumnArrayStream* impl =
        new ColumnArrayStream(std::move(inner_), std::move(column_name_));
    nanoarrow::ArrayStreamFactory<ColumnArrayStream>::InitArrayStream(impl, out);
  }

 private:
  struct ArrowError last_error_{};
  nanoarrow::UniqueArrayStream inner_;
  std::string column_name_;
  int64_t column_index_{-1};

  friend class nanoarrow::ArrayStreamFactory<ColumnArrayStream>;

  int GetSchema(struct ArrowSchema* schema) {
    NANOARROW_RETURN_NOT_OK(ResolveColumnIndex());
    nanoarrow::UniqueSchema inner_schema;
    NANOARROW_RETURN_NOT_OK(
        ArrowArrayStreamGetSchema(inner_.get(), inner_schema.get(), &last_error_));
    ArrowSchemaMove(inner_schema->children[column_index_], schema);
    return NANOARROW_OK;
  }

  int GetNext(struct ArrowArray* array) {
    NANOARROW_RETURN_NOT_OK(ResolveColumnIndex());
    nanoarrow::UniqueArray inner_array;
    NANOARROW_RETURN_NOT_OK(
        ArrowArrayStreamGetNext(inner_.get(), inner_array.get(), &last_error_));
    if (inner_array->release == nullptr) {
      ArrowArrayMove(inner_array.get(), array);
    } else {
      ArrowArrayMove(inner_array->children[column_index_], array);
    }

    return NANOARROW_OK;
  }

  const char* GetLastError() { return last_error_.message; }

  int ResolveColumnIndex() {
    if (column_index_ != -1) {
      return NANOARROW_OK;
    }

    nanoarrow::UniqueSchema inner_schema;
    NANOARROW_RETURN_NOT_OK(
        ArrowArrayStreamGetSchema(inner_.get(), inner_schema.get(), &last_error_));
    for (int64_t i = 0; i < inner_schema->n_children; i++) {
      if (inner_schema->children[i]->name != nullptr &&
          inner_schema->children[i]->name == column_name_) {
        column_index_ = i;
        return NANOARROW_OK;
      }
    }

    ArrowErrorSet(&last_error_, "Can't resolve column %s from inner schema",
                  column_name_.c_str());
    return EINVAL;
  }
};

void ArrayStreamFromIpc(const std::string& filename, std::string geometry_column,
                        struct ArrowArrayStream* out) {
  FILE* file = fopen(filename.c_str(), "rb");
  if (file == nullptr) {
    throw std::runtime_error("Failed to open " + filename);
  }

  nanoarrow::ipc::UniqueInputStream input_stream;
  NANOARROW_THROW_NOT_OK(ArrowIpcInputStreamInitFile(input_stream.get(), file, true));

  nanoarrow::UniqueArrayStream inner;
  NANOARROW_THROW_NOT_OK(
      ArrowIpcArrayStreamReaderInit(inner.get(), input_stream.get(), nullptr));
  ColumnArrayStream(std::move(inner), std::move(geometry_column)).ToArrayStream(out);
}

}  // namespace gpuspatial
