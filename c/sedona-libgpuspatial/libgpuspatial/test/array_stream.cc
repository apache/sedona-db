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
#include "test_common.hpp"

#include "nanoarrow/nanoarrow.hpp"

#include "arrow/api.h"
#include "parquet/arrow/reader.h"

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

// Function to read a single Parquet file and extract a column.
arrow::Status ReadParquetFromFile(
    arrow::fs::FileSystem* fs,     // 1. Filesystem pointer (e.g., LocalFileSystem)
    const std::string& file_path,  // 2. Single file path instead of a folder
    int64_t batch_size, const char* column_name,
    std::vector<std::shared_ptr<arrow::Array>>& out_arrays) {
  // 1. Get FileInfo for the single path
  ARROW_ASSIGN_OR_RAISE(auto file_info, fs->GetFileInfo(file_path));

  // Check if the path points to a file
  if (file_info.type() != arrow::fs::FileType::File) {
    return arrow::Status::Invalid("Path is not a file: ", file_path);
  }

  // 2. Open the input file
  ARROW_ASSIGN_OR_RAISE(auto input_file, fs->OpenInputFile(file_info));

  // 3. Open the Parquet file and create an Arrow reader
  ARROW_ASSIGN_OR_RAISE(auto arrow_reader, parquet::arrow::OpenFile(
                                               input_file, arrow::default_memory_pool()));

  // 4. Set the batch size
  arrow_reader->set_batch_size(batch_size);

  // 5. Get the RecordBatchReader
  auto rb_reader = arrow_reader->GetRecordBatchReader().ValueOrDie();
  // 6. Read all record batches and extract the column
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;

    // Read the next batch
    ARROW_THROW_NOT_OK(rb_reader->ReadNext(&batch));

    // Check for end of stream
    if (!batch) {
      break;
    }

    // Extract the specified column and add to the output vector
    std::shared_ptr<arrow::Array> column_array = batch->GetColumnByName(column_name);
    if (!column_array) {
      return arrow::Status::Invalid("Column not found: ", column_name);
    }
    out_arrays.push_back(column_array);
  }

  return arrow::Status::OK();
}

std::vector<std::shared_ptr<arrow::Array>> ReadParquet(const std::string& path,
                                                       int batch_size) {
  using namespace TestUtils;

  auto fs = std::make_shared<arrow::fs::LocalFileSystem>();

  std::vector<std::shared_ptr<arrow::Array>> build_arrays;
  ARROW_THROW_NOT_OK(
      ReadParquetFromFile(fs.get(), path, batch_size, "geometry", build_arrays));
  return build_arrays;
}
}  // namespace gpuspatial
