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
#include "array_stream.hpp"
#include "gpuspatial/index/spatial_joiner.cuh"
#include "gpuspatial/loader/device_geometries.cuh"
#include "test_common.hpp"

#include "geoarrow_geos/geoarrow_geos.hpp"
#include "nanoarrow/nanoarrow.hpp"

#include <geoarrow/geoarrow.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <numeric>  // For std::iota

namespace gpuspatial {
// Function to read a single Parquet file and extract a column.
static arrow::Status ReadParquetFromFile(
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

  std::cout << "--- Processing Parquet file: " << file_path << " ---" << std::endl;

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

  std::cout << "Finished reading. Total arrays extracted: " << out_arrays.size()
            << std::endl;
  return arrow::Status::OK();
}

using GeosBinaryPredicateFn = char (*)(GEOSContextHandle_t, const GEOSGeometry*,
                                       const GEOSGeometry*);
static GeosBinaryPredicateFn GetGeosPredicateFn(Predicate predicate) {
  switch (predicate) {
    case Predicate::kContains:
      return &GEOSContains_r;
    case Predicate::kIntersects:
      return &GEOSIntersects_r;
    case Predicate::kWithin:
      return &GEOSWithin_r;
    case Predicate::kEquals:
      return &GEOSEquals_r;
    case Predicate::kTouches:
      return &GEOSTouches_r;
    default:
      throw std::out_of_range("Unsupported GEOS predicate enumeration value.");
  }
}

void TestJoiner(const std::string& build_parquet_path,
                const std::string& stream_parquet_path, Predicate predicate,
                int batch_size = 10) {
  using namespace TestUtils;
  auto fs = std::make_shared<arrow::fs::LocalFileSystem>();
  SpatialJoiner::SpatialJoinerConfig config;
  std::string ptx_root = TestUtils::GetTestShaderPath();

  config.ptx_root = ptx_root.c_str();
  SpatialJoiner spatial_joiner;

  spatial_joiner.Init(&config);
  spatial_joiner.Clear();

  geoarrow::geos::ArrayReader reader;

  class GEOSCppHandle {
   public:
    GEOSContextHandle_t handle;

    GEOSCppHandle() { handle = GEOS_init_r(); }

    ~GEOSCppHandle() { GEOS_finish_r(handle); }
  };
  GEOSCppHandle handle;

  reader.InitFromEncoding(handle.handle, GEOARROW_GEOS_ENCODING_WKB);

  geoarrow::geos::GeometryVector geom_build(handle.handle);

  auto get_total_length = [](const std::vector<std::shared_ptr<arrow::Array>>& arrays) {
    size_t total_length = 0;
    for (const auto& array : arrays) {
      total_length += array->length();
    }
    return total_length;
  };

  std::vector<std::shared_ptr<arrow::Array>> build_arrays;
  ARROW_THROW_NOT_OK(ReadParquetFromFile(fs.get(), build_parquet_path, batch_size,
                                         "geometry", build_arrays));

  // Using GEOS for reference
  geom_build.resize(get_total_length(build_arrays));
  size_t tail_build = 0;
  auto* tree = GEOSSTRtree_create_r(handle.handle, 10);

  for (auto& array : build_arrays) {
    nanoarrow::UniqueArray unique_array;
    nanoarrow::UniqueSchema unique_schema;

    ARROW_THROW_NOT_OK(
        arrow::ExportArray(*array, unique_array.get(), unique_schema.get()));

    spatial_joiner.PushBuild(unique_schema.get(), unique_array.get(), 0,
                             unique_array->length);

    // geos for reference
    size_t n_build;

    ASSERT_EQ(reader.Read(unique_array.get(), 0, unique_array->length,
                          geom_build.mutable_data() + tail_build, &n_build),
              GEOARROW_GEOS_OK);

    for (size_t offset = tail_build; offset < tail_build + n_build; offset++) {
      auto* geom = geom_build.borrow(offset);
      auto* box = GEOSEnvelope_r(handle.handle, geom);
      GEOSGeom_setUserData_r(handle.handle, (GEOSGeometry*)geom, (void*)offset);
      GEOSSTRtree_insert_r(handle.handle, tree, box, (void*)geom);
      GEOSGeom_destroy_r(handle.handle, box);
    }
    tail_build += n_build;
  }
  spatial_joiner.FinishBuilding();
  ASSERT_EQ(GEOSSTRtree_build_r(handle.handle, tree), 1);

  std::vector<std::shared_ptr<arrow::Array>> stream_arrays;
  ARROW_THROW_NOT_OK(ReadParquetFromFile(
      fs.get(), stream_parquet_path, batch_size, "geometry", stream_arrays));
  int array_index_offset = 0;
  auto context = spatial_joiner.CreateContext();

  for (auto& array : stream_arrays) {
    nanoarrow::UniqueArray unique_array;
    nanoarrow::UniqueSchema unique_schema;

    ARROW_THROW_NOT_OK(
        arrow::ExportArray(*array, unique_array.get(), unique_schema.get()));
    std::vector<uint32_t> build_indices, stream_indices;

    spatial_joiner.PushStream(context.get(), unique_schema.get(), unique_array.get(), 0,
                              unique_array->length, predicate, &build_indices,
                              &stream_indices, array_index_offset);

    geoarrow::geos::GeometryVector geom_stream(handle.handle);
    size_t n_stream;
    geom_stream.resize(array->length());
    ASSERT_EQ(reader.Read(unique_array.get(), 0, unique_array->length,
                          geom_stream.mutable_data(), &n_stream),
              GEOARROW_GEOS_OK);
    struct Payload {
      GEOSContextHandle_t handle;
      const GEOSGeometry* geom;
      int64_t stream_index_offset;
      std::vector<uint32_t> build_indices;
      std::vector<uint32_t> stream_indices;
      Predicate predicate;
    };

    Payload payload;
    payload.predicate = predicate;
    payload.handle = handle.handle;

    payload.stream_index_offset = array_index_offset;

    for (size_t offset = 0; offset < n_stream; offset++) {
      auto* geom = geom_stream.borrow(offset);
      GEOSGeom_setUserData_r(handle.handle, (GEOSGeometry*)geom, (void*)offset);
      payload.geom = geom;

      GEOSSTRtree_query_r(
          handle.handle, tree, geom,
          [](void* item, void* data) {
            auto* geom_build = (GEOSGeometry*)item;
            auto* payload = (Payload*)data;
            auto* geom_stream = payload->geom;

            if (GetGeosPredicateFn(payload->predicate)(payload->handle, geom_build,
                                                       geom_stream) == 1) {
              auto build_id = (size_t)GEOSGeom_getUserData_r(payload->handle, geom_build);
              auto stream_id =
                  (size_t)GEOSGeom_getUserData_r(payload->handle, geom_stream);
              payload->build_indices.push_back(build_id);
              payload->stream_indices.push_back(payload->stream_index_offset + stream_id);
            }
          },
          (void*)&payload);
    }

    ASSERT_EQ(payload.build_indices.size(), build_indices.size());
    ASSERT_EQ(payload.stream_indices.size(), stream_indices.size());
    sort_vectors_by_index(payload.build_indices, payload.stream_indices);
    sort_vectors_by_index(build_indices, stream_indices);
    for (size_t j = 0; j < build_indices.size(); j++) {
      ASSERT_EQ(payload.build_indices[j], build_indices[j]);
      ASSERT_EQ(payload.stream_indices[j], stream_indices[j]);
    }
    array_index_offset += array->length();
  }
  GEOSSTRtree_destroy_r(handle.handle, tree);
}

TEST(JoinerTest, PIPContainsParquet) {
  using namespace TestUtils;
  auto fs = std::make_shared<arrow::fs::LocalFileSystem>();

  std::vector<std::string> polys{
      GetTestDataPath("cities/natural-earth_cities_geo.parquet"),
      GetTestDataPath("countries/natural-earth_countries_geo.parquet")};
  std::vector<std::string> points{GetTestDataPath("cities/generated_points.parquet"),
                                  GetTestDataPath("countries/generated_points.parquet")};

  for (int i = 0; i < polys.size(); i++) {
    auto poly_path = TestUtils::GetTestDataPath(polys[i]);
    auto point_path = TestUtils::GetCanonicalPath(points[i]);
    TestJoiner(poly_path, point_path, Predicate::kContains, 10);
  }
}

TEST(JoinerTest, PIPWithinParquet) {
  using namespace TestUtils;
  auto fs = std::make_shared<arrow::fs::LocalFileSystem>();

  std::vector<std::string> polys{
      GetTestDataPath("cities/natural-earth_cities_geo.parquet"),
      GetTestDataPath("countries/natural-earth_countries_geo.parquet")};
  std::vector<std::string> points{GetTestDataPath("cities/generated_points.parquet"),
                                  GetTestDataPath("countries/generated_points.parquet")};

  for (int i = 0; i < polys.size(); i++) {
    auto poly_path = TestUtils::GetTestDataPath(polys[i]);
    auto point_path = TestUtils::GetCanonicalPath(points[i]);
    TestJoiner(point_path, poly_path, Predicate::kWithin, 10);
  }
}

TEST(JoinerTest, PolyPointIntersectsParquet) {
  using namespace TestUtils;
  auto fs = std::make_shared<arrow::fs::LocalFileSystem>();

  std::vector<std::string> polys{
      GetTestDataPath("cities/natural-earth_cities_geo.parquet"),
      GetTestDataPath("countries/natural-earth_countries_geo.parquet")};
  std::vector<std::string> points{GetTestDataPath("cities/generated_points.parquet"),
                                  GetTestDataPath("countries/generated_points.parquet")};

  for (int i = 0; i < polys.size(); i++) {
    auto poly_path = TestUtils::GetTestDataPath(polys[i]);
    auto point_path = TestUtils::GetCanonicalPath(points[i]);
    TestJoiner(point_path, poly_path, Predicate::kIntersects, 10);
  }
}

TEST(JoinerTest, PolygonPolygonContains) {
  SpatialJoiner::SpatialJoinerConfig config;
  std::string ptx_root = TestUtils::GetTestShaderPath();
  config.ptx_root = ptx_root.c_str();
  SpatialJoiner spatial_joiner;

  nanoarrow::UniqueArrayStream poly1_stream, poly2_stream;

  auto poly1_path = TestUtils::GetTestDataPath("arrowipc/test_polygons1.arrows");
  auto poly2_path = TestUtils::GetTestDataPath("arrowipc/test_polygons2.arrows");

  ArrayStreamFromIpc(poly1_path, "geometry", poly1_stream.get());
  ArrayStreamFromIpc(poly2_path, "geometry", poly2_stream.get());

  nanoarrow::UniqueSchema build_schema, stream_schema;
  nanoarrow::UniqueArray build_array, stream_array;
  ArrowError error;
  ArrowErrorSet(&error, "");
  int n_row_groups = 100;
  int array_index_offset = 0;
  std::vector<uint32_t> build_indices, stream_indices;
  geoarrow::geos::ArrayReader reader;

  class GEOSCppHandle {
   public:
    GEOSContextHandle_t handle;

    GEOSCppHandle() { handle = GEOS_init_r(); }

    ~GEOSCppHandle() { GEOS_finish_r(handle); }
  };
  GEOSCppHandle handle;

  reader.InitFromEncoding(handle.handle, GEOARROW_GEOS_ENCODING_WKB);

  geoarrow::geos::GeometryVector geom_polygons1(handle.handle);
  geoarrow::geos::GeometryVector geom_polygons2(handle.handle);
  struct Payload {
    GEOSContextHandle_t handle;
    const GEOSGeometry* geom;
    int64_t build_index_offset;
    int64_t stream_index_offset;
    std::vector<int64_t> build_indices;
    std::vector<int64_t> stream_indices;
  };

  int64_t build_count = 0;
  spatial_joiner.Init(&config);
  for (int i = 0; i < n_row_groups; i++) {
    ASSERT_EQ(ArrowArrayStreamGetNext(poly1_stream.get(), build_array.get(), &error),
              NANOARROW_OK);
    ASSERT_EQ(ArrowArrayStreamGetSchema(poly1_stream.get(), build_schema.get(), &error),
              NANOARROW_OK);

    ASSERT_EQ(ArrowArrayStreamGetNext(poly2_stream.get(), stream_array.get(), &error),
              NANOARROW_OK);
    ASSERT_EQ(ArrowArrayStreamGetSchema(poly2_stream.get(), stream_schema.get(), &error),
              NANOARROW_OK);

    spatial_joiner.Clear();
    spatial_joiner.PushBuild(nullptr, build_array.get(), 0, build_array->length);
    auto context = spatial_joiner.CreateContext();

    build_indices.clear();
    stream_indices.clear();
    spatial_joiner.FinishBuilding();
    spatial_joiner.PushStream(context.get(), nullptr, stream_array.get(), 0,
                              stream_array->length, Predicate::kContains, &build_indices,
                              &stream_indices, array_index_offset);
    geom_polygons1.resize(build_array->length);
    geom_polygons2.resize(stream_array->length);

    size_t n_polygons1 = 0, n_polygons2 = 0;
    ASSERT_EQ(reader.Read(build_array.get(), 0, build_array->length,
                          geom_polygons1.mutable_data(), &n_polygons1),
              GEOARROW_GEOS_OK);
    ASSERT_EQ(reader.Read(stream_array.get(), 0, stream_array->length,
                          geom_polygons2.mutable_data(), &n_polygons2),
              GEOARROW_GEOS_OK);

    auto* tree = GEOSSTRtree_create_r(handle.handle, 10);

    for (size_t j = 0; j < n_polygons1; j++) {
      auto* geom_polygon = geom_polygons1.borrow(j);
      auto* box = GEOSEnvelope_r(handle.handle, geom_polygon);
      GEOSGeom_setUserData_r(handle.handle, (GEOSGeometry*)geom_polygon, (void*)j);
      GEOSSTRtree_insert_r(handle.handle, tree, box, (void*)geom_polygon);
      GEOSGeom_destroy_r(handle.handle, box);
    }
    ASSERT_EQ(GEOSSTRtree_build_r(handle.handle, tree), 1);

    Payload payload;
    payload.handle = handle.handle;

    payload.build_index_offset = build_count;
    payload.stream_index_offset = array_index_offset;

    for (size_t j = 0; j < n_polygons2; j++) {
      auto* geom_poly2 = geom_polygons2.borrow(j);
      GEOSGeom_setUserData_r(handle.handle, (GEOSGeometry*)geom_poly2, (void*)j);

      payload.geom = geom_poly2;

      GEOSSTRtree_query_r(
          handle.handle, tree, geom_poly2,
          [](void* item, void* data) {
            auto* polygon1 = (GEOSGeometry*)item;
            auto* payload = (Payload*)data;
            auto* polygon2 = payload->geom;

            if (GEOSContains_r(payload->handle, polygon1, polygon2) == 1) {
              auto polygon1_id =
                  (size_t)GEOSGeom_getUserData_r(payload->handle, polygon1);
              auto polygon2_id =
                  (size_t)GEOSGeom_getUserData_r(payload->handle, polygon2);
              payload->build_indices.push_back(payload->build_index_offset + polygon1_id);
              payload->stream_indices.push_back(payload->stream_index_offset +
                                                polygon2_id);
            }
          },
          (void*)&payload);
    }

    GEOSSTRtree_destroy_r(handle.handle, tree);

    ASSERT_EQ(payload.build_indices.size(), build_indices.size());

    build_count += build_array->length;
    array_index_offset += stream_array->length;
  }
}

}  // namespace gpuspatial
