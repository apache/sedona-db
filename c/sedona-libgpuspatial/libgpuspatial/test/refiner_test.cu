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
#include "gpuspatial/index/rt_spatial_index.hpp"
#include "gpuspatial/loader/device_geometries.cuh"
#include "gpuspatial/refine/rt_spatial_refiner.hpp"
#include "test_common.hpp"

#include "geoarrow_geos/geoarrow_geos.hpp"
#include "nanoarrow/nanoarrow.hpp"

#include <geoarrow/geoarrow.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <numeric>  // For std::iota

#include "gpuspatial/index/rt_spatial_index.cuh"
#include "gpuspatial/refine/rt_spatial_refiner.cuh"

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

// Helper to concatenate C-style ArrowArrays
arrow::Result<std::shared_ptr<arrow::Array>> ConcatCArrays(
    const std::vector<ArrowArray*>& c_arrays, ArrowSchema* c_schema) {
  // 1. Import the schema ONCE into a C++ DataType object.
  //    This effectively "consumes" c_schema.
  ARROW_ASSIGN_OR_RAISE(auto type, arrow::ImportType(c_schema));

  arrow::ArrayVector arrays_to_concat;
  arrays_to_concat.reserve(c_arrays.size());

  // 2. Loop through arrays using the C++ type object.
  for (ArrowArray* c_arr : c_arrays) {
    // Use the ImportArray overload that takes std::shared_ptr<DataType>.
    // This validates c_arr against 'type' without consuming 'type'.
    ARROW_ASSIGN_OR_RAISE(auto arr, arrow::ImportArray(c_arr, type));
    arrays_to_concat.push_back(arr);
  }

  return arrow::Concatenate(arrays_to_concat);
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

std::vector<std::shared_ptr<arrow::Array>> ReadParquet(const std::string& path,
                                                       int batch_size = 100) {
  using namespace TestUtils;

  auto fs = std::make_shared<arrow::fs::LocalFileSystem>();

  std::vector<std::shared_ptr<arrow::Array>> build_arrays;
  ARROW_THROW_NOT_OK(
      ReadParquetFromFile(fs.get(), path, batch_size, "geometry", build_arrays));
  return build_arrays;
}

void ReadArrowIPC(const std::string& path, std::vector<nanoarrow::UniqueArray>& arrays,
                  std::vector<nanoarrow::UniqueSchema>& schemas,
                  uint32_t limit = std::numeric_limits<uint32_t>::max()) {
  nanoarrow::UniqueArrayStream stream;
  ArrowError error;

  // Assuming this helper exists in your context or you implement it via Arrow C++
  // (It populates the C-stream from the file)
  ArrayStreamFromIpc(path, "geometry", stream.get());
  uint32_t count = 0;
  while (true) {
    // 1. Create fresh objects for this iteration
    nanoarrow::UniqueArray array;
    nanoarrow::UniqueSchema schema;

    // 2. Get the next batch
    // Note: This function expects 'array' to be empty/released.
    int code = ArrowArrayStreamGetNext(stream.get(), array.get(), &error);
    if (code != NANOARROW_OK) {
      // Handle error (log or throw)
      break;
    }

    // 3. CHECK END OF STREAM
    // If release is NULL, the stream is finished.
    if (array->release == nullptr) {
      break;
    }

    // 4. Get the schema for this specific batch
    // ArrowArrayStreamGetSchema creates a deep copy of the schema into 'schema'.
    code = ArrowArrayStreamGetSchema(stream.get(), schema.get(), &error);
    if (code != NANOARROW_OK) {
      // Handle error
      break;
    }

    // 5. Move ownership to the output vectors
    arrays.push_back(std::move(array));
    schemas.push_back(std::move(schema));
    count += array->length;
    if (count >= limit) break;
  }
}

void TestJoiner(ArrowSchema* build_schema, std::vector<ArrowArray*>& build_arrays,
                ArrowSchema* probe_schema, std::vector<ArrowArray*>& probe_arrays,
                Predicate predicate) {
  using namespace TestUtils;
  using coord_t = double;
  using fpoint_t = Point<coord_t, 2>;
  using box_t = Box<fpoint_t>;

  auto rt_engine = std::make_shared<RTEngine>();

  {
    std::string ptx_root = GetTestShaderPath();
    auto config = get_default_rt_config(ptx_root);
    rt_engine->Init(config);
  }

  RTSpatialIndexConfig idx_config;
  idx_config.rt_engine = rt_engine;
  auto rt_index = CreateRTSpatialIndex<coord_t, 2>(idx_config);
  RTSpatialRefinerConfig refiner_config;
  refiner_config.rt_engine = rt_engine;
  auto rt_refiner = CreateRTSpatialRefiner(refiner_config);

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
  size_t total_build_length = 0;

  for (auto& array : build_arrays) {
    total_build_length += array->length;
  }

  // Using GEOS for reference
  geom_build.resize(total_build_length);
  size_t tail_build = 0;
  auto* tree = GEOSSTRtree_create_r(handle.handle, 10);
  for (auto& array : build_arrays) {
    // geos for reference
    size_t n_build;

    ASSERT_EQ(reader.Read((ArrowArray*)array, 0, array->length,
                          geom_build.mutable_data() + tail_build, &n_build),
              GEOARROW_GEOS_OK);
    ASSERT_EQ(array->length, n_build);
    std::vector<box_t> rects;

    for (size_t offset = tail_build; offset < tail_build + n_build; offset++) {
      auto* geom = geom_build.borrow(offset);
      auto* box = GEOSEnvelope_r(handle.handle, geom);

      double xmin, ymin, xmax, ymax;
      if (GEOSGeom_getExtent_r(handle.handle, box, &xmin, &ymin, &xmax, &ymax) == 0) {
        printf("Error getting extent\n");
        xmin = 0;
        ymin = 0;
        xmax = -1;
        ymax = -1;
      }

      box_t bbox(fpoint_t((float)xmin, (float)ymin), fpoint_t((float)xmax, (float)ymax));

      rects.push_back(bbox);

      GEOSGeom_setUserData_r(handle.handle, (GEOSGeometry*)geom, (void*)offset);
      GEOSSTRtree_insert_r(handle.handle, tree, box, (void*)geom);
      GEOSGeom_destroy_r(handle.handle, box);
    }
    rt_index->PushBuild(rects.data(), rects.size());
    tail_build += n_build;
  }
  rt_index->FinishBuilding();
  ASSERT_EQ(GEOSSTRtree_build_r(handle.handle, tree), 1);

  auto build_array_ptr = ConcatCArrays(build_arrays, build_schema).ValueOrDie();

  nanoarrow::UniqueArray uniq_build_array;
  nanoarrow::UniqueSchema uniq_build_schema;
  ARROW_THROW_NOT_OK(arrow::ExportArray(*build_array_ptr, uniq_build_array.get(),
                                        uniq_build_schema.get()));
  // Start stream processing

  for (auto& array : probe_arrays) {
    geoarrow::geos::GeometryVector geom_stream(handle.handle);
    size_t n_stream;
    geom_stream.resize(array->length);

    ASSERT_EQ(reader.Read(array, 0, array->length, geom_stream.mutable_data(), &n_stream),
              GEOARROW_GEOS_OK);

    std::vector<box_t> queries;

    for (size_t i = 0; i < array->length; i++) {
      auto* geom = geom_stream.borrow(i);
      double xmin, ymin, xmax, ymax;
      int result = GEOSGeom_getExtent_r(handle.handle, geom, &xmin, &ymin, &xmax, &ymax);
      ASSERT_EQ(result, 1);
      box_t bbox(fpoint_t((float)xmin, (float)ymin), fpoint_t((float)xmax, (float)ymax));
      queries.push_back(bbox);
    }

    std::vector<uint32_t> build_indices, stream_indices;

    rt_index->Probe(queries.data(), queries.size(), &build_indices, &stream_indices);
    auto old_size = build_indices.size();

    auto new_size = rt_refiner->Refine(
        uniq_build_schema.get(), uniq_build_array.get(), probe_schema, array, predicate,
        build_indices.data(), stream_indices.data(), build_indices.size());

    build_indices.resize(new_size);
    stream_indices.resize(new_size);

    struct Payload {
      GEOSContextHandle_t handle;
      const GEOSGeometry* geom;
      std::vector<uint32_t> build_indices;
      std::vector<uint32_t> stream_indices;
      Predicate predicate;
    };

    Payload payload;
    payload.predicate = predicate;
    payload.handle = handle.handle;

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
              payload->stream_indices.push_back(stream_id);
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
  }
  GEOSSTRtree_destroy_r(handle.handle, tree);
}

void TestJoinerLoaded(ArrowSchema* build_schema, std::vector<ArrowArray*>& build_arrays,
                      ArrowSchema* probe_schema, std::vector<ArrowArray*>& probe_arrays,
                      Predicate predicate, bool pipelined = false) {
  using namespace TestUtils;
  using coord_t = double;
  using fpoint_t = Point<coord_t, 2>;
  using box_t = Box<fpoint_t>;

  auto rt_engine = std::make_shared<RTEngine>();

  {
    std::string ptx_root = TestUtils::GetTestShaderPath();
    auto config = get_default_rt_config(ptx_root);
    rt_engine->Init(config);
  }

  RTSpatialIndexConfig idx_config;
  idx_config.rt_engine = rt_engine;
  auto rt_index = CreateRTSpatialIndex<coord_t, 2>(idx_config);

  RTSpatialRefinerConfig refiner_config;
  refiner_config.rt_engine = rt_engine;
  if (pipelined) {
    refiner_config.pipeline_batches = 10;
  }
  auto rt_refiner = CreateRTSpatialRefiner(refiner_config);
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
  size_t total_build_length = 0;

  for (auto& array : build_arrays) {
    total_build_length += array->length;
  }

  // Using GEOS for reference
  geom_build.resize(total_build_length);
  size_t tail_build = 0;
  auto* tree = GEOSSTRtree_create_r(handle.handle, 10);
  for (auto& array : build_arrays) {
    // geos for reference
    size_t n_build;

    ASSERT_EQ(reader.Read((ArrowArray*)array, 0, array->length,
                          geom_build.mutable_data() + tail_build, &n_build),
              GEOARROW_GEOS_OK);
    ASSERT_EQ(array->length, n_build);
    std::vector<box_t> rects;

    for (size_t offset = tail_build; offset < tail_build + n_build; offset++) {
      auto* geom = geom_build.borrow(offset);
      auto* box = GEOSEnvelope_r(handle.handle, geom);

      double xmin, ymin, xmax, ymax;
      if (GEOSGeom_getExtent_r(handle.handle, box, &xmin, &ymin, &xmax, &ymax) == 0) {
        xmin = 0;
        ymin = 0;
        xmax = -1;
        ymax = -1;
      }

      box_t bbox(fpoint_t((float)xmin, (float)ymin), fpoint_t((float)xmax, (float)ymax));

      rects.push_back(bbox);

      GEOSGeom_setUserData_r(handle.handle, (GEOSGeometry*)geom, (void*)offset);
      GEOSSTRtree_insert_r(handle.handle, tree, box, (void*)geom);
      GEOSGeom_destroy_r(handle.handle, box);
    }
    rt_index->PushBuild(rects.data(), rects.size());
    tail_build += n_build;
  }
  rt_index->FinishBuilding();
  ASSERT_EQ(GEOSSTRtree_build_r(handle.handle, tree), 1);

  auto build_array_ptr = ConcatCArrays(build_arrays, build_schema).ValueOrDie();

  nanoarrow::UniqueArray uniq_build_array;
  nanoarrow::UniqueSchema uniq_build_schema;
  ARROW_THROW_NOT_OK(arrow::ExportArray(*build_array_ptr, uniq_build_array.get(),
                                        uniq_build_schema.get()));
  // Start stream processing

  rt_refiner->PushBuild(uniq_build_schema.get(), uniq_build_array.get());
  rt_refiner->FinishBuilding();

  for (auto& array : probe_arrays) {
    geoarrow::geos::GeometryVector geom_stream(handle.handle);
    size_t n_stream;
    geom_stream.resize(array->length);

    ASSERT_EQ(reader.Read(array, 0, array->length, geom_stream.mutable_data(), &n_stream),
              GEOARROW_GEOS_OK);

    std::vector<box_t> queries;

    for (size_t i = 0; i < array->length; i++) {
      auto* geom = geom_stream.borrow(i);
      double xmin, ymin, xmax, ymax;
      int result = GEOSGeom_getExtent_r(handle.handle, geom, &xmin, &ymin, &xmax, &ymax);
      ASSERT_EQ(result, 1);
      box_t bbox(fpoint_t((float)xmin, (float)ymin), fpoint_t((float)xmax, (float)ymax));
      queries.push_back(bbox);
    }

    std::vector<uint32_t> build_indices, stream_indices;

    rt_index->Probe(queries.data(), queries.size(), &build_indices, &stream_indices);
    auto old_size = build_indices.size();

    auto new_size =
        rt_refiner->Refine(probe_schema, array, predicate, build_indices.data(),
                           stream_indices.data(), build_indices.size());

    printf("Old size %u, new size %u\n", (unsigned)old_size, (unsigned)new_size);
    build_indices.resize(new_size);
    stream_indices.resize(new_size);

    struct Payload {
      GEOSContextHandle_t handle;
      const GEOSGeometry* geom;
      std::vector<uint32_t> build_indices;
      std::vector<uint32_t> stream_indices;
      Predicate predicate;
    };

    Payload payload;
    payload.predicate = predicate;
    payload.handle = handle.handle;

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
              payload->stream_indices.push_back(stream_id);
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
    auto poly_arrays = ReadParquet(poly_path, 1000);
    auto point_arrays = ReadParquet(point_path, 1000);
    std::vector<nanoarrow::UniqueArray> poly_uniq_arrays, point_uniq_arrays;
    std::vector<nanoarrow::UniqueSchema> poly_uniq_schema, point_uniq_schema;

    for (auto& arr : poly_arrays) {
      ARROW_THROW_NOT_OK(arrow::ExportArray(*arr, poly_uniq_arrays.emplace_back().get(),
                                            poly_uniq_schema.emplace_back().get()));
    }
    for (auto& arr : point_arrays) {
      ARROW_THROW_NOT_OK(arrow::ExportArray(*arr, point_uniq_arrays.emplace_back().get(),
                                            point_uniq_schema.emplace_back().get()));
    }

    std::vector<ArrowArray*> poly_c_arrays, point_c_arrays;
    for (auto& arr : poly_uniq_arrays) {
      poly_c_arrays.push_back(arr.get());
    }
    for (auto& arr : point_uniq_arrays) {
      point_c_arrays.push_back(arr.get());
    }
    TestJoinerLoaded(poly_uniq_schema[0].get(), poly_c_arrays, point_uniq_schema[0].get(),
                     point_c_arrays, Predicate::kContains);
  }
}

TEST(JoinerTest, PIPContainsParquetLoaded) {
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
    auto poly_arrays = ReadParquet(poly_path, 1000);
    auto point_arrays = ReadParquet(point_path, 1000);
    std::vector<nanoarrow::UniqueArray> poly_uniq_arrays, point_uniq_arrays;
    std::vector<nanoarrow::UniqueSchema> poly_uniq_schema, point_uniq_schema;

    for (auto& arr : poly_arrays) {
      ARROW_THROW_NOT_OK(arrow::ExportArray(*arr, poly_uniq_arrays.emplace_back().get(),
                                            poly_uniq_schema.emplace_back().get()));
    }
    for (auto& arr : point_arrays) {
      ARROW_THROW_NOT_OK(arrow::ExportArray(*arr, point_uniq_arrays.emplace_back().get(),
                                            point_uniq_schema.emplace_back().get()));
    }

    std::vector<ArrowArray*> poly_c_arrays, point_c_arrays;
    for (auto& arr : poly_uniq_arrays) {
      poly_c_arrays.push_back(arr.get());
    }
    for (auto& arr : point_uniq_arrays) {
      point_c_arrays.push_back(arr.get());
    }
    TestJoinerLoaded(poly_uniq_schema[0].get(), poly_c_arrays, point_uniq_schema[0].get(),
                     point_c_arrays, Predicate::kContains);
  }
}

TEST(JoinerTest, PIPContainsParquetPipelined) {
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
    auto poly_arrays = ReadParquet(poly_path, 1000);
    auto point_arrays = ReadParquet(point_path, 1000);
    std::vector<nanoarrow::UniqueArray> poly_uniq_arrays, point_uniq_arrays;
    std::vector<nanoarrow::UniqueSchema> poly_uniq_schema, point_uniq_schema;

    for (auto& arr : poly_arrays) {
      ARROW_THROW_NOT_OK(arrow::ExportArray(*arr, poly_uniq_arrays.emplace_back().get(),
                                            poly_uniq_schema.emplace_back().get()));
    }
    for (auto& arr : point_arrays) {
      ARROW_THROW_NOT_OK(arrow::ExportArray(*arr, point_uniq_arrays.emplace_back().get(),
                                            point_uniq_schema.emplace_back().get()));
    }

    std::vector<ArrowArray*> poly_c_arrays, point_c_arrays;
    for (auto& arr : poly_uniq_arrays) {
      poly_c_arrays.push_back(arr.get());
    }
    for (auto& arr : point_uniq_arrays) {
      point_c_arrays.push_back(arr.get());
    }
    TestJoinerLoaded(poly_uniq_schema[0].get(), poly_c_arrays, point_uniq_schema[0].get(),
                     point_c_arrays, Predicate::kContains, true);
  }
}

TEST(JoinerTest, PIPContainsArrowIPC) {
  using namespace TestUtils;
  auto fs = std::make_shared<arrow::fs::LocalFileSystem>();

  std::vector<std::string> polys{GetTestDataPath("arrowipc/test_polygons.arrows")};
  std::vector<std::string> points{GetTestDataPath("arrowipc/test_points.arrows")};

  for (int i = 0; i < polys.size(); i++) {
    auto poly_path = TestUtils::GetTestDataPath(polys[i]);
    auto point_path = TestUtils::GetCanonicalPath(points[i]);
    std::vector<nanoarrow::UniqueArray> poly_uniq_arrays, point_uniq_arrays;
    std::vector<nanoarrow::UniqueSchema> poly_uniq_schema, point_uniq_schema;

    ReadArrowIPC(poly_path, poly_uniq_arrays, poly_uniq_schema);
    ReadArrowIPC(point_path, point_uniq_arrays, point_uniq_schema);

    std::vector<ArrowArray*> poly_c_arrays, point_c_arrays;
    for (auto& arr : poly_uniq_arrays) {
      poly_c_arrays.push_back(arr.get());
    }
    for (auto& arr : point_uniq_arrays) {
      point_c_arrays.push_back(arr.get());
    }

    TestJoiner(poly_uniq_schema[0].get(), poly_c_arrays, point_uniq_schema[0].get(),
               point_c_arrays, Predicate::kContains);
  }
}

TEST(JoinerTest, PIPWithinArrowIPC) {
  using namespace TestUtils;
  auto fs = std::make_shared<arrow::fs::LocalFileSystem>();

  std::vector<std::string> polys{GetTestDataPath("arrowipc/test_polygons.arrows")};
  std::vector<std::string> points{GetTestDataPath("arrowipc/test_points.arrows")};

  for (int i = 0; i < polys.size(); i++) {
    auto poly_path = TestUtils::GetTestDataPath(polys[i]);
    auto point_path = TestUtils::GetCanonicalPath(points[i]);
    std::vector<nanoarrow::UniqueArray> poly_uniq_arrays, point_uniq_arrays;
    std::vector<nanoarrow::UniqueSchema> poly_uniq_schema, point_uniq_schema;

    ReadArrowIPC(poly_path, poly_uniq_arrays, poly_uniq_schema);
    ReadArrowIPC(point_path, point_uniq_arrays, point_uniq_schema);

    std::vector<ArrowArray*> poly_c_arrays, point_c_arrays;
    for (auto& arr : poly_uniq_arrays) {
      poly_c_arrays.push_back(arr.get());
    }
    for (auto& arr : point_uniq_arrays) {
      point_c_arrays.push_back(arr.get());
    }

    TestJoiner(point_uniq_schema[0].get(), point_c_arrays, poly_uniq_schema[0].get(),
               poly_c_arrays, Predicate::kWithin);
  }
}

TEST(JoinerTest, PolygonPolygonContains) {
  using namespace TestUtils;
  auto fs = std::make_shared<arrow::fs::LocalFileSystem>();

  std::vector<std::string> polys1{GetTestDataPath("arrowipc/test_polygons1.arrows")};
  std::vector<std::string> polys2{GetTestDataPath("arrowipc/test_polygons2.arrows")};

  for (int i = 0; i < polys1.size(); i++) {
    auto poly1_path = TestUtils::GetTestDataPath(polys1[i]);
    auto poly2_path = TestUtils::GetCanonicalPath(polys2[i]);
    std::vector<nanoarrow::UniqueArray> poly1_uniq_arrays, poly2_uniq_arrays;
    std::vector<nanoarrow::UniqueSchema> poly1_uniq_schema, poly2_uniq_schema;

    ReadArrowIPC(poly1_path, poly1_uniq_arrays, poly1_uniq_schema, 100);
    ReadArrowIPC(poly2_path, poly2_uniq_arrays, poly2_uniq_schema, 100);

    std::vector<ArrowArray*> poly1_c_arrays, poly2_c_arrays;
    for (auto& arr : poly1_uniq_arrays) {
      poly1_c_arrays.push_back(arr.get());
    }
    for (auto& arr : poly2_uniq_arrays) {
      poly2_c_arrays.push_back(arr.get());
    }

    TestJoiner(poly1_uniq_schema[0].get(), poly1_c_arrays, poly2_uniq_schema[0].get(),
               poly2_c_arrays, Predicate::kIntersects);
  }
}
}  // namespace gpuspatial
