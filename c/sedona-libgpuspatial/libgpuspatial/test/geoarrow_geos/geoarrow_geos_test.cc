
#include <gtest/gtest.h>

#include "nanoarrow/nanoarrow.hpp"

#include "geoarrow_geos.hpp"

class GEOSCppHandle {
 public:
  GEOSContextHandle_t handle;

  GEOSCppHandle() { handle = GEOS_init_r(); }

  ~GEOSCppHandle() { GEOS_finish_r(handle); }
};

class GEOSCppWKTReader {
 public:
  GEOSWKTReader* ptr;
  GEOSContextHandle_t handle;

  GEOSCppWKTReader(GEOSContextHandle_t handle) : handle(handle), ptr(nullptr) {
    ptr = GEOSWKTReader_create_r(handle);
  }

  GeoArrowGEOSErrorCode Read(const std::string& wkt, GEOSGeometry** out) {
    if (wkt == "") {
      *out = nullptr;
      return NANOARROW_OK;
    }

    GEOSGeometry* result = GEOSWKTReader_read_r(handle, ptr, wkt.c_str());
    if (result == nullptr) {
      return EINVAL;
    }

    *out = result;
    return GEOARROW_GEOS_OK;
  }

  ~GEOSCppWKTReader() {
    if (ptr != NULL) {
      GEOSWKTReader_destroy_r(handle, ptr);
    }
  }
};

TEST(GeoArrowGEOSTest, TestVersions) {
  ASSERT_EQ(std::string(GeoArrowGEOSVersionGEOS()).substr(0, 1), "3");
  ASSERT_STREQ(GeoArrowGEOSVersionGeoArrow(), "0.2.0-SNAPSHOT");
}

void TestBuilderRoundtripWKT(const std::string& wkt) {
  GEOSCppHandle handle;
  GEOSCppWKTReader reader(handle.handle);
  geoarrow::geos::GeometryVector geom(handle.handle);
  geoarrow::geos::ArrayBuilder builder;

  ASSERT_EQ(builder.InitFromEncoding(handle.handle, GEOARROW_GEOS_ENCODING_WKT),
            GEOARROW_GEOS_OK);

  geom.resize(1);
  ASSERT_EQ(reader.Read(wkt, geom.mutable_data()), GEOARROW_GEOS_OK);
  size_t n = 0;
  ASSERT_EQ(builder.Append(geom.data(), 1, &n), GEOARROW_GEOS_OK)
      << "WKT: " << wkt << "\n Error: " << builder.GetLastError();
  ASSERT_EQ(n, 1);

  nanoarrow::UniqueArray array;
  ASSERT_EQ(builder.Finish(array.get()), GEOARROW_GEOS_OK);

  ASSERT_EQ(array->length, 1);
  ASSERT_EQ(array->n_buffers, 3);

  const auto offsets = reinterpret_cast<const int32_t*>(array->buffers[1]);
  const auto data = reinterpret_cast<const char*>(array->buffers[2]);

  std::string wkt_out(data + offsets[0], offsets[1] - offsets[0]);
  EXPECT_EQ(wkt_out, wkt);

  if (wkt_out == "") {
    ASSERT_NE(array->buffers[0], nullptr);
    const auto validity = reinterpret_cast<const uint8_t*>(array->buffers[0]);
    EXPECT_EQ(validity[0] & (1 << 0), 0);
  }
}

TEST(GeoArrowGEOSTest, TestArrayBuilderRoundtripWKTNull) { TestBuilderRoundtripWKT(""); }

TEST(GeoArrowGEOSTest, TestArrayBuilderRoundtripWKTPoint) {
  TestBuilderRoundtripWKT("POINT EMPTY");
  TestBuilderRoundtripWKT("POINT (0 1)");
  TestBuilderRoundtripWKT("POINT Z EMPTY");
  TestBuilderRoundtripWKT("POINT Z (0 1 2)");
}

TEST(GeoArrowGEOSTest, TestArrayBuilderRoundtripWKTLinestring) {
  TestBuilderRoundtripWKT("LINESTRING EMPTY");
  TestBuilderRoundtripWKT("LINESTRING (0 1, 2 3)");
  TestBuilderRoundtripWKT("LINESTRING Z EMPTY");
  TestBuilderRoundtripWKT("LINESTRING Z (0 1 2, 3 4 5)");
}

TEST(GeoArrowGEOSTest, TestArrayBuilderRoundtripWKTPolygon) {
  TestBuilderRoundtripWKT("POLYGON EMPTY");
  TestBuilderRoundtripWKT("POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))");
  TestBuilderRoundtripWKT(
      "POLYGON ((35 10, 45 45, 15 40, 10 20, 35 10), (20 30, 35 35, 30 20, 20 30))");
}

TEST(GeoArrowGEOSTest, TestArrayBuilderRoundtripWKTCollection) {
  TestBuilderRoundtripWKT("MULTIPOINT EMPTY");
  TestBuilderRoundtripWKT("MULTIPOINT (30 10)");
  TestBuilderRoundtripWKT("MULTIPOINT (30 10, 40 30, 20 20)");
}

void TestReaderRoundtripWKTVec(
    const std::vector<std::string>& wkt, int wkb_type,
    GeoArrowGEOSEncoding encoding = GEOARROW_GEOS_ENCODING_GEOARROW) {
  GEOSCppHandle handle;
  geoarrow::geos::ArrayBuilder builder;
  geoarrow::geos::ArrayReader reader;

  // Initialize builder + build a target array
  ASSERT_EQ(builder.InitFromEncoding(handle.handle, encoding, wkb_type),
            GEOARROW_GEOS_OK);

  GEOSCppWKTReader wkt_reader(handle.handle);

  geoarrow::geos::GeometryVector geoms_in(handle.handle);
  geoms_in.resize(wkt.size());
  geoarrow::geos::GeometryVector geoms_out(handle.handle);
  geoms_out.resize(wkt.size());

  for (size_t i = 0; i < wkt.size(); i++) {
    ASSERT_EQ(wkt_reader.Read(wkt[i], geoms_in.mutable_data() + i), GEOARROW_GEOS_OK)
        << "Failed to append " << wkt[i];
  }

  size_t n = 0;
  ASSERT_EQ(builder.Append(geoms_in.data(), wkt.size(), &n), GEOARROW_GEOS_OK);
  ASSERT_EQ(n, wkt.size());

  nanoarrow::UniqueArray array;
  ASSERT_EQ(builder.Finish(array.get()), GEOARROW_GEOS_OK);

  // Read it back!
  ASSERT_EQ(reader.InitFromEncoding(handle.handle, encoding, wkb_type), GEOARROW_GEOS_OK);

  size_t n_out = 0;
  ASSERT_EQ(reader.Read(array.get(), 0, array->length, geoms_out.mutable_data(), &n_out),
            GEOARROW_GEOS_OK)
      << "WKT[0]: " << wkt[0] << " n = " << n << "\n Error: " << reader.GetLastError();
  ASSERT_EQ(n_out, n);

  // Check for GEOS equality
  for (size_t i = 0; i < n; i++) {
    if (geoms_out.borrow(i) == nullptr || geoms_in.borrow(i) == nullptr) {
      EXPECT_EQ(geoms_out.borrow(i), geoms_in.borrow(i));
    } else {
      EXPECT_EQ(
          GEOSEqualsExact_r(handle.handle, geoms_out.borrow(i), geoms_in.borrow(i), 0), 1)
          << "WKT: " << wkt[i] << " at index " << i;
    }
  }
}

void TestReaderRoundtripWKT(
    const std::string& wkt, int wkb_type,
    GeoArrowGEOSEncoding encoding = GEOARROW_GEOS_ENCODING_GEOARROW) {
  TestReaderRoundtripWKTVec({wkt}, wkb_type, encoding);
}

class EncodingTestFixture : public ::testing::TestWithParam<GeoArrowGEOSEncoding> {
 protected:
  GeoArrowGEOSEncoding encoding;
};

TEST_P(EncodingTestFixture, TestArrayReaderPoint) {
  GeoArrowGEOSEncoding encoding = GetParam();

  TestReaderRoundtripWKT("", 1, encoding);
  TestReaderRoundtripWKT("POINT EMPTY", 1, encoding);
  TestReaderRoundtripWKT("POINT (0 1)", 1, encoding);
  TestReaderRoundtripWKT("POINT Z EMPTY", 1001, encoding);
  TestReaderRoundtripWKT("POINT Z (0 1 2)", 1001, encoding);

  TestReaderRoundtripWKTVec({}, 1, encoding);
  TestReaderRoundtripWKTVec({}, 1001, encoding);
  TestReaderRoundtripWKTVec(
      {"POINT EMPTY", "POINT (0 1)", "POINT (2 3)", "POINT EMPTY", ""}, 1, encoding);
  TestReaderRoundtripWKTVec(
      {"POINT Z EMPTY", "POINT Z (0 1 2)", "POINT Z (3 4 5)", "POINT Z EMPTY", ""}, 1001,
      encoding);
}

TEST_P(EncodingTestFixture, TestArrayReaderLinestring) {
  GeoArrowGEOSEncoding encoding = GetParam();

  TestReaderRoundtripWKT("", 2, encoding);
  TestReaderRoundtripWKT("LINESTRING EMPTY", 2, encoding);
  TestReaderRoundtripWKT("LINESTRING (0 1, 2 3)", 2, encoding);
  TestReaderRoundtripWKT("LINESTRING Z EMPTY", 2, encoding);
  TestReaderRoundtripWKT("LINESTRING Z (0 1 2, 3 4 5)", 1002, encoding);

  TestReaderRoundtripWKTVec({}, 2, encoding);
  TestReaderRoundtripWKTVec({}, 1002, encoding);
  TestReaderRoundtripWKTVec({"LINESTRING EMPTY", "LINESTRING (0 1, 2 3)",
                             "LINESTRING (4 5, 6 7, 8 9)", "LINESTRING EMPTY", ""},
                            2, encoding);
  TestReaderRoundtripWKTVec(
      {"LINESTRING Z EMPTY", "LINESTRING Z (0 1 2, 3 4 5)",
       "LINESTRING Z (6 7 8, 9 10 11, 12 13 14)", "LINESTRING Z EMPTY", ""},
      1002, encoding);
}

TEST_P(EncodingTestFixture, TestArrayReaderPolygon) {
  GeoArrowGEOSEncoding encoding = GetParam();

  TestReaderRoundtripWKT("", 3, encoding);
  TestReaderRoundtripWKT("POLYGON EMPTY", 3, encoding);
  TestReaderRoundtripWKT("POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))", 3);
  TestReaderRoundtripWKT(
      "POLYGON ((35 10, 45 45, 15 40, 10 20, 35 10), (20 30, 35 35, 30 20, 20 30))", 3);
  TestReaderRoundtripWKT("POLYGON Z EMPTY", 1003, encoding);
  TestReaderRoundtripWKT("POLYGON Z ((30 10 40, 40 40 80, 20 40 60, 10 20 30, 30 10 40))",
                         1003, encoding);
  TestReaderRoundtripWKT(
      "POLYGON Z ((35 10 45, 45 45 90, 15 40 55, 10 20 30, 35 10 45), (20 30 50, 35 35 "
      "70, 30 20 50, 20 30 50))",
      1003, encoding);
  TestReaderRoundtripWKT(
      "POLYGON Z ((35 10 45, 45 45 90, 15 40 55, 10 20 30, 35 10 45), (20 30 50, 35 35 "
      "70, 30 20 50, 20 30 50))",
      1003, encoding);

  TestReaderRoundtripWKTVec({}, 3, encoding);
  TestReaderRoundtripWKTVec({}, 1003, encoding);
  TestReaderRoundtripWKTVec(
      {"POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))",
       "POLYGON ((35 10, 45 45, 15 40, 10 20, 35 10), (20 30, 35 35, 30 20, 20 30))",
       "POLYGON EMPTY", ""},
      3, encoding);

  TestReaderRoundtripWKTVec(
      {"POLYGON Z ((30 10 40, 40 40 80, 20 40 60, 10 20 30, 30 10 40))",
       "POLYGON Z ((35 10 45, 45 45 90, 15 40 55, 10 20 30, 35 10 45), (20 30 50, 35 35 "
       "70, 30 20 50, 20 30 50))",
       "POLYGON Z EMPTY", ""},
      1003, encoding);
}

TEST_P(EncodingTestFixture, TestArrayReaderMultipoint) {
  GeoArrowGEOSEncoding encoding = GetParam();

  TestReaderRoundtripWKT("", 4, encoding);
  TestReaderRoundtripWKT("MULTIPOINT EMPTY", 4, encoding);
  TestReaderRoundtripWKT("MULTIPOINT (10 40, 40 30, 20 20, 30 10)", 4, encoding);
  TestReaderRoundtripWKT("MULTIPOINT (30 10)", 4, encoding);

  TestReaderRoundtripWKTVec({}, 4, encoding);
  TestReaderRoundtripWKTVec({}, 1004, encoding);
  TestReaderRoundtripWKTVec(
      {"MULTIPOINT ((30 10))", "MULTIPOINT ((10 40), (40 30), (20 20), (30 10))",
       "MULTIPOINT ((10 40), (40 30), (20 20), (30 10))", ""},
      4, encoding);

  TestReaderRoundtripWKTVec(
      {"MULTIPOINT Z ((30 10 40))",
       "MULTIPOINT Z ((10 40 50), (40 30 70), (20 20 40), (30 10 40))",
       "MULTIPOINT Z ((10 40 50), (40 30 70), (20 20 40), (30 10 40))",
       "MULTIPOINT Z EMPTY", ""},
      1004, encoding);
}

TEST_P(EncodingTestFixture, TestArrayReaderMultilinestring) {
  GeoArrowGEOSEncoding encoding = GetParam();

  TestReaderRoundtripWKT("", 5, encoding);
  TestReaderRoundtripWKT("MULTILINESTRING EMPTY", 5, encoding);
  TestReaderRoundtripWKT("MULTILINESTRING ((30 10, 10 30, 40 40))", 5, encoding);
  TestReaderRoundtripWKT(
      "MULTILINESTRING ((10 10, 20 20, 10 40), (40 40, 30 30, 40 20, 30 10))", 5,
      encoding);

  TestReaderRoundtripWKTVec(
      {"MULTILINESTRING ((30 10, 10 30, 40 40))",
       "MULTILINESTRING ((10 10, 20 20, 10 40), (40 40, 30 30, 40 20, 30 10))",
       "MULTILINESTRING EMPTY", ""},
      5, encoding);

  TestReaderRoundtripWKTVec({}, 5, encoding);
  TestReaderRoundtripWKTVec({}, 1005, encoding);
  TestReaderRoundtripWKTVec({"MULTILINESTRING Z ((30 10 40, 10 30 40, 40 40 80))",
                             "MULTILINESTRING Z ((10 10 20, 20 20 40, 10 40 50), (40 40 "
                             "80, 30 30 60, 40 20 60, 30 10 40))",
                             "MULTILINESTRING Z EMPTY", ""},
                            1005, encoding);
}

TEST_P(EncodingTestFixture, TestArrayReaderMultipolygon) {
  GeoArrowGEOSEncoding encoding = GetParam();

  TestReaderRoundtripWKT("", 6, encoding);
  TestReaderRoundtripWKT("MULTIPOLYGON EMPTY", 6, encoding);
  TestReaderRoundtripWKT(
      "MULTIPOLYGON (((30 20, 45 40, 10 40, 30 20)), ((15 5, 40 10, 10 20, 5 10, 15 5)))",
      6, encoding);
  TestReaderRoundtripWKT(
      "MULTIPOLYGON (((40 40, 20 45, 45 30, 40 40)), ((20 35, 10 30, 10 10, 30 5, 45 20, "
      "20 35), (30 20, 20 15, 20 25, 30 20)))",
      6, encoding);

  TestReaderRoundtripWKTVec({}, 6, encoding);
  TestReaderRoundtripWKTVec({}, 1006, encoding);
  TestReaderRoundtripWKTVec(
      {"MULTIPOLYGON (((30 10, 40 40, 20 40, 10 20, 30 10)))",
       "MULTIPOLYGON (((30 20, 45 40, 10 40, 30 20)), ((15 5, 40 10, 10 20, 5 10, 15 "
       "5)))",
       "MULTIPOLYGON (((40 40, 20 45, 45 30, 40 40)), ((20 35, 10 30, 10 10, 30 5, 45 "
       "20, 20 35), (30 20, 20 15, 20 25, 30 20)))",
       "MULTIPOLYGON EMPTY", ""},
      6, encoding);

  TestReaderRoundtripWKTVec(
      {"MULTIPOLYGON Z (((30 10 40, 40 40 80, 20 40 60, 10 20 30, 30 10 40)))",
       "MULTIPOLYGON Z (((30 20 50, 45 40 85, 10 40 50, 30 20 50)), ((15 5 20, 40 10 50, "
       "10 20 30, 5 10 15, 15 5 20)))",
       "MULTIPOLYGON Z (((40 40 80, 20 45 65, 45 30 75, 40 40 80)), ((20 35 55, 10 30 "
       "40, 10 10 20, 30 5 35, 45 20 65, 20 35 55), (30 20 50, 20 15 35, 20 25 45, 30 20 "
       "50)))",
       "MULTIPOLYGON Z EMPTY", ""},
      1006, encoding);
}

INSTANTIATE_TEST_SUITE_P(GeoArrowGEOSTest, EncodingTestFixture,
                         ::testing::Values(GEOARROW_GEOS_ENCODING_GEOARROW,
                                           GEOARROW_GEOS_ENCODING_GEOARROW_INTERLEAVED,
                                           GEOARROW_GEOS_ENCODING_WKB,
                                           GEOARROW_GEOS_ENCODING_WKT));

TEST(GeoArrowGEOSTest, TestHppGeometryVector) {
  GEOSCppHandle handle;
  geoarrow::geos::GeometryVector geom(handle.handle);

  geom.reserve(3);
  geom.resize(3);
  ASSERT_EQ(geom.size(), 3);
  ASSERT_EQ(geom.borrow(0), nullptr);
  ASSERT_EQ(geom.borrow(1), nullptr);
  ASSERT_EQ(geom.borrow(2), nullptr);

  geom.set(0, GEOSGeom_createEmptyPolygon_r(handle.handle));
  geom.set(1, GEOSGeom_createEmptyLineString_r(handle.handle));
  geom.set(2, GEOSGeom_createEmptyPoint_r(handle.handle));

  geom.resize(2);
  geom.resize(3);
  ASSERT_NE(geom.borrow(0), nullptr);
  ASSERT_NE(geom.borrow(1), nullptr);
  ASSERT_EQ(geom.borrow(2), nullptr);

  GEOSGeometry* geom1 = geom.take_ownership_of(1);
  ASSERT_NE(geom1, nullptr);
  GEOSGeom_destroy_r(handle.handle, geom1);
  ASSERT_EQ(geom.borrow(1), nullptr);

  geoarrow::geos::GeometryVector other = std::move(geom);
  ASSERT_EQ(geom.size(), 0);
  ASSERT_EQ(other.size(), 3);
  ASSERT_NE(other.borrow(0), nullptr);
  ASSERT_EQ(other.borrow(1), nullptr);
  ASSERT_EQ(other.borrow(2), nullptr);
}

TEST(GeoArrowGEOSTest, TestHppArrayBuilder) {
  GEOSCppHandle handle;
  geoarrow::geos::ArrayBuilder builder;
  EXPECT_STREQ(builder.GetLastError(), "");

  ASSERT_EQ(builder.InitFromEncoding(handle.handle, GEOARROW_GEOS_ENCODING_UNKNOWN),
            EINVAL);
  ASSERT_EQ(builder.InitFromEncoding(handle.handle, GEOARROW_GEOS_ENCODING_WKT),
            GEOARROW_GEOS_OK);
  EXPECT_STREQ(builder.GetLastError(), "");

  geoarrow::geos::ArrayBuilder builder2 = std::move(builder);
  nanoarrow::UniqueArray array;
  builder2.Finish(array.get());
  ASSERT_EQ(array->length, 0);
  ASSERT_EQ(array->n_buffers, 3);
}

TEST(GeoArrowGEOSTest, TestHppArrayReader) {
  GEOSCppHandle handle;
  geoarrow::geos::ArrayReader reader;
  EXPECT_STREQ(reader.GetLastError(), "");

  ASSERT_EQ(reader.InitFromEncoding(handle.handle, GEOARROW_GEOS_ENCODING_UNKNOWN),
            EINVAL);
  ASSERT_EQ(reader.InitFromEncoding(handle.handle, GEOARROW_GEOS_ENCODING_WKT),
            GEOARROW_GEOS_OK);
  EXPECT_STREQ(reader.GetLastError(), "");

  geoarrow::geos::ArrayReader reader2 = std::move(reader);
}

GeoArrowGEOSErrorCode SchemaFromWkbType(const std::vector<int32_t>& wkb_type,
                                        enum GeoArrowGEOSEncoding encoding,
                                        ArrowSchema* out) {
  geoarrow::geos::SchemaCalculator calc;
  calc.Ingest(wkb_type.data(), wkb_type.size());

  return calc.Finish(encoding, out);
}

GeoArrowGEOSErrorCode SchemaFromWKT(const std::vector<std::string>& wkt,
                                    enum GeoArrowGEOSEncoding encoding,
                                    ArrowSchema* out) {
  GEOSCppHandle handle;
  GEOSCppWKTReader wkt_reader(handle.handle);
  geoarrow::geos::GeometryVector geom(handle.handle);
  geom.resize(wkt.size());
  std::vector<int32_t> wkb_type(wkt.size());

  for (size_t i = 0; i < wkt.size(); i++) {
    if (wkt[i] == "") {
      wkb_type[i] = 0;
      continue;
    }

    wkt_reader.Read(wkt[i], geom.mutable_data() + i);
    wkb_type[i] = GeoArrowGEOSWKBType(handle.handle, geom.borrow(i));
  }

  return SchemaFromWkbType(wkb_type, encoding, out);
}

std::string SchemaExtensionName(ArrowSchema* schema) {
  ArrowStringView value;
  value.data = "";
  value.size_bytes = 0;
  ArrowMetadataGetValue(schema->metadata, ArrowCharView("ARROW:extension:name"), &value);
  return std::string(value.data, value.size_bytes);
}

std::string SchemaExtensionDims(ArrowSchema* schema) {
  if (std::string(schema->format) == "+l") {
    return SchemaExtensionDims(schema->children[0]);
  }

  std::stringstream ss;
  for (int64_t i = 0; i < schema->n_children; i++) {
    ss << schema->children[i]->name;
  }

  return ss.str();
}

TEST(GeoArrowGEOSTest, TestSchemaCalcEmpty) {
  nanoarrow::UniqueSchema schema;
  ASSERT_EQ(SchemaFromWkbType({}, GEOARROW_GEOS_ENCODING_UNKNOWN, schema.get()), EINVAL);

  ASSERT_EQ(SchemaFromWkbType({}, GEOARROW_GEOS_ENCODING_WKT, schema.get()),
            NANOARROW_OK);
  ASSERT_EQ(SchemaExtensionName(schema.get()), "geoarrow.wkt");

  schema.reset();
  ASSERT_EQ(SchemaFromWkbType({}, GEOARROW_GEOS_ENCODING_WKB, schema.get()),
            NANOARROW_OK);
  ASSERT_EQ(SchemaExtensionName(schema.get()), "geoarrow.wkb");

  schema.reset();
  ASSERT_EQ(SchemaFromWkbType({}, GEOARROW_GEOS_ENCODING_GEOARROW, schema.get()),
            NANOARROW_OK);
  ASSERT_EQ(SchemaExtensionName(schema.get()), "geoarrow.wkb");

  schema.reset();
  ASSERT_EQ(
      SchemaFromWkbType({}, GEOARROW_GEOS_ENCODING_GEOARROW_INTERLEAVED, schema.get()),
      NANOARROW_OK);
  ASSERT_EQ(SchemaExtensionName(schema.get()), "geoarrow.wkb");
}

TEST(GeoArrowGEOSTest, TestSchemaCalcZM) {
  nanoarrow::UniqueSchema schema;

  ASSERT_EQ(SchemaFromWkbType({1, 2001}, GEOARROW_GEOS_ENCODING_GEOARROW, schema.get()),
            NANOARROW_OK);
  ASSERT_EQ(SchemaExtensionName(schema.get()), "geoarrow.point");
  EXPECT_EQ(SchemaExtensionDims(schema.get()), "xyz");

  schema.reset();
  ASSERT_EQ(SchemaFromWkbType({2001, 1}, GEOARROW_GEOS_ENCODING_GEOARROW, schema.get()),
            NANOARROW_OK);
  ASSERT_EQ(SchemaExtensionName(schema.get()), "geoarrow.point");
  EXPECT_EQ(SchemaExtensionDims(schema.get()), "xyz");

  schema.reset();
  ASSERT_EQ(
      SchemaFromWkbType({2001, 2001}, GEOARROW_GEOS_ENCODING_GEOARROW, schema.get()),
      NANOARROW_OK);
  ASSERT_EQ(SchemaExtensionName(schema.get()), "geoarrow.point");
  EXPECT_EQ(SchemaExtensionDims(schema.get()), "xyz");

  schema.reset();
  ASSERT_EQ(SchemaFromWkbType({1, 3001}, GEOARROW_GEOS_ENCODING_GEOARROW, schema.get()),
            NANOARROW_OK);
  ASSERT_EQ(SchemaExtensionName(schema.get()), "geoarrow.point");
  EXPECT_EQ(SchemaExtensionDims(schema.get()), "xym");

  schema.reset();
  ASSERT_EQ(SchemaFromWkbType({3001, 1}, GEOARROW_GEOS_ENCODING_GEOARROW, schema.get()),
            NANOARROW_OK);
  ASSERT_EQ(SchemaExtensionName(schema.get()), "geoarrow.point");
  EXPECT_EQ(SchemaExtensionDims(schema.get()), "xym");

  schema.reset();
  ASSERT_EQ(
      SchemaFromWkbType({3001, 3001}, GEOARROW_GEOS_ENCODING_GEOARROW, schema.get()),
      NANOARROW_OK);
  ASSERT_EQ(SchemaExtensionName(schema.get()), "geoarrow.point");
  EXPECT_EQ(SchemaExtensionDims(schema.get()), "xym");

  schema.reset();
  ASSERT_EQ(
      SchemaFromWkbType({3001, 3001}, GEOARROW_GEOS_ENCODING_GEOARROW, schema.get()),
      NANOARROW_OK);
  ASSERT_EQ(SchemaExtensionName(schema.get()), "geoarrow.point");
  EXPECT_EQ(SchemaExtensionDims(schema.get()), "xym");

  schema.reset();
  ASSERT_EQ(
      SchemaFromWkbType({2001, 3001}, GEOARROW_GEOS_ENCODING_GEOARROW, schema.get()),
      NANOARROW_OK);
  ASSERT_EQ(SchemaExtensionName(schema.get()), "geoarrow.point");
  EXPECT_EQ(SchemaExtensionDims(schema.get()), "xyzm");

  schema.reset();
  ASSERT_EQ(
      SchemaFromWkbType({3001, 2001}, GEOARROW_GEOS_ENCODING_GEOARROW, schema.get()),
      NANOARROW_OK);
  ASSERT_EQ(SchemaExtensionName(schema.get()), "geoarrow.point");
  EXPECT_EQ(SchemaExtensionDims(schema.get()), "xyzm");

  schema.reset();
  ASSERT_EQ(
      SchemaFromWkbType({2001, 4001}, GEOARROW_GEOS_ENCODING_GEOARROW, schema.get()),
      NANOARROW_OK);
  ASSERT_EQ(SchemaExtensionName(schema.get()), "geoarrow.point");
  EXPECT_EQ(SchemaExtensionDims(schema.get()), "xyzm");

  schema.reset();
  ASSERT_EQ(
      SchemaFromWkbType({4001, 2001}, GEOARROW_GEOS_ENCODING_GEOARROW, schema.get()),
      NANOARROW_OK);
  ASSERT_EQ(SchemaExtensionName(schema.get()), "geoarrow.point");
  EXPECT_EQ(SchemaExtensionDims(schema.get()), "xyzm");

  schema.reset();
  ASSERT_EQ(
      SchemaFromWkbType({3001, 4001}, GEOARROW_GEOS_ENCODING_GEOARROW, schema.get()),
      NANOARROW_OK);
  ASSERT_EQ(SchemaExtensionName(schema.get()), "geoarrow.point");
  EXPECT_EQ(SchemaExtensionDims(schema.get()), "xyzm");

  schema.reset();
  ASSERT_EQ(
      SchemaFromWkbType({4001, 3001}, GEOARROW_GEOS_ENCODING_GEOARROW, schema.get()),
      NANOARROW_OK);
  ASSERT_EQ(SchemaExtensionName(schema.get()), "geoarrow.point");
  EXPECT_EQ(SchemaExtensionDims(schema.get()), "xyzm");
}

class SchemaCalcFixture : public ::testing::TestWithParam<std::vector<std::string>> {
 protected:
  std::vector<std::string> params;
};

TEST_P(SchemaCalcFixture, TestSchemaCalcSingleType) {
  auto params = GetParam();
  std::string extension_name = params[0];
  std::string dimensions = params[1];
  std::string non_null = params[2];
  std::string non_null_simple = params[3];
  std::string non_null_mixed = params[4];

  nanoarrow::UniqueSchema schema;

  // Length 1
  schema.reset();
  ASSERT_EQ(SchemaFromWKT({non_null}, GEOARROW_GEOS_ENCODING_GEOARROW, schema.get()),
            NANOARROW_OK);
  EXPECT_EQ(SchemaExtensionName(schema.get()), extension_name);
  EXPECT_EQ(SchemaExtensionDims(schema.get()), dimensions);

  // non-null, null
  schema.reset();
  ASSERT_EQ(SchemaFromWKT({non_null, ""}, GEOARROW_GEOS_ENCODING_GEOARROW, schema.get()),
            NANOARROW_OK);
  EXPECT_EQ(SchemaExtensionName(schema.get()), extension_name);
  EXPECT_EQ(SchemaExtensionDims(schema.get()), dimensions);

  // null, non-null
  schema.reset();
  ASSERT_EQ(SchemaFromWKT({"", non_null}, GEOARROW_GEOS_ENCODING_GEOARROW, schema.get()),
            NANOARROW_OK);
  EXPECT_EQ(SchemaExtensionName(schema.get()), extension_name);
  EXPECT_EQ(SchemaExtensionDims(schema.get()), dimensions);

  // non-null, non-null
  schema.reset();
  ASSERT_EQ(
      SchemaFromWKT({non_null, non_null}, GEOARROW_GEOS_ENCODING_GEOARROW, schema.get()),
      NANOARROW_OK);
  EXPECT_EQ(SchemaExtensionName(schema.get()), extension_name);
  EXPECT_EQ(SchemaExtensionDims(schema.get()), dimensions);

  // non-null, EMPTY
  schema.reset();
  ASSERT_EQ(SchemaFromWKT({non_null, "POINT EMPTY"}, GEOARROW_GEOS_ENCODING_GEOARROW,
                          schema.get()),
            NANOARROW_OK);
  EXPECT_EQ(SchemaExtensionName(schema.get()), extension_name);
  EXPECT_EQ(SchemaExtensionDims(schema.get()), dimensions);

  // simple, multi
  schema.reset();
  ASSERT_EQ(SchemaFromWKT({non_null_simple, non_null}, GEOARROW_GEOS_ENCODING_GEOARROW,
                          schema.get()),
            NANOARROW_OK);
  EXPECT_EQ(SchemaExtensionName(schema.get()), extension_name);
  EXPECT_EQ(SchemaExtensionDims(schema.get()), dimensions);

  // multi, simple
  schema.reset();
  ASSERT_EQ(SchemaFromWKT({non_null, non_null_simple}, GEOARROW_GEOS_ENCODING_GEOARROW,
                          schema.get()),
            NANOARROW_OK);
  EXPECT_EQ(SchemaExtensionName(schema.get()), extension_name);
  EXPECT_EQ(SchemaExtensionDims(schema.get()), dimensions);

  // mixed
  schema.reset();
  ASSERT_EQ(SchemaFromWKT({non_null, non_null_mixed}, GEOARROW_GEOS_ENCODING_GEOARROW,
                          schema.get()),
            NANOARROW_OK);
  EXPECT_EQ(SchemaExtensionName(schema.get()), "geoarrow.wkb");

  schema.reset();
  ASSERT_EQ(SchemaFromWKT({non_null_mixed, non_null}, GEOARROW_GEOS_ENCODING_GEOARROW,
                          schema.get()),
            NANOARROW_OK);
  EXPECT_EQ(SchemaExtensionName(schema.get()), "geoarrow.wkb");
}

INSTANTIATE_TEST_SUITE_P(
    GeoArrowGEOSTest, SchemaCalcFixture,
    ::testing::Values(
        // XY
        std::vector<std::string>({"geoarrow.point", "xy", "POINT (0 1)", "",
                                  "LINESTRING (0 1, 2 3)"}),
        std::vector<std::string>({"geoarrow.linestring", "xy", "LINESTRING (0 1, 2 3)",
                                  "", "POINT (0 1)"}),
        std::vector<std::string>({"geoarrow.polygon", "xy",
                                  "POLYGON ((0 0, 1 0, 0 1, 0 0))", "", "POINT (0 1)"}),
        std::vector<std::string>({"geoarrow.multipoint", "xy", "MULTIPOINT (0 1)",
                                  "POINT (0 1)", "LINESTRING (0 1, 2 3)"}),
        std::vector<std::string>({"geoarrow.multilinestring", "xy",
                                  "MULTILINESTRING ((0 1, 2 3))", "LINESTRING (0 1, 2 3)",
                                  "POINT (0 1)"}),
        std::vector<std::string>({"geoarrow.multipolygon", "xy",
                                  "MULTIPOLYGON (((0 0, 1 0, 0 1, 0 0)))",
                                  "POLYGON ((0 0, 1 0, 0 1, 0 0))", "POINT (0 1)"}),
        std::vector<std::string>({"geoarrow.wkb", "", "GEOMETRYCOLLECTION (POINT (0 1))",
                                  "", ""}),
        // XYZ
        std::vector<std::string>({"geoarrow.point", "xyz", "POINT Z (0 1 2)",
                                  "POINT (0 1)", "LINESTRING (0 1, 2 3)"})

            ));
