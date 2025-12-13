
#include <vector>

#include "geoarrow_geos.h"

namespace geoarrow {

namespace geos {

class GeometryVector {
 public:
  GeometryVector(GEOSContextHandle_t handle) : handle_(handle) {}

  GeometryVector(GeometryVector&& rhs)
      : handle_(rhs.handle_), data_(std::move(rhs.data_)) {
    rhs.data_.clear();
  }

  GeometryVector(GeometryVector& rhs) = delete;

  void reset(size_t offset, size_t length = 1) {
    for (size_t i = 0; i < length; i++) {
      GEOSGeometry* item = data_[offset + i];
      if (item != nullptr) {
        GEOSGeom_destroy_r(handle_, item);
      }
    }
  }

  ~GeometryVector() { reset(0, data_.size()); }

  void reserve(size_t n) { data_.reserve(n); }

  size_t size() { return data_.size(); }

  GEOSGeometry* take_ownership_of(size_t i) {
    GEOSGeometry* item = data_[i];
    data_[i] = nullptr;
    return item;
  }

  const GEOSGeometry* borrow(size_t i) { return data_[i]; }

  void set(size_t i, GEOSGeometry* value) {
    reset(i);
    data_[i] = value;
  }

  const GEOSGeometry** data() { return const_cast<const GEOSGeometry**>(data_.data()); }

  GEOSGeometry** mutable_data() { return data_.data(); }

  void resize(size_t n) {
    size_t current_size = size();
    if (n >= current_size) {
      data_.resize(n);
      for (size_t i = current_size; i < n; i++) {
        data_[i] = nullptr;
      }
    } else {
      reset(n, current_size - n);
      data_.resize(n);
    }
  }

 private:
  GEOSContextHandle_t handle_;
  std::vector<GEOSGeometry*> data_;
};

class ArrayBuilder {
 public:
  ArrayBuilder() : builder_(nullptr) {}

  ArrayBuilder(ArrayBuilder&& rhs) : builder_(rhs.builder_) { rhs.builder_ = nullptr; }

  ArrayBuilder(ArrayBuilder& rhs) = delete;

  ~ArrayBuilder() {
    if (builder_ != nullptr) {
      GeoArrowGEOSArrayBuilderDestroy(builder_);
    }
  }

  const char* GetLastError() {
    if (builder_ == nullptr) {
      return "";
    } else {
      return GeoArrowGEOSArrayBuilderGetLastError(builder_);
    }
  }

  GeoArrowGEOSErrorCode InitFromEncoding(GEOSContextHandle_t handle,
                                         GeoArrowGEOSEncoding encoding,
                                         int wkb_type = 0) {
    ArrowSchema tmp_schema;
    tmp_schema.release = nullptr;
    int result = GeoArrowGEOSMakeSchema(encoding, wkb_type, &tmp_schema);
    if (result != GEOARROW_GEOS_OK) {
      return result;
    }

    result = InitFromSchema(handle, &tmp_schema);
    tmp_schema.release(&tmp_schema);
    return result;
  }

  GeoArrowGEOSErrorCode InitFromSchema(GEOSContextHandle_t handle, ArrowSchema* schema) {
    if (builder_ != nullptr) {
      GeoArrowGEOSArrayBuilderDestroy(builder_);
    }

    return GeoArrowGEOSArrayBuilderCreate(handle, schema, &builder_);
  }

  GeoArrowGEOSErrorCode Append(const GEOSGeometry** geom, size_t geom_size,
                               size_t* n_appended) {
    return GeoArrowGEOSArrayBuilderAppend(builder_, geom, geom_size, n_appended);
  }

  GeoArrowGEOSErrorCode Finish(struct ArrowArray* out) {
    return GeoArrowGEOSArrayBuilderFinish(builder_, out);
  }

 private:
  GeoArrowGEOSArrayBuilder* builder_;
};

class ArrayReader {
 public:
  ArrayReader() : reader_(nullptr) {}

  ArrayReader(ArrayReader&& rhs) : reader_(rhs.reader_) { rhs.reader_ = nullptr; }

  ArrayReader(ArrayReader& rhs) = delete;

  ~ArrayReader() {
    if (reader_ != nullptr) {
      GeoArrowGEOSArrayReaderDestroy(reader_);
    }
  }

  const char* GetLastError() {
    if (reader_ == nullptr) {
      return "";
    } else {
      return GeoArrowGEOSArrayReaderGetLastError(reader_);
    }
  }

  GeoArrowGEOSErrorCode InitFromEncoding(GEOSContextHandle_t handle,
                                         GeoArrowGEOSEncoding encoding,
                                         int wkb_type = 0) {
    ArrowSchema tmp_schema;
    tmp_schema.release = nullptr;
    int result = GeoArrowGEOSMakeSchema(encoding, wkb_type, &tmp_schema);
    if (result != GEOARROW_GEOS_OK) {
      return result;
    }

    result = InitFromSchema(handle, &tmp_schema);
    tmp_schema.release(&tmp_schema);
    return result;
  }

  GeoArrowGEOSErrorCode InitFromSchema(GEOSContextHandle_t handle, ArrowSchema* schema) {
    if (reader_ != nullptr) {
      GeoArrowGEOSArrayReaderDestroy(reader_);
    }

    return GeoArrowGEOSArrayReaderCreate(handle, schema, &reader_);
  }

  GeoArrowGEOSErrorCode Read(ArrowArray* array, int64_t offset, int64_t length,
                             GEOSGeometry** out, size_t* n_out) {
    return GeoArrowGEOSArrayReaderRead(reader_, array, offset, length, out, n_out);
  }

 private:
  GeoArrowGEOSArrayReader* reader_;
};

class SchemaCalculator {
 public:
  SchemaCalculator() : calc_(nullptr) { GeoArrowGEOSSchemaCalculatorCreate(&calc_); }

  SchemaCalculator(SchemaCalculator&& rhs) : calc_(rhs.calc_) { rhs.calc_ = nullptr; }

  SchemaCalculator(SchemaCalculator& rhs) = delete;

  ~SchemaCalculator() {
    if (calc_ != nullptr) {
      GeoArrowGEOSSchemaCalculatorDestroy(calc_);
    }
  }

  void Ingest(const int32_t* wkb_type, size_t n) {
    GeoArrowGEOSSchemaCalculatorIngest(calc_, wkb_type, n);
  }

  GeoArrowGEOSErrorCode Finish(enum GeoArrowGEOSEncoding encoding, ArrowSchema* out) {
    return GeoArrowGEOSSchemaCalculatorFinish(calc_, encoding, out);
  }

 private:
  GeoArrowGEOSSchemaCalculator* calc_;
};

}  // namespace geos

}  // namespace geoarrow
