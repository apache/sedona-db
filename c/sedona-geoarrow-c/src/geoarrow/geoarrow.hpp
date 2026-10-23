#include "geoarrow/geoarrow.h"

#ifndef GEOARROW_HPP_EXCEPTION_INCLUDED
#define GEOARROW_HPP_EXCEPTION_INCLUDED

#include <exception>
#include <string>



#if defined(GEOARROW_DEBUG)
#define _GEOARROW_THROW_NOT_OK_IMPL(NAME, EXPR, EXPR_STR, ERR)                      \
  do {                                                                              \
    const int NAME = (EXPR);                                                        \
    if (NAME) {                                                                     \
      throw ::geoarrow::ErrnoException(                                             \
          NAME,                                                                     \
          std::string(EXPR_STR) + std::string(" failed with errno ") +              \
              std::to_string(NAME) + std::string("\n * ") + std::string(__FILE__) + \
              std::string(":") + std::to_string(__LINE__),                          \
          ERR);                                                                     \
    }                                                                               \
  } while (0)
#else
#define _GEOARROW_THROW_NOT_OK_IMPL(NAME, EXPR, EXPR_STR, ERR)                  \
  do {                                                                          \
    const int NAME = (EXPR);                                                    \
    if (NAME) {                                                                 \
      throw ::geoarrow::ErrnoException(NAME,                                    \
                                       std::string(EXPR_STR) +                  \
                                           std::string(" failed with errno ") + \
                                           std::to_string(NAME),                \
                                       ERR);                                    \
    }                                                                           \
  } while (0)
#endif

#define GEOARROW_THROW_NOT_OK(ERR, EXPR) \
  _GEOARROW_THROW_NOT_OK_IMPL(           \
      _GEOARROW_MAKE_NAME(errno_status_, _GEOARROW_UNIQUE_SUFFIX), EXPR, #EXPR, ERR)

namespace geoarrow {

class Exception : public std::exception {
 public:
  std::string message;

  Exception() = default;

  explicit Exception(const std::string& msg) : message(msg){};

  const char* what() const noexcept override { return message.c_str(); }
};

class ErrnoException : public Exception {
 public:
  GeoArrowErrorCode code{};

  ErrnoException(GeoArrowErrorCode code, const std::string& msg,
                 struct GeoArrowError* error)
      : code(code) {
    if (error != nullptr) {
      message = msg + ": \n" + error->message;
    } else {
      message = msg;
    }
  }
};

}  // namespace geoarrow

#endif

#ifndef GEOARROW_HPP_INTERNAL_INCLUDED
#define GEOARROW_HPP_INTERNAL_INCLUDED



namespace geoarrow {

namespace internal {
struct SchemaHolder {
  struct ArrowSchema schema {};
  ~SchemaHolder() {
    if (schema.release != nullptr) {
      schema.release(&schema);
    }
  }
};

struct ArrayHolder {
  struct ArrowArray array {};
  ~ArrayHolder() {
    if (array.release != nullptr) {
      array.release(&array);
    }
  }
};

template <typename T>
static inline void FreeWrappedBuffer(uint8_t* ptr, int64_t size, void* private_data) {
  GEOARROW_UNUSED(ptr);
  GEOARROW_UNUSED(size);
  auto obj = reinterpret_cast<T*>(private_data);
  delete obj;
}

template <typename T>
static inline struct GeoArrowBufferView BufferView(const T& v) {
  if (v.size() == 0) {
    return {nullptr, 0};
  } else {
    return {reinterpret_cast<const uint8_t*>(v.data()),
            static_cast<int64_t>(v.size() * sizeof(typename T::value_type))};
  }
}

}  // namespace internal

}  // namespace geoarrow

#endif

#ifndef GEOARROW_HPP_GEOMETRY_DATA_TYPE_INCLUDED
#define GEOARROW_HPP_GEOMETRY_DATA_TYPE_INCLUDED

#include <cstring>
#include <string>
#include <vector>





namespace geoarrow {

class GeometryDataType {
 public:
  GeometryDataType() = default;

  GeometryDataType(const GeometryDataType& other)
      : schema_view_(other.schema_view_),
        metadata_view_(other.metadata_view_),
        crs_(other.crs_) {
    metadata_view_.crs.data = crs_.data();
  }

  GeometryDataType& operator=(const GeometryDataType& other) {
    this->schema_view_ = other.schema_view_;
    this->metadata_view_ = other.metadata_view_;
    this->crs_ = other.crs_;
    this->metadata_view_.crs.data = this->crs_.data();
    return *this;
  }

  GeometryDataType(GeometryDataType&& other) { MoveFrom(&other); }

  GeometryDataType& operator=(GeometryDataType&& other) {
    MoveFrom(&other);
    return *this;
  }

  void MoveFrom(GeometryDataType* other) {
    schema_view_ = other->schema_view_;
    metadata_view_ = other->metadata_view_;
    crs_ = std::move(other->crs_);
    metadata_view_.crs.data = crs_.data();

    std::memset(&other->schema_view_, 0, sizeof(struct GeoArrowSchemaView));
    std::memset(&other->metadata_view_, 0, sizeof(struct GeoArrowMetadataView));
  }

  /// \brief Make a GeometryDataType from a geometry type, dimensions, and coordinate
  /// type.
  static GeometryDataType Make(
      enum GeoArrowGeometryType geometry_type,
      enum GeoArrowDimensions dimensions = GEOARROW_DIMENSIONS_XY,
      enum GeoArrowCoordType coord_type = GEOARROW_COORD_TYPE_SEPARATE,
      const std::string& metadata = "") {
    enum GeoArrowType type = GeoArrowMakeType(geometry_type, dimensions, coord_type);
    if (type == GEOARROW_TYPE_UNINITIALIZED) {
      throw ::geoarrow::Exception(
          std::string("Combination of geometry type/dimensions/coord type not valid: ") +
          GeoArrowGeometryTypeString(geometry_type) + "/" +
          GeoArrowDimensionsString(dimensions) + "/" +
          GeoArrowCoordTypeString(coord_type));
    }

    return Make(type, metadata);
  }

  /// \brief Make a GeometryDataType from a type identifier and optional extension
  /// metadata.
  static GeometryDataType Make(enum GeoArrowType type, const std::string& metadata = "") {
    switch (type) {
      case GEOARROW_TYPE_UNINITIALIZED:
        throw Exception(
            "Can't construct GeometryDataType from GEOARROW_TYPE_UNINITIALIZED");
      default:
        break;
    }

    GeoArrowError error{};
    GeoArrowSchemaView schema_view;
    GEOARROW_THROW_NOT_OK(nullptr, GeoArrowSchemaViewInitFromType(&schema_view, type));

    GeoArrowStringView metadata_str_view = {metadata.data(), (int64_t)metadata.size()};
    GeoArrowMetadataView metadata_view{};
    GEOARROW_THROW_NOT_OK(
        &error, GeoArrowMetadataViewInit(&metadata_view, metadata_str_view, &error));

    return GeometryDataType(schema_view, metadata_view);
  }

  /// \brief Make a GeometryDataType from an ArrowSchema extension type
  ///
  /// The caller retains ownership of schema.
  static GeometryDataType Make(const struct ArrowSchema* schema) {
    GeoArrowError error{};
    GeoArrowSchemaView schema_view{};
    GEOARROW_THROW_NOT_OK(&error, GeoArrowSchemaViewInit(&schema_view, schema, &error));

    GeoArrowMetadataView metadata_view{};
    GEOARROW_THROW_NOT_OK(
        &error,
        GeoArrowMetadataViewInit(&metadata_view, schema_view.extension_metadata, &error));

    return GeometryDataType(schema_view, metadata_view);
  }

  /// \brief Make a GeometryDataType from an ArrowSchema storage type
  ///
  /// The caller retains ownership of schema. If schema is an extension type,
  /// any extension type or metadata is ignored.
  static GeometryDataType Make(struct ArrowSchema* schema,
                               const std::string& extension_name,
                               const std::string& metadata = "") {
    struct GeoArrowError error {};
    struct GeoArrowSchemaView schema_view {};

    struct GeoArrowStringView extension_name_view = {extension_name.data(),
                                                     (int64_t)extension_name.size()};
    GEOARROW_THROW_NOT_OK(&error, GeoArrowSchemaViewInitFromStorage(
                                      &schema_view, schema, extension_name_view, &error));

    struct GeoArrowStringView metadata_str_view = {metadata.data(),
                                                   (int64_t)metadata.size()};
    struct GeoArrowMetadataView metadata_view {};
    GEOARROW_THROW_NOT_OK(
        &error, GeoArrowMetadataViewInit(&metadata_view, metadata_str_view, &error));

    return GeometryDataType(schema_view, metadata_view);
  }

  GeometryDataType WithGeometryType(enum GeoArrowGeometryType geometry_type) const {
    return Make(geometry_type, dimensions(), coord_type(), extension_metadata());
  }

  GeometryDataType WithCoordType(enum GeoArrowCoordType coord_type) const {
    return Make(geometry_type(), dimensions(), coord_type, extension_metadata());
  }

  GeometryDataType WithDimensions(enum GeoArrowDimensions dimensions) const {
    return Make(geometry_type(), dimensions, coord_type(), extension_metadata());
  }

  GeometryDataType WithEdgeType(enum GeoArrowEdgeType edge_type) const {
    GeometryDataType new_type(*this);
    new_type.metadata_view_.edge_type = edge_type;
    return new_type;
  }

  GeometryDataType WithCrs(const std::string& crs,
                           enum GeoArrowCrsType crs_type = GEOARROW_CRS_TYPE_UNKNOWN) {
    struct GeoArrowMetadataView metadata_view_copy = metadata_view_;
    metadata_view_copy.crs.data = crs.data();
    metadata_view_copy.crs.size_bytes = crs.size();
    metadata_view_copy.crs_type = crs_type;

    return GeometryDataType(schema_view_, metadata_view_copy);
  }

  GeometryDataType WithCrsLonLat() {
    struct GeoArrowMetadataView metadata_view_copy = metadata_view_;
    GeoArrowMetadataSetLonLat(&metadata_view_copy);
    return GeometryDataType(schema_view_, metadata_view_copy);
  }

  GeometryDataType XY() const { return WithDimensions(GEOARROW_DIMENSIONS_XY); }

  GeometryDataType XYZ() const { return WithDimensions(GEOARROW_DIMENSIONS_XYZ); }

  GeometryDataType XYM() const { return WithDimensions(GEOARROW_DIMENSIONS_XYM); }

  GeometryDataType XYZM() const { return WithDimensions(GEOARROW_DIMENSIONS_XYZM); }

  GeometryDataType Simple() const {
    switch (geometry_type()) {
      case GEOARROW_GEOMETRY_TYPE_POINT:
      case GEOARROW_GEOMETRY_TYPE_MULTIPOINT:
        return WithGeometryType(GEOARROW_GEOMETRY_TYPE_POINT);
      case GEOARROW_GEOMETRY_TYPE_LINESTRING:
      case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING:
        return WithGeometryType(GEOARROW_GEOMETRY_TYPE_LINESTRING);
      case GEOARROW_GEOMETRY_TYPE_POLYGON:
      case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON:
        return WithGeometryType(GEOARROW_GEOMETRY_TYPE_POLYGON);
      default:
        throw ::geoarrow::Exception(
            std::string("Can't make simple type from geometry type ") +
            GeoArrowGeometryTypeString(geometry_type()));
    }
  }

  GeometryDataType Multi() const {
    switch (geometry_type()) {
      case GEOARROW_GEOMETRY_TYPE_POINT:
      case GEOARROW_GEOMETRY_TYPE_MULTIPOINT:
        return WithGeometryType(GEOARROW_GEOMETRY_TYPE_MULTIPOINT);
      case GEOARROW_GEOMETRY_TYPE_LINESTRING:
      case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING:
        return WithGeometryType(GEOARROW_GEOMETRY_TYPE_MULTILINESTRING);
      case GEOARROW_GEOMETRY_TYPE_POLYGON:
      case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON:
        return WithGeometryType(GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON);
      default:
        throw ::geoarrow::Exception(
            std::string("Can't make multi type from geometry type ") +
            GeoArrowGeometryTypeString(geometry_type()));
    }
  }

  void InitSchema(struct ArrowSchema* schema_out) const {
    GEOARROW_THROW_NOT_OK(nullptr,
                          GeoArrowSchemaInitExtension(schema_out, schema_view_.type));
    GEOARROW_THROW_NOT_OK(nullptr,
                          GeoArrowSchemaSetMetadata(schema_out, &metadata_view_));
  }

  void InitStorageSchema(struct ArrowSchema* schema_out) const {
    GEOARROW_THROW_NOT_OK(nullptr, GeoArrowSchemaInit(schema_out, schema_view_.type));
  }

  std::string extension_name() const {
    const char* name = GeoArrowExtensionNameFromType(schema_view_.type);
    if (name == NULL) {
      throw ::geoarrow::Exception(
          std::string("Extension name not available for type with id ") +
          std::to_string(schema_view_.type));
    }

    return name;
  }

  std::string extension_metadata() const {
    int64_t metadata_size = GeoArrowMetadataSerialize(&metadata_view_, nullptr, 0);
    char* out = reinterpret_cast<char*>(malloc(metadata_size));
    GeoArrowMetadataSerialize(&metadata_view_, out, metadata_size);
    std::string metadata(out, metadata_size);
    free(out);
    return metadata;
  }

  enum GeoArrowType id() const { return schema_view_.type; }

  enum GeoArrowGeometryType geometry_type() const { return schema_view_.geometry_type; }
  enum GeoArrowCoordType coord_type() const { return schema_view_.coord_type; }

  enum GeoArrowDimensions dimensions() const { return schema_view_.dimensions; }

  int num_dimensions() const {
    switch (dimensions()) {
      case GEOARROW_DIMENSIONS_XY:
        return 2;
      case GEOARROW_DIMENSIONS_XYZ:
      case GEOARROW_DIMENSIONS_XYM:
        return 3;
      case GEOARROW_DIMENSIONS_XYZM:
        return 4;
      default:
        return -1;
    }
  }

  enum GeoArrowEdgeType edge_type() const { return metadata_view_.edge_type; }

  enum GeoArrowCrsType crs_type() const { return metadata_view_.crs_type; }

  std::string crs() const {
    int64_t len = GeoArrowUnescapeCrs(metadata_view_.crs, nullptr, 0);
    char* out = reinterpret_cast<char*>(malloc(len));
    GeoArrowUnescapeCrs(metadata_view_.crs, out, len);
    std::string out_str(out, len);
    free(out);
    return out_str;
  }

  std::string ToString(size_t max_crs_size = 40) const {
    if (id() == GEOARROW_TYPE_UNINITIALIZED) {
      return "<Uninitialized GeometryDataType>";
    }

    std::string ext_name = extension_name();

    std::string dims;
    switch (dimensions()) {
      case GEOARROW_DIMENSIONS_UNKNOWN:
      case GEOARROW_DIMENSIONS_XY:
        break;
      default:
        dims = std::string("_") +
               std::string(GeoArrowDimensionsString(dimensions())).substr(2);
        break;
    }

    std::vector<std::string> modifiers;
    if (id() == GEOARROW_TYPE_LARGE_WKT || id() == GEOARROW_TYPE_LARGE_WKB) {
      modifiers.push_back("large");
    }

    if (id() == GEOARROW_TYPE_WKT_VIEW || id() == GEOARROW_TYPE_WKB_VIEW) {
      modifiers.push_back("view");
    }

    if (edge_type() != GEOARROW_EDGE_TYPE_PLANAR) {
      modifiers.push_back(GeoArrowEdgeTypeString(edge_type()));
    }

    if (coord_type() == GEOARROW_COORD_TYPE_INTERLEAVED) {
      modifiers.push_back("interleaved");
    }

    std::string type_prefix;
    for (const auto& modifier : modifiers) {
      type_prefix += modifier + " ";
    }

    std::string crs_suffix;
    switch (crs_type()) {
      case GEOARROW_CRS_TYPE_NONE:
        break;
      case GEOARROW_CRS_TYPE_UNKNOWN:
        crs_suffix = crs();
        break;
      default:
        crs_suffix = std::string(GeoArrowCrsTypeString(crs_type())) + ":" + crs();
        break;
    }

    if (!crs_suffix.empty()) {
      crs_suffix = std::string("<") + crs_suffix + ">";
    }

    if (crs_suffix.size() >= max_crs_size) {
      crs_suffix = crs_suffix.substr(0, max_crs_size - 4) + "...>";
    }

    return type_prefix + ext_name + dims + crs_suffix;
  }

 private:
  struct GeoArrowSchemaView schema_view_ {};
  struct GeoArrowMetadataView metadata_view_ {};
  std::string crs_;

  GeometryDataType(struct GeoArrowSchemaView schema_view,
                   struct GeoArrowMetadataView metadata_view) {
    schema_view_.geometry_type = schema_view.geometry_type;
    schema_view_.dimensions = schema_view.dimensions;
    schema_view_.coord_type = schema_view.coord_type;
    schema_view_.type = schema_view.type;

    metadata_view_.edge_type = metadata_view.edge_type;
    crs_ = std::string(metadata_view.crs.data, metadata_view.crs.size_bytes);
    metadata_view_.crs_type = metadata_view.crs_type;
    metadata_view_.crs.data = crs_.data();
    metadata_view_.crs.size_bytes = crs_.size();
  }
};

static inline GeometryDataType Wkb() { return GeometryDataType::Make(GEOARROW_TYPE_WKB); }

static inline GeometryDataType Wkt() { return GeometryDataType::Make(GEOARROW_TYPE_WKT); }

static inline GeometryDataType Box() { return GeometryDataType::Make(GEOARROW_TYPE_BOX); }

static inline GeometryDataType Point() {
  return GeometryDataType::Make(GEOARROW_TYPE_POINT);
}

static inline GeometryDataType Linestring() {
  return GeometryDataType::Make(GEOARROW_TYPE_LINESTRING);
}

static inline GeometryDataType Polygon() {
  return GeometryDataType::Make(GEOARROW_TYPE_POLYGON);
}

}  // namespace geoarrow

#endif
#ifndef GEOARROW_HPP_ARRAY_READER_INCLUDED
#define GEOARROW_HPP_ARRAY_READER_INCLUDED





namespace geoarrow {

class ArrayView {
 public:
  explicit ArrayView(const struct GeoArrowArrayView* array_view = nullptr)
      : array_view_(array_view) {}

  void Init(const struct GeoArrowArrayView* array_view) { array_view_ = array_view; }

  bool is_valid() {
    return array_view_ != nullptr &&
           array_view_->schema_view.type != GEOARROW_TYPE_UNINITIALIZED;
  }

  const struct GeoArrowArrayView* array_view() { return array_view_; }

 private:
  const struct GeoArrowArrayView* array_view_{};
};

class ArrayReader {
 public:
  explicit ArrayReader(GeoArrowType type) {
    GEOARROW_THROW_NOT_OK(nullptr, GeoArrowArrayReaderInitFromType(&reader_, type));
  }

  explicit ArrayReader(const GeometryDataType& type) : ArrayReader(type.id()) {}

  explicit ArrayReader(const ArrowSchema* schema) {
    struct GeoArrowError error {};
    GEOARROW_THROW_NOT_OK(&error,
                          GeoArrowArrayReaderInitFromSchema(&reader_, schema, &error));
  }

  ~ArrayReader() {
    if (reader_.private_data != nullptr) {
      GeoArrowArrayReaderReset(&reader_);
    }
  }

  void SetArray(struct ArrowArray* array) {
    std::memcpy(&array_.array, array, sizeof(struct ArrowArray));
    array->release = nullptr;
    SetArrayNonOwning(&array_.array);
  }

  void SetArrayNonOwning(const struct ArrowArray* array) {
    struct GeoArrowError error {};
    GEOARROW_THROW_NOT_OK(&error, GeoArrowArrayReaderSetArray(&reader_, array, &error));
  }

  GeoArrowErrorCode Visit(struct GeoArrowVisitor* visitor, int64_t offset, int64_t length,
                          struct GeoArrowError* error = nullptr) {
    visitor->error = error;
    return GeoArrowArrayReaderVisit(&reader_, offset, length, visitor);
  }

  ArrayView& View() {
    if (!view_.is_valid()) {
      const struct GeoArrowArrayView* array_view = nullptr;
      GEOARROW_THROW_NOT_OK(nullptr, GeoArrowArrayReaderArrayView(&reader_, &array_view));
      view_.Init(array_view);
    }

    return view_;
  }

 private:
  struct GeoArrowArrayReader reader_ {};
  ArrayView view_;
  internal::ArrayHolder array_;
};
}  // namespace geoarrow

#endif
#ifndef GEOARROW_HPP_ARRAY_WRITER_INCLUDED
#define GEOARROW_HPP_ARRAY_WRITER_INCLUDED





namespace geoarrow {

class ArrayBuilder {
 public:
  explicit ArrayBuilder(GeoArrowType type) {
    GEOARROW_THROW_NOT_OK(nullptr, GeoArrowBuilderInitFromType(&builder_, type));
  }
  explicit ArrayBuilder(const GeometryDataType& type) : ArrayBuilder(type.id()) {}
  explicit ArrayBuilder(const ArrowSchema* schema) {
    struct GeoArrowError error {};
    GEOARROW_THROW_NOT_OK(&error,
                          GeoArrowBuilderInitFromSchema(&builder_, schema, &error));
  }

  ~ArrayBuilder() {
    if (builder_.private_data != nullptr) {
      GeoArrowBuilderReset(&builder_);
    }
  }

  void Finish(struct ArrowArray* out) {
    struct GeoArrowError error {};
    GEOARROW_THROW_NOT_OK(&error, GeoArrowBuilderFinish(&builder_, out, &error));
  }

  struct GeoArrowBuilder* builder() { return &builder_; }

  template <typename T>
  void SetBufferWrapped(int64_t i, T obj, GeoArrowBufferView value) {
    T* obj_moved = new T(std::move(obj));
    GeoArrowErrorCode result = GeoArrowBuilderSetOwnedBuffer(
        builder_, i, value, &internal::FreeWrappedBuffer<T>, obj_moved);
    if (result != GEOARROW_OK) {
      delete obj_moved;
      throw ::geoarrow::ErrnoException(result, "GeoArrowBuilderSetOwnedBuffer()",
                                       nullptr);
    }
  }

  template <typename T>
  void SetBufferWrapped(int64_t i, T obj) {
    T* obj_moved = new T(std::move(obj));
    struct GeoArrowBufferView value {
      reinterpret_cast<const uint8_t*>(obj_moved->data()),
          static_cast<int64_t>(obj_moved->size() * sizeof(typename T::value_type))
    };

    GeoArrowErrorCode result = GeoArrowBuilderSetOwnedBuffer(
        &builder_, i, value, &internal::FreeWrappedBuffer<T>, obj_moved);
    if (result != GEOARROW_OK) {
      delete obj_moved;
      throw ::geoarrow::ErrnoException(result, "GeoArrowBuilderSetOwnedBuffer()",
                                       nullptr);
    }
  }

  template <typename T>
  void AppendToBuffer(int64_t i, const T& obj) {
    GEOARROW_THROW_NOT_OK(
        nullptr, GeoArrowBuilderAppendBuffer(&builder_, i, internal::BufferView(obj)));
  }

  template <typename T>
  void AppendToOffsetBuffer(int64_t i, const T& obj) {
    GEOARROW_THROW_NOT_OK(nullptr, GeoArrowBuilderOffsetAppend(
                                       &builder_, static_cast<int32_t>(i), obj.data(),
                                       static_cast<int64_t>(obj.size())));
  }

  void AppendCoords(const GeoArrowCoordView* coords, enum GeoArrowDimensions dimensions,
                    int64_t offset, int64_t length) {
    GEOARROW_THROW_NOT_OK(nullptr, GeoArrowBuilderCoordsAppend(
                                       &builder_, coords, dimensions, offset, length));
  }

 protected:
  struct GeoArrowBuilder builder_ {};
};

class ArrayWriter {
 public:
  explicit ArrayWriter(GeoArrowType type) {
    GEOARROW_THROW_NOT_OK(nullptr, GeoArrowArrayWriterInitFromType(&writer_, type));
  }
  explicit ArrayWriter(const GeometryDataType& type) : ArrayWriter(type.id()) {}
  explicit ArrayWriter(const ArrowSchema* schema) {
    GEOARROW_THROW_NOT_OK(nullptr, GeoArrowArrayWriterInitFromSchema(&writer_, schema));
  }

  ~ArrayWriter() {
    if (writer_.private_data != nullptr) {
      GeoArrowArrayWriterReset(&writer_);
    }
  }

  void SetPrecision(int precision) {
    GEOARROW_THROW_NOT_OK(nullptr, GeoArrowArrayWriterSetPrecision(&writer_, precision));
  }

  void SetFlatMultipoint(bool flat_multipoint) {
    GEOARROW_THROW_NOT_OK(nullptr,
                          GeoArrowArrayWriterSetPrecision(&writer_, flat_multipoint));
  }

  struct GeoArrowVisitor* visitor() {
    if (visitor_.coords == nullptr) {
      GEOARROW_THROW_NOT_OK(nullptr, GeoArrowArrayWriterInitVisitor(&writer_, &visitor_));
    }

    return &visitor_;
  }

  void Finish(struct ArrowArray* out) {
    struct GeoArrowError error {};
    GEOARROW_THROW_NOT_OK(&error, GeoArrowArrayWriterFinish(&writer_, out, &error));
  }

 private:
  struct GeoArrowArrayWriter writer_ {};
  struct GeoArrowVisitor visitor_ {};
};

}  // namespace geoarrow

#endif

#ifndef GEOARROW_HPP_ARRAY_UTIL_INCLUDED
#define GEOARROW_HPP_ARRAY_UTIL_INCLUDED

#include <array>
#include <iterator>
#include <limits>
#include <type_traits>



/// \defgroup hpp-array-utility Array iteration utilities
///
/// This header provides utilities for iterating over native GeoArrow arrays
/// (i.e., nested lists of coordinates where the coordinates are contiguous).
/// This header only depends on geoarrow_type.h (i.e., it does not use any symbols
/// from the geoarrow-c runtime) and most logic is written such that it the
/// dependency on the types could be removed.
///
/// @{

namespace geoarrow {

namespace array_util {

namespace internal {

// The verbose bits of a random access iterator for simple outer + index-based
// iteration. Requires that an implementation defineds value_type operator[]
// and value_type operator*.
template <typename T>
T SafeLoadAs(const uint8_t* unaligned) {
  T out;
  std::memcpy(&out, unaligned, sizeof(T));
  return out;
}

template <typename Outer>
class BaseRandomAccessIterator {
 public:
  explicit BaseRandomAccessIterator(const Outer& outer, int64_t i)
      : outer_(outer), i_(i) {}
  BaseRandomAccessIterator& operator++() {
    i_++;
    return *this;
  }
  BaseRandomAccessIterator& operator--() {
    i_--;
    return *this;
  }
  BaseRandomAccessIterator& operator+=(int64_t n) {
    i_ += n;
    return *this;
  }
  BaseRandomAccessIterator& operator-=(int64_t n) {
    i_ -= n;
    return *this;
  }
  int64_t operator-(const BaseRandomAccessIterator& other) const { return i_ - other.i_; }
  bool operator<(const BaseRandomAccessIterator& other) const { return i_ < other.i_; }
  bool operator>(const BaseRandomAccessIterator& other) const { return i_ > other.i_; }
  bool operator<=(const BaseRandomAccessIterator& other) const { return i_ <= other.i_; }
  bool operator>=(const BaseRandomAccessIterator& other) const { return i_ >= other.i_; }
  bool operator==(const BaseRandomAccessIterator& other) const { return i_ == other.i_; }
  bool operator!=(const BaseRandomAccessIterator& other) const { return i_ != other.i_; }

 protected:
  const Outer& outer_;
  int64_t i_;
};

/// \brief Iterator implementation for CoordSequence
template <typename CoordSequence>
class CoordSequenceIterator : public BaseRandomAccessIterator<CoordSequence> {
 public:
  using iterator_category = std::random_access_iterator_tag;
  using difference_type = int64_t;
  using value_type = typename CoordSequence::value_type;
  using reference = value_type&;
  using pointer = value_type*;

  explicit CoordSequenceIterator(const CoordSequence& outer, int64_t i)
      : BaseRandomAccessIterator<CoordSequence>(outer, i) {}
  value_type operator*() const { return this->outer_.coord(this->i_); }
  value_type operator[](int64_t i) const { return this->outer_.coord(this->i_ + i); }
};

/// \brief Iterator implementation for ListSequence
template <typename ListSequence>
class ListSequenceIterator : public BaseRandomAccessIterator<ListSequence> {
 public:
  using iterator_category = std::random_access_iterator_tag;
  using difference_type = int64_t;
  using value_type = typename ListSequence::value_type;

  explicit ListSequenceIterator(const ListSequence& outer, int64_t i)
      : BaseRandomAccessIterator<ListSequence>(outer, i), stashed_(outer.child) {}

  typename ListSequence::child_type& operator*() {
    this->outer_.UpdateChild(&stashed_, this->i_);
    return stashed_;
  }

  // Not quite a random access iterator because it doesn't implement []
  // (which would necessitate a copy, which we don't really want to do)
 private:
  typename ListSequence::child_type stashed_;
};

// Iterator implementation for a binary sequence. It might be faster to
// cache some offset buffer dereferences when performing sequential scans.
template <typename BinarySequence>
class BinarySequenceIterator : public BaseRandomAccessIterator<BinarySequence> {
 public:
  explicit BinarySequenceIterator(const BinarySequence& outer, int64_t i)
      : BaseRandomAccessIterator<BinarySequence>(outer, i) {}

  using iterator_category = std::random_access_iterator_tag;
  using difference_type = int64_t;
  using value_type = typename BinarySequence::value_type;

  value_type operator*() { return this->outer_.blob(this->i_); }

  value_type operator[](int64_t i) { return this->outer_.blob(this->i_ + i); }
};

// Iterator for dimension begin/end
template <typename T>
class StridedIterator {
 public:
  explicit StridedIterator(const T* ptr, int64_t stride) : ptr_(ptr), stride_(stride) {}
  StridedIterator& operator++() {
    ptr_ += stride_;
    return *this;
  }
  T operator++(int) {
    T retval = *ptr_;
    ptr_ += stride_;
    return retval;
  }
  StridedIterator& operator--() {
    ptr_ -= stride_;
    return *this;
  }
  StridedIterator& operator+=(int64_t n) {
    ptr_ += (n * stride_);
    return *this;
  }
  StridedIterator& operator-=(int64_t n) {
    ptr_ -= (n * stride_);
    return *this;
  }
  int64_t operator-(const StridedIterator& other) const {
    return (ptr_ - other.ptr_) / stride_;
  }
  bool operator<(const StridedIterator& other) const { return ptr_ < other.ptr_; }
  bool operator>(const StridedIterator& other) const { return ptr_ > other.ptr_; }
  bool operator<=(const StridedIterator& other) const { return ptr_ <= other.ptr_; }
  bool operator>=(const StridedIterator& other) const { return ptr_ >= other.ptr_; }
  bool operator==(const StridedIterator& other) const { return ptr_ == other.ptr_; }
  bool operator!=(const StridedIterator& other) const { return ptr_ != other.ptr_; }

  T operator*() const { return *ptr_; }
  T operator[](ptrdiff_t i) const { return ptr_[i]; }

  using iterator_category = std::random_access_iterator_tag;
  using difference_type = int64_t;
  using value_type = T;
  using reference = T&;
  using pointer = T*;

 protected:
  const T* ptr_;
  ptrdiff_t stride_;
};

template <typename T>
struct LoadIdentity {
  T operator()(const uint8_t* unaligned) const {
    return internal::SafeLoadAs<T>(unaligned);
  };
};

template <typename T>
struct LoadSwapped {
  T operator()(const uint8_t* unaligned) const {
    uint8_t swapped[sizeof(T)];
    for (size_t i = 0; i < sizeof(T); i++) {
      swapped[sizeof(T) - i - 1] = unaligned[i];
    }
    return internal::SafeLoadAs<T>(swapped);
  };
};

template <typename T, typename Load>
class UnalignedStridedIterator {
 public:
  explicit UnalignedStridedIterator(const uint8_t* ptr, int64_t stride)
      : ptr_(ptr), stride_(stride) {}
  UnalignedStridedIterator& operator++() {
    ptr_ += stride_;
    return *this;
  }
  T operator++(int) {
    T retval = *ptr_;
    ptr_ += stride_;
    return retval;
  }
  UnalignedStridedIterator& operator--() {
    ptr_ -= stride_;
    return *this;
  }
  UnalignedStridedIterator& operator+=(int64_t n) {
    ptr_ += (n * stride_);
    return *this;
  }
  UnalignedStridedIterator& operator-=(int64_t n) {
    ptr_ -= (n * stride_);
    return *this;
  }
  int64_t operator-(const UnalignedStridedIterator& other) const {
    return (ptr_ - other.ptr_) / stride_;
  }
  bool operator<(const UnalignedStridedIterator& other) const {
    return ptr_ < other.ptr_;
  }
  bool operator>(const UnalignedStridedIterator& other) const {
    return ptr_ > other.ptr_;
  }
  bool operator<=(const UnalignedStridedIterator& other) const {
    return ptr_ <= other.ptr_;
  }
  bool operator>=(const UnalignedStridedIterator& other) const {
    return ptr_ >= other.ptr_;
  }
  bool operator==(const UnalignedStridedIterator& other) const {
    return ptr_ == other.ptr_;
  }
  bool operator!=(const UnalignedStridedIterator& other) const {
    return ptr_ != other.ptr_;
  }

  T operator*() const { return load_(ptr_); }
  T operator[](ptrdiff_t i) const { return load_(ptr_ + (i * stride_)); }

  using iterator_category = std::random_access_iterator_tag;
  using difference_type = int64_t;
  using value_type = T;
  using reference = T&;
  using pointer = T*;

 protected:
  const uint8_t* ptr_;
  int64_t stride_;
  static constexpr Load load_{};
};

}  // namespace internal

template <typename T>
struct BoxXY;
template <typename T>
struct BoxXYZ;
template <typename T>
struct BoxXYM;
template <typename T>
struct BoxXYZM;
template <typename CoordSrc, typename CoordDst>
CoordDst CoordCast(CoordSrc src);

/// \brief Coord implementation for XY
template <typename T>
struct XY : public std::array<T, 2> {
  using box_type = BoxXY<T>;
  static constexpr enum GeoArrowDimensions dimensions = GEOARROW_DIMENSIONS_XY;

  T x() const { return this->at(0); }
  T y() const { return this->at(1); }
  T z() const { return std::numeric_limits<T>::quiet_NaN(); }
  T m() const { return std::numeric_limits<T>::quiet_NaN(); }

  template <typename CoordDst, typename Func>
  void VisitVertices(Func&& func) const {
    func(CoordCast<XY, CoordDst>(*this));
  }

  template <typename CoordDst, typename Func>
  void VisitEdges(Func&& func) const {
    CoordDst coord = CoordCast<XY, CoordDst>(*this);
    func(coord, coord);
  }

  static XY FromXYZM(T x, T y, T z, T m) {
    GEOARROW_UNUSED(z);
    GEOARROW_UNUSED(m);
    return XY{x, y};
  }
};

/// \brief Coord implementation for XYZ
template <typename T>
struct XYZ : public std::array<T, 3> {
  using box_type = BoxXYZ<T>;
  static constexpr enum GeoArrowDimensions dimensions = GEOARROW_DIMENSIONS_XYZ;

  T x() const { return this->at(0); }
  T y() const { return this->at(1); }
  T z() const { return this->at(2); }
  T m() const { return std::numeric_limits<T>::quiet_NaN(); }

  template <typename CoordDst, typename Func>
  void VisitVertices(Func&& func) const {
    func(CoordCast<XYZ, CoordDst>(*this));
  }

  template <typename CoordDst, typename Func>
  void VisitEdges(Func&& func) const {
    CoordDst coord = CoordCast<XYZ, CoordDst>(*this);
    func(coord, coord);
  }

  static XYZ FromXYZM(T x, T y, T z, T m) {
    GEOARROW_UNUSED(m);
    return XYZ{x, y, z};
  }
};

/// \brief Coord implementation for XYM
template <typename T>
struct XYM : public std::array<T, 3> {
  using box_type = BoxXYM<T>;
  static constexpr enum GeoArrowDimensions dimensions = GEOARROW_DIMENSIONS_XYM;

  T x() const { return this->at(0); }
  T y() const { return this->at(1); }
  T z() const { return std::numeric_limits<T>::quiet_NaN(); }
  T m() const { return this->at(2); }

  template <typename CoordDst, typename Func>
  void VisitVertices(Func&& func) const {
    func(CoordCast<XYM, CoordDst>(*this));
  }

  template <typename CoordDst, typename Func>
  void VisitEdges(Func&& func) const {
    CoordDst coord = CoordCast<XYM, CoordDst>(*this);
    func(coord, coord);
  }

  static XYM FromXYZM(T x, T y, T z, T m) {
    GEOARROW_UNUSED(z);
    return XYM{x, y, m};
  }
};

/// \brief Coord implementation for XYZM
template <typename T>
struct XYZM : public std::array<T, 4> {
  using box_type = BoxXYZM<T>;
  static constexpr enum GeoArrowDimensions dimensions = GEOARROW_DIMENSIONS_XYZM;

  T x() const { return this->at(0); }
  T y() const { return this->at(1); }
  T z() const { return this->at(2); }
  T m() const { return this->at(3); }

  template <typename CoordDst, typename Func>
  void VisitVertices(Func&& func) const {
    func(CoordCast<XYZM, CoordDst>(*this));
  }

  template <typename CoordDst, typename Func>
  void VisitEdges(Func&& func) const {
    CoordDst coord = CoordCast<XYZM, CoordDst>(*this);
    func(coord, coord);
  }

  static XYZM FromXYZM(T x, T y, T z, T m) { return XYZM{x, y, z, m}; }
};

/// \brief Cast a coordinate from one type to another
///
/// When one type of coordinate is requested but another exists, perform
/// the transformation by (1) dropping dimensions or (2) filling dimensions
/// that did not previously exist with NaN. A difference between ordinate type
/// (e.g., double/float) is handled using static_cast<>().
template <typename CoordSrc, typename CoordDst>
CoordDst CoordCast(CoordSrc src) {
  if constexpr (std::is_same<CoordSrc, CoordDst>::value) {
    return src;
  } else if constexpr (std::is_same<CoordDst, XY<typename CoordSrc::value_type>>::value) {
    return XY<typename CoordSrc::value_type>{src.x(), src.y()};
  } else {
    return CoordDst::FromXYZM(static_cast<typename CoordDst::value_type>(src.x()),
                              static_cast<typename CoordDst::value_type>(src.y()),
                              static_cast<typename CoordDst::value_type>(src.z()),
                              static_cast<typename CoordDst::value_type>(src.m()));
  }
}

/// \brief Coord implementation for Box
template <typename T>
struct BoxXY : public std::array<T, 4> {
  using bound_type = XY<T>;
  T xmin() const { return this->at(0); }
  T ymin() const { return this->at(1); }
  T zmin() const { return std::numeric_limits<T>::infinity(); }
  T mmin() const { return std::numeric_limits<T>::infinity(); }
  T xmax() const { return this->at(2); }
  T ymax() const { return this->at(3); }
  T zmax() const { return -std::numeric_limits<T>::infinity(); }
  T mmax() const { return -std::numeric_limits<T>::infinity(); }
  bound_type lower_bound() const { return {xmin(), ymin()}; }
  bound_type upper_bound() const { return {xmax(), ymax()}; }
  static BoxXY Empty() {
    return {std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity(),
            -std::numeric_limits<T>::infinity(), -std::numeric_limits<T>::infinity()};
  }
};

/// \brief Coord implementation for BoxZ
template <typename T>
struct BoxXYZ : public std::array<T, 6> {
  using bound_type = XYZ<T>;
  T xmin() const { return this->at(0); }
  T ymin() const { return this->at(1); }
  T zmin() const { return this->at(2); }
  T mmin() const { return std::numeric_limits<T>::infinity(); }
  T xmax() const { return this->at(3); }
  T ymax() const { return this->at(4); }
  T zmax() const { return this->at(5); }
  T mmax() const { return -std::numeric_limits<T>::infinity(); }
  bound_type lower_bound() const { return {xmin(), ymin(), zmin()}; }
  bound_type upper_bound() const { return {xmax(), ymax(), zmax()}; }
  static BoxXYZ Empty() {
    return {std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity(),
            -std::numeric_limits<T>::infinity(), -std::numeric_limits<T>::infinity()};
  }
};

/// \brief Coord implementation for BoxM
template <typename T>
struct BoxXYM : public std::array<T, 6> {
  using bound_type = XYM<T>;
  T xmin() const { return this->at(0); }
  T ymin() const { return this->at(1); }
  T zmin() const { return std::numeric_limits<T>::infinity(); }
  T mmin() const { return this->at(2); }
  T xmax() const { return this->at(3); }
  T ymax() const { return this->at(4); }
  T zmax() const { return -std::numeric_limits<T>::infinity(); }
  T mmax() const { return this->at(5); }
  bound_type lower_bound() const { return {xmin(), ymin(), mmin()}; }
  bound_type upper_bound() const { return {xmax(), ymax(), mmax()}; }
  static BoxXYM Empty() {
    return {std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity(),
            -std::numeric_limits<T>::infinity(), -std::numeric_limits<T>::infinity()};
  }
};

/// \brief Coord implementation for BoxZM
template <typename T>
struct BoxXYZM : public std::array<T, 8> {
  using bound_type = XYZM<T>;
  T xmin() const { return this->at(0); }
  T ymin() const { return this->at(1); }
  T zmin() const { return this->at(2); }
  T mmin() const { return this->at(3); }
  T xmax() const { return this->at(4); }
  T ymax() const { return this->at(5); }
  T zmax() const { return this->at(6); }
  T mmax() const { return this->at(7); }
  bound_type lower_bound() const { return {xmin(), ymin(), zmin(), mmin()}; }
  bound_type upper_bound() const { return {xmax(), ymax(), zmax(), mmax()}; }
  static BoxXYZM Empty() {
    return {std::numeric_limits<T>::infinity(),  std::numeric_limits<T>::infinity(),
            std::numeric_limits<T>::infinity(),  std::numeric_limits<T>::infinity(),
            -std::numeric_limits<T>::infinity(), -std::numeric_limits<T>::infinity(),
            -std::numeric_limits<T>::infinity(), -std::numeric_limits<T>::infinity()};
  }
};

/// \brief View of a GeoArrow coordinate sequence
///
/// A view of zero or more coordinates. This data structure can handle either interleaved
/// coordinates, separated coordinates, or any other sequence where values are aligned
/// and equally spaced. For example, this sequence can be used to view only the
/// interleaved XY portion of an interleaved XYZM sequence.
template <typename Coord>
struct CoordSequence {
  /// \brief The C++ Coordinate type
  using value_type = Coord;

  /// \brief The C++ numeric type for ordinate storage
  using ordinate_type = typename value_type::value_type;

  /// \brief The number of values in each coordinate
  static constexpr uint32_t coord_size = Coord().size();

  /// \brief The offset into values to apply
  int64_t offset{};

  /// \brief The number of coordinates in the sequence
  int64_t length{};

  /// \brief Pointers to the first ordinate values in each dimension
  ///
  /// This structure can accommodate either interleaved or separated
  /// coordinates. For interleaved coordinates, these pointers will be
  /// contiguous; for separated coordinates these pointers will point
  /// to separate arrays. Each ordinate value is accessed by using the
  /// expression `values[dimension_id][(offset + coord_id) * stride]`.
  std::array<const ordinate_type*, coord_size> values{};

  /// \brief The distance (in elements) between sequential coordinates in
  /// each values array.
  ///
  /// For interleaved coordinates this is the number of dimensions in the
  /// input; for separated coordinates this is 1. This does not need to be
  /// equal to coord_size (e.g., when providing a CoordSequence<XY> view
  /// of an interleaved sequence of XYZM coordinates).
  int64_t stride{};

  /// \brief Initialize a dimension pointer for this array
  void InitValue(uint32_t i, const ordinate_type* value) { values[i] = value; }

  /// \brief Initialize from a GeoArrowCoordView
  GeoArrowErrorCode InitFrom(const struct GeoArrowCoordView* view) {
    if (static_cast<uint32_t>(view->n_values) < coord_size ||
        !std::is_same<ordinate_type, double>::value) {
      return EINVAL;
    }

    this->offset = 0;
    this->length = view->n_coords;
    this->stride = view->coords_stride;
    for (uint32_t i = 0; i < coord_size; i++) {
      this->InitValue(i, reinterpret_cast<const ordinate_type*>(view->values[i]));
    }
    return GEOARROW_OK;
  }

  /// \brief Initialize from a GeoArrowArrayView
  GeoArrowErrorCode InitFrom(const struct GeoArrowArrayView* view, int level = 0) {
    if (level != view->n_offsets) {
      return EINVAL;
    }

    GEOARROW_RETURN_NOT_OK(InitFrom(&view->coords));
    this->offset = view->offset[level];
    this->length = view->length[level];
    return GEOARROW_OK;
  }

  /// \brief Initialize an interleaved coordinate sequence from a pointer to its start
  GeoArrowErrorCode InitInterleaved(int64_t length_elements, const ordinate_type* data,
                                    int64_t stride_elements = coord_size) {
    if (data == nullptr && length_elements != 0) {
      return EINVAL;
    }

    stride = stride_elements;
    offset = 0;
    length = length_elements;
    if (data != nullptr) {
      for (uint32_t i = 0; i < coord_size; ++i) {
        InitValue(i, data + i);
      }
    }

    return GEOARROW_OK;
  }

  /// \brief Initialize a separated coordinate sequence from pointers to each
  /// dimension start
  GeoArrowErrorCode InitSeparated(int64_t length_elements,
                                  std::array<const ordinate_type*, coord_size> dimensions,
                                  int64_t stride_elements = 1) {
    this->offset = 0;
    this->length = length_elements;
    this->stride = stride_elements;
    for (uint32_t i = 0; i < dimensions.size(); i++) {
      this->InitValue(i, dimensions[i]);
    }

    return GEOARROW_OK;
  }

  /// \brief Return a coordinate at the given position
  Coord coord(int64_t i) const {
    Coord out;
    for (size_t j = 0; j < out.size(); j++) {
      out[j] = values[j][(offset + i) * stride];
    }
    return out;
  }

  /// \brief Return the number of coordinates in the sequence
  int64_t size() const { return length; }

  /// \brief Return a new coordinate sequence that is a subset of this one
  ///
  /// Caller is responsible for ensuring that offset + length is within the bounds
  /// of this sequence.
  CoordSequence<Coord> Slice(int64_t offset, int64_t length) const {
    CoordSequence<Coord> out = *this;
    out.offset += offset;
    out.length = length;
    return out;
  }

  /// \brief Call func once for each vertex in this sequence
  ///
  /// This function is templated on the desired coordinate output type.
  /// This allows, for example, iteration along all XYZM dimensions of an
  /// arbitrary sequence, even if some of those dimensions don't exist
  /// in the sequence. Similarly, one can iterate over fewer dimensions than
  /// are strictly in the output (discarding dimensions not of interest).
  template <typename CoordDst, typename Func>
  void VisitVertices(Func&& func) const {
    for (const auto vertex : *this) {
      func(CoordCast<Coord, CoordDst>(vertex));
    }
  }

  /// \brief Call func once for each sequential pair of vertices in this sequence
  ///
  /// This function is templated on the desired coordinate output type, performing
  /// the same coordinate conversion as VisitVertices. Note that sequential vertices
  /// may not be meaningful as edges for some types of sequences.
  template <typename CoordDst, typename Func>
  void VisitEdges(Func&& func) const {
    if (this->length < 2) {
      return;
    }

    auto it = begin();
    CoordDst start = CoordCast<Coord, CoordDst>(*it);
    ++it;
    while (it != end()) {
      CoordDst end = *it;
      func(start, end);
      start = end;
      ++it;
    }
  }

  using const_iterator = internal::CoordSequenceIterator<CoordSequence>;
  const_iterator begin() const { return const_iterator(*this, 0); }
  const_iterator end() const { return const_iterator(*this, length); }

  using dimension_iterator = internal::StridedIterator<ordinate_type>;
  dimension_iterator dbegin(uint32_t j) const {
    return dimension_iterator(values[j] + (offset * stride), stride);
  }
  dimension_iterator dend(uint32_t j) const {
    return dimension_iterator(values[j] + ((offset + length) * stride), stride);
  }
};

/// \brief View of an unaligned GeoArrow coordinate sequence
///
/// A view of zero or more coordinates. This data structure can handle either interleaved
/// or separated coordinates. This coordinate sequence type is intended to wrap
/// arbitrary bytes (e.g., WKB).
///
/// Like the CoordSequence, this sequence can handle structures beyond strictly
/// interleaved or separated coordinates. For example, the UnalignedCoordSequence can wrap
/// a sequence of contiguous WKB points (because the memory representing XY[Z[M]] values
/// are equally spaced, although the spacing is not an even multiple of the coordinate
/// size).
template <typename Coord,
          typename Load = internal::LoadIdentity<typename Coord::value_type>>
struct UnalignedCoordSequence {
  /// \brief The C++ Coordinate type
  using value_type = Coord;

  /// \brief The C++ numeric type for ordinate storage
  using ordinate_type = typename value_type::value_type;

  /// \brief The number of values in each coordinate
  static constexpr uint32_t coord_size = Coord().size();

  /// \brief The number of bytes in each coordinate
  static constexpr uint32_t coord_size_bytes = sizeof(Coord);

  /// \brief The offset into values to apply
  int64_t offset{};

  /// \brief The number of coordinates in the sequence
  int64_t length{};

  /// \brief Pointers to the first ordinate values in each dimension
  std::array<const uint8_t*, coord_size> values{};

  /// \brief The distance (in bytes) between sequential coordinates in
  /// each values array.
  int64_t stride_bytes{};

  /// \brief Initialize a dimension pointer for this array
  void InitValue(uint32_t i, const void* value) {
    values[i] = reinterpret_cast<const uint8_t*>(value);
  }

  /// \brief Initialize from a GeoArrowCoordView
  GeoArrowErrorCode InitFrom(struct GeoArrowCoordView* view) {
    if (view->n_values < coord_size || !std::is_same<ordinate_type, double>::value) {
      return EINVAL;
    }

    this->offset = 0;
    this->length = view->n_coords;
    this->stride_bytes = view->coords_stride * sizeof(ordinate_type);
    for (uint32_t i = 0; i < coord_size; i++) {
      this->InitValue(i, view->values[i]);
    }
    return GEOARROW_OK;
  }

  /// \brief Initialize from a GeoArrowArrayView
  GeoArrowErrorCode InitFrom(const struct GeoArrowArrayView* view, int level = 0) {
    if (level != view->n_offsets) {
      return EINVAL;
    }

    GEOARROW_RETURN_NOT_OK(InitFrom(&view->coords));
    this->offset = view->offset[level];
    this->length = view->length[level];
    return GEOARROW_OK;
  }

  /// \brief Initialize an interleaved coordinate sequence from a pointer to its start
  GeoArrowErrorCode InitInterleaved(int64_t length_elements, const void* data,
                                    int64_t stride_elements = coord_size) {
    if (data == nullptr && length_elements != 0) {
      return EINVAL;
    }

    this->stride_bytes = stride_elements * sizeof(ordinate_type);
    this->offset = 0;
    this->length = length_elements;
    if (data != nullptr) {
      for (uint32_t i = 0; i < coord_size; ++i) {
        this->InitValue(
            i, reinterpret_cast<const uint8_t*>(data) + i * sizeof(ordinate_type));
      }
    }

    return GEOARROW_OK;
  }

  /// \brief Initialize a separated coordinate sequence from pointers to each
  /// dimension start
  GeoArrowErrorCode InitSeparated(int64_t length_elements,
                                  std::array<const void*, coord_size> dimensions,
                                  int64_t stride_elements = 1) {
    this->offset = 0;
    this->length = length_elements;
    this->stride_bytes = stride_elements * sizeof(ordinate_type);
    for (uint32_t i = 0; i < dimensions.size(); i++) {
      this->InitValue(i, dimensions[i]);
    }

    return GEOARROW_OK;
  }

  /// \brief Return a coordinate at the given position
  Coord coord(int64_t i) const {
    Coord out;
    for (size_t j = 0; j < out.size(); j++) {
      out[j] = load_(values[j] + ((offset + i) * stride_bytes));
    }
    return out;
  }

  /// \brief Return the number of coordinates in the sequence
  int64_t size() const { return length; }

  /// \brief Return a new coordinate sequence that is a subset of this one
  ///
  /// Caller is responsible for ensuring that offset + length is within the bounds
  /// of this function.
  UnalignedCoordSequence<Coord> Slice(int64_t offset, int64_t length) const {
    UnalignedCoordSequence<Coord> out = *this;
    out.offset += offset;
    out.length = length;
    return out;
  }

  /// \brief Call func once for each vertex in this sequence
  ///
  /// This function is templated on the desired coordinate output type.
  /// This allows, for example, iteration along all XYZM dimensions of an
  /// arbitrary sequence, even if some of those dimensions don't exist
  /// in the sequence. Similarly, one can iterate over fewer dimensions than
  /// are strictly in the output (discarding dimensions not of interest).
  template <typename CoordDst, typename Func>
  void VisitVertices(Func&& func) const {
    for (const auto vertex : *this) {
      func(CoordCast<Coord, CoordDst>(vertex));
    }
  }

  /// \brief Call func once for each sequential pair of vertices in this sequence
  ///
  /// This function is templated on the desired coordinate output type, performing
  /// the same coordinate conversion as VisitVertices. Note that sequential vertices
  /// may not be meaningful as edges for some types of sequences.
  template <typename CoordDst, typename Func>
  void VisitEdges(Func&& func) const {
    if (this->length < 2) {
      return;
    }

    auto it = begin();
    CoordDst start = CoordCast<Coord, CoordDst>(*it);
    ++it;
    while (it != end()) {
      CoordDst end = CoordCast<Coord, CoordDst>(*it);
      func(start, end);
      start = end;
      ++it;
    }
  }

  using const_iterator = internal::CoordSequenceIterator<UnalignedCoordSequence>;
  const_iterator begin() const { return const_iterator(*this, 0); }
  const_iterator end() const { return const_iterator(*this, length); }

  using dimension_iterator =
      internal::UnalignedStridedIterator<typename value_type::value_type, Load>;
  dimension_iterator dbegin(uint32_t j) const {
    return dimension_iterator(values[j] + (offset * stride_bytes), stride_bytes);
  }
  dimension_iterator dend(uint32_t j) const {
    return dimension_iterator(values[j] + ((offset + length) * stride_bytes),
                              stride_bytes);
  }

 private:
  static constexpr Load load_{};
};

/// \brief View of a sequence of lists
template <typename T>
struct ListSequence {
  /// \brief The child view type (either a ListSequence or a CoordSequence)
  using child_type = T;

  /// \brief For the purposes of iteration, the value type is a const reference
  /// to the child type (stashed in the iterator).
  using value_type = const T&;

  /// \brief The logical offset into the sequence
  int64_t offset{};

  /// \brief The number of lists in the sequence
  int64_t length{};

  /// \brief The pointer to the first offset
  ///
  /// These offsets are sequential such that the offset of the ith element in the
  /// sequence begins at offsets[i]. This means there must be (offset + length + 1)
  /// accessible elements in offsets. This is exactly equal to the definition of the
  /// offsets in the Apache Arrow list type.
  const int32_t* offsets{};

  /// \brief The item from which slices are to be taken according to each offset pair
  ///
  /// Note that the child may have its own non-zero offset which must also be applied.
  T child{};

  /// \brief Initialize from a GeoArrowArrayView
  GeoArrowErrorCode InitFrom(const struct GeoArrowArrayView* view, int level = 0) {
    if (level > (view->n_offsets - 1)) {
      return EINVAL;
    }

    this->offsets = view->offsets[level];
    GEOARROW_RETURN_NOT_OK(this->child.InitFrom(view, level + 1));
    this->offset = view->offset[level];
    this->length = view->length[level];
    return GEOARROW_OK;
  }

  /// \brief Slice the child based on this sequence's offset/length
  T ValidChildElements() const {
    if (length == 0) {
      return child.Slice(0, 0);
    } else {
      int64_t first_offset = offsets[offset];
      int64_t last_offset = offsets[offset + length];
      return child.Slice(first_offset, last_offset - first_offset);
    }
  }

  /// \brief Return the number of elements in the sequence
  int64_t size() const { return length; }

  /// \brief Initialize a child whose offset and length are unset.
  void InitChild(T* child_p) const { *child_p = child; }

  /// \brief Update a child initialized with InitChild such that it represents the
  /// ith element of the array.
  void UpdateChild(T* child_p, int64_t i) const {
    int32_t child_offset = offsets[offset + i];
    child_p->offset = child.offset + child_offset;
    child_p->length = offsets[offset + i + 1] - child_offset;
  }

  /// \brief Return a new coordinate sequence that is a subset of this one
  ///
  /// Caller is responsible for ensuring that offset + length is within the bounds
  /// of this function.
  ListSequence<T> Slice(int64_t offset, int64_t length) const {
    ListSequence<T> out = *this;
    out.offset += offset;
    out.length = length;
    return out;
  }

  /// \brief Call func once for each vertex in the child sequence
  template <typename CoordDst, typename Func>
  void VisitVertices(Func&& func) const {
    // Vertices can always just use the child array
    ValidChildElements().template VisitVertices<CoordDst>(func);
  }

  /// \brief Call func once for each edge in each child sequence
  template <typename CoordDst, typename Func>
  void VisitEdges(Func&& func) const {
    // Edges need to treat each child element separately
    for (const auto& item : *this) {
      item.template VisitEdges<CoordDst>(func);
    }
  }

  using const_iterator = internal::ListSequenceIterator<ListSequence>;
  const_iterator begin() const { return const_iterator(*this, 0); }
  const_iterator end() const { return const_iterator(*this, length); }
};

/// \brief View of a sequence of blobs
template <typename Offset>
struct BinarySequence {
  /// \brief The value type of this sequence
  using value_type = GeoArrowBufferView;

  /// \brief The logical offset into the sequence
  int64_t offset{};

  /// \brief The number of blobs in the sequence
  int64_t length{};

  /// \brief The pointer to the first offset
  ///
  /// These offsets are sequential such that the offset of the ith element in the
  /// sequence begins at offsets[i]. This means there must be (offset + length + 1)
  /// accessible elements in offsets. This is exactly equal to the definition of the
  /// offsets in the Apache Arrow binary and utf8 types.
  const Offset* offsets{};

  /// \brief The pointer to the contiguous data buffer
  const uint8_t* data{};

  /// \brief Initialize from a GeoArrowArrayView
  GeoArrowErrorCode InitFrom(const struct GeoArrowArrayView* view) {
    this->offsets = view->offsets[0];
    this->offset = view->offset[0];
    this->length = view->length[0];
    this->data = view->data;
    return GEOARROW_OK;
  }

  value_type blob(int64_t i) const {
    Offset element_begin = offsets[offset + i];
    Offset element_end = offsets[offset + i + 1];
    return {data + element_begin, element_end - element_begin};
  }

  using const_iterator = internal::BinarySequenceIterator<BinarySequence>;
  const_iterator begin() const { return const_iterator(*this, 0); }
  const_iterator end() const { return const_iterator(*this, length); }
};

/// \brief A nullable sequence (either a ListSequence or a CoordSequence)
///
/// Unlike the ListSequence and CoordSequence types, elements may be nullable.
template <typename T>
struct Array {
  /// \brief The sequence type (either a ListSequence or a CoordSequence)
  using sequence_type = T;

  /// \brief An instance of the ListSequence or CoordSequence
  T value{};

  /// \brief A validity bitmap where a set bit indicates a non-null value
  /// and an unset bit indicates a null value.
  ///
  /// The pointer itself may be (C++) nullptr, which indicates that all values
  /// in the array are non-null. Bits use least-significant bit ordering such that
  /// the validity of the ith element in the array is calculated with the expression
  /// `validity[i / 8] & (1 << (i % 8))`. This is exactly equal to the definition of
  /// the validity bitmap in the Apache Arrow specification.
  ///
  /// Note that the offset of the underlying sequence must be applied.
  const uint8_t* validity{};

  /// \brief Return the validity of a given element
  ///
  /// Note that this is not an efficient mechanism to check for nullability in a loop.
  bool is_valid(int64_t i) const {
    i += value.offset;
    return validity == nullptr || validity[i / 8] & (1 << (i % 8));
  }

  /// \brief Return the nullness of a given element
  ///
  /// Note that this is not an efficient mechanism to check for nullability in a loop.
  bool is_null(int64_t i) const {
    i += value.offset;
    return validity != nullptr && !(validity[i / 8] & (1 << (i % 8)));
  }

  /// \brief Call func once for each vertex in this array (excluding null elements)
  template <typename CoordDst, typename Func>
  void VisitVertices(Func&& func) const {
    if (this->validity) {
      // TODO: optimize the nullable case
      auto it = this->value.begin();
      int64_t i = 0;
      while (it != this->value.end()) {
        if (this->is_valid(i)) {
          const auto& item = *it;
          item.template VisitVertices<CoordDst>(func);
        }
        ++i;
        ++it;
      }
    } else {
      this->value.template VisitVertices<CoordDst>(func);
    }
  }

  /// \brief Call func once for each edge in this array (excluding null elements)
  ///
  /// For the purposes of this function, points are considered degenerate edges.
  template <typename CoordDst, typename Func>
  void VisitEdges(Func&& func) const {
    if (this->validity) {
      // TODO: optimize the nullable case
      auto it = this->value.begin();
      int64_t i = 0;
      while (it != this->value.end()) {
        if (this->is_valid(i)) {
          const auto& item = *it;
          item.template VisitEdges<CoordDst>(func);
        }
        ++i;
        ++it;
      }
    } else {
      this->value.template VisitEdges<CoordDst>(func);
    }
  }

  /// \brief Initialize an Array from a GeoArrowArrayView
  ///
  /// Returns EINVAL if the nesting levels and/or coordinate size
  /// is incompatible with the values in the view.
  GeoArrowErrorCode Init(const struct GeoArrowArrayView* view) {
    GEOARROW_RETURN_NOT_OK(value.InitFrom(view));
    validity = view->validity_bitmap;
    return GEOARROW_OK;
  }

 protected:
  template <typename Impl>
  Impl SliceImpl(Impl self, int64_t offset, int64_t length) {
    self.value.offset += offset;
    self.value.length = length;
    return self;
  }
};

/// \brief An Array of points
template <typename Coord>
struct PointArray : public Array<CoordSequence<Coord>> {
  static constexpr enum GeoArrowGeometryType geometry_type = GEOARROW_GEOMETRY_TYPE_POINT;
  static constexpr enum GeoArrowDimensions dimensions = Coord::dimensions;

  /// \brief Return a view of all coordinates in this array
  ///
  /// Note that in the presence of null values, some of the coordinates values
  /// are not present in the array (e.g., for the purposes of calculating aggregate
  /// statistics).
  CoordSequence<Coord> Coords() const { return this->value; }

  /// \brief Return a new array that is a subset of this one
  ///
  /// Caller is responsible for ensuring that offset + length is within the bounds
  /// of this array.
  PointArray Slice(int64_t offset, int64_t length) {
    return this->template SliceImpl<PointArray>(*this, offset, length);
  }

  /// \brief Call func once for each edge in this array (excluding null elements)
  ///
  /// For the purposes of this function, points are considered degenerate edges.
  template <typename CoordDst, typename Func>
  void VisitEdges(Func&& func) const {
    this->template VisitVertices<CoordDst>([&](Coord coord) { func(coord, coord); });
  }
};

/// \brief An Array of boxes
template <typename Coord>
struct BoxArray : public Array<CoordSequence<typename Coord::box_type>> {
  static constexpr enum GeoArrowGeometryType geometry_type = GEOARROW_GEOMETRY_TYPE_BOX;
  static constexpr enum GeoArrowDimensions dimensions = Coord::dimensions;

  /// \brief Return the xmin/ymin/zmin/xmin tuple from this box array
  PointArray<Coord> LowerBound() {
    PointArray<Coord> out;
    out.validity = this->validity;
    out.value.stride = this->value.stride;
    out.value.offset = this->value.offset;
    out.value.length = this->value.length;
    for (size_t i = 0; i < Coord().size(); i++) {
      out.value.values[i] = this->value.values[i];
    }

    return out;
  }

  /// \brief Return the xmax/ymax/zmax/xmax tuple from this box array
  PointArray<Coord> UpperBound() {
    PointArray<Coord> out;
    out.validity = this->validity;
    out.value.stride = this->value.stride;
    out.value.offset = this->value.offset;
    out.value.length = this->value.length;
    for (size_t i = 0; i < Coord().size(); i++) {
      out.value.values[i] = this->value.values[Coord().size() + i];
    }

    return out;
  }

  /// \brief Return a new array that is a subset of this one
  ///
  /// Caller is responsible for ensuring that offset + length is within the bounds
  /// of this array.
  BoxArray Slice(int64_t offset, int64_t length) {
    return this->template SliceImpl<BoxArray>(*this, offset, length);
  }
};

/// \brief An Array of linestrings
template <typename Coord>
struct LinestringArray : public Array<ListSequence<CoordSequence<Coord>>> {
  static constexpr enum GeoArrowGeometryType geometry_type =
      GEOARROW_GEOMETRY_TYPE_LINESTRING;
  static constexpr enum GeoArrowDimensions dimensions = Coord::dimensions;

  /// \brief Return a view of all coordinates in this array
  ///
  /// Note that in the presence of null values, some of the coordinates values
  /// are not present in the array (e.g., for the purposes of calculating aggregate
  /// statistics).
  CoordSequence<Coord> Coords() const { return this->value.ValidChildElements(); }

  /// \brief Return a new array that is a subset of this one
  ///
  /// Caller is responsible for ensuring that offset + length is within the bounds
  /// of this array.
  LinestringArray Slice(int64_t offset, int64_t length) {
    return this->template SliceImpl<LinestringArray>(*this, offset, length);
  }
};

/// \brief An Array of polygons
template <typename Coord>
struct PolygonArray : public Array<ListSequence<ListSequence<CoordSequence<Coord>>>> {
  static constexpr enum GeoArrowGeometryType geometry_type =
      GEOARROW_GEOMETRY_TYPE_POLYGON;
  static constexpr enum GeoArrowDimensions dimensions = Coord::dimensions;

  /// \brief Return a view of all coordinates in this array
  ///
  /// Note that in the presence of null values, some of the coordinates values
  /// are not present in the array (e.g., for the purposes of calculating aggregate
  /// statistics).
  CoordSequence<Coord> Coords() const {
    return this->value.ValidChildElements().ValidChildElements();
  }

  /// \brief Return a new array that is a subset of this one
  ///
  /// Caller is responsible for ensuring that offset + length is within the bounds
  /// of this array.
  PolygonArray Slice(int64_t offset, int64_t length) {
    return this->template SliceImpl<PolygonArray>(*this, offset, length);
  }
};

/// \brief An Array of multipoints
template <typename Coord>
struct MultipointArray : public Array<ListSequence<CoordSequence<Coord>>> {
  static constexpr enum GeoArrowGeometryType geometry_type =
      GEOARROW_GEOMETRY_TYPE_MULTIPOINT;
  static constexpr enum GeoArrowDimensions dimensions = Coord::dimensions;

  /// \brief Return a view of all coordinates in this array
  ///
  /// Note that in the presence of null values, some of the coordinates values
  /// are not present in the array (e.g., for the purposes of calculating aggregate
  /// statistics).
  CoordSequence<Coord> Coords() const { return this->value.ValidChildElements(); }

  /// \brief Return a new array that is a subset of this one
  ///
  /// Caller is responsible for ensuring that offset + length is within the bounds
  /// of this array.
  MultipointArray Slice(int64_t offset, int64_t length) {
    return this->template SliceImpl<MultipointArray>(*this, offset, length);
  }

  /// \brief Call func once for each edge in this array (excluding null elements)
  ///
  /// For the purposes of this function, points are considered degenerate edges.
  template <typename CoordDst, typename Func>
  void VisitEdges(Func&& func) const {
    this->template VisitVertices<CoordDst>([&](Coord coord) { func(coord, coord); });
  }
};

/// \brief An Array of multilinestrings
template <typename Coord>
struct MultiLinestringArray
    : public Array<ListSequence<ListSequence<CoordSequence<Coord>>>> {
  static constexpr enum GeoArrowGeometryType geometry_type =
      GEOARROW_GEOMETRY_TYPE_MULTILINESTRING;
  static constexpr enum GeoArrowDimensions dimensions = Coord::dimensions;

  /// \brief Return a view of all coordinates in this array
  ///
  /// Note that in the presence of null values, some of the coordinates values
  /// are not present in the array (e.g., for the purposes of calculating aggregate
  /// statistics).
  CoordSequence<Coord> Coords() const {
    return this->value.ValidChildElements().ValidChildElements();
  }

  /// \brief Return a new array that is a subset of this one
  ///
  /// Caller is responsible for ensuring that offset + length is within the bounds
  /// of this array.
  MultiLinestringArray Slice(int64_t offset, int64_t length) {
    return this->template SliceImpl<MultiLinestringArray>(*this, offset, length);
  }
};

/// \brief An Array of multipolygons
template <typename Coord>
struct MultiPolygonArray
    : public Array<ListSequence<ListSequence<ListSequence<CoordSequence<Coord>>>>> {
  static constexpr enum GeoArrowGeometryType geometry_type =
      GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON;
  static constexpr enum GeoArrowDimensions dimensions = Coord::dimensions;

  /// \brief Return a view of all coordinates in this array
  ///
  /// Note that in the presence of null values, some of the coordinates values
  /// are not present in the array (e.g., for the purposes of calculating aggregate
  /// statistics).
  CoordSequence<Coord> Coords() const {
    return this->value.ValidChildElements().ValidChildElements().ValidChildElements();
  }

  /// \brief Return a new array that is a subset of this one
  ///
  /// Caller is responsible for ensuring that offset + length is within the bounds
  /// of this array.
  MultiPolygonArray Slice(int64_t offset, int64_t length) {
    return this->template SliceImpl<MultiPolygonArray>(*this, offset, length);
  }
};

}  // namespace array_util
}  // namespace geoarrow

/// @}

#endif

#ifndef GEOARROW_HPP_WKB_UTIL_INCLUDED
#define GEOARROW_HPP_WKB_UTIL_INCLUDED

#include <vector>




/// \defgroup hpp-binary-utility Binary iteration utilities
///
/// This header provides utilities for iterating over serialized GeoArrow arrays
/// (e.g., WKB).
///
/// @{

namespace geoarrow {

namespace wkb_util {

namespace internal {

static constexpr uint8_t kBigEndian = 0x00;
static constexpr uint8_t kLittleEndian = 0x01;

#if GEOARROW_NATIVE_ENDIAN == 0x00
static constexpr uint8_t kNativeEndian = kBigEndian;
static constexpr uint8_t kSwappedEndian = kLittleEndian;
#else
static constexpr uint8_t kNativeEndian = kLittleEndian;
static constexpr uint8_t kSwappedEndian = kBigEndian;

#endif

template <uint8_t endian>
struct Endian;

template <>
struct Endian<kNativeEndian> {
  using LoadUInt32 = array_util::internal::LoadIdentity<uint32_t>;
  using LoadDouble = array_util::internal::LoadIdentity<double>;
};

template <>
struct Endian<kSwappedEndian> {
  using LoadUInt32 = array_util::internal::LoadSwapped<uint32_t>;
  using LoadDouble = array_util::internal::LoadSwapped<double>;
};

}  // namespace internal

/// \brief Location and structure of a coordinate sequence within a WKB blob
struct WKBSequence {
  const uint8_t* data{};
  uint32_t length{};
  uint32_t stride{};
  enum GeoArrowDimensions dimensions {};
  uint8_t endianness;

  /// \brief The number of coordinates in the sequence
  uint32_t size() const { return length; }

  using NativeXYSequence = array_util::UnalignedCoordSequence<array_util::XY<double>>;

  /// \brief For native-endian sequences, return an XY sequence directly
  ///
  /// This may be faster or more convenient than using a visitor-based approach
  /// when only the XY dimensions are needed.
  NativeXYSequence ViewAsNativeXY() const {
    NativeXYSequence seq;
    seq.InitInterleaved(length, data, stride);
    return seq;
  }

  /// \brief Call func once for each vertex in this sequence
  ///
  /// This function is templated on the desired coordinate output type.
  /// This allows, for example, iteration along all XYZM dimensions of an
  /// arbitrary sequence, even if some of those dimensions don't exist
  /// in the sequence. Similarly, one can iterate over fewer dimensions than
  /// are strictly in the output (discarding dimensions not of interest).
  template <typename CoordDst, typename Func>
  void VisitVertices(Func&& f) const {
    switch (dimensions) {
      case GEOARROW_DIMENSIONS_XY:
        VisitVerticesInternal<array_util::XY<double>, CoordDst>(f);
        break;
      case GEOARROW_DIMENSIONS_XYZ:
        VisitVerticesInternal<array_util::XYZ<double>, CoordDst>(f);
        break;
      case GEOARROW_DIMENSIONS_XYM:
        VisitVerticesInternal<array_util::XYM<double>, CoordDst>(f);
        break;
      case GEOARROW_DIMENSIONS_XYZM:
        VisitVerticesInternal<array_util::XYZM<double>, CoordDst>(f);
        break;
      default:
        throw Exception("Unknown dimensions");
    }
  }

  /// \brief Call func once for each sequential pair of vertices in this sequence
  ///
  /// This function is templated on the desired coordinate output type, performing
  /// the same coordinate conversion as VisitVertices. Note that sequential vertices
  /// may not be meaningful as edges for some types of sequences.
  template <typename CoordDst, typename Func>
  void VisitEdges(Func&& f) const {
    switch (dimensions) {
      case GEOARROW_DIMENSIONS_XY:
        VisitEdgesInternal<array_util::XY<double>, CoordDst>(f);
        break;
      case GEOARROW_DIMENSIONS_XYZ:
        VisitEdgesInternal<array_util::XYZ<double>, CoordDst>(f);
        break;
      case GEOARROW_DIMENSIONS_XYM:
        VisitEdgesInternal<array_util::XYM<double>, CoordDst>(f);
        break;
      case GEOARROW_DIMENSIONS_XYZM:
        VisitEdgesInternal<array_util::XYZM<double>, CoordDst>(f);
        break;
      default:
        throw Exception("Unknown dimensions");
    }
  }

 private:
  template <typename CoordSrc, typename CoordDst, typename Func>
  void VisitVerticesInternal(Func&& f) const {
    if (endianness == internal::kLittleEndian) {
      using LoadDouble = internal::Endian<internal::kLittleEndian>::LoadDouble;
      using Sequence = array_util::UnalignedCoordSequence<CoordSrc, LoadDouble>;
      Sequence seq;
      seq.InitInterleaved(length, data, stride);
      seq.template VisitVertices<CoordDst>(f);
    } else {
      using LoadDouble = internal::Endian<internal::kBigEndian>::LoadDouble;
      using Sequence = array_util::UnalignedCoordSequence<CoordSrc, LoadDouble>;
      Sequence seq;
      seq.InitInterleaved(length, data, stride);
      seq.template VisitVertices<CoordDst>(f);
    }
  }

  template <typename CoordSrc, typename CoordDst, typename Func>
  void VisitEdgesInternal(Func&& f) const {
    if (endianness == internal::kLittleEndian) {
      using LoadDouble = internal::Endian<internal::kLittleEndian>::LoadDouble;
      using Sequence = array_util::UnalignedCoordSequence<CoordSrc, LoadDouble>;
      Sequence seq;
      seq.InitInterleaved(length, data, stride);
      seq.template VisitEdges<CoordDst>(f);
    } else {
      using LoadDouble = internal::Endian<internal::kBigEndian>::LoadDouble;
      using Sequence = array_util::UnalignedCoordSequence<CoordSrc, LoadDouble>;
      Sequence seq;
      seq.InitInterleaved(length, data, stride);
      seq.template VisitEdges<CoordDst>(f);
    }
  }
};

/// \brief Tokenized WKB Geometry
///
/// The result of parsing a well-known binary blob. Geometries are represented as:
///
/// - Point: A single sequence that contains exactly one point.
/// - Linestring: A single sequence.
/// - Polygon: Each ring's sequence is stored in sequences.
/// - Multipoint, Multilinestring, Multipolygon, Geometrycollection: Each child feature
//    is given its own WKBGeometry.
///
/// The motivation for this structure is to ensure that, given a single stack-allocated
/// WKBGeometry, many well-known binary blobs can be parsed and efficiently reuse the
/// heap-allocated vectors. Because of this, care is taken to private manage the
/// storage for WKBSequence and child WKBGeometry objects to ensure we don't call any
/// child deleters when a Reset() is requested and a new geometry is to be parsed.
///
/// This class can also be used to represent serialized geometries that are not WKB
/// but follow the same structure (e.g., gserialized, DuckDB GEOMETRY).
class WKBGeometry {
 public:
  static constexpr uint32_t kSridUnset = 0xFFFFFFFF;

  enum GeoArrowGeometryType geometry_type {};
  enum GeoArrowDimensions dimensions {};
  uint32_t srid{kSridUnset};

  const WKBSequence& Sequence(uint32_t i) const { return sequences_[i]; }

  uint32_t NumSequences() const { return num_sequences_; }

  const WKBGeometry& Geometry(uint32_t i) const { return geometries_[i]; }

  uint32_t NumGeometries() const { return num_geometries_; }

  /// \brief Call func once for each vertex in this geometry
  template <typename CoordDst, typename Func>
  void VisitVertices(Func&& func) const {
    for (uint32_t i = 0; i < NumSequences(); i++) {
      Sequence(i).VisitVertices<CoordDst>(func);
    }

    for (uint32_t i = 0; i < NumGeometries(); i++) {
      Geometry(i).VisitVertices<CoordDst>(func);
    }
  }

  /// \brief Call func once for each edge in this geometry
  ///
  /// For the purposes of this function, points are considered degenerate edges.
  template <typename CoordDst, typename Func>
  void VisitEdges(Func&& func) const {
    switch (geometry_type) {
      case GEOARROW_GEOMETRY_TYPE_POINT:
      case GEOARROW_GEOMETRY_TYPE_MULTIPOINT:
        VisitVertices<CoordDst>([&](CoordDst v0) { func(v0, v0); });
        return;
      default:
        for (uint32_t i = 0; i < NumSequences(); i++) {
          Sequence(i).VisitEdges<CoordDst>(func);
        }
    }

    for (uint32_t i = 0; i < NumGeometries(); i++) {
      Geometry(i).VisitEdges<CoordDst>(func);
    }
  }

  /// \brief Append a new sequence to this WKBGeometry and return a pointer to it
  WKBSequence* AppendSequence() {
    ++num_sequences_;
    if (sequences_.size() < num_sequences_) {
      sequences_.push_back({});
    }
    return &sequences_[num_sequences_ - 1];
  }

  WKBGeometry* AppendGeometry() {
    ++num_geometries_;
    if (geometries_.size() < num_geometries_) {
      geometries_.push_back({});
    }
    return &geometries_[num_geometries_ - 1];
  }

  void Reset() {
    geometry_type = GEOARROW_GEOMETRY_TYPE_GEOMETRY;
    dimensions = GEOARROW_DIMENSIONS_UNKNOWN;
    srid = kSridUnset;
    num_sequences_ = 0;
    num_geometries_ = 0;
  }

 private:
  // Vectors managed privately to ensure that all the stack-allocated memory associated
  // with the geometries is not freed when the vector is resized to zero.
  std::vector<WKBSequence> sequences_;
  uint32_t num_sequences_;
  std::vector<WKBGeometry> geometries_;
  uint32_t num_geometries_;
};

/// \brief Parse Well-known binary blobs
class WKBParser {
 public:
  /// \brief Outcomes of parsing a WKB blob
  enum Status {
    /// \brief Success!
    OK = 0,
    /// \brief The provided size was not sufficient to complete parsing
    TOO_FEW_BYTES,
    /// \brief Parse was successful, but the provided size was larger than the parsed WKB
    TOO_MANY_BYTES,
    /// \brief An endian value other than 0x00 or 0x01 was encountered (e.g., corrupted
    /// data)
    INVALID_ENDIAN,
    /// \brief An unexpected geometry type value was encountered (e.g., corrupted data or
    /// curved/complex geometry)
    INVALID_GEOMETRY_TYPE
  };

  WKBParser() = default;

  /// \brief Parse a GeoArrowBufferView into out, placing the end of the sequence in
  /// cursor
  Status Parse(struct GeoArrowBufferView data, WKBGeometry* out,
               const uint8_t** cursor = nullptr) {
    return Parse(data.data, static_cast<uint32_t>(data.size_bytes), out, cursor);
  }

  /// \brief Parse the specified bytes into out, placing the end of the sequence in
  /// focursor
  Status Parse(const uint8_t* data, size_t size, WKBGeometry* out,
               const uint8_t** cursor = nullptr) {
    data_ = cursor_ = data;
    remaining_ = size;
    Status status = ParseGeometry(out);
    if (status != OK) {
      return status;
    }

    if (cursor != nullptr) {
      *cursor = cursor_;
    }

    if (static_cast<size_t>(cursor_ - data_) < size) {
      return TOO_MANY_BYTES;
    }

    return OK;
  }

  std::string ErrorToString(Status status) {
    switch (status) {
      case OK:
        return "OK";
      case TOO_FEW_BYTES:
        return "TOO_FEW_BYTES";
      case TOO_MANY_BYTES:
        return "TOO_MANY_BYTES";
      case INVALID_ENDIAN:
        return "INVALID_ENDIAN";
      case INVALID_GEOMETRY_TYPE:
        return "INVALID_GEOMETRY_TYPE";
      default:
        return "UNKNOWN";
    }
  }

 private:
  const uint8_t* data_{};
  const uint8_t* cursor_{};
  size_t remaining_{};
  uint8_t last_endian_;
  enum GeoArrowDimensions last_dimensions_;
  uint32_t last_coord_stride_;
  internal::Endian<internal::kBigEndian>::LoadUInt32 load_uint32_be_;
  internal::Endian<internal::kLittleEndian>::LoadUInt32 load_uint32_le_;

  static constexpr uint32_t kEWKBZ = 0x80000000;
  static constexpr uint32_t kEWKBM = 0x40000000;
  static constexpr uint32_t kEWKBSrid = 0x20000000;
  static constexpr uint32_t kEWKBMask = 0x00FFFFFF;

  Status ParseGeometry(WKBGeometry* out) {
    out->Reset();

    Status status = CheckRemaining(sizeof(uint8_t) + sizeof(uint32_t));
    if (status != OK) {
      return status;
    }

    status = ReadEndianUnchecked();
    if (status != OK) {
      return status;
    }

    uint32_t geometry_type = ReadUInt32Unchecked();

    if (geometry_type & kEWKBSrid) {
      status = CheckRemaining(sizeof(uint32_t));
      if (status != OK) {
        return status;
      }

      out->srid = ReadUInt32Unchecked();
    }

    bool has_z = geometry_type & kEWKBZ;
    bool has_m = geometry_type & kEWKBM;
    geometry_type &= kEWKBMask;

    switch (geometry_type / 1000) {
      case 1:
        has_z = true;
        break;
      case 2:
        has_m = true;
        break;
      case 3:
        has_z = true;
        has_m = true;
        break;
    }

    last_coord_stride_ = 2 + has_z + has_m;
    if (has_z && has_m) {
      last_dimensions_ = out->dimensions = GEOARROW_DIMENSIONS_XYZM;
    } else if (has_z) {
      last_dimensions_ = out->dimensions = GEOARROW_DIMENSIONS_XYZ;
    } else if (has_m) {
      last_dimensions_ = out->dimensions = GEOARROW_DIMENSIONS_XYM;
    } else {
      last_dimensions_ = out->dimensions = GEOARROW_DIMENSIONS_XY;
    }

    geometry_type = geometry_type % 1000;
    switch (geometry_type) {
      case GEOARROW_GEOMETRY_TYPE_POINT:
        out->geometry_type = static_cast<enum GeoArrowGeometryType>(geometry_type);
        return ParsePoint(out);
      case GEOARROW_GEOMETRY_TYPE_LINESTRING:
        out->geometry_type = static_cast<enum GeoArrowGeometryType>(geometry_type);
        return ParseSequence(out);
      case GEOARROW_GEOMETRY_TYPE_POLYGON:
        out->geometry_type = static_cast<enum GeoArrowGeometryType>(geometry_type);
        return ParsePolygon(out);
      case GEOARROW_GEOMETRY_TYPE_MULTIPOINT:
      case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING:
      case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON:
      case GEOARROW_GEOMETRY_TYPE_GEOMETRYCOLLECTION:
        out->geometry_type = static_cast<enum GeoArrowGeometryType>(geometry_type);
        return ParseCollection(out);
      default:
        return INVALID_GEOMETRY_TYPE;
    }
  }

  Status ParsePoint(WKBGeometry* out) {
    uint32_t bytes_required = 1 * last_coord_stride_ * sizeof(double);
    Status status = CheckRemaining(bytes_required);
    if (status != OK) {
      return status;
    }

    NewSequenceAtCursor(out, 1);
    Advance(bytes_required);
    return OK;
  }

  Status ParseSequence(WKBGeometry* out) {
    Status status = CheckRemaining(sizeof(uint32_t));
    if (status != OK) {
      return status;
    }

    uint32_t size = ReadUInt32Unchecked();
    size_t bytes_required = sizeof(double) * size * last_coord_stride_;
    status = CheckRemaining(bytes_required);
    if (status != OK) {
      return status;
    }

    NewSequenceAtCursor(out, size);
    Advance(bytes_required);
    return OK;
  }

  Status ParsePolygon(WKBGeometry* out) {
    Status status = CheckRemaining(sizeof(uint32_t));
    if (status != OK) {
      return status;
    }

    uint32_t size = ReadUInt32Unchecked();
    for (uint32_t i = 0; i < size; ++i) {
      status = ParseSequence(out);
      if (status != OK) {
        return status;
      }
    }

    return OK;
  }

  Status ParseCollection(WKBGeometry* out) {
    Status status = CheckRemaining(sizeof(uint32_t));
    if (status != OK) {
      return status;
    }

    uint32_t size = ReadUInt32Unchecked();
    for (uint32_t i = 0; i < size; ++i) {
      status = ParseGeometry(out->AppendGeometry());
      if (status != OK) {
        return status;
      }
    }

    return OK;
  }

  uint32_t NewSequenceAtCursor(WKBGeometry* out, uint32_t size) {
    WKBSequence* seq = out->AppendSequence();
    seq->length = size;
    seq->data = cursor_;
    seq->dimensions = last_dimensions_;
    seq->stride = last_coord_stride_;
    seq->endianness = last_endian_;
    return size * last_coord_stride_ * sizeof(double);
  }

  Status CheckRemaining(size_t bytes) {
    if (bytes <= remaining_) {
      return OK;
    } else {
      return TOO_FEW_BYTES;
    }
  }

  Status ReadEndianUnchecked() {
    last_endian_ = *cursor_;
    switch (last_endian_) {
      case internal::kLittleEndian:
      case internal::kBigEndian:
        Advance(sizeof(uint8_t));
        return OK;
      default:
        return INVALID_ENDIAN;
    }
  }

  uint32_t ReadUInt32Unchecked() {
    uint32_t out;
    if (last_endian_ == internal::kLittleEndian) {
      out = load_uint32_le_(cursor_);
    } else {
      out = load_uint32_be_(cursor_);
    }

    Advance(sizeof(uint32_t));
    return out;
  }

  void Advance(size_t bytes) {
    cursor_ += bytes;
    remaining_ -= bytes;
  }
};

namespace internal {

template <typename WKBSequence>
class WKBSequenceIterator
    : public array_util::internal::BaseRandomAccessIterator<WKBSequence> {
 public:
  explicit WKBSequenceIterator(const WKBSequence& outer, int64_t i)
      : array_util::internal::BaseRandomAccessIterator<WKBSequence>(outer, i) {}

  using iterator_category = std::random_access_iterator_tag;
  using difference_type = int64_t;
  using value_type = const WKBGeometry&;

  value_type operator*() {
    WKBParser::Status status = parser_.Parse(this->outer_.blob(this->i_), &stashed_);
    if (status != WKBParser::OK) {
      throw Exception("Failed to parse WKB at index " + std::to_string(this->i_) +
                      " with error " + parser_.ErrorToString(status));
    }

    return stashed_;
  }

 private:
  WKBGeometry stashed_;
  WKBParser parser_;
};

}  // namespace internal

template <typename Offset>
struct WKBBlobSequence : public array_util::BinarySequence<Offset> {
  using value_type = const WKBGeometry&;

  template <typename CoordDst, typename Func>
  void VisitVertices(Func&& func) const {
    for (const auto& item : *this) {
      item.template VisitVertices<CoordDst>(func);
    }
  }

  template <typename CoordDst, typename Func>
  void VisitEdges(Func&& func) const {
    for (const auto& item : *this) {
      item.template VisitEdges<CoordDst>(func);
    }
  }

  using const_iterator = internal::WKBSequenceIterator<WKBBlobSequence>;
  const_iterator begin() const { return const_iterator(*this, 0); }
  const_iterator end() const { return const_iterator(*this, this->length); }
};

/// \brief An Array of WKB blobs
template <typename Offset>
struct WKBArray : public array_util::Array<WKBBlobSequence<Offset>> {
  static constexpr enum GeoArrowGeometryType geometry_type =
      GEOARROW_GEOMETRY_TYPE_GEOMETRY;
  static constexpr enum GeoArrowDimensions dimensions = GEOARROW_DIMENSIONS_UNKNOWN;

  /// \brief Return a new array that is a subset of this one
  ///
  /// Caller is responsible for ensuring that offset + length is within the bounds
  /// of this array.
  WKBArray Slice(int64_t offset, int64_t length) {
    return this->template SliceImpl<WKBArray>(*this, offset, length);
  }
};

}  // namespace wkb_util

}  // namespace geoarrow

/// @}

#endif
#ifndef GEOARROW_HPP_TYPE_TRAITS_INCLUDED
#define GEOARROW_HPP_TYPE_TRAITS_INCLUDED




namespace geoarrow {

namespace type_traits {

namespace internal {

/// \brief A template to resolve the coordinate type based on a dimensions constant
///
/// This may change in the future if more coordinate storage options are added
/// (e.g., float) or if the coordinate sequence is updated such that the Coord level
/// of templating is no longer needed.
template <enum GeoArrowDimensions dimensions>
struct DimensionTraits;

template <>
struct DimensionTraits<GEOARROW_DIMENSIONS_XY> {
  using coord_type = array_util::XY<double>;
};

template <>
struct DimensionTraits<GEOARROW_DIMENSIONS_XYZ> {
  using coord_type = array_util::XYZ<double>;
};

template <>
struct DimensionTraits<GEOARROW_DIMENSIONS_XYM> {
  using coord_type = array_util::XYM<double>;
};

template <>
struct DimensionTraits<GEOARROW_DIMENSIONS_XYZM> {
  using coord_type = array_util::XYZM<double>;
};

template <enum GeoArrowGeometryType geometry_type, enum GeoArrowDimensions dimensions>
struct ResolveArrayType;

// GCC Won't compile template specializations within a template, so we stamp this
// out using a macro (there may be a better way).
#define _GEOARROW_SPECIALIZE_ARRAY_TYPE(geometry_type, dimensions, array_cls) \
  template <>                                                                 \
  struct ResolveArrayType<geometry_type, dimensions> {                        \
    using coord_type = typename DimensionTraits<dimensions>::coord_type;      \
    using array_type = array_util::array_cls<coord_type>;                     \
  }

#define _GEOARROW_SPECIALIZE_GEOMETRY_TYPE(geometry_type, array_cls)                  \
  _GEOARROW_SPECIALIZE_ARRAY_TYPE(geometry_type, GEOARROW_DIMENSIONS_XY, array_cls);  \
  _GEOARROW_SPECIALIZE_ARRAY_TYPE(geometry_type, GEOARROW_DIMENSIONS_XYZ, array_cls); \
  _GEOARROW_SPECIALIZE_ARRAY_TYPE(geometry_type, GEOARROW_DIMENSIONS_XYM, array_cls); \
  _GEOARROW_SPECIALIZE_ARRAY_TYPE(geometry_type, GEOARROW_DIMENSIONS_XYZM, array_cls);

_GEOARROW_SPECIALIZE_GEOMETRY_TYPE(GEOARROW_GEOMETRY_TYPE_POINT, PointArray);
_GEOARROW_SPECIALIZE_GEOMETRY_TYPE(GEOARROW_GEOMETRY_TYPE_LINESTRING, LinestringArray);
_GEOARROW_SPECIALIZE_GEOMETRY_TYPE(GEOARROW_GEOMETRY_TYPE_POLYGON, PolygonArray);
_GEOARROW_SPECIALIZE_GEOMETRY_TYPE(GEOARROW_GEOMETRY_TYPE_MULTIPOINT, MultipointArray);
_GEOARROW_SPECIALIZE_GEOMETRY_TYPE(GEOARROW_GEOMETRY_TYPE_MULTILINESTRING,
                                   MultiLinestringArray);
_GEOARROW_SPECIALIZE_GEOMETRY_TYPE(GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON,
                                   MultiPolygonArray);

#undef _GEOARROW_SPECIALIZE_GEOMETRY_TYPE
#undef _GEOARROW_SPECIALIZE_ARRAY_TYPE

}  // namespace internal

/// \brief Resolve compile-time type definitions and constants from a geometry type and
/// dimension identifier.
template <enum GeoArrowGeometryType geometry_type, enum GeoArrowDimensions dimensions>
struct GeometryTypeTraits {
  using coord_type = typename internal::DimensionTraits<dimensions>::coord_type;
  using sequence_type = array_util::CoordSequence<coord_type>;
  using array_type =
      typename internal::ResolveArrayType<geometry_type, dimensions>::array_type;

  static constexpr enum GeoArrowType type_id(enum GeoArrowCoordType coord_type =
                                                 GEOARROW_COORD_TYPE_SEPARATE){
      return static_cast<enum GeoArrowType>((coord_type - 1) * 10000 +
                                            (dimensions - 1) * 1000 + geometry_type);}

};  // namespace type_traits

/// \brief Resolve compile-time type definitions and constants from a type identifier
template <enum GeoArrowType type>
struct TypeTraits {
  static constexpr enum GeoArrowCoordType coord_type_id =
      static_cast<enum GeoArrowCoordType>(type / 10000 + 1);
  static constexpr enum GeoArrowDimensions dimensions =
      static_cast<enum GeoArrowDimensions>((type % 10000) / 1000 + 1);
  static constexpr enum GeoArrowGeometryType geometry_type =
      static_cast<enum GeoArrowGeometryType>((type % 10000) % 1000);

  using coord_type = typename internal::DimensionTraits<dimensions>::coord_type;
  using array_type =
      typename internal::ResolveArrayType<geometry_type, dimensions>::array_type;
};

template <>
struct TypeTraits<GEOARROW_TYPE_WKT> {};

template <>
struct TypeTraits<GEOARROW_TYPE_LARGE_WKT> {};

template <>
struct TypeTraits<GEOARROW_TYPE_WKB> {
  using array_type = wkb_util::WKBArray<int32_t>;
  using coord_type = array_util::XYZM<double>;
};

template <>
struct TypeTraits<GEOARROW_TYPE_LARGE_WKB> {
  using array_type = wkb_util::WKBArray<int64_t>;
  using coord_type = array_util::XYZM<double>;
};

}  // namespace geoarrow

}  // namespace geoarrow

#define _GEOARROW_NATIVE_ARRAY_CALL_INTERNAL(geometry_type, dimensions, expr, ...)    \
  do {                                                                                \
    using array_type_internal =                                                       \
        typename ::geoarrow::type_traits::GeometryTypeTraits<geometry_type,           \
                                                             dimensions>::array_type; \
    array_type_internal array_instance_internal;                                      \
    expr(array_instance_internal, __VA_ARGS__);                                       \
  } while (false)

#define _GEOARROW_DISPATCH_NATIVE_ARRAY_CALL_DIMENSIONS(geometry_type, dimensions, expr, \
                                                        ...)                             \
  do {                                                                                   \
    switch (dimensions) {                                                                \
      case GEOARROW_DIMENSIONS_XY:                                                       \
        _GEOARROW_NATIVE_ARRAY_CALL_INTERNAL(geometry_type, GEOARROW_DIMENSIONS_XY,      \
                                             expr, __VA_ARGS__);                         \
        break;                                                                           \
      case GEOARROW_DIMENSIONS_XYZ:                                                      \
        _GEOARROW_NATIVE_ARRAY_CALL_INTERNAL(geometry_type, GEOARROW_DIMENSIONS_XYZ,     \
                                             expr, __VA_ARGS__);                         \
        break;                                                                           \
      case GEOARROW_DIMENSIONS_XYM:                                                      \
        _GEOARROW_NATIVE_ARRAY_CALL_INTERNAL(geometry_type, GEOARROW_DIMENSIONS_XYM,     \
                                             expr, __VA_ARGS__);                         \
        break;                                                                           \
      case GEOARROW_DIMENSIONS_XYZM:                                                     \
        _GEOARROW_NATIVE_ARRAY_CALL_INTERNAL(geometry_type, GEOARROW_DIMENSIONS_XYZM,    \
                                             expr, __VA_ARGS__);                         \
        break;                                                                           \
      default:                                                                           \
        throw ::geoarrow::Exception("Unknown dimension type");                           \
    }                                                                                    \
  } while (false)

/// \brief Dispatch a call to a function accepting a native Array specialization
///
/// This allows writing generic code against any Array type in the form
/// `template <typename Array> DoSomething(Array& array, ...) { ... }`, which can be
/// dispatched using `GEOARROW_DISPATCH_NATIVE_ARRAY_CALL(type_id, DoSomething, ...);`.
/// Currently requires that `DoSomething` handles all native array types and accepts
/// a parameter other than array (e.g., a `GeoArrowArrayView`).
///
/// This version only dispatches to "native" array types. Use
/// `GEOARROW_DISPATCH_ARRAY_CALL` to also dispatch serialized types.
#define GEOARROW_DISPATCH_NATIVE_ARRAY_CALL(type_id, expr, ...)                         \
  do {                                                                                  \
    auto geometry_type_internal = GeoArrowGeometryTypeFromType(type_id);                \
    auto dimensions_internal = GeoArrowDimensionsFromType(type_id);                     \
    switch (geometry_type_internal) {                                                   \
      case GEOARROW_GEOMETRY_TYPE_POINT:                                                \
        _GEOARROW_DISPATCH_NATIVE_ARRAY_CALL_DIMENSIONS(                                \
            GEOARROW_GEOMETRY_TYPE_POINT, dimensions_internal, expr, __VA_ARGS__);      \
        break;                                                                          \
      case GEOARROW_GEOMETRY_TYPE_LINESTRING:                                           \
        _GEOARROW_DISPATCH_NATIVE_ARRAY_CALL_DIMENSIONS(                                \
            GEOARROW_GEOMETRY_TYPE_LINESTRING, dimensions_internal, expr, __VA_ARGS__); \
        break;                                                                          \
      case GEOARROW_GEOMETRY_TYPE_POLYGON:                                              \
        _GEOARROW_DISPATCH_NATIVE_ARRAY_CALL_DIMENSIONS(                                \
            GEOARROW_GEOMETRY_TYPE_POLYGON, dimensions_internal, expr, __VA_ARGS__);    \
        break;                                                                          \
      case GEOARROW_GEOMETRY_TYPE_MULTIPOINT:                                           \
        _GEOARROW_DISPATCH_NATIVE_ARRAY_CALL_DIMENSIONS(                                \
            GEOARROW_GEOMETRY_TYPE_MULTIPOINT, dimensions_internal, expr, __VA_ARGS__); \
        break;                                                                          \
      case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING:                                      \
        _GEOARROW_DISPATCH_NATIVE_ARRAY_CALL_DIMENSIONS(                                \
            GEOARROW_GEOMETRY_TYPE_MULTILINESTRING, dimensions_internal, expr,          \
            __VA_ARGS__);                                                               \
        break;                                                                          \
      case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON:                                         \
        _GEOARROW_DISPATCH_NATIVE_ARRAY_CALL_DIMENSIONS(                                \
            GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON, dimensions_internal, expr,             \
            __VA_ARGS__);                                                               \
        break;                                                                          \
      default:                                                                          \
        throw ::geoarrow::Exception("Unknown geometry type");                           \
    }                                                                                   \
  } while (false)

/// \brief Dispatch a call to a function accepting a native or serialized Array
/// specialization
///
/// Identical to `GEOARROW_DISPATCH_NATIVE_ARRAY_CALL()`, but also handles dispatching to
/// WKB arrays. WKB arrays might not support the same operations or have the same syntax
/// and may need to be dispatched separately for anything that can't use VisitEdges()
/// or VisitVertices().
#define GEOARROW_DISPATCH_ARRAY_CALL(type_id, expr, ...)                                 \
  do {                                                                                   \
    switch (type_id) {                                                                   \
      case GEOARROW_TYPE_WKB: {                                                          \
        using array_type_internal =                                                      \
            typename ::geoarrow::type_traits::TypeTraits<GEOARROW_TYPE_WKB>::array_type; \
        array_type_internal array_instance_internal;                                     \
        expr(array_instance_internal, __VA_ARGS__);                                      \
        break;                                                                           \
      }                                                                                  \
      case GEOARROW_TYPE_LARGE_WKB:                                                      \
      case GEOARROW_TYPE_WKT:                                                            \
      case GEOARROW_TYPE_LARGE_WKT:                                                      \
        throw ::geoarrow::Exception("WKT/Large WKB not handled by generic dispatch");    \
      default:                                                                           \
        GEOARROW_DISPATCH_NATIVE_ARRAY_CALL(type_id, expr, __VA_ARGS__);                 \
    }                                                                                    \
  } while (false)

#endif
