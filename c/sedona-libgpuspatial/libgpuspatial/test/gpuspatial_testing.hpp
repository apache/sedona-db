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

#include "geoarrow/geoarrow.hpp"
#include "nanoarrow/nanoarrow.hpp"

namespace gpuspatial::testing {

inline void MakeWKBArrayFromWKT(const std::vector<std::string>& wkts,
                                struct ArrowArray* out) {
  // Build a WKT array using nanoarrow
  nanoarrow::UniqueArray wkt_array;
  NANOARROW_THROW_NOT_OK(ArrowArrayInitFromType(wkt_array.get(), NANOARROW_TYPE_STRING));
  NANOARROW_THROW_NOT_OK(ArrowArrayStartAppending(wkt_array.get()));
  for (const auto& wkt : wkts) {
    // Use "" (empty string) as the null sentinel for testing
    if (wkt.empty()) {
      NANOARROW_THROW_NOT_OK(ArrowArrayAppendNull(wkt_array.get(), 1));
    } else {
      NANOARROW_THROW_NOT_OK(ArrowArrayAppendString(
          wkt_array.get(), {wkt.data(), static_cast<int64_t>(wkt.size())}));
    }
  }

  NANOARROW_THROW_NOT_OK(ArrowArrayFinishBuildingDefault(wkt_array.get(), nullptr));

  // Convert it to WKB using the ArrayReader and ArrayWriter
  geoarrow::ArrayReader reader(GEOARROW_TYPE_WKT);
  geoarrow::ArrayWriter writer(GEOARROW_TYPE_WKB);
  struct GeoArrowError error{};

  reader.SetArrayNonOwning(wkt_array.get());
  GEOARROW_THROW_NOT_OK(&error,
                        reader.Visit(writer.visitor(), 0, wkt_array->length, &error));
  writer.Finish(out);
}

inline std::vector<std::string> ReadWKBArray(const struct ArrowArray* wkb_array) {
  // Convert array to WKT using the ArrayReader and ArrayWriter
  geoarrow::ArrayReader reader(GEOARROW_TYPE_WKB);
  geoarrow::ArrayWriter writer(GEOARROW_TYPE_WKT);
  struct GeoArrowError error{};

  reader.SetArrayNonOwning(wkb_array);
  GEOARROW_THROW_NOT_OK(&error,
                        reader.Visit(writer.visitor(), 0, wkb_array->length, &error));

  nanoarrow::UniqueArray wkt_array;
  writer.Finish(wkt_array.get());

  std::vector<std::string> out;
  auto view = nanoarrow::ViewArrayAsBytes<32>(wkt_array.get());
  for (const auto& item : view) {
    auto item_or_sentinel = item.value_or({nullptr, 0});
    out.push_back(
        {item_or_sentinel.data, static_cast<size_t>(item_or_sentinel.size_bytes)});
  }

  return out;
}

class WKBBounder {
 public:
  using BoxXY = geoarrow::array_util::BoxXY<double>;

  WKBBounder() {
    GEOARROW_THROW_NOT_OK(nullptr, GeoArrowWKBReaderInit(&reader_));
    ArrowArrayViewInitFromType(&array_view_, NANOARROW_TYPE_BINARY);
  }

  ~WKBBounder() { GeoArrowWKBReaderReset(&reader_); }

  const BoxXY& Bounds() const { return bounds_; }

  void Read(const struct ArrowArray* array) {
    NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&array_view_, array, nullptr));
    struct ArrowBufferView item;
    struct GeoArrowGeometryView geom;
    for (int64_t i = 0; i < array_view_.length; i++) {
      if (!ArrowArrayViewIsNull(&array_view_, i)) {
        item = ArrowArrayViewGetBytesUnsafe(&array_view_, i);
        GEOARROW_THROW_NOT_OK(
            &error_,
            GeoArrowWKBReaderRead(&reader_, {item.data.as_uint8, item.size_bytes}, &geom,
                                  &error_));
        ReadGeometry(geom);
      }
    }
  }

  void ReadGeometry(const GeoArrowGeometryView& geom) {
    const struct GeoArrowGeometryNode* node;
    const struct GeoArrowGeometryNode* end;
    const uint8_t* px;
    const uint8_t* py;
    int32_t dx, dy;
    double x, y;

    end = geom.root + geom.size_nodes;
    for (node = geom.root; node < end; node++) {
      switch (node->geometry_type) {
        case GEOARROW_GEOMETRY_TYPE_POINT:
        case GEOARROW_GEOMETRY_TYPE_LINESTRING:
          px = geom.root->coords[0];
          py = geom.root->coords[1];
          dx = geom.root->coord_stride[0];
          dy = geom.root->coord_stride[1];

          if (node->flags & GEOARROW_GEOMETRY_NODE_FLAG_SWAP_ENDIAN) {
            throw std::runtime_error("big endian not supported");
          }

          for (uint32_t i = 0; i < node->size; i++) {
            std::memcpy(&x, px, sizeof(double));
            std::memcpy(&y, py, sizeof(double));

            bounds_[0] = std::min(bounds_[0], x);
            bounds_[1] = std::min(bounds_[1], y);
            bounds_[2] = std::max(bounds_[2], x);
            bounds_[3] = std::max(bounds_[3], y);

            px += dx;
            py += dy;
          }
          break;
        default:
          break;
      }
    }
  }

 private:
  struct GeoArrowError error_{};
  struct ArrowArrayView array_view_;
  struct GeoArrowWKBReader reader_{};
  BoxXY bounds_{BoxXY::Empty()};
};

}  // namespace gpuspatial::testing
