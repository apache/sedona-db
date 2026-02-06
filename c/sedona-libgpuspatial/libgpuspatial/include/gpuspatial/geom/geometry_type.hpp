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
#pragma once

#include "geoarrow/geoarrow.hpp"

#include <string>

namespace gpuspatial {
// N.B. The order of this enum must match GeoArrowGeometryType
enum class GeometryType {
  kGeometry,
  kPoint,  // 1
  kLineString,
  kPolygon,
  kMultiPoint,
  kMultiLineString,
  kMultiPolygon,
  kGeometryCollection,  // 7
  kBox,
  kNull,
  kNumGeometryTypes
};

inline std::string GeometryTypeToString(GeometryType type) {
  switch (type) {
    case GeometryType::kPoint:
      return "Point";
    case GeometryType::kMultiPoint:
      return "MultiPoint";
    case GeometryType::kLineString:
      return "LineString";
    case GeometryType::kMultiLineString:
      return "MultiLineString";
    case GeometryType::kPolygon:
      return "Polygon";
    case GeometryType::kMultiPolygon:
      return "MultiPolygon";
    case GeometryType::kGeometryCollection:
      return "GeometryCollection";
    case GeometryType::kBox:
      return "Box";
    default:
      return "Unknown";
  }
}

inline GeometryType FromGeoArrowGeometryType(GeoArrowGeometryType geo_arrow_type) {
  GeometryType type = GeometryType::kNumGeometryTypes;
  switch (geo_arrow_type) {
    case GEOARROW_GEOMETRY_TYPE_POINT: {
      type = GeometryType::kPoint;
      break;
    }
    case GEOARROW_GEOMETRY_TYPE_LINESTRING: {
      type = GeometryType::kLineString;
      break;
    }
    case GEOARROW_GEOMETRY_TYPE_POLYGON: {
      type = GeometryType::kPolygon;
      break;
    }
    case GEOARROW_GEOMETRY_TYPE_MULTIPOINT: {
      type = GeometryType::kMultiPoint;
      break;
    }
    case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING: {
      type = GeometryType::kMultiLineString;
      break;
    }
    case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON: {
      type = GeometryType::kMultiPolygon;
      break;
    }
    case GEOARROW_GEOMETRY_TYPE_GEOMETRYCOLLECTION: {
      type = GeometryType::kGeometryCollection;
      break;
    }
    default: {
      throw std::runtime_error("Unsupported type " +
                               std::string(GeoArrowGeometryTypeString(geo_arrow_type)));
    }
  }
  return type;
}

namespace detail {
inline bool IsPointType(GeometryType type) {
  return type == GeometryType::kPoint || type == GeometryType::kMultiPoint;
}

inline bool IsLineType(GeometryType type) {
  return type == GeometryType::kLineString || type == GeometryType::kMultiLineString;
}

inline bool IsPolygonType(GeometryType type) {
  return type == GeometryType::kPolygon || type == GeometryType::kMultiPolygon;
}
}  // namespace detail

inline GeometryType GetCompatibleGeometryType(GeometryType type1, GeometryType type2) {
  if (type1 != type2) {
    if (detail::IsPointType(type1) && detail::IsPointType(type2)) {
      return GeometryType::kMultiPoint;
    } else if (detail::IsLineType(type1) && detail::IsLineType(type2)) {
      return GeometryType::kMultiLineString;
    } else if (detail::IsPolygonType(type1) && detail::IsPolygonType(type2)) {
      return GeometryType::kMultiPolygon;
    } else {
      return GeometryType::kGeometryCollection;
    }
  }
  return type1;
}
}  // namespace gpuspatial
