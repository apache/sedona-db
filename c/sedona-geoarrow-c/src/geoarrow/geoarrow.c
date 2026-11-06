#include "geoarrow/geoarrow.h"

#include <errno.h>
#include <string.h>

#include "nanoarrow/nanoarrow.h"



static GeoArrowErrorCode GeoArrowSchemaInitCoordFixedSizeList(struct ArrowSchema* schema,
                                                              const char* dims) {
  int64_t n_dims = strlen(dims);
  ArrowSchemaInit(schema);
  NANOARROW_RETURN_NOT_OK(ArrowSchemaSetTypeFixedSize(
      schema, NANOARROW_TYPE_FIXED_SIZE_LIST, (int32_t)n_dims));
  NANOARROW_RETURN_NOT_OK(ArrowSchemaSetName(schema->children[0], dims));
  NANOARROW_RETURN_NOT_OK(ArrowSchemaSetType(schema->children[0], NANOARROW_TYPE_DOUBLE));

  // Set child field non-nullable
  schema->children[0]->flags = 0;

  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowSchemaInitCoordStruct(struct ArrowSchema* schema,
                                                       const char* dims) {
  int64_t n_dims = strlen(dims);
  char dim_name[] = {'\0', '\0'};

  NANOARROW_RETURN_NOT_OK(ArrowSchemaInitFromType(schema, NANOARROW_TYPE_STRUCT));
  NANOARROW_RETURN_NOT_OK(ArrowSchemaAllocateChildren(schema, n_dims));

  for (int64_t i = 0; i < n_dims; i++) {
    dim_name[0] = dims[i];
    NANOARROW_RETURN_NOT_OK(
        ArrowSchemaInitFromType(schema->children[i], NANOARROW_TYPE_DOUBLE));
    NANOARROW_RETURN_NOT_OK(ArrowSchemaSetName(schema->children[i], dim_name));
    // Set child non-nullable
    schema->children[i]->flags = 0;
  }

  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowSchemaInitRect(struct ArrowSchema* schema,
                                                const char* dims) {
  int64_t n_dims = strlen(dims);
  char dim_name_min[] = {'\0', 'm', 'i', 'n', '\0'};
  char dim_name_max[] = {'\0', 'm', 'a', 'x', '\0'};

  NANOARROW_RETURN_NOT_OK(ArrowSchemaInitFromType(schema, NANOARROW_TYPE_STRUCT));
  NANOARROW_RETURN_NOT_OK(ArrowSchemaAllocateChildren(schema, n_dims * 2));

  for (int64_t i = 0; i < n_dims; i++) {
    dim_name_min[0] = dims[i];
    NANOARROW_RETURN_NOT_OK(
        ArrowSchemaInitFromType(schema->children[i], NANOARROW_TYPE_DOUBLE));
    NANOARROW_RETURN_NOT_OK(ArrowSchemaSetName(schema->children[i], dim_name_min));

    dim_name_max[0] = dims[i];
    NANOARROW_RETURN_NOT_OK(
        ArrowSchemaInitFromType(schema->children[n_dims + i], NANOARROW_TYPE_DOUBLE));
    NANOARROW_RETURN_NOT_OK(
        ArrowSchemaSetName(schema->children[n_dims + i], dim_name_max));

    // Set children non-nullable
    schema->children[i]->flags = 0;
    schema->children[i + n_dims]->flags = 0;
  }

  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowSchemaInitListOf(struct ArrowSchema* schema,
                                                  enum GeoArrowCoordType coord_type,
                                                  const char* dims, int n,
                                                  const char** child_names) {
  if (n == 0) {
    switch (coord_type) {
      case GEOARROW_COORD_TYPE_SEPARATE:
        return GeoArrowSchemaInitCoordStruct(schema, dims);
      case GEOARROW_COORD_TYPE_INTERLEAVED:
        return GeoArrowSchemaInitCoordFixedSizeList(schema, dims);
      default:
        return EINVAL;
    }
  } else {
    ArrowSchemaInit(schema);
    NANOARROW_RETURN_NOT_OK(ArrowSchemaSetFormat(schema, "+l"));
    NANOARROW_RETURN_NOT_OK(ArrowSchemaAllocateChildren(schema, 1));
    NANOARROW_RETURN_NOT_OK(GeoArrowSchemaInitListOf(schema->children[0], coord_type,
                                                     dims, n - 1, child_names + 1));
    NANOARROW_RETURN_NOT_OK(ArrowSchemaSetName(schema->children[0], child_names[0]));

    // Set child field non-nullable
    schema->children[0]->flags = 0;

    return NANOARROW_OK;
  }
}

#define CHILD_NAMES_LINESTRING \
  (const char*[]) { "vertices" }
#define CHILD_NAMES_POLYGON \
  (const char*[]) { "rings", "vertices" }
#define CHILD_NAMES_MULTIPOINT \
  (const char*[]) { "points" }
#define CHILD_NAMES_MULTILINESTRING \
  (const char*[]) { "linestrings", "vertices" }
#define CHILD_NAMES_MULTIPOLYGON \
  (const char*[]) { "polygons", "rings", "vertices" }

GeoArrowErrorCode GeoArrowSchemaInit(struct ArrowSchema* schema, enum GeoArrowType type) {
  schema->release = NULL;

  switch (type) {
    case GEOARROW_TYPE_WKB:
      return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_BINARY);
    case GEOARROW_TYPE_LARGE_WKB:
      return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_LARGE_BINARY);
    case GEOARROW_TYPE_WKB_VIEW:
      return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_BINARY_VIEW);

    case GEOARROW_TYPE_WKT:
      return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_STRING);
    case GEOARROW_TYPE_LARGE_WKT:
      return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_LARGE_STRING);
    case GEOARROW_TYPE_WKT_VIEW:
      return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_STRING_VIEW);

    default:
      break;
  }

  enum GeoArrowDimensions dimensions = GeoArrowDimensionsFromType(type);
  enum GeoArrowCoordType coord_type = GeoArrowCoordTypeFromType(type);
  enum GeoArrowGeometryType geometry_type = GeoArrowGeometryTypeFromType(type);

  const char* dims;
  switch (dimensions) {
    case GEOARROW_DIMENSIONS_XY:
      dims = "xy";
      break;
    case GEOARROW_DIMENSIONS_XYZ:
      dims = "xyz";
      break;
    case GEOARROW_DIMENSIONS_XYM:
      dims = "xym";
      break;
    case GEOARROW_DIMENSIONS_XYZM:
      dims = "xyzm";
      break;
    default:
      return EINVAL;
  }

  switch (geometry_type) {
    case GEOARROW_GEOMETRY_TYPE_BOX:
      switch (coord_type) {
        case GEOARROW_COORD_TYPE_SEPARATE:
          NANOARROW_RETURN_NOT_OK(GeoArrowSchemaInitRect(schema, dims));
          break;
        default:
          return EINVAL;
      }
      break;

    case GEOARROW_GEOMETRY_TYPE_POINT:
      switch (coord_type) {
        case GEOARROW_COORD_TYPE_SEPARATE:
          NANOARROW_RETURN_NOT_OK(GeoArrowSchemaInitCoordStruct(schema, dims));
          break;
        case GEOARROW_COORD_TYPE_INTERLEAVED:
          NANOARROW_RETURN_NOT_OK(GeoArrowSchemaInitCoordFixedSizeList(schema, dims));
          break;
        default:
          return EINVAL;
      }
      break;

    case GEOARROW_GEOMETRY_TYPE_LINESTRING:
      NANOARROW_RETURN_NOT_OK(
          GeoArrowSchemaInitListOf(schema, coord_type, dims, 1, CHILD_NAMES_LINESTRING));
      break;
    case GEOARROW_GEOMETRY_TYPE_MULTIPOINT:
      NANOARROW_RETURN_NOT_OK(
          GeoArrowSchemaInitListOf(schema, coord_type, dims, 1, CHILD_NAMES_MULTIPOINT));
      break;
    case GEOARROW_GEOMETRY_TYPE_POLYGON:
      NANOARROW_RETURN_NOT_OK(
          GeoArrowSchemaInitListOf(schema, coord_type, dims, 2, CHILD_NAMES_POLYGON));
      break;
    case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING:
      NANOARROW_RETURN_NOT_OK(GeoArrowSchemaInitListOf(schema, coord_type, dims, 2,
                                                       CHILD_NAMES_MULTILINESTRING));
      break;
    case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON:
      NANOARROW_RETURN_NOT_OK(GeoArrowSchemaInitListOf(schema, coord_type, dims, 3,
                                                       CHILD_NAMES_MULTIPOLYGON));
      break;

    default:
      return ENOTSUP;
  }

  return NANOARROW_OK;
}

GeoArrowErrorCode GeoArrowSchemaInitExtension(struct ArrowSchema* schema,
                                              enum GeoArrowType type) {
  const char* ext_type = GeoArrowExtensionNameFromType(type);
  if (ext_type == NULL) {
    return EINVAL;
  }

  struct ArrowBuffer metadata;
  NANOARROW_RETURN_NOT_OK(ArrowMetadataBuilderInit(&metadata, NULL));
  int result = ArrowMetadataBuilderAppend(
      &metadata, ArrowCharView("ARROW:extension:name"), ArrowCharView(ext_type));
  if (result != NANOARROW_OK) {
    ArrowBufferReset(&metadata);
    return result;
  }

  result = ArrowMetadataBuilderAppend(
      &metadata, ArrowCharView("ARROW:extension:metadata"), ArrowCharView("{}"));
  if (result != NANOARROW_OK) {
    ArrowBufferReset(&metadata);
    return result;
  }

  result = GeoArrowSchemaInit(schema, type);
  if (result != NANOARROW_OK) {
    ArrowBufferReset(&metadata);
    return result;
  }

  result = ArrowSchemaSetMetadata(schema, (const char*)metadata.data);
  ArrowBufferReset(&metadata);
  return result;
}

#include <errno.h>
#include <stddef.h>
#include <string.h>


#include "nanoarrow/nanoarrow.h"

static int GeoArrowParsePointFixedSizeList(const struct ArrowSchema* schema,
                                           struct GeoArrowSchemaView* schema_view,
                                           struct ArrowError* error,
                                           const char* ext_name) {
  if (schema->n_children != 1 || strcmp(schema->children[0]->format, "g") != 0) {
    ArrowErrorSet(
        error,
        "Expected fixed-size list coordinate child 0 to have storage type of double for "
        "extension '%s'",
        ext_name);
    return EINVAL;
  }

  struct ArrowSchemaView na_schema_view;
  NANOARROW_RETURN_NOT_OK(ArrowSchemaViewInit(&na_schema_view, schema, error));
  const char* maybe_dims = schema->children[0]->name;
  if (maybe_dims == NULL) {
    maybe_dims = "<NULL>";
  }

  if (strcmp(maybe_dims, "xy") == 0) {
    schema_view->dimensions = GEOARROW_DIMENSIONS_XY;
  } else if (strcmp(maybe_dims, "xyz") == 0) {
    schema_view->dimensions = GEOARROW_DIMENSIONS_XYZ;
  } else if (strcmp(maybe_dims, "xym") == 0) {
    schema_view->dimensions = GEOARROW_DIMENSIONS_XYM;
  } else if (strcmp(maybe_dims, "xyzm") == 0) {
    schema_view->dimensions = GEOARROW_DIMENSIONS_XYZM;
  } else {
    switch (na_schema_view.fixed_size) {
      case 2:
        schema_view->dimensions = GEOARROW_DIMENSIONS_XY;
        break;
      case 3:
        schema_view->dimensions = GEOARROW_DIMENSIONS_XYZ;
        break;
      case 4:
        schema_view->dimensions = GEOARROW_DIMENSIONS_XYZM;
        break;
      default:
        ArrowErrorSet(error,
                      "Can't guess dimensions for fixed size list coord array with child "
                      "name '%s' and fixed size %d for extension '%s'",
                      maybe_dims, na_schema_view.fixed_size, ext_name);
        return EINVAL;
    }
  }

  int expected_n_dims = _GeoArrowkNumDimensions[schema_view->dimensions];
  if (expected_n_dims != na_schema_view.fixed_size) {
    ArrowErrorSet(error,
                  "Expected fixed size list coord array with child name '%s' to have "
                  "fixed size %d but found fixed size %d for extension '%s'",
                  maybe_dims, expected_n_dims, na_schema_view.fixed_size, ext_name);
    return EINVAL;
  }

  schema_view->coord_type = GEOARROW_COORD_TYPE_INTERLEAVED;
  return NANOARROW_OK;
}

static int GeoArrowParsePointStruct(const struct ArrowSchema* schema,
                                    struct GeoArrowSchemaView* schema_view,
                                    struct ArrowError* error, const char* ext_name) {
  if (schema->n_children < 2 || schema->n_children > 4) {
    ArrowErrorSet(
        error,
        "Expected 2, 3, or 4 children for coord array for extension '%s' but got %d",
        ext_name, (int)schema->n_children);
    return EINVAL;
  }

  char dim[5];
  memset(dim, 0, sizeof(dim));
  for (int64_t i = 0; i < schema->n_children; i++) {
    const char* child_name = schema->children[i]->name;
    if (child_name == NULL || strlen(child_name) != 1) {
      ArrowErrorSet(error,
                    "Expected coordinate child %d to have single character name for "
                    "extension '%s'",
                    (int)i, ext_name);
      return EINVAL;
    }

    if (strcmp(schema->children[i]->format, "g") != 0) {
      ArrowErrorSet(error,
                    "Expected coordinate child %d to have storage type of double for "
                    "extension '%s'",
                    (int)i, ext_name);
      return EINVAL;
    }

    dim[i] = child_name[0];
  }

  if (strcmp(dim, "xy") == 0) {
    schema_view->dimensions = GEOARROW_DIMENSIONS_XY;
  } else if (strcmp(dim, "xyz") == 0) {
    schema_view->dimensions = GEOARROW_DIMENSIONS_XYZ;
  } else if (strcmp(dim, "xym") == 0) {
    schema_view->dimensions = GEOARROW_DIMENSIONS_XYM;
  } else if (strcmp(dim, "xyzm") == 0) {
    schema_view->dimensions = GEOARROW_DIMENSIONS_XYZM;
  } else {
    ArrowErrorSet(error,
                  "Expected dimensions 'xy', 'xyz', 'xym', or 'xyzm' for extension "
                  "'%s' but found '%s'",
                  ext_name, dim);
    return EINVAL;
  }

  schema_view->coord_type = GEOARROW_COORD_TYPE_SEPARATE;
  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowParseNestedSchema(const struct ArrowSchema* schema,
                                                   int n,
                                                   struct GeoArrowSchemaView* schema_view,
                                                   struct ArrowError* error,
                                                   const char* ext_name) {
  if (n == 0) {
    if (strcmp(schema->format, "+s") == 0) {
      return GeoArrowParsePointStruct(schema, schema_view, error, ext_name);
    } else if (strncmp(schema->format, "+w:", 3) == 0) {
      return GeoArrowParsePointFixedSizeList(schema, schema_view, error, ext_name);
    } else {
      ArrowErrorSet(error,
                    "Expected storage type fixed-size list or struct for coord array for "
                    "extension '%s'",
                    ext_name);
      return EINVAL;
    }
  } else {
    if (strcmp(schema->format, "+l") != 0 || schema->n_children != 1) {
      ArrowErrorSet(error,
                    "Expected valid list type for coord parent %d for extension '%s'", n,
                    ext_name);
      return EINVAL;
    }

    return GeoArrowParseNestedSchema(schema->children[0], n - 1, schema_view, error,
                                     ext_name);
  }
}

static int GeoArrowParseBoxChild(const struct ArrowSchema* schema, char* dim,
                                 const char** minmax, int64_t i,
                                 struct ArrowError* error) {
  const char* name = schema->name;
  if (name == NULL || strlen(name) != 4) {
    ArrowErrorSet(error, "Expected box child %d to have exactly four characters", (int)i);
    return EINVAL;
  }

  if (strcmp(schema->format, "g") != 0) {
    ArrowErrorSet(error, "Expected box child %d to have storage type of double", (int)i);
    return EINVAL;
  }

  *dim = schema->name[0];
  *minmax = schema->name + 1;
  return GEOARROW_OK;
}

static int GeoArrowParseBox(const struct ArrowSchema* schema,
                            struct GeoArrowSchemaView* schema_view,
                            struct ArrowError* error) {
  if (strcmp(schema->format, "+s") != 0) {
    ArrowErrorSet(error, "Expected struct storage for 'geoarrow.box'");
    return EINVAL;
  }

  switch (schema->n_children) {
    case 4:
    case 6:
    case 8:
      break;
    default:
      ArrowErrorSet(
          error, "Expected 4, 6, or 8 children for extension 'geoarrow.box' but got %d",
          (int)schema->n_children);
      return EINVAL;
  }

  int64_t n_dims = schema->n_children / 2;
  char dim[5];
  memset(dim, 0, sizeof(dim));

  const char* minmax = "";
  for (int64_t i = 0; i < n_dims; i++) {
    NANOARROW_RETURN_NOT_OK(
        GeoArrowParseBoxChild(schema->children[i], dim + i, &minmax, i, error));
    if (strcmp(minmax, "min") != 0) {
      ArrowErrorSet(error, "Expected box child %d to have suffix 'min' but got '%s'",
                    (int)i, schema->children[i]->name);
      return EINVAL;
    }
  }

  if (strcmp(dim, "xy") == 0) {
    schema_view->dimensions = GEOARROW_DIMENSIONS_XY;
  } else if (strcmp(dim, "xyz") == 0) {
    schema_view->dimensions = GEOARROW_DIMENSIONS_XYZ;
  } else if (strcmp(dim, "xym") == 0) {
    schema_view->dimensions = GEOARROW_DIMENSIONS_XYM;
  } else if (strcmp(dim, "xyzm") == 0) {
    schema_view->dimensions = GEOARROW_DIMENSIONS_XYZM;
  } else {
    ArrowErrorSet(error,
                  "Expected dimensions 'xy', 'xyz', 'xym', or 'xyzm' for extension "
                  "'geoarrow.box' but found '%s'",
                  dim);
    return EINVAL;
  }

  for (int64_t i = 0; i < n_dims; i++) {
    char max_dim = '\0';
    NANOARROW_RETURN_NOT_OK(GeoArrowParseBoxChild(schema->children[n_dims + i], &max_dim,
                                                  &minmax, n_dims + i, error));
    if (strcmp(minmax, "max") != 0) {
      ArrowErrorSet(error, "Expected box child %d to have suffix 'max' but got '%s'",
                    (int)(n_dims + i), schema->children[n_dims + i]->name);
      return EINVAL;
    }

    if (max_dim != dim[i]) {
      ArrowErrorSet(
          error,
          "Expected box child %d name to match name for dimension '%s' but got '%s'",
          (int)i, schema->children[i]->name, schema->children[n_dims + i]->name);
      return EINVAL;
    }
  }

  schema_view->coord_type = GEOARROW_COORD_TYPE_SEPARATE;
  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowSchemaViewInitInternal(
    struct GeoArrowSchemaView* schema_view, const struct ArrowSchema* schema,
    struct ArrowSchemaView* na_schema_view, struct ArrowError* na_error) {
  const char* ext_name = na_schema_view->extension_name.data;
  int64_t ext_len = na_schema_view->extension_name.size_bytes;

  if (ext_len == 12 && strncmp(ext_name, "geoarrow.box", 12) == 0) {
    schema_view->geometry_type = GEOARROW_GEOMETRY_TYPE_BOX;
    NANOARROW_RETURN_NOT_OK(GeoArrowParseBox(schema, schema_view, na_error));
    schema_view->type = GeoArrowMakeType(
        schema_view->geometry_type, schema_view->dimensions, schema_view->coord_type);
  } else if (ext_len == 14 && strncmp(ext_name, "geoarrow.point", 14) == 0) {
    schema_view->geometry_type = GEOARROW_GEOMETRY_TYPE_POINT;
    NANOARROW_RETURN_NOT_OK(
        GeoArrowParseNestedSchema(schema, 0, schema_view, na_error, "geoarrow.point"));
    schema_view->type = GeoArrowMakeType(
        schema_view->geometry_type, schema_view->dimensions, schema_view->coord_type);
  } else if (ext_len == 19 && strncmp(ext_name, "geoarrow.linestring", 19) == 0) {
    schema_view->geometry_type = GEOARROW_GEOMETRY_TYPE_LINESTRING;
    NANOARROW_RETURN_NOT_OK(GeoArrowParseNestedSchema(schema, 1, schema_view, na_error,
                                                      "geoarrow.linestring"));
    schema_view->type = GeoArrowMakeType(
        schema_view->geometry_type, schema_view->dimensions, schema_view->coord_type);
  } else if (ext_len == 16 && strncmp(ext_name, "geoarrow.polygon", 16) == 0) {
    schema_view->geometry_type = GEOARROW_GEOMETRY_TYPE_POLYGON;
    NANOARROW_RETURN_NOT_OK(
        GeoArrowParseNestedSchema(schema, 2, schema_view, na_error, "geoarrow.polygon"));
    schema_view->type = GeoArrowMakeType(
        schema_view->geometry_type, schema_view->dimensions, schema_view->coord_type);
  } else if (ext_len == 19 && strncmp(ext_name, "geoarrow.multipoint", 19) == 0) {
    schema_view->geometry_type = GEOARROW_GEOMETRY_TYPE_MULTIPOINT;
    NANOARROW_RETURN_NOT_OK(GeoArrowParseNestedSchema(schema, 1, schema_view, na_error,
                                                      "geoarrow.multipoint"));
    schema_view->type = GeoArrowMakeType(
        schema_view->geometry_type, schema_view->dimensions, schema_view->coord_type);
  } else if (ext_len == 24 && strncmp(ext_name, "geoarrow.multilinestring", 24) == 0) {
    schema_view->geometry_type = GEOARROW_GEOMETRY_TYPE_MULTILINESTRING;
    NANOARROW_RETURN_NOT_OK(GeoArrowParseNestedSchema(schema, 2, schema_view, na_error,
                                                      "geoarrow.multilinestring"));
    schema_view->type = GeoArrowMakeType(
        schema_view->geometry_type, schema_view->dimensions, schema_view->coord_type);
  } else if (ext_len == 21 && strncmp(ext_name, "geoarrow.multipolygon", 21) == 0) {
    schema_view->geometry_type = GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON;
    NANOARROW_RETURN_NOT_OK(GeoArrowParseNestedSchema(schema, 3, schema_view, na_error,
                                                      "geoarrow.multipolygon"));
    schema_view->type = GeoArrowMakeType(
        schema_view->geometry_type, schema_view->dimensions, schema_view->coord_type);
  } else if (ext_len == 12 && strncmp(ext_name, "geoarrow.wkt", 12) == 0) {
    switch (na_schema_view->type) {
      case NANOARROW_TYPE_STRING:
        schema_view->type = GEOARROW_TYPE_WKT;
        break;
      case NANOARROW_TYPE_LARGE_STRING:
        schema_view->type = GEOARROW_TYPE_LARGE_WKT;
        break;
      case NANOARROW_TYPE_STRING_VIEW:
        schema_view->type = GEOARROW_TYPE_WKT_VIEW;
        break;
      default:
        ArrowErrorSet(na_error,
                      "Expected storage type of string or large_string for extension "
                      "'geoarrow.wkt'");
        return EINVAL;
    }

    schema_view->geometry_type = GeoArrowGeometryTypeFromType(schema_view->type);
    schema_view->dimensions = GeoArrowDimensionsFromType(schema_view->type);
    schema_view->coord_type = GeoArrowCoordTypeFromType(schema_view->type);
  } else if (ext_len >= 12 && strncmp(ext_name, "geoarrow.wkb", 12) == 0) {
    switch (na_schema_view->type) {
      case NANOARROW_TYPE_BINARY:
        schema_view->type = GEOARROW_TYPE_WKB;
        break;
      case NANOARROW_TYPE_LARGE_BINARY:
        schema_view->type = GEOARROW_TYPE_LARGE_WKB;
        break;
      case NANOARROW_TYPE_BINARY_VIEW:
        schema_view->type = GEOARROW_TYPE_WKB_VIEW;
        break;
      default:
        ArrowErrorSet(na_error,
                      "Expected storage type of binary or large_binary for extension "
                      "'geoarrow.wkb'");
        return EINVAL;
    }

    schema_view->geometry_type = GeoArrowGeometryTypeFromType(schema_view->type);
    schema_view->dimensions = GeoArrowDimensionsFromType(schema_view->type);
    schema_view->coord_type = GeoArrowCoordTypeFromType(schema_view->type);
  } else {
    ArrowErrorSet(na_error, "Unrecognized GeoArrow extension name: '%.*s'", (int)ext_len,
                  ext_name);
    return EINVAL;
  }

  schema_view->extension_name.data = na_schema_view->extension_name.data;
  schema_view->extension_name.size_bytes = na_schema_view->extension_name.size_bytes;
  schema_view->extension_metadata.data = na_schema_view->extension_metadata.data;
  schema_view->extension_metadata.size_bytes =
      na_schema_view->extension_metadata.size_bytes;

  return GEOARROW_OK;
}

GeoArrowErrorCode GeoArrowSchemaViewInit(struct GeoArrowSchemaView* schema_view,
                                         const struct ArrowSchema* schema,
                                         struct GeoArrowError* error) {
  struct ArrowError* na_error = (struct ArrowError*)error;
  struct ArrowSchemaView na_schema_view;
  NANOARROW_RETURN_NOT_OK(ArrowSchemaViewInit(&na_schema_view, schema, na_error));

  const char* ext_name = na_schema_view.extension_name.data;
  if (ext_name == NULL) {
    ArrowErrorSet(na_error, "Expected extension type");
    return EINVAL;
  }

  return GeoArrowSchemaViewInitInternal(schema_view, schema, &na_schema_view, na_error);
}

GeoArrowErrorCode GeoArrowSchemaViewInitFromStorage(
    struct GeoArrowSchemaView* schema_view, const struct ArrowSchema* schema,
    struct GeoArrowStringView extension_name, struct GeoArrowError* error) {
  struct ArrowError* na_error = (struct ArrowError*)error;
  struct ArrowSchemaView na_schema_view;
  NANOARROW_RETURN_NOT_OK(ArrowSchemaViewInit(&na_schema_view, schema, na_error));
  na_schema_view.extension_name.data = extension_name.data;
  na_schema_view.extension_name.size_bytes = extension_name.size_bytes;
  return GeoArrowSchemaViewInitInternal(schema_view, schema, &na_schema_view, na_error);
}

GeoArrowErrorCode GeoArrowSchemaViewInitFromType(struct GeoArrowSchemaView* schema_view,
                                                 enum GeoArrowType type) {
  schema_view->schema = NULL;
  schema_view->extension_name.data = NULL;
  schema_view->extension_name.size_bytes = 0;
  schema_view->extension_metadata.data = NULL;
  schema_view->extension_metadata.size_bytes = 0;
  schema_view->type = type;
  schema_view->geometry_type = GeoArrowGeometryTypeFromType(type);
  schema_view->dimensions = GeoArrowDimensionsFromType(type);
  schema_view->coord_type = GeoArrowCoordTypeFromType(type);

  if (type == GEOARROW_TYPE_UNINITIALIZED) {
    return GEOARROW_OK;
  }

  const char* extension_name = GeoArrowExtensionNameFromType(type);
  if (extension_name == NULL) {
    return EINVAL;
  }

  schema_view->extension_name.data = extension_name;
  schema_view->extension_name.size_bytes = strlen(extension_name);

  return GEOARROW_OK;
}

#include <errno.h>
#include <stdio.h>

#include "nanoarrow/nanoarrow.h"



#define CHECK_POS(n)                               \
  if ((pos + (int32_t)(n)) > ((int32_t)pos_max)) { \
    return EINVAL;                                 \
  }

static int ParseChar(struct ArrowStringView* s, char c) {
  if (s->size_bytes > 0 && s->data[0] == c) {
    s->size_bytes--;
    s->data++;
    return GEOARROW_OK;
  } else {
    return EINVAL;
  }
}

static void SkipWhitespace(struct ArrowStringView* s) {
  while (s->size_bytes > 0) {
    char c = *(s->data);
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
      s->size_bytes--;
      s->data++;
    } else {
      break;
    }
  }
}

static int SkipUntil(struct ArrowStringView* s, const char* items) {
  int64_t n_items = strlen(items);
  while (s->size_bytes > 0) {
    char c = *(s->data);
    if (c == '\0') {
      return 0;
    }

    for (int64_t i = 0; i < n_items; i++) {
      if (c == items[i]) {
        return 1;
      }
    }

    s->size_bytes--;
    s->data++;
  }

  return 0;
}

static GeoArrowErrorCode FindNull(struct ArrowStringView* s,
                                  struct ArrowStringView* out) {
  if (s->size_bytes < 4) {
    return EINVAL;
  }

  if (strncmp(s->data, "null", 4) != 0) {
    return EINVAL;
  }

  out->data = s->data;
  out->size_bytes = 4;
  s->size_bytes -= 4;
  s->data += 4;
  return GEOARROW_OK;
}

static GeoArrowErrorCode FindString(struct ArrowStringView* s,
                                    struct ArrowStringView* out) {
  out->data = s->data;
  if (s->data[0] != '\"') {
    return EINVAL;
  }

  s->size_bytes--;
  s->data++;

  int is_escape = 0;
  while (s->size_bytes > 0) {
    char c = *(s->data);
    if (!is_escape && c == '\\') {
      is_escape = 1;
      s->size_bytes--;
      s->data++;
      continue;
    }

    if (!is_escape && c == '\"') {
      s->size_bytes--;
      s->data++;
      out->size_bytes = s->data - out->data;
      return GEOARROW_OK;
    }

    s->size_bytes--;
    s->data++;
    is_escape = 0;
  }

  return EINVAL;
}

static GeoArrowErrorCode FindObject(struct ArrowStringView* s,
                                    struct ArrowStringView* out);

static GeoArrowErrorCode FindList(struct ArrowStringView* s,
                                  struct ArrowStringView* out) {
  out->data = s->data;
  if (s->data[0] != '[') {
    return EINVAL;
  }

  s->size_bytes--;
  s->data++;
  struct ArrowStringView tmp_value;
  while (s->size_bytes > 0) {
    if (SkipUntil(s, "[{\"]")) {
      char c = *(s->data);
      switch (c) {
        case '\"':
          NANOARROW_RETURN_NOT_OK(FindString(s, &tmp_value));
          break;
        case '[':
          NANOARROW_RETURN_NOT_OK(FindList(s, &tmp_value));
          break;
        case '{':
          NANOARROW_RETURN_NOT_OK(FindObject(s, &tmp_value));
          break;
        case ']':
          s->size_bytes--;
          s->data++;
          out->size_bytes = s->data - out->data;
          return GEOARROW_OK;
        default:
          break;
      }
    }
  }

  return EINVAL;
}

static GeoArrowErrorCode FindObject(struct ArrowStringView* s,
                                    struct ArrowStringView* out) {
  out->data = s->data;
  if (s->data[0] != '{') {
    return EINVAL;
  }

  s->size_bytes--;
  s->data++;
  struct ArrowStringView tmp_value;
  while (s->size_bytes > 0) {
    if (SkipUntil(s, "{[\"}")) {
      char c = *(s->data);
      switch (c) {
        case '\"':
          NANOARROW_RETURN_NOT_OK(FindString(s, &tmp_value));
          break;
        case '[':
          NANOARROW_RETURN_NOT_OK(FindList(s, &tmp_value));
          break;
        case '{':
          NANOARROW_RETURN_NOT_OK(FindObject(s, &tmp_value));
          break;
        case '}':
          s->size_bytes--;
          s->data++;
          out->size_bytes = s->data - out->data;
          return GEOARROW_OK;
        default:
          break;
      }
    }
  }

  return EINVAL;
}

static GeoArrowErrorCode ParseJSONMetadata(struct GeoArrowMetadataView* metadata_view,
                                           struct ArrowStringView* s) {
  NANOARROW_RETURN_NOT_OK(ParseChar(s, '{'));
  SkipWhitespace(s);
  struct ArrowStringView k;
  struct ArrowStringView v;

  while (s->size_bytes > 0 && s->data[0] != '}') {
    SkipWhitespace(s);
    NANOARROW_RETURN_NOT_OK(FindString(s, &k));
    SkipWhitespace(s);
    NANOARROW_RETURN_NOT_OK(ParseChar(s, ':'));
    SkipWhitespace(s);

    switch (s->data[0]) {
      case '[':
        NANOARROW_RETURN_NOT_OK(FindList(s, &v));
        break;
      case '{':
        NANOARROW_RETURN_NOT_OK(FindObject(s, &v));
        break;
      case '\"':
        NANOARROW_RETURN_NOT_OK(FindString(s, &v));
        break;
      case 'n':
        NANOARROW_RETURN_NOT_OK(FindNull(s, &v));
        break;
      default:
        // e.g., a number or boolean
        return EINVAL;
    }

    if (k.size_bytes == 7 && strncmp(k.data, "\"edges\"", 7) == 0) {
      if (v.size_bytes == 11 && strncmp(v.data, "\"spherical\"", 11) == 0) {
        metadata_view->edge_type = GEOARROW_EDGE_TYPE_SPHERICAL;
      } else if (v.size_bytes == 8 && strncmp(v.data, "\"planar\"", 8) == 0) {
        metadata_view->edge_type = GEOARROW_EDGE_TYPE_PLANAR;
      } else if (v.size_bytes == 10 && strncmp(v.data, "\"vincenty\"", 10) == 0) {
        metadata_view->edge_type = GEOARROW_EDGE_TYPE_VINCENTY;
      } else if (v.size_bytes == 8 && strncmp(v.data, "\"thomas\"", 8) == 0) {
        metadata_view->edge_type = GEOARROW_EDGE_TYPE_THOMAS;
      } else if (v.size_bytes == 9 && strncmp(v.data, "\"andoyer\"", 9) == 0) {
        metadata_view->edge_type = GEOARROW_EDGE_TYPE_ANDOYER;
      } else if (v.size_bytes == 8 && strncmp(v.data, "\"karney\"", 8) == 0) {
        metadata_view->edge_type = GEOARROW_EDGE_TYPE_KARNEY;
      } else if (v.data[0] == 'n') {
        metadata_view->edge_type = GEOARROW_EDGE_TYPE_PLANAR;
      } else {
        return EINVAL;
      }
    } else if (k.size_bytes == 5 && strncmp(k.data, "\"crs\"", 5) == 0) {
      if (v.data[0] == '{') {
        metadata_view->crs.data = v.data;
        metadata_view->crs.size_bytes = v.size_bytes;
        if (metadata_view->crs_type == GEOARROW_CRS_TYPE_NONE) {
          metadata_view->crs_type = GEOARROW_CRS_TYPE_UNKNOWN;
        }
      } else if (v.data[0] == '\"') {
        metadata_view->crs.data = v.data;
        metadata_view->crs.size_bytes = v.size_bytes;
        if (metadata_view->crs_type == GEOARROW_CRS_TYPE_NONE) {
          metadata_view->crs_type = GEOARROW_CRS_TYPE_UNKNOWN;
        }
      } else if (v.data[0] == 'n') {
        // A null explicitly un-sets the CRS
        metadata_view->crs_type = GEOARROW_CRS_TYPE_NONE;
      } else {
        // Reject an unknown JSON type
        return EINVAL;
      }
    } else if (k.size_bytes == 10 && strncmp(k.data, "\"crs_type\"", 10) == 0) {
      if (v.data[0] == '\"') {
        if (v.size_bytes == 10 && strncmp(k.data, "\"projjson\"", 10)) {
          metadata_view->crs_type = GEOARROW_CRS_TYPE_PROJJSON;
        } else if (v.size_bytes == 11 && strncmp(k.data, "\"wkt2:2019\"", 11)) {
          metadata_view->crs_type = GEOARROW_CRS_TYPE_WKT2_2019;
        } else if (v.size_bytes == 16 && strncmp(k.data, "\"authority_code\"", 16)) {
          metadata_view->crs_type = GEOARROW_CRS_TYPE_AUTHORITY_CODE;
        } else if (v.size_bytes == 6 && strncmp(k.data, "\"srid\"", 6)) {
          metadata_view->crs_type = GEOARROW_CRS_TYPE_SRID;
        } else {
          // Accept unrecognized string values but ignore them
          metadata_view->crs_type = GEOARROW_CRS_TYPE_UNKNOWN;
        }
      } else {
        // Reject values that are not a string
        return EINVAL;
      }
    }

    SkipUntil(s, ",}");
    if (s->data[0] == ',') {
      s->size_bytes--;
      s->data++;
    }
  }

  if (s->size_bytes > 0 && s->data[0] == '}') {
    s->size_bytes--;
    s->data++;
    return GEOARROW_OK;
  } else {
    return EINVAL;
  }
}

static GeoArrowErrorCode GeoArrowMetadataViewInitJSON(
    struct GeoArrowMetadataView* metadata_view, struct GeoArrowError* error) {
  struct ArrowStringView metadata;
  metadata.data = metadata_view->metadata.data;
  metadata.size_bytes = metadata_view->metadata.size_bytes;

  struct ArrowStringView s = metadata;
  SkipWhitespace(&s);

  if (ParseJSONMetadata(metadata_view, &s) != GEOARROW_OK) {
    GeoArrowErrorSet(error, "Expected valid GeoArrow JSON metadata but got '%.*s'",
                     (int)metadata.size_bytes, metadata.data);
    return EINVAL;
  }

  SkipWhitespace(&s);
  if (s.data != (metadata.data + metadata.size_bytes)) {
    ArrowErrorSet(
        (struct ArrowError*)error,
        "Expected JSON object with no trailing characters but found trailing '%.*s'",
        (int)s.size_bytes, s.data);
    return EINVAL;
  }

  // Do one final canonicalization: it is possible that the crs_type was set
  // but the crs was not. If this is the case, we need to unset the crs_type to
  // NONE.
  if (metadata_view->crs.size_bytes == 0) {
    metadata_view->crs_type = GEOARROW_CRS_TYPE_NONE;
  }

  return GEOARROW_OK;
}

GeoArrowErrorCode GeoArrowMetadataViewInit(struct GeoArrowMetadataView* metadata_view,
                                           struct GeoArrowStringView metadata,
                                           struct GeoArrowError* error) {
  metadata_view->metadata = metadata;
  metadata_view->edge_type = GEOARROW_EDGE_TYPE_PLANAR;
  metadata_view->crs_type = GEOARROW_CRS_TYPE_NONE;
  metadata_view->crs.data = NULL;
  metadata_view->crs.size_bytes = 0;

  if (metadata.size_bytes == 0) {
    return GEOARROW_OK;
  }

  return GeoArrowMetadataViewInitJSON(metadata_view, error);
}

static int GeoArrowMetadataCrsNeedsEscape(struct GeoArrowStringView crs) {
  return (crs.size_bytes == 0) || (*crs.data != '{' && *crs.data != '"');
}

static int64_t GeoArrowMetadataCalculateSerializedSize(
    const struct GeoArrowMetadataView* metadata_view) {
  const int64_t kSizeOuterBraces = 2;
  const int64_t kSizeQuotes = 2;
  const int64_t kSizeColon = 1;
  const int64_t kSizeComma = 1;
  const int64_t kSizeEdgesKey = 5 + kSizeQuotes + kSizeColon;
  const int64_t kSizeCrsTypeKey = 8 + kSizeQuotes + kSizeColon;
  const int64_t kSizeCrsKey = 3 + kSizeQuotes + kSizeColon;

  int n_keys = 0;
  int64_t size_out = 0;
  size_out += kSizeOuterBraces;

  if (metadata_view->edge_type != GEOARROW_EDGE_TYPE_PLANAR) {
    n_keys += 1;
    size_out += kSizeEdgesKey + kSizeQuotes +
                strlen(GeoArrowEdgeTypeString(metadata_view->edge_type));
  }

  if (metadata_view->crs_type != GEOARROW_CRS_TYPE_UNKNOWN &&
      metadata_view->crs_type != GEOARROW_CRS_TYPE_NONE) {
    n_keys += 1;
    size_out += kSizeCrsTypeKey + kSizeQuotes +
                strlen(GeoArrowCrsTypeString(metadata_view->crs_type));
  }

  if (metadata_view->crs_type != GEOARROW_CRS_TYPE_NONE) {
    n_keys += 1;
    size_out += kSizeCrsKey;

    if (GeoArrowMetadataCrsNeedsEscape(metadata_view->crs)) {
      size_out += kSizeQuotes + metadata_view->crs.size_bytes;
      for (int64_t i = 0; i < metadata_view->crs.size_bytes; i++) {
        char val = metadata_view->crs.data[i];
        size_out += val == '\\' || val == '"';
      }
    } else {
      size_out += metadata_view->crs.size_bytes;
    }
  }

  if (n_keys > 1) {
    size_out += kSizeComma * (n_keys - 1);
  }

  return size_out;
}

static void GeoArrowWriteStringView(struct ArrowStringView sv, char** out) {
  if (sv.size_bytes == 0) {
    return;
  }

  memcpy(*out, sv.data, sv.size_bytes);
  (*out) += sv.size_bytes;
}

static void GeoArrowWriteString(const char* value, char** out) {
  GeoArrowWriteStringView(ArrowCharView(value), out);
}

static int64_t GeoArrowMetadataSerializeInternal(
    const struct GeoArrowMetadataView* metadata_view, char* out) {
  const struct ArrowStringView kEdgesKey = ArrowCharView("\"edges\":");
  const struct ArrowStringView kCrsTypeKey = ArrowCharView("\"crs_type\":");
  const struct ArrowStringView kCrsKey = ArrowCharView("\"crs\":");

  char* out_initial = out;
  int n_keys = 0;

  *out++ = '{';

  if (metadata_view->edge_type != GEOARROW_EDGE_TYPE_PLANAR) {
    n_keys += 1;
    GeoArrowWriteStringView(kEdgesKey, &out);
    *out++ = '"';
    GeoArrowWriteString(GeoArrowEdgeTypeString(metadata_view->edge_type), &out);
    *out++ = '"';
  }

  if (metadata_view->crs_type != GEOARROW_CRS_TYPE_UNKNOWN &&
      metadata_view->crs_type != GEOARROW_CRS_TYPE_NONE) {
    if (n_keys > 0) {
      *out++ = ',';
    }

    n_keys += 1;
    GeoArrowWriteStringView(kCrsTypeKey, &out);
    *out++ = '"';
    GeoArrowWriteString(GeoArrowCrsTypeString(metadata_view->crs_type), &out);
    *out++ = '"';
  }

  if (metadata_view->crs_type != GEOARROW_CRS_TYPE_NONE) {
    if (n_keys > 0) {
      *out++ = ',';
    }

    n_keys += 1;
    GeoArrowWriteStringView(kCrsKey, &out);

    if (GeoArrowMetadataCrsNeedsEscape(metadata_view->crs)) {
      *out++ = '"';
      for (int64_t i = 0; i < metadata_view->crs.size_bytes; i++) {
        char val = metadata_view->crs.data[i];
        if (val == '"') {
          *out++ = '\\';
        }

        *out++ = val;
      }
      *out++ = '"';
    } else {
      struct ArrowStringView sv;
      sv.data = metadata_view->crs.data;
      sv.size_bytes = metadata_view->crs.size_bytes;
      GeoArrowWriteStringView(sv, &out);
    }
  }

  *out++ = '}';
  return out - out_initial;
}

static GeoArrowErrorCode GeoArrowSchemaSetMetadataInternal(
    struct ArrowSchema* schema, const struct GeoArrowMetadataView* metadata_view) {
  int64_t metadata_size = GeoArrowMetadataCalculateSerializedSize(metadata_view);
  char* metadata = (char*)ArrowMalloc(metadata_size);
  if (metadata == NULL) {
    return ENOMEM;
  }

  int64_t chars_written = GeoArrowMetadataSerializeInternal(metadata_view, metadata);
  NANOARROW_DCHECK(chars_written == metadata_size);
  NANOARROW_UNUSED(chars_written);

  struct ArrowBuffer existing_buffer;
  int result = ArrowMetadataBuilderInit(&existing_buffer, schema->metadata);
  if (result != GEOARROW_OK) {
    ArrowFree(metadata);
    return result;
  }

  struct ArrowStringView value;
  value.data = metadata;
  value.size_bytes = metadata_size;
  result = ArrowMetadataBuilderSet(&existing_buffer,
                                   ArrowCharView("ARROW:extension:metadata"), value);
  ArrowFree(metadata);
  if (result != GEOARROW_OK) {
    ArrowBufferReset(&existing_buffer);
    return result;
  }

  result = ArrowSchemaSetMetadata(schema, (const char*)existing_buffer.data);
  ArrowBufferReset(&existing_buffer);
  return result;
}

int64_t GeoArrowMetadataSerialize(const struct GeoArrowMetadataView* metadata_view,
                                  char* out, int64_t n) {
  int64_t metadata_size = GeoArrowMetadataCalculateSerializedSize(metadata_view);
  if (metadata_size <= n) {
    int64_t chars_written = GeoArrowMetadataSerializeInternal(metadata_view, out);
    NANOARROW_DCHECK(chars_written == metadata_size);
    NANOARROW_UNUSED(chars_written);
  }

  // If there is room, write the null terminator
  if (metadata_size < n) {
    out[metadata_size] = '\0';
  }

  return metadata_size;
}

GeoArrowErrorCode GeoArrowSchemaSetMetadata(
    struct ArrowSchema* schema, const struct GeoArrowMetadataView* metadata_view) {
  return GeoArrowSchemaSetMetadataInternal(schema, metadata_view);
}

GeoArrowErrorCode GeoArrowSchemaSetMetadataFrom(struct ArrowSchema* schema,
                                                const struct ArrowSchema* schema_src) {
  struct ArrowSchemaView schema_view;
  NANOARROW_RETURN_NOT_OK(ArrowSchemaViewInit(&schema_view, schema_src, NULL));

  struct ArrowBuffer buffer;
  NANOARROW_RETURN_NOT_OK(ArrowMetadataBuilderInit(&buffer, schema->metadata));
  int result = ArrowMetadataBuilderSet(&buffer, ArrowCharView("ARROW:extension:metadata"),
                                       schema_view.extension_metadata);
  if (result != GEOARROW_OK) {
    ArrowBufferReset(&buffer);
    return result;
  }

  result = ArrowSchemaSetMetadata(schema, (const char*)buffer.data);
  ArrowBufferReset(&buffer);
  return result;
}

int64_t GeoArrowUnescapeCrs(struct GeoArrowStringView crs, char* out, int64_t n) {
  if (crs.size_bytes == 0) {
    if (n > 0) {
      out[0] = '\0';
    }
    return 0;
  }

  if (crs.data[0] != '\"') {
    if (n > crs.size_bytes) {
      memcpy(out, crs.data, crs.size_bytes);
      out[crs.size_bytes] = '\0';
    } else if (out != NULL) {
      memcpy(out, crs.data, n);
    }

    return crs.size_bytes;
  }

  int64_t out_i = 0;
  int is_escape = 0;
  for (int64_t i = 1; i < (crs.size_bytes - 1); i++) {
    if (!is_escape && crs.data[i] == '\\') {
      is_escape = 1;
      continue;
    } else {
      is_escape = 0;
    }

    if (out_i < n) {
      out[out_i] = crs.data[i];
    }

    out_i++;
  }

  if (out_i < n) {
    out[out_i] = '\0';
  }

  return out_i;
}

static const char* kCrsWgs84 =
    "{\"type\":\"GeographicCRS\",\"name\":\"WGS 84 "
    "(CRS84)\",\"datum_ensemble\":{\"name\":\"World Geodetic System 1984 "
    "ensemble\",\"members\":[{\"name\":\"World Geodetic System 1984 "
    "(Transit)\",\"id\":{\"authority\":\"EPSG\",\"code\":1166}},{\"name\":\"World "
    "Geodetic System 1984 "
    "(G730)\",\"id\":{\"authority\":\"EPSG\",\"code\":1152}},{\"name\":\"World Geodetic "
    "System 1984 "
    "(G873)\",\"id\":{\"authority\":\"EPSG\",\"code\":1153}},{\"name\":\"World Geodetic "
    "System 1984 "
    "(G1150)\",\"id\":{\"authority\":\"EPSG\",\"code\":1154}},{\"name\":\"World Geodetic "
    "System 1984 "
    "(G1674)\",\"id\":{\"authority\":\"EPSG\",\"code\":1155}},{\"name\":\"World Geodetic "
    "System 1984 "
    "(G1762)\",\"id\":{\"authority\":\"EPSG\",\"code\":1156}},{\"name\":\"World Geodetic "
    "System 1984 "
    "(G2139)\",\"id\":{\"authority\":\"EPSG\",\"code\":1309}}],\"ellipsoid\":{\"name\":"
    "\"WGS "
    "84\",\"semi_major_axis\":6378137,\"inverse_flattening\":298.257223563},\"accuracy\":"
    "\"2.0\",\"id\":{\"authority\":\"EPSG\",\"code\":6326}},\"coordinate_system\":{"
    "\"subtype\":\"ellipsoidal\",\"axis\":[{\"name\":\"Geodetic "
    "longitude\",\"abbreviation\":\"Lon\",\"direction\":\"east\",\"unit\":\"degree\"},{"
    "\"name\":\"Geodetic "
    "latitude\",\"abbreviation\":\"Lat\",\"direction\":\"north\",\"unit\":\"degree\"}]},"
    "\"scope\":\"Not "
    "known.\",\"area\":\"World.\",\"bbox\":{\"south_latitude\":-90,\"west_longitude\":-"
    "180,\"north_latitude\":90,\"east_longitude\":180},\"id\":{\"authority\":\"OGC\","
    "\"code\":\"CRS84\"}}";

void GeoArrowMetadataSetLonLat(struct GeoArrowMetadataView* metadata_view) {
  metadata_view->crs.data = kCrsWgs84;
  metadata_view->crs.size_bytes = strlen(kCrsWgs84);
  metadata_view->crs_type = GEOARROW_CRS_TYPE_PROJJSON;
}

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "nanoarrow/nanoarrow.h"



static int kernel_start_void(struct GeoArrowKernel* kernel, struct ArrowSchema* schema,
                             const char* options, struct ArrowSchema* out,
                             struct GeoArrowError* error) {
  NANOARROW_UNUSED(kernel);
  NANOARROW_UNUSED(schema);
  NANOARROW_UNUSED(options);
  NANOARROW_UNUSED(error);

  return ArrowSchemaInitFromType(out, NANOARROW_TYPE_NA);
}

static int kernel_push_batch_void(struct GeoArrowKernel* kernel, struct ArrowArray* array,
                                  struct ArrowArray* out, struct GeoArrowError* error) {
  NANOARROW_UNUSED(kernel);
  NANOARROW_UNUSED(array);
  NANOARROW_UNUSED(error);

  struct ArrowArray tmp;
  NANOARROW_RETURN_NOT_OK(ArrowArrayInitFromType(&tmp, NANOARROW_TYPE_NA));
  tmp.length = array->length;
  tmp.null_count = array->length;
  ArrowArrayMove(&tmp, out);
  return NANOARROW_OK;
}

static int kernel_finish_void(struct GeoArrowKernel* kernel, struct ArrowArray* out,
                              struct GeoArrowError* error) {
  NANOARROW_UNUSED(kernel);
  NANOARROW_UNUSED(error);

  if (out != NULL) {
    return EINVAL;
  }

  return NANOARROW_OK;
}

static void kernel_release_void(struct GeoArrowKernel* kernel) { kernel->release = NULL; }

static void GeoArrowKernelInitVoid(struct GeoArrowKernel* kernel) {
  kernel->start = &kernel_start_void;
  kernel->push_batch = &kernel_push_batch_void;
  kernel->finish = &kernel_finish_void;
  kernel->release = &kernel_release_void;
  kernel->private_data = NULL;
}

static int kernel_push_batch_void_agg(struct GeoArrowKernel* kernel,
                                      struct ArrowArray* array, struct ArrowArray* out,
                                      struct GeoArrowError* error) {
  NANOARROW_UNUSED(kernel);
  NANOARROW_UNUSED(array);
  NANOARROW_UNUSED(out);
  NANOARROW_UNUSED(error);

  if (out != NULL) {
    return EINVAL;
  }

  return NANOARROW_OK;
}

static int kernel_finish_void_agg(struct GeoArrowKernel* kernel, struct ArrowArray* out,
                                  struct GeoArrowError* error) {
  NANOARROW_UNUSED(kernel);
  NANOARROW_UNUSED(out);
  NANOARROW_UNUSED(error);

  struct ArrowArray tmp;
  NANOARROW_RETURN_NOT_OK(ArrowArrayInitFromType(&tmp, NANOARROW_TYPE_NA));
  tmp.length = 1;
  tmp.null_count = 1;
  ArrowArrayMove(&tmp, out);
  return NANOARROW_OK;
}

static void GeoArrowKernelInitVoidAgg(struct GeoArrowKernel* kernel) {
  kernel->start = &kernel_start_void;
  kernel->push_batch = &kernel_push_batch_void_agg;
  kernel->finish = &kernel_finish_void_agg;
  kernel->release = &kernel_release_void;
  kernel->private_data = NULL;
}

// Visitor-based kernels
//
// These kernels implement generic operations by visiting each feature in
// the input (since all GeoArrow types including WKB/WKT can be visited).
// This for conversion to/from WKB and WKT whose readers and writers are
// visitor-based. Most other operations are probably faster phrased as
// "cast to GeoArrow in batches then do the thing" (but require these kernels to
// do the "cast to GeoArrow" step).

struct GeoArrowGeometryTypesVisitorPrivate {
  enum GeoArrowGeometryType geometry_type;
  enum GeoArrowDimensions dimensions;
  uint64_t geometry_types_mask;
};

struct GeoArrowBox2DPrivate {
  int feat_null;
  double min_values[2];
  double max_values[2];
  struct ArrowBitmap validity;
  struct ArrowBuffer values[4];
  int64_t null_count;
};

struct GeoArrowVisitorKernelPrivate {
  struct GeoArrowVisitor v;
  int visit_by_feature;
  struct GeoArrowArrayReader reader;
  struct GeoArrowArrayWriter writer;
  struct GeoArrowWKTWriter wkt_writer;
  struct GeoArrowGeometryTypesVisitorPrivate geometry_types_private;
  struct GeoArrowBox2DPrivate box2d_private;
  int (*finish_push_batch)(struct GeoArrowVisitorKernelPrivate* private_data,
                           struct ArrowArray* out, struct GeoArrowError* error);
  int (*finish_start)(struct GeoArrowVisitorKernelPrivate* private_data,
                      struct ArrowSchema* schema, const char* options,
                      struct ArrowSchema* out, struct GeoArrowError* error);
};

static int kernel_get_arg_long(const char* options, const char* key, long* out,
                               int required, struct GeoArrowError* error) {
  struct ArrowStringView type_str;
  type_str.data = NULL;
  type_str.size_bytes = 0;
  NANOARROW_RETURN_NOT_OK(ArrowMetadataGetValue(options, ArrowCharView(key), &type_str));
  if (type_str.data == NULL && required) {
    GeoArrowErrorSet(error, "Missing required parameter '%s'", key);
    return EINVAL;
  } else if (type_str.data == NULL && !required) {
    return NANOARROW_OK;
  }

  char type_str0[16];
  memset(type_str0, 0, sizeof(type_str0));
  snprintf(type_str0, sizeof(type_str0), "%.*s", (int)type_str.size_bytes, type_str.data);
  *out = atoi(type_str0);
  return NANOARROW_OK;
}

static int finish_push_batch_do_nothing(struct GeoArrowVisitorKernelPrivate* private_data,
                                        struct ArrowArray* out,
                                        struct GeoArrowError* error) {
  NANOARROW_UNUSED(private_data);
  NANOARROW_UNUSED(out);
  NANOARROW_UNUSED(error);

  return NANOARROW_OK;
}

static void kernel_release_visitor(struct GeoArrowKernel* kernel) {
  struct GeoArrowVisitorKernelPrivate* private_data =
      (struct GeoArrowVisitorKernelPrivate*)kernel->private_data;
  if (private_data->reader.private_data != NULL) {
    GeoArrowArrayReaderReset(&private_data->reader);
  }

  if (private_data->writer.private_data != NULL) {
    GeoArrowArrayWriterReset(&private_data->writer);
  }

  if (private_data->wkt_writer.private_data != NULL) {
    GeoArrowWKTWriterReset(&private_data->wkt_writer);
  }

  for (int i = 0; i < 4; i++) {
    ArrowBufferReset(&private_data->box2d_private.values[i]);
  }

  ArrowBitmapReset(&private_data->box2d_private.validity);

  ArrowFree(private_data);
  kernel->release = NULL;
}

static int kernel_push_batch(struct GeoArrowKernel* kernel, struct ArrowArray* array,
                             struct ArrowArray* out, struct GeoArrowError* error) {
  struct GeoArrowVisitorKernelPrivate* private_data =
      (struct GeoArrowVisitorKernelPrivate*)kernel->private_data;

  NANOARROW_RETURN_NOT_OK(
      GeoArrowArrayReaderSetArray(&private_data->reader, array, error));

  private_data->v.error = error;
  NANOARROW_RETURN_NOT_OK(GeoArrowArrayReaderVisit(&private_data->reader, 0,
                                                   array->length, &private_data->v));

  return private_data->finish_push_batch(private_data, out, error);
}

static int kernel_push_batch_by_feature(struct GeoArrowKernel* kernel,
                                        struct ArrowArray* array, struct ArrowArray* out,
                                        struct GeoArrowError* error) {
  struct GeoArrowVisitorKernelPrivate* private_data =
      (struct GeoArrowVisitorKernelPrivate*)kernel->private_data;

  NANOARROW_RETURN_NOT_OK(
      GeoArrowArrayReaderSetArray(&private_data->reader, array, error));

  private_data->v.error = error;
  int result;
  for (int64_t i = 0; i < array->length; i++) {
    result = GeoArrowArrayReaderVisit(&private_data->reader, i, 1, &private_data->v);

    if (result == EAGAIN) {
      NANOARROW_RETURN_NOT_OK(private_data->v.feat_end(&private_data->v));
    } else if (result != NANOARROW_OK) {
      return result;
    }
  }

  return private_data->finish_push_batch(private_data, out, error);
}

static int kernel_visitor_start(struct GeoArrowKernel* kernel, struct ArrowSchema* schema,
                                const char* options, struct ArrowSchema* out,
                                struct GeoArrowError* error) {
  struct GeoArrowVisitorKernelPrivate* private_data =
      (struct GeoArrowVisitorKernelPrivate*)kernel->private_data;

  struct GeoArrowSchemaView schema_view;
  NANOARROW_RETURN_NOT_OK(GeoArrowSchemaViewInit(&schema_view, schema, error));

  switch (schema_view.type) {
    case GEOARROW_TYPE_UNINITIALIZED:
      return EINVAL;
    default:
      NANOARROW_RETURN_NOT_OK(
          GeoArrowArrayReaderInitFromSchema(&private_data->reader, schema, error));
      if (private_data->visit_by_feature) {
        kernel->push_batch = &kernel_push_batch_by_feature;
      } else {
        kernel->push_batch = &kernel_push_batch;
      }
      break;
  }

  return private_data->finish_start(private_data, schema, options, out, error);
}

// Kernel visit_void_agg
//
// This kernel visits every feature and returns a single null item at the end.
// This is useful for (1) testing and (2) validating well-known text or well-known
// binary.

static int finish_start_visit_void_agg(struct GeoArrowVisitorKernelPrivate* private_data,
                                       struct ArrowSchema* schema, const char* options,
                                       struct ArrowSchema* out,
                                       struct GeoArrowError* error) {
  NANOARROW_UNUSED(private_data);
  NANOARROW_UNUSED(schema);
  NANOARROW_UNUSED(options);
  NANOARROW_UNUSED(error);

  return ArrowSchemaInitFromType(out, NANOARROW_TYPE_NA);
}

// Kernel format_wkt
//
// Visits every feature in the input and writes the corresponding well-known text output,
// optionally specifying precision and max_element_size_bytes.

static int finish_start_format_wkt(struct GeoArrowVisitorKernelPrivate* private_data,
                                   struct ArrowSchema* schema, const char* options,
                                   struct ArrowSchema* out, struct GeoArrowError* error) {
  NANOARROW_UNUSED(schema);
  NANOARROW_UNUSED(options);

  long precision = private_data->wkt_writer.precision;
  NANOARROW_RETURN_NOT_OK(
      kernel_get_arg_long(options, "precision", &precision, 0, error));
  private_data->wkt_writer.precision = (int)precision;

  long max_element_size_bytes = private_data->wkt_writer.max_element_size_bytes;
  NANOARROW_RETURN_NOT_OK(kernel_get_arg_long(options, "max_element_size_bytes",
                                              &max_element_size_bytes, 0, error));
  private_data->wkt_writer.max_element_size_bytes = max_element_size_bytes;

  GeoArrowWKTWriterInitVisitor(&private_data->wkt_writer, &private_data->v);

  NANOARROW_RETURN_NOT_OK(ArrowSchemaInitFromType(out, NANOARROW_TYPE_STRING));
  return GEOARROW_OK;
}

static int finish_push_batch_format_wkt(struct GeoArrowVisitorKernelPrivate* private_data,
                                        struct ArrowArray* out,
                                        struct GeoArrowError* error) {
  return GeoArrowWKTWriterFinish(&private_data->wkt_writer, out, error);
}

// Kernel as_geoarrow
//
// Visits every feature in the input and writes an array of the specified type.
// Takes option 'type' as the desired integer enum GeoArrowType.

static int finish_start_as_geoarrow(struct GeoArrowVisitorKernelPrivate* private_data,
                                    struct ArrowSchema* schema, const char* options,
                                    struct ArrowSchema* out,
                                    struct GeoArrowError* error) {
  long out_type_long;
  NANOARROW_RETURN_NOT_OK(kernel_get_arg_long(options, "type", &out_type_long, 1, error));
  enum GeoArrowType out_type = (enum GeoArrowType)out_type_long;

  if (private_data->writer.private_data != NULL) {
    GeoArrowErrorSet(error, "Expected exactly one call to start(as_geoarrow)");
    return EINVAL;
  }

  NANOARROW_RETURN_NOT_OK(
      GeoArrowArrayWriterInitFromType(&private_data->writer, out_type));
  NANOARROW_RETURN_NOT_OK(
      GeoArrowArrayWriterInitVisitor(&private_data->writer, &private_data->v));

  struct ArrowSchema tmp;
  NANOARROW_RETURN_NOT_OK(GeoArrowSchemaInitExtension(&tmp, out_type));

  int result = GeoArrowSchemaSetMetadataFrom(&tmp, schema);
  if (result != GEOARROW_OK) {
    GeoArrowErrorSet(error, "GeoArrowSchemaSetMetadataFrom() failed");
    tmp.release(&tmp);
    return result;
  }

  ArrowSchemaMove(&tmp, out);
  return GEOARROW_OK;
}

static int finish_push_batch_as_geoarrow(
    struct GeoArrowVisitorKernelPrivate* private_data, struct ArrowArray* out,
    struct GeoArrowError* error) {
  return GeoArrowArrayWriterFinish(&private_data->writer, out, error);
}

// Kernel unique_geometry_types_agg
//
// This kernel collects all geometry type + dimension combinations in the
// input. EMPTY values are not counted as any particular geometry type;
// however, note that POINTs as represented in WKB or GeoArrow cannot be
// EMPTY and this kernel does not check for the convention of EMPTY as
// all coordinates == nan. This is mostly to facilitate choosing an appropriate
// destination type (e.g., point, linestring, etc.). This visitor is not exposed as a
// standalone visitor in the geoarrow.h header.
//
// The internals use GeoArrowDimensions * 8 + GeoArrowGeometryType as the
// "key" for a given combination. This gives an integer between 0 and 39.
// The types are accumulated in a uint64_t bitmask and translated into the
// corresponding ISO WKB type codes at the end.
static int32_t kGeoArrowGeometryTypeWkbValues[] = {
    -1000, -999, -998, -997, -996, -995, -994, -993, 0,    1,    2,    3,    4,    5,
    6,     7,    1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 2000, 2001, 2002, 2003,
    2004,  2005, 2006, 2007, 3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007};

static int feat_start_geometry_types(struct GeoArrowVisitor* v) {
  struct GeoArrowVisitorKernelPrivate* private_data =
      (struct GeoArrowVisitorKernelPrivate*)v->private_data;
  private_data->geometry_types_private.geometry_type = GEOARROW_GEOMETRY_TYPE_GEOMETRY;
  private_data->geometry_types_private.dimensions = GEOARROW_DIMENSIONS_UNKNOWN;
  return GEOARROW_OK;
}

static int geom_start_geometry_types(struct GeoArrowVisitor* v,
                                     enum GeoArrowGeometryType geometry_type,
                                     enum GeoArrowDimensions dimensions) {
  struct GeoArrowVisitorKernelPrivate* private_data =
      (struct GeoArrowVisitorKernelPrivate*)v->private_data;

  // Only record the first seen geometry type/dimension combination
  if (private_data->geometry_types_private.geometry_type ==
      GEOARROW_GEOMETRY_TYPE_GEOMETRY) {
    private_data->geometry_types_private.geometry_type = geometry_type;
    private_data->geometry_types_private.dimensions = dimensions;
  }

  return GEOARROW_OK;
}

static int coords_geometry_types(struct GeoArrowVisitor* v,
                                 const struct GeoArrowCoordView* coords) {
  if (coords->n_coords > 0) {
    struct GeoArrowVisitorKernelPrivate* private_data =
        (struct GeoArrowVisitorKernelPrivate*)v->private_data;

    // At the first coordinate, add the geometry type to the bitmask
    int bitshift = private_data->geometry_types_private.dimensions * 8 +
                   private_data->geometry_types_private.geometry_type;
    uint64_t bitmask = ((uint64_t)1) << bitshift;
    private_data->geometry_types_private.geometry_types_mask |= bitmask;
    return EAGAIN;
  } else {
    return GEOARROW_OK;
  }
}

static int finish_start_unique_geometry_types_agg(
    struct GeoArrowVisitorKernelPrivate* private_data, struct ArrowSchema* schema,
    const char* options, struct ArrowSchema* out, struct GeoArrowError* error) {
  NANOARROW_UNUSED(schema);
  NANOARROW_UNUSED(options);
  NANOARROW_UNUSED(error);

  private_data->v.feat_start = &feat_start_geometry_types;
  private_data->v.geom_start = &geom_start_geometry_types;
  private_data->v.coords = &coords_geometry_types;
  private_data->v.private_data = private_data;
  return ArrowSchemaInitFromType(out, NANOARROW_TYPE_INT32);
}

static int kernel_finish_unique_geometry_types_agg(struct GeoArrowKernel* kernel,
                                                   struct ArrowArray* out,
                                                   struct GeoArrowError* error) {
  NANOARROW_UNUSED(error);

  struct GeoArrowVisitorKernelPrivate* private_data =
      (struct GeoArrowVisitorKernelPrivate*)kernel->private_data;
  uint64_t result_mask = private_data->geometry_types_private.geometry_types_mask;

  int n_types = 0;
  for (int i = 0; i < 40; i++) {
    uint64_t bitmask = ((uint64_t)1) << i;
    n_types += (result_mask & bitmask) != 0;
  }

  struct ArrowArray tmp;
  NANOARROW_RETURN_NOT_OK(ArrowArrayInitFromType(&tmp, NANOARROW_TYPE_INT32));
  struct ArrowBuffer* data = ArrowArrayBuffer(&tmp, 1);
  int result = ArrowBufferReserve(data, n_types * sizeof(int32_t));
  if (result != NANOARROW_OK) {
    tmp.release(&tmp);
    return result;
  }

  int result_i = 0;
  int32_t* data_int32 = (int32_t*)data->data;
  for (int i = 0; i < 40; i++) {
    uint64_t bitmask = ((uint64_t)1) << i;
    if (result_mask & bitmask) {
      data_int32[result_i++] = kGeoArrowGeometryTypeWkbValues[i];
    }
  }

  result = ArrowArrayFinishBuildingDefault(&tmp, NULL);
  if (result != NANOARROW_OK) {
    tmp.release(&tmp);
    return result;
  }

  tmp.length = n_types;
  tmp.null_count = 0;
  ArrowArrayMove(&tmp, out);
  return GEOARROW_OK;
}

// Kernel box + box_agg
//
// Calculate bounding box values by feature or as an aggregate.
// This visitor is not exposed as a standalone visitor in the geoarrow.h header.

static ArrowErrorCode schema_box(struct ArrowSchema* schema,
                                 struct GeoArrowStringView extension_metadata,
                                 struct GeoArrowError* error) {
  struct GeoArrowMetadataView metadata;
  NANOARROW_RETURN_NOT_OK(GeoArrowMetadataViewInit(&metadata, extension_metadata, error));

  // This bounder only works for planar edges
  if (metadata.edge_type != GEOARROW_EDGE_TYPE_PLANAR) {
    GeoArrowErrorSet(error, "box kernel does not support non-planar edges");
    return EINVAL;
  }

  NANOARROW_RETURN_NOT_OK(GeoArrowSchemaInitExtension(schema, GEOARROW_TYPE_BOX));
  NANOARROW_RETURN_NOT_OK(GeoArrowSchemaSetMetadata(schema, &metadata));
  return GEOARROW_OK;
}

static ArrowErrorCode array_box(struct ArrowArray* array) {
  NANOARROW_RETURN_NOT_OK(ArrowArrayInitFromType(array, NANOARROW_TYPE_STRUCT));
  NANOARROW_RETURN_NOT_OK(ArrowArrayAllocateChildren(array, 4));
  for (int i = 0; i < 4; i++) {
    NANOARROW_RETURN_NOT_OK(
        ArrowArrayInitFromType(array->children[i], NANOARROW_TYPE_DOUBLE));
  }

  return GEOARROW_OK;
}

static ArrowErrorCode box_flush(struct GeoArrowVisitorKernelPrivate* private_data) {
  NANOARROW_RETURN_NOT_OK(ArrowBufferAppendDouble(
      &private_data->box2d_private.values[0], private_data->box2d_private.min_values[0]));
  NANOARROW_RETURN_NOT_OK(ArrowBufferAppendDouble(
      &private_data->box2d_private.values[1], private_data->box2d_private.min_values[1]));
  NANOARROW_RETURN_NOT_OK(ArrowBufferAppendDouble(
      &private_data->box2d_private.values[2], private_data->box2d_private.max_values[0]));
  NANOARROW_RETURN_NOT_OK(ArrowBufferAppendDouble(
      &private_data->box2d_private.values[3], private_data->box2d_private.max_values[1]));

  return NANOARROW_OK;
}

static ArrowErrorCode box_finish(struct GeoArrowVisitorKernelPrivate* private_data,
                                 struct ArrowArray* out, struct ArrowError* error) {
  struct ArrowArray tmp;
  tmp.release = NULL;
  int result = array_box(&tmp);
  if (result != GEOARROW_OK) {
    if (tmp.release != NULL) {
      tmp.release(&tmp);
    }
  }

  int64_t length = private_data->box2d_private.values[0].size_bytes / sizeof(double);

  for (int i = 0; i < 4; i++) {
    NANOARROW_RETURN_NOT_OK(
        ArrowArraySetBuffer(tmp.children[i], 1, &private_data->box2d_private.values[i]));
    tmp.children[i]->length = length;
  }

  tmp.length = length;
  if (private_data->box2d_private.null_count > 0) {
    ArrowArraySetValidityBitmap(&tmp, &private_data->box2d_private.validity);
  } else {
    ArrowBitmapReset(&private_data->box2d_private.validity);
  }

  result = ArrowArrayFinishBuildingDefault(&tmp, ((struct ArrowError*)error));
  if (result != GEOARROW_OK) {
    tmp.release(&tmp);
    return result;
  }

  tmp.null_count = private_data->box2d_private.null_count;
  private_data->box2d_private.null_count = 0;
  ArrowArrayMove(&tmp, out);
  return GEOARROW_OK;
}

static int feat_start_box(struct GeoArrowVisitor* v) {
  struct GeoArrowVisitorKernelPrivate* private_data =
      (struct GeoArrowVisitorKernelPrivate*)v->private_data;
  private_data->box2d_private.max_values[0] = -INFINITY;
  private_data->box2d_private.max_values[1] = -INFINITY;
  private_data->box2d_private.min_values[0] = INFINITY;
  private_data->box2d_private.min_values[1] = INFINITY;
  private_data->box2d_private.feat_null = 0;
  return GEOARROW_OK;
}

static int null_feat_box(struct GeoArrowVisitor* v) {
  struct GeoArrowVisitorKernelPrivate* private_data =
      (struct GeoArrowVisitorKernelPrivate*)v->private_data;
  private_data->box2d_private.feat_null = 1;
  return GEOARROW_OK;
}

static int coords_box(struct GeoArrowVisitor* v, const struct GeoArrowCoordView* coords) {
  struct GeoArrowVisitorKernelPrivate* private_data =
      (struct GeoArrowVisitorKernelPrivate*)v->private_data;

  double value;
  for (int dim_i = 0; dim_i < 2; dim_i++) {
    for (int64_t i = 0; i < coords->n_coords; i++) {
      value = GEOARROW_COORD_VIEW_VALUE(coords, i, dim_i);
      if (value < private_data->box2d_private.min_values[dim_i]) {
        private_data->box2d_private.min_values[dim_i] = value;
      }

      if (value > private_data->box2d_private.max_values[dim_i]) {
        private_data->box2d_private.max_values[dim_i] = value;
      }
    }
  }

  return GEOARROW_OK;
}

static int feat_end_box(struct GeoArrowVisitor* v) {
  struct GeoArrowVisitorKernelPrivate* private_data =
      (struct GeoArrowVisitorKernelPrivate*)v->private_data;

  if (private_data->box2d_private.feat_null) {
    if (private_data->box2d_private.validity.buffer.data == NULL) {
      int64_t length = private_data->box2d_private.values[0].size_bytes / sizeof(double);
      NANOARROW_RETURN_NOT_OK(
          ArrowBitmapAppend(&private_data->box2d_private.validity, 1, length));
    }

    NANOARROW_RETURN_NOT_OK(
        ArrowBitmapAppend(&private_data->box2d_private.validity, 0, 1));
    private_data->box2d_private.null_count++;
  } else if (private_data->box2d_private.validity.buffer.data != NULL) {
    NANOARROW_RETURN_NOT_OK(
        ArrowBitmapAppend(&private_data->box2d_private.validity, 1, 1));
  }

  NANOARROW_RETURN_NOT_OK(box_flush(private_data));
  return GEOARROW_OK;
}

static int finish_start_box_agg(struct GeoArrowVisitorKernelPrivate* private_data,
                                struct ArrowSchema* schema, const char* options,
                                struct ArrowSchema* out, struct GeoArrowError* error) {
  NANOARROW_UNUSED(options);
  NANOARROW_UNUSED(error);

  private_data->v.coords = &coords_box;
  private_data->v.private_data = private_data;

  private_data->box2d_private.max_values[0] = -INFINITY;
  private_data->box2d_private.max_values[1] = -INFINITY;
  private_data->box2d_private.min_values[0] = INFINITY;
  private_data->box2d_private.min_values[1] = INFINITY;
  private_data->box2d_private.feat_null = 0;

  struct GeoArrowSchemaView schema_view;
  NANOARROW_RETURN_NOT_OK(GeoArrowSchemaViewInit(&schema_view, schema, error));
  NANOARROW_RETURN_NOT_OK(schema_box(out, schema_view.extension_metadata, error));
  return GEOARROW_OK;
}

static int kernel_finish_box_agg(struct GeoArrowKernel* kernel, struct ArrowArray* out,
                                 struct GeoArrowError* error) {
  struct GeoArrowVisitorKernelPrivate* private_data =
      (struct GeoArrowVisitorKernelPrivate*)kernel->private_data;

  NANOARROW_RETURN_NOT_OK(box_flush(private_data));
  NANOARROW_RETURN_NOT_OK(box_finish(private_data, out, (struct ArrowError*)error));
  return GEOARROW_OK;
}

static int finish_start_box(struct GeoArrowVisitorKernelPrivate* private_data,
                            struct ArrowSchema* schema, const char* options,
                            struct ArrowSchema* out, struct GeoArrowError* error) {
  NANOARROW_UNUSED(options);
  NANOARROW_UNUSED(error);

  private_data->v.feat_start = &feat_start_box;
  private_data->v.null_feat = &null_feat_box;
  private_data->v.coords = &coords_box;
  private_data->v.feat_end = &feat_end_box;
  private_data->v.private_data = private_data;

  struct GeoArrowSchemaView schema_view;
  NANOARROW_RETURN_NOT_OK(GeoArrowSchemaViewInit(&schema_view, schema, error));
  NANOARROW_RETURN_NOT_OK(schema_box(out, schema_view.extension_metadata, error));
  return GEOARROW_OK;
}

static int finish_push_batch_box(struct GeoArrowVisitorKernelPrivate* private_data,
                                 struct ArrowArray* out, struct GeoArrowError* error) {
  NANOARROW_RETURN_NOT_OK(box_finish(private_data, out, (struct ArrowError*)error));
  return GEOARROW_OK;
}

static int GeoArrowInitVisitorKernelInternal(struct GeoArrowKernel* kernel,
                                             const char* name) {
  struct GeoArrowVisitorKernelPrivate* private_data =
      (struct GeoArrowVisitorKernelPrivate*)ArrowMalloc(
          sizeof(struct GeoArrowVisitorKernelPrivate));
  if (private_data == NULL) {
    return ENOMEM;
  }

  memset(private_data, 0, sizeof(struct GeoArrowVisitorKernelPrivate));
  private_data->finish_push_batch = &finish_push_batch_do_nothing;
  GeoArrowVisitorInitVoid(&private_data->v);
  private_data->visit_by_feature = 0;

  ArrowBitmapInit(&private_data->box2d_private.validity);
  for (int i = 0; i < 4; i++) {
    ArrowBufferInit(&private_data->box2d_private.values[i]);
  }

  int result = GEOARROW_OK;

  if (strcmp(name, "visit_void_agg") == 0) {
    kernel->finish = &kernel_finish_void_agg;
    private_data->finish_start = &finish_start_visit_void_agg;
  } else if (strcmp(name, "format_wkt") == 0) {
    kernel->finish = &kernel_finish_void;
    private_data->finish_start = &finish_start_format_wkt;
    private_data->finish_push_batch = &finish_push_batch_format_wkt;
    result = GeoArrowWKTWriterInit(&private_data->wkt_writer);
    private_data->visit_by_feature = 1;
  } else if (strcmp(name, "as_geoarrow") == 0) {
    kernel->finish = &kernel_finish_void;
    private_data->finish_start = &finish_start_as_geoarrow;
    private_data->finish_push_batch = &finish_push_batch_as_geoarrow;
  } else if (strcmp(name, "unique_geometry_types_agg") == 0) {
    kernel->finish = &kernel_finish_unique_geometry_types_agg;
    private_data->finish_start = &finish_start_unique_geometry_types_agg;
    private_data->visit_by_feature = 1;
  } else if (strcmp(name, "box") == 0) {
    kernel->finish = &kernel_finish_void;
    private_data->finish_start = &finish_start_box;
    private_data->finish_push_batch = &finish_push_batch_box;
  } else if (strcmp(name, "box_agg") == 0) {
    kernel->finish = &kernel_finish_box_agg;
    private_data->finish_start = &finish_start_box_agg;
  }

  if (result != GEOARROW_OK) {
    ArrowFree(private_data);
    return result;
  }

  kernel->start = &kernel_visitor_start;
  kernel->push_batch = &kernel_push_batch_void_agg;
  kernel->release = &kernel_release_visitor;
  kernel->private_data = private_data;

  return GEOARROW_OK;
}

GeoArrowErrorCode GeoArrowKernelInit(struct GeoArrowKernel* kernel, const char* name,
                                     const char* options) {
  NANOARROW_UNUSED(options);

  if (strcmp(name, "void") == 0) {
    GeoArrowKernelInitVoid(kernel);
    return NANOARROW_OK;
  } else if (strcmp(name, "void_agg") == 0) {
    GeoArrowKernelInitVoidAgg(kernel);
    return NANOARROW_OK;
  } else if (strcmp(name, "visit_void_agg") == 0) {
    return GeoArrowInitVisitorKernelInternal(kernel, name);
  } else if (strcmp(name, "format_wkt") == 0) {
    return GeoArrowInitVisitorKernelInternal(kernel, name);
  } else if (strcmp(name, "as_geoarrow") == 0) {
    return GeoArrowInitVisitorKernelInternal(kernel, name);
  } else if (strcmp(name, "unique_geometry_types_agg") == 0) {
    return GeoArrowInitVisitorKernelInternal(kernel, name);
  } else if (strcmp(name, "box") == 0) {
    return GeoArrowInitVisitorKernelInternal(kernel, name);
  } else if (strcmp(name, "box_agg") == 0) {
    return GeoArrowInitVisitorKernelInternal(kernel, name);
  }

  return ENOTSUP;
}

#include <string.h>

#include "nanoarrow/nanoarrow.h"



struct BuilderPrivate {
  // The ArrowSchema (without extension) for this builder
  struct ArrowSchema schema;

  // The ArrowArray responsible for owning the memory
  struct ArrowArray array;

  // Cached pointers pointing inside the array's private data
  // Depending on what exactly is being built, these pointers
  // might be NULL.
  struct ArrowBitmap* validity;
  struct ArrowBuffer* buffers[9];
};

static ArrowErrorCode GeoArrowBuilderInitArrayAndCachePointers(
    struct GeoArrowBuilder* builder) {
  struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;

  NANOARROW_RETURN_NOT_OK(
      ArrowArrayInitFromSchema(&private->array, &private->schema, NULL));

  private->validity = ArrowArrayValidityBitmap(&private->array);

  struct _GeoArrowFindBufferResult res;
  for (int64_t i = 0; i < builder->view.n_buffers; i++) {
    res.array = NULL;
    _GeoArrowArrayFindBuffer(&private->array, &res, i, 0, 0);
    if (res.array == NULL) {
      return EINVAL;
    }

    private->buffers[i] = ArrowArrayBuffer(res.array, res.i);
    builder->view.buffers[i].data.as_uint8 = NULL;
    builder->view.buffers[i].size_bytes = 0;
    builder->view.buffers[i].capacity_bytes = 0;
  }

  // Reset the coordinate counts and values
  builder->view.coords.size_coords = 0;
  builder->view.coords.capacity_coords = 0;
  for (int i = 0; i < 4; i++) {
    builder->view.coords.values[i] = NULL;
  }

  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowBuilderInitInternal(struct GeoArrowBuilder* builder) {
  enum GeoArrowType type = builder->view.schema_view.type;

  // View types aren't supported by the visitor nor the buffer builder
  switch (type) {
    case GEOARROW_TYPE_WKB_VIEW:
    case GEOARROW_TYPE_WKT_VIEW:
      return ENOTSUP;
    default:
      break;
  }

  // Initialize an array view to help set some fields
  struct GeoArrowArrayView array_view;
  NANOARROW_RETURN_NOT_OK(GeoArrowArrayViewInitFromType(&array_view, type));

  struct BuilderPrivate* private =
      (struct BuilderPrivate*)ArrowMalloc(sizeof(struct BuilderPrivate));
  if (private == NULL) {
    return ENOMEM;
  }

  memset(private, 0, sizeof(struct BuilderPrivate));
  builder->private_data = private;

  // Initialize our copy of the schema for the storage type
  int result = GeoArrowSchemaInit(&private->schema, type);
  if (result != GEOARROW_OK) {
    ArrowFree(private);
    builder->private_data = NULL;
    return result;
  }

  // Update a few things about the writable view from the regular view
  // that never change.
  builder->view.coords.n_values = array_view.coords.n_values;
  builder->view.coords.coords_stride = array_view.coords.coords_stride;
  builder->view.n_offsets = array_view.n_offsets;
  switch (builder->view.schema_view.coord_type) {
    case GEOARROW_COORD_TYPE_SEPARATE:
      builder->view.n_buffers = 1 + array_view.n_offsets + array_view.coords.n_values;
      break;

    // interleaved + WKB + WKT
    default:
      builder->view.n_buffers = 1 + array_view.n_offsets + 1;
      break;
  }

  // Initialize an empty array; cache the ArrowBitmap and ArrowBuffer pointers we need
  result = GeoArrowBuilderInitArrayAndCachePointers(builder);
  if (result != GEOARROW_OK) {
    private->schema.release(&private->schema);
    ArrowFree(private);
    builder->private_data = NULL;
    return result;
  }

  return GEOARROW_OK;
}

GeoArrowErrorCode GeoArrowBuilderInitFromType(struct GeoArrowBuilder* builder,
                                              enum GeoArrowType type) {
  memset(builder, 0, sizeof(struct GeoArrowBuilder));
  NANOARROW_RETURN_NOT_OK(
      GeoArrowSchemaViewInitFromType(&builder->view.schema_view, type));
  return GeoArrowBuilderInitInternal(builder);
}

GeoArrowErrorCode GeoArrowBuilderInitFromSchema(struct GeoArrowBuilder* builder,
                                                const struct ArrowSchema* schema,
                                                struct GeoArrowError* error) {
  memset(builder, 0, sizeof(struct GeoArrowBuilder));
  NANOARROW_RETURN_NOT_OK(
      GeoArrowSchemaViewInit(&builder->view.schema_view, schema, error));
  return GeoArrowBuilderInitInternal(builder);
}

GeoArrowErrorCode GeoArrowBuilderReserveBuffer(struct GeoArrowBuilder* builder, int64_t i,
                                               int64_t additional_size_bytes) {
  struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;
  struct ArrowBuffer* buffer_src = private->buffers[i];
  struct GeoArrowWritableBufferView* buffer_dst = builder->view.buffers + i;

  // Sync any changes from the builder's view of the buffer to nanoarrow's
  buffer_src->size_bytes = buffer_dst->size_bytes;

  // Use nanoarrow's reserve
  NANOARROW_RETURN_NOT_OK(ArrowBufferReserve(buffer_src, additional_size_bytes));

  // Sync any changes back to the builder's view
  builder->view.buffers[i].data.data = buffer_src->data;
  builder->view.buffers[i].capacity_bytes = buffer_src->capacity_bytes;
  return GEOARROW_OK;
}

struct GeoArrowBufferDeallocatorPrivate {
  void (*custom_free)(uint8_t* ptr, int64_t size, void* private_data);
  void* private_data;
};

static void GeoArrowBufferDeallocateWrapper(struct ArrowBufferAllocator* allocator,
                                            uint8_t* ptr, int64_t size) {
  struct GeoArrowBufferDeallocatorPrivate* private_data =
      (struct GeoArrowBufferDeallocatorPrivate*)allocator->private_data;
  private_data->custom_free(ptr, size, private_data->private_data);
  ArrowFree(private_data);
}

GeoArrowErrorCode GeoArrowBuilderSetOwnedBuffer(
    struct GeoArrowBuilder* builder, int64_t i, struct GeoArrowBufferView value,
    void (*custom_free)(uint8_t* ptr, int64_t size, void* private_data),
    void* private_data) {
  if (i < 0 || i >= builder->view.n_buffers) {
    return EINVAL;
  }

  struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;
  struct ArrowBuffer* buffer_src = private->buffers[i];

  struct GeoArrowBufferDeallocatorPrivate* deallocator =
      (struct GeoArrowBufferDeallocatorPrivate*)ArrowMalloc(
          sizeof(struct GeoArrowBufferDeallocatorPrivate));
  if (deallocator == NULL) {
    return ENOMEM;
  }

  deallocator->custom_free = custom_free;
  deallocator->private_data = private_data;

  ArrowBufferReset(buffer_src);
  buffer_src->allocator =
      ArrowBufferDeallocator(&GeoArrowBufferDeallocateWrapper, deallocator);
  buffer_src->data = (uint8_t*)value.data;
  buffer_src->size_bytes = value.size_bytes;
  buffer_src->capacity_bytes = value.size_bytes;

  // Sync this information to the writable view
  builder->view.buffers[i].data.data = buffer_src->data;
  builder->view.buffers[i].size_bytes = buffer_src->size_bytes;
  builder->view.buffers[i].capacity_bytes = buffer_src->capacity_bytes;

  return GEOARROW_OK;
}

static void GeoArrowSetArrayLengthFromBufferLength(struct GeoArrowSchemaView* schema_view,
                                                   struct _GeoArrowFindBufferResult* res,
                                                   int64_t size_bytes);

static void GeoArrowSetCoordContainerLength(struct GeoArrowBuilder* builder);

GeoArrowErrorCode GeoArrowBuilderFinish(struct GeoArrowBuilder* builder,
                                        struct ArrowArray* array,
                                        struct GeoArrowError* error) {
  struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;

  // If the coordinate appender was used, we may need to update the buffer sizes
  struct GeoArrowWritableCoordView* writable_view = &builder->view.coords;
  int64_t last_buffer = builder->view.n_buffers - 1;
  int n_values = writable_view->n_values;
  int64_t size_by_coords;

  switch (builder->view.schema_view.coord_type) {
    case GEOARROW_COORD_TYPE_INTERLEAVED:
      size_by_coords = writable_view->size_coords * sizeof(double) * n_values;
      if (size_by_coords > builder->view.buffers[last_buffer].size_bytes) {
        builder->view.buffers[last_buffer].size_bytes = size_by_coords;
      }
      break;

    case GEOARROW_COORD_TYPE_SEPARATE:
      for (int64_t i = last_buffer - n_values + 1; i <= last_buffer; i++) {
        size_by_coords = writable_view->size_coords * sizeof(double);
        if (size_by_coords > builder->view.buffers[i].size_bytes) {
          builder->view.buffers[i].size_bytes = size_by_coords;
        }
      }
      break;

    default:
      break;
  }

  // If the validity bitmap was used, we need to update the validity buffer size
  if (private->validity->buffer.data != NULL &&
      builder->view.buffers[0].data.data == NULL) {
    builder->view.buffers[0].data.as_uint8 = private->validity->buffer.data;
    builder->view.buffers[0].size_bytes = private->validity->buffer.size_bytes;
    builder->view.buffers[0].capacity_bytes = private->validity->buffer.capacity_bytes;
  }

  // Sync builder's buffers back to the array; set array lengths from buffer sizes
  struct _GeoArrowFindBufferResult res;
  for (int64_t i = 0; i < builder->view.n_buffers; i++) {
    private->buffers[i]->size_bytes = builder->view.buffers[i].size_bytes;

    res.array = NULL;
    _GeoArrowArrayFindBuffer(&private->array, &res, i, 0, 0);
    if (res.array == NULL) {
      return EINVAL;
    }

    GeoArrowSetArrayLengthFromBufferLength(&builder->view.schema_view, &res,
                                           private->buffers[i]->size_bytes);
  }

  // Set the struct or fixed-size list container length
  GeoArrowSetCoordContainerLength(builder);

  // Call finish building, which will flush the buffer pointers into the array
  // and validate sizes.
  NANOARROW_RETURN_NOT_OK(
      ArrowArrayFinishBuildingDefault(&private->array, (struct ArrowError*)error));

  // If the first buffer is non-null, we don't know what it is
  if (private->array.buffers[0] != NULL) {
    private->array.null_count = -1;
  }

  // Move the result out of private so we can maybe prepare for the next round
  struct ArrowArray tmp;
  ArrowArrayMove(&private->array, &tmp);

  // Prepare for another round of building
  int result = GeoArrowBuilderInitArrayAndCachePointers(builder);
  if (result != GEOARROW_OK) {
    tmp.release(&tmp);
    return result;
  }

  // Move the result
  ArrowArrayMove(&tmp, array);
  return GEOARROW_OK;
}

void GeoArrowBuilderReset(struct GeoArrowBuilder* builder) {
  if (builder->private_data != NULL) {
    struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;

    if (private->schema.release != NULL) {
      private->schema.release(&private->schema);
    }

    if (private->array.release != NULL) {
      private->array.release(&private->array);
    }

    ArrowFree(private);
    builder->private_data = NULL;
  }
}

static void GeoArrowSetArrayLengthFromBufferLength(struct GeoArrowSchemaView* schema_view,
                                                   struct _GeoArrowFindBufferResult* res,
                                                   int64_t size_bytes) {
  // By luck, buffer index 1 for every array is the one we use to infer the length;
  // however, this is a slightly different formula for each type/depth
  if (res->i != 1) {
    return;
  }

  // ...but in all cases, if the size is 0, the length is 0
  if (size_bytes == 0) {
    res->array->length = 0;
    return;
  }

  switch (schema_view->type) {
    case GEOARROW_TYPE_WKB:
    case GEOARROW_TYPE_WKT:
      res->array->length = (size_bytes / sizeof(int32_t)) - 1;
      return;
    case GEOARROW_TYPE_LARGE_WKB:
    case GEOARROW_TYPE_LARGE_WKT:
      res->array->length = (size_bytes / sizeof(int64_t)) - 1;
      return;
    default:
      break;
  }

  int coord_level;
  switch (schema_view->geometry_type) {
    case GEOARROW_GEOMETRY_TYPE_BOX:
    case GEOARROW_GEOMETRY_TYPE_POINT:
      coord_level = 0;
      break;
    case GEOARROW_GEOMETRY_TYPE_LINESTRING:
    case GEOARROW_GEOMETRY_TYPE_MULTIPOINT:
      coord_level = 1;
      break;
    case GEOARROW_GEOMETRY_TYPE_POLYGON:
    case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING:
      coord_level = 2;
      break;
    case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON:
      coord_level = 3;
      break;
    default:
      return;
  }

  if (res->level < coord_level) {
    // This is an offset buffer
    res->array->length = (size_bytes / sizeof(int32_t)) - 1;
  } else {
    // This is a data buffer
    res->array->length = size_bytes / sizeof(double);
  }
}

static void GeoArrowSetCoordContainerLength(struct GeoArrowBuilder* builder) {
  struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;

  // At this point all the array lengths should be set except for the
  // fixed-size list or struct parent to the coordinate array(s).
  int scale = -1;
  switch (builder->view.schema_view.coord_type) {
    case GEOARROW_COORD_TYPE_SEPARATE:
      scale = 1;
      break;
    case GEOARROW_COORD_TYPE_INTERLEAVED:
      switch (builder->view.schema_view.dimensions) {
        case GEOARROW_DIMENSIONS_XY:
          scale = 2;
          break;
        case GEOARROW_DIMENSIONS_XYZ:
        case GEOARROW_DIMENSIONS_XYM:
          scale = 3;
          break;
        case GEOARROW_DIMENSIONS_XYZM:
          scale = 4;
          break;
        default:
          return;
      }
      break;
    default:
      // e.g., WKB
      break;
  }

  switch (builder->view.schema_view.geometry_type) {
    case GEOARROW_GEOMETRY_TYPE_BOX:
    case GEOARROW_GEOMETRY_TYPE_POINT:
      private
      ->array.length = private->array.children[0]->length / scale;
      break;
    case GEOARROW_GEOMETRY_TYPE_LINESTRING:
    case GEOARROW_GEOMETRY_TYPE_MULTIPOINT:
      private
      ->array.children[0]->length =
          private->array.children[0]->children[0]->length / scale;
      break;
    case GEOARROW_GEOMETRY_TYPE_POLYGON:
    case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING:
      private
      ->array.children[0]->children[0]->length =
          private->array.children[0]->children[0]->children[0]->length / scale;
      break;
    case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON:
      private
      ->array.children[0]->children[0]->children[0]->length =
          private->array.children[0]->children[0]->children[0]->children[0]->length /
          scale;
      break;
    default:
      // e.g., WKB
      break;
  }
}

#include <errno.h>



#include "nanoarrow/nanoarrow.h"

static int32_t kZeroInt32 = 0;

static int GeoArrowArrayViewInitInternal(struct GeoArrowArrayView* array_view) {
  switch (array_view->schema_view.geometry_type) {
    case GEOARROW_GEOMETRY_TYPE_BOX:
    case GEOARROW_GEOMETRY_TYPE_POINT:
      array_view->n_offsets = 0;
      break;
    case GEOARROW_GEOMETRY_TYPE_LINESTRING:
    case GEOARROW_GEOMETRY_TYPE_MULTIPOINT:
      array_view->n_offsets = 1;
      break;
    case GEOARROW_GEOMETRY_TYPE_POLYGON:
    case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING:
      array_view->n_offsets = 2;
      break;
    case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON:
      array_view->n_offsets = 3;
      break;
    default:
      // i.e., serialized type
      array_view->n_offsets = 1;
      break;
  }

  for (int i = 0; i < 4; i++) {
    array_view->length[i] = 0;
    array_view->offset[i] = 0;
  }

  array_view->validity_bitmap = NULL;
  for (int i = 0; i < 3; i++) {
    array_view->offsets[i] = NULL;
  }
  array_view->data = NULL;

  array_view->coords.n_coords = 0;
  switch (array_view->schema_view.dimensions) {
    case GEOARROW_DIMENSIONS_XY:
      array_view->coords.n_values = 2;
      break;
    case GEOARROW_DIMENSIONS_XYZ:
    case GEOARROW_DIMENSIONS_XYM:
      array_view->coords.n_values = 3;
      break;
    case GEOARROW_DIMENSIONS_XYZM:
      array_view->coords.n_values = 4;
      break;
    default:
      // i.e., serialized type
      array_view->coords.n_coords = 0;
      break;
  }

  if (array_view->schema_view.geometry_type == GEOARROW_GEOMETRY_TYPE_BOX) {
    array_view->coords.n_values *= 2;
  }

  switch (array_view->schema_view.coord_type) {
    case GEOARROW_COORD_TYPE_SEPARATE:
      array_view->coords.coords_stride = 1;
      break;
    case GEOARROW_COORD_TYPE_INTERLEAVED:
      array_view->coords.coords_stride = array_view->coords.n_values;
      break;
    default:
      // i.e., serialized type
      array_view->coords.coords_stride = 0;
      break;
  }

  return GEOARROW_OK;
}

GeoArrowErrorCode GeoArrowArrayViewInitFromType(struct GeoArrowArrayView* array_view,
                                                enum GeoArrowType type) {
  memset(array_view, 0, sizeof(struct GeoArrowArrayView));
  NANOARROW_RETURN_NOT_OK(GeoArrowSchemaViewInitFromType(&array_view->schema_view, type));
  return GeoArrowArrayViewInitInternal(array_view);
}

GeoArrowErrorCode GeoArrowArrayViewInitFromSchema(struct GeoArrowArrayView* array_view,
                                                  const struct ArrowSchema* schema,
                                                  struct GeoArrowError* error) {
  memset(array_view, 0, sizeof(struct GeoArrowArrayView));
  NANOARROW_RETURN_NOT_OK(
      GeoArrowSchemaViewInit(&array_view->schema_view, schema, error));
  return GeoArrowArrayViewInitInternal(array_view);
}

static int GeoArrowArrayViewSetArrayInternal(struct GeoArrowArrayView* array_view,
                                             const struct ArrowArray* array,
                                             struct GeoArrowError* error, int level) {
  // Set offset + length of the array
  array_view->offset[level] = array->offset;
  array_view->length[level] = array->length;

  if (level == array_view->n_offsets) {
    // We're at the coord array!

    // n_coords is last_offset[level - 1] or array->length if level == 0
    if (level > 0) {
      int32_t first_offset = array_view->first_offset[level - 1];
      array_view->coords.n_coords = array_view->last_offset[level - 1] - first_offset;
    } else {
      array_view->coords.n_coords = array->length;
    }

    switch (array_view->schema_view.coord_type) {
      case GEOARROW_COORD_TYPE_SEPARATE:
        if (array->n_children != array_view->coords.n_values) {
          GeoArrowErrorSet(error,
                           "Unexpected number of children for struct coordinate array "
                           "in GeoArrowArrayViewSetArray()");
          return EINVAL;
        }

        // Set the coord pointers to the data buffer of each child (applying
        // offset before assigning the pointer)
        for (int32_t i = 0; i < array_view->coords.n_values; i++) {
          if (array->children[i]->n_buffers != 2) {
            ArrowErrorSet(
                (struct ArrowError*)error,
                "Unexpected number of buffers for struct coordinate array child "
                "in GeoArrowArrayViewSetArray()");
            return EINVAL;
          }

          array_view->coords.values[i] = ((const double*)array->children[i]->buffers[1]) +
                                         array->children[i]->offset;
        }

        break;

      case GEOARROW_COORD_TYPE_INTERLEAVED:
        if (array->n_children != 1) {
          GeoArrowErrorSet(
              error,
              "Unexpected number of children for interleaved coordinate array "
              "in GeoArrowArrayViewSetArray()");
          return EINVAL;
        }

        if (array->children[0]->n_buffers != 2) {
          ArrowErrorSet(
              (struct ArrowError*)error,
              "Unexpected number of buffers for interleaved coordinate array child "
              "in GeoArrowArrayViewSetArray()");
          return EINVAL;
        }

        // Set the coord pointers to the first four doubles in the data buffers

        for (int32_t i = 0; i < array_view->coords.n_values; i++) {
          array_view->coords.values[i] = ((const double*)array->children[0]->buffers[1]) +
                                         array->children[0]->offset + i;
        }

        break;

      default:
        GeoArrowErrorSet(error, "Unexpected coordinate type GeoArrowArrayViewSetArray()");
        return EINVAL;
    }

    return GEOARROW_OK;
  }

  if (array->n_buffers != 2) {
    ArrowErrorSet(
        (struct ArrowError*)error,
        "Unexpected number of buffers in list array in GeoArrowArrayViewSetArray()");
    return EINVAL;
  }

  if (array->n_children != 1) {
    ArrowErrorSet(
        (struct ArrowError*)error,
        "Unexpected number of children in list array in GeoArrowArrayViewSetArray()");
    return EINVAL;
  }

  // Set the offsets buffer and the last_offset value of level
  if (array->length > 0) {
    array_view->offsets[level] = (const int32_t*)array->buffers[1];
    array_view->first_offset[level] = array_view->offsets[level][array->offset];
    array_view->last_offset[level] =
        array_view->offsets[level][array->offset + array->length];
  } else {
    array_view->offsets[level] = &kZeroInt32;
    array_view->first_offset[level] = 0;
    array_view->last_offset[level] = 0;
  }

  return GeoArrowArrayViewSetArrayInternal(array_view, array->children[0], error,
                                           level + 1);
}

static GeoArrowErrorCode GeoArrowArrayViewSetArraySerialized(
    struct GeoArrowArrayView* array_view, const struct ArrowArray* array) {
  array_view->length[0] = array->length;
  array_view->offset[0] = array->offset;

  array_view->offsets[0] = (const int32_t*)array->buffers[1];
  array_view->data = (const uint8_t*)array->buffers[2];
  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowArrayViewSetArrayBox(
    struct GeoArrowArrayView* array_view, const struct ArrowArray* array,
    struct GeoArrowError* error) {
  array_view->length[0] = array->length;
  array_view->offset[0] = array->offset;
  array_view->coords.n_coords = array->length;

  if (array->n_children != array_view->coords.n_values) {
    GeoArrowErrorSet(error,
                     "Unexpected number of children for box array struct "
                     "in GeoArrowArrayViewSetArray()");
    return EINVAL;
  }

  // Set the coord pointers to the data buffer of each child (applying
  // offset before assigning the pointer)
  for (int32_t i = 0; i < array_view->coords.n_values; i++) {
    if (array->children[i]->n_buffers != 2) {
      ArrowErrorSet((struct ArrowError*)error,
                    "Unexpected number of buffers for box array child "
                    "in GeoArrowArrayViewSetArray()");
      return EINVAL;
    }

    array_view->coords.values[i] =
        ((const double*)array->children[i]->buffers[1]) + array->children[i]->offset;
  }

  return GEOARROW_OK;
}

GeoArrowErrorCode GeoArrowArrayViewSetArray(struct GeoArrowArrayView* array_view,
                                            const struct ArrowArray* array,
                                            struct GeoArrowError* error) {
  switch (array_view->schema_view.type) {
    case GEOARROW_TYPE_WKT:
    case GEOARROW_TYPE_WKB:
      NANOARROW_RETURN_NOT_OK(GeoArrowArrayViewSetArraySerialized(array_view, array));
      break;
    case GEOARROW_TYPE_BOX:
    case GEOARROW_TYPE_BOX_Z:
    case GEOARROW_TYPE_BOX_M:
    case GEOARROW_TYPE_BOX_ZM:
      NANOARROW_RETURN_NOT_OK(GeoArrowArrayViewSetArrayBox(array_view, array, error));
      break;
    default:
      NANOARROW_RETURN_NOT_OK(
          GeoArrowArrayViewSetArrayInternal(array_view, array, error, 0));
      break;
  }

  array_view->validity_bitmap = array->buffers[0];
  return GEOARROW_OK;
}

static inline void GeoArrowCoordViewUpdate(const struct GeoArrowCoordView* src,
                                           struct GeoArrowCoordView* dst, int64_t offset,
                                           int64_t length) {
  for (int j = 0; j < dst->n_values; j++) {
    dst->values[j] = src->values[j] + (offset * src->coords_stride);
  }
  dst->n_coords = length;
}

static GeoArrowErrorCode GeoArrowArrayViewVisitNativePoint(
    const struct GeoArrowArrayView* array_view, int64_t offset, int64_t length,
    struct GeoArrowVisitor* v) {
  struct GeoArrowCoordView coords = array_view->coords;

  for (int64_t i = 0; i < length; i++) {
    NANOARROW_RETURN_NOT_OK(v->feat_start(v));
    if (!array_view->validity_bitmap ||
        ArrowBitGet(array_view->validity_bitmap, array_view->offset[0] + offset + i)) {
      NANOARROW_RETURN_NOT_OK(v->geom_start(v, GEOARROW_GEOMETRY_TYPE_POINT,
                                            array_view->schema_view.dimensions));
      GeoArrowCoordViewUpdate(&array_view->coords, &coords,
                              array_view->offset[0] + offset + i, 1);
      NANOARROW_RETURN_NOT_OK(v->coords(v, &coords));
      NANOARROW_RETURN_NOT_OK(v->geom_end(v));
    } else {
      NANOARROW_RETURN_NOT_OK(v->null_feat(v));
    }

    NANOARROW_RETURN_NOT_OK(v->feat_end(v));

    for (int j = 0; j < coords.n_values; j++) {
      coords.values[j] += coords.coords_stride;
    }
  }

  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowArrayViewVisitNativeLinestring(
    const struct GeoArrowArrayView* array_view, int64_t offset, int64_t length,
    struct GeoArrowVisitor* v) {
  struct GeoArrowCoordView coords = array_view->coords;

  int64_t coord_offset;
  int64_t n_coords;
  for (int64_t i = 0; i < length; i++) {
    NANOARROW_RETURN_NOT_OK(v->feat_start(v));
    if (!array_view->validity_bitmap ||
        ArrowBitGet(array_view->validity_bitmap, array_view->offset[0] + offset + i)) {
      NANOARROW_RETURN_NOT_OK(v->geom_start(v, GEOARROW_GEOMETRY_TYPE_LINESTRING,
                                            array_view->schema_view.dimensions));
      coord_offset = array_view->offsets[0][array_view->offset[0] + offset + i];
      n_coords =
          array_view->offsets[0][array_view->offset[0] + offset + i + 1] - coord_offset;
      coord_offset += array_view->offset[1];
      GeoArrowCoordViewUpdate(&array_view->coords, &coords, coord_offset, n_coords);
      NANOARROW_RETURN_NOT_OK(v->coords(v, &coords));
      NANOARROW_RETURN_NOT_OK(v->geom_end(v));
    } else {
      NANOARROW_RETURN_NOT_OK(v->null_feat(v));
    }

    NANOARROW_RETURN_NOT_OK(v->feat_end(v));
  }

  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowArrayViewVisitNativePolygon(
    const struct GeoArrowArrayView* array_view, int64_t offset, int64_t length,
    struct GeoArrowVisitor* v) {
  struct GeoArrowCoordView coords = array_view->coords;

  int64_t ring_offset;
  int64_t n_rings;
  int64_t coord_offset;
  int64_t n_coords;
  for (int64_t i = 0; i < length; i++) {
    NANOARROW_RETURN_NOT_OK(v->feat_start(v));
    if (!array_view->validity_bitmap ||
        ArrowBitGet(array_view->validity_bitmap, array_view->offset[0] + offset + i)) {
      NANOARROW_RETURN_NOT_OK(v->geom_start(v, GEOARROW_GEOMETRY_TYPE_POLYGON,
                                            array_view->schema_view.dimensions));
      ring_offset = array_view->offsets[0][array_view->offset[0] + offset + i];
      n_rings =
          array_view->offsets[0][array_view->offset[0] + offset + i + 1] - ring_offset;
      ring_offset += array_view->offset[1];

      for (int64_t j = 0; j < n_rings; j++) {
        NANOARROW_RETURN_NOT_OK(v->ring_start(v));
        coord_offset = array_view->offsets[1][ring_offset + j];
        n_coords = array_view->offsets[1][ring_offset + j + 1] - coord_offset;
        coord_offset += array_view->offset[2];
        GeoArrowCoordViewUpdate(&array_view->coords, &coords, coord_offset, n_coords);
        NANOARROW_RETURN_NOT_OK(v->coords(v, &coords));
        NANOARROW_RETURN_NOT_OK(v->ring_end(v));
      }

      NANOARROW_RETURN_NOT_OK(v->geom_end(v));
    } else {
      NANOARROW_RETURN_NOT_OK(v->null_feat(v));
    }

    NANOARROW_RETURN_NOT_OK(v->feat_end(v));
  }

  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowArrayViewVisitNativeMultipoint(
    const struct GeoArrowArrayView* array_view, int64_t offset, int64_t length,
    struct GeoArrowVisitor* v) {
  struct GeoArrowCoordView coords = array_view->coords;

  int64_t coord_offset;
  int64_t n_coords;
  for (int64_t i = 0; i < length; i++) {
    NANOARROW_RETURN_NOT_OK(v->feat_start(v));
    if (!array_view->validity_bitmap ||
        ArrowBitGet(array_view->validity_bitmap, array_view->offset[0] + offset + i)) {
      NANOARROW_RETURN_NOT_OK(v->geom_start(v, GEOARROW_GEOMETRY_TYPE_MULTIPOINT,
                                            array_view->schema_view.dimensions));
      coord_offset = array_view->offsets[0][array_view->offset[0] + offset + i];
      n_coords =
          array_view->offsets[0][array_view->offset[0] + offset + i + 1] - coord_offset;
      coord_offset += array_view->offset[1];
      for (int64_t j = 0; j < n_coords; j++) {
        NANOARROW_RETURN_NOT_OK(v->geom_start(v, GEOARROW_GEOMETRY_TYPE_POINT,
                                              array_view->schema_view.dimensions));
        GeoArrowCoordViewUpdate(&array_view->coords, &coords, coord_offset + j, 1);
        NANOARROW_RETURN_NOT_OK(v->coords(v, &coords));
        NANOARROW_RETURN_NOT_OK(v->geom_end(v));
      }
      NANOARROW_RETURN_NOT_OK(v->geom_end(v));
    } else {
      NANOARROW_RETURN_NOT_OK(v->null_feat(v));
    }

    NANOARROW_RETURN_NOT_OK(v->feat_end(v));
  }

  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowArrayViewVisitNativeMultilinestring(
    const struct GeoArrowArrayView* array_view, int64_t offset, int64_t length,
    struct GeoArrowVisitor* v) {
  struct GeoArrowCoordView coords = array_view->coords;

  int64_t linestring_offset;
  int64_t n_linestrings;
  int64_t coord_offset;
  int64_t n_coords;
  for (int64_t i = 0; i < length; i++) {
    NANOARROW_RETURN_NOT_OK(v->feat_start(v));
    if (!array_view->validity_bitmap ||
        ArrowBitGet(array_view->validity_bitmap, array_view->offset[0] + offset + i)) {
      NANOARROW_RETURN_NOT_OK(v->geom_start(v, GEOARROW_GEOMETRY_TYPE_MULTILINESTRING,
                                            array_view->schema_view.dimensions));
      linestring_offset = array_view->offsets[0][array_view->offset[0] + offset + i];
      n_linestrings = array_view->offsets[0][array_view->offset[0] + offset + i + 1] -
                      linestring_offset;
      linestring_offset += array_view->offset[1];

      for (int64_t j = 0; j < n_linestrings; j++) {
        NANOARROW_RETURN_NOT_OK(v->geom_start(v, GEOARROW_GEOMETRY_TYPE_LINESTRING,
                                              array_view->schema_view.dimensions));
        coord_offset = array_view->offsets[1][linestring_offset + j];
        n_coords = array_view->offsets[1][linestring_offset + j + 1] - coord_offset;
        coord_offset += array_view->offset[2];
        GeoArrowCoordViewUpdate(&array_view->coords, &coords, coord_offset, n_coords);
        NANOARROW_RETURN_NOT_OK(v->coords(v, &coords));
        NANOARROW_RETURN_NOT_OK(v->geom_end(v));
      }

      NANOARROW_RETURN_NOT_OK(v->geom_end(v));
    } else {
      NANOARROW_RETURN_NOT_OK(v->null_feat(v));
    }

    NANOARROW_RETURN_NOT_OK(v->feat_end(v));
  }

  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowArrayViewVisitNativeMultipolygon(
    const struct GeoArrowArrayView* array_view, int64_t offset, int64_t length,
    struct GeoArrowVisitor* v) {
  struct GeoArrowCoordView coords = array_view->coords;

  int64_t polygon_offset;
  int64_t n_polygons;
  int64_t ring_offset;
  int64_t n_rings;
  int64_t coord_offset;
  int64_t n_coords;
  for (int64_t i = 0; i < length; i++) {
    NANOARROW_RETURN_NOT_OK(v->feat_start(v));
    if (!array_view->validity_bitmap ||
        ArrowBitGet(array_view->validity_bitmap, array_view->offset[0] + offset + i)) {
      NANOARROW_RETURN_NOT_OK(v->geom_start(v, GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON,
                                            array_view->schema_view.dimensions));

      polygon_offset = array_view->offsets[0][array_view->offset[0] + offset + i];
      n_polygons =
          array_view->offsets[0][array_view->offset[0] + offset + i + 1] - polygon_offset;
      polygon_offset += array_view->offset[1];

      for (int64_t j = 0; j < n_polygons; j++) {
        NANOARROW_RETURN_NOT_OK(v->geom_start(v, GEOARROW_GEOMETRY_TYPE_POLYGON,
                                              array_view->schema_view.dimensions));

        ring_offset = array_view->offsets[1][polygon_offset + j];
        n_rings = array_view->offsets[1][polygon_offset + j + 1] - ring_offset;
        ring_offset += array_view->offset[2];

        for (int64_t k = 0; k < n_rings; k++) {
          NANOARROW_RETURN_NOT_OK(v->ring_start(v));
          coord_offset = array_view->offsets[2][ring_offset + k];
          n_coords = array_view->offsets[2][ring_offset + k + 1] - coord_offset;
          coord_offset += array_view->offset[3];
          GeoArrowCoordViewUpdate(&array_view->coords, &coords, coord_offset, n_coords);
          NANOARROW_RETURN_NOT_OK(v->coords(v, &coords));
          NANOARROW_RETURN_NOT_OK(v->ring_end(v));
        }

        NANOARROW_RETURN_NOT_OK(v->geom_end(v));
      }

      NANOARROW_RETURN_NOT_OK(v->geom_end(v));
    } else {
      NANOARROW_RETURN_NOT_OK(v->null_feat(v));
    }

    NANOARROW_RETURN_NOT_OK(v->feat_end(v));
  }

  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowArrayViewVisitNativeBox(
    const struct GeoArrowArrayView* array_view, int64_t offset, int64_t length,
    struct GeoArrowVisitor* v) {
  // We aren't going to attempt Z, M, or ZM boxes since there is no canonical
  // way to do this (maybe only if the non-XY dimensions are constant?).
  if (array_view->schema_view.dimensions != GEOARROW_DIMENSIONS_XY) {
    GeoArrowErrorSet(v->error, "Can't visit box with non-XY dimensions");
    return ENOTSUP;
  }

  // These are the polygon coords and the arrays to back them
  struct GeoArrowCoordView poly_coords;
  memset(&poly_coords, 0, sizeof(struct GeoArrowCoordView));

  int n_dim = array_view->coords.n_values / 2;
  double x[5];
  double y[5];
  poly_coords.n_values = n_dim;
  poly_coords.n_coords = 5;
  poly_coords.coords_stride = 1;
  poly_coords.values[0] = x;
  poly_coords.values[1] = y;

  // index into each box coord's values[] for each polygon coordinate
  int box_coord_poly_map_x[] = {0, n_dim, n_dim, 0, 0};
  int box_coord_poly_map_y[] = {1, 1, n_dim + 1, n_dim + 1, 1};

  for (int64_t i = 0; i < length; i++) {
    int64_t raw_offset = array_view->offset[0] + offset + i;
    NANOARROW_RETURN_NOT_OK(v->feat_start(v));
    if (!array_view->validity_bitmap ||
        ArrowBitGet(array_view->validity_bitmap, raw_offset)) {
      // Check for empty dimensions
      int n_empty_dims = 0;
      for (int i = 0; i < n_dim; i++) {
        double dim_min = GEOARROW_COORD_VIEW_VALUE(&array_view->coords, raw_offset, i);
        double dim_max =
            GEOARROW_COORD_VIEW_VALUE(&array_view->coords, raw_offset, n_dim + i);
        n_empty_dims += dim_min > dim_max;
      }

      NANOARROW_RETURN_NOT_OK(v->geom_start(v, GEOARROW_GEOMETRY_TYPE_POLYGON,
                                            array_view->schema_view.dimensions));

      // If any dimension has a negative range, we consider the polygon empty
      // (i.e., there are no points for which...)
      if (n_empty_dims == 0) {
        // Populate the polygon coordinates
        for (int i = 0; i < 5; i++) {
          x[i] = GEOARROW_COORD_VIEW_VALUE(&array_view->coords, raw_offset,
                                           box_coord_poly_map_x[i]);
          y[i] = GEOARROW_COORD_VIEW_VALUE(&array_view->coords, raw_offset,
                                           box_coord_poly_map_y[i]);
        }

        // Call the visitor
        NANOARROW_RETURN_NOT_OK(v->ring_start(v));
        NANOARROW_RETURN_NOT_OK(v->coords(v, &poly_coords));
        NANOARROW_RETURN_NOT_OK(v->ring_end(v));
      }

      NANOARROW_RETURN_NOT_OK(v->geom_end(v));
    } else {
      NANOARROW_RETURN_NOT_OK(v->null_feat(v));
    }

    NANOARROW_RETURN_NOT_OK(v->feat_end(v));
  }

  return GEOARROW_OK;
}

GeoArrowErrorCode GeoArrowArrayViewVisitNative(const struct GeoArrowArrayView* array_view,
                                               int64_t offset, int64_t length,
                                               struct GeoArrowVisitor* v) {
  switch (array_view->schema_view.geometry_type) {
    case GEOARROW_GEOMETRY_TYPE_BOX:
      return GeoArrowArrayViewVisitNativeBox(array_view, offset, length, v);
    case GEOARROW_GEOMETRY_TYPE_POINT:
      return GeoArrowArrayViewVisitNativePoint(array_view, offset, length, v);
    case GEOARROW_GEOMETRY_TYPE_LINESTRING:
      return GeoArrowArrayViewVisitNativeLinestring(array_view, offset, length, v);
    case GEOARROW_GEOMETRY_TYPE_POLYGON:
      return GeoArrowArrayViewVisitNativePolygon(array_view, offset, length, v);
    case GEOARROW_GEOMETRY_TYPE_MULTIPOINT:
      return GeoArrowArrayViewVisitNativeMultipoint(array_view, offset, length, v);
    case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING:
      return GeoArrowArrayViewVisitNativeMultilinestring(array_view, offset, length, v);
    case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON:
      return GeoArrowArrayViewVisitNativeMultipolygon(array_view, offset, length, v);
    default:
      return ENOTSUP;
  }
}

#include <errno.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>



const char* GeoArrowVersion(void) { return GEOARROW_VERSION; }

int GeoArrowVersionInt(void) { return GEOARROW_VERSION_INT; }

GeoArrowErrorCode GeoArrowErrorSet(struct GeoArrowError* error, const char* fmt, ...) {
  if (error == NULL) {
    return GEOARROW_OK;
  }

  memset(error->message, 0, sizeof(error->message));

  va_list args;
  va_start(args, fmt);
  int chars_needed = vsnprintf(error->message, sizeof(error->message), fmt, args);
  va_end(args);

  if (chars_needed < 0) {
    return EINVAL;
  } else if (((size_t)chars_needed) >= sizeof(error->message)) {
    return ERANGE;
  } else {
    return GEOARROW_OK;
  }
}

#include <stddef.h>


#include "nanoarrow/nanoarrow.h"

static int feat_start_void(struct GeoArrowVisitor* v) {
  NANOARROW_UNUSED(v);
  return GEOARROW_OK;
}

static int null_feat_void(struct GeoArrowVisitor* v) {
  NANOARROW_UNUSED(v);
  return GEOARROW_OK;
}

static int geom_start_void(struct GeoArrowVisitor* v,
                           enum GeoArrowGeometryType geometry_type,
                           enum GeoArrowDimensions dimensions) {
  NANOARROW_UNUSED(v);
  NANOARROW_UNUSED(geometry_type);
  NANOARROW_UNUSED(dimensions);
  return GEOARROW_OK;
}

static int ring_start_void(struct GeoArrowVisitor* v) {
  NANOARROW_UNUSED(v);
  return GEOARROW_OK;
}

static int coords_void(struct GeoArrowVisitor* v,
                       const struct GeoArrowCoordView* coords) {
  NANOARROW_UNUSED(v);
  NANOARROW_UNUSED(coords);
  return GEOARROW_OK;
}

static int ring_end_void(struct GeoArrowVisitor* v) {
  NANOARROW_UNUSED(v);
  return GEOARROW_OK;
}

static int geom_end_void(struct GeoArrowVisitor* v) {
  NANOARROW_UNUSED(v);
  return GEOARROW_OK;
}

static int feat_end_void(struct GeoArrowVisitor* v) {
  NANOARROW_UNUSED(v);
  return GEOARROW_OK;
}

void GeoArrowVisitorInitVoid(struct GeoArrowVisitor* v) {
  v->feat_start = &feat_start_void;
  v->null_feat = &null_feat_void;
  v->geom_start = &geom_start_void;
  v->ring_start = &ring_start_void;
  v->coords = &coords_void;
  v->ring_end = &ring_end_void;
  v->geom_end = &geom_end_void;
  v->feat_end = &feat_end_void;
  v->error = NULL;
  v->private_data = NULL;
}

#include <errno.h>


#include "nanoarrow/nanoarrow.h"

struct GeoArrowGeometryPrivate {
  // Note that the GeoArrowGeometry has cached data/size/capacity that needs
  // to be kept in sync
  struct ArrowBuffer nodes;
  struct ArrowBuffer coords;
  // For the builder interface
  int current_level;
};

GeoArrowErrorCode GeoArrowGeometryInit(struct GeoArrowGeometry* geom) {
  struct GeoArrowGeometryPrivate* private_data =
      (struct GeoArrowGeometryPrivate*)ArrowMalloc(
          sizeof(struct GeoArrowGeometryPrivate));
  if (private_data == NULL) {
    return ENOMEM;
  }

  geom->root = NULL;
  geom->size_nodes = 0;
  geom->capacity_nodes = 0;
  ArrowBufferInit(&private_data->nodes);
  ArrowBufferInit(&private_data->coords);
  private_data->current_level = 0;
  geom->private_data = private_data;

  return GEOARROW_OK;
}

GeoArrowErrorCode GeoArrowGeometryShallowCopy(struct GeoArrowGeometryView src,
                                              struct GeoArrowGeometry* dst) {
  GEOARROW_RETURN_NOT_OK(GeoArrowGeometryResizeNodes(dst, src.size_nodes));
  if (src.size_nodes > 0) {
    memcpy(dst->root, src.root, src.size_nodes * sizeof(struct GeoArrowGeometryNode));
  }

  return GEOARROW_OK;
}

void GeoArrowGeometryReset(struct GeoArrowGeometry* geom) {
  struct GeoArrowGeometryPrivate* private_data =
      (struct GeoArrowGeometryPrivate*)geom->private_data;
  ArrowBufferReset(&private_data->coords);
  ArrowBufferReset(&private_data->nodes);
  ArrowFree(geom->private_data);
  geom->private_data = NULL;
}

static inline void GeoArrowGeometrySyncGeomToNodes(struct GeoArrowGeometry* geom) {
  struct GeoArrowGeometryPrivate* private_data =
      (struct GeoArrowGeometryPrivate*)geom->private_data;
  // Inline methods should never modify the capacity or data pointer
  NANOARROW_DCHECK(private_data->nodes.capacity_bytes ==
                   (geom->capacity_nodes * (int64_t)sizeof(struct GeoArrowGeometryNode)));
  NANOARROW_DCHECK(private_data->nodes.data == (uint8_t*)geom->root);

  // But may have updated the size
  private_data->nodes.size_bytes = geom->size_nodes * sizeof(struct GeoArrowGeometryNode);
}

static inline void GeoArrowGeometrySyncNodesToGeom(struct GeoArrowGeometry* geom) {
  struct GeoArrowGeometryPrivate* private_data =
      (struct GeoArrowGeometryPrivate*)geom->private_data;
  geom->root = (struct GeoArrowGeometryNode*)private_data->nodes.data;
  geom->size_nodes = private_data->nodes.size_bytes / sizeof(struct GeoArrowGeometryNode);
  geom->capacity_nodes =
      private_data->nodes.capacity_bytes / sizeof(struct GeoArrowGeometryNode);
}

GeoArrowErrorCode GeoArrowGeometryResizeNodes(struct GeoArrowGeometry* geom,
                                              int64_t size_nodes) {
  struct GeoArrowGeometryPrivate* private_data =
      (struct GeoArrowGeometryPrivate*)geom->private_data;
  GEOARROW_RETURN_NOT_OK(ArrowBufferResize(
      &private_data->nodes, size_nodes * sizeof(struct GeoArrowGeometryNode), 0));
  GeoArrowGeometrySyncNodesToGeom(geom);
  return GEOARROW_OK;
}

GeoArrowErrorCode GeoArrowGeometryAppendNode(struct GeoArrowGeometry* geom,
                                             struct GeoArrowGeometryNode** out) {
  struct GeoArrowGeometryPrivate* private_data =
      (struct GeoArrowGeometryPrivate*)geom->private_data;
  GeoArrowGeometrySyncGeomToNodes(geom);
  GEOARROW_RETURN_NOT_OK(
      ArrowBufferReserve(&private_data->nodes, sizeof(struct GeoArrowGeometryNode)));
  GeoArrowGeometrySyncNodesToGeom(geom);
  return GeoArrowGeometryAppendNodeInline(geom, out);
}

GeoArrowErrorCode GeoArrowGeometryDeepCopy(struct GeoArrowGeometryView src,
                                           struct GeoArrowGeometry* dst) {
  struct GeoArrowGeometryPrivate* private_data =
      (struct GeoArrowGeometryPrivate*)dst->private_data;

  // Calculate the number of coordinate storage bytes required
  int64_t total_bytes_required = 0;
  const struct GeoArrowGeometryNode* src_end = src.root + src.size_nodes;
  for (const struct GeoArrowGeometryNode* node = src.root; node < src_end; ++node) {
    switch (node->geometry_type) {
      case GEOARROW_GEOMETRY_TYPE_POINT:
      case GEOARROW_GEOMETRY_TYPE_LINESTRING:
        total_bytes_required += GeoArrowGeometryNodeWriteSequence(node, NULL, 0);
        break;
      default:
        break;
    }
  }

  // Resize the destination coords array
  GEOARROW_RETURN_NOT_OK(
      ArrowBufferResize(&private_data->coords, total_bytes_required, 0));

  // Copy the nodes
  GEOARROW_RETURN_NOT_OK(GeoArrowGeometryShallowCopy(src, dst));

  // Copy the data and update the nodes to point to the internal data
  uint8_t* coords_buf = private_data->coords.data;
  int64_t coords_buf_size = private_data->coords.size_bytes;
  int64_t bytes_written;
  int n_values;

  struct GeoArrowGeometryNode* dst_end = dst->root + dst->size_nodes;
  for (struct GeoArrowGeometryNode* node = dst->root; node < dst_end; ++node) {
    switch (node->geometry_type) {
      case GEOARROW_GEOMETRY_TYPE_POINT:
      case GEOARROW_GEOMETRY_TYPE_LINESTRING:
        n_values = _GeoArrowkNumDimensions[node->dimensions];
        bytes_written =
            GeoArrowGeometryNodeWriteSequence(node, coords_buf, coords_buf_size);
        // NANOARROW_DCHECK(bytes_written <= coords_buf_size);

        for (int i = 0; i < n_values; i++) {
          node->coords[i] = coords_buf + (i * sizeof(double));
          node->coord_stride[i] = n_values * sizeof(double);
        }

        coords_buf += bytes_written;
        coords_buf_size -= bytes_written;
        break;
      default:
        break;
    }
  }

  return GEOARROW_OK;
}

// We define a maximum nesting to simplify collecting sizes based on levels
#define GEOARROW_GEOMETRY_VISITOR_MAX_NESTING 31

static inline GeoArrowErrorCode GeoArrowGeometryReallocCoords(
    struct GeoArrowGeometry* geom, struct ArrowBuffer* new_coords,
    int64_t additional_size_bytes) {
  struct GeoArrowGeometryPrivate* private_data =
      (struct GeoArrowGeometryPrivate*)geom->private_data;

  int64_t new_size_bytes;
  if ((private_data->coords.size_bytes + additional_size_bytes) >
      private_data->coords.size_bytes * 2) {
    new_size_bytes = private_data->coords.size_bytes + additional_size_bytes;
  } else {
    new_size_bytes = private_data->coords.size_bytes * 2;
  }

  GEOARROW_RETURN_NOT_OK(ArrowBufferReserve(new_coords, new_size_bytes));
  GEOARROW_RETURN_NOT_OK(ArrowBufferAppend(new_coords, private_data->coords.data,
                                           private_data->coords.size_bytes));
  struct GeoArrowGeometryNode* node = geom->root;
  for (int64_t i = 0; i < geom->size_nodes; i++) {
    for (int i = 0; i < 4; i++) {
      if (node->coords[i] == _GeoArrowkEmptyPointCoords) {
        continue;
      }

      ptrdiff_t offset = node->coords[i] - private_data->coords.data;
      node->coords[i] = new_coords->data + offset;
    }

    ++node;
  }

  return GEOARROW_OK;
}

static inline GeoArrowErrorCode GeoArrowGeometryReserveCoords(
    struct GeoArrowGeometry* geom, int64_t additional_doubles, double** out) {
  struct GeoArrowGeometryPrivate* private_data =
      (struct GeoArrowGeometryPrivate*)geom->private_data;

  int64_t bytes_required =
      private_data->coords.size_bytes + (additional_doubles * sizeof(double));
  if (bytes_required > private_data->coords.capacity_bytes) {
    struct ArrowBuffer new_coords;
    ArrowBufferInit(&new_coords);
    GeoArrowErrorCode result = GeoArrowGeometryReallocCoords(
        geom, &new_coords, (additional_doubles * sizeof(double)));
    if (result != GEOARROW_OK) {
      ArrowBufferReset(&new_coords);
      return result;
    }

    ArrowBufferReset(&private_data->coords);
    ArrowBufferMove(&new_coords, &private_data->coords);
  }

  *out = (double*)(private_data->coords.data + private_data->coords.size_bytes);
  return GEOARROW_OK;
}

static int feat_start_geometry(struct GeoArrowVisitor* v) {
  struct GeoArrowGeometry* geom = (struct GeoArrowGeometry*)v->private_data;
  struct GeoArrowGeometryPrivate* private_data =
      (struct GeoArrowGeometryPrivate*)geom->private_data;

  GEOARROW_RETURN_NOT_OK(GeoArrowGeometryResizeNodes(geom, 0));
  GEOARROW_RETURN_NOT_OK(ArrowBufferResize(&private_data->coords, 0, 0));
  private_data->current_level = 0;
  return GEOARROW_OK;
}

static int null_feat_geometry(struct GeoArrowVisitor* v) {
  struct GeoArrowGeometry* geom = (struct GeoArrowGeometry*)v->private_data;
  struct GeoArrowGeometryNode* node;
  GEOARROW_RETURN_NOT_OK(GeoArrowGeometryAppendNodeInline(geom, &node));
  return GEOARROW_OK;
}

static int geom_start_geometry(struct GeoArrowVisitor* v,
                               enum GeoArrowGeometryType geometry_type,
                               enum GeoArrowDimensions dimensions) {
  struct GeoArrowGeometry* geom = (struct GeoArrowGeometry*)v->private_data;
  struct GeoArrowGeometryPrivate* private_data =
      (struct GeoArrowGeometryPrivate*)geom->private_data;
  struct GeoArrowGeometryNode* node;
  GEOARROW_RETURN_NOT_OK(GeoArrowGeometryAppendNodeInline(geom, &node));
  node->geometry_type = (uint8_t)geometry_type;
  node->dimensions = (uint8_t)dimensions;
  node->level = (uint8_t)private_data->current_level;
  switch (geometry_type) {
    case GEOARROW_GEOMETRY_TYPE_POINT:
    case GEOARROW_GEOMETRY_TYPE_LINESTRING: {
      // Reserve coords and set pointers here, even when we're not sure if
      // there will be any coordinates. This lets us calculate sequences sizes
      // after the fact because we can always use the pointer differences between
      // the start of the x value in one sequence and the start of the x value
      // in the next sequence to calculate sequence size.
      int n_values = _GeoArrowkNumDimensions[dimensions];
      double* coords_start;
      GEOARROW_RETURN_NOT_OK(
          GeoArrowGeometryReserveCoords(geom, n_values, &coords_start));
      for (int i = 0; i < n_values; i++) {
        node->coords[i] = (uint8_t*)(coords_start + i);
        node->coord_stride[i] = n_values * sizeof(double);
      }
      break;
    }
    default:
      break;
  }

  if (private_data->current_level == GEOARROW_GEOMETRY_VISITOR_MAX_NESTING) {
    GeoArrowErrorSet(v->error, "Maximum recursion for GeoArrowGeometry visitor reached");
    return EINVAL;
  }

  private_data->current_level++;
  return GEOARROW_OK;
}

static int ring_start_geometry(struct GeoArrowVisitor* v) {
  struct GeoArrowGeometry* geom = (struct GeoArrowGeometry*)v->private_data;
  if (geom->size_nodes == 0) {
    GeoArrowErrorSet(v->error,
                     "Call to ring_start before geom_start in GeoArrowGeometry visitor");
    return EINVAL;
  }

  struct GeoArrowGeometryNode* last_node = geom->root + geom->size_nodes - 1;
  return geom_start_geometry(v, GEOARROW_GEOMETRY_TYPE_LINESTRING, last_node->dimensions);
}

static int coords_geometry(struct GeoArrowVisitor* v,
                           const struct GeoArrowCoordView* coords) {
  struct GeoArrowGeometry* geom = (struct GeoArrowGeometry*)v->private_data;
  struct GeoArrowGeometryPrivate* private_data =
      (struct GeoArrowGeometryPrivate*)geom->private_data;

  if (geom->size_nodes == 0) {
    GeoArrowErrorSet(v->error,
                     "Call to coords before geom_start in GeoArrowGeometry visitor");
    return EINVAL;
  }

  double* values;
  GEOARROW_RETURN_NOT_OK(
      GeoArrowGeometryReserveCoords(geom, coords->n_coords * coords->n_values, &values));

  for (int64_t i = 0; i < coords->n_coords; i++) {
    for (int j = 0; j < coords->n_values; j++) {
      *values++ = GEOARROW_COORD_VIEW_VALUE(coords, i, j);
    }
  }

  private_data->coords.size_bytes += coords->n_coords * coords->n_values * sizeof(double);
  return GEOARROW_OK;
}

static int ring_end_geometry(struct GeoArrowVisitor* v) {
  struct GeoArrowGeometry* geom = (struct GeoArrowGeometry*)v->private_data;
  struct GeoArrowGeometryPrivate* private_data =
      (struct GeoArrowGeometryPrivate*)geom->private_data;

  if (private_data->current_level == 0) {
    GeoArrowErrorSet(v->error,
                     "Incorrect nesting in GeoArrowGeometry visitor (level < 0)");
    return EINVAL;
  }

  private_data->current_level--;
  return GEOARROW_OK;
}

static int geom_end_geometry(struct GeoArrowVisitor* v) {
  struct GeoArrowGeometry* geom = (struct GeoArrowGeometry*)v->private_data;
  struct GeoArrowGeometryPrivate* private_data =
      (struct GeoArrowGeometryPrivate*)geom->private_data;

  if (private_data->current_level == 0) {
    GeoArrowErrorSet(v->error,
                     "Incorrect nesting in GeoArrowGeometry visitor (level < 0)");
    return EINVAL;
  }

  private_data->current_level--;
  return GEOARROW_OK;
}

static int feat_end_geometry(struct GeoArrowVisitor* v) {
  struct GeoArrowGeometry* geom = (struct GeoArrowGeometry*)v->private_data;
  struct GeoArrowGeometryPrivate* private_data =
      (struct GeoArrowGeometryPrivate*)geom->private_data;

  if (geom->size_nodes == 0) {
    GeoArrowErrorSet(v->error,
                     "Call to feat_end before geom_start in GeoArrowGeometry visitor");
    return EINVAL;
  }

  if (private_data->coords.size_bytes == 0) {
    // Can happen for empty collections where there was never a sequence
    return GEOARROW_OK;
  }

  // Set sequence lengths by rolling backwards through the sequences. We recorded
  // the start position for each sequence in geom_start, so we can calculate the
  // size based on the distance between the pointers.
  const uint8_t* end = private_data->coords.data + private_data->coords.size_bytes;
  struct GeoArrowGeometryNode* node_begin = geom->root;
  struct GeoArrowGeometryNode* node_end = geom->root + geom->size_nodes;

  uint32_t sizes[GEOARROW_GEOMETRY_VISITOR_MAX_NESTING + 1];
  memset(sizes, 0, sizeof(sizes));
  ptrdiff_t sequence_bytes;
  for (struct GeoArrowGeometryNode* node = (node_end - 1); node >= node_begin; node--) {
    sizes[node->level]++;

    switch (node->geometry_type) {
      case GEOARROW_GEOMETRY_TYPE_POINT:
      case GEOARROW_GEOMETRY_TYPE_LINESTRING:
        sequence_bytes = end - node->coords[0];
        node->size = (uint32_t)(sequence_bytes / node->coord_stride[0]);
        end = node->coords[0];
        break;
      case GEOARROW_GEOMETRY_TYPE_POLYGON:
      case GEOARROW_GEOMETRY_TYPE_MULTIPOINT:
      case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING:
      case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON:
      case GEOARROW_GEOMETRY_TYPE_GEOMETRYCOLLECTION:
        node->size = sizes[node->level + 1];
        sizes[node->level + 1] = 0;
        break;
      default:
        break;
    }
  }

  return GEOARROW_OK;
}

void GeoArrowGeometryInitVisitor(struct GeoArrowGeometry* geom,
                                 struct GeoArrowVisitor* v) {
  v->feat_start = &feat_start_geometry;
  v->null_feat = &null_feat_geometry;
  v->geom_start = &geom_start_geometry;
  v->ring_start = &ring_start_geometry;
  v->coords = &coords_geometry;
  v->ring_end = &ring_end_geometry;
  v->geom_end = &geom_end_geometry;
  v->feat_end = &feat_end_geometry;
  v->private_data = geom;
}

// This must be divisible by 2, 3, and 4
#define COORD_CACHE_SIZE_ELEMENTS 384

static GeoArrowErrorCode GeoArrowGeometryVisitSequence(
    const struct GeoArrowGeometryNode* node, struct GeoArrowVisitor* v) {
  double coords[COORD_CACHE_SIZE_ELEMENTS];
  struct GeoArrowCoordView coord_view;
  coord_view.n_values = _GeoArrowkNumDimensions[node->dimensions];
  for (int i = 0; i < coord_view.n_values; i++) {
    coord_view.values[i] = coords + i;
    coord_view.coords_stride = coord_view.n_values;
  }

  int32_t chunk_size = COORD_CACHE_SIZE_ELEMENTS / coord_view.n_values;
  coord_view.n_coords = chunk_size;

  // Process full chunks
  int64_t n_coords = node->size;

  struct GeoArrowGeometryNode node_cursor = *node;
  node_cursor.size = chunk_size;

  while (n_coords > chunk_size) {
    GeoArrowGeometryNodeWriteSequence(&node_cursor, (uint8_t*)&coords, sizeof(coords));
    coord_view.n_coords = chunk_size;
    GEOARROW_RETURN_NOT_OK(v->coords(v, &coord_view));
    n_coords -= chunk_size;
    for (int i = 0; i < coord_view.n_values; i++) {
      node_cursor.coords[i] += chunk_size * node_cursor.coord_stride[i];
    }
  }

  // Process the last chunk
  node_cursor.size = (uint32_t)n_coords;
  coord_view.n_coords = n_coords;
  GeoArrowGeometryNodeWriteSequence(&node_cursor, (uint8_t*)&coords, sizeof(coords));
  GEOARROW_RETURN_NOT_OK(v->coords(v, &coord_view));

  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowGeometryVisitNode(
    const struct GeoArrowGeometryNode* node, int64_t* n_nodes,
    struct GeoArrowVisitor* v) {
  if ((*n_nodes)-- <= 0) {
    GeoArrowErrorSet(v->error, "Too few nodes provided to GeoArrowGeometryVisit()");
  }

  GEOARROW_RETURN_NOT_OK(v->geom_start(v, (enum GeoArrowGeometryType)node->geometry_type,
                                       (enum GeoArrowDimensions)node->dimensions));
  switch (node->geometry_type) {
    case GEOARROW_GEOMETRY_TYPE_POINT:
    case GEOARROW_GEOMETRY_TYPE_LINESTRING:
      GEOARROW_RETURN_NOT_OK(GeoArrowGeometryVisitSequence(node, v));
      break;
    case GEOARROW_GEOMETRY_TYPE_POLYGON:
      if (*n_nodes < ((int64_t)node->size)) {
        return EINVAL;
      }

      for (uint32_t i = 0; i < node->size; i++) {
        GEOARROW_RETURN_NOT_OK(v->ring_start(v));
        GEOARROW_RETURN_NOT_OK(GeoArrowGeometryVisitSequence(node + i + 1, v));
        GEOARROW_RETURN_NOT_OK(v->ring_end(v));
      }

      (*n_nodes) -= node->size;
      break;
    case GEOARROW_GEOMETRY_TYPE_MULTIPOINT:
    case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING:
    case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON:
    case GEOARROW_GEOMETRY_TYPE_GEOMETRYCOLLECTION: {
      const struct GeoArrowGeometryNode* child = node + 1;
      for (uint32_t i = 0; i < node->size; i++) {
        int64_t n_nodes_before_child = *n_nodes;
        GEOARROW_RETURN_NOT_OK(GeoArrowGeometryVisitNode(child, n_nodes, v));
        int64_t nodes_consumed = n_nodes_before_child - *n_nodes;
        child += nodes_consumed;
      }
      break;
    }
    default:
      GeoArrowErrorSet(v->error, "Invalid geometry_type: %d", (int)node->geometry_type);
      return EINVAL;
  }

  GEOARROW_RETURN_NOT_OK(v->geom_end(v));
  return GEOARROW_OK;
}

GeoArrowErrorCode GeoArrowGeometryViewVisit(struct GeoArrowGeometryView geometry,
                                            struct GeoArrowVisitor* v) {
  int64_t n_nodes = geometry.size_nodes;
  GEOARROW_RETURN_NOT_OK(v->feat_start(v));
  GEOARROW_RETURN_NOT_OK(GeoArrowGeometryVisitNode(geometry.root, &n_nodes, v));
  if (n_nodes != 0) {
    GeoArrowErrorSet(
        v->error, "Too many nodes provided to GeoArrowGeometryVisit() for root geometry");
    return EINVAL;
  }

  GEOARROW_RETURN_NOT_OK(v->feat_end(v));
  return GEOARROW_OK;
}

GeoArrowErrorCode GeoArrowGeometryVisit(const struct GeoArrowGeometry* geometry,
                                        struct GeoArrowVisitor* v) {
  int64_t n_nodes = geometry->size_nodes;
  GEOARROW_RETURN_NOT_OK(v->feat_start(v));
  GEOARROW_RETURN_NOT_OK(GeoArrowGeometryVisitNode(geometry->root, &n_nodes, v));
  if (n_nodes != 0) {
    GeoArrowErrorSet(
        v->error, "Too many nodes provided to GeoArrowGeometryVisit() for root geometry");
    return EINVAL;
  }

  GEOARROW_RETURN_NOT_OK(v->feat_end(v));
  return GEOARROW_OK;
}

#include <string.h>

#include "nanoarrow/nanoarrow.h"



// Bytes for four quiet (little-endian) NANs
static uint8_t kEmptyPointCoords[] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f,
                                      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f,
                                      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f,
                                      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f};

struct GeoArrowNativeWriterPrivate {
  struct GeoArrowBuilder builder;

  struct ArrowBitmap validity;
  int64_t null_count;
  int64_t current_length;
  int32_t last_offset[4];

  // Fields to keep track of state
  int output_initialized;
  int feat_is_null;
  int nesting_multipoint;
  double empty_coord_values[4];
  struct GeoArrowCoordView empty_coord;
  enum GeoArrowDimensions last_dimensions;
  int64_t size[32];
  int32_t level;
};

static GeoArrowErrorCode GeoArrowNativeWriterEnsureOutputInitialized(
    struct GeoArrowNativeWriter* writer);

GeoArrowErrorCode GeoArrowNativeWriterInit(struct GeoArrowNativeWriter* writer,
                                           enum GeoArrowType type) {
  struct GeoArrowNativeWriterPrivate* private_data =
      (struct GeoArrowNativeWriterPrivate*)ArrowMalloc(
          sizeof(struct GeoArrowNativeWriterPrivate));
  if (private_data == NULL) {
    return ENOMEM;
  }

  memset(private_data, 0, sizeof(struct GeoArrowNativeWriterPrivate));

  GeoArrowErrorCode result = GeoArrowBuilderInitFromType(&private_data->builder, type);
  if (result != GEOARROW_OK) {
    ArrowFree(private_data);
    return result;
  }

  ArrowBitmapInit(&private_data->validity);

  // Initialize one empty coordinate
  memcpy(private_data->empty_coord_values, kEmptyPointCoords, 4 * sizeof(double));
  private_data->empty_coord.values[0] = private_data->empty_coord_values;
  private_data->empty_coord.values[1] = private_data->empty_coord_values + 1;
  private_data->empty_coord.values[2] = private_data->empty_coord_values + 2;
  private_data->empty_coord.values[3] = private_data->empty_coord_values + 3;
  private_data->empty_coord.n_coords = 1;
  private_data->empty_coord.n_values = 4;
  private_data->empty_coord.coords_stride = 1;

  writer->private_data = private_data;

  result = GeoArrowNativeWriterEnsureOutputInitialized(writer);
  if (result != GEOARROW_OK) {
    GeoArrowNativeWriterReset(writer);
    return result;
  }

  return GEOARROW_OK;
}

void GeoArrowNativeWriterReset(struct GeoArrowNativeWriter* writer) {
  struct GeoArrowNativeWriterPrivate* private_data =
      (struct GeoArrowNativeWriterPrivate*)writer->private_data;
  GeoArrowBuilderReset(&private_data->builder);
  ArrowBitmapReset(&private_data->validity);
  ArrowFree(private_data);
}

static GeoArrowErrorCode GeoArrowNativeWriterEnsureOutputInitialized(
    struct GeoArrowNativeWriter* writer) {
  struct GeoArrowNativeWriterPrivate* private_data =
      (struct GeoArrowNativeWriterPrivate*)writer->private_data;

  if (private_data->output_initialized) {
    return GEOARROW_OK;
  }

  struct GeoArrowBuilder* builder = &private_data->builder;

  int32_t zero = 0;
  for (int i = 0; i < private_data->builder.view.n_offsets; i++) {
    NANOARROW_RETURN_NOT_OK(GeoArrowBuilderOffsetAppend(builder, i, &zero, 1));
  }

  private_data->null_count = 0;
  private_data->current_length = 0;
  NANOARROW_RETURN_NOT_OK(ArrowBitmapResize(&private_data->validity, 0, 0));
  memset(private_data->last_offset, 0, sizeof(private_data->last_offset));

  private_data->builder.view.coords.size_coords = 0;
  private_data->builder.view.coords.capacity_coords = 0;

  private_data->output_initialized = 1;
  return GEOARROW_OK;
}

GeoArrowErrorCode GeoArrowNativeWriterFinish(struct GeoArrowNativeWriter* writer,
                                             struct ArrowArray* array,
                                             struct GeoArrowError* error) {
  struct GeoArrowNativeWriterPrivate* private_data =
      (struct GeoArrowNativeWriterPrivate*)writer->private_data;

  // We could in theory wrap this buffer instead of copy it
  struct GeoArrowBufferView validity_view;
  validity_view.data = private_data->validity.buffer.data;
  validity_view.size_bytes = private_data->validity.buffer.size_bytes;
  if (validity_view.size_bytes > 0) {
    GEOARROW_RETURN_NOT_OK(
        GeoArrowBuilderAppendBuffer(&private_data->builder, 0, validity_view));
  }

  struct ArrowArray tmp;
  GEOARROW_RETURN_NOT_OK(GeoArrowBuilderFinish(&private_data->builder, &tmp, error));
  tmp.null_count = private_data->null_count;

  private_data->output_initialized = 0;

  GeoArrowErrorCode result = GeoArrowNativeWriterEnsureOutputInitialized(writer);
  if (result != GEOARROW_OK) {
    ArrowArrayRelease(&tmp);
    GeoArrowErrorSet(error, "Failed to reinitialize writer");
    return result;
  }

  ArrowArrayMove(&tmp, array);
  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowNativeWriterAppendSize(
    struct GeoArrowNativeWriter* writer, uint32_t offset, uint32_t size,
    struct GeoArrowError* error) {
  struct GeoArrowNativeWriterPrivate* private_data =
      (struct GeoArrowNativeWriterPrivate*)writer->private_data;

  // Handle the offsets (and determine if this geometry can, in fact, be added to the
  // builder)
  if (size > INT32_MAX) {
    GeoArrowErrorSet(
        error, "Can't append geometry with size > INT32_MAX to GeoArrow native array");
    return EOVERFLOW;
  }

  int32_t* p_last_offset = private_data->last_offset + offset;
  (*p_last_offset) += size;

  GEOARROW_RETURN_NOT_OK(
      GeoArrowBuilderOffsetAppend(&private_data->builder, offset, p_last_offset, 1));

  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowNativeWriterAppendEmpty(
    struct GeoArrowNativeWriter* writer, struct GeoArrowError* error) {
  struct GeoArrowNativeWriterPrivate* private_data =
      (struct GeoArrowNativeWriterPrivate*)writer->private_data;

  switch (private_data->builder.view.schema_view.geometry_type) {
    case GEOARROW_GEOMETRY_TYPE_POINT: {
      GEOARROW_RETURN_NOT_OK(GeoArrowBuilderCoordsAppend(
          &private_data->builder, &private_data->empty_coord,
          private_data->builder.view.schema_view.dimensions, 0, 1));
      return NANOARROW_OK;
    }
    case GEOARROW_GEOMETRY_TYPE_LINESTRING:
    case GEOARROW_GEOMETRY_TYPE_POLYGON:
    case GEOARROW_GEOMETRY_TYPE_MULTIPOINT:
    case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING:
    case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON:
    case GEOARROW_GEOMETRY_TYPE_GEOMETRYCOLLECTION:
      GEOARROW_RETURN_NOT_OK(GeoArrowNativeWriterAppendSize(writer, 0, 0, error));
      return NANOARROW_OK;
    default:
      NANOARROW_DCHECK(0 && "unreachable");
      return EINVAL;
  }
}

static void GeoArrowNativeWriterAppendCoordsUnsafe(struct GeoArrowNativeWriter* writer,
                                                   struct GeoArrowGeometryView geom) {
  struct GeoArrowNativeWriterPrivate* private_data =
      (struct GeoArrowNativeWriterPrivate*)writer->private_data;

  struct GeoArrowWritableCoordView* out_coords = &private_data->builder.view.coords;
  uint8_t* out[4];
  int32_t strides[4];
  for (int i = 0; i < 4; i++) {
    out[i] = (uint8_t*)(out_coords->values[i] +
                        out_coords->coords_stride * out_coords->size_coords);
    strides[i] = out_coords->coords_stride * sizeof(double);
  }

  GeoArrowGeometryViewCopyCoordsGeneric(
      geom, out, strides, private_data->builder.view.schema_view.dimensions);
}

static GeoArrowErrorCode GeoArrowNativeWriterAppendLinestringOffsets(
    struct GeoArrowNativeWriter* writer, const struct GeoArrowGeometryNode* node,
    uint32_t n, int32_t offset, struct GeoArrowError* error) {
  for (int64_t i = 0; i < n; i++) {
    GEOARROW_RETURN_NOT_OK(
        GeoArrowNativeWriterAppendSize(writer, offset, node->size, error));
    ++node;
  }

  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowNativeWriterAppendPolygonOffsets(
    struct GeoArrowNativeWriter* writer, const struct GeoArrowGeometryNode* node,
    uint32_t n, int32_t offset, struct GeoArrowError* error) {
  for (int64_t i = 0; i < n; i++) {
    GEOARROW_RETURN_NOT_OK(
        GeoArrowNativeWriterAppendSize(writer, offset, node->size, error));
    GEOARROW_RETURN_NOT_OK(GeoArrowNativeWriterAppendLinestringOffsets(
        writer, node + 1, node->size, offset + 1, error));

    node += node->size + 1;
  }

  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowNativeWriterAppendMultiPolygonOffsets(
    struct GeoArrowNativeWriter* writer, const struct GeoArrowGeometryNode* node,
    struct GeoArrowError* error) {
  GEOARROW_RETURN_NOT_OK(GeoArrowNativeWriterAppendSize(writer, 0, node->size, error));
  uint32_t n_polygons = node->size;
  ++node;
  for (uint32_t i = 0; i < n_polygons; i++) {
    GEOARROW_RETURN_NOT_OK(GeoArrowNativeWriterAppendSize(writer, 1, node->size, error));
    GEOARROW_RETURN_NOT_OK(GeoArrowNativeWriterAppendLinestringOffsets(
        writer, node + 1, node->size, 2, error));
    node += node->size + 1;
  }

  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowNativeWriterAppendValidity(
    struct GeoArrowNativeWriter* writer, int is_valid) {
  struct GeoArrowNativeWriterPrivate* private_data =
      (struct GeoArrowNativeWriterPrivate*)writer->private_data;

  private_data->current_length++;

  if (!is_valid) {
    if (private_data->validity.buffer.data == NULL) {
      NANOARROW_RETURN_NOT_OK(
          ArrowBitmapReserve(&private_data->validity, private_data->current_length));
      ArrowBitmapAppendUnsafe(&private_data->validity, 1,
                              private_data->current_length - 1);
    }

    private_data->null_count++;
    NANOARROW_RETURN_NOT_OK(ArrowBitmapAppend(&private_data->validity, 0, 1));
  } else if (private_data->validity.buffer.data != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowBitmapAppend(&private_data->validity, 1, 1));
  }

  return GEOARROW_OK;
}

GeoArrowErrorCode GeoArrowNativeWriterAppendNull(struct GeoArrowNativeWriter* writer) {
  GEOARROW_RETURN_NOT_OK(GeoArrowNativeWriterAppendEmpty(writer, NULL));
  GEOARROW_RETURN_NOT_OK(GeoArrowNativeWriterAppendValidity(writer, 0));
  return GEOARROW_OK;
}

GeoArrowErrorCode GeoArrowNativeWriterAppend(struct GeoArrowNativeWriter* writer,
                                             struct GeoArrowGeometryView geom,
                                             struct GeoArrowError* error) {
  struct GeoArrowNativeWriterPrivate* private_data =
      (struct GeoArrowNativeWriterPrivate*)writer->private_data;

  uint32_t coord_count = GeoArrowGeometryViewNumCoords(geom);

  const struct GeoArrowGeometryNode* node = geom.root;

  // Any EMPTY can be appended to any builder
  if (coord_count == 0) {
    GEOARROW_RETURN_NOT_OK(GeoArrowNativeWriterAppendEmpty(writer, error));
    GEOARROW_RETURN_NOT_OK(GeoArrowNativeWriterAppendValidity(writer, 1));
    return GEOARROW_OK;
  }

  // Handle the offsets (and determine if this geometry can, in fact, be
  // added to the builder)
  switch (private_data->builder.view.schema_view.geometry_type) {
    case GEOARROW_GEOMETRY_TYPE_POINT:
      if (coord_count > 1) {
        GeoArrowErrorSet(
            error, "Can't append geometry with coord count >1 to array of type POINT");
        return EINVAL;
      }
      break;
    case GEOARROW_GEOMETRY_TYPE_LINESTRING:
      switch (node->geometry_type) {
        case GEOARROW_GEOMETRY_TYPE_LINESTRING:
          GEOARROW_RETURN_NOT_OK(
              GeoArrowNativeWriterAppendLinestringOffsets(writer, node, 1, 0, error));
          break;

        case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING:
          if (node->size > 1) {
            GeoArrowErrorSet(
                error,
                "Can't append MULTILINESTRING with size >1 to array of type LINESTRING");
            return EINVAL;
          }

          GEOARROW_RETURN_NOT_OK(
              GeoArrowNativeWriterAppendLinestringOffsets(writer, node + 1, 1, 0, error));
          break;

        default:
          GeoArrowErrorSet(error, "Can't append %s to array of type LINESTRING",
                           GeoArrowGeometryTypeString(node->geometry_type));
          return EINVAL;
      }
      break;
    case GEOARROW_GEOMETRY_TYPE_POLYGON:
      switch (node->geometry_type) {
        case GEOARROW_GEOMETRY_TYPE_POLYGON: {
          GEOARROW_RETURN_NOT_OK(
              GeoArrowNativeWriterAppendPolygonOffsets(writer, node, 1, 0, error));
          break;
        }
        case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON:
          if (node->size > 1) {
            GeoArrowErrorSet(
                error, "Can't append MULTIPOLYGON with size >1 to array of type POLYGON");
            return EINVAL;
          }

          GEOARROW_RETURN_NOT_OK(
              GeoArrowNativeWriterAppendPolygonOffsets(writer, node + 1, 1, 0, error));
          break;

        default:
          GeoArrowErrorSet(error, "Can't append %s to array of type POLYGON",
                           GeoArrowGeometryTypeString(node->geometry_type));
          return EINVAL;
      }
      break;
    case GEOARROW_GEOMETRY_TYPE_MULTIPOINT:
      switch (node->geometry_type) {
        case GEOARROW_GEOMETRY_TYPE_POINT:
        case GEOARROW_GEOMETRY_TYPE_MULTIPOINT:
          GEOARROW_RETURN_NOT_OK(
              GeoArrowNativeWriterAppendSize(writer, 0, coord_count, error));
          break;

        default:
          GeoArrowErrorSet(error, "Can't append %s to array of type MULTIPOINT",
                           GeoArrowGeometryTypeString(node->geometry_type));
          return EINVAL;
      }
      break;
    case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING:
      switch (node->geometry_type) {
        case GEOARROW_GEOMETRY_TYPE_LINESTRING:
          // Append a one to the first offset buffer
          GEOARROW_RETURN_NOT_OK(GeoArrowNativeWriterAppendSize(writer, 0, 1, error));

          // Append the linestring to the second offset buffer
          GEOARROW_RETURN_NOT_OK(
              GeoArrowNativeWriterAppendLinestringOffsets(writer, node, 1, 1, error));
          break;

        case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING:
          // Same math as appending a Polygon
          GEOARROW_RETURN_NOT_OK(
              GeoArrowNativeWriterAppendPolygonOffsets(writer, node, 1, 0, error));
          break;

        default:
          GeoArrowErrorSet(error, "Can't append %s to array of type MULTILINESTRING",
                           GeoArrowGeometryTypeString(node->geometry_type));
          return EINVAL;
      }
      break;
    case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON:
      switch (node->geometry_type) {
        case GEOARROW_GEOMETRY_TYPE_POLYGON:
          // Append a one to the first offset buffer
          GEOARROW_RETURN_NOT_OK(GeoArrowNativeWriterAppendSize(writer, 0, 1, error));

          // Append the polygon to the inner offset buffer
          GEOARROW_RETURN_NOT_OK(
              GeoArrowNativeWriterAppendPolygonOffsets(writer, node, 1, 1, error));
          break;

        case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON:
          GEOARROW_RETURN_NOT_OK(
              GeoArrowNativeWriterAppendMultiPolygonOffsets(writer, node, error));
          break;

        default:
          GeoArrowErrorSet(error, "Can't append %s to array of type MULTIPOLYGON",
                           GeoArrowGeometryTypeString(node->geometry_type));
          return EINVAL;
      }
      break;
    default:
      GeoArrowErrorSet(error, "Unexpected builder geometry type");
      return EINVAL;
  }

  // Append coords
  GEOARROW_RETURN_NOT_OK(
      GeoArrowBuilderCoordsReserve(&private_data->builder, coord_count));
  GeoArrowNativeWriterAppendCoordsUnsafe(writer, geom);
  private_data->builder.view.coords.size_coords += coord_count;

  // Append to validity buffer
  GEOARROW_RETURN_NOT_OK(GeoArrowNativeWriterAppendValidity(writer, 1));

  return GEOARROW_OK;
}

static int feat_start_point(struct GeoArrowVisitor* v) {
  struct GeoArrowNativeWriter* writer = (struct GeoArrowNativeWriter*)v->private_data;
  struct GeoArrowNativeWriterPrivate* private_data =
      (struct GeoArrowNativeWriterPrivate*)writer->private_data;
  private_data->level = 0;
  private_data->size[0] = 0;
  private_data->feat_is_null = 0;
  return GEOARROW_OK;
}

static int geom_start_point(struct GeoArrowVisitor* v,
                            enum GeoArrowGeometryType geometry_type,
                            enum GeoArrowDimensions dimensions) {
  NANOARROW_UNUSED(geometry_type);

  // level++, geometry type, dimensions, reset size
  // validate dimensions, maybe against some options that indicate
  // error for mismatch, fill, or drop behaviour
  struct GeoArrowNativeWriter* writer = (struct GeoArrowNativeWriter*)v->private_data;
  struct GeoArrowNativeWriterPrivate* private_data =
      (struct GeoArrowNativeWriterPrivate*)writer->private_data;
  private_data->last_dimensions = dimensions;
  return GEOARROW_OK;
}

static int ring_start_point(struct GeoArrowVisitor* v) {
  NANOARROW_UNUSED(v);
  return GEOARROW_OK;
}

static int coords_point(struct GeoArrowVisitor* v,
                        const struct GeoArrowCoordView* coords) {
  struct GeoArrowNativeWriter* writer = (struct GeoArrowNativeWriter*)v->private_data;
  struct GeoArrowNativeWriterPrivate* private_data =
      (struct GeoArrowNativeWriterPrivate*)writer->private_data;
  private_data->size[0] += coords->n_coords;
  return GeoArrowBuilderCoordsAppend(&private_data->builder, coords,
                                     private_data->last_dimensions, 0, coords->n_coords);
}

static int ring_end_point(struct GeoArrowVisitor* v) {
  NANOARROW_UNUSED(v);
  return GEOARROW_OK;
}

static int geom_end_point(struct GeoArrowVisitor* v) {
  NANOARROW_UNUSED(v);
  return GEOARROW_OK;
}

static int null_feat_point(struct GeoArrowVisitor* v) {
  struct GeoArrowNativeWriter* writer = (struct GeoArrowNativeWriter*)v->private_data;
  struct GeoArrowNativeWriterPrivate* private_data =
      (struct GeoArrowNativeWriterPrivate*)writer->private_data;
  private_data->feat_is_null = 1;
  return GEOARROW_OK;
}

static int feat_end_point(struct GeoArrowVisitor* v) {
  struct GeoArrowNativeWriter* writer = (struct GeoArrowNativeWriter*)v->private_data;
  struct GeoArrowNativeWriterPrivate* private_data =
      (struct GeoArrowNativeWriterPrivate*)writer->private_data;

  // If there weren't any coords (i.e., EMPTY), we need to write some NANs here
  // if there was >1 coords, we also need to error or we'll get misaligned output
  if (private_data->size[0] == 0) {
    int n_dim =
        _GeoArrowkNumDimensions[private_data->builder.view.schema_view.dimensions];
    private_data->empty_coord.n_values = n_dim;
    NANOARROW_RETURN_NOT_OK(coords_point(v, &private_data->empty_coord));
  } else if (private_data->size[0] != 1) {
    GeoArrowErrorSet(v->error, "Can't convert feature with >1 coordinate to POINT");
    return EINVAL;
  }

  if (private_data->feat_is_null) {
    int64_t current_length = private_data->builder.view.coords.size_coords;
    if (private_data->validity.buffer.data == NULL) {
      NANOARROW_RETURN_NOT_OK(
          ArrowBitmapReserve(&private_data->validity, current_length));
      ArrowBitmapAppendUnsafe(&private_data->validity, 1, current_length - 1);
    }

    private_data->null_count++;
    NANOARROW_RETURN_NOT_OK(ArrowBitmapAppend(&private_data->validity, 0, 1));
  } else if (private_data->validity.buffer.data != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowBitmapAppend(&private_data->validity, 1, 1));
  }

  return GEOARROW_OK;
}

static void GeoArrowVisitorInitPoint(struct GeoArrowVisitor* v) {
  struct GeoArrowError* previous_error = v->error;
  GeoArrowVisitorInitVoid(v);
  v->error = previous_error;

  v->feat_start = &feat_start_point;
  v->null_feat = &null_feat_point;
  v->geom_start = &geom_start_point;
  v->ring_start = &ring_start_point;
  v->coords = &coords_point;
  v->ring_end = &ring_end_point;
  v->geom_end = &geom_end_point;
  v->feat_end = &feat_end_point;
}

static int feat_start_multipoint(struct GeoArrowVisitor* v) {
  struct GeoArrowNativeWriter* writer = (struct GeoArrowNativeWriter*)v->private_data;
  struct GeoArrowNativeWriterPrivate* private_data =
      (struct GeoArrowNativeWriterPrivate*)writer->private_data;
  private_data->level = 0;
  private_data->size[0] = 0;
  private_data->size[1] = 0;
  private_data->feat_is_null = 0;
  private_data->nesting_multipoint = 0;
  return GEOARROW_OK;
}

static int geom_start_multipoint(struct GeoArrowVisitor* v,
                                 enum GeoArrowGeometryType geometry_type,
                                 enum GeoArrowDimensions dimensions) {
  struct GeoArrowNativeWriter* writer = (struct GeoArrowNativeWriter*)v->private_data;
  struct GeoArrowNativeWriterPrivate* private_data =
      (struct GeoArrowNativeWriterPrivate*)writer->private_data;
  private_data->last_dimensions = dimensions;

  switch (geometry_type) {
    case GEOARROW_GEOMETRY_TYPE_LINESTRING:
      private_data->level++;
      break;
    case GEOARROW_GEOMETRY_TYPE_MULTIPOINT:
      private_data->nesting_multipoint = 1;
      private_data->level++;
      break;
    case GEOARROW_GEOMETRY_TYPE_POINT:
      if (private_data->nesting_multipoint) {
        private_data->nesting_multipoint++;
      }
    default:
      break;
  }

  return GEOARROW_OK;
}

static int ring_start_multipoint(struct GeoArrowVisitor* v) {
  struct GeoArrowNativeWriter* writer = (struct GeoArrowNativeWriter*)v->private_data;
  struct GeoArrowNativeWriterPrivate* private_data =
      (struct GeoArrowNativeWriterPrivate*)writer->private_data;
  private_data->level++;
  return GEOARROW_OK;
}

static int coords_multipoint(struct GeoArrowVisitor* v,
                             const struct GeoArrowCoordView* coords) {
  struct GeoArrowNativeWriter* writer = (struct GeoArrowNativeWriter*)v->private_data;
  struct GeoArrowNativeWriterPrivate* private_data =
      (struct GeoArrowNativeWriterPrivate*)writer->private_data;
  private_data->size[1] += coords->n_coords;
  return GeoArrowBuilderCoordsAppend(&private_data->builder, coords,
                                     private_data->last_dimensions, 0, coords->n_coords);
}

static int ring_end_multipoint(struct GeoArrowVisitor* v) {
  struct GeoArrowNativeWriter* writer = (struct GeoArrowNativeWriter*)v->private_data;
  struct GeoArrowNativeWriterPrivate* private_data =
      (struct GeoArrowNativeWriterPrivate*)writer->private_data;

  private_data->level--;
  private_data->size[0]++;
  if (private_data->builder.view.coords.size_coords > 2147483647) {
    return EOVERFLOW;
  }
  int32_t n_coord32 = (int32_t)private_data->builder.view.coords.size_coords;
  NANOARROW_RETURN_NOT_OK(
      GeoArrowBuilderOffsetAppend(&private_data->builder, 0, &n_coord32, 1));

  return GEOARROW_OK;
}

static int geom_end_multipoint(struct GeoArrowVisitor* v) {
  struct GeoArrowNativeWriter* writer = (struct GeoArrowNativeWriter*)v->private_data;
  struct GeoArrowNativeWriterPrivate* private_data =
      (struct GeoArrowNativeWriterPrivate*)writer->private_data;

  // Ignore geom_end calls from the end of a POINT nested within a MULTIPOINT
  if (private_data->nesting_multipoint == 2) {
    private_data->nesting_multipoint--;
    return GEOARROW_OK;
  }

  if (private_data->level == 1) {
    private_data->size[0]++;
    private_data->level--;
    if (private_data->builder.view.coords.size_coords > 2147483647) {
      return EOVERFLOW;
    }
    int32_t n_coord32 = (int32_t)private_data->builder.view.coords.size_coords;
    NANOARROW_RETURN_NOT_OK(
        GeoArrowBuilderOffsetAppend(&private_data->builder, 0, &n_coord32, 1));
  }

  return GEOARROW_OK;
}

static int null_feat_multipoint(struct GeoArrowVisitor* v) {
  struct GeoArrowNativeWriter* writer = (struct GeoArrowNativeWriter*)v->private_data;
  struct GeoArrowNativeWriterPrivate* private_data =
      (struct GeoArrowNativeWriterPrivate*)writer->private_data;
  private_data->feat_is_null = 1;
  return GEOARROW_OK;
}

static int feat_end_multipoint(struct GeoArrowVisitor* v) {
  struct GeoArrowNativeWriter* writer = (struct GeoArrowNativeWriter*)v->private_data;
  struct GeoArrowNativeWriterPrivate* private_data =
      (struct GeoArrowNativeWriterPrivate*)writer->private_data;

  // If we didn't finish any sequences, finish at least one. This is usually an
  // EMPTY but could also be a single point.
  if (private_data->size[0] == 0) {
    if (private_data->builder.view.coords.size_coords > 2147483647) {
      return EOVERFLOW;
    }
    int32_t n_coord32 = (int32_t)private_data->builder.view.coords.size_coords;
    NANOARROW_RETURN_NOT_OK(
        GeoArrowBuilderOffsetAppend(&private_data->builder, 0, &n_coord32, 1));
  } else if (private_data->size[0] != 1) {
    GeoArrowErrorSet(v->error, "Can't convert feature with >1 sequence to LINESTRING");
    return EINVAL;
  }

  if (private_data->feat_is_null) {
    int64_t current_length =
        private_data->builder.view.buffers[1].size_bytes / sizeof(int32_t) - 1;
    if (private_data->validity.buffer.data == NULL) {
      NANOARROW_RETURN_NOT_OK(
          ArrowBitmapReserve(&private_data->validity, current_length));
      ArrowBitmapAppendUnsafe(&private_data->validity, 1, current_length - 1);
    }

    private_data->null_count++;
    NANOARROW_RETURN_NOT_OK(ArrowBitmapAppend(&private_data->validity, 0, 1));
  } else if (private_data->validity.buffer.data != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowBitmapAppend(&private_data->validity, 1, 1));
  }

  return GEOARROW_OK;
}

static void GeoArrowVisitorInitLinestring(struct GeoArrowVisitor* v) {
  struct GeoArrowError* previous_error = v->error;
  GeoArrowVisitorInitVoid(v);
  v->error = previous_error;

  v->feat_start = &feat_start_multipoint;
  v->null_feat = &null_feat_multipoint;
  v->geom_start = &geom_start_multipoint;
  v->ring_start = &ring_start_multipoint;
  v->coords = &coords_multipoint;
  v->ring_end = &ring_end_multipoint;
  v->geom_end = &geom_end_multipoint;
  v->feat_end = &feat_end_multipoint;
}

static int feat_start_multilinestring(struct GeoArrowVisitor* v) {
  struct GeoArrowNativeWriter* writer = (struct GeoArrowNativeWriter*)v->private_data;
  struct GeoArrowNativeWriterPrivate* private_data =
      (struct GeoArrowNativeWriterPrivate*)writer->private_data;
  private_data->level = 0;
  private_data->size[0] = 0;
  private_data->size[1] = 0;
  private_data->feat_is_null = 0;
  return GEOARROW_OK;
}

static int geom_start_multilinestring(struct GeoArrowVisitor* v,
                                      enum GeoArrowGeometryType geometry_type,
                                      enum GeoArrowDimensions dimensions) {
  struct GeoArrowNativeWriter* writer = (struct GeoArrowNativeWriter*)v->private_data;
  struct GeoArrowNativeWriterPrivate* private_data =
      (struct GeoArrowNativeWriterPrivate*)writer->private_data;
  private_data->last_dimensions = dimensions;

  switch (geometry_type) {
    case GEOARROW_GEOMETRY_TYPE_LINESTRING:
    case GEOARROW_GEOMETRY_TYPE_MULTIPOINT:
      private_data->level++;
      break;
    default:
      break;
  }

  return GEOARROW_OK;
}

static int ring_start_multilinestring(struct GeoArrowVisitor* v) {
  struct GeoArrowNativeWriter* writer = (struct GeoArrowNativeWriter*)v->private_data;
  struct GeoArrowNativeWriterPrivate* private_data =
      (struct GeoArrowNativeWriterPrivate*)writer->private_data;
  private_data->level++;
  return GEOARROW_OK;
}

static int coords_multilinestring(struct GeoArrowVisitor* v,
                                  const struct GeoArrowCoordView* coords) {
  struct GeoArrowNativeWriter* writer = (struct GeoArrowNativeWriter*)v->private_data;
  struct GeoArrowNativeWriterPrivate* private_data =
      (struct GeoArrowNativeWriterPrivate*)writer->private_data;
  private_data->size[1] += coords->n_coords;
  return GeoArrowBuilderCoordsAppend(&private_data->builder, coords,
                                     private_data->last_dimensions, 0, coords->n_coords);
}

static int ring_end_multilinestring(struct GeoArrowVisitor* v) {
  struct GeoArrowNativeWriter* writer = (struct GeoArrowNativeWriter*)v->private_data;
  struct GeoArrowNativeWriterPrivate* private_data =
      (struct GeoArrowNativeWriterPrivate*)writer->private_data;

  private_data->level--;
  if (private_data->size[1] > 0) {
    if (private_data->builder.view.coords.size_coords > 2147483647) {
      return EOVERFLOW;
    }
    int32_t n_coord32 = (int32_t)private_data->builder.view.coords.size_coords;
    NANOARROW_RETURN_NOT_OK(
        GeoArrowBuilderOffsetAppend(&private_data->builder, 1, &n_coord32, 1));
    private_data->size[0]++;
    private_data->size[1] = 0;
  }

  return GEOARROW_OK;
}

static int geom_end_multilinestring(struct GeoArrowVisitor* v) {
  struct GeoArrowNativeWriter* writer = (struct GeoArrowNativeWriter*)v->private_data;
  struct GeoArrowNativeWriterPrivate* private_data =
      (struct GeoArrowNativeWriterPrivate*)writer->private_data;

  if (private_data->level == 1) {
    private_data->level--;
    if (private_data->size[1] > 0) {
      if (private_data->builder.view.coords.size_coords > 2147483647) {
        return EOVERFLOW;
      }
      int32_t n_coord32 = (int32_t)private_data->builder.view.coords.size_coords;
      NANOARROW_RETURN_NOT_OK(
          GeoArrowBuilderOffsetAppend(&private_data->builder, 1, &n_coord32, 1));
      private_data->size[0]++;
      private_data->size[1] = 0;
    }
  }

  return GEOARROW_OK;
}

static int null_feat_multilinestring(struct GeoArrowVisitor* v) {
  struct GeoArrowNativeWriter* writer = (struct GeoArrowNativeWriter*)v->private_data;
  struct GeoArrowNativeWriterPrivate* private_data =
      (struct GeoArrowNativeWriterPrivate*)writer->private_data;
  private_data->feat_is_null = 1;
  return GEOARROW_OK;
}

static int feat_end_multilinestring(struct GeoArrowVisitor* v) {
  struct GeoArrowNativeWriter* writer = (struct GeoArrowNativeWriter*)v->private_data;
  struct GeoArrowNativeWriterPrivate* private_data =
      (struct GeoArrowNativeWriterPrivate*)writer->private_data;

  // If we have an unfinished sequence left over, finish it now. This could have
  // occurred if the last geometry that was visited was a POINT.
  if (private_data->size[1] > 0) {
    if (private_data->builder.view.coords.size_coords > 2147483647) {
      return EOVERFLOW;
    }
    int32_t n_coord32 = (int32_t)private_data->builder.view.coords.size_coords;
    NANOARROW_RETURN_NOT_OK(
        GeoArrowBuilderOffsetAppend(&private_data->builder, 1, &n_coord32, 1));
  }

  // Finish off the sequence of sequences. This is a polygon or multilinestring
  // so it can any number of them.
  int32_t n_seq32 =
      (int32_t)(private_data->builder.view.buffers[2].size_bytes / sizeof(int32_t)) - 1;
  NANOARROW_RETURN_NOT_OK(
      GeoArrowBuilderOffsetAppend(&private_data->builder, 0, &n_seq32, 1));

  if (private_data->feat_is_null) {
    int64_t current_length =
        private_data->builder.view.buffers[1].size_bytes / sizeof(int32_t) - 1;
    if (private_data->validity.buffer.data == NULL) {
      NANOARROW_RETURN_NOT_OK(
          ArrowBitmapReserve(&private_data->validity, current_length));
      ArrowBitmapAppendUnsafe(&private_data->validity, 1, current_length - 1);
    }

    private_data->null_count++;
    NANOARROW_RETURN_NOT_OK(ArrowBitmapAppend(&private_data->validity, 0, 1));
  } else if (private_data->validity.buffer.data != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowBitmapAppend(&private_data->validity, 1, 1));
  }

  return GEOARROW_OK;
}

static void GeoArrowVisitorInitMultiLinestring(struct GeoArrowVisitor* v) {
  struct GeoArrowError* previous_error = v->error;
  GeoArrowVisitorInitVoid(v);
  v->error = previous_error;

  v->feat_start = &feat_start_multilinestring;
  v->null_feat = &null_feat_multilinestring;
  v->geom_start = &geom_start_multilinestring;
  v->ring_start = &ring_start_multilinestring;
  v->coords = &coords_multilinestring;
  v->ring_end = &ring_end_multilinestring;
  v->geom_end = &geom_end_multilinestring;
  v->feat_end = &feat_end_multilinestring;
}

static int feat_start_multipolygon(struct GeoArrowVisitor* v) {
  struct GeoArrowNativeWriter* writer = (struct GeoArrowNativeWriter*)v->private_data;
  struct GeoArrowNativeWriterPrivate* private_data =
      (struct GeoArrowNativeWriterPrivate*)writer->private_data;
  private_data->level = 0;
  private_data->size[0] = 0;
  private_data->size[1] = 0;
  private_data->size[2] = 0;
  private_data->feat_is_null = 0;
  return GEOARROW_OK;
}

static int geom_start_multipolygon(struct GeoArrowVisitor* v,
                                   enum GeoArrowGeometryType geometry_type,
                                   enum GeoArrowDimensions dimensions) {
  struct GeoArrowNativeWriter* writer = (struct GeoArrowNativeWriter*)v->private_data;
  struct GeoArrowNativeWriterPrivate* private_data =
      (struct GeoArrowNativeWriterPrivate*)writer->private_data;
  private_data->last_dimensions = dimensions;

  switch (geometry_type) {
    case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING:
    case GEOARROW_GEOMETRY_TYPE_POLYGON:
    case GEOARROW_GEOMETRY_TYPE_LINESTRING:
    case GEOARROW_GEOMETRY_TYPE_MULTIPOINT:
      private_data->level++;
      break;
    default:
      break;
  }

  return GEOARROW_OK;
}

static int ring_start_multipolygon(struct GeoArrowVisitor* v) {
  struct GeoArrowNativeWriter* writer = (struct GeoArrowNativeWriter*)v->private_data;
  struct GeoArrowNativeWriterPrivate* private_data =
      (struct GeoArrowNativeWriterPrivate*)writer->private_data;
  private_data->level++;
  return GEOARROW_OK;
}

static int coords_multipolygon(struct GeoArrowVisitor* v,
                               const struct GeoArrowCoordView* coords) {
  struct GeoArrowNativeWriter* writer = (struct GeoArrowNativeWriter*)v->private_data;
  struct GeoArrowNativeWriterPrivate* private_data =
      (struct GeoArrowNativeWriterPrivate*)writer->private_data;
  private_data->size[2] += coords->n_coords;
  return GeoArrowBuilderCoordsAppend(&private_data->builder, coords,
                                     private_data->last_dimensions, 0, coords->n_coords);
}

static int ring_end_multipolygon(struct GeoArrowVisitor* v) {
  struct GeoArrowNativeWriter* writer = (struct GeoArrowNativeWriter*)v->private_data;
  struct GeoArrowNativeWriterPrivate* private_data =
      (struct GeoArrowNativeWriterPrivate*)writer->private_data;

  private_data->level--;
  if (private_data->size[2] > 0) {
    if (private_data->builder.view.coords.size_coords > 2147483647) {
      return EOVERFLOW;
    }
    int32_t n_coord32 = (int32_t)private_data->builder.view.coords.size_coords;
    NANOARROW_RETURN_NOT_OK(
        GeoArrowBuilderOffsetAppend(&private_data->builder, 2, &n_coord32, 1));
    private_data->size[1]++;
    private_data->size[2] = 0;
  }

  return GEOARROW_OK;
}

static int geom_end_multipolygon(struct GeoArrowVisitor* v) {
  struct GeoArrowNativeWriter* writer = (struct GeoArrowNativeWriter*)v->private_data;
  struct GeoArrowNativeWriterPrivate* private_data =
      (struct GeoArrowNativeWriterPrivate*)writer->private_data;

  if (private_data->level == 2) {
    private_data->level--;
    if (private_data->size[2] > 0) {
      if (private_data->builder.view.coords.size_coords > 2147483647) {
        return EOVERFLOW;
      }
      int32_t n_coord32 = (int32_t)private_data->builder.view.coords.size_coords;
      NANOARROW_RETURN_NOT_OK(
          GeoArrowBuilderOffsetAppend(&private_data->builder, 2, &n_coord32, 1));
      private_data->size[1]++;
      private_data->size[2] = 0;
    }
  } else if (private_data->level == 1) {
    private_data->level--;
    if (private_data->size[1] > 0) {
      int32_t n_seq32 =
          (int32_t)(private_data->builder.view.buffers[3].size_bytes / sizeof(int32_t)) -
          1;
      NANOARROW_RETURN_NOT_OK(
          GeoArrowBuilderOffsetAppend(&private_data->builder, 1, &n_seq32, 1));
      private_data->size[0]++;
      private_data->size[1] = 0;
    }
  }

  return GEOARROW_OK;
}

static int null_feat_multipolygon(struct GeoArrowVisitor* v) {
  struct GeoArrowNativeWriter* writer = (struct GeoArrowNativeWriter*)v->private_data;
  struct GeoArrowNativeWriterPrivate* private_data =
      (struct GeoArrowNativeWriterPrivate*)writer->private_data;
  private_data->feat_is_null = 1;
  return GEOARROW_OK;
}

static int feat_end_multipolygon(struct GeoArrowVisitor* v) {
  struct GeoArrowNativeWriter* writer = (struct GeoArrowNativeWriter*)v->private_data;
  struct GeoArrowNativeWriterPrivate* private_data =
      (struct GeoArrowNativeWriterPrivate*)writer->private_data;

  // If we have an unfinished sequence left over, finish it now. This could have
  // occurred if the last geometry that was visited was a POINT.
  if (private_data->size[2] > 0) {
    if (private_data->builder.view.coords.size_coords > 2147483647) {
      return EOVERFLOW;
    }
    int32_t n_coord32 = (int32_t)private_data->builder.view.coords.size_coords;
    NANOARROW_RETURN_NOT_OK(
        GeoArrowBuilderOffsetAppend(&private_data->builder, 2, &n_coord32, 1));
    private_data->size[1]++;
  }

  // If we have an unfinished sequence of sequences left over, finish it now.
  // This could have occurred if the last geometry that was visited was a POINT.
  if (private_data->size[1] > 0) {
    int32_t n_seq32 =
        (int32_t)(private_data->builder.view.buffers[3].size_bytes / sizeof(int32_t)) - 1;
    NANOARROW_RETURN_NOT_OK(
        GeoArrowBuilderOffsetAppend(&private_data->builder, 1, &n_seq32, 1));
  }

  // Finish off the sequence of sequence of sequences. This is a multipolygon
  // so it can be any number of them.
  int32_t n_seq_seq32 =
      (int32_t)(private_data->builder.view.buffers[2].size_bytes / sizeof(int32_t)) - 1;
  NANOARROW_RETURN_NOT_OK(
      GeoArrowBuilderOffsetAppend(&private_data->builder, 0, &n_seq_seq32, 1));

  if (private_data->feat_is_null) {
    int64_t current_length =
        private_data->builder.view.buffers[1].size_bytes / sizeof(int32_t) - 1;
    if (private_data->validity.buffer.data == NULL) {
      NANOARROW_RETURN_NOT_OK(
          ArrowBitmapReserve(&private_data->validity, current_length));
      ArrowBitmapAppendUnsafe(&private_data->validity, 1, current_length - 1);
    }

    private_data->null_count++;
    NANOARROW_RETURN_NOT_OK(ArrowBitmapAppend(&private_data->validity, 0, 1));
  } else if (private_data->validity.buffer.data != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowBitmapAppend(&private_data->validity, 1, 1));
  }

  return GEOARROW_OK;
}

static void GeoArrowVisitorInitMultiPolygon(struct GeoArrowVisitor* v) {
  struct GeoArrowError* previous_error = v->error;
  GeoArrowVisitorInitVoid(v);
  v->error = previous_error;

  v->feat_start = &feat_start_multipolygon;
  v->null_feat = &null_feat_multipolygon;
  v->geom_start = &geom_start_multipolygon;
  v->ring_start = &ring_start_multipolygon;
  v->coords = &coords_multipolygon;
  v->ring_end = &ring_end_multipolygon;
  v->geom_end = &geom_end_multipolygon;
  v->feat_end = &feat_end_multipolygon;
}

GeoArrowErrorCode GeoArrowNativeWriterInitVisitor(struct GeoArrowNativeWriter* writer,
                                                  struct GeoArrowVisitor* v) {
  struct GeoArrowNativeWriterPrivate* private_data =
      (struct GeoArrowNativeWriterPrivate*)writer->private_data;

  switch (private_data->builder.view.schema_view.geometry_type) {
    case GEOARROW_GEOMETRY_TYPE_POINT:
      GeoArrowVisitorInitPoint(v);
      break;
    case GEOARROW_GEOMETRY_TYPE_MULTIPOINT:
    case GEOARROW_GEOMETRY_TYPE_LINESTRING:
      GeoArrowVisitorInitLinestring(v);
      break;
    case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING:
    case GEOARROW_GEOMETRY_TYPE_POLYGON:
      GeoArrowVisitorInitMultiLinestring(v);
      break;
    case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON:
      GeoArrowVisitorInitMultiPolygon(v);
      break;
    default:
      return EINVAL;
  }

  v->private_data = writer;
  return GEOARROW_OK;
}

#include "nanoarrow/nanoarrow.h"



static void GeoArrowScalarUdfVoidFactoryInit(struct GeoArrowScalarUdfFactory* self,
                                             struct GeoArrowScalarUdf* out);

static void GeoArrowScalarUdfVoidFactoryRelease(struct GeoArrowScalarUdfFactory* self) {
  GEOARROW_UNUSED(self);
}

GeoArrowErrorCode GeoArrowScalarUdfFactoryInit(struct GeoArrowScalarUdfFactory* out,
                                               const char* name, const char* options,
                                               struct GeoArrowError* error) {
  GEOARROW_UNUSED(out);
  GEOARROW_UNUSED(name);
  GEOARROW_UNUSED(options);

  if (strcmp(name, "void") == 0) {
    out->new_scalar_udf_impl = &GeoArrowScalarUdfVoidFactoryInit;
  } else {
    GeoArrowErrorSet(
        error, "GeoArrow C scalar implementation with name '%s' does not exist", name);
    return ENOTSUP;
  }

  out->private_data = NULL;
  out->release = &GeoArrowScalarUdfVoidFactoryRelease;
  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowScalarUdfVoidInit(struct GeoArrowScalarUdf* self,
                                                   const struct ArrowSchema** arg_types,
                                                   struct ArrowArray** scalar_args,
                                                   int64_t n_args,
                                                   struct ArrowSchema* out) {
  GEOARROW_UNUSED(self);
  GEOARROW_UNUSED(arg_types);
  GEOARROW_UNUSED(scalar_args);
  GEOARROW_UNUSED(n_args);

  GEOARROW_RETURN_NOT_OK(ArrowSchemaInitFromType(out, NANOARROW_TYPE_NA));
  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowScalarUdfVoidExecute(struct GeoArrowScalarUdf* self,
                                                      struct ArrowArray** args,
                                                      int64_t n_args, int64_t n_rows,
                                                      struct ArrowArray* out) {
  GEOARROW_UNUSED(self);
  GEOARROW_UNUSED(args);
  GEOARROW_UNUSED(n_args);

  GEOARROW_RETURN_NOT_OK(ArrowArrayInitFromType(out, NANOARROW_TYPE_NA));
  out->length = n_rows;
  out->null_count = n_rows;
  GEOARROW_RETURN_NOT_OK(ArrowArrayFinishBuildingDefault(out, NULL));

  return GEOARROW_OK;
}

static const char* GeoArrowScalarUdfVoidGetLastError(struct GeoArrowScalarUdf* self) {
  GEOARROW_UNUSED(self);
  return NULL;
}

static void GeoArrowScalarUdfVoidRelease(struct GeoArrowScalarUdf* self) {
  GEOARROW_UNUSED(self);
}

static void GeoArrowScalarUdfVoidFactoryInit(struct GeoArrowScalarUdfFactory* self,
                                             struct GeoArrowScalarUdf* out) {
  GEOARROW_UNUSED(self);
  out->init = &GeoArrowScalarUdfVoidInit;
  out->execute = &GeoArrowScalarUdfVoidExecute;
  out->get_last_error = &GeoArrowScalarUdfVoidGetLastError;
  out->release = &GeoArrowScalarUdfVoidRelease;
}

#include <errno.h>
#include <stdint.h>



#include "nanoarrow/nanoarrow.h"

#define EWKB_Z_BIT 0x80000000
#define EWKB_M_BIT 0x40000000
#define EWKB_SRID_BIT 0x20000000

struct WKBReaderPrivate {
  const uint8_t* data;
  int64_t size_bytes;
  const uint8_t* data0;
  int need_swapping;
  struct GeoArrowGeometry geom;
};

static inline int WKBReaderReadEndian(struct WKBReaderPrivate* s,
                                      struct GeoArrowError* error) {
  if (s->size_bytes > 0) {
    s->need_swapping = s->data[0] != GEOARROW_NATIVE_ENDIAN;
    s->data++;
    s->size_bytes--;
    return GEOARROW_OK;
  } else {
    GeoArrowErrorSet(error, "Expected endian byte but found end of buffer at byte %ld",
                     (long)(s->data - s->data0));
    return EINVAL;
  }
}

static inline int WKBReaderReadUInt32(struct WKBReaderPrivate* s, uint32_t* out,
                                      struct GeoArrowError* error) {
  if (s->size_bytes >= 4) {
    memcpy(out, s->data, sizeof(uint32_t));
    s->data += sizeof(uint32_t);
    s->size_bytes -= sizeof(uint32_t);
    if (s->need_swapping) {
      *out = GEOARROW_BSWAP32(*out);
    }
    return GEOARROW_OK;
  } else {
    GeoArrowErrorSet(error, "Expected uint32 but found end of buffer at byte %ld",
                     (long)(s->data - s->data0));
    return EINVAL;
  }
}

static inline GeoArrowErrorCode WKBReaderReadNodeCoordinates(
    struct WKBReaderPrivate* s, uint32_t n_coords, uint32_t coord_size_elements,
    struct GeoArrowGeometryNode* node, struct GeoArrowError* error) {
  int64_t bytes_needed = n_coords * coord_size_elements * sizeof(double);
  if (s->size_bytes < bytes_needed) {
    GeoArrowErrorSet(
        error,
        "Expected coordinate sequence of %ld coords (%ld bytes) but found %ld bytes "
        "remaining at byte %ld",
        (long)n_coords, (long)bytes_needed, (long)s->size_bytes,
        (long)(s->data - s->data0));
    return EINVAL;
  }

  if (n_coords > 0) {
    for (uint32_t i = 0; i < coord_size_elements; i++) {
      node->coord_stride[i] = (int32_t)coord_size_elements * sizeof(double);
      node->coords[i] = s->data + (i * sizeof(double));
    }
  }

  s->data += bytes_needed;
  return GEOARROW_OK;
}

static inline GeoArrowErrorCode WKBReaderReadNodeGeometry(
    struct WKBReaderPrivate* s, struct GeoArrowGeometryNode* node,
    struct GeoArrowError* error) {
  NANOARROW_RETURN_NOT_OK(WKBReaderReadEndian(s, error));
  uint32_t geometry_type;
  const uint8_t* data_at_geom_type = s->data;
  NANOARROW_RETURN_NOT_OK(WKBReaderReadUInt32(s, &geometry_type, error));

  int has_z = 0;
  int has_m = 0;

  // Handle EWKB high bits
  if (geometry_type & EWKB_Z_BIT) {
    has_z = 1;
  }

  if (geometry_type & EWKB_M_BIT) {
    has_m = 1;
  }

  if (geometry_type & EWKB_SRID_BIT) {
    // We ignore this because it's hard to work around if a user somehow
    // has embedded srid but still wants the data and doesn't have another way
    // to convert
    uint32_t embedded_srid;
    NANOARROW_RETURN_NOT_OK(WKBReaderReadUInt32(s, &embedded_srid, error));
  }

  geometry_type = geometry_type & 0x0000ffff;

  // Handle ISO X000 geometry types
  if (geometry_type >= 3000) {
    geometry_type = geometry_type - 3000;
    has_z = 1;
    has_m = 1;
  } else if (geometry_type >= 2000) {
    geometry_type = geometry_type - 2000;
    has_m = 1;
  } else if (geometry_type >= 1000) {
    geometry_type = geometry_type - 1000;
    has_z = 1;
  }

  // Read the number of coordinates/rings/parts
  uint32_t size;
  if (geometry_type != GEOARROW_GEOMETRY_TYPE_POINT) {
    NANOARROW_RETURN_NOT_OK(WKBReaderReadUInt32(s, &size, error));
  } else {
    size = 1;
  }

  // Set coord size
  uint32_t coord_size_elements = 2 + has_z + has_m;

  // Resolve dimensions
  enum GeoArrowDimensions dimensions;
  if (has_z && has_m) {
    dimensions = GEOARROW_DIMENSIONS_XYZM;
  } else if (has_z) {
    dimensions = GEOARROW_DIMENSIONS_XYZ;
  } else if (has_m) {
    dimensions = GEOARROW_DIMENSIONS_XYM;
  } else {
    dimensions = GEOARROW_DIMENSIONS_XY;
  }

  // Populate the node
  node->geometry_type = (uint8_t)geometry_type;
  node->dimensions = (uint8_t)dimensions;
  node->size = size;
  if (s->need_swapping) {
    node->flags = GEOARROW_GEOMETRY_NODE_FLAG_SWAP_ENDIAN;
  }

  switch (geometry_type) {
    case GEOARROW_GEOMETRY_TYPE_POINT:
    case GEOARROW_GEOMETRY_TYPE_LINESTRING:
      NANOARROW_RETURN_NOT_OK(
          WKBReaderReadNodeCoordinates(s, size, coord_size_elements, node, error));
      break;
    case GEOARROW_GEOMETRY_TYPE_POLYGON:
      if (node->level == 255) {
        GeoArrowErrorSet(error, "WKBReader exceeded maximum recursion");
        return ENOTSUP;
      }

      struct GeoArrowGeometryNode ring_template = *node;
      ring_template.geometry_type = (uint8_t)GEOARROW_GEOMETRY_TYPE_LINESTRING;
      ring_template.level++;

      struct GeoArrowGeometryNode* ring;
      uint32_t ring_size;
      for (uint32_t i = 0; i < size; i++) {
        GEOARROW_RETURN_NOT_OK(WKBReaderReadUInt32(s, &ring_size, error));
        GEOARROW_RETURN_NOT_OK(GeoArrowGeometryAppendNodeInline(&s->geom, &ring));
        *ring = ring_template;
        ring->size = ring_size;

        GEOARROW_RETURN_NOT_OK(
            WKBReaderReadNodeCoordinates(s, ring_size, coord_size_elements, ring, error));
      }
      break;
    case GEOARROW_GEOMETRY_TYPE_MULTIPOINT:
    case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING:
    case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON:
    case GEOARROW_GEOMETRY_TYPE_GEOMETRYCOLLECTION:
      if (node->level == 255) {
        GeoArrowErrorSet(error, "WKBReader exceeded maximum recursion");
        return ENOTSUP;
      }

      uint8_t child_level = node->level + 1;
      struct GeoArrowGeometryNode* child;
      for (uint32_t i = 0; i < size; i++) {
        GEOARROW_RETURN_NOT_OK(GeoArrowGeometryAppendNodeInline(&s->geom, &child));
        child->level = child_level;
        GEOARROW_RETURN_NOT_OK(WKBReaderReadNodeGeometry(s, child, error));
      }
      break;
    default:
      GeoArrowErrorSet(error,
                       "Expected valid geometry type code but found %u at byte %ld",
                       (unsigned int)geometry_type, (long)(data_at_geom_type - s->data0));
      return EINVAL;
  }

  return GEOARROW_OK;
}

GeoArrowErrorCode GeoArrowWKBReaderInit(struct GeoArrowWKBReader* reader) {
  struct WKBReaderPrivate* s =
      (struct WKBReaderPrivate*)ArrowMalloc(sizeof(struct WKBReaderPrivate));

  if (s == NULL) {
    return ENOMEM;
  }

  s->data0 = NULL;
  s->data = NULL;
  s->size_bytes = 0;
  s->need_swapping = 0;

  GeoArrowErrorCode result = GeoArrowGeometryInit(&s->geom);
  if (result != GEOARROW_OK) {
    ArrowFree(s);
    return result;
  }

  reader->private_data = s;
  return GEOARROW_OK;
}

void GeoArrowWKBReaderReset(struct GeoArrowWKBReader* reader) {
  struct WKBReaderPrivate* s = (struct WKBReaderPrivate*)reader->private_data;
  GeoArrowGeometryReset(&s->geom);
  ArrowFree(reader->private_data);
}

GeoArrowErrorCode GeoArrowWKBReaderVisit(struct GeoArrowWKBReader* reader,
                                         struct GeoArrowBufferView src,
                                         struct GeoArrowVisitor* v) {
  struct GeoArrowGeometryView geometry;
  GEOARROW_RETURN_NOT_OK(GeoArrowWKBReaderRead(reader, src, &geometry, v->error));
  GEOARROW_RETURN_NOT_OK(GeoArrowGeometryViewVisit(geometry, v));
  return GEOARROW_OK;
}

GeoArrowErrorCode GeoArrowWKBReaderRead(struct GeoArrowWKBReader* reader,
                                        struct GeoArrowBufferView src,
                                        struct GeoArrowGeometryView* out,
                                        struct GeoArrowError* error) {
  struct WKBReaderPrivate* s = (struct WKBReaderPrivate*)reader->private_data;
  s->data0 = src.data;
  s->data = src.data;
  s->size_bytes = src.size_bytes;

  // Reset the nodes list
  GEOARROW_RETURN_NOT_OK(GeoArrowGeometryResizeNodesInline(&s->geom, 0));

  // Make a root node at level 0
  struct GeoArrowGeometryNode* node;
  GEOARROW_RETURN_NOT_OK(GeoArrowGeometryAppendNodeInline(&s->geom, &node));
  node->level = 0;

  // Read
  GEOARROW_RETURN_NOT_OK(WKBReaderReadNodeGeometry(s, node, error));

  // Populate output on success
  *out = GeoArrowGeometryAsView(&s->geom);
  return GEOARROW_OK;
}

#include <limits.h>
#include <string.h>

#include "nanoarrow/nanoarrow.h"



struct WKBWriterPrivate {
  enum ArrowType storage_type;
  struct ArrowBitmap validity;
  struct ArrowBuffer offsets;
  struct ArrowBuffer values;
  enum GeoArrowGeometryType geometry_type[32];
  enum GeoArrowDimensions dimensions[32];
  int64_t size_pos[32];
  uint32_t size[32];
  int32_t level;
  int64_t length;
  int64_t null_count;
  int feat_is_null;
};

static uint8_t kWKBWriterEmptyPointCoords2[] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                                0xf8, 0x7f, 0x00, 0x00, 0x00, 0x00,
                                                0x00, 0x00, 0xf8, 0x7f};
static uint8_t kWKBWriterEmptyPointCoords3[] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0xf8, 0x7f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f};
static uint8_t kWKBWriterEmptyPointCoords4[] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0xf8, 0x7f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0xf8, 0x7f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f};

static inline int WKBWriterCheckLevel(struct WKBWriterPrivate* private) {
  if (private->level >= 0 && private->level <= 30) {
    return GEOARROW_OK;
  } else {
    return EINVAL;
  }
}

static int feat_start_wkb(struct GeoArrowVisitor* v) {
  struct WKBWriterPrivate* private = (struct WKBWriterPrivate*)v->private_data;
  private->level = 0;
  private->size[private->level] = 0;
  private->length++;
  private->feat_is_null = 0;

  if (private->values.size_bytes > 2147483647) {
    return EOVERFLOW;
  }
  return ArrowBufferAppendInt32(&private->offsets, (int32_t) private->values.size_bytes);
}

static int null_feat_wkb(struct GeoArrowVisitor* v) {
  struct WKBWriterPrivate* private = (struct WKBWriterPrivate*)v->private_data;
  private->feat_is_null = 1;
  return GEOARROW_OK;
}

static int geom_start_wkb(struct GeoArrowVisitor* v,
                          enum GeoArrowGeometryType geometry_type,
                          enum GeoArrowDimensions dimensions) {
  struct WKBWriterPrivate* private = (struct WKBWriterPrivate*)v->private_data;
  NANOARROW_RETURN_NOT_OK(WKBWriterCheckLevel(private));
  private->size[private->level]++;
  private->level++;
  private->geometry_type[private->level] = geometry_type;
  private->dimensions[private->level] = dimensions;
  private->size[private->level] = 0;

  NANOARROW_RETURN_NOT_OK(
      ArrowBufferAppendUInt8(&private->values, GEOARROW_NATIVE_ENDIAN));
  NANOARROW_RETURN_NOT_OK(ArrowBufferAppendUInt32(
      &private->values, geometry_type + ((dimensions - 1) * 1000)));
  if (geometry_type != GEOARROW_GEOMETRY_TYPE_POINT) {
    private->size_pos[private->level] = private->values.size_bytes;
    NANOARROW_RETURN_NOT_OK(ArrowBufferAppendUInt32(&private->values, 0));
  }

  return GEOARROW_OK;
}

static int ring_start_wkb(struct GeoArrowVisitor* v) {
  struct WKBWriterPrivate* private = (struct WKBWriterPrivate*)v->private_data;
  NANOARROW_RETURN_NOT_OK(WKBWriterCheckLevel(private));
  private->size[private->level]++;
  private->level++;
  private->geometry_type[private->level] = GEOARROW_GEOMETRY_TYPE_GEOMETRY;
  private->size_pos[private->level] = private->values.size_bytes;
  private->size[private->level] = 0;
  return ArrowBufferAppendUInt32(&private->values, 0);
}

static int coords_wkb(struct GeoArrowVisitor* v, const struct GeoArrowCoordView* coords) {
  struct WKBWriterPrivate* private = (struct WKBWriterPrivate*)v->private_data;
  NANOARROW_RETURN_NOT_OK(WKBWriterCheckLevel(private));

  NANOARROW_DCHECK(coords->n_coords <= UINT32_MAX);
  private->size[private->level] += (uint32_t)coords->n_coords;

  NANOARROW_RETURN_NOT_OK(ArrowBufferReserve(
      &private->values, coords->n_values * coords->n_coords * sizeof(double)));
  for (int64_t i = 0; i < coords->n_coords; i++) {
    for (int32_t j = 0; j < coords->n_values; j++) {
      ArrowBufferAppendUnsafe(&private->values,
                              coords->values[j] + i * coords->coords_stride,
                              sizeof(double));
    }
  }

  return GEOARROW_OK;
}

static int ring_end_wkb(struct GeoArrowVisitor* v) {
  struct WKBWriterPrivate* private = (struct WKBWriterPrivate*)v->private_data;
  NANOARROW_RETURN_NOT_OK(WKBWriterCheckLevel(private));
  if (private->values.data == NULL) {
    return EINVAL;
  }
  memcpy(private->values.data + private->size_pos[private->level],
         private->size + private->level, sizeof(uint32_t));
  private->level--;
  return GEOARROW_OK;
}

static int geom_end_wkb(struct GeoArrowVisitor* v) {
  struct WKBWriterPrivate* private = (struct WKBWriterPrivate*)v->private_data;
  NANOARROW_RETURN_NOT_OK(WKBWriterCheckLevel(private));
  if (private->values.data == NULL) {
    return EINVAL;
  }

  if (private->geometry_type[private->level] != GEOARROW_GEOMETRY_TYPE_POINT) {
    memcpy(private->values.data + private->size_pos[private->level],
           private->size + private->level, sizeof(uint32_t));
  } else if (private->size[private->level] == 0) {
    switch (private->dimensions[private->level]) {
      case GEOARROW_DIMENSIONS_XY:
        NANOARROW_RETURN_NOT_OK(ArrowBufferAppend(&private->values,
                                                  kWKBWriterEmptyPointCoords2,
                                                  sizeof(kWKBWriterEmptyPointCoords2)));
        break;
      case GEOARROW_DIMENSIONS_XYZ:
      case GEOARROW_DIMENSIONS_XYM:
        NANOARROW_RETURN_NOT_OK(ArrowBufferAppend(&private->values,
                                                  kWKBWriterEmptyPointCoords3,
                                                  sizeof(kWKBWriterEmptyPointCoords3)));
        break;
      case GEOARROW_DIMENSIONS_XYZM:
        NANOARROW_RETURN_NOT_OK(ArrowBufferAppend(&private->values,
                                                  kWKBWriterEmptyPointCoords4,
                                                  sizeof(kWKBWriterEmptyPointCoords4)));
        break;
      default:
        return EINVAL;
    }
  }

  private->level--;
  return GEOARROW_OK;
}

static int feat_end_wkb(struct GeoArrowVisitor* v) {
  struct WKBWriterPrivate* private = (struct WKBWriterPrivate*)v->private_data;

  if (private->feat_is_null) {
    if (private->validity.buffer.data == NULL) {
      NANOARROW_RETURN_NOT_OK(ArrowBitmapReserve(&private->validity, private->length));
      ArrowBitmapAppendUnsafe(&private->validity, 1, private->length - 1);
    }

    private->null_count++;
    return ArrowBitmapAppend(&private->validity, 0, 1);
  } else if (private->validity.buffer.data != NULL) {
    return ArrowBitmapAppend(&private->validity, 1, 1);
  }

  return GEOARROW_OK;
}

GeoArrowErrorCode GeoArrowWKBWriterInit(struct GeoArrowWKBWriter* writer) {
  struct WKBWriterPrivate* private =
      (struct WKBWriterPrivate*)ArrowMalloc(sizeof(struct WKBWriterPrivate));
  if (private == NULL) {
    return ENOMEM;
  }

  private->storage_type = NANOARROW_TYPE_BINARY;
  private->length = 0;
  private->level = 0;
  private->null_count = 0;
  ArrowBitmapInit(&private->validity);
  ArrowBufferInit(&private->offsets);
  ArrowBufferInit(&private->values);
  writer->private_data = private;

  return GEOARROW_OK;
}

void GeoArrowWKBWriterInitVisitor(struct GeoArrowWKBWriter* writer,
                                  struct GeoArrowVisitor* v) {
  GeoArrowVisitorInitVoid(v);

  v->private_data = writer->private_data;
  v->feat_start = &feat_start_wkb;
  v->null_feat = &null_feat_wkb;
  v->geom_start = &geom_start_wkb;
  v->ring_start = &ring_start_wkb;
  v->coords = &coords_wkb;
  v->ring_end = &ring_end_wkb;
  v->geom_end = &geom_end_wkb;
  v->feat_end = &feat_end_wkb;
}

GeoArrowErrorCode GeoArrowWKBWriterFinish(struct GeoArrowWKBWriter* writer,
                                          struct ArrowArray* array,
                                          struct GeoArrowError* error) {
  struct WKBWriterPrivate* private = (struct WKBWriterPrivate*)writer->private_data;
  array->release = NULL;

  if (private->values.size_bytes > 2147483647) {
    return EOVERFLOW;
  }

  NANOARROW_RETURN_NOT_OK(
      ArrowBufferAppendInt32(&private->offsets, (int32_t) private->values.size_bytes));
  NANOARROW_RETURN_NOT_OK(ArrowArrayInitFromType(array, private->storage_type));
  ArrowArraySetValidityBitmap(array, &private->validity);
  NANOARROW_RETURN_NOT_OK(ArrowArraySetBuffer(array, 1, &private->offsets));
  NANOARROW_RETURN_NOT_OK(ArrowArraySetBuffer(array, 2, &private->values));
  array->length = private->length;
  array->null_count = private->null_count;
  private->length = 0;
  private->null_count = 0;
  return ArrowArrayFinishBuildingDefault(array, (struct ArrowError*)error);
}

void GeoArrowWKBWriterReset(struct GeoArrowWKBWriter* writer) {
  struct WKBWriterPrivate* private = (struct WKBWriterPrivate*)writer->private_data;
  ArrowBitmapReset(&private->validity);
  ArrowBufferReset(&private->offsets);
  ArrowBufferReset(&private->values);
  ArrowFree(private);
  writer->private_data = NULL;
}

static GeoArrowErrorCode GeoArrowWKBWriterAppendValidity(struct GeoArrowWKBWriter* writer,
                                                         int is_valid) {
  struct WKBWriterPrivate* private_data = (struct WKBWriterPrivate*)writer->private_data;

  private_data->length++;

  if (!is_valid) {
    if (private_data->validity.buffer.data == NULL) {
      NANOARROW_RETURN_NOT_OK(
          ArrowBitmapReserve(&private_data->validity, private_data->length));
      ArrowBitmapAppendUnsafe(&private_data->validity, 1, private_data->length - 1);
    }

    private_data->null_count++;
    NANOARROW_RETURN_NOT_OK(ArrowBitmapAppend(&private_data->validity, 0, 1));
  } else if (private_data->validity.buffer.data != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowBitmapAppend(&private_data->validity, 1, 1));
  }

  return GEOARROW_OK;
}

GeoArrowErrorCode GeoArrowWKBWriterAppendNull(struct GeoArrowWKBWriter* writer) {
  struct WKBWriterPrivate* private_data = (struct WKBWriterPrivate*)writer->private_data;
  GEOARROW_RETURN_NOT_OK(ArrowBufferAppendInt32(
      &private_data->offsets, (int32_t)private_data->values.size_bytes));
  GEOARROW_RETURN_NOT_OK(GeoArrowWKBWriterAppendValidity(writer, 0));
  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowWKBWriterAppendSequence(
    struct ArrowBuffer* values, const struct GeoArrowGeometryNode* node) {
  uint8_t* dst = values->data + values->size_bytes;
  int64_t dst_size = values->capacity_bytes - values->size_bytes;
  int64_t bytes_required = GeoArrowGeometryNodeWriteSequence(node, dst, dst_size);
  if (bytes_required <= dst_size) {
    values->size_bytes += bytes_required;
    return GEOARROW_OK;
  }

  NANOARROW_RETURN_NOT_OK(ArrowBufferReserve(values, bytes_required));
  dst_size = values->capacity_bytes - values->size_bytes;
  NANOARROW_DCHECK(dst_size >= bytes_required);
  return GeoArrowWKBWriterAppendSequence(values, node);
}

static GeoArrowErrorCode GeoArrowWKBWriterAppendBuffer(struct ArrowBuffer* values,
                                                       struct GeoArrowGeometryView geom) {
  uint32_t remaining_rings = 0;

  const struct GeoArrowGeometryNode* end = geom.root + geom.size_nodes;
  for (const struct GeoArrowGeometryNode* node = geom.root; node < end; ++node) {
    if (remaining_rings > 0) {
      GEOARROW_RETURN_NOT_OK(ArrowBufferAppendUInt32(values, node->size));
      GEOARROW_RETURN_NOT_OK(GeoArrowWKBWriterAppendSequence(values, node));
      --remaining_rings;
      continue;
    }

    uint32_t iso_geometry_type = node->geometry_type + ((node->dimensions - 1) * 1000);

    switch (node->geometry_type) {
      case GEOARROW_GEOMETRY_TYPE_POINT: {
        GEOARROW_RETURN_NOT_OK(ArrowBufferAppendUInt8(values, GEOARROW_NATIVE_ENDIAN));
        GEOARROW_RETURN_NOT_OK(ArrowBufferAppendUInt32(values, iso_geometry_type));
        if (node->size == 0) {
          uint32_t n_values = _GeoArrowkNumDimensions[node->dimensions];
          GEOARROW_RETURN_NOT_OK(ArrowBufferAppend(values, _GeoArrowkEmptyPointCoords,
                                                   n_values * sizeof(double)));
        } else {
          GEOARROW_RETURN_NOT_OK(GeoArrowWKBWriterAppendSequence(values, node));
        }

        break;
      }
      case GEOARROW_GEOMETRY_TYPE_LINESTRING: {
        GEOARROW_RETURN_NOT_OK(ArrowBufferAppendUInt8(values, GEOARROW_NATIVE_ENDIAN));
        GEOARROW_RETURN_NOT_OK(ArrowBufferAppendUInt32(values, iso_geometry_type));
        GEOARROW_RETURN_NOT_OK(ArrowBufferAppendUInt32(values, node->size));
        GEOARROW_RETURN_NOT_OK(GeoArrowWKBWriterAppendSequence(values, node));
        break;
      }
      case GEOARROW_GEOMETRY_TYPE_POLYGON: {
        GEOARROW_RETURN_NOT_OK(ArrowBufferAppendUInt8(values, GEOARROW_NATIVE_ENDIAN));
        GEOARROW_RETURN_NOT_OK(ArrowBufferAppendUInt32(values, iso_geometry_type));
        GEOARROW_RETURN_NOT_OK(ArrowBufferAppendUInt32(values, node->size));
        remaining_rings = node->size;
        break;
      }
      case GEOARROW_GEOMETRY_TYPE_MULTIPOINT:
      case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING:
      case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON:
      case GEOARROW_GEOMETRY_TYPE_GEOMETRYCOLLECTION:
        GEOARROW_RETURN_NOT_OK(ArrowBufferAppendUInt8(values, GEOARROW_NATIVE_ENDIAN));
        GEOARROW_RETURN_NOT_OK(ArrowBufferAppendUInt32(values, iso_geometry_type));
        GEOARROW_RETURN_NOT_OK(ArrowBufferAppendUInt32(values, node->size));
        break;
      default:
        return EINVAL;
    }
  }

  return GEOARROW_OK;
}

GeoArrowErrorCode GeoArrowWKBWriterAppend(struct GeoArrowWKBWriter* writer,
                                          struct GeoArrowGeometryView geom) {
  struct WKBWriterPrivate* private_data = (struct WKBWriterPrivate*)writer->private_data;
  GEOARROW_RETURN_NOT_OK(ArrowBufferAppendInt32(
      &private_data->offsets, (int32_t)private_data->values.size_bytes));
  GEOARROW_RETURN_NOT_OK(GeoArrowWKBWriterAppendBuffer(&private_data->values, geom));
  GEOARROW_RETURN_NOT_OK(GeoArrowWKBWriterAppendValidity(writer, 1));
  return GEOARROW_OK;
}
#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include "nanoarrow/nanoarrow.h"



#define COORD_CACHE_SIZE_COORDS 64

struct WKTReaderPrivate {
  const char* data;
  int64_t size_bytes;
  const char* data0;
  double coords[4 * COORD_CACHE_SIZE_COORDS];
  struct GeoArrowCoordView coord_view;
};

static inline void GeoArrowWKTAdvanceUnsafe(struct WKTReaderPrivate* s, int64_t n) {
  s->data += n;
  s->size_bytes -= n;
}

static inline void GeoArrowWKTSkipWhitespace(struct WKTReaderPrivate* s) {
  while (s->size_bytes > 0) {
    char c = *(s->data);
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
      s->size_bytes--;
      s->data++;
    } else {
      break;
    }
  }
}

static inline int GeoArrowWKTSkipUntil(struct WKTReaderPrivate* s, const char* items) {
  int64_t n_items = strlen(items);
  while (s->size_bytes > 0) {
    char c = *(s->data);
    if (c == '\0') {
      return 0;
    }

    for (int64_t i = 0; i < n_items; i++) {
      if (c == items[i]) {
        return 1;
      }
    }

    s->size_bytes--;
    s->data++;
  }

  return 0;
}

static inline void GeoArrowWKTSkipUntilSep(struct WKTReaderPrivate* s) {
  GeoArrowWKTSkipUntil(s, " \n\t\r,()");
}

static inline char PeekChar(struct WKTReaderPrivate* s) {
  if (s->size_bytes > 0) {
    return s->data[0];
  } else {
    return '\0';
  }
}

static inline struct ArrowStringView GeoArrowWKTPeekUntilSep(struct WKTReaderPrivate* s,
                                                             int max_chars) {
  struct WKTReaderPrivate tmp = *s;
  if (tmp.size_bytes > max_chars) {
    tmp.size_bytes = max_chars;
  }

  GeoArrowWKTSkipUntilSep(&tmp);
  struct ArrowStringView out = {s->data, tmp.data - s->data};
  return out;
}

static inline void SetParseErrorAuto(const char* expected, struct WKTReaderPrivate* s,
                                     struct GeoArrowError* error) {
  long pos = s->data - s->data0;
  // TODO: "but found ..." from s
  GeoArrowErrorSet(error, "Expected %s at byte %ld", expected, pos);
}

static inline int GeoArrowWKTAssertChar(struct WKTReaderPrivate* s, char c,
                                        struct GeoArrowError* error) {
  if (s->size_bytes > 0 && s->data[0] == c) {
    GeoArrowWKTAdvanceUnsafe(s, 1);
    return GEOARROW_OK;
  } else {
    char expected[4] = {'\'', c, '\'', '\0'};
    SetParseErrorAuto(expected, s, error);
    return EINVAL;
  }
}

static inline int GeoArrowWKTAssertWhitespace(struct WKTReaderPrivate* s,
                                              struct GeoArrowError* error) {
  if (s->size_bytes > 0 && (s->data[0] == ' ' || s->data[0] == '\t' ||
                            s->data[0] == '\r' || s->data[0] == '\n')) {
    GeoArrowWKTSkipWhitespace(s);
    return GEOARROW_OK;
  } else {
    SetParseErrorAuto("whitespace", s, error);
    return EINVAL;
  }
}

static inline int GeoArrowWKTAssertWordEmpty(struct WKTReaderPrivate* s,
                                             struct GeoArrowError* error) {
  struct ArrowStringView word = GeoArrowWKTPeekUntilSep(s, 6);
  if (word.size_bytes == 5 && strncmp(word.data, "EMPTY", 5) == 0) {
    GeoArrowWKTAdvanceUnsafe(s, 5);
    return GEOARROW_OK;
  }

  SetParseErrorAuto("'(' or 'EMPTY'", s, error);
  return EINVAL;
}

static inline int ReadOrdinate(struct WKTReaderPrivate* s, double* out,
                               struct GeoArrowError* error) {
  const char* start = s->data;
  GeoArrowWKTSkipUntilSep(s);
  int result = GeoArrowFromChars(start, s->data, out);
  if (result != GEOARROW_OK) {
    s->size_bytes += s->data - start;
    s->data = start;
    SetParseErrorAuto("number", s, error);
  }

  return result;
}

static inline void ResetCoordCache(struct WKTReaderPrivate* s) {
  s->coord_view.n_coords = 0;
}

static inline int FlushCoordCache(struct WKTReaderPrivate* s, struct GeoArrowVisitor* v) {
  if (s->coord_view.n_coords > 0) {
    int result = v->coords(v, &s->coord_view);
    s->coord_view.n_coords = 0;
    return result;
  } else {
    return GEOARROW_OK;
  }
}

static inline int ReadCoordinate(struct WKTReaderPrivate* s, struct GeoArrowVisitor* v) {
  if (s->coord_view.n_coords == COORD_CACHE_SIZE_COORDS) {
    NANOARROW_RETURN_NOT_OK(FlushCoordCache(s, v));
  }

  NANOARROW_RETURN_NOT_OK(ReadOrdinate(
      s, (double*)s->coord_view.values[0] + s->coord_view.n_coords, v->error));
  for (int i = 1; i < s->coord_view.n_values; i++) {
    NANOARROW_RETURN_NOT_OK(GeoArrowWKTAssertWhitespace(s, v->error));
    NANOARROW_RETURN_NOT_OK(ReadOrdinate(
        s, (double*)s->coord_view.values[i] + s->coord_view.n_coords, v->error));
  }

  s->coord_view.n_coords++;
  return NANOARROW_OK;
}

static inline int ReadEmptyOrCoordinates(struct WKTReaderPrivate* s,
                                         struct GeoArrowVisitor* v) {
  GeoArrowWKTSkipWhitespace(s);
  if (PeekChar(s) == '(') {
    GeoArrowWKTAdvanceUnsafe(s, 1);
    GeoArrowWKTSkipWhitespace(s);

    ResetCoordCache(s);

    // Read the first coordinate (there must always be one)
    NANOARROW_RETURN_NOT_OK(ReadCoordinate(s, v));
    GeoArrowWKTSkipWhitespace(s);

    // Read the rest of the coordinates
    while (PeekChar(s) != ')') {
      GeoArrowWKTSkipWhitespace(s);
      NANOARROW_RETURN_NOT_OK(GeoArrowWKTAssertChar(s, ',', v->error));
      GeoArrowWKTSkipWhitespace(s);
      NANOARROW_RETURN_NOT_OK(ReadCoordinate(s, v));
      GeoArrowWKTSkipWhitespace(s);
    }

    NANOARROW_RETURN_NOT_OK(FlushCoordCache(s, v));

    GeoArrowWKTAdvanceUnsafe(s, 1);
    return GEOARROW_OK;
  }

  return GeoArrowWKTAssertWordEmpty(s, v->error);
}

static inline int ReadMultipointFlat(struct WKTReaderPrivate* s,
                                     struct GeoArrowVisitor* v,
                                     enum GeoArrowDimensions dimensions) {
  NANOARROW_RETURN_NOT_OK(GeoArrowWKTAssertChar(s, '(', v->error));

  // Read the first coordinate (there must always be one)
  NANOARROW_RETURN_NOT_OK(v->geom_start(v, GEOARROW_GEOMETRY_TYPE_POINT, dimensions));
  ResetCoordCache(s);
  NANOARROW_RETURN_NOT_OK(ReadCoordinate(s, v));
  NANOARROW_RETURN_NOT_OK(FlushCoordCache(s, v));
  NANOARROW_RETURN_NOT_OK(v->geom_end(v));
  GeoArrowWKTSkipWhitespace(s);

  // Read the rest of the coordinates
  while (PeekChar(s) != ')') {
    GeoArrowWKTSkipWhitespace(s);
    NANOARROW_RETURN_NOT_OK(GeoArrowWKTAssertChar(s, ',', v->error));
    GeoArrowWKTSkipWhitespace(s);
    NANOARROW_RETURN_NOT_OK(v->geom_start(v, GEOARROW_GEOMETRY_TYPE_POINT, dimensions));
    ResetCoordCache(s);
    NANOARROW_RETURN_NOT_OK(ReadCoordinate(s, v));
    NANOARROW_RETURN_NOT_OK(FlushCoordCache(s, v));
    NANOARROW_RETURN_NOT_OK(v->geom_end(v));
    GeoArrowWKTSkipWhitespace(s);
  }

  GeoArrowWKTAdvanceUnsafe(s, 1);
  return GEOARROW_OK;
}

static inline int ReadEmptyOrPointCoordinate(struct WKTReaderPrivate* s,
                                             struct GeoArrowVisitor* v) {
  GeoArrowWKTSkipWhitespace(s);
  if (PeekChar(s) == '(') {
    GeoArrowWKTAdvanceUnsafe(s, 1);

    GeoArrowWKTSkipWhitespace(s);
    ResetCoordCache(s);
    NANOARROW_RETURN_NOT_OK(ReadCoordinate(s, v));
    NANOARROW_RETURN_NOT_OK(FlushCoordCache(s, v));
    GeoArrowWKTSkipWhitespace(s);
    NANOARROW_RETURN_NOT_OK(GeoArrowWKTAssertChar(s, ')', v->error));
    return GEOARROW_OK;
  }

  return GeoArrowWKTAssertWordEmpty(s, v->error);
}

static inline int ReadPolygon(struct WKTReaderPrivate* s, struct GeoArrowVisitor* v) {
  GeoArrowWKTSkipWhitespace(s);
  if (PeekChar(s) == '(') {
    GeoArrowWKTAdvanceUnsafe(s, 1);
    GeoArrowWKTSkipWhitespace(s);

    // Read the first ring (there must always be one)
    NANOARROW_RETURN_NOT_OK(v->ring_start(v));
    NANOARROW_RETURN_NOT_OK(ReadEmptyOrCoordinates(s, v));
    NANOARROW_RETURN_NOT_OK(v->ring_end(v));
    GeoArrowWKTSkipWhitespace(s);

    // Read the rest of the rings
    while (PeekChar(s) != ')') {
      GeoArrowWKTSkipWhitespace(s);
      NANOARROW_RETURN_NOT_OK(GeoArrowWKTAssertChar(s, ',', v->error));
      GeoArrowWKTSkipWhitespace(s);
      NANOARROW_RETURN_NOT_OK(v->ring_start(v));
      NANOARROW_RETURN_NOT_OK(ReadEmptyOrCoordinates(s, v));
      NANOARROW_RETURN_NOT_OK(v->ring_end(v));
      GeoArrowWKTSkipWhitespace(s);
    }

    GeoArrowWKTAdvanceUnsafe(s, 1);
    return GEOARROW_OK;
  }

  return GeoArrowWKTAssertWordEmpty(s, v->error);
}

static inline int ReadMultipoint(struct WKTReaderPrivate* s, struct GeoArrowVisitor* v,
                                 enum GeoArrowDimensions dimensions) {
  GeoArrowWKTSkipWhitespace(s);
  if (PeekChar(s) == '(') {
    GeoArrowWKTAdvanceUnsafe(s, 1);
    GeoArrowWKTSkipWhitespace(s);

    // Both MULTIPOINT (1 2, 2 3) and MULTIPOINT ((1 2), (2 3)) have to parse here
    // if it doesn't look like the verbose version, try the flat version
    if (PeekChar(s) != '(' && PeekChar(s) != 'E') {
      s->data--;
      s->size_bytes++;
      return ReadMultipointFlat(s, v, dimensions);
    }

    // Read the first geometry (there must always be one)
    NANOARROW_RETURN_NOT_OK(v->geom_start(v, GEOARROW_GEOMETRY_TYPE_POINT, dimensions));
    NANOARROW_RETURN_NOT_OK(ReadEmptyOrPointCoordinate(s, v));
    NANOARROW_RETURN_NOT_OK(v->geom_end(v));
    GeoArrowWKTSkipWhitespace(s);

    // Read the rest of the geometries
    while (PeekChar(s) != ')') {
      GeoArrowWKTSkipWhitespace(s);
      NANOARROW_RETURN_NOT_OK(GeoArrowWKTAssertChar(s, ',', v->error));
      GeoArrowWKTSkipWhitespace(s);
      NANOARROW_RETURN_NOT_OK(v->geom_start(v, GEOARROW_GEOMETRY_TYPE_POINT, dimensions));
      NANOARROW_RETURN_NOT_OK(ReadEmptyOrPointCoordinate(s, v));
      NANOARROW_RETURN_NOT_OK(v->geom_end(v));
      GeoArrowWKTSkipWhitespace(s);
    }

    GeoArrowWKTAdvanceUnsafe(s, 1);
    return GEOARROW_OK;
  }

  return GeoArrowWKTAssertWordEmpty(s, v->error);
}

static inline int ReadMultilinestring(struct WKTReaderPrivate* s,
                                      struct GeoArrowVisitor* v,
                                      enum GeoArrowDimensions dimensions) {
  GeoArrowWKTSkipWhitespace(s);
  if (PeekChar(s) == '(') {
    GeoArrowWKTAdvanceUnsafe(s, 1);
    GeoArrowWKTSkipWhitespace(s);

    // Read the first geometry (there must always be one)
    NANOARROW_RETURN_NOT_OK(
        v->geom_start(v, GEOARROW_GEOMETRY_TYPE_LINESTRING, dimensions));
    NANOARROW_RETURN_NOT_OK(ReadEmptyOrCoordinates(s, v));
    NANOARROW_RETURN_NOT_OK(v->geom_end(v));
    GeoArrowWKTSkipWhitespace(s);

    // Read the rest of the geometries
    while (PeekChar(s) != ')') {
      GeoArrowWKTSkipWhitespace(s);
      NANOARROW_RETURN_NOT_OK(GeoArrowWKTAssertChar(s, ',', v->error));
      GeoArrowWKTSkipWhitespace(s);
      NANOARROW_RETURN_NOT_OK(
          v->geom_start(v, GEOARROW_GEOMETRY_TYPE_LINESTRING, dimensions));
      NANOARROW_RETURN_NOT_OK(ReadEmptyOrCoordinates(s, v));
      NANOARROW_RETURN_NOT_OK(v->geom_end(v));
      GeoArrowWKTSkipWhitespace(s);
    }

    GeoArrowWKTAdvanceUnsafe(s, 1);
    return GEOARROW_OK;
  }

  return GeoArrowWKTAssertWordEmpty(s, v->error);
}

static inline int ReadMultipolygon(struct WKTReaderPrivate* s, struct GeoArrowVisitor* v,
                                   enum GeoArrowDimensions dimensions) {
  GeoArrowWKTSkipWhitespace(s);
  if (PeekChar(s) == '(') {
    GeoArrowWKTAdvanceUnsafe(s, 1);
    GeoArrowWKTSkipWhitespace(s);

    // Read the first geometry (there must always be one)
    NANOARROW_RETURN_NOT_OK(v->geom_start(v, GEOARROW_GEOMETRY_TYPE_POLYGON, dimensions));
    NANOARROW_RETURN_NOT_OK(ReadPolygon(s, v));
    NANOARROW_RETURN_NOT_OK(v->geom_end(v));
    GeoArrowWKTSkipWhitespace(s);

    // Read the rest of the geometries
    while (PeekChar(s) != ')') {
      GeoArrowWKTSkipWhitespace(s);
      NANOARROW_RETURN_NOT_OK(GeoArrowWKTAssertChar(s, ',', v->error));
      GeoArrowWKTSkipWhitespace(s);
      NANOARROW_RETURN_NOT_OK(
          v->geom_start(v, GEOARROW_GEOMETRY_TYPE_POLYGON, dimensions));
      NANOARROW_RETURN_NOT_OK(ReadPolygon(s, v));
      NANOARROW_RETURN_NOT_OK(v->geom_end(v));
      GeoArrowWKTSkipWhitespace(s);
    }

    GeoArrowWKTAdvanceUnsafe(s, 1);
    return GEOARROW_OK;
  }

  return GeoArrowWKTAssertWordEmpty(s, v->error);
}

static inline int ReadTaggedGeometry(struct WKTReaderPrivate* s,
                                     struct GeoArrowVisitor* v);

static inline int ReadGeometryCollection(struct WKTReaderPrivate* s,
                                         struct GeoArrowVisitor* v) {
  GeoArrowWKTSkipWhitespace(s);
  if (PeekChar(s) == '(') {
    GeoArrowWKTAdvanceUnsafe(s, 1);
    GeoArrowWKTSkipWhitespace(s);

    // Read the first geometry (there must always be one)
    NANOARROW_RETURN_NOT_OK(ReadTaggedGeometry(s, v));
    GeoArrowWKTSkipWhitespace(s);

    // Read the rest of the geometries
    while (PeekChar(s) != ')') {
      GeoArrowWKTSkipWhitespace(s);
      NANOARROW_RETURN_NOT_OK(GeoArrowWKTAssertChar(s, ',', v->error));
      GeoArrowWKTSkipWhitespace(s);
      NANOARROW_RETURN_NOT_OK(ReadTaggedGeometry(s, v));
      GeoArrowWKTSkipWhitespace(s);
    }

    GeoArrowWKTAdvanceUnsafe(s, 1);
    return GEOARROW_OK;
  }

  return GeoArrowWKTAssertWordEmpty(s, v->error);
}

static inline int ReadTaggedGeometry(struct WKTReaderPrivate* s,
                                     struct GeoArrowVisitor* v) {
  GeoArrowWKTSkipWhitespace(s);

  struct ArrowStringView word = GeoArrowWKTPeekUntilSep(s, 19);
  enum GeoArrowGeometryType geometry_type;
  if (word.size_bytes == 5 && strncmp(word.data, "POINT", 5) == 0) {
    geometry_type = GEOARROW_GEOMETRY_TYPE_POINT;
  } else if (word.size_bytes == 10 && strncmp(word.data, "LINESTRING", 10) == 0) {
    geometry_type = GEOARROW_GEOMETRY_TYPE_LINESTRING;
  } else if (word.size_bytes == 7 && strncmp(word.data, "POLYGON", 7) == 0) {
    geometry_type = GEOARROW_GEOMETRY_TYPE_POLYGON;
  } else if (word.size_bytes == 10 && strncmp(word.data, "MULTIPOINT", 10) == 0) {
    geometry_type = GEOARROW_GEOMETRY_TYPE_MULTIPOINT;
  } else if (word.size_bytes == 15 && strncmp(word.data, "MULTILINESTRING", 15) == 0) {
    geometry_type = GEOARROW_GEOMETRY_TYPE_MULTILINESTRING;
  } else if (word.size_bytes == 12 && strncmp(word.data, "MULTIPOLYGON", 12) == 0) {
    geometry_type = GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON;
  } else if (word.size_bytes == 18 && strncmp(word.data, "GEOMETRYCOLLECTION", 18) == 0) {
    geometry_type = GEOARROW_GEOMETRY_TYPE_GEOMETRYCOLLECTION;
  } else {
    SetParseErrorAuto("geometry type", s, v->error);
    return EINVAL;
  }

  GeoArrowWKTAdvanceUnsafe(s, word.size_bytes);
  GeoArrowWKTSkipWhitespace(s);

  enum GeoArrowDimensions dimensions = GEOARROW_DIMENSIONS_XY;
  s->coord_view.n_values = 2;
  word = GeoArrowWKTPeekUntilSep(s, 3);
  if (word.size_bytes == 1 && strncmp(word.data, "Z", 1) == 0) {
    dimensions = GEOARROW_DIMENSIONS_XYZ;
    s->coord_view.n_values = 3;
    GeoArrowWKTAdvanceUnsafe(s, 1);
  } else if (word.size_bytes == 1 && strncmp(word.data, "M", 1) == 0) {
    dimensions = GEOARROW_DIMENSIONS_XYM;
    s->coord_view.n_values = 3;
    GeoArrowWKTAdvanceUnsafe(s, 1);
  } else if (word.size_bytes == 2 && strncmp(word.data, "ZM", 2) == 0) {
    dimensions = GEOARROW_DIMENSIONS_XYZM;
    s->coord_view.n_values = 4;
    GeoArrowWKTAdvanceUnsafe(s, 2);
  }

  NANOARROW_RETURN_NOT_OK(v->geom_start(v, geometry_type, dimensions));

  switch (geometry_type) {
    case GEOARROW_GEOMETRY_TYPE_POINT:
      NANOARROW_RETURN_NOT_OK(ReadEmptyOrPointCoordinate(s, v));
      break;
    case GEOARROW_GEOMETRY_TYPE_LINESTRING:
      NANOARROW_RETURN_NOT_OK(ReadEmptyOrCoordinates(s, v));
      break;
    case GEOARROW_GEOMETRY_TYPE_POLYGON:
      NANOARROW_RETURN_NOT_OK(ReadPolygon(s, v));
      break;
    case GEOARROW_GEOMETRY_TYPE_MULTIPOINT:
      NANOARROW_RETURN_NOT_OK(ReadMultipoint(s, v, dimensions));
      break;
    case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING:
      NANOARROW_RETURN_NOT_OK(ReadMultilinestring(s, v, dimensions));
      break;
    case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON:
      NANOARROW_RETURN_NOT_OK(ReadMultipolygon(s, v, dimensions));
      break;
    case GEOARROW_GEOMETRY_TYPE_GEOMETRYCOLLECTION:
      NANOARROW_RETURN_NOT_OK(ReadGeometryCollection(s, v));
      break;
    default:
      GeoArrowErrorSet(v->error, "Internal error: unrecognized geometry type id");
      return EINVAL;
  }

  return v->geom_end(v);
}

GeoArrowErrorCode GeoArrowWKTReaderInit(struct GeoArrowWKTReader* reader) {
  struct WKTReaderPrivate* s =
      (struct WKTReaderPrivate*)ArrowMalloc(sizeof(struct WKTReaderPrivate));

  if (s == NULL) {
    return ENOMEM;
  }

  s->data0 = NULL;
  s->data = NULL;
  s->size_bytes = 0;

  s->coord_view.coords_stride = 1;
  s->coord_view.values[0] = s->coords;
  for (int i = 1; i < 4; i++) {
    s->coord_view.values[i] = s->coord_view.values[i - 1] + COORD_CACHE_SIZE_COORDS;
  }

  reader->private_data = s;
  return GEOARROW_OK;
}

void GeoArrowWKTReaderReset(struct GeoArrowWKTReader* reader) {
  ArrowFree(reader->private_data);
}

GeoArrowErrorCode GeoArrowWKTReaderVisit(struct GeoArrowWKTReader* reader,
                                         struct GeoArrowStringView src,
                                         struct GeoArrowVisitor* v) {
  struct WKTReaderPrivate* s = (struct WKTReaderPrivate*)reader->private_data;
  s->data0 = src.data;
  s->data = src.data;
  s->size_bytes = src.size_bytes;

  NANOARROW_RETURN_NOT_OK(v->feat_start(v));
  NANOARROW_RETURN_NOT_OK(ReadTaggedGeometry(s, v));
  NANOARROW_RETURN_NOT_OK(v->feat_end(v));
  GeoArrowWKTSkipWhitespace(s);
  if (PeekChar(s) != '\0') {
    SetParseErrorAuto("end of input", s, v->error);
    return EINVAL;
  }

  return GEOARROW_OK;
}

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "nanoarrow/nanoarrow.h"



struct WKTWriterPrivate {
  enum ArrowType storage_type;
  struct ArrowBitmap validity;
  struct ArrowBuffer offsets;
  struct ArrowBuffer values;
  enum GeoArrowGeometryType geometry_type[32];
  int64_t i[32];
  int32_t level;
  int64_t length;
  int64_t null_count;
  int64_t values_feat_start;
  int precision;
  int use_flat_multipoint;
  int64_t max_element_size_bytes;
  int feat_is_null;
};

static inline int WKTWriterCheckLevel(struct WKTWriterPrivate* private) {
  if (private->level >= 0 && private->level <= 31) {
    return GEOARROW_OK;
  } else {
    return EINVAL;
  }
}

static inline int WKTWriterWrite(struct WKTWriterPrivate* private, const char* value) {
  return ArrowBufferAppend(&private->values, value, strlen(value));
}

static GeoArrowErrorCode WKTWriterWriteGeometryType(
    struct WKTWriterPrivate* private_data, enum GeoArrowGeometryType geometry_type,
    enum GeoArrowDimensions dimensions) {
  const char* geometry_type_name = GeoArrowGeometryTypeString(geometry_type);
  NANOARROW_RETURN_NOT_OK(WKTWriterWrite(private_data, geometry_type_name));

  switch (dimensions) {
    case GEOARROW_DIMENSIONS_XY:
      break;
    case GEOARROW_DIMENSIONS_XYZ:
      NANOARROW_RETURN_NOT_OK(WKTWriterWrite(private_data, " Z"));
      break;
    case GEOARROW_DIMENSIONS_XYM:
      NANOARROW_RETURN_NOT_OK(WKTWriterWrite(private_data, " M"));
      break;
    case GEOARROW_DIMENSIONS_XYZM:
      NANOARROW_RETURN_NOT_OK(WKTWriterWrite(private_data, " ZM"));
      break;
    default:
      NANOARROW_RETURN_NOT_OK(WKTWriterWrite(private_data, " <not valid>"));
      return EINVAL;
  }

  GEOARROW_RETURN_NOT_OK(WKTWriterWrite(private_data, " "));

  return GEOARROW_OK;
}

static GeoArrowErrorCode WKTWriterReserveCoords(struct WKTWriterPrivate* private_data,
                                                int64_t n_coords, int n_values) {
  int64_t max_chars_per_coord_theoretical =
      // space + comma after coordinate
      (n_coords * 2) +
      // spaces between ordinates
      (n_coords * (n_values - 1)) +
      // GeoArrowPrintDouble might require up to 40 accessible bytes per call
      (40 * n_values);

  // Use a heuristic to estimate the number of characters we are about to write
  // to avoid more then one allocation for this call. This is normally substantially
  // less than the theoretical amount.
  int64_t max_chars_estimated =
      (n_coords * 2) +               // space + comma after coordinate
      (n_coords * (n_values - 1)) +  // spaces between ordinates
      // precision + decimal + estimate of normal
      // digits to the left of the decimal
      ((private_data->precision + 1 + 8) * n_coords * n_values) +
      // Ensure that the last reserve() call doesn't trigger an allocation
      max_chars_per_coord_theoretical;

  GEOARROW_RETURN_NOT_OK(ArrowBufferReserve(&private_data->values, max_chars_estimated));
  return GEOARROW_OK;
}

static GeoArrowErrorCode WKTWriterCheckElementOutputSize(
    struct WKTWriterPrivate* private_data) {
  // Check if we've hit our max number of bytes for this feature
  if (private_data->max_element_size_bytes >= 0 &&
      (private_data->values.size_bytes - private_data->values_feat_start) >=
          private_data->max_element_size_bytes) {
    return EAGAIN;
  } else {
    return GEOARROW_OK;
  }
}

static void WKTWriterUpdateOptions(struct GeoArrowWKTWriter* writer,
                                   struct WKTWriterPrivate* private_data) {
  // Clamp writer->precision to a specific range of valid values
  if (writer->precision < 0 || writer->precision > 16) {
    private_data->precision = 16;
  } else {
    private_data->precision = writer->precision;
  }

  private_data->use_flat_multipoint = writer->use_flat_multipoint;
  private_data->max_element_size_bytes = writer->max_element_size_bytes;
}

static inline void WKTWriterWriteDoubleUnsafe(struct WKTWriterPrivate* private,
                                              double value) {
  // Always ensure that we have at least 40 writable bytes remaining before calling
  // GeoArrowPrintDouble()
  NANOARROW_DCHECK((private->values.capacity_bytes - private->values.size_bytes) >= 40);
  private->values.size_bytes +=
      GeoArrowPrintDouble(value, private->precision,
                          ((char*)private->values.data) + private->values.size_bytes);
}

static inline int WKTWriterCoordIsEmptyPoint(const struct GeoArrowCoordView* coords,
                                             int64_t i) {
  for (int32_t j = 0; j < coords->n_values; j++) {
    if (!isnan(GEOARROW_COORD_VIEW_VALUE(coords, i, j))) {
      return 0;
    }
  }

  return 1;
}

static inline int WKTWriterNodeIsEmptyPoint(const struct GeoArrowGeometryNode* node) {
  int n_values = _GeoArrowkNumDimensions[node->dimensions];
  if (node->geometry_type != GEOARROW_GEOMETRY_TYPE_POINT || node->size != 1 ||
      n_values < 2) {
    return 0;
  }

  for (int j = 0; j < n_values; j++) {
    double value;
    memcpy(&value, node->coords[j], sizeof(double));
    if (!isnan(value)) {
      return 0;
    }
  }

  return 1;
}

static int feat_start_wkt(struct GeoArrowVisitor* v) {
  struct WKTWriterPrivate* private = (struct WKTWriterPrivate*)v->private_data;
  private->level = -1;
  private->length++;
  private->feat_is_null = 0;
  private->values_feat_start = private->values.size_bytes;

  if (private->values.size_bytes > 2147483647) {
    return EOVERFLOW;
  }
  return ArrowBufferAppendInt32(&private->offsets, (int32_t) private->values.size_bytes);
}

static int null_feat_wkt(struct GeoArrowVisitor* v) {
  struct WKTWriterPrivate* private = (struct WKTWriterPrivate*)v->private_data;
  private->feat_is_null = 1;
  return GEOARROW_OK;
}

static int geom_start_wkt(struct GeoArrowVisitor* v,
                          enum GeoArrowGeometryType geometry_type,
                          enum GeoArrowDimensions dimensions) {
  struct WKTWriterPrivate* private = (struct WKTWriterPrivate*)v->private_data;
  private->level++;
  NANOARROW_RETURN_NOT_OK(WKTWriterCheckLevel(private));

  if (private->level > 0 && private->i[private->level - 1] > 0) {
    NANOARROW_RETURN_NOT_OK(WKTWriterWrite(private, ", "));
  } else if (private->level > 0) {
    NANOARROW_RETURN_NOT_OK(WKTWriterWrite(private, "("));
  }

  if (private->level == 0 || private->geometry_type[private->level - 1] ==
                                 GEOARROW_GEOMETRY_TYPE_GEOMETRYCOLLECTION) {
    if (geometry_type < GEOARROW_GEOMETRY_TYPE_POINT ||
        geometry_type > GEOARROW_GEOMETRY_TYPE_GEOMETRYCOLLECTION) {
      GeoArrowErrorSet(v->error, "WKTWriter::geom_start(): Unexpected `geometry_type`");
      return EINVAL;
    }

    GEOARROW_RETURN_NOT_OK(
        WKTWriterWriteGeometryType(private, geometry_type, dimensions));
  }

  if (private->level > 0) {
    private->i[private->level - 1]++;
  }

  private->geometry_type[private->level] = geometry_type;
  private->i[private->level] = 0;
  return GEOARROW_OK;
}

static int ring_start_wkt(struct GeoArrowVisitor* v) {
  struct WKTWriterPrivate* private = (struct WKTWriterPrivate*)v->private_data;
  private->level++;
  NANOARROW_RETURN_NOT_OK(WKTWriterCheckLevel(private));

  if (private->level > 0 && private->i[private->level - 1] > 0) {
    NANOARROW_RETURN_NOT_OK(WKTWriterWrite(private, ", "));
  } else {
    NANOARROW_RETURN_NOT_OK(WKTWriterWrite(private, "("));
  }

  if (private->level > 0) {
    private->i[private->level - 1]++;
  }

  private->geometry_type[private->level] = GEOARROW_GEOMETRY_TYPE_GEOMETRY;
  private->i[private->level] = 0;
  return GEOARROW_OK;
}

static int coords_wkt(struct GeoArrowVisitor* v, const struct GeoArrowCoordView* coords) {
  int64_t n_coords = coords->n_coords;
  int32_t n_dims = coords->n_values;
  if (n_coords == 0) {
    return GEOARROW_OK;
  }

  struct WKTWriterPrivate* private = (struct WKTWriterPrivate*)v->private_data;
  NANOARROW_RETURN_NOT_OK(WKTWriterCheckLevel(private));

  // GeoArrow's fixed-size point representation uses an all-NaN coordinate as
  // the sentinel for POINT EMPTY. Preserve NaNs in every other context.
  if (private->geometry_type[private->level] == GEOARROW_GEOMETRY_TYPE_POINT &&
      private->i[private->level] == 0 && n_coords == 1 &&
      WKTWriterCoordIsEmptyPoint(coords, 0)) {
    return GEOARROW_OK;
  }

  GEOARROW_RETURN_NOT_OK(WKTWriterReserveCoords(private, n_coords, n_dims));

  int64_t max_chars_per_coord_theoretical =
      // space + comma after coordinate
      (n_coords * 2) +
      // spaces between ordinates
      (n_coords * (n_dims - 1)) +
      // GeoArrowPrintDouble might require up to 40 accessible bytes per call
      (40 * n_dims);

  // Use a heuristic to estimate the number of characters we are about to write
  // to avoid more then one allocation for this call. This is normally substantially
  // less than the theoretical amount.
  int64_t max_chars_estimated =
      (n_coords * 2) +             // space + comma after coordinate
      (n_coords * (n_dims - 1)) +  // spaces between ordinates
      // precision + decimal + estimate of normal
      // digits to the left of the decimal
      ((private->precision + 1 + 8) * n_coords * n_dims) +
      // Ensure that the last reserve() call doesn't trigger an allocation
      max_chars_per_coord_theoretical;
  NANOARROW_RETURN_NOT_OK(ArrowBufferReserve(&private->values, max_chars_estimated));

  // Write the first coordinate, possibly with a leading comma if there was
  // a previous call to coords, or the opening (if it wasn't). Special case
  // for the flat multipoint output MULTIPOINT (1 2, 3 4, ...) which doesn't
  // have extra () for inner POINTs
  if (private->i[private->level] != 0) {
    ArrowBufferAppendUnsafe(&private->values, ", ", 2);
  } else if (private->level < 1 || !private->use_flat_multipoint ||
             private->geometry_type[private->level - 1] !=
                 GEOARROW_GEOMETRY_TYPE_MULTIPOINT) {
    ArrowBufferAppendUnsafe(&private->values, "(", 1);
  }

  // Actually write the first coordinate (no leading comma)
  // Reserve the theoretical amount for each coordinate because we need this to guarantee
  // that there won't be a segfault when writing a coordinate. This probably results in
  // a few dozen bytes of of overallocation.
  NANOARROW_RETURN_NOT_OK(
      ArrowBufferReserve(&private->values, max_chars_per_coord_theoretical));
  WKTWriterWriteDoubleUnsafe(private, GEOARROW_COORD_VIEW_VALUE(coords, 0, 0));
  for (int32_t j = 1; j < n_dims; j++) {
    ArrowBufferAppendUnsafe(&private->values, " ", 1);
    WKTWriterWriteDoubleUnsafe(private, GEOARROW_COORD_VIEW_VALUE(coords, 0, j));
  }

  // Write the remaining coordinates (which all have leading commas)
  for (int64_t i = 1; i < n_coords; i++) {
    // Check if we've hit our max number of bytes for this feature
    if (private->max_element_size_bytes >= 0 &&
        (private->values.size_bytes - private->values_feat_start) >=
            private->max_element_size_bytes) {
      return EAGAIN;
    }

    NANOARROW_RETURN_NOT_OK(
        ArrowBufferReserve(&private->values, max_chars_per_coord_theoretical));
    ArrowBufferAppendUnsafe(&private->values, ", ", 2);
    WKTWriterWriteDoubleUnsafe(private, GEOARROW_COORD_VIEW_VALUE(coords, i, 0));
    for (int32_t j = 1; j < n_dims; j++) {
      ArrowBufferAppendUnsafe(&private->values, " ", 1);
      WKTWriterWriteDoubleUnsafe(private, GEOARROW_COORD_VIEW_VALUE(coords, i, j));
    }
  }

  private->i[private->level] += n_coords;
  return GEOARROW_OK;
}

static int ring_end_wkt(struct GeoArrowVisitor* v) {
  struct WKTWriterPrivate* private = (struct WKTWriterPrivate*)v->private_data;
  NANOARROW_RETURN_NOT_OK(WKTWriterCheckLevel(private));
  if (private->i[private->level] == 0) {
    private->level--;
    return WKTWriterWrite(private, "EMPTY");
  } else {
    private->level--;
    return WKTWriterWrite(private, ")");
  }
}

static int geom_end_wkt(struct GeoArrowVisitor* v) {
  struct WKTWriterPrivate* private = (struct WKTWriterPrivate*)v->private_data;
  NANOARROW_RETURN_NOT_OK(WKTWriterCheckLevel(private));

  if (private->i[private->level] == 0) {
    private->level--;
    return WKTWriterWrite(private, "EMPTY");
  } else if (private->level < 1 || !private->use_flat_multipoint ||
             private->geometry_type[private->level - 1] !=
                 GEOARROW_GEOMETRY_TYPE_MULTIPOINT) {
    private->level--;
    return WKTWriterWrite(private, ")");
  } else {
    private->level--;
    return GEOARROW_OK;
  }
}

static int feat_end_wkt(struct GeoArrowVisitor* v) {
  struct WKTWriterPrivate* private = (struct WKTWriterPrivate*)v->private_data;

  if (private->feat_is_null) {
    if (private->validity.buffer.data == NULL) {
      NANOARROW_RETURN_NOT_OK(ArrowBitmapReserve(&private->validity, private->length));
      ArrowBitmapAppendUnsafe(&private->validity, 1, private->length - 1);
    }

    private->null_count++;
    return ArrowBitmapAppend(&private->validity, 0, 1);
  } else if (private->validity.buffer.data != NULL) {
    return ArrowBitmapAppend(&private->validity, 1, 1);
  }

  if (private->max_element_size_bytes >= 0 &&
      (private->values.size_bytes - private->values_feat_start) >
          private->max_element_size_bytes) {
    private->values.size_bytes =
        private->values_feat_start + private->max_element_size_bytes;
  }

  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowWKTWriterAppendValidity(struct GeoArrowWKTWriter* writer,
                                                         int is_valid) {
  struct WKTWriterPrivate* private_data = (struct WKTWriterPrivate*)writer->private_data;

  private_data->length++;

  if (!is_valid) {
    if (private_data->validity.buffer.data == NULL) {
      NANOARROW_RETURN_NOT_OK(
          ArrowBitmapReserve(&private_data->validity, private_data->length));
      ArrowBitmapAppendUnsafe(&private_data->validity, 1, private_data->length - 1);
    }

    private_data->null_count++;
    NANOARROW_RETURN_NOT_OK(ArrowBitmapAppend(&private_data->validity, 0, 1));
  } else if (private_data->validity.buffer.data != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowBitmapAppend(&private_data->validity, 1, 1));
  }

  return GEOARROW_OK;
}

static inline void GeoArrowWKTWriterAppendCoordUnsafe(
    struct WKTWriterPrivate* private_data, const uint8_t** cursor, const int32_t* strides,
    int n_values) {
  double value;

  memcpy(&value, cursor[0], sizeof(double));
  WKTWriterWriteDoubleUnsafe(private_data, value);
  cursor[0] += strides[0];

  for (int j = 1; j < n_values; j++) {
    ArrowBufferAppendUnsafe(&private_data->values, " ", 1);
    memcpy(&value, cursor[j], sizeof(double));
    WKTWriterWriteDoubleUnsafe(private_data, value);
    cursor[j] += strides[j];
  }
}

static GeoArrowErrorCode GeoArrowWKTWriterAppendCoordsUnsafe(
    struct WKTWriterPrivate* private_data, const struct GeoArrowGeometryNode* node,
    int n_values) {
  const uint8_t* cursor[4];
  memcpy(cursor, node->coords, sizeof(cursor));

  int32_t stride[4];
  memcpy(stride, node->coord_stride, sizeof(stride));

  // Append the first coord
  GeoArrowWKTWriterAppendCoordUnsafe(private_data, cursor, stride, n_values);

  // Append the rest of the coords
  for (uint32_t i = 1; i < node->size; i++) {
    GEOARROW_RETURN_NOT_OK(WKTWriterCheckElementOutputSize(private_data));
    ArrowBufferAppendUnsafe(&private_data->values, ", ", 2);
    GeoArrowWKTWriterAppendCoordUnsafe(private_data, cursor, stride, n_values);
  }

  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowWKTWriterAppendSequenceOrEmpty(
    struct WKTWriterPrivate* private_data, const struct GeoArrowGeometryNode* node) {
  if (node->size == 0 || WKTWriterNodeIsEmptyPoint(node)) {
    GEOARROW_RETURN_NOT_OK(WKTWriterWrite(private_data, "EMPTY"));
    return GEOARROW_OK;
  }

  uint32_t n_coords = node->size;
  int n_values = _GeoArrowkNumDimensions[node->dimensions];

  GEOARROW_RETURN_NOT_OK(WKTWriterReserveCoords(private_data, n_coords, n_values));

  GEOARROW_RETURN_NOT_OK(WKTWriterWrite(private_data, "("));
  GEOARROW_RETURN_NOT_OK(
      GeoArrowWKTWriterAppendCoordsUnsafe(private_data, node, n_values));
  GEOARROW_RETURN_NOT_OK(WKTWriterWrite(private_data, ")"));

  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowWKTWriterAppendFlatMultipoint(
    struct WKTWriterPrivate* private_data, const struct GeoArrowGeometryNode* node,
    const struct GeoArrowGeometryNode** node_iter) {
  int32_t stride[4] = {0, 0, 0, 0};

  GEOARROW_RETURN_NOT_OK(WKTWriterWrite(private_data, "("));

  for (uint32_t i = 0; i < node->size; i++) {
    const struct GeoArrowGeometryNode* child_node = (*node_iter)++;

    if (i > 0) {
      GEOARROW_RETURN_NOT_OK(WKTWriterCheckElementOutputSize(private_data));
      GEOARROW_RETURN_NOT_OK(WKTWriterWrite(private_data, ", "));
    }

    if (child_node->size == 0 || WKTWriterNodeIsEmptyPoint(child_node)) {
      GEOARROW_RETURN_NOT_OK(WKTWriterWrite(private_data, "EMPTY"));
    } else {
      int n_values = _GeoArrowkNumDimensions[child_node->dimensions];
      const uint8_t* cursor[4];
      GEOARROW_RETURN_NOT_OK(
          WKTWriterReserveCoords(private_data, child_node->size, n_values));
      memcpy(cursor, child_node->coords, sizeof(cursor));
      GeoArrowWKTWriterAppendCoordUnsafe(private_data, cursor, stride, n_values);
    }
  }

  GEOARROW_RETURN_NOT_OK(WKTWriterWrite(private_data, ")"));
  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowWKTWriterAppendImpl(
    struct WKTWriterPrivate* private_data, const struct GeoArrowGeometryNode** node_iter,
    int tag) {
  const struct GeoArrowGeometryNode* node = (*node_iter)++;

  if (tag) {
    enum GeoArrowGeometryType geometry_type =
        (enum GeoArrowGeometryType)node->geometry_type;
    enum GeoArrowDimensions dimensions = (enum GeoArrowDimensions)node->dimensions;

    GEOARROW_RETURN_NOT_OK(
        WKTWriterWriteGeometryType(private_data, geometry_type, dimensions));
  }

  switch (node->geometry_type) {
    case GEOARROW_GEOMETRY_TYPE_POINT:
    case GEOARROW_GEOMETRY_TYPE_LINESTRING:
      GEOARROW_RETURN_NOT_OK(GeoArrowWKTWriterAppendSequenceOrEmpty(private_data, node));
      return GEOARROW_OK;

    case GEOARROW_GEOMETRY_TYPE_MULTIPOINT:
      if (node->size == 0) {
        GEOARROW_RETURN_NOT_OK(WKTWriterWrite(private_data, "EMPTY"));
        return GEOARROW_OK;
      }

      if (private_data->use_flat_multipoint) {
        return GeoArrowWKTWriterAppendFlatMultipoint(private_data, node, node_iter);
      }
#if defined(__GNUC__) || defined(__clang__)
      __attribute__((fallthrough));
#endif
    case GEOARROW_GEOMETRY_TYPE_POLYGON:
    case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING:
    case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON:
    case GEOARROW_GEOMETRY_TYPE_GEOMETRYCOLLECTION: {
      if (node->size == 0) {
        GEOARROW_RETURN_NOT_OK(WKTWriterWrite(private_data, "EMPTY"));
        return GEOARROW_OK;
      }

      GEOARROW_RETURN_NOT_OK(WKTWriterWrite(private_data, "("));
      GEOARROW_RETURN_NOT_OK(GeoArrowWKTWriterAppendImpl(
          private_data, node_iter,
          node->geometry_type == GEOARROW_GEOMETRY_TYPE_GEOMETRYCOLLECTION));
      for (uint32_t i = 1; i < node->size; i++) {
        GEOARROW_RETURN_NOT_OK(WKTWriterWrite(private_data, ", "));
        GEOARROW_RETURN_NOT_OK(GeoArrowWKTWriterAppendImpl(
            private_data, node_iter,
            node->geometry_type == GEOARROW_GEOMETRY_TYPE_GEOMETRYCOLLECTION));
      }
      GEOARROW_RETURN_NOT_OK(WKTWriterWrite(private_data, ")"));
      return GEOARROW_OK;
    }
    default:
      return EINVAL;
  }
}

GeoArrowErrorCode GeoArrowWKTWriterAppendNull(struct GeoArrowWKTWriter* writer) {
  struct WKTWriterPrivate* private_data = (struct WKTWriterPrivate*)writer->private_data;
  GEOARROW_RETURN_NOT_OK(ArrowBufferAppendInt32(
      &private_data->offsets, (int32_t)private_data->values.size_bytes));
  GEOARROW_RETURN_NOT_OK(GeoArrowWKTWriterAppendValidity(writer, 0));
  return GEOARROW_OK;
}

GeoArrowErrorCode GeoArrowWKTWriterAppend(struct GeoArrowWKTWriter* writer,
                                          struct GeoArrowGeometryView geom) {
  struct WKTWriterPrivate* private_data = (struct WKTWriterPrivate*)writer->private_data;
  WKTWriterUpdateOptions(writer, private_data);

  private_data->values_feat_start = private_data->values.size_bytes;

  GEOARROW_RETURN_NOT_OK(ArrowBufferAppendInt32(
      &private_data->offsets, (int32_t)private_data->values.size_bytes));

  const struct GeoArrowGeometryNode* node_iter = geom.root;
  GeoArrowErrorCode result = GeoArrowWKTWriterAppendImpl(private_data, &node_iter, 1);
  if (result != GEOARROW_OK && result != EAGAIN) {
    return result;
  }

  if (private_data->max_element_size_bytes >= 0 &&
      (private_data->values.size_bytes - private_data->values_feat_start) >
          private_data->max_element_size_bytes) {
    private_data->values.size_bytes =
        private_data->values_feat_start + private_data->max_element_size_bytes;
  }

  GEOARROW_RETURN_NOT_OK(GeoArrowWKTWriterAppendValidity(writer, 1));
  return GEOARROW_OK;
}

GeoArrowErrorCode GeoArrowWKTWriterInit(struct GeoArrowWKTWriter* writer) {
  struct WKTWriterPrivate* private =
      (struct WKTWriterPrivate*)ArrowMalloc(sizeof(struct WKTWriterPrivate));
  if (private == NULL) {
    return ENOMEM;
  }

  private->storage_type = NANOARROW_TYPE_STRING;
  private->length = 0;
  private->level = 0;
  private->null_count = 0;
  ArrowBitmapInit(&private->validity);
  ArrowBufferInit(&private->offsets);
  ArrowBufferInit(&private->values);
  writer->precision = 16;
  private->precision = 16;
  writer->use_flat_multipoint = 1;
  private->use_flat_multipoint = 1;
  writer->max_element_size_bytes = -1;
  private->max_element_size_bytes = -1;
  writer->private_data = private;

  return GEOARROW_OK;
}

void GeoArrowWKTWriterInitVisitor(struct GeoArrowWKTWriter* writer,
                                  struct GeoArrowVisitor* v) {
  GeoArrowVisitorInitVoid(v);

  struct WKTWriterPrivate* private = (struct WKTWriterPrivate*)writer->private_data;

  WKTWriterUpdateOptions(writer, private);

  v->private_data = writer->private_data;
  v->feat_start = &feat_start_wkt;
  v->null_feat = &null_feat_wkt;
  v->geom_start = &geom_start_wkt;
  v->ring_start = &ring_start_wkt;
  v->coords = &coords_wkt;
  v->ring_end = &ring_end_wkt;
  v->geom_end = &geom_end_wkt;
  v->feat_end = &feat_end_wkt;
}

GeoArrowErrorCode GeoArrowWKTWriterFinish(struct GeoArrowWKTWriter* writer,
                                          struct ArrowArray* array,
                                          struct GeoArrowError* error) {
  struct WKTWriterPrivate* private = (struct WKTWriterPrivate*)writer->private_data;
  array->release = NULL;

  if (private->values.size_bytes > 2147483647) {
    return EOVERFLOW;
  }
  NANOARROW_RETURN_NOT_OK(
      ArrowBufferAppendInt32(&private->offsets, (int32_t) private->values.size_bytes));
  NANOARROW_RETURN_NOT_OK(ArrowArrayInitFromType(array, private->storage_type));
  ArrowArraySetValidityBitmap(array, &private->validity);
  NANOARROW_RETURN_NOT_OK(ArrowArraySetBuffer(array, 1, &private->offsets));
  NANOARROW_RETURN_NOT_OK(ArrowArraySetBuffer(array, 2, &private->values));
  array->length = private->length;
  array->null_count = private->null_count;
  private->length = 0;
  private->null_count = 0;
  return ArrowArrayFinishBuildingDefault(array, (struct ArrowError*)error);
}

void GeoArrowWKTWriterReset(struct GeoArrowWKTWriter* writer) {
  struct WKTWriterPrivate* private = (struct WKTWriterPrivate*)writer->private_data;
  ArrowBitmapReset(&private->validity);
  ArrowBufferReset(&private->offsets);
  ArrowBufferReset(&private->values);
  ArrowFree(private);
  writer->private_data = NULL;
}



#include "nanoarrow/nanoarrow.h"

union GeoArrowArrayReaderSrc {
  struct GeoArrowArrayView geoarrow;
  struct ArrowArrayView arrow;
};

struct GeoArrowArrayReaderPrivate {
  enum GeoArrowType type;
  union GeoArrowArrayReaderSrc src;
  struct GeoArrowWKTReader wkt_reader;
  struct GeoArrowWKBReader wkb_reader;
};

static GeoArrowErrorCode GeoArrowArrayReaderVisitWKT(
    const struct GeoArrowArrayView* array_view, int64_t offset, int64_t length,
    struct GeoArrowWKTReader* reader, struct GeoArrowVisitor* v) {
  struct GeoArrowStringView item;
  const int32_t* offset_begin = array_view->offsets[0] + array_view->offset[0] + offset;

  for (int64_t i = 0; i < length; i++) {
    if (!array_view->validity_bitmap ||
        ArrowBitGet(array_view->validity_bitmap, array_view->offset[0] + offset + i)) {
      item.data = (const char*)(array_view->data + offset_begin[i]);
      item.size_bytes = offset_begin[i + 1] - offset_begin[i];
      NANOARROW_RETURN_NOT_OK(GeoArrowWKTReaderVisit(reader, item, v));
    } else {
      NANOARROW_RETURN_NOT_OK(v->feat_start(v));
      NANOARROW_RETURN_NOT_OK(v->null_feat(v));
      NANOARROW_RETURN_NOT_OK(v->feat_end(v));
    }
  }

  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowArrayReaderVisitWKTArrow(
    const struct ArrowArrayView* array_view, int64_t offset, int64_t length,
    struct GeoArrowWKTReader* reader, struct GeoArrowVisitor* v) {
  struct ArrowStringView arrow_item;
  struct GeoArrowStringView item;

  for (int64_t i = 0; i < length; i++) {
    if (!ArrowArrayViewIsNull(array_view, offset + i)) {
      arrow_item = ArrowArrayViewGetStringUnsafe(array_view, offset + i);
      item.data = arrow_item.data;
      item.size_bytes = arrow_item.size_bytes;
      NANOARROW_RETURN_NOT_OK(GeoArrowWKTReaderVisit(reader, item, v));
    } else {
      NANOARROW_RETURN_NOT_OK(v->feat_start(v));
      NANOARROW_RETURN_NOT_OK(v->null_feat(v));
      NANOARROW_RETURN_NOT_OK(v->feat_end(v));
    }
  }

  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowArrayReaderVisitWKB(
    const struct GeoArrowArrayView* array_view, int64_t offset, int64_t length,
    struct GeoArrowWKBReader* reader, struct GeoArrowVisitor* v) {
  struct GeoArrowBufferView item;
  const int32_t* offset_begin = array_view->offsets[0] + array_view->offset[0] + offset;

  for (int64_t i = 0; i < length; i++) {
    if (!array_view->validity_bitmap ||
        ArrowBitGet(array_view->validity_bitmap, array_view->offset[0] + offset + i)) {
      item.data = array_view->data + offset_begin[i];
      item.size_bytes = offset_begin[i + 1] - offset_begin[i];
      NANOARROW_RETURN_NOT_OK(GeoArrowWKBReaderVisit(reader, item, v));
    } else {
      NANOARROW_RETURN_NOT_OK(v->feat_start(v));
      NANOARROW_RETURN_NOT_OK(v->null_feat(v));
      NANOARROW_RETURN_NOT_OK(v->feat_end(v));
    }
  }

  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowArrayReaderVisitWKBArrow(
    const struct ArrowArrayView* array_view, int64_t offset, int64_t length,
    struct GeoArrowWKBReader* reader, struct GeoArrowVisitor* v) {
  struct ArrowBufferView arrow_item;
  struct GeoArrowBufferView item;

  for (int64_t i = 0; i < length; i++) {
    if (!ArrowArrayViewIsNull(array_view, offset + i)) {
      arrow_item = ArrowArrayViewGetBytesUnsafe(array_view, offset + i);
      item.data = arrow_item.data.as_uint8;
      item.size_bytes = arrow_item.size_bytes;
      NANOARROW_RETURN_NOT_OK(GeoArrowWKBReaderVisit(reader, item, v));
    } else {
      NANOARROW_RETURN_NOT_OK(v->feat_start(v));
      NANOARROW_RETURN_NOT_OK(v->null_feat(v));
      NANOARROW_RETURN_NOT_OK(v->feat_end(v));
    }
  }

  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowArrayReaderInitInternal(
    struct GeoArrowArrayReaderPrivate* private_data, enum GeoArrowType type) {
  private_data->type = type;

  switch (type) {
    case GEOARROW_TYPE_LARGE_WKB:
      ArrowArrayViewInitFromType(&private_data->src.arrow, NANOARROW_TYPE_LARGE_BINARY);
      break;
    case GEOARROW_TYPE_WKB_VIEW:
      ArrowArrayViewInitFromType(&private_data->src.arrow, NANOARROW_TYPE_BINARY_VIEW);
      break;
    case GEOARROW_TYPE_LARGE_WKT:
      ArrowArrayViewInitFromType(&private_data->src.arrow, NANOARROW_TYPE_LARGE_STRING);
      break;
    case GEOARROW_TYPE_WKT_VIEW:
      ArrowArrayViewInitFromType(&private_data->src.arrow, NANOARROW_TYPE_STRING_VIEW);
      break;
    default:
      GEOARROW_RETURN_NOT_OK(
          GeoArrowArrayViewInitFromType(&private_data->src.geoarrow, type));
      break;
  }

  // Independent of the source view, we might need a parser
  switch (type) {
    case GEOARROW_TYPE_WKT:
    case GEOARROW_TYPE_LARGE_WKT:
    case GEOARROW_TYPE_WKT_VIEW:
      return GeoArrowWKTReaderInit(&private_data->wkt_reader);
    case GEOARROW_TYPE_WKB:
    case GEOARROW_TYPE_LARGE_WKB:
    case GEOARROW_TYPE_WKB_VIEW:
      return GeoArrowWKBReaderInit(&private_data->wkb_reader);
    default:
      return GEOARROW_OK;
  }
}

GeoArrowErrorCode GeoArrowArrayReaderInitFromType(struct GeoArrowArrayReader* reader,
                                                  enum GeoArrowType type) {
  struct GeoArrowArrayReaderPrivate* private_data =
      (struct GeoArrowArrayReaderPrivate*)ArrowMalloc(
          sizeof(struct GeoArrowArrayReaderPrivate));

  if (private_data == NULL) {
    return ENOMEM;
  }

  memset(private_data, 0, sizeof(struct GeoArrowArrayReaderPrivate));

  int result = GeoArrowArrayReaderInitInternal(private_data, type);
  if (result != GEOARROW_OK) {
    ArrowFree(private_data);
    return result;
  }

  reader->private_data = private_data;
  return GEOARROW_OK;
}

GeoArrowErrorCode GeoArrowArrayReaderInitFromSchema(struct GeoArrowArrayReader* reader,
                                                    const struct ArrowSchema* schema,
                                                    struct GeoArrowError* error) {
  struct GeoArrowArrayReaderPrivate* private_data =
      (struct GeoArrowArrayReaderPrivate*)ArrowMalloc(
          sizeof(struct GeoArrowArrayReaderPrivate));

  if (private_data == NULL) {
    GeoArrowErrorSet(error, "Failed to allocate GeoArrowArrayReaderPrivate");
    return ENOMEM;
  }

  memset(private_data, 0, sizeof(struct GeoArrowArrayReaderPrivate));

  struct GeoArrowSchemaView schema_view;
  int result = GeoArrowSchemaViewInit(&schema_view, schema, error);
  if (result != GEOARROW_OK) {
    ArrowFree(private_data);
    return result;
  }

  result = GeoArrowArrayReaderInitInternal(private_data, schema_view.type);
  if (result != GEOARROW_OK) {
    ArrowFree(private_data);
    GeoArrowErrorSet(error, "GeoArrowArrayReaderInitInternal() failed");
    return result;
  }

  reader->private_data = private_data;
  return GEOARROW_OK;
}

void GeoArrowArrayReaderReset(struct GeoArrowArrayReader* reader) {
  struct GeoArrowArrayReaderPrivate* private_data =
      (struct GeoArrowArrayReaderPrivate*)reader->private_data;

  if (private_data->wkb_reader.private_data != NULL) {
    GeoArrowWKBReaderReset(&private_data->wkb_reader);
  }

  if (private_data->wkt_reader.private_data != NULL) {
    GeoArrowWKTReaderReset(&private_data->wkt_reader);
  }

  ArrowFree(reader->private_data);
  reader->private_data = NULL;
}

GeoArrowErrorCode GeoArrowArrayReaderSetArray(struct GeoArrowArrayReader* reader,
                                              const struct ArrowArray* array,
                                              struct GeoArrowError* error) {
  NANOARROW_DCHECK(reader != NULL);
  NANOARROW_DCHECK(array != NULL);
  struct GeoArrowArrayReaderPrivate* private_data =
      (struct GeoArrowArrayReaderPrivate*)reader->private_data;
  NANOARROW_DCHECK(private_data != NULL);

  switch (private_data->type) {
    case GEOARROW_TYPE_LARGE_WKT:
    case GEOARROW_TYPE_WKT_VIEW:
    case GEOARROW_TYPE_LARGE_WKB:
    case GEOARROW_TYPE_WKB_VIEW:
      GEOARROW_RETURN_NOT_OK(ArrowArrayViewSetArray(&private_data->src.arrow, array,
                                                    (struct ArrowError*)error));
      break;
    default:
      GEOARROW_RETURN_NOT_OK(
          GeoArrowArrayViewSetArray(&private_data->src.geoarrow, array, error));
      break;
  }

  return GEOARROW_OK;
}

GeoArrowErrorCode GeoArrowArrayReaderVisit(struct GeoArrowArrayReader* reader,
                                           int64_t offset, int64_t length,
                                           struct GeoArrowVisitor* v) {
  struct GeoArrowArrayReaderPrivate* private_data =
      (struct GeoArrowArrayReaderPrivate*)reader->private_data;

  if (length == 0) {
    return GEOARROW_OK;
  }

  switch (private_data->type) {
    case GEOARROW_TYPE_WKT:
      return GeoArrowArrayReaderVisitWKT(&private_data->src.geoarrow, offset, length,
                                         &private_data->wkt_reader, v);
    case GEOARROW_TYPE_WKB:
      return GeoArrowArrayReaderVisitWKB(&private_data->src.geoarrow, offset, length,
                                         &private_data->wkb_reader, v);

    case GEOARROW_TYPE_LARGE_WKB:
    case GEOARROW_TYPE_WKB_VIEW:
      return GeoArrowArrayReaderVisitWKBArrow(&private_data->src.arrow, offset, length,
                                              &private_data->wkb_reader, v);

    case GEOARROW_TYPE_LARGE_WKT:
    case GEOARROW_TYPE_WKT_VIEW:
      return GeoArrowArrayReaderVisitWKTArrow(&private_data->src.arrow, offset, length,
                                              &private_data->wkt_reader, v);

    default:
      return GeoArrowArrayViewVisitNative(&private_data->src.geoarrow, offset, length, v);
  }
}

GeoArrowErrorCode GeoArrowArrayReaderArrayView(struct GeoArrowArrayReader* reader,
                                               const struct GeoArrowArrayView** out) {
  NANOARROW_DCHECK(reader->private_data != NULL);
  struct GeoArrowArrayReaderPrivate* private_data =
      (struct GeoArrowArrayReaderPrivate*)reader->private_data;
  NANOARROW_DCHECK(private_data != NULL);

  switch (private_data->type) {
    case GEOARROW_TYPE_LARGE_WKT:
    case GEOARROW_TYPE_WKT_VIEW:
    case GEOARROW_TYPE_LARGE_WKB:
    case GEOARROW_TYPE_WKB_VIEW:
      return ENOTSUP;
    default:
      *out = &private_data->src.geoarrow;
      return GEOARROW_OK;
  }
}

#include <string.h>

#include "nanoarrow/nanoarrow.h"



struct GeoArrowArrayWriterPrivate {
  struct GeoArrowNativeWriter native_writer;
  struct GeoArrowWKTWriter wkt_writer;
  struct GeoArrowWKBWriter wkb_writer;
  enum GeoArrowType type;
};

GeoArrowErrorCode GeoArrowArrayWriterInitFromType(struct GeoArrowArrayWriter* writer,
                                                  enum GeoArrowType type) {
  struct GeoArrowArrayWriterPrivate* private_data =
      (struct GeoArrowArrayWriterPrivate*)ArrowMalloc(
          sizeof(struct GeoArrowArrayWriterPrivate));

  if (private_data == NULL) {
    return ENOMEM;
  }

  memset(private_data, 0, sizeof(struct GeoArrowArrayWriterPrivate));

  int result;
  switch (type) {
    case GEOARROW_TYPE_LARGE_WKT:
    case GEOARROW_TYPE_LARGE_WKB:
    case GEOARROW_TYPE_WKT_VIEW:
    case GEOARROW_TYPE_WKB_VIEW:
      return ENOTSUP;
    case GEOARROW_TYPE_WKT:
      result = GeoArrowWKTWriterInit(&private_data->wkt_writer);
      break;
    case GEOARROW_TYPE_WKB:
      result = GeoArrowWKBWriterInit(&private_data->wkb_writer);
      break;
    default:
      result = GeoArrowNativeWriterInit(&private_data->native_writer, type);
      break;
  }

  if (result != GEOARROW_OK) {
    ArrowFree(private_data);
    return result;
  }

  private_data->type = type;
  writer->private_data = private_data;
  return GEOARROW_OK;
}

GeoArrowErrorCode GeoArrowArrayWriterInitFromSchema(struct GeoArrowArrayWriter* writer,
                                                    const struct ArrowSchema* schema) {
  struct GeoArrowSchemaView schema_view;
  NANOARROW_RETURN_NOT_OK(GeoArrowSchemaViewInit(&schema_view, schema, NULL));
  return GeoArrowArrayWriterInitFromType(writer, schema_view.type);
}

GeoArrowErrorCode GeoArrowArrayWriterSetPrecision(struct GeoArrowArrayWriter* writer,
                                                  int precision) {
  struct GeoArrowArrayWriterPrivate* private_data =
      (struct GeoArrowArrayWriterPrivate*)writer->private_data;

  if (private_data->type != GEOARROW_TYPE_WKT &&
      private_data->type != GEOARROW_TYPE_LARGE_WKT) {
    return EINVAL;
  }

  private_data->wkt_writer.precision = precision;
  return GEOARROW_OK;
}

GeoArrowErrorCode GeoArrowArrayWriterSetFlatMultipoint(struct GeoArrowArrayWriter* writer,
                                                       int flat_multipoint) {
  struct GeoArrowArrayWriterPrivate* private_data =
      (struct GeoArrowArrayWriterPrivate*)writer->private_data;

  if (private_data->type != GEOARROW_TYPE_WKT &&
      private_data->type != GEOARROW_TYPE_LARGE_WKT) {
    return EINVAL;
  }

  private_data->wkt_writer.use_flat_multipoint = flat_multipoint;
  return GEOARROW_OK;
}

GeoArrowErrorCode GeoArrowArrayWriterInitVisitor(struct GeoArrowArrayWriter* writer,
                                                 struct GeoArrowVisitor* v) {
  struct GeoArrowArrayWriterPrivate* private_data =
      (struct GeoArrowArrayWriterPrivate*)writer->private_data;

  switch (private_data->type) {
    case GEOARROW_TYPE_WKT:
      GeoArrowWKTWriterInitVisitor(&private_data->wkt_writer, v);
      return GEOARROW_OK;
    case GEOARROW_TYPE_WKB:
      GeoArrowWKBWriterInitVisitor(&private_data->wkb_writer, v);
      return GEOARROW_OK;
    default:
      return GeoArrowNativeWriterInitVisitor(&private_data->native_writer, v);
  }
}

GeoArrowErrorCode GeoArrowArrayWriterFinish(struct GeoArrowArrayWriter* writer,
                                            struct ArrowArray* array,
                                            struct GeoArrowError* error) {
  struct GeoArrowArrayWriterPrivate* private_data =
      (struct GeoArrowArrayWriterPrivate*)writer->private_data;

  switch (private_data->type) {
    case GEOARROW_TYPE_WKT:
      return GeoArrowWKTWriterFinish(&private_data->wkt_writer, array, error);
    case GEOARROW_TYPE_WKB:
      return GeoArrowWKBWriterFinish(&private_data->wkb_writer, array, error);
    default:
      return GeoArrowNativeWriterFinish(&private_data->native_writer, array, error);
  }
}

void GeoArrowArrayWriterReset(struct GeoArrowArrayWriter* writer) {
  struct GeoArrowArrayWriterPrivate* private_data =
      (struct GeoArrowArrayWriterPrivate*)writer->private_data;

  if (private_data->wkt_writer.private_data != NULL) {
    GeoArrowWKTWriterReset(&private_data->wkt_writer);
  }

  if (private_data->wkb_writer.private_data != NULL) {
    GeoArrowWKBWriterReset(&private_data->wkb_writer);
  }

  if (private_data->native_writer.private_data != NULL) {
    GeoArrowNativeWriterReset(&private_data->native_writer);
  }

  ArrowFree(private_data);
  writer->private_data = NULL;
}



#include <stdio.h>

#if defined(GEOARROW_USE_RYU) && GEOARROW_USE_RYU

#include "ryu/ryu.h"

int64_t GeoArrowPrintDouble(double f, uint32_t precision, char* result) {
  // Use exponential to serialize very large numbers in scientific notation
  // and ignore user precision for these cases.
  if (f > 1.0e17 || f < -1.0e17) {
    return GeoArrowd2sexp_buffered_n(f, 17, result);
  } else {
    // Note: d2sfixed_buffered_n() may write up to 310 characters into result
    // for the case where f is the minimum possible double value.
    return GeoArrowd2sfixed_buffered_n(f, precision, result);
  }
}

#else

int64_t GeoArrowPrintDouble(double f, uint32_t precision, char* result) {
  // For very large numbers, use scientific notation ignoring user precision
  if (f > 1.0e17 || f < -1.0e17) {
    return snprintf(result, 40, "%0.*e", 16, f);
  }

  int64_t n_chars = snprintf(result, 40, "%0.*f", precision, f);
  if (n_chars > 39) {
    n_chars = 39;
  }

  // Strip trailing zeroes + decimal
  for (int64_t i = n_chars - 1; i >= 0; i--) {
    if (result[i] == '0') {
      n_chars--;
    } else if (result[i] == '.') {
      n_chars--;
      break;
    } else {
      break;
    }
  }

  return n_chars;
}

#endif

#include <errno.h>
#include <stdlib.h>
#include <string.h>



#if !defined(GEOARROW_USE_FAST_FLOAT) || !GEOARROW_USE_FAST_FLOAT

GeoArrowErrorCode GeoArrowFromChars(const char* first, const char* last, double* out) {
  if (first == last) {
    return EINVAL;
  }

  int64_t size_bytes = last - first;

  // There is no guarantee that src.data is null-terminated. The maximum size of
  // a double is 24 characters, but if we can't fit all of src for some reason, error.
  char src_copy[64];
  if (size_bytes >= ((int64_t)sizeof(src_copy))) {
    return EINVAL;
  }

  memcpy(src_copy, first, size_bytes);
  char* last_copy = src_copy + size_bytes;
  *last_copy = '\0';

  char* end_ptr;
  double result = strtod(src_copy, &end_ptr);
  if (end_ptr != last_copy) {
    return EINVAL;
  } else {
    *out = result;
    return GEOARROW_OK;
  }
}

#endif
