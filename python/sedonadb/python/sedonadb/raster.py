# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import struct
import math

from typing import List, Optional, TYPE_CHECKING, Tuple, Any, Iterable
import geoarrow.types as gat
import pyarrow as pa

from sedonadb._lib import raster_type

if TYPE_CHECKING:
    import numpy as np

EXTENSION_NAME = "sedona.raster"

# Band data type IDs (matches sedona-schema BandDataType enum discriminants)
BAND_DATA_TYPES = {
    1: "UInt8",
    2: "UInt16",
    3: "Int16",
    4: "UInt32",
    5: "Int32",
    6: "Float32",
    7: "Float64",
    8: "UInt64",
    9: "Int64",
    10: "Int8",
}

BAND_DATA_TYPE_IDS = {v.lower(): k for k, v in BAND_DATA_TYPES.items()}

# Python struct module format characters for band data types
BAND_DATA_TYPE_STRUCT_CHARS = {
    1: "B",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "d",
    8: "Q",
    9: "q",
    10: "b",
}


def _resolve_dim_names(ndim, dim_names):
    """Resolve a band's dimension names. The trailing two are the spatial
    ``(y, x)`` pair; a 2-D raster defaults to ``["y", "x"]`` and higher
    dimensionalities must name every axis explicitly."""
    if dim_names is None:
        if ndim == 2:
            return ["y", "x"]
        raise ValueError(
            f"dim_names is required for a {ndim}-dimensional raster "
            "(only 2-D defaults to ['y', 'x'])"
        )
    dim_names = list(dim_names)
    if len(dim_names) != ndim:
        raise ValueError(
            f"dim_names has {len(dim_names)} entries but the raster has {ndim} dimensions"
        )
    return dim_names


def _build_raster(
    dim_names,
    shape,
    data_type_id,
    *,
    data,
    crs=None,
    nodata=None,
    transform=None,
    outdb_uri=None,
    outdb_format=None,
):
    """Assemble a single-band raster from its components, shared by the
    `Raster.lazy` (OutDb) and `Raster.from_numpy` (InDb) constructors. The
    trailing two `dim_names`/`shape` entries are the spatial `(y, x)` axes."""
    if crs is not None:
        crs = gat.type_spec(crs=crs).crs.to_json()
    nodata_bytes = None
    if nodata is not None:
        nodata_bytes = struct.pack(
            "<" + BAND_DATA_TYPE_STRUCT_CHARS[data_type_id], nodata
        )

    # spatial_dims / spatial_shape reference the trailing (y, x) axes, in x,y order.
    y_name, x_name = dim_names[-2], dim_names[-1]
    height, width = shape[-2], shape[-1]

    band = {
        "name": None,
        "dim_names": list(dim_names),
        "source_shape": list(shape),
        "data_type": data_type_id,
        "nodata": nodata_bytes,
        "view": None,
        "outdb_uri": outdb_uri,
        "outdb_format": outdb_format,
        "data": data,
    }
    raster = {
        "crs": crs,
        "transform": list(transform)
        if transform is not None
        else [0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
        "spatial_dims": [x_name, y_name],
        "spatial_shape": [width, height],
        "bands": [band],
    }
    storage_type = pa.DataType._import_from_c_capsule(
        raster_type().__arrow_c_schema__()
    )
    return Raster(pa.array([raster], type=storage_type))


class Raster:
    """Python representation of a sedona.raster scalar value."""

    @staticmethod
    def lazy(
        uri: str,
        shape: Iterable[int],
        dtype: str,
        *,
        format: Optional[str] = None,
        crs: Any = None,
        dim_names: Optional[Iterable[str]] = None,
    ) -> "Raster":
        """Create a lazy raster that references external data without loading it.

        This creates a raster with a single band that points to an external data
        source (e.g., a file on disk or cloud storage) without reading the actual
        pixel data into memory.

        Args:
            uri: The URI of the external data source (e.g., file path or cloud URL).
                This URI is not validated unless the pixels are read.
            shape: The shape of the raster. The trailing two axes are the spatial
                `(y, x)` pair; any leading axes are extra (e.g. time) dimensions.
            dtype: The pixel data type (e.g., 'uint8', 'float32', 'int16').
            format: The format of the external data (e.g., 'tif', or 'zarr'). If None,
                the format will be inferred when the data is accessed.
            crs: The coordinate reference system. Can be any value accepted by
                geoarrow.types.type_spec (e.g., string, pyproj.CRS).
            dim_names: Names of the dimensions. Defaults to `["y", "x"]` for a 2-D
                shape; required when the shape has more than two dimensions.

        Returns:
            A new Raster instance with a single band referencing the external data.

        Examples:
            >>> raster = Raster.lazy("s3://bucket/image.tif", (1024, 2048), "uint8")
            >>> cube = Raster.lazy(
            ...     "s3://bucket/cube.zarr", (12, 1024, 2048), "float32",
            ...     dim_names=["time", "y", "x"],
            ... )
        """
        shape = list(shape)
        if len(shape) < 2:
            raise ValueError("lazy() requires at least two (y, x) dimensions")
        dim_names = _resolve_dim_names(len(shape), dim_names)

        dtype = dtype.lower()
        if dtype not in BAND_DATA_TYPE_IDS:
            raise ValueError(f"Unsupported raster dtype: {dtype}")

        return _build_raster(
            dim_names,
            shape,
            BAND_DATA_TYPE_IDS[dtype],
            data=b"",
            crs=crs,
            outdb_uri=uri,
            outdb_format=format,
        )

    @staticmethod
    def from_numpy(
        array: "np.ndarray",
        *,
        dim_names: Optional[Iterable[str]] = None,
        crs: Any = None,
        nodata: Any = None,
        transform: Optional[Iterable[float]] = None,
    ) -> "Raster":
        """Create an in-database raster from a NumPy array, holding pixels inline.

        The mirror of `lazy`: identical dimension/CRS conventions, but the
        pixel data is carried in the raster rather than referenced by URI.

        **Warning:** this copies the *entire* array into the raster value; it
        is not zero-copy.

        Args:
            array: The pixel data. The trailing two axes are the spatial `(y, x)`
                pair; any leading axes are extra (e.g. time) dimensions. The dtype
                must be a supported raster type (uint8, int16, float32, ...).
            dim_names: Names of the dimensions. Defaults to `["y", "x"]` for a 2-D
                array; required when the array has more than two dimensions.
            crs: The coordinate reference system (any value accepted by
                geoarrow.types.type_spec).
            nodata: Optional nodata sentinel, packed in the array's dtype.
            transform: Optional GDAL-order geotransform
                `[origin_x, scale_x, skew_x, origin_y, skew_y, scale_y]`;
                defaults to a north-up identity.

        Returns:
            A new Raster instance with a single in-database band.

        Examples:
            >>> import numpy as np
            >>> cube = Raster.from_numpy(
            ...     np.arange(2 * 3 * 4, dtype="uint8").reshape(2, 3, 4),
            ...     dim_names=["time", "y", "x"],
            ... )
        """
        shape = list(array.shape)
        if len(shape) < 2:
            raise ValueError(
                "from_numpy() requires an array with at least two (y, x) dimensions"
            )
        dim_names = _resolve_dim_names(len(shape), dim_names)

        dtype = str(array.dtype)
        if dtype not in BAND_DATA_TYPE_IDS:
            raise ValueError(f"Unsupported raster dtype: {dtype}")

        return _build_raster(
            dim_names,
            shape,
            BAND_DATA_TYPE_IDS[dtype],
            data=array.tobytes(),
            crs=crs,
            nodata=nodata,
            transform=transform,
        )

    def __init__(self, array, i=0):
        """Create a Raster from an Arrow array at index i."""
        if isinstance(array, pa.ExtensionArray):
            array = array.storage

        self._array = pa.array(array.slice(i, i + 1))

    def _py_field(self, k):
        """Extract a field value as a Python object."""
        return self._array.field(k)[0].as_py()

    @property
    def crs(self) -> gat.Crs:
        """The coordinate reference system of this raster."""
        return gat.type_spec(crs=self._py_field("crs")).crs

    @property
    def width(self) -> int:
        """The width of this raster in pixels."""
        return self._py_field("spatial_shape")[0]

    @property
    def height(self) -> int:
        """The height of this raster in pixels."""
        return self._py_field("spatial_shape")[1]

    @property
    def transform(self) -> List[float]:
        """The affine transform coefficients for this raster."""
        return self._py_field("transform")

    @property
    def bands(self) -> List["Band"]:
        """The list of bands in this raster."""
        bands_array = self._array.field("bands").flatten()
        return [Band(bands_array, i) for i in range(len(bands_array))]

    def __repr__(self) -> str:
        """Return a string representation of this raster."""
        return f"<Raster {self.width}x{self.height}, {len(self.bands)} band(s)>"

    def __arrow_c_array__(self, requested_schema=None):
        """Implement the array protocol so this works with lit()"""
        extension_type = RasterType(self._array.type)
        extension_array = extension_type.wrap_array(self._array)
        return extension_array.__arrow_c_array__(requested_schema=requested_schema)


class Band:
    """Python representation of a raster band."""

    def __init__(self, array, i=0):
        """Create a Band from an Arrow array at index i."""
        self._array = pa.array(array.slice(i, i + 1))

    def _py_field(self, k):
        """Extract a field value as a Python object."""
        return self._array.field(k)[0].as_py()

    @property
    def name(self) -> Optional[str]:
        """The name of this band, if any."""
        return self._py_field("name")

    @property
    def shape(self) -> Tuple[int, ...]:
        """The shape of this band's data after applying any views."""
        views = self._py_field("view")
        if views:
            raise NotImplementedError("Lazy views are not yet supported")

        return self.source_shape

    @property
    def source_shape(self) -> Tuple[int, ...]:
        """The shape of this band's source data."""
        return tuple(self._py_field("source_shape"))

    @property
    def outdb_uri(self) -> Optional[str]:
        """The URI for out-of-database storage, if any."""
        return self._py_field("outdb_uri")

    @property
    def data_type(self) -> str:
        """The pixel data type name (e.g., 'uint8', 'float32')."""
        type_id = self._py_field("data_type")
        return BAND_DATA_TYPES[type_id].lower()

    @property
    def source_data(self) -> memoryview:
        """The raw source data buffer as a memoryview."""
        view_scalar = self._array.field("data")[0]
        return memoryview(view_scalar.as_buffer())

    @property
    def source_data_size(self) -> int:
        """The number of bytes consumed by soure_data if it were loaded"""
        buffer_type_id = self._py_field("data_type")
        buffer_type_char = BAND_DATA_TYPE_STRUCT_CHARS[buffer_type_id]
        element_size = struct.calcsize(buffer_type_char)
        return math.prod(self.source_shape) * element_size

    @property
    def data_size(self) -> int:
        """The number of bytes consumed by data if it were loaded"""
        buffer_type_id = self._py_field("data_type")
        buffer_type_char = BAND_DATA_TYPE_STRUCT_CHARS[buffer_type_id]
        element_size = struct.calcsize(buffer_type_char)
        return math.prod(self.shape) * element_size

    @property
    def data(self) -> memoryview:
        """The band data as a typed, shaped memoryview."""
        buffer_type_id = self._py_field("data_type")
        buffer_type_char = BAND_DATA_TYPE_STRUCT_CHARS[buffer_type_id]

        # This is not quite right, but shapes that contain zeroes are not well
        # supported by the memoryview yet. Callers should check data_size for
        # empty handling with non-numpy views.
        if self.data_size == 0:
            return memoryview(b"")

        source_data = self.source_data
        if self.outdb_uri is not None and len(source_data) == 0:
            raise ValueError("Can't extract buffer from a reference to external data.")

        # When views are supported, we would need to calculate the striding
        # to export a zero copy view.
        views = self._py_field("view")
        if views:
            raise NotImplementedError("Lazy views are not yet supported")

        return self.source_data.cast(buffer_type_char, self.shape)

    def to_numpy(self) -> "np.ndarray":
        """Convert this band's data to a numpy array."""
        import numpy as np

        if self.data_size == 0:
            return np.empty(self.shape, dtype=self.data_type)

        return np.array(self.data)

    def __repr__(self) -> str:
        """Return a string representation of this band."""
        name_part = f" {self.name!r}" if self.name else ""
        return f"<Band{name_part} {self.data_type} {'x'.join(map(str, self.shape))}>"


class RasterScalar(pa.ExtensionScalar):
    """Scalar type for sedona.raster extension arrays."""

    def as_py(self):
        return Raster(pa.array([self.value]))


class RasterArray(pa.ExtensionArray):
    """Array type for sedona.raster extension arrays."""

    pass


class RasterType(pa.ExtensionType):
    """PyArrow extension type for sedona.raster.

    This extension type wraps a struct storage type representing raster data.
    """

    def __init__(self, storage_type: Any = None):
        """Create a RasterType with the given storage type.

        Parameters
        ----------
        storage_type : pa.DataType
            The underlying Arrow storage type (must be a struct type).
        """
        if storage_type is None:
            storage_type = RASTER_STORAGE_TYPE

        if not pa.types.is_struct(storage_type):
            raise TypeError(f"storage_type must be a struct type, not {storage_type}")
        super().__init__(storage_type, EXTENSION_NAME)

    def __arrow_ext_serialize__(self) -> bytes:
        """Serialize extension type metadata."""
        return b""

    @classmethod
    def __arrow_ext_deserialize__(
        cls, storage_type: pa.DataType, serialized: bytes
    ) -> "RasterType":
        return RasterType(storage_type)

    def __arrow_ext_class__(self):
        return RasterArray

    def __arrow_ext_scalar_class__(self):
        return RasterScalar


def register_extension_type():
    """Register the sedona.raster extension type with PyArrow.

    This should be called once at module initialization to enable
    automatic deserialization of sedona.raster arrays from IPC.
    """
    # Create a dummy storage type for registration - the actual storage
    # type will be determined during deserialization
    dummy_storage = pa.struct([("_placeholder", pa.int32())])
    try:
        pa.register_extension_type(RasterType(dummy_storage))
    except pa.ArrowKeyError:
        # Already registered
        pass


# The storage type for a raster. To ensure we get this exactly right,
# import from Rust via the Arrow PyCapsule interface.
RASTER_STORAGE_TYPE = pa.DataType._import_from_c_capsule(
    raster_type().__arrow_c_schema__()
)
