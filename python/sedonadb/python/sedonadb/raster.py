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

        This operation is zero-copy for contiguous arrays.

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
            data=array,
            crs=crs,
            nodata=nodata,
            transform=transform,
        )

    def __init__(self, array, i=0):
        """Create a Raster from an Arrow array at index i."""
        # It is convenient to handle this case because this is what to_arrow_table()
        # returns.
        if isinstance(array, pa.ChunkedArray):
            for chunk in array.chunks:
                if i < len(chunk):
                    chunk = (
                        chunk.storage if isinstance(chunk, pa.ExtensionArray) else chunk
                    )
                    self._array = chunk.slice(i, 1)
                    return
                i -= len(chunk)
            raise IndexError("Index out of bounds for chunked array")

        if isinstance(array, pa.ExtensionArray):
            array = array.storage

        # Use slice directly - pa.array() would copy
        self._array = array.slice(i, 1)

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
        # Use slice directly - pa.array() would copy
        self._array = array.slice(i, 1)

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
        """The raw source data buffer as a memoryview.

        Zero-copy for out-of-line BinaryView data by decoding the view descriptor
        and resolving the correct variadic buffer. Falls back to copying for
        inline data (≤12 bytes).
        """
        # Try zero-copy path for out-of-line data
        data_array = self._array.field("data")
        result = _get_binary_view_buffer(data_array, index=0)
        if result is not None:
            return result

        # Fallback: inline data or missing buffer - must copy
        return memoryview(data_array[0].as_buffer())

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

        # Shapes that contain zeroes are supported by memroyview.cast().
        # Callers should check data_size for empty handling with non-numpy views
        # if they want to avoid a numpy dependency.
        if self.data_size == 0:
            return memoryview(self.to_numpy())

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
        """Convert this band's data to a numpy array (zero-copy when possible)."""
        import numpy as np

        if self.data_size == 0:
            return np.empty(self.shape, dtype=self.data_type)

        return np.array(self.data, dtype=self.data_type, copy=False).reshape(self.shape)

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

    def to_pandas(self, **kwargs):
        """Convert to a Pandas Series of Raster objects (zero-copy where possible)."""
        import pandas as pd

        _check_pandas_version()

        # Create Raster objects that reference the underlying storage directly
        rasters = [Raster(self.storage, i) for i in range(len(self))]
        return pd.array(rasters, dtype=object)


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

    def to_pandas_dtype(self):
        """Return the Pandas dtype for this extension type."""
        return RasterDtype()


class RasterDtype:
    """Pandas-compatible dtype for Raster arrays.

    This dtype enables conversion from Arrow to Pandas with zero-copy
    Raster objects that reference the underlying Arrow buffers.
    """

    name = "raster"
    na_value = None

    def __repr__(self):
        return "RasterDtype()"

    def __eq__(self, other):
        return isinstance(other, RasterDtype)

    def __hash__(self):
        return hash("RasterDtype")

    @classmethod
    def __from_arrow__(cls, arr):
        """Convert an Arrow array to a numpy array of Raster objects.

        This is called by PyArrow when converting to Pandas. Each Raster
        object wraps a slice of the Arrow array for zero-copy access.
        """
        import pandas as pd

        _check_pandas_version()

        # Handle ChunkedArray by iterating over chunks
        if isinstance(arr, pa.ChunkedArray):
            rasters = []
            for chunk in arr.chunks:
                storage = (
                    chunk.storage if isinstance(chunk, pa.ExtensionArray) else chunk
                )
                for i in range(len(storage)):
                    rasters.append(Raster(storage, i))
            return pd.array(rasters, dtype=object)

        # Handle single array
        storage = arr.storage if isinstance(arr, pa.ExtensionArray) else arr
        rasters = [Raster(storage, i) for i in range(len(storage))]
        return pd.array(rasters, dtype=object)


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


def _get_binary_view_buffer(
    data_array: pa.BinaryViewArray, index: int = 0
) -> Optional[memoryview]:
    """Extract a zero-copy memoryview from a BinaryViewArray element.

    Decodes the BinaryView format to resolve the correct variadic buffer
    for out-of-line data. Returns None for inline data (≤12 bytes) which
    requires copying. This in theory should be possible with a normal
    pyarrow array[index].as_py().as_buffer(); however, something about
    this operation forces a copy with the current PyArrow version.

    Args:
        data_array: A BinaryViewArray (may be sliced).
        index: The element index within the array (after any slicing).

    Returns:
        A memoryview pointing directly into the variadic buffer for out-of-line
        data, or None if the data is inline and must be copied.
    """
    # BinaryView layout: [validity, views, variadic_buffers...]
    # Each view is 16 bytes:
    #   - Inline (len ≤ 12): length:i4, data:12bytes
    #   - Out-of-line (len > 12): length:i4, prefix:4bytes, buf_idx:i4, offset:i4
    buffers = data_array.buffers()
    if len(buffers) < 2 or buffers[1] is None:
        return None

    views_buf = memoryview(buffers[1])
    # Account for array offset (e.g., from slicing) plus the requested index
    array_offset = data_array.offset + index
    view_start = array_offset * 16
    view_end = view_start + 16
    if view_end > len(views_buf):
        raise IndexError(
            f"index {index} is out of bounds for BinaryViewArray with "
            f"{len(views_buf) // 16} elements"
        )
    view_bytes = views_buf[view_start:view_end]

    # Decode length from first 4 bytes
    length = struct.unpack_from("=I", view_bytes, 0)[0]

    if length <= 12:
        # Inline data - caller must copy
        return None

    # Out-of-line: decode buffer_index and offset
    buf_idx = struct.unpack_from("=I", view_bytes, 8)[0]
    offset = struct.unpack_from("=I", view_bytes, 12)[0]

    # Variadic buffers start at index 2
    variadic_buf_idx = 2 + buf_idx
    if variadic_buf_idx >= len(buffers) or buffers[variadic_buf_idx] is None:
        raise IndexError(
            f"BinaryView references buffer index {buf_idx} but only "
            f"{len(buffers) - 2} variadic buffers are available"
        )

    data_buf = memoryview(buffers[variadic_buf_idx])
    return data_buf[offset : offset + length]


def _wrap_data_zero_copy(data) -> pa.BinaryViewArray:
    """Wrap data as a single-element BinaryViewArray without copying.

    For numpy arrays and memoryviews, this creates a zero-copy view.
    For bytes, pyarrow handles the wrapping efficiently.
    """
    import numpy as np

    view = memoryview(data)
    if not view.c_contiguous:
        # Must copy if not contiguous
        data = np.ascontiguousarray(data)
        view = memoryview(data)

    nbytes = view.nbytes
    if nbytes == 0:
        return pa.array([b""], type=pa.binary_view())
    elif nbytes >= 2**31:
        raise ValueError(f"Can't store {nbytes} (>2GB) in a single raster band")

    # Build BinaryArray zero-copy using from_buffers, then cast to binary_view
    # BinaryArray buffers: [validity, offsets, data]
    offsets = pa.py_buffer(np.array([0, nbytes], dtype=np.int32))
    data_buffer = pa.py_buffer(view)
    binary_array = pa.BinaryArray.from_buffers(
        pa.binary(),
        length=1,
        buffers=[None, offsets, data_buffer],
    )

    # Cast to binary_view is zero-copy at array buffer level
    return binary_array.cast(pa.binary_view())


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

    # Build band struct array with zero-copy data field
    band_data_array = _wrap_data_zero_copy(data)

    band_struct = pa.StructArray.from_arrays(
        [
            pa.array([None], type=pa.utf8()),  # name
            pa.array([list(dim_names)], type=pa.list_(pa.utf8())),  # dim_names
            pa.array([list(shape)], type=pa.list_(pa.int64())),  # source_shape
            pa.array([data_type_id], type=pa.uint32()),  # data_type
            pa.array([nodata_bytes], type=pa.binary()),  # nodata
            pa.array(
                [None],
                type=pa.list_(
                    pa.struct(
                        [  # view
                            pa.field("source_axis", pa.int64()),
                            pa.field("start", pa.int64()),
                            pa.field("step", pa.int64()),
                            pa.field("steps", pa.int64()),
                        ]
                    )
                ),
            ),
            pa.array([outdb_uri], type=pa.utf8()),  # outdb_uri
            pa.array([outdb_format], type=pa.string_view()),  # outdb_format
            band_data_array,  # data (zero-copy)
        ],
        names=[
            "name",
            "dim_names",
            "source_shape",
            "data_type",
            "nodata",
            "view",
            "outdb_uri",
            "outdb_format",
            "data",
        ],
    )

    # Wrap band in a list array (single element list containing one band)
    bands_list = pa.ListArray.from_arrays(
        pa.array([0, 1], type=pa.int32()),
        band_struct,
    )

    transform_values = (
        list(transform) if transform is not None else [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]
    )

    # Build raster struct array
    raster_struct = pa.StructArray.from_arrays(
        [
            pa.array([crs], type=pa.string_view()),  # crs
            pa.array([transform_values], type=pa.list_(pa.float64())),  # transform
            pa.array(
                [[x_name, y_name]], type=pa.list_(pa.string_view())
            ),  # spatial_dims
            pa.array([[width, height]], type=pa.list_(pa.int64())),  # spatial_shape
            bands_list,  # bands
        ],
        names=["crs", "transform", "spatial_dims", "spatial_shape", "bands"],
    )

    # Cast to the canonical storage type to ensure nullability matches
    raster_struct = raster_struct.cast(RASTER_STORAGE_TYPE)

    return Raster(raster_struct)


def _check_pandas_version():
    """Check that pandas version is >= 3.0 for raster to_pandas conversion.

    Older Pandas versions have an issue with Arrow conversion for the
    current strategy of Raster array -> Pandas conversion. This may
    be able to be worked around (the original error was a BlockManager
    error regarding a 1D/2D block).

    Raises:
        ImportError: If pandas version is less than 3.0.
    """
    import pandas as pd
    from packaging.version import Version

    if Version(pd.__version__) < Version("3.0"):
        raise ImportError(
            f"Converting raster columns to pandas requires pandas >= 3.0, "
            f"but found pandas {pd.__version__}. Use .to_arrow_table() instead."
        )
