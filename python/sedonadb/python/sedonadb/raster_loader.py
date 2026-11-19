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

"""Raster loader interface for Python-backed raster data sources."""

from typing import Any, Sequence, Union

from sedonadb._lib import (
    PyChunkRef,
    PyRasterLoadRequest,
    PyRasterLoadResult,
    PyViewEntry,
    py_raster_loader,
)

__all__ = [
    "PyChunkRef",
    "PyRasterLoadRequest",
    "PyRasterLoadResult",
    "PyViewEntry",
    "RasterLoadResult",
    "RasterLoader",
]


class RasterLoadResult:
    """Result of loading raster data.

    This class provides convenience constructors for creating raster load results.
    Use :meth:`unresolved` when returning the full source data (letting the engine
    apply the view), or :meth:`resolved` when returning pre-sliced data.
    """

    @staticmethod
    def unresolved(
        data: Union[bytes, bytearray], request: PyRasterLoadRequest
    ) -> PyRasterLoadResult:
        """Create a result with unresolved view (full source data).

        The returned bytes represent the full source array. The engine will
        apply the view from the request to extract the needed slice.

        Args:
            data: The raw bytes of the full source array.
            request: The original load request (view is preserved).

        Returns:
            A :class:`PyRasterLoadResult` ready to return from :meth:`RasterLoader.load`.
        """
        return PyRasterLoadResult.unresolved(data, request)

    @staticmethod
    def resolved(data: Any, shape: Sequence[int]) -> PyRasterLoadResult:
        """Create a result with resolved view (pre-sliced data).

        The returned bytes represent the final sliced array matching `shape`.
        The view is set to identity (each axis starts at 0, step 1).

        Args:
            data: The raw bytes of the pre-sliced array.
            shape: The shape of the returned array.

        Returns:
            A :class:`PyRasterLoadResult` ready to return from :meth:`RasterLoader.load`.
        """
        # Identity view: each dimension maps 1:1 from source
        view = [
            PyViewEntry(source_axis=i, start=0, step=1, steps=dim)
            for i, dim in enumerate(shape)
        ]
        return PyRasterLoadResult(
            bytes=data,
            source_shape=list(shape),
            view=view,
        )


class RasterLoader:
    """Abstract base class for Python-backed raster loaders.

    This class provides an interface for implementing custom raster data loaders
    that can be registered with a SedonaDB context. Subclasses implement the
    loading logic, and the wrapper is automatically created via
    `__sedonadb_raster_loader__`.

    The loader is invoked during `RS_EnsureLoaded` to materialize out-of-database
    (OutDb) raster references into in-memory data.
    """

    def name(self) -> str:
        """Return the loader name for diagnostics and identification.

        Returns:
            A string identifier for this loader.
        """
        raise NotImplementedError()

    def supports(self, codec_names: Sequence[str]) -> bool:
        """Check whether this loader decodes a band with these codecs.

        An OutDb band carries a `codecs` pipeline (Zarr v3 codec JSON) shared
        by every chunk of the band. The engine matches each band against the
        registered loaders by the pipeline's codec names, in pipeline order,
        most recently registered first; a pipeline no loader claims is an
        error, so nothing guesses.

        Args:
            codec_names: The codec names of the band's pipeline, e.g.
                `["sedona.gdal"]` for a whole file decoded by GDAL,
                `["sedona.zarr"]` for a Zarr chunk, or `["bytes", "zstd"]`
                for a zstd-compressed raw chunk.

        Returns:
            `True` if this loader decodes every codec named, `False` otherwise.
        """
        raise NotImplementedError()

    def load(
        self, requests: Sequence[PyRasterLoadRequest]
    ) -> Sequence[PyRasterLoadResult]:
        """Load raster data for the given requests.

        This method is called to materialize OutDb raster references. Each
        request describes one band: the chunks to fetch, how to decode them,
        the shape, the view, and the data type.

        Args:
            requests: A sequence of request objects, each with:
                - `chunks`: The band's chunk grid in row-major order, one
                  :class:`PyChunkRef` per position (`uri`, `offset`, `size`,
                  `content_type`, `checksum`) or `None` for a missing chunk.
                  A band from `RS_FromPath` or `Raster.lazy` has exactly one
                  chunk, the whole object at its `uri`.
                - `chunk_shape`: Pixels per chunk per dimension
                - `codecs`: The band's codec pipeline as Zarr v3 codec JSON
                - `dim_names`: Dimension names
                - `source_shape`: The shape of the source array
                - `view`: A sequence of view entries describing the slice
                - `data_type`: A data type object with `name` and `byte_size`

        Returns:
            A sequence of results, one per request.
            Use :meth:`RasterLoadResult.unresolved` for simple cases where the
            full source is loaded, or :meth:`RasterLoadResult.resolved` when
            returning pre-sliced data.
        """
        raise NotImplementedError()

    def __sedonadb_raster_loader__(self):
        """Return the internal wrapper for registration with SedonaDB.

        This method is called automatically when registering the loader
        with a SedonaDB context via `sd.register(loader)`.
        """
        return py_raster_loader(
            self.name,
            self.supports,
            self.load,
        )
