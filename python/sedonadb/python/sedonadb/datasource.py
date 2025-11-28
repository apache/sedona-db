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

import sys
from typing import Any, Mapping

from sedonadb._lib import PyExternalFormat, PyProjectedRecordBatchReader


class ExternalFormatSpec:
    """Python file format specification

    This class defines an abstract "file format", which maps to the DataFusion
    concept of a `FileFormat`. This is a layer on top of the `TableProvider` that
    provides standard support for querying collections of files using globs
    or directories of files with compatible schemas. This abstraction allows for
    basic support for pruning and partial filter pushdown (e.g., a bounding box
    is available if one was provided in the underlying query); however, data
    providers with more advanced features may wish to implement a `TableProvider`
    in Rust to take advantage of a wider range of DataFusion features.

    Implementations are only required to implement `open_reader()`; however, if
    opening a reader is expensive and there is a more efficient way to infer a
    schema from a given source, implementers may wish to also implement
    `infer_schema()`.

    This extension point is experimental and may evolve to serve the needs of
    various file formats.
    """

    @property
    def extension(self):
        """A file extension for files that match this format

        If this concept is not important for this format, returns an empty string.
        """
        return ""

    def with_options(self, options: Mapping[str, Any]):
        """Clone this instance and return a new instance with options applied

        Apply an arbitrary set of format-defined key/value options. It is useful
        to raise an error in this method if an option or value will later result
        in an error; however, implementation may defer the error until later if
        required by the underlying producer.

        The default implementation of this method errors for any attempt to
        pass options.
        """
        raise NotImplementedError(
            f"key/value options not supported by {type(self).__name__}"
        )

    def open_reader(self, args: Any):
        """Open an ArrowArrayStream/RecordBatchReader of batches given input information

        Note that the output stream must take into account `args.file_projection`, if one
        exists (`PyProjectedRecordBatchReader` may be used to ensure a set of output
        columns or apply an output projection on an input stream.

        The internals will keep a strong (Python) reference to the returned object
        for as long as batches are being produced.

        Args:
            args: An object with attributes
                - `src`: An object/file abstraction. Currently, `.to_url()` is the best way
                  to extract the underlying URL from the source.
                - `filter`: An object representing the filter expression that was pushed
                  down, if one exists. Currently, `.bounding_box(column_index)` is the only
                  way to interact with this object.
                - `file_schema`: An optional schema. If `None`, the implementation must
                  infer the schema.
                - `file_projection`: An optional list of integers of the columns of
                  `file_schema` that must be produced by this implementation (in the
                  exact order specified).
                - `batch_size`: An optional integer specifying the number of rows requested
                  for each output batch.

        """
        raise NotImplementedError()

    def infer_schema(self, src):
        """Infer the output schema

        Implementations can leave this unimplemented, in which case the internals will call
        `open_reader()` and query the provided schema without pulling any batches.

        Args:
            src: An object/file abstraction. Currently, `.to_url()` is the best way
                to extract the underlying URL from the source.
        """
        raise NotImplementedError()

    def __sedona_external_format__(self):
        return PyExternalFormat(self)


class PyogrioFormatSpec(ExternalFormatSpec):
    """An `ExternalFormatSpec` implementation wrapping GDAL/OGR via pyogrio"""

    def __init__(self, extension=""):
        self._extension = extension
        self._options = {}

    def with_options(self, options):
        cloned = type(self)(self.extension)
        cloned._options.update(options)
        return cloned

    @property
    def extension(self) -> str:
        return self._extension

    def open_reader(self, args):
        import pyogrio.raw

        url = args.src.to_url()
        if url is None:
            raise ValueError(f"Can't convert {args.src} to OGR-openable object")

        if url.startswith("http://") or url.startswith("https://"):
            ogr_src = f"/vsicurl/{url}"
        elif url.startswith("file://") and sys.platform != "win32":
            ogr_src = url.removeprefix("file://")
        elif url.startswith("file:///"):
            ogr_src = url.removeprefix("file:///")
        else:
            raise ValueError(f"Can't open {url} with OGR")

        if ogr_src.endswith(".zip"):
            ogr_src = f"/vsizip/{ogr_src}"

        if args.is_projected():
            file_columns = args.file_schema.names
            columns = [file_columns[i] for i in args.file_projection]
        else:
            columns = None

        batch_size = args.batch_size if args.batch_size is not None else 0

        if args.filter and args.file_schema is not None:
            geometry_column_indices = args.file_schema.geometry_column_indices
            file_columns = args.file_schema.names
            if len(geometry_column_indices) == 1:
                bbox = args.filter.bounding_box(
                    file_columns[geometry_column_indices[0]]
                )
            else:
                bbox = None
        else:
            bbox = None

        return PyogrioReaderShelter(
            pyogrio.raw.ogr_open_arrow(
                ogr_src, {}, columns=columns, batch_size=batch_size, bbox=bbox
            ),
            columns,
        )


class PyogrioReaderShelter:
    """Python object wrapper around the context manager returned by pyogrio

    The pyogrio object returned by `pyogrio.raw.ogr_open_arrow()` is a context
    manager; however, the internals can only manage Rust object references.
    This object ensures that the context manager is closed when the object
    is deleted (which occurs as soon as possible when the returned reader
    is no longer required).
    """

    def __init__(self, inner, output_names=None):
        self._inner = inner
        self._output_names = output_names
        self._meta, self._reader = self._inner.__enter__()

    def __del__(self):
        self._inner.__exit__(None, None, None)

    def __arrow_c_stream__(self, requested_schema=None):
        if self._output_names is None:
            return self._reader.__arrow_c_stream__()
        else:
            projected = PyProjectedRecordBatchReader(
                self._reader, None, self._output_names
            )
            return projected.__arrow_c_stream__()
