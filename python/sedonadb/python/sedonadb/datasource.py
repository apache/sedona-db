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

from sedonadb._lib import PyExternalFormat, PyProjectedRecordBatchReader


class ExternalFormatSpec:
    def clone(self):
        raise NotImplementedError()

    @property
    def extension(self):
        return ""

    def with_options(self, options):
        raise NotImplementedError(
            f"key/value options not supported by {type(self).__name__}"
        )

    def open_reader(self, args):
        raise NotImplementedError()

    def infer_schema(self, object):
        raise NotImplementedError()

    def __sedona_external_format__(self):
        return PyExternalFormat(self)


class PyogrioFormatSpec(ExternalFormatSpec):
    def __init__(self, extension=""):
        self._extension = extension
        self._options = {}

    def clone(self):
        cloned = type(self)(self.extension)
        cloned._options.update(self._options)
        return cloned

    def with_options(self, options):
        cloned = self.clone()
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
        elif url.startswith("file://"):
            ogr_src = url.removeprefix("file://")
        else:
            raise ValueError(f"Can't open {url} with OGR")

        if ogr_src.endswith(".zip"):
            ogr_src = f"/vsizip/{ogr_src}"

        if args.is_projected():
            file_names = args.file_schema.names
            columns = [file_names[i] for i in args.file_projection]
        else:
            columns = None

        batch_size = args.batch_size if args.batch_size is not None else 0

        if args.filter and args.file_schema is not None:
            geometry_column_indices = args.file_schema.geometry_column_indices
            if len(geometry_column_indices) == 1:
                bbox = args.filter.bounding_box(geometry_column_indices[0])
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
