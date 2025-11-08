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

from sedonadb._lib import PyExternalFormat


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
        import pyogrio.raw

        self._raw = pyogrio.raw
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
        url = args.src.to_url()
        if url is None:
            raise ValueError(f"Can't convert {args.src} to OGR-openable object")

        if url.startswith("http://") or url.startswith("https://"):
            ogr_src = f"/vsicurl/{url}"
        elif url.startswith("file://"):
            ogr_src = url.removeprefix("file://")
        else:
            raise ValueError(f"Can't open {url} with OGR")

        return PyogrioReaderShelter(self._raw.ogr_open_arrow(ogr_src, {}))


class PyogrioReaderShelter:
    def __init__(self, inner):
        self._inner = inner
        self._meta, self._reader = self._inner.__enter__()

    def __del__(self):
        self._inner.__exit__(None, None, None)

    def __arrow_c_stream__(self, requested_schema=None):
        return self._reader.__arrow_c_stream__()
