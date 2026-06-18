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

"""Zarr support for SedonaDB.

```python
import sedona.db
from sedonadb_zarr import ZarrExtension

sd = sedona.db.connect()
sd.register(ZarrExtension())
sd.read("file:///path/to/foo.zarr").show()
```

Importing `sedonadb_zarr` is opt-in — applications that don't import
it pay no runtime cost.
"""

from typing import Any, Mapping, Optional

from sedonadb.context import SedonaContext
from sedonadb.datasource import ExternalFormatSpec
from sedonadb.raster_loader import RasterLoader
from sedonadb.utility import sedona  # noqa: F401

from sedonadb_zarr._lib import (
    PyZarrChunkReader,
    PyZarrRasterLoader,
)


class ZarrExtension:
    """SedonaDB Zarr extension entrypoint

    This interface enables registration of Zarr components with a Python
    SedonaContext.

    Examples:
        >>> from sedonadb_zarr import ZarrExtension
        >>> sd = sedona.db.connect()
        >>> sd.register(ZarrExtension())
    """

    def __sedonadb_extension__(self, sd: SedonaContext, **kwargs) -> None:
        if kwargs:
            raise ValueError("Registration options not supported for ZarrExtension")

        # Register the ZarrRasterLoader
        sd.register(ZarrRasterLoader())

        # Register the Zarr() format as a FileFormatFactory for SQL and .read(..., format="zarr")
        sd.register(Zarr())


class Zarr(ExternalFormatSpec):
    """`ExternalFormatSpec` for Zarr groups.

    This is registered automatically when registering the module with a
    SedonaContext. Use with `sd.read(uri, format=Zarr())` or format="zarr"
    after registering the extension:

    ```python
    sd.read("file:///path/to/foo.zarr", format="zarr")
    ```

    Args:
        options: Supported options include
            - `arrays` (`list[str]`) — explicit subset of group arrays to read.
    """

    _SUPPORTED_OPTIONS = frozenset({"arrays"})

    def __init__(self, options: Optional[Mapping[str, Any]] = None):
        self._options: dict = dict(options) if options else {}

    @property
    def extension(self) -> str:
        return "zarr"

    @property
    def list_single_object(self) -> bool:
        # Zarr groups are directories; the DataFusion listing layer
        # returns zero objects at a `.zarr` prefix.
        return True

    def with_options(self, options: Mapping[str, Any]) -> "Zarr":
        unknown = set(options) - self._SUPPORTED_OPTIONS
        if unknown:
            raise ValueError(
                f"Zarr: unknown option(s) {sorted(unknown)!r}; "
                f"supported: {sorted(self._SUPPORTED_OPTIONS)!r}"
            )
        merged = {**self._options, **options}
        return Zarr(merged)

    def open_reader(self, args: Any) -> PyZarrChunkReader:
        uri = args.src.to_url()
        if uri is None:
            raise ValueError("Zarr: could not resolve a URL from the source object")
        arrays = self._options.get("arrays")
        batch_size = args.batch_size if args.batch_size is not None else 8192
        return PyZarrChunkReader(uri, arrays, batch_size)


class ZarrRasterLoader(RasterLoader):
    """Zarr RasterLoader implementation

    This is registered automatically when registering the ZarrExtension
    and enables RS_EnsureLoaded() can resolve pixels of a Zarr.
    """

    def __init__(self):
        self._impl = PyZarrRasterLoader()

    def name(self):
        return self._impl.name()

    def supports_format(self, format):
        return self._impl.supports_format(format)

    def load(self, requests):
        return self._impl.load(requests)


__all__ = ["Zarr", "ZarrExtension"]
