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

Activate by calling `register` on a SedonaDB connection. After
registration, two surfaces work:

1. `con.sql("SELECT * FROM sd_read_zarr('s3://...')")` — SQL UDTF.
2. `con.read_format(ZarrFormatSpec(), uri)` — DataFrame API via
   `ExternalFormatSpec`.

```python
import sedonadb
import sedonadb_zarr

con = sedonadb.connect()
sedonadb_zarr.register(con)
con.sql("SELECT count(*) FROM sd_read_zarr('file:///path/to/foo.zarr')").show()
```

Importing `sedonadb_zarr` is opt-in — applications that don't import
it pay no runtime cost.
"""

import json
from typing import Any, Mapping, Optional

from sedonadb.datasource import ExternalFormatSpec

from sedonadb_zarr._lib import PyZarrChunkReader, ZarrTableFunction


def register(con) -> None:
    """Attach Zarr SQL support to a SedonaDB connection.

    After this call, `con.sql("SELECT * FROM sd_read_zarr(...)")` works.
    Idempotent: calling twice re-registers the UDTF without error.

    Args:
        con: A `sedonadb` connection (returned by `sedonadb.connect()`).
    """
    internal_ctx = getattr(con, "_impl", None)
    if internal_ctx is None:
        raise TypeError(
            "sedonadb_zarr.register: could not locate the InternalContext on "
            f"{type(con).__name__}; expected attribute `_impl`."
        )
    internal_ctx.register_udtf_capsule("sd_read_zarr", ZarrTableFunction())


class ZarrFormatSpec(ExternalFormatSpec):
    """`ExternalFormatSpec` for Zarr groups.

    Use with `con.read_format(spec, uri)`:

    ```python
    con.read_format(ZarrFormatSpec(), "file:///path/to/foo.zarr")
    ```

    Supported `with_options` keys:

    - `arrays` (`list[str]` or JSON-string) — explicit subset of group
      arrays to read.
    """

    def __init__(self, options: Optional[Mapping[str, Any]] = None):
        self._options: dict = dict(options) if options else {}

    @property
    def extension(self) -> str:
        return ".zarr"

    @property
    def list_single_object(self) -> bool:
        # Zarr groups are directories; the DataFusion listing layer
        # returns zero objects at a `.zarr` prefix.
        return True

    def with_options(self, options: Mapping[str, Any]) -> "ZarrFormatSpec":
        merged = {**self._options, **options}
        return ZarrFormatSpec(merged)

    def open_reader(self, args: Any) -> PyZarrChunkReader:
        uri = args.src.to_url()
        if uri is None:
            raise ValueError(
                "ZarrFormatSpec: could not resolve a URL from the source object"
            )
        arrays = self._options.get("arrays")
        if isinstance(arrays, str):
            arrays = json.loads(arrays)
        batch_size = args.batch_size if args.batch_size is not None else 8192
        return PyZarrChunkReader(uri, arrays, batch_size)


__all__ = ["register", "ZarrFormatSpec", "PyZarrChunkReader"]
