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

Activate by calling :func:`register` on a SedonaDB connection. After
registration, two surfaces work:

1. ``con.sql("SELECT * FROM sd_read_zarr('s3://...')")`` — SQL UDTF.
2. ``con.read_format(ZarrFormatSpec(), uri)`` — DataFrame API via
   ``ExternalFormatSpec``.

>>> import sedonadb
>>> import sedonadb_zarr
>>> con = sedonadb.connect()
>>> sedonadb_zarr.register(con)
>>> con.sql("SELECT count(*) FROM sd_read_zarr('file:///path/to/foo.zarr')").show()  # doctest: +SKIP

The plugin is opt-in: SedonaDB itself does not bundle Zarr support, so
applications that don't import ``sedonadb_zarr`` pay no zarr build or
runtime cost.
"""

import json
from typing import Any, Mapping, Optional

from sedonadb.datasource import ExternalFormatSpec

from sedonadb_zarr._lib import PyZarrChunkReader
from sedonadb_zarr._lib import register_udtf as _register_udtf


def register(con) -> None:
    """Attach Zarr SQL support to a SedonaDB connection.

    After this call, ``con.sql("SELECT * FROM sd_read_zarr(...)")``
    works. Idempotent — calling twice on the same connection re-
    registers the UDTF without error.

    Parameters
    ----------
    con
        A ``sedonadb`` ``Context`` (the object returned by
        ``sedonadb.connect()``). Internally, this function extracts
        the underlying ``InternalContext`` PyO3 handle and registers
        the UDTF on its DataFusion ``SessionContext``.
    """
    internal_ctx = getattr(con, "_impl", None)
    if internal_ctx is None:
        raise TypeError(
            "sedonadb_zarr.register: could not locate the InternalContext on "
            f"{type(con).__name__}; expected attribute `_impl`."
        )
    _register_udtf(internal_ctx)


class ZarrFormatSpec(ExternalFormatSpec):
    """`ExternalFormatSpec` for Zarr groups.

    Use with ``con.read_format(spec, uri)``:

    >>> con.read_format(ZarrFormatSpec(), "file:///path/to/foo.zarr")  # doctest: +SKIP

    Supported options (via :meth:`with_options`):

    - ``load_eager`` (``bool``) — ``False`` (default) emits chunk-anchor
      URIs only; ``True`` currently errors pending the async resolver.
    - ``arrays`` (``list[str]`` or JSON-string) — explicit subset of
      group arrays to read.
    """

    def __init__(self, options: Optional[Mapping[str, Any]] = None):
        self._options: dict = dict(options) if options else {}

    @property
    def extension(self) -> str:
        return ".zarr"

    def with_options(self, options: Mapping[str, Any]) -> "ZarrFormatSpec":
        merged = {**self._options, **options}
        return ZarrFormatSpec(merged)

    def open_reader(self, args: Any) -> PyZarrChunkReader:
        uri = args.src.to_url()
        if uri is None:
            raise ValueError(
                "ZarrFormatSpec: could not resolve a URL from the source object"
            )
        if self._options.get("load_eager"):
            raise NotImplementedError(
                "ZarrFormatSpec: load_eager=True is not yet supported. "
                "Pixel-byte materialisation lands with the async RS_EnsureLoaded "
                "resolver."
            )
        arrays = self._options.get("arrays")
        if isinstance(arrays, str):
            arrays = json.loads(arrays)
        batch_size = args.batch_size if args.batch_size is not None else 8192
        return PyZarrChunkReader(uri, arrays, batch_size)


__all__ = ["register", "ZarrFormatSpec", "PyZarrChunkReader"]
