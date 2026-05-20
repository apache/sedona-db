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
registration, the ``sd_read_zarr`` SQL UDTF reads Zarr groups as
N-D raster columns:

>>> import sedonadb
>>> import sedonadb_zarr
>>> con = sedonadb.connect()
>>> sedonadb_zarr.register(con)
>>> con.sql("SELECT count(*) FROM sd_read_zarr('file:///path/to/foo.zarr')").show()  # doctest: +SKIP

The plugin is opt-in: SedonaDB itself does not bundle Zarr support, so
applications that don't import ``sedonadb_zarr`` pay no zarr build or
runtime cost.
"""

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
    # The `Context` Python object wraps an `InternalContext`; the
    # attribute name follows sedonadb's internal convention.
    internal_ctx = getattr(con, "_impl", None)
    if internal_ctx is None:
        raise TypeError(
            "sedonadb_zarr.register: could not locate the InternalContext on "
            f"{type(con).__name__}; expected attribute `_impl`."
        )
    _register_udtf(internal_ctx)


__all__ = ["register"]
