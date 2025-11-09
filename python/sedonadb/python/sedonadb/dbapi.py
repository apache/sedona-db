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

from typing import Mapping, Any

import adbc_driver_manager.dbapi

import sedonadb.adbc
from sedonadb.utility import sedona  # noqa: F401


def connect(**kwargs: Mapping[str, Any]) -> "Connection":
    """Connect to Sedona via Python DBAPI

    Creates a DBAPI-compatible connection as a thin wrapper around the
    ADBC Python driver manager's DBAPI compatibility layer. Support for
    DBAPI is experimental.

    Args:
        kwargs: Extra keyword arguments passed to
            `adbc_driver_manager.dbapi.Connection()`.

    Examples:

        >>> con = sedona.dbapi.connect()
        >>> with con.cursor() as cur:
        ...     cur.execute("SELECT 1 as one")
        ...     cur.fetchall()
        [(1,)]
    """
    db = None
    conn = None

    try:
        db = sedonadb.adbc.connect()
        conn = adbc_driver_manager.AdbcConnection(db)
        return adbc_driver_manager.dbapi.Connection(db, conn, **kwargs)
    except Exception:
        if conn:
            conn.close()
        if db:
            db.close()
        raise


Connection = adbc_driver_manager.dbapi.Connection
Cursor = adbc_driver_manager.dbapi.Cursor
