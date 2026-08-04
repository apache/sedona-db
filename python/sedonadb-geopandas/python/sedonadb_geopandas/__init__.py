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
"""GeoPandas-compatible API on top of SedonaDB.

**EXPERIMENTAL.** This package provides `GeoDataFrame` / `GeoSeries` wrappers
whose methods mirror GeoPandas but delegate to a lazy SedonaDB engine. It is a
compatibility layer, not a drop-in replacement: see the package README for the
intentional differences (laziness, no row index, immutability). The API may
change without notice.
"""

from sedonadb_geopandas._context import default_context
from sedonadb_geopandas._frame import GeoDataFrame
from sedonadb_geopandas._series import GeoSeries, Series

__all__ = ["GeoDataFrame", "GeoSeries", "Series", "from_geopandas"]


def from_geopandas(data, *, context=None, geometry=None):
    """Load a `geopandas.GeoDataFrame` into a SedonaDB-backed `GeoDataFrame`.

    **EXPERIMENTAL.**

    Args:
        data: A `geopandas.GeoDataFrame` (or any object accepted by
            `SedonaContext.create_data_frame`, which surfaces an error for
            anything that cannot be turned into a DataFrame).
        context: An optional SedonaDB context. Defaults to a shared,
            lazily-created one.
        geometry: The active geometry column name. Defaults to SedonaDB's
            primary-geometry heuristic (the same one `to_geopandas` uses).

    Returns:
        A `GeoDataFrame`.
    """
    ctx = context or default_context()
    df = ctx.create_data_frame(data)
    if geometry is None:
        return GeoDataFrame(df)
    return GeoDataFrame(df, geometry=geometry)
