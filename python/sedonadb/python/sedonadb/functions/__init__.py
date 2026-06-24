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

from functools import cached_property
from typing import TYPE_CHECKING, Union

from sedonadb._lib import SedonaError
from sedonadb.expr.expression import ScalarUdf, AggregateUdf

if TYPE_CHECKING:
    from sedonadb.functions.table import TableFunctions
    from sedonadb.expr.expression import Expr
    from sedonadb_expr import GeoFunctions, RasterFunctions


class Functions:
    """Functions accessor

    This class provides Pythonic wrappers to call SedonaDB functions
    given a specific SedonaDB context.
    """

    def __init__(self, ctx, expr=None):
        self._ctx = ctx
        self._expr = expr

    @cached_property
    def table(self) -> "TableFunctions":
        """Access SedonaDB Table functions"""
        from sedonadb.functions.table import TableFunctions

        if self._expr is not None:
            raise ValueError("Expr piping into table functions is not supported")

        return TableFunctions(self._ctx)

    @property
    def geo(self) -> "GeoFunctions[Expr]":
        from sedonadb_expr import GeoFunctions

        return GeoFunctions(self)

    @property
    def rst(self) -> "RasterFunctions[Expr]":
        from sedonadb_expr import RasterFunctions

        return RasterFunctions(self)

    def __getattr__(self, name) -> Union["ScalarUdf", "AggregateUdf"]:
        try:
            return ScalarUdf(self._ctx._impl.scalar_udf(name), self._ctx, self._expr)
        except SedonaError:
            pass

        try:
            return AggregateUdf(
                self._ctx._impl.aggregate_udf(name), self._ctx, self._expr
            )
        except SedonaError:
            pass

        raise AttributeError(f"Can't find scalar or aggregate function '{name}'")

    def __getitem__(self, key) -> Union["ScalarUdf", "AggregateUdf"]:
        try:
            return self.__getattr__(key)
        except AttributeError:
            raise KeyError(f"Can't find scalar or aggregate function '{key}'")

    def __dir__(self):
        return (
            self._ctx._impl.list_scalar_udfs()
            + self._ctx._impl.list_aggregate_udfs()
            + super().__dir__()
        )

    def _ipython_key_completions_(self):
        """Enable tab completion for f["name"] in IPython/Jupyter."""
        return (
            self._ctx._impl.list_scalar_udfs() + self._ctx._impl.list_aggregate_udfs()
        )
