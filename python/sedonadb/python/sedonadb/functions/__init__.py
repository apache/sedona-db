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
    from sedonadb.functions.scalar import ScalarFunctions
    from sedonadb.functions.aggregate import AggregateFunctions


class Functions:
    """Functions accessor

    This class provides Pythonic wrappers to call SedonaDB functions
    given a specific SedonaDB context.
    """

    def __init__(self, ctx):
        self._ctx = ctx

    @cached_property
    def table(self) -> "TableFunctions":
        """Access SedonaDB Table functions"""
        from sedonadb.functions.table import TableFunctions

        return TableFunctions(self._ctx)

    @cached_property
    def scalar(self) -> "ScalarFunctions":
        """Access SedonaDB Scalar functions"""
        from sedonadb.functions.scalar import ScalarFunctions

        return ScalarFunctions(self._ctx)

    @cached_property
    def aggregate(self) -> "AggregateFunctions":
        """Access SedonaDB Aggregate functions"""
        from sedonadb.functions.aggregate import AggregateFunctions

        return AggregateFunctions(self._ctx)

    def __getattr__(self, name) -> Union["ScalarUdf", "AggregateUdf"]:
        try:
            return ScalarUdf(self._ctx._impl.scalar_udf(name))
        except SedonaError:
            pass

        try:
            return AggregateUdf(self._ctx._impl.aggregate_udf(name))
        except SedonaError:
            pass

        return AttributeError(f"Can't find scalar or aggregate function '{name}'")

    def __getitem__(self, key) -> Union["ScalarUdf", "AggregateUdf"]:
        try:
            return self.__getattr__(key)
        except AttributeError:
            KeyError(f"Can't find scalar or aggregate function '{key}'")
