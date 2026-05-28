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

from sedonadb.expr.expression import ScalarUdf


class ScalarFunctions:
    """Catalog of available scalar functions"""

    def __init__(self, ctx):
        self._ctx = ctx

    def __getitem__(self, key: str) -> ScalarUdf:
        return ScalarUdf(self._ctx._impl.scalar_udf(key))

    def __getattr__(self, name) -> ScalarUdf:
        return ScalarUdf(self._ctx._impl.scalar_udf(name))

    def __dir__(self) -> str:
        return self._ctx._impl.list_scalar_udfs()
