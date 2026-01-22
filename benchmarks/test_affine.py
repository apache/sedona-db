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
import pytest
from test_bench_base import TestBenchBase
from sedonadb.testing import (
    SedonaDBSingleThread,
)


class TestBenchAffine(TestBenchBase):
    @pytest.mark.parametrize("eng", [SedonaDBSingleThread])
    @pytest.mark.parametrize("case", ["scalar", "array", "mixed"])
    def test_st_affine_2d(self, benchmark, eng, case):
        eng = self._get_eng(eng)

        if case == "scalar":
            query = (
                "SELECT ST_Affine(geom1, 1.0, 0.1, 0.0, 1.0, 2.0, 3.0) "
                "FROM affine_params"
            )
        elif case == "array":
            query = "SELECT ST_Affine(geom1, a, b, d, e, xoff, yoff) FROM affine_params"
        else:
            query = (
                "SELECT ST_Affine(geom1, a, 0.1, d, 1.0, xoff, 2.0) FROM affine_params"
            )

        def queries():
            eng.execute_and_collect(query)

        benchmark(queries)
