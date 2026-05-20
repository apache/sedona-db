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

"""Architectural regression test: `sedonadb` alone has no Zarr.

This locks in the plugin separation. A fresh sedonadb connection,
created without importing `sedonadb_zarr`, must not know about the
`sd_read_zarr` UDTF. If a future change accidentally re-bundles Zarr
into the main package — for example by adding a convenience import or
re-attaching the registration to `SedonaContext::new_from_context` —
this test fails and forces a conscious decision.

The test is here (not in `sedonadb-zarr`) on purpose: it's the test
that catches *re-bundling*, which would happen in `sedonadb`'s code,
not in the plugin.
"""

import pytest

import sedonadb


def test_sd_read_zarr_is_not_a_known_function_without_plugin():
    con = sedonadb.connect()
    # Use a non-existent path; we expect the planner to fail because
    # `sd_read_zarr` itself is unknown, not because the file doesn't
    # exist. The error message should mention the function name.
    with pytest.raises(Exception, match=r"sd_read_zarr|function|table function"):
        con.sql("SELECT * FROM sd_read_zarr('file:///nowhere/foo.zarr')")
