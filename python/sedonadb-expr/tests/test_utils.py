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
from sedonadb_expr import GeoFunctions, GeoMethods
from sedonadb_expr.utils import MISSING, filter_missing_args


class MockExpr:
    """Mock expression that records _call invocations."""

    def __init__(self):
        self.calls = []

    def _call(self, name, *args):
        self.calls.append((name, args))
        return self


class MockFunctions:
    """Mock function mapping that records _call invocations."""

    def __init__(self):
        self.calls = []

    def __getitem__(self, name, *args):
        def fn(*args):
            self.calls.append((name, args))

        return fn


def test_filter_missing_args():
    """Tests for filter_missing_args utility."""
    # Passthrough when no missing
    assert filter_missing_args(1, 2, 3) == (1, 2, 3)

    # All missing returns empty
    assert filter_missing_args(MISSING, MISSING) == ()

    # Trailing missing are filtered
    assert filter_missing_args(1, 2, MISSING, MISSING) == (1, 2)
    assert filter_missing_args(1, MISSING) == (1,)

    # Empty args
    assert filter_missing_args() == ()

    # Missing before non-missing raises
    with pytest.raises(ValueError, match="Missing arguments must be at the end"):
        filter_missing_args(MISSING, 1)

    # Missing in middle raises
    with pytest.raises(ValueError, match="Missing arguments must be at the end"):
        filter_missing_args(1, MISSING, 2)


def test_geo_methods_missing_args():
    """Tests for MISSING argument handling in generated GeoMethods."""
    mock = MockExpr()
    geo = GeoMethods(mock)

    # force4d with no args passes no extra arguments
    geo.force4d()
    assert mock.calls[-1] == ("st_force4d", ())

    # force4d with z only passes just z
    geo.force4d(z=1.0)
    assert mock.calls[-1] == ("st_force4d", (1.0,))

    # force4d with both z and m passes both
    geo.force4d(z=1.0, m=2.0)
    assert mock.calls[-1] == ("st_force4d", (1.0, 2.0))

    # translate with partial missing filters correctly
    geo.translate(deltaX=1.0, deltaY=2.0)
    assert mock.calls[-1] == ("st_translate", (1.0, 2.0))

    # force4d with MISSING z but non-MISSING m should raise
    with pytest.raises(ValueError, match="Missing arguments must be at the end"):
        geo.force4d(z=MISSING, m=2.0)

    # translate with MISSING in middle should raise
    with pytest.raises(ValueError, match="Missing arguments must be at the end"):
        geo.translate(deltaX=1.0, deltaY=MISSING, deltaZ=3.0)


def test_geo_functions():
    """Tests for GeoFunctions property access."""
    factory = MockFunctions()
    geo_fns = GeoFunctions(factory)

    # Properties return callables
    assert callable(geo_fns.affine)
    assert callable(geo_fns.buffer)

    # Calling returned function invokes factory
    geo_fns.envelope("geom_arg")
    assert factory.calls[-1] == ("st_envelope", ("geom_arg",))
