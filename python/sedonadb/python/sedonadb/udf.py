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

import inspect
from typing import Literal, Optional

from sedonadb._lib import sedona_scalar_udf


def arrow_udf(
    return_type,
    input_types=None,
    volatility: Literal["immutable", "stable", "volatile"] = "immutable",
    name: Optional[str] = None,
):
    def decorator(func):
        kwarg_names = callable_kwarg_only_names(func)
        if "return_type" in kwarg_names and "num_rows" in kwarg_names:

            def func_wrapper(args, return_type, num_rows):
                return func(*args, return_type=return_type, num_rows=num_rows)
        elif "return_type" in kwarg_names:

            def func_wrapper(args, return_type, num_rows):
                return func(*args, return_type=return_type)
        elif "num_rows" in kwarg_names:

            def func_wrapper(args, return_type, num_rows):
                return func(*args, num_rows=num_rows)
        else:

            def func_wrapper(args, return_type, num_rows):
                return func(*args)

        name_arg = func.__name__ if name is None and hasattr(func, "__name__") else name
        return ScalarUdfImpl(
            func_wrapper, return_type, input_types, volatility, name_arg
        )

    return decorator


class TypeMatcher(str):
    """Helper class to mark type matchers that can be used as the `input_types` for
    user-defined functions

    Note that the internal storage of the type matcher (currently a string) is
    arbitrary and may change in a future release. Use the constants provided by
    the `udf` module.
    """

    pass


BINARY: TypeMatcher = "binary"
"""Match any binary argument (i.e., binary, binary view, large binary,
fixed-size binary)"""

BOOLEAN: TypeMatcher = "boolean"
"""Match a boolean argument"""

GEOGRAPHY: TypeMatcher = "geography"
"""Match a geography argument"""

GEOMETRY: TypeMatcher = "geometry"
"""Match a geometry argument"""

NUMERIC: TypeMatcher = "numeric"
"""Match any numeric argument"""

STRING: TypeMatcher = "string"
"""Match any string argument (i.e., string, string view, large string)"""


class ScalarUdfImpl:
    def __init__(
        self,
        invoke_batch,
        return_type,
        input_types=None,
        volatility: Literal["immutable", "stable", "volatile"] = "immutable",
        name: Optional[str] = None,
    ):
        if input_types is None and not callable(return_type):

            def return_type_impl(*args, **kwargs):
                return return_type

            self._return_type = return_type_impl
        else:
            self._return_type = return_type

        self._invoke_batch = invoke_batch
        self._input_types = input_types
        if name is None and hasattr(invoke_batch, "__name__"):
            self._name = invoke_batch.__name__
        else:
            self._name = name

        self._volatility = volatility

    def __sedona_internal_udf__(self):
        return sedona_scalar_udf(
            self._invoke_batch,
            self._return_type,
            self._input_types,
            self._volatility,
            self._name,
        )


def callable_kwarg_only_names(f):
    sig = inspect.signature(f)
    return [
        k for k, p in sig.parameters.items() if p.kind == inspect.Parameter.KEYWORD_ONLY
    ]
