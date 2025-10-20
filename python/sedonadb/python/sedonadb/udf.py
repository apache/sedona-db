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

from typing import Literal, Optional

from sedonadb._lib import sedona_scalar_udf


def arrow_udf(
    return_type,
    input_types=None,
    volatility: Literal["immutable", "stable", "volatile"] = "immutable",
    name: Optional[str] = None,
):
    def decorator(func):
        def func_wrapper(args, return_type, num_rows):
            return func(*args)

        name = func.__name__ if hasattr(func, "__name__") else None
        return ScalarUdfImpl(func_wrapper, return_type, input_types, volatility, name)

    # Decorator must always be used with parentheses
    return decorator


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

    def __datafusion_scalar_udf__(self):
        return sedona_scalar_udf(
            self._invoke_batch,
            self._return_type,
            self._input_types,
            self._volatility,
            self._name,
        )
