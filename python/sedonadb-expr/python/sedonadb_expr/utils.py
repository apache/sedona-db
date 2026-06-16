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

"""Utility classes and functions for sedonadb-expr."""

from typing import Any


class _MissingType:
    """Sentinel type for missing/omitted arguments.

    This is distinct from None, which represents a valid NULL value.
    Use the MISSING singleton instance rather than creating new instances.
    """


MISSING = _MissingType()
"""Sentinel value for missing/omitted arguments.

Use this as the default value for optional parameters.
"""


def filter_missing_args(*args: Any):
    """Filter out trailing MISSING arguments, validating ordering.

    Args:
        *args: Arguments to filter

    Returns:
        Tuple of non-MISSING arguments

    Raises:
        ValueError: If MISSING arguments appear before non-MISSING arguments
    """
    if not args:
        return ()

    # Find indices of missing args
    is_missing = [arg is MISSING for arg in args]

    if not any(is_missing):
        return args

    # Find last non-missing arg
    last_non_missing = -1
    for i in range(len(args) - 1, -1, -1):
        if not is_missing[i]:
            last_non_missing = i
            break

    # Check no missing args before non-missing args
    if last_non_missing >= 0 and any(is_missing[: last_non_missing + 1]):
        raise ValueError("Missing arguments must be at the end of the argument list")

    # Return args up to and including last non-missing
    if last_non_missing < 0:
        return ()
    return args[: last_non_missing + 1]
