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
import os
from typing import Literal, Optional, Union

from sedonadb.utility import sedona  # noqa: F401


class Options:
    """Global SedonaDB options

    Options are divided into two categories:

    **Display options** can be changed at any time and affect how results are
    presented:

    - `interactive`: Enable/disable auto-display of DataFrames.
    - `width`: Override the detected terminal width.

    **Runtime options** configure the execution environment and must be set
    *before* the first query is executed (i.e., before the internal context
    is initialized). Attempting to change these after the context has been
    created will raise a `RuntimeError`:

    - `memory_limit`: Maximum memory for execution, in bytes or as a
      human-readable string (e.g., `"4gb"`, `"512m"`).
    - `temp_dir`: Directory for temporary/spill files.
    - `memory_pool_type`: Memory pool type (`"greedy"` or `"fair"`).
    - `unspillable_reserve_ratio`: Fraction of memory reserved for
      unspillable consumers (only applies to the `"fair"` pool type).

    Examples:

        >>> sd = sedona.db.connect()
        >>> sd.options.memory_limit = "4gb"
        >>> sd.options.memory_pool_type = "fair"
        >>> sd.options.temp_dir = "/tmp/sedona-spill"
        >>> sd.options.interactive = True
        >>> sd.sql("SELECT 1 as one")
        ┌───────┐
        │  one  │
        │ int64 │
        ╞═══════╡
        │     1 │
        └───────┘
    """

    def __init__(self):
        # Display options (can be changed at any time)
        self._interactive = False
        self._width = None

        # Runtime options (must be set before first query)
        self._memory_limit = None
        self._temp_dir = None
        self._memory_pool_type = "greedy"
        self._unspillable_reserve_ratio = None

        # Set to True once the internal context is created; after this,
        # runtime options become read-only.
        self._runtime_frozen = False

    def freeze_runtime(self) -> None:
        """Mark runtime options as read-only.

        Called after the internal context has been successfully created.
        """
        self._runtime_frozen = True

    def _check_runtime_mutable(self, name: str) -> None:
        if self._runtime_frozen:
            raise RuntimeError(
                f"Cannot change '{name}' after the context has been initialized. "
                f"Set this option before executing your first query."
            )

    # --- Display options ---

    @property
    def interactive(self) -> bool:
        """Use interactive mode

        In interactive mode, data frames are shown interactively without explicitly
        calling `.show()`. This is helpful when exploring data frames but may
        execute more queries than expected. Defaults to `False`.
        """
        return self._interactive

    @interactive.setter
    def interactive(self, value: bool) -> None:
        self._interactive = value

    @property
    def width(self) -> Optional[int]:
        """Set the terminal width

        Usually the terminal width can be automatically determined; however, in
        interactive environments like Jupyter notebooks this may not be possible
        or may be guessed incorrectly. Defaults to `None` (i.e., fall back to the
        detected value or 100 characters if nothing can be guessed).
        """
        return self._width

    @width.setter
    def width(self, value: Optional[int]):
        self._width = value

    # --- Runtime options (must be set before first query) ---

    @property
    def memory_limit(self) -> Union[int, str, None]:
        """Maximum memory for query execution.

        Accepts an integer (bytes) or a human-readable string such as
        `"4gb"`, `"512m"`, or `"1.5g"`. When set, a bounded memory pool is
        created to enforce this limit. Without a memory limit, DataFusion's
        default unbounded pool is used.

        Must be set before the first query is executed.

        Examples:

            >>> sd = sedona.db.connect()
            >>> sd.options.memory_limit = "4gb"
            >>> sd.options.memory_limit = 4 * 1024 * 1024 * 1024  # equivalent
        """
        return self._memory_limit

    @memory_limit.setter
    def memory_limit(self, value: Union[int, str, None]) -> None:
        self._check_runtime_mutable("memory_limit")
        if value is not None and not isinstance(value, (int, str)):
            raise TypeError(
                f"memory_limit must be an int, str, or None, got {type(value).__name__}"
            )
        self._memory_limit = value

    @property
    def temp_dir(self) -> Optional[str]:
        """Directory for temporary/spill files.

        When set, disk-based spilling will use this directory. When `None`,
        DataFusion's default temporary directory is used.

        Must be set before the first query is executed.
        """
        return self._temp_dir

    @temp_dir.setter
    def temp_dir(self, value: "Optional[Union[str, os.PathLike[str]]]") -> None:
        self._check_runtime_mutable("temp_dir")
        if value is None:
            self._temp_dir = None
        elif isinstance(value, os.PathLike):
            self._temp_dir = os.fspath(value)
        elif isinstance(value, str):
            self._temp_dir = value
        else:
            raise TypeError(
                f"temp_dir must be a str, PathLike, or None, got {type(value).__name__}"
            )

    @property
    def memory_pool_type(self) -> str:
        """Memory pool type: `"greedy"` or `"fair"`.

        - `"greedy"`: A simple pool that grants reservations on a
          first-come-first-served basis. This is the default.
        - `"fair"`: A pool that fairly distributes memory among spillable
          consumers and reserves a fraction for unspillable consumers
          (configured via `unspillable_reserve_ratio`).

        Only takes effect when `memory_limit` is set.
        Must be set before the first query is executed.
        """
        return self._memory_pool_type

    @memory_pool_type.setter
    def memory_pool_type(self, value: Literal["greedy", "fair"]) -> None:
        self._check_runtime_mutable("memory_pool_type")
        if value not in ("greedy", "fair"):
            raise ValueError(
                f"memory_pool_type must be 'greedy' or 'fair', got '{value}'"
            )
        self._memory_pool_type = value

    @property
    def unspillable_reserve_ratio(self) -> Optional[float]:
        """Fraction of memory reserved for unspillable consumers (0.0 - 1.0).

        Only applies when `memory_pool_type` is `"fair"` and
        `memory_limit` is set. Defaults to 0.2 when not explicitly set.

        Must be set before the first query is executed.
        """
        return self._unspillable_reserve_ratio

    @unspillable_reserve_ratio.setter
    def unspillable_reserve_ratio(self, value: Optional[float]) -> None:
        self._check_runtime_mutable("unspillable_reserve_ratio")
        if value is None:
            self._unspillable_reserve_ratio = None
            return
        if not isinstance(value, (int, float)):
            raise TypeError(
                "unspillable_reserve_ratio must be a number between 0.0 and 1.0 "
                f"or None, got {type(value).__name__}"
            )
        value = float(value)
        if not (0.0 <= value <= 1.0):
            raise ValueError(
                f"unspillable_reserve_ratio must be between 0.0 and 1.0, got {value}"
            )
        self._unspillable_reserve_ratio = value
