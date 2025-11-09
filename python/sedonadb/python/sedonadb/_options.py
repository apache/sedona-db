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
from typing import Optional


class Options:
    """Global SedonaDB options"""

    def __init__(self):
        self._interactive = False
        self._width = None

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
