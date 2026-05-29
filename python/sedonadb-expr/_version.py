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

"""
Version source for hatchling - reads from workspace Cargo.toml.

This file is used by hatchling at build time to determine the version.
The build hook then generates a static _version.py inside the package.
"""

import re
from pathlib import Path


def get_version() -> str:
    """Read version from the workspace root Cargo.toml."""
    here = Path(__file__).parent
    cargo_toml = here.parent.parent / "Cargo.toml"

    if not cargo_toml.exists():
        raise FileNotFoundError(f"Could not find workspace Cargo.toml at {cargo_toml}")

    content = cargo_toml.read_text()

    match = re.search(
        r'\[workspace\.package\].*?version\s*=\s*"([^"]+)"',
        content,
        re.DOTALL,
    )
    if match:
        return match.group(1)

    raise ValueError("Could not find workspace.package.version in Cargo.toml")
