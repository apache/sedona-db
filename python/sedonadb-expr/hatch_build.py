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
Hatch build hook for sedonadb-expr.

This hook runs during sdist and wheel builds to generate Python source
files from the docs/reference/sql documentation files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Custom build hook that generates Python sources from SQL docs."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        """
        Called before the build process starts.

        Args:
            version: The version being built
            build_data: Mutable dict to modify build behavior
        """
        self._generate_version(version)
        self._generate_sources()

    def _generate_version(self, version: str) -> None:
        """Generate _version.py with the static version string."""
        here = Path(__file__).parent
        version_file = here / "python" / "sedonadb_expr" / "_version.py"

        content = f'''# Auto-generated at build time - do not edit
__version__ = "{version}"
'''
        version_file.write_text(content)
        self.app.display_info(f"Generated _version.py with version {version}")

    def _generate_sources(self) -> None:
        """Generate Python source files from docs/reference/sql."""
        # Import here to avoid circular imports and allow standalone usage
        import sys

        here = Path(__file__).parent
        # Add the package to sys.path so we can import _codegen
        sys.path.insert(0, str(here / "python"))
        from sedonadb_expr._codegen import generate_sources

        docs_sql = here.parent.parent / "docs" / "reference" / "sql"
        output_dir = here / "python" / "sedonadb_expr" / "_generated"

        result = generate_sources(docs_sql, output_dir)

        if result.total_functions == 0 and not docs_sql.exists():
            self.app.display_warning(
                f"docs/reference/sql not found at {docs_sql}, skipping generation"
            )
            return

        self.app.display_info(
            f"Generated {result.total_functions} functions total, "
            f"{result.geo_method_count} geo methods"
        )
