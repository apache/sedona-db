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
        here = Path(__file__).parent
        docs_sql = here.parent.parent / "docs" / "reference" / "sql"
        output_dir = here / "python" / "sedonadb_expr" / "_generated"

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create __init__.py for the generated module
        init_file = output_dir / "__init__.py"
        init_file.write_text(
            "# Auto-generated module - do not edit\n"
            "# Generated from docs/reference/sql\n"
        )

        if not docs_sql.exists():
            self.app.display_warning(
                f"docs/reference/sql not found at {docs_sql}, skipping generation"
            )
            return

        # Find all .qmd files (source files, not rendered .md)
        qmd_files = sorted(docs_sql.glob("*.qmd"))

        functions = []
        for qmd_file in qmd_files:
            # Skip index and special files
            if qmd_file.stem in ("index", "barrier", "_quarto"):
                continue

            func_name = qmd_file.stem.upper()
            functions.append(func_name)

        # Generate a simple functions module listing all available functions
        functions_file = output_dir / "functions.py"
        functions_content = [
            "# Auto-generated - do not edit",
            "# Generated from docs/reference/sql/*.qmd",
            "",
            "FUNCTIONS = [",
        ]
        for func in sorted(functions):
            functions_content.append(f'    "{func}",')
        functions_content.append("]")
        functions_content.append("")

        functions_file.write_text("\n".join(functions_content))

        self.app.display_info(f"Generated {len(functions)} function definitions")
