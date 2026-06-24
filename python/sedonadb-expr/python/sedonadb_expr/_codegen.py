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
Code generation for sedonadb-expr.

This module generates Python source files from docs/reference/sql documentation files.
It can be invoked during the build process or run as a standalone script.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Any, List, Optional

import yaml


# Type to parameter name mapping (matches R version)
TYPE_TO_PARAM: dict[str, str] = {
    "geometry": "geom",
    "geography": "geom",
    "raster": "rast",
    "float64": "x",
    "double": "x",
    "integer": "n",
    "int64": "n",
    "string": "s",
    "boolean": "b",
    "crs": "crs",
}

# Types that qualify for geo methods (first arg piped in)
GEO_TYPES = {"geometry", "geography"}

# Types that qualify for raster methods (first arg piped in)
RASTER_TYPES = {"raster"}

DOCS_BASE_URL = "https://sedona.apache.org/sedonadb/latest/reference/sql"


def camel_to_snake(name: str) -> str:
    """Convert CamelCase/PascalCase to snake_case.

    Examples:
        AsBinary -> as_binary
        GeomFromWKB -> geom_from_wkb
        AsEWKT -> as_ewkt
        LineInterpolatePoint -> line_interpolate_point
    """
    # Insert underscore before uppercase letters that follow lowercase letters
    # or before uppercase letters that are followed by lowercase letters
    result = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", name)
    result = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", result)
    return result.lower()


LICENSE_HEADER = """\
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


class ArgInfo:
    """Information about a kernel argument."""

    def __init__(
        self,
        type: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        optional: bool = False,
    ):
        self.type = type
        self.name = name
        self.description = description
        self.optional = optional


class KernelInfo:
    """Parsed kernel information."""

    def __init__(
        self,
        args: Optional[List[ArgInfo]] = None,
        returns: str = "unknown",
        variadic: bool = False,
        kernel_signatures: Optional[List[str]] = None,
    ):
        self.args = args if args is not None else []
        self.returns = returns
        self.variadic = variadic
        self.kernel_signatures = (
            kernel_signatures if kernel_signatures is not None else []
        )

    @property
    def has_optional_args(self) -> bool:
        """Return True if any argument is optional."""
        return any(arg.optional for arg in self.args)


class FunctionInfo:
    """Parsed function information from a .qmd file."""

    def __init__(
        self,
        name: str,
        title: str,
        description: str,
        kernels: list[dict[str, Any]],
        is_geo_method: bool = False,
        is_raster_method: bool = False,
        kernel_info: Optional[KernelInfo] = None,
        sql_name: Optional[str] = None,
    ):
        self.name = name
        self.title = title
        self.description = description
        self.kernels = kernels
        self.is_geo_method = is_geo_method
        self.is_raster_method = is_raster_method
        self.kernel_info = kernel_info
        self.sql_name = sql_name or name  # e.g., "ST_AsBinary" or "RS_Height"

    @property
    def method_name(self) -> str:
        """Return the snake_case method name derived from the SQL function name.

        e.g., ST_AsBinary -> as_binary, ST_GeomFromWKB -> geom_from_wkb
        """
        sql = self.sql_name
        # Strip prefix (ST_, RS_, S2_, SD_)
        for prefix in ("ST_", "RS_", "S2_", "SD_"):
            if sql.upper().startswith(prefix):
                sql = sql[len(prefix) :]
                break
        return camel_to_snake(sql)


def extract_frontmatter(file_path: Path) -> dict[str, Any]:
    """Extract YAML frontmatter from a .qmd file."""
    content = file_path.read_text()
    lines = content.split("\n")

    # Find YAML delimiters
    delimiters = [i for i, line in enumerate(lines) if line.strip() == "---"]
    if len(delimiters) < 2:
        raise ValueError(f"Could not find YAML frontmatter in {file_path}")

    yaml_text = "\n".join(lines[delimiters[0] + 1 : delimiters[1]])
    return yaml.safe_load(yaml_text)


def extract_description_section(file_path: Path) -> str | None:
    """Extract the ## Description section from the .qmd file body."""
    content = file_path.read_text()
    lines = content.split("\n")

    # Find end of frontmatter
    delimiters = [i for i, line in enumerate(lines) if line.strip() == "---"]
    if len(delimiters) < 2:
        return None

    body_lines = lines[delimiters[1] + 1 :]

    # Find ## Description section
    desc_start = None
    for i, line in enumerate(body_lines):
        if line.startswith("## Description"):
            desc_start = i
            break

    if desc_start is None:
        return None

    # Find next section or end
    remaining = body_lines[desc_start + 1 :]
    next_section = None
    for i, line in enumerate(remaining):
        if line.startswith("## "):
            next_section = i
            break

    if next_section is None:
        desc_lines = remaining
    else:
        desc_lines = remaining[:next_section]

    # Process lines: preserve markdown lists, join paragraphs
    result_lines: list[str] = []
    current_paragraph: list[str] = []

    for line in desc_lines:
        stripped = line.strip()
        # Check if this is a list item (-, *, or numbered)
        is_list_item = bool(re.match(r"^[-*]|\d+\.", stripped))

        if not stripped:
            # Empty line: flush current paragraph and add blank line for separation
            if current_paragraph:
                result_lines.append(" ".join(current_paragraph))
                current_paragraph = []
                result_lines.append("")  # Preserve paragraph break
        elif is_list_item:
            # List item: flush paragraph first, then add list item
            if current_paragraph:
                result_lines.append(" ".join(current_paragraph))
                current_paragraph = []
            result_lines.append(stripped)
        else:
            # Regular text: accumulate into paragraph
            current_paragraph.append(stripped)

    # Flush any remaining paragraph
    if current_paragraph:
        result_lines.append(" ".join(current_paragraph))

    desc_text = "\n".join(result_lines).strip()
    return desc_text if desc_text else None


def type_to_param_name(
    arg_type: str, index: int = 0, needs_suffix: bool = False
) -> str:
    """Generate parameter name from type."""
    base_name = TYPE_TO_PARAM.get(arg_type, "arg")
    if needs_suffix:
        suffix = chr(ord("a") + index)  # 0=a, 1=b, 2=c, ...
        return f"{base_name}_{suffix}"
    return base_name


def parse_kernel_args(kernel_args: list) -> list[ArgInfo]:
    """Parse kernel arguments into ArgInfo objects."""
    result = []
    for arg in kernel_args:
        if isinstance(arg, str):
            result.append(ArgInfo(type=arg))
        elif isinstance(arg, dict):
            result.append(
                ArgInfo(
                    type=arg.get("type", "unknown"),
                    name=arg.get("name"),
                    description=arg.get("description"),
                )
            )
        else:
            result.append(ArgInfo(type="unknown"))
    return result


def generate_arg_names(arg_info_list: list[ArgInfo]) -> list[str]:
    """Generate argument names for a kernel's args."""
    types = [info.type for info in arg_info_list]
    type_counts: dict[str, int] = {}
    type_totals: dict[str, int] = {}

    # Count total occurrences of each type
    for t in types:
        type_totals[t] = type_totals.get(t, 0) + 1

    arg_names = []
    for info in arg_info_list:
        arg_type = info.type
        arg_name = info.name

        if arg_name is None:
            type_counts[arg_type] = type_counts.get(arg_type, 0) + 1
            needs_suffix = type_totals.get(arg_type, 0) > 1
            arg_name = type_to_param_name(
                arg_type, type_counts[arg_type] - 1, needs_suffix
            )

        arg_names.append(arg_name)

    return arg_names


def parse_kernel_params(kernels: list[dict], fn_name: str = "unknown") -> KernelInfo:
    """Parse kernel arguments and generate parameter info."""
    if not kernels:
        return KernelInfo()

    # Process all kernels
    all_kernel_info = [parse_kernel_args(k.get("args", [])) for k in kernels]
    all_kernel_args = [generate_arg_names(info) for info in all_kernel_info]

    # Find max args
    kernel_lengths = [len(args) for args in all_kernel_args]
    max_args = max(kernel_lengths) if kernel_lengths else 0

    # Check for argument name conflicts
    has_conflict = False
    for pos in range(max_args):
        names_at_pos = set()
        for args in all_kernel_args:
            if pos < len(args):
                names_at_pos.add(args[pos])
        if len(names_at_pos) > 1:
            has_conflict = True
            break

    returns = kernels[0].get("returns", "unknown")

    if has_conflict:
        # Build signature strings for documentation
        kernel_signatures = []
        for i, args in enumerate(all_kernel_args):
            types = [info.type for info in all_kernel_info[i]]
            sig = ", ".join(f"{arg} ({t})" for arg, t in zip(args, types))
            kernel_signatures.append(sig)

        return KernelInfo(
            args=[],
            returns=returns,
            variadic=True,
            kernel_signatures=kernel_signatures,
        )

    # Use kernel with most arguments as reference
    ref_idx = kernel_lengths.index(max(kernel_lengths)) if kernel_lengths else 0
    arg_info = all_kernel_info[ref_idx] if all_kernel_info else []
    arg_names = all_kernel_args[ref_idx] if all_kernel_args else []

    # Determine minimum args (args present in all kernels)
    min_args = min(kernel_lengths) if kernel_lengths else 0

    # Update ArgInfo with generated names and optional flag
    for i, info in enumerate(arg_info):
        if info.name is None:
            info.name = arg_names[i]
        # Args beyond min_args are optional (not present in all kernels)
        info.optional = i >= min_args

    return KernelInfo(args=arg_info, returns=returns, variadic=False)


def parse_qmd_file(qmd_path: Path) -> FunctionInfo | None:
    """Parse a .qmd file and return FunctionInfo."""
    fn_name = qmd_path.stem  # e.g., "st_envelope" or "rs_height"

    try:
        frontmatter = extract_frontmatter(qmd_path)
    except Exception:
        return None

    kernels = frontmatter.get("kernels", [])
    if not kernels:
        return None

    # Check if first argument of any kernel is geometry/geography or raster
    is_geo_method = False
    is_raster_method = False
    for kernel in kernels:
        args = kernel.get("args", [])
        if args:
            first_arg = args[0]
            first_type = (
                first_arg if isinstance(first_arg, str) else first_arg.get("type", "")
            )
            if first_type in GEO_TYPES:
                is_geo_method = True
                break
            if first_type in RASTER_TYPES:
                is_raster_method = True
                break

    # Get properly-cased SQL function name from title field
    sql_name = frontmatter.get("title", fn_name)
    title = frontmatter.get("description", frontmatter.get("title", fn_name))
    description = extract_description_section(qmd_path) or ""

    kernel_info = parse_kernel_params(kernels, fn_name)

    return FunctionInfo(
        name=fn_name,
        title=title,
        description=description,
        kernels=kernels,
        is_geo_method=is_geo_method,
        is_raster_method=is_raster_method,
        kernel_info=kernel_info,
        sql_name=sql_name,
    )


def wrap_docstring(text: str, width: int = 88, indent: str = "        ") -> str:
    """Wrap text for docstrings, preserving markdown lists."""
    if not text:
        return ""

    result_lines: list[str] = []
    for i, line in enumerate(text.split("\n")):
        if not line.strip():
            result_lines.append("")
            continue

        # Wrap each line separately
        wrapped = textwrap.fill(line, width=width - len(indent))
        for j, wrapped_line in enumerate(wrapped.split("\n")):
            if i == 0 and j == 0:
                # First line of first paragraph - no indent
                result_lines.append(wrapped_line)
            else:
                result_lines.append(indent + wrapped_line)

    return "\n".join(result_lines)


def generate_method_docstring(func: FunctionInfo) -> str:
    """Generate docstring for a method."""
    title = func.title.strip()
    parts = [f'"""{title}']

    if func.description and func.description.strip() != title:
        parts.append("")
        parts.append(wrap_docstring(func.description, indent="        "))

    kernel_info = func.kernel_info
    if kernel_info:
        if kernel_info.variadic and kernel_info.kernel_signatures:
            # Variadic mode: document with bulleted list of supported combinations
            # Skip the first arg (piped in via self._expr) from each signature
            parts.append("")
            parts.append("Variants:")
            for sig in kernel_info.kernel_signatures:
                # Split signature, skip first arg, rejoin
                arg_parts = [p.strip() for p in sig.split(",")]
                remaining = ", ".join(arg_parts[1:]) if len(arg_parts) > 1 else ""
                if remaining:
                    parts.append(f"    - {remaining}")
                else:
                    parts.append("    - (no additional arguments)")
        elif kernel_info.args:
            # Skip first arg (piped in via self._expr)
            remaining_args = kernel_info.args[1:] if len(kernel_info.args) > 1 else []
            if remaining_args:
                parts.append("")
                parts.append("Args:")
                for arg in remaining_args:
                    desc = arg.description or f"Input {arg.type}"
                    parts.append(f"    {arg.name}: {desc}")

    parts.append("")
    parts.append("See Also:")
    parts.append(f"    {DOCS_BASE_URL}/{func.name}/")
    parts.append('"""')

    joined = "\n        ".join(parts)
    return "\n".join(line.rstrip() for line in joined.split("\n"))


def generate_function_docstring(func: FunctionInfo) -> str:
    """Generate docstring for a standalone function property."""
    title = func.title.strip()
    parts = [f'"""{title}']

    if func.description and func.description.strip() != title:
        parts.append("")
        parts.append(wrap_docstring(func.description, indent="        "))

    kernel_info = func.kernel_info
    if kernel_info:
        if kernel_info.variadic and kernel_info.kernel_signatures:
            # Variadic mode: document with bulleted list of supported combinations
            parts.append("")
            parts.append("Variants:")
            for sig in kernel_info.kernel_signatures:
                parts.append(f"    - {sig}")
        elif kernel_info.args:
            parts.append("")
            parts.append("Args:")
            for arg in kernel_info.args:
                desc = arg.description or f"Input {arg.type}"
                parts.append(f"    {arg.name}: {desc}")

    parts.append("")
    parts.append("See Also:")
    parts.append(f"    {DOCS_BASE_URL}/{func.name}/")
    parts.append('"""')

    joined = "\n        ".join(parts)
    return "\n".join(line.rstrip() for line in joined.split("\n"))


def generate_geo_methods_py(functions: list[FunctionInfo]) -> str:
    """Generate geo_methods.py content."""
    # Filter to only geo methods (first arg is geometry/geography)
    geo_funcs = [f for f in functions if f.is_geo_method]

    lines = [
        LICENSE_HEADER,
        "",
        '"""Auto-generated geometry/geography methods - do not edit."""',
        "",
        "from typing import Generic, TypeVar",
        "",
        "from sedonadb_expr.utils import MISSING, filter_missing_args",
        "",
        'ExprT = TypeVar("ExprT")',
        "",
        "",
        "class GeoMethods(Generic[ExprT]):",
        '    """Geometry and geography methods accessible via expr.geo."""',
        "",
        "    def __init__(self, expr: ExprT) -> None:",
        "        self._expr = expr",
    ]

    for func in sorted(geo_funcs, key=lambda f: f.name):
        # Method name: derived from SQL function name (e.g., ST_AsBinary -> as_binary)
        method_name = func.method_name

        kernel_info = func.kernel_info
        if not kernel_info:
            continue

        # Build method signature - skip first arg (piped in)
        remaining_args = kernel_info.args[1:] if len(kernel_info.args) > 1 else []
        # Check if any remaining args are optional
        has_optional = any(arg.optional for arg in remaining_args)

        if kernel_info.variadic:
            params = "self, *args"
            call_args = "*args"
            use_filter = False
        elif remaining_args:
            # Build param strings with MISSING default for optional args
            param_strs = []
            for arg in remaining_args:
                if arg.optional:
                    param_strs.append(f"{arg.name}=MISSING")
                else:
                    param_strs.append(arg.name)
            params = "self, " + ", ".join(param_strs)
            call_args = ", ".join(arg.name for arg in remaining_args)
            use_filter = has_optional
        else:
            params = "self"
            call_args = ""
            use_filter = False

        docstring = generate_method_docstring(func)

        lines.extend(
            [
                "",
                f"    def {method_name}({params}) -> ExprT:",
                f"        {docstring}",
            ]
        )

        if call_args:
            if use_filter:
                lines.append(
                    f'        return self._expr._call("{func.name}", *filter_missing_args({call_args}))'
                )
            else:
                lines.append(
                    f'        return self._expr._call("{func.name}", {call_args})'
                )
        else:
            lines.append(f'        return self._expr._call("{func.name}")')

    lines.append("")
    return "\n".join(lines)


def generate_geo_functions_py(functions: list[FunctionInfo]) -> str:
    """Generate geo_functions.py content."""
    # Filter to only geo methods (these become callable properties)
    geo_funcs = [f for f in functions if f.is_geo_method]

    lines = [
        LICENSE_HEADER,
        "",
        '"""Auto-generated geometry/geography functions - do not edit."""',
        "",
        "from typing import Callable, Generic, TypeVar",
        "",
        'ExprT = TypeVar("ExprT")',
        "",
        "",
        "class GeoFunctions(Generic[ExprT]):",
        '    """Geometry and geography functions accessible via a factory."""',
        "",
        "    def __init__(self, factory) -> None:",
        "        self._factory = factory",
    ]

    for func in sorted(geo_funcs, key=lambda f: f.name):
        # Property name: derived from SQL function name (e.g., ST_AsBinary -> as_binary)
        prop_name = func.method_name

        docstring = generate_function_docstring(func)

        lines.extend(
            [
                "",
                "    @property",
                f"    def {prop_name}(self) -> Callable[..., ExprT]:",
                f"        {docstring}",
                f'        return self._factory["{func.name}"]',
            ]
        )

    lines.append("")
    return "\n".join(lines)


def generate_raster_methods_py(functions: list[FunctionInfo]) -> str:
    """Generate raster_methods.py content."""
    # Filter to only raster methods (first arg is raster)
    raster_funcs = [f for f in functions if f.is_raster_method]

    lines = [
        LICENSE_HEADER,
        "",
        '"""Auto-generated raster methods - do not edit."""',
        "",
        "from typing import Generic, TypeVar",
        "",
        "from sedonadb_expr.utils import MISSING, filter_missing_args",
        "",
        'ExprT = TypeVar("ExprT")',
        "",
        "",
        "class RasterMethods(Generic[ExprT]):",
        '    """Raster methods accessible via expr.rast."""',
        "",
        "    def __init__(self, expr: ExprT) -> None:",
        "        self._expr = expr",
    ]

    for func in sorted(raster_funcs, key=lambda f: f.name):
        # Method name: derived from SQL function name (e.g., RS_Height -> height)
        method_name = func.method_name

        kernel_info = func.kernel_info
        if not kernel_info:
            continue

        # Build method signature - skip first arg (piped in)
        remaining_args = kernel_info.args[1:] if len(kernel_info.args) > 1 else []
        # Check if any remaining args are optional
        has_optional = any(arg.optional for arg in remaining_args)

        if kernel_info.variadic:
            params = "self, *args"
            call_args = "*args"
            use_filter = False
        elif remaining_args:
            # Build param strings with MISSING default for optional args
            param_strs = []
            for arg in remaining_args:
                if arg.optional:
                    param_strs.append(f"{arg.name}=MISSING")
                else:
                    param_strs.append(arg.name)
            params = "self, " + ", ".join(param_strs)
            call_args = ", ".join(arg.name for arg in remaining_args)
            use_filter = has_optional
        else:
            params = "self"
            call_args = ""
            use_filter = False

        docstring = generate_method_docstring(func)

        lines.extend(
            [
                "",
                f"    def {method_name}({params}) -> ExprT:",
                f"        {docstring}",
            ]
        )

        if call_args:
            if use_filter:
                lines.append(
                    f'        return self._expr._call("{func.name}", *filter_missing_args({call_args}))'
                )
            else:
                lines.append(
                    f'        return self._expr._call("{func.name}", {call_args})'
                )
        else:
            lines.append(f'        return self._expr._call("{func.name}")')

    lines.append("")
    return "\n".join(lines)


def generate_raster_functions_py(functions: list[FunctionInfo]) -> str:
    """Generate raster_functions.py content."""
    # Filter to only raster methods (these become callable properties)
    raster_funcs = [f for f in functions if f.is_raster_method]

    lines = [
        LICENSE_HEADER,
        "",
        '"""Auto-generated raster functions - do not edit."""',
        "",
        "from typing import Callable, Generic, TypeVar",
        "",
        'ExprT = TypeVar("ExprT")',
        "",
        "",
        "class RasterFunctions(Generic[ExprT]):",
        '    """Raster functions accessible via a factory."""',
        "",
        "    def __init__(self, factory) -> None:",
        "        self._factory = factory",
    ]

    for func in sorted(raster_funcs, key=lambda f: f.name):
        # Property name: derived from SQL function name (e.g., RS_Height -> height)
        prop_name = func.method_name

        docstring = generate_function_docstring(func)

        lines.extend(
            [
                "",
                "    @property",
                f"    def {prop_name}(self) -> Callable[..., ExprT]:",
                f"        {docstring}",
                f'        return self._factory["{func.name}"]',
            ]
        )

    lines.append("")
    return "\n".join(lines)


class GenerationResult:
    """Result of code generation."""

    def __init__(
        self,
        total_functions: int,
        geo_method_count: int,
        raster_method_count: int,
        generated_files: list[Path],
    ):
        self.total_functions = total_functions
        self.geo_method_count = geo_method_count
        self.raster_method_count = raster_method_count
        self.generated_files = generated_files


def parse_qmd_files(docs_sql: Path, pattern: str) -> list[FunctionInfo]:
    """Parse all .qmd files in a directory and return function definitions.

    Args:
        docs_sql: Path to directory containing .qmd files.

    Returns:
        List of parsed FunctionInfo objects.
    """
    qmd_files = sorted(docs_sql.glob(pattern))
    functions: list[FunctionInfo] = []
    for qmd_file in qmd_files:
        func = parse_qmd_file(qmd_file)
        if func:
            functions.append(func)
    return functions


def generate_sources(docs_sql: Path, output_dir: Path) -> GenerationResult:
    """Generate Python source files from docs/reference/sql.

    Args:
        docs_sql: Path to docs/reference/sql directory containing .qmd files.
        output_dir: Path to output directory for generated files.

    Returns:
        GenerationResult with statistics about generated code.
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create __init__.py for the generated module
    init_file = output_dir / "__init__.py"
    init_file.write_text(
        f"{LICENSE_HEADER}\n"
        "# Auto-generated module - do not edit\n"
        "# Generated from docs/reference/sql\n"
    )

    generated_files: list[Path] = [init_file]

    if not docs_sql.exists():
        return GenerationResult(
            total_functions=0,
            geo_method_count=0,
            raster_method_count=0,
            generated_files=generated_files,
        )

    # Parse ST_* functions for geo methods/functions
    geo_functions = parse_qmd_files(docs_sql, "st_*.qmd")

    # Generate geo_methods.py
    geo_methods_content = generate_geo_methods_py(geo_functions)
    geo_methods_file = output_dir / "geo_methods.py"
    geo_methods_file.write_text(geo_methods_content)
    generated_files.append(geo_methods_file)

    # Generate geo_functions.py
    geo_functions_content = generate_geo_functions_py(geo_functions)
    geo_functions_file = output_dir / "geo_functions.py"
    geo_functions_file.write_text(geo_functions_content)
    generated_files.append(geo_functions_file)

    # Parse RS_* functions for raster methods/functions
    raster_functions = parse_qmd_files(docs_sql, "rs_*.qmd")

    # Generate raster_methods.py
    raster_methods_content = generate_raster_methods_py(raster_functions)
    raster_methods_file = output_dir / "raster_methods.py"
    raster_methods_file.write_text(raster_methods_content)
    generated_files.append(raster_methods_file)

    # Generate raster_functions.py
    raster_functions_content = generate_raster_functions_py(raster_functions)
    raster_functions_file = output_dir / "raster_functions.py"
    raster_functions_file.write_text(raster_functions_content)
    generated_files.append(raster_functions_file)

    # Count stats
    geo_method_count = sum(1 for f in geo_functions if f.is_geo_method)
    raster_method_count = sum(1 for f in raster_functions if f.is_raster_method)

    return GenerationResult(
        total_functions=len(geo_functions) + len(raster_functions),
        geo_method_count=geo_method_count,
        raster_method_count=raster_method_count,
        generated_files=generated_files,
    )


if __name__ == "__main__":
    # Allow running as a standalone script for development/debugging
    here = Path(__file__).parent
    docs_sql = here.parent.parent.parent.parent / "docs" / "reference" / "sql"
    output_dir = here / "_generated"

    result = generate_sources(docs_sql, output_dir)

    print(f"Generated {result.total_functions} functions total")
    print(f"Generated {result.geo_method_count} geo methods")
    print(f"Generated {result.raster_method_count} raster methods")
    print("Output files:")
    for f in result.generated_files:
        print(f"  - {f}")
