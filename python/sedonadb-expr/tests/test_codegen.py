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

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from sedonadb_expr import _codegen

SAMPLE_QMD = """\
---
title: ST_Buffer
description: Computes a buffered geometry.
kernels:
  - returns: geometry
    args:
    - geometry
    - name: distance
      type: float64
      description: Radius of the buffer
---

## Description

Returns a geometry covering all points within a given distance.
This paragraph could have more than one line.

This is the second paragraph.

- Followed by a list!
- Second bullet point

## The Next Section

...if there is one
"""


@pytest.fixture
def sample_qmd_path():
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "st_buffer.qmd"
        path.write_text(SAMPLE_QMD)
        yield path


def test_camel_to_snake():
    assert _codegen.camel_to_snake("AsBinary") == "as_binary"
    assert _codegen.camel_to_snake("GeomFromWKB") == "geom_from_wkb"
    assert _codegen.camel_to_snake("AsEWKT") == "as_ewkt"
    assert _codegen.camel_to_snake("LineInterpolatePoint") == "line_interpolate_point"


def test_extract_frontmatter(sample_qmd_path: Path):
    fm = _codegen.extract_frontmatter(sample_qmd_path)
    assert fm["title"] == "ST_Buffer"


def test_extract_description_section(sample_qmd_path: Path):
    desc = _codegen.extract_description_section(sample_qmd_path)
    expected = """\
Returns a geometry covering all points within a given distance. This paragraph could have more than one line.

This is the second paragraph.

- Followed by a list!
- Second bullet point"""
    assert desc == expected


def test_generate_method_docstring_with_args():
    # Test case: method with description and args (first arg skipped)
    func = _codegen.FunctionInfo(
        name="st_buffer",
        title="ST_Buffer",
        description="Returns a buffered geometry.",
        kernels=[],
        kernel_info=_codegen.KernelInfo(
            args=[
                _codegen.ArgInfo(
                    type="geometry", name="geom", description="Input geometry"
                ),
                _codegen.ArgInfo(
                    type="float64", name="distance", description="Buffer distance"
                ),
            ],
            returns="geometry",
        ),
    )
    docstring = _codegen.generate_method_docstring(func)
    expected = '''\
"""ST_Buffer

        Returns a buffered geometry.

        Args:
            distance: Buffer distance

        See Also:
            https://sedona.apache.org/sedonadb/latest/reference/sql/st_buffer/
        """'''
    assert docstring == expected


def test_generate_method_docstring_variadic():
    # Variadic mode is triggered when parameter names conflict across kernels
    # e.g., ST_Buffer has (geom, distance, params) vs (geog, distance, num_quad_segs)
    # where position 3 conflicts: "params" vs "num_quad_segs"
    func = _codegen.FunctionInfo(
        name="st_buffer",
        title="ST_Buffer",
        description="Creates a buffer.",
        kernels=[],
        kernel_info=_codegen.KernelInfo(
            args=[],
            returns="geometry",
            variadic=True,
            kernel_signatures=[
                "geom (geometry), distance (float64)",
                "geom (geometry), distance (float64), params (string)",
                "geog (geography), distance (float64)",
                "geog (geography), distance (float64), num_quad_segs (integer)",
                "geog (geography), distance (float64), params (string)",
            ],
        ),
    )
    docstring = _codegen.generate_method_docstring(func)
    # Method docstring skips the first arg (piped in via self._expr)
    expected = '''\
"""ST_Buffer

        Creates a buffer.

        Variants:
            - distance (float64)
            - distance (float64), params (string)
            - distance (float64)
            - distance (float64), num_quad_segs (integer)
            - distance (float64), params (string)

        See Also:
            https://sedona.apache.org/sedonadb/latest/reference/sql/st_buffer/
        """'''
    assert docstring == expected


def test_generate_function_docstring_with_args():
    # Test case: function with description and args (all args included)
    func = _codegen.FunctionInfo(
        name="st_buffer",
        title="ST_Buffer",
        description="Returns a buffered geometry.",
        kernels=[],
        kernel_info=_codegen.KernelInfo(
            args=[
                _codegen.ArgInfo(
                    type="geometry", name="geom", description="Input geometry"
                ),
                _codegen.ArgInfo(
                    type="float64", name="distance", description="Buffer distance"
                ),
            ],
            returns="geometry",
        ),
    )
    docstring = _codegen.generate_function_docstring(func)
    expected = '''\
"""ST_Buffer

        Returns a buffered geometry.

        Args:
            geom: Input geometry
            distance: Buffer distance

        See Also:
            https://sedona.apache.org/sedonadb/latest/reference/sql/st_buffer/
        """'''
    assert docstring == expected


def test_generate_function_docstring_variadic():
    # Variadic mode is triggered when parameter names conflict across kernels
    func = _codegen.FunctionInfo(
        name="st_buffer",
        title="ST_Buffer",
        description="Creates a buffer.",
        kernels=[],
        kernel_info=_codegen.KernelInfo(
            args=[],
            returns="geometry",
            variadic=True,
            kernel_signatures=[
                "geom (geometry), distance (float64)",
                "geom (geometry), distance (float64), params (string)",
                "geom (geography), distance (float64)",
                "geom (geography), distance (float64), num_quad_segs (integer)",
                "geom (geography), distance (float64), params (string)",
            ],
        ),
    )
    docstring = _codegen.generate_function_docstring(func)
    expected = '''\
"""ST_Buffer

        Creates a buffer.

        Variants:
            - geom (geometry), distance (float64)
            - geom (geometry), distance (float64), params (string)
            - geom (geography), distance (float64)
            - geom (geography), distance (float64), num_quad_segs (integer)
            - geom (geography), distance (float64), params (string)

        See Also:
            https://sedona.apache.org/sedonadb/latest/reference/sql/st_buffer/
        """'''
    assert docstring == expected


def test_generate_sources(sample_qmd_path: Path):
    docs_sql = sample_qmd_path.parent
    with TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"

        result = _codegen.generate_sources(docs_sql, output_dir)

        assert result.total_functions == 1
        assert result.geo_method_count == 1
        assert len(result.generated_files) == 5

        # Verify generated files compile as valid Python
        for file_path in result.generated_files:
            code = file_path.read_text()
            compile(code, str(file_path), "exec")
