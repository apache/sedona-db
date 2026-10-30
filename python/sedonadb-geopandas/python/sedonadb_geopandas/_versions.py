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
"""`show_versions()`, as GeoPandas has it."""

import importlib.metadata
import platform
import sys

# The packages this one uses, needs for GeoPandas interop, or can use (for
# reading and writing files, plotting and maps).
_DEPENDENCIES = (
    "sedonadb-geopandas",
    "sedonadb-expr",
    "pyarrow",
    "shapely",
    "pyproj",
    "numpy",
    "pandas",
    "geopandas",
    "geoarrow-pyarrow",
    "pyogrio",
    "matplotlib",
    "folium",
    "mapclassify",
)


def _version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _section(title, items):
    width = max(len(name) for name, _ in items)
    lines = [title, "-" * len(title)]
    lines += [f"{name:<{width}}: {value}" for name, value in items]
    return "\n".join(lines)


def show_versions():
    """Print the versions of Python, SedonaDB and the packages this one uses,
    as `geopandas.show_versions` does for GeoPandas', to include in a bug
    report."""
    import sedonadb

    sections = [
        _section(
            "SYSTEM INFO",
            [
                ("python", sys.version.replace("\n", " ")),
                ("executable", sys.executable),
                ("machine", platform.platform()),
            ],
        ),
        _section(
            "SEDONADB INFO",
            [
                ("sedonadb", sedonadb.__version__),
                ("features", ", ".join(getattr(sedonadb, "__features__", [])) or None),
            ],
        ),
        _section(
            "PYTHON DEPENDENCIES",
            [(name, _version(name)) for name in _DEPENDENCIES],
        ),
    ]
    print("\n" + "\n\n".join(sections))
