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

import json
from typing import Optional, Literal, Union, Tuple, Iterable

from sedonadb.dataframe import DataFrame
from sedonadb.utility import sedona  # noqa: F401


class TableFunctions:
    def __init__(self, ctx):
        self._ctx = ctx

    def sd_random_geometry(
        self,
        geom_type: Optional[
            Literal[
                "Geometry",
                "Point",
                "LineString",
                "Polygon",
                "MultiPoint",
                "MultiLineString",
                "MultiPolygon",
                "GeometryCollection",
            ]
        ] = None,
        num_rows: Optional[int] = None,
        *,
        num_vertices: Union[int, Tuple[int, int], None] = None,
        num_parts: Union[int, Tuple[int, int], None] = None,
        size: Union[float, Tuple[float, float], None] = None,
        bounds: Optional[Iterable[float]] = None,
        hole_rate: Optional[float] = None,
        empty_rate: Optional[float] = None,
        null_rate: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> DataFrame:
        """
        Generate a DataFrame with random geometries for testing purposes.
        This function creates a DataFrame containing randomly generated geometries with
        configurable parameters for geometry type, size, complexity, and spatial distribution.
        Returns a DataFrame with columns 'id', 'dist', and 'geometry' containing randomly
        generated geometries and distances.

        Parameters
        ----------
        geom_type : str, default "Point"
            The type of geometry to generate. One of "Geometry",
            "Point", "LineString",  "Polygon", "MultiPoint", "MultiLineString",
            "MultiPolygon", or "GeometryCollection".
        num_rows : int, default 1024
            Number of rows to generate.
        num_vertices : int or tuple of (int, int), default 4
            Number of vertices per geometry. If a tuple, specifies (min, max) range.
        num_parts : int or tuple of (int, int), default (1, 3)
            Number of parts for multi-geometries. If a tuple, specifies (min, max) range.
        size : float or tuple of (float, float), default (1.0, 10.0)
            Spatial size of geometries. If a tuple, specifies (min, max) range.
        bounds : iterable of float, default [0.0, 0.0, 100.0, 100.0]
            Spatial bounds as [xmin, ymin, xmax, ymax] to constrain generated geometries.
        hole_rate : float, default 0.0
            Rate of polygons with holes, between 0.0 and 1.0.
        empty_rate : float, default 0.0
            Rate of empty geometries, between 0.0 and 1.0.
        null_rate : float, default 0.0
            Rate of null geometries, between 0.0 and 1.0.
        seed : int, optional
            Random seed for reproducible geometry generation. If omitted, the result is
            non-deterministic.

        Examples
        --------
        >>> sd = sedona.db.connect()
        >>> sd.funcs.table.sd_random_geometry("Point", 1, seed=938).show()
        ┌───────┬───────────────────┬────────────────────────────────────────────┐
        │   id  ┆        dist       ┆                  geometry                  │
        │ int32 ┆      float64      ┆                  geometry                  │
        ╞═══════╪═══════════════════╪════════════════════════════════════════════╡
        │     0 ┆ 58.86528701627309 ┆ POINT(94.77686827801787 17.65107885959438) │
        └───────┴───────────────────┴────────────────────────────────────────────┘
        """

        args = {
            "bounds": bounds,
            "empty_rate": empty_rate,
            "geom_type": geom_type,
            "null_rate": null_rate,
            "num_parts": num_parts,
            "hole_rate": hole_rate,
            "seed": seed,
            "size": size,
            "num_rows": num_rows,
            "num_vertices": num_vertices,
        }

        args = {k: v for k, v in args.items() if v is not None}

        return self._ctx.sql(f"SELECT * FROM sd_random_geometry('{json.dumps(args)}')")
