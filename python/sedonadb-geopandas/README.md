<!---
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

# sedonadb-geopandas

A GeoPandas-compatible API on top of [SedonaDB](https://sedona.apache.org/sedonadb/).

The goal is to let existing GeoPandas code run against SedonaDB's relational
engine with minimal changes, by providing `GeoDataFrame` / `GeoSeries` wrappers
whose methods mirror GeoPandas but delegate to SedonaDB expressions.

```python
import geopandas
import sedonadb_geopandas as sgpd

gdf = sgpd.from_geopandas(geopandas.read_file("cities.geojson"))
regions = sgpd.from_geopandas(geopandas.read_file("regions.geojson"))
big = gdf[gdf["pop"] > 1_000_000]          # boolean-mask filter
gdf["density"] = gdf["pop"] / gdf["area"]  # assign a computed column
buffered = gdf.geometry.buffer(0.5)        # element-wise .geo operation
web = gdf.to_crs("EPSG:3857")              # reproject (CRS tracked through)

joined = gdf.sjoin(regions, predicate="within")  # spatial join
zones = joined.dissolve(by="region")              # group and union geometry

result = zones.to_geopandas()               # back to a real GeoDataFrame
```

## Intentional differences from GeoPandas

This is a compatibility layer over a lazy, relational engine, so it is
deliberately *not* identical to GeoPandas:

- **Lazy, not eager**: operations build a query; data materializes on
  `to_geopandas()` / `to_pandas()` / display.
- **No row index / alignment**: there is no pandas `Index`; joins and filters
  are positional/relational, not index-aligned. Consequently `sjoin()`
  produces no `index_left`/`index_right` column, and `dissolve()` leaves the
  group keys as ordinary columns instead of moving them into the index.
- **Immutable under the hood**: "in-place" style operations return a new frame.
  A `Series` read from a frame stays usable across assignments that only *add*
  columns (so `g = gdf.geometry` can supply `g.area`, `g.length`, ... in turn),
  but replacing a column or filtering rebinds the frame, and a `Series` read
  before that is stale and raises rather than silently resolving to different
  values.
- **Columns cannot be mixed across frames**: without row alignment, combining
  columns from two different frames raises rather than guessing. Join first.
- **Plotting and arbitrary `apply`**: use the `to_geopandas()` escape hatch and
  operate on the materialized result.
- **Assignment takes a column or a scalar, not a bare expression**: a SedonaDB
  expression records no origin, so one built from another frame would resolve
  against the destination and silently write the wrong values. Assign a `Series`
  read from the same frame, or a scalar (a geometry included). For anything the
  wrapper does not cover, drop to the SedonaDB `DataFrame` API directly.

Division follows pandas rather than SQL: `/` is true division, so integer
columns do not silently truncate. `//` is not implemented, since SQL division
truncates toward zero where Python floors. Duration arithmetic whose result
overflows the 64-bit tick range raises when the result is computed, as in
pandas 3 (pandas 2 silently wraps integer overflow, and pandas clamps finite
positive float overflow to `Timedelta.max`; this layer raises for both). An
integer operand that itself exceeds the 64-bit range raises `OverflowError`
immediately, as in every pandas version. Mixed-unit duration
operands are rebuilt in the column's own unit when exactly representable,
rather than widening to the engine's interval type.

`GeoSeries` properties and methods follow GeoPandas, with a few differences
in form: `to_wkt()` writes equivalent but differently spaced WKT
(`POINT(1 2)`), `to_wkb()` writes ISO WKB (GeoPandas' `flavor="iso"`),
`is_simple` and `boundary` answer for geometry collections where GEOS leaves
them undefined, and `x`/`y`/`z` on a non-point raise when the result is
computed rather than when the property is read.

Binary geometry operations (`intersects`, `within`, `distance`,
`intersection`, ...) take either a single Shapely geometry, which takes the
column's CRS, or a `GeoSeries` from the same frame, matched row by row as with
GeoPandas' `align=False`; there is no index to align on. `touches` inherits
an engine issue with geometry collections that mix dimensions
(apache/sedona-db#1383).

`sjoin()` requires both geometry columns to share a CRS: GeoPandas warns on a
mismatch and joins anyway, which is almost always a mistake, so this raises and
points at `to_crs()`. `on_attribute` is not supported yet.

`dissolve()` aggregates non-geometry columns with `"first"`, which is an
unordered aggregate: it returns *some* value from the group rather than the one
from the first row, and unlike GeoPandas it does not skip missing values, so a
group containing a null or NaN may aggregate to that. Dissolving an empty frame
without a group key returns one row — empty geometry collection, null attribute
values — rather than zero rows, because that is what a grouping-free SQL
aggregate produces, and a group mixing 2D and 3D geometries raises rather than
being promoted to 3D. Grouping is observed-only: unused categories of a
categorical key do not produce empty groups the way GeoPandas' default
`observed=False` does, because the category domain does not survive a relational
aggregation.

`read_parquet()`, `read_file()` and `to_parquet()` follow GeoPandas, with
differences that come from SedonaDB's readers and GeoParquet writer:

- The active geometry column of a file that is read, and the `primary_column`
  written to one, follow SedonaDB's heuristic (a column named `geometry`,
  `geography`, `geom` or `geog`, else the first geometry column) rather than the
  file's metadata or the frame's active geometry. This matters only for files
  with several geometry columns (apache/sedona-db#1404).
- A file that declares an unknown CRS (`"crs": null`, which GeoPandas writes for
  a frame without one) reads back as OGC:CRS84, and `to_parquet()` refuses a
  geometry column without a CRS (apache/sedona-db#1404). EPSG:4326 is written as
  OGC:CRS84.
- Reading a GeoParquet 1.1 file keeps its covering bounding-box column, which
  `to_parquet(write_covering_bbox=True)` recomputes. A covering is written for
  every geometry column, not only the active one, and exactly with
  `schema_version="1.1.0"`.
- `read_file()` keeps the source's name for the geometry column (`geom` in a
  GeoPackage) where GeoPandas renames it to `geometry`: renaming it would break
  later column selections (apache/sedona-db#1405).
- `to_parquet()` writes one file and needs a file extension such as `.parquet`,
  since SedonaDB writes a directory of files to a path without one.

See the SedonaDB "Migrating from GeoPandas" guide for the relational model that
underlies each method.
