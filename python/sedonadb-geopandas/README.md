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
big = gdf[gdf["pop"] > 1_000_000]          # boolean-mask filter
gdf["density"] = gdf["pop"] / gdf["area"]  # assign a computed column
buffered = gdf.geometry.buffer(0.5)        # element-wise .geo operation
web = gdf.to_crs("EPSG:3857")              # reproject (CRS tracked through)

zones = gdf.dissolve(by="region")          # group and union geometry

result = zones.to_geopandas()               # back to a real GeoDataFrame
```

## Intentional differences from GeoPandas

This is a compatibility layer over a lazy, relational engine, so it is
deliberately *not* identical to GeoPandas:

- **Lazy, not eager**: operations build a query; data materializes on
  `to_geopandas()` / `to_pandas()` / display.
- **No row index / alignment**: there is no pandas `Index`; joins and filters
  are positional/relational, not index-aligned. Consequently `dissolve()`
  leaves the group keys as ordinary columns instead of moving them into the
  index.
- **Immutable under the hood**: "in-place" style operations return a new frame.
  Assigning a column with `gdf["x"] = ...` rebinds the frame, so a `Series` read
  before the assignment is stale and cannot be combined with later reads.
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
truncates toward zero where Python floors.

`dissolve()` aggregates non-geometry columns with `"first"`, which is an
unordered aggregate: it returns *some* value from the group rather than the one
from the first row, and unlike GeoPandas it does not skip missing values, so a
group containing a null or NaN may aggregate to that. Dissolving an empty frame
without a group key returns one row — empty geometry collection, null attribute
values — rather than zero rows, because that is what a grouping-free SQL
aggregate produces, and a group mixing 2D and 3D geometries raises rather than
being promoted to 3D.

See the SedonaDB "Migrating from GeoPandas" guide for the relational model that
underlies each method.
