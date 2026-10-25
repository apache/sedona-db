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
parts = zones.explode()                           # one row per geometry part

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

The numeric parameters of geometry methods (`segmentize()`, `set_precision()`,
`concave_hull()`, `force_3d()`, `snap()`, `get_geometry()`) take a number or a
numeric `Series` of the same frame, applied row by row. `segmentize()` only
adds vertices, where GEOS also repairs an invalid polygon and turns a one-part
multi-line into a line; `force_3d()` keeps Z values and drops M, where Shapely
leaves a geometry with M but no Z as it is and replaces the Z values of one
with both; and `snap()` leaves an empty geometry as it is, where GEOS can turn
it into a 3D empty, or one of another type (`POLYGON EMPTY` becomes
`LINEARRING Z EMPTY`). `set_precision()` modes other than `"valid_output"` are
not supported.

The affine transformations (`affine_transform()`, `translate()`, `rotate()`,
`scale()`, `skew()`) give Shapely's coordinates to the last bit, but keep M
values, which Shapely drops (and overwrites, with Z, as NaN in a geometry
with both). `interpolate()` and `project()` raise for a multi-line when the
result is computed, as the engine works along single lines only, and
`interpolate()` gives a 2D point on a 3D line.

`explode()` takes parts one level deep and drops rows whose geometry has no
parts, as GeoPandas does. Without an index it cannot number the parts
(`index_parts=True`), and it explodes the active geometry column only:
GeoPandas explodes the active geometry even when `column=` names another
geometry column, and explodes non-geometry columns as lists.

`sjoin()` requires both geometry columns to share a CRS: GeoPandas warns on a
mismatch and joins anyway, which is almost always a mistake, so this raises and
points at `to_crs()`. `on_attribute` is not supported yet.

Without an index, `drop()` and `rename()` work on columns only, `rename()`
raises rather than producing duplicate column names, and a sort from
`sort_values()` holds until an operation that does not preserve row order (a
join or an aggregation). `astype()` to a string keeps missing values missing,
like pandas' `"string"` dtype rather than `str`, and formats floats the way
Arrow does (`"1"`, `"NaN"`); `astype(object)` leaves the values as they are,
since an Arrow-backed column has no object type.

Reductions (`sum()`, `mean()`, `min()`, `max()`, `count()`, `median()`,
`std()`, `var()`, `quantile()`, `nunique()`) compute their result, skipping
missing values (NaN included) as pandas does by default; an integer column keeps
its integer type where pandas, having turned its missing values into NaN, would
report a float. `unique()` and `value_counts()` cannot follow the order of first
appearance, since a lazy frame has no row order: `value_counts()` breaks ties by
value.

`where()`, `mask()`, and `clip()` keep a column to one kind of value: a
replacement or bound of another kind (text in a number column, say) raises,
where pandas would turn the column into Python objects. `round()` rounds half to
even as pandas (NumPy) does, and raises for a non-numeric column, as pandas 2
does (pandas 3 returns it unchanged). `dropna()` and `head()` on a column return
a column of a new frame, which, like any column of another frame, can't be
combined with the original's; `head()` takes no negative count, since there is
no row order to count back from.

`merge()` joins on columns as pandas does, including in matching a missing key
(NaN or None) with a missing key, unlike SQL; without an index, its rows don't
keep the left frame's order (`sort=True` orders them by key), and
`left_index`/`right_index` are not supported. A key both frames share keeps the
left's type in an inner or left join and the right's in a right join; in an
outer join it takes a type both sides fit, which pandas chooses by whether some
row lacks the left's key, but a lazy frame's types can't depend on its rows.
When both frames have a geometry column of the same name, the left's stays
active under its suffixed name, where
GeoPandas keeps naming the unsuffixed column. `concat()` stacks rows only
(`axis=0`), matching columns by name; a column holds one kind of value, so
values of different kinds (numbers and text, say) raise rather than make a
column of Python objects, and geometry columns must share their CRS, as in
GeoPandas: one without a CRS takes the others' only if it holds no geometry,
which is checked as the result is computed (GeoPandas relabels its coordinates
whatever they are).

`groupby()` aggregates as pandas does, skipping missing values and treating
NaN and None (and -0.0 and 0.0) as one key, but there is no index to hold the
keys, so they are columns of the result whatever `as_index` says. Several
aggregations of one column (`agg({"v": ["sum", "max"]})`) would make two-level
column names, which aren't supported: name each aggregate instead
(`agg(total=("v", "sum"), largest=("v", "max"))`). Without a row order,
`first()` and `last()` give a value of each group that is not missing, the
smallest and the largest, rather than the first and last rows'.

`fillna()` follows the same rule: a scalar fills the columns that hold its kind
of value and leaves the others as they are (pandas would turn them into Python
objects, and GeoPandas refuses a non-geometry value for a geometry column), and
a dict value of another kind raises. `drop_duplicates()` keeps one row of each
group of duplicates, but with no row order, `keep="first"` and `keep="last"`
cannot say which. `dropna()` drops rows only (`axis=0`), since choosing columns
would take computing the data first. `assign()` evaluates a value that is not a
callable on the frame as it was, even if an earlier keyword replaced a column it
reads, as pandas does.

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
- EPSG:4326 is written as OGC:CRS84.
- The covering (bounding-box) columns a file's metadata declares are left out
  on read, as in GeoPandas, for local files only: a remote file's metadata is
  not read, so its coverings stay. A covering is written for every geometry
  column, not only the active one, and exactly with `schema_version="1.1.0"`.
- `read_file()` keeps the source's name for the geometry column (`geom` in a
  GeoPackage) where GeoPandas renames it to `geometry`: renaming it would break
  later column selections (apache/sedona-db#1405).
- `to_parquet()` writes one local file and needs a file extension such as
  `.parquet`, since SedonaDB writes a directory of files to a path without one.
  It cannot write to a remote URL (`s3://`, `gs://`) yet
  (apache/sedona-db#1440); reading one works, with the object store's options
  passed as `storage_options` using SedonaDB's keys (for example
  `{"aws.skip_signature": True, "aws.region": "us-west-2"}`), not fsspec's.

See the SedonaDB "Migrating from GeoPandas" guide for the relational model that
underlies each method.
