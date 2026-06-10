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

# Working with Zarr and NDArray data in SedonaDB

SedonaDB's raster type is **N-dimensional**: a band isn't limited to a 2-D
`(y, x)` grid — it can carry additional axes such as `time` or `band`. This
makes it a natural fit for *datacubes*: climate reanalyses, satellite time
series, and model outputs.

The `sedonadb-zarr` extension reads [Zarr](https://zarr.dev/) groups —
local or in cloud object storage — directly into that raster type, so a
datacube becomes a table you can query in SQL.

This page walks through loading a Zarr datacube, inspecting its dimensions,
slicing out a 2-D plane, and handing the result to NumPy.

## Install

`sedonadb-zarr` is an extension, distributed separately from the core
SedonaDB package:

```bash
pip install "apache-sedona[db]" sedonadb-zarr zarr numpy geoarrow-pyarrow
```

`zarr` and `numpy` are used to build the sample datacube below;
`geoarrow-pyarrow` is required to export query results with
`to_arrow_table()`.

## Create a sample datacube

If you don't already have a Zarr group handy, this creates a small
`[time, y, x]` temperature cube to follow along with:

```python
import numpy as np
import zarr  # zarr-python >= 3.0

store = "/tmp/temperature.zarr"
root = zarr.open_group(store, mode="w")
arr = root.create_array(
    "temperature",
    shape=(3, 10, 20),
    chunks=(3, 5, 5),
    dtype="uint16",
    dimension_names=["time", "y", "x"],
)
arr[:] = np.arange(3 * 10 * 20, dtype="uint16").reshape(3, 10, 20)
```

The `chunks=(3, 5, 5)` argument leaves `time` un-chunked — every chunk
holds all three time steps — and tiles the `10 × 20` spatial grid into a
`2 × 4` grid of `5 × 5` patches. That's `8` chunks in total, and that
chunking is what determines how the data loads, as we'll see next.

## Connect and load

Register the extension on your connection, then read the Zarr group with
its format spec:

```python
import sedona.db
import sedonadb_zarr

sd = sedona.db.connect()
sedonadb_zarr.register(sd)

df = sd.read_format(sedonadb_zarr.ZarrFormatSpec(), f"file://{store}")
df.to_view("cube")
```

`sedonadb-zarr` emits **one row per Zarr chunk**, with one band per array
in the group. Our cube tiles into eight chunks, so it loads as eight rows
— each a `[3, 5, 5]` tile holding all three time steps for one `5 × 5`
spatial patch:

```python
sd.sql("SELECT COUNT(*) AS n_chunks FROM cube").show()
```

```text
┌──────────┐
│ n_chunks │
╞══════════╡
│        8 │
└──────────┘
```

## Inspect the dimensions

The dimension-query functions read the raster's schema only — **no pixel
data is loaded** — so they return near-instantly even against a large
remote cube. Each row reports its **chunk's** shape, not the full cube
extent. All eight chunks share the same shape here, so we look at one:

```python
sd.sql("""
    SELECT
        RS_NumDimensions(raster)   AS ndim,
        RS_DimNames(raster)        AS dims,
        RS_Shape(raster)           AS shape,
        RS_DimSize(raster, 'time') AS n_time
    FROM cube
    LIMIT 1
""").show()
```

```text
┌──────┬──────────────┬───────────┬────────┐
│ ndim ┆ dims         ┆ shape     ┆ n_time │
╞══════╪══════════════╪═══════════╪════════╡
│    3 ┆ [time, y, x] ┆ [3, 5, 5] ┆      3 │
└──────┴──────────────┴───────────┴────────┘
```

Each chunk is 3-dimensional (`[time, y, x]`) with all three time steps
(`n_time = 3`) and a `5 × 5` spatial footprint — one tile of the full
`10 × 20` grid.

## Slice out a 2-D plane

`RS_Slice` selects a single index along a named dimension and drops it.
Picking time step `1` turns every `[3, 5, 5]` chunk into a `[y, x]` plane
— the `5 × 5` patch at that time step, one per row:

```python
sliced = sd.sql("SELECT RS_Slice(raster, 'time', 1) AS plane FROM cube")
sliced.to_view("plane")

sd.sql("SELECT RS_DimNames(plane) AS dims, RS_Shape(plane) AS shape FROM plane LIMIT 1").show()
```

```text
┌────────┬────────┐
│ dims   ┆ shape  │
╞════════╪════════╡
│ [y, x] ┆ [5, 5] │
└────────┴────────┘
```

`RS_Slice` needs pixel data, so SedonaDB resolves each row's Zarr chunk on
demand before slicing — you never call a loader yourself. Because `time`
isn't chunked, the slice index is the global time step: `0`, `1`, or `2`
all select a real plane.

Related functions reshape a cube in other ways:

- `RS_SliceRange(raster, dim, start, end)` keeps a contiguous range of a
  dimension instead of a single index.
- `RS_DimToBand(raster, dim)` / `RS_BandToDim(raster, dim_name)` move an
  axis between the dimension list and the band list.

## Bring a slice into NumPy

A raster band carries its bytes, shape, and pixel type, so decoding a
materialized band to NumPy is a small `frombuffer` + `reshape`:

```python
_NP_DTYPE = {
    1: np.uint8, 2: np.uint16, 3: np.int16, 4: np.uint32, 5: np.int32,
    6: np.float32, 7: np.float64, 8: np.uint64, 9: np.int64, 10: np.int8,
}

def band_to_numpy(raster, band_index=0):
    band = raster["bands"][band_index]
    dtype = _NP_DTYPE[band["data_type"]]
    return np.frombuffer(band["data"], dtype=dtype).reshape(band["source_shape"])

planes = sliced.to_arrow_table()["plane"]
raster = planes[0].as_py()  # each row is one 5x5 spatial tile at time step 1
print(band_to_numpy(raster))
```

```text
[[200 201 202 203 204]
 [220 221 222 223 224]
 [240 241 242 243 244]
 [260 261 262 263 264]
 [280 281 282 283 284]]
```

Each of the eight rows decodes to a `5 × 5` spatial tile at time step `1`;
this is one of them. Rows correspond to chunks rather than a guaranteed
order, so apply your own `ORDER BY` (or carry a chunk identifier) if you
need to know which spatial tile a given plane covers.

## Reading from cloud storage

The same code reads a datacube over S3 or HTTP(S) — only the URI changes:

```python
df = sd.read_format(sedonadb_zarr.ZarrFormatSpec(), "s3://my-bucket/temperature.zarr")
```

Supported URI schemes are `file://` (and bare local paths), `s3://`,
`http://`, and `https://`. S3 credentials are read from the standard AWS
environment variables (for example `AWS_ACCESS_KEY_ID` and `AWS_REGION`).

Because each row corresponds to one chunk, a `LIMIT` or row filter
directly bounds how many chunks SedonaDB fetches — handy for sampling a
large remote cube before committing to a full scan.
