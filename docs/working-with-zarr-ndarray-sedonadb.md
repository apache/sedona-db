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
    shape=(3, 4, 5),
    chunks=(1, 4, 5),
    dtype="uint16",
    dimension_names=["time", "y", "x"],
)
arr[:] = np.arange(3 * 4 * 5, dtype="uint16").reshape(3, 4, 5)
```

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

Each array in the Zarr group becomes one row with a `raster` column.

## Inspect the dimensions

The dimension-query functions read the raster's schema only — **no pixel
data is loaded** — so they return instantly even against a large remote
cube:

```python
sd.sql("""
    SELECT
        RS_NumDimensions(raster)   AS ndim,
        RS_DimNames(raster)        AS dims,
        RS_Shape(raster)           AS shape,
        RS_DimSize(raster, 'time') AS n_time
    FROM cube
""").show()
```

```text
┌──────┬──────────────┬───────────┬────────┐
│ ndim ┆ dims         ┆ shape     ┆ n_time │
╞══════╪══════════════╪═══════════╪════════╡
│    3 ┆ [time, y, x] ┆ [3, 4, 5] ┆      3 │
└──────┴──────────────┴───────────┴────────┘
```

## Slice out a 2-D plane

`RS_Slice` selects a single index along a named dimension and drops it,
turning the `[time, y, x]` cube into a `[y, x]` raster:

```python
sliced = sd.sql("SELECT RS_Slice(raster, 'time', 1) AS plane FROM cube")
sliced.to_view("plane")

sd.sql("SELECT RS_DimNames(plane) AS dims, RS_Shape(plane) AS shape FROM plane").show()
```

```text
┌────────┬────────┐
│ dims   ┆ shape  │
╞════════╪════════╡
│ [y, x] ┆ [4, 5] │
└────────┴────────┘
```

`RS_Slice` needs pixel data, so SedonaDB resolves the Zarr chunks for that
time step on demand before slicing — you never call a loader yourself.

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

raster = sliced.to_arrow_table()["plane"][0].as_py()
print(band_to_numpy(raster))
```

```text
[[20 21 22 23 24]
 [25 26 27 28 29]
 [30 31 32 33 34]
 [35 36 37 38 39]]
```

That's the `time = 1` plane of the cube (values `20`–`39`).

## Reading from cloud storage

The same code reads a datacube over S3 or HTTP(S) — only the URI changes:

```python
df = sd.read_format(sedonadb_zarr.ZarrFormatSpec(), "s3://my-bucket/temperature.zarr")
```

Supported URI schemes are `file://` (and bare local paths), `s3://`,
`http://`, and `https://`. S3 credentials are read from the standard AWS
environment variables (for example `AWS_ACCESS_KEY_ID` and `AWS_REGION`).
