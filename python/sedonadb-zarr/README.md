<!--
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

# sedonadb-zarr

Zarr support for [SedonaDB](https://sedona.apache.org/) as an opt-in
plugin package. Adds the `sd_read_zarr` SQL UDTF that reads Zarr v3
groups (with sharding, vlen-utf8 dims, etc.) as a column of N-D
rasters.

```python
import sedonadb
import sedonadb_zarr

con = sedonadb.connect()
sedonadb_zarr.register(con)

df = con.sql("SELECT count(*) FROM sd_read_zarr('file:///path/to/foo.zarr')")
df.show()
```

The main `sedonadb` package does not bundle Zarr support — applications
that don't import `sedonadb_zarr` pay no zarr build or runtime cost.

## Status

- SQL UDTF (`sd_read_zarr`): supported.
- `ExternalFormatSpec` (`con.read_format(ZarrFormatSpec(), uri)`): the
  Rust impl exists in `sedona-raster-zarr` and is callable from Rust
  code; a Python wrapper that exposes it via `con.read_format` is a
  follow-up.

## Architecture

This is a maturin-built mixed Rust/Python package. The Rust side is a
thin shim around `sedona-raster-zarr` that exposes a `register_udtf`
PyO3 function. The Python side calls it from `register(con)`.

See the [design notes](https://...) (TODO) for the plugin pattern this
package follows. The same shape applies to future formats (`sedonadb-cog`,
`sedonadb-icechunk`, …).
