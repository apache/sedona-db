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

Zarr support for [SedonaDB](https://sedona.apache.org/) as an opt-in plugin package. Reads Zarr v3 groups (with sharding, vlen-utf8 dims, etc.) as a column of N-D rasters via two equivalent surfaces:

```python
import sedonadb
import sedonadb_zarr

con = sedonadb.connect()
sedonadb_zarr.register(con)

# SQL UDTF:
con.sql("SELECT count(*) FROM sd_read_zarr('file:///path/to/foo.zarr')").show()

# DataFrame API via ExternalFormatSpec:
con.read_format(sedonadb_zarr.ZarrFormatSpec(), 'file:///path/to/foo.zarr').show()
```

The main `sedonadb` package does not bundle Zarr support — applications that don't import `sedonadb_zarr` pay no zarr build or runtime cost.

## Architecture

This is a maturin-built mixed Rust/Python package. The Rust side is a thin shim around `sedona-raster-zarr` that exposes a `register_udtf` PyO3 function and a `PyZarrChunkReader` class implementing `__arrow_c_stream__`. The Python side defines `ZarrFormatSpec(ExternalFormatSpec)` and a `register(con)` helper that wires the UDTF onto a session.

The same plugin shape applies to future formats (`sedonadb-cog`, `sedonadb-icechunk`, …).
