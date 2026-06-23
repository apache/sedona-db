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
# Zarr Extension (Python) API Reference

`sedonadb-zarr` reads [Zarr](https://zarr.dev/) groups into the SedonaDB
raster type. Install it alongside SedonaDB, register the extension on your
connection, and read a group through its format spec:

```python
import sedona.db
import sedonadb_zarr

sd = sedona.db.connect()
sd.register(sedonadb_zarr.ZarrExtension())
sd.read("file:///path/to/foo.zarr").show()
```

For an end-to-end walkthrough, see
[Working with Zarr and NDArray data in SedonaDB](../working-with-zarr-ndarray-sedonadb.md).

::: sedonadb_zarr.ZarrExtension

::: sedonadb_zarr.Zarr
    options:
      members:
        - with_options
