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

# SedonaDB + FlatGeobuf

This page explains how to read FlatGeobuf files with SedonaDB.

FlatGeobuf is a cloud-optimized binary format for geographic vector data designed for fast streaming and spatial filtering over HTTP.

It has a built-in spatial index, is easily compactible, contains CRS information, and is supported by many engines.

The examples on this page show you how to query FlatGeobuf files with SedonaDB over HTTP.


```python
import sedona.db

sd = sedona.db.connect()
```

# Read Microsoft Buildings FlatGeobuf data with SedonaDB

The Microsoft buildings dataset is a comprehensive open dataset of building footprints extracted from satellite imagery using computer vision and deep learning.

Here's how to read the Microsoft buildings dataset into a SedonaDB DataFrame and print a few rows.


```python
url = "https://github.com/geoarrow/geoarrow-data/releases/download/v0.2.0/microsoft-buildings_point.fgb.zip"
df = sd.read_pyogrio(url)
df.show(3)
```

    ┌─────────────────────────────────┐
    │           wkb_geometry          │
    │             geometry            │
    ╞═════════════════════════════════╡
    │ POINT(-97.16154292 26.08759861) │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ POINT(-97.1606625 26.08481)     │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ POINT(-97.16133375 26.08519809) │
    └─────────────────────────────────┘


You can see that the Microsoft Buildings dataset contains the building centroids.

Take a look at the schema and see how it contains the `wkb_geometry` column and the CRS.


```python
df.schema
```




    SedonaSchema with 1 field:
      wkb_geometry: geometry<Wkb(ogc:crs84)>



Now lets see how to read another FlatGeobuf dataset.

# Read Vermont boundary FlatGeobuf data with SedonaDB

The Vermont boundary dataset contains the polygon for the state of Vermont.

The following example shows how to read the Vermont FlatGeobuf dataset and plot it.

```python
url = "https://raw.githubusercontent.com/geoarrow/geoarrow-data/v0.2.0/example-crs/files/example-crs_vermont-utm.fgb"
sd.read_pyogrio(url).to_pandas().plot()
```
