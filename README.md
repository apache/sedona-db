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

# SedonaDB

SedonaDB is a high-performance, dependency-free geospatial compute engine.

You can easily run SedonaDB locally or in the cloud.  The first release supports a core set of vector operations, but the full-suite of common vector and raster computations will be supported soon.

SedonaDB only runs on a single machine, so it’s perfect for processing smaller datasets.  You can use SedonaSpark, SedonaFlink, or SedonaSnow for operations on larger datasets.

## Install

You can install Python SedonaDB with PyPI:

```sh
pip install "apache-sedona[db]"
```

## Overture buildings example

This section shows how to query the Overture buildings data.

Start by establishing a connection:

```python
import sedona.db
sd = sedona.db.connect()
```

Set some AWS environment variables to access the data:

```python
os.environ["AWS_SKIP_SIGNATURE"] = "true"
os.environ["AWS_DEFAULT_REGION"] = "us-west-2"
```

Read the dataset into a Python SedonaDB `DataFrame`. This is lazy: even though the Overture buildings table contains millions of rows, SedonaDB will only fetch the data required for the query.

```python
df = sd.read_parquet(
    "s3://overturemaps-us-west-2/release/2025-08-20.0/theme=buildings/type=building/"
)
```

Now run a query to compute the centroids of tall buildings (above 20 meters) in New York City:

```python
nyc_bbox_wkt = (
    "POLYGON((-74.2591 40.4774, -74.2591 40.9176, -73.7004 40.9176, -73.7004 40.4774, -74.2591 40.4774))"
)

sd.sql(f"""
SELECT
    id,
    height,
    num_floors,
    roof_shape,
    ST_Centroid(geometry) as centroid
FROM
    buildings
WHERE
    is_underground = FALSE
    AND height IS NOT NULL
    AND height > 20
    AND ST_Intersects(geometry, ST_SetSRID(ST_GeomFromText('{nyc_bbox_wkt}'), 4326))
LIMIT 5;
""").show()
```

Here's the query output:

```
┌─────────────────────────┬────────────────────┬────────────┬────────────┬─────────────────────────┐
│            id           ┆       height       ┆ num_floors ┆ roof_shape ┆         centroid        │
│         utf8view        ┆       float64      ┆    int32   ┆  utf8view  ┆     wkb <ogc:crs84>     │
╞═════════════════════════╪════════════════════╪════════════╪════════════╪═════════════════════════╡
│ 1b9040c2-2e79-4f56-aba… ┆               22.4 ┆            ┆            ┆ POINT(-74.230407502993… │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ 1b5e1cd2-d697-489e-892… ┆               21.5 ┆            ┆            ┆ POINT(-74.231451103592… │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ c1afdf78-bf84-4b8f-ae1… ┆               20.9 ┆            ┆            ┆ POINT(-74.232593032240… │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ 88f36399-b09f-491b-bb6… ┆               24.5 ┆            ┆            ┆ POINT(-74.231878209597… │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ df37a283-f5bd-4822-a05… ┆ 24.154542922973633 ┆            ┆            ┆ POINT(-74.241910239840… │
└─────────────────────────┴────────────────────┴────────────┴────────────┴─────────────────────────┘
```

## Features of SedonaDB

SedonaDB has several advantages:

* The code is written in Rust and runs fast.
* It supports both vector and raster functions.  You can use a single library to access a full-suite of spatial functionality.
* It always propagates the coordinate reference system (CRS).
* It supports legacy and modern file formats.
* It has Python and SQL APIs and users can seamlessly switch between them.
* It’s easily extensible and customized.
* It is interoperable with other PyArrow compatible libraries like GeoPandas, DuckDB, and Polars.
* It has a great community of maintainers and encourages external contributions.

## Contributing

There are many different ways to contribute to SedonaDB:

* Join the Discord and chat with us
* Open a GitHub Discussion with questions or ideas
* Work on an existing issue.  Just comment “take” on the issue and we will assign you the task.
* Brainstorm features with the contributors and then contribute a pull request.

The contributors meet on a monthly basis and we’re happy to add you to the call if you would like to join the community!

## Community

SedonaDB is a subproject of Apache Sedona, an Apache Software Foundation project.

The project is governed by the Apache Software Foundation and subject to all the rules and oversight requirements.
