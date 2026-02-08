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

# SedonaDB + PostGIS

This page demonstrates how to integrate PostGIS with SedonaDB.

Two approaches are covered:

1. A GeoPandas-based workflow for simplicity and exploratory use.
2. A high-performance ADBC-based workflow for large datasets and production use cases.

## Prerequisites

This notebook assumes:

- A running PostgreSQL instance with PostGIS enabled
- Python 3.9+
- The following Python packages available:
  - `geopandas`
  - `sqlalchemy`
  - `psycopg2-binary`
  - `adbc-driver-postgresql`

### Optional: Installing dependencies in a Jupyter environment

If you are running this notebook interactively, you can install the required
dependencies using:

````bash
pip install geopandas sqlalchemy psycopg2-binary adbc-driver-postgresql


## PostGIS Setup

This tutorial assumes a running PostgreSQL instance with PostGIS enabled.

For development and testing, the SedonaDB repository provides a PostGIS
Docker container that can be started with:

```bash
docker compose up --detach



```python
import psycopg2

conn = psycopg2.connect(
    host="127.0.0.1",
    port=5432,
    dbname="postgres",
    user="postgres",
)
print("CONNECTED OK")
conn.close()
````

    CONNECTED OK

```python
import geopandas as gpd
from shapely.geometry import Point
from sqlalchemy import create_engine

gdf = gpd.GeoDataFrame(
    {
        "name": ["New York", "Los Angeles", "Chicago"],
        "geometry": [
            Point(-74.006, 40.7128),
            Point(-118.2437, 34.0522),
            Point(-87.6298, 41.8781),
        ],
    },
    crs="EPSG:4326",
)

gdf

```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>New York</td>
      <td>POINT (-74.006 40.7128)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Los Angeles</td>
      <td>POINT (-118.2437 34.0522)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Chicago</td>
      <td>POINT (-87.6298 41.8781)</td>
    </tr>
  </tbody>
</table>
</div>

Ensure PostGIS is running and accessible before executing these cells

```python
from sqlalchemy import create_engine

engine = create_engine(
    "postgresql+psycopg2://postgres:password@127.0.0.1:5432/postgres",
    pool_pre_ping=True,
)
```

```python
from sqlalchemy import text

with engine.connect() as conn:
    print(conn.execute(text("SELECT current_user")).fetchall())

```

    [('postgres',)]

## PostGIS → SedonaDB using GeoPandas

This approach reads a PostGIS table into a GeoPandas DataFrame and then converts it into a SedonaDB DataFrame.

```python
import geopandas as gpd
from sqlalchemy import create_engine
import sedona.db
```

```python
gdf.to_postgis(
    "my_places",
    engine,
    if_exists="replace",
    index=False,
)
```

```python
gdf = gpd.read_postgis(
    "SELECT * FROM my_places",
    engine,
    geom_col="geometry",
)

import sedona.db

sd = sedona.db.connect()
df = sd.create_data_frame(gdf)
df.show()
df.schema

```

    ┌─────────────┬──────────────────────────┐
    │     name    ┆         geometry         │
    │     utf8    ┆         geometry         │
    ╞═════════════╪══════════════════════════╡
    │ New York    ┆ POINT(-74.006 40.7128)   │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Los Angeles ┆ POINT(-118.2437 34.0522) │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Chicago     ┆ POINT(-87.6298 41.8781)  │
    └─────────────┴──────────────────────────┘





    SedonaSchema with 2 fields:
      name: utf8<Utf8>
      geometry: geometry<Wkb(epsg:4326)>

## High-performance PostGIS integration using ADBC

Apache Arrow Database Connectivity (ADBC) enables efficient, zero-copy data
transfer between databases and analytical engines. This approach is especially useful when working with large tables or when minimizing memory overhead is important.

By using `adbc_ingest()` and `fetch_arrow()`, this approach avoids row-wise
iteration and intermediate Pandas DataFrames, making it well suited for
large datasets and performance-critical pipelines.

```python
import sedona.db
import adbc_driver_postgresql.dbapi

sd = sedona.db.connect()

conn = adbc_driver_postgresql.dbapi.connect(
    "postgresql://postgres:password@127.0.0.1:5432/postgres"
)

```

### Writing data from SedonaDB to PostGIS using ADBC

```python
with conn.cursor() as cur:
    url = "https://github.com/geoarrow/geoarrow-data/releases/download/v0.2.0/ns-water_water-point_geo.parquet"

    sd.read_parquet(url).to_view("ns_water_point", overwrite=True)

    df = sd.sql("""
        SELECT "OBJECTID", ST_AsBinary(geometry) AS geometry
        FROM ns_water_point
    """)

    cur.adbc_ingest("ns_water_point_temp", df, temporary=True)
```

```python
with conn.cursor() as cur:
    cur.executescript("""
        CREATE TABLE ns_water_point AS
        SELECT
            "OBJECTID",
            ST_GeomFromWKB(geometry) AS geometry
        FROM ns_water_point_temp
    """)
```

### Reading data from PostGIS into SedonaDB using ADBC

```python
with conn.cursor() as cur:
    cur.execute("""
        SELECT "OBJECTID", ST_AsBinary(geometry) AS geom_wkb
        FROM ns_water_point
    """)

    sd.create_data_frame(cur.fetch_arrow()).to_view("postgis_result", overwrite=True)

    df = sd.sql("""
        SELECT ST_GeomFromWKB(geom_wkb) AS geometry
        FROM postgis_result
    """).to_memtable()
```

```python
df.head(5).show()
```

    ┌──────────────────────────────────────────────────────────────────┐
    │                             geometry                             │
    │                             geometry                             │
    ╞══════════════════════════════════════════════════════════════════╡
    │ POINT Z(300175.22580000013 4910284.878799999 166.39999999999418) │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ POINT Z(300229.72580000013 4910146.878799999 166.39999999999418) │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ POINT Z(300258.0247999998 4910111.278899999 166.39999999999418)  │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ POINT Z(300267.62480000034 4910089.1789 166.39999999999418)      │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ POINT Z(300321.22580000013 4910120.1778 166.39999999999418)      │
    └──────────────────────────────────────────────────────────────────┘

### Choosing an approach

- Use the GeoPandas-based approach for simplicity and exploratory workflows.
- Use the ADBC-based approach for large datasets or production pipelines
  where performance and memory efficiency are critical.
