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
import geopandas as gpd
from shapely.geometry import Point
from sqlalchemy import create_engine

# Create example GeoDataFrame
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

# Connect to PostGIS (Docker container)
engine = create_engine(
    "postgresql://postgres:password@localhost:5432/postgres"
)

# Write data into PostGIS
gdf.to_postgis(
    "my_places",
    engine,
    if_exists="replace",
    index=False,
)

gdf
````

## PostGIS → SedonaDB using GeoPandas

This approach reads a PostGIS table into a GeoPandas DataFrame and then converts it into a SedonaDB DataFrame.

```python
import geopandas as gpd
from sqlalchemy import create_engine
import sedona.db
```

```python
engine = create_engine(
    "postgresql://<user>:<password>@localhost:5432/<database>"
)
```

```python
gdf = gpd.read_postgis(
    "SELECT * FROM my_places",
    engine,
    geom_col="geom"
)
```

```python
sd = sedona.db.connect()
df = sd.create_data_frame(gdf)
df.show()
df.schema
```

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
    "postgresql://<user>:<password>@localhost:5432/<database>"
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

    cur.adbc_ingest(
        "ns_water_point_temp",
        df,
        temporary=True
    )

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

    sd.create_data_frame(
        cur.fetch_arrow()
    ).to_view("postgis_result", overwrite=True)

    df = sd.sql("""
        SELECT ST_GeomFromWKB(geom_wkb) AS geometry
        FROM postgis_result
    """).to_memtable()

```

```python
df.head(5).show()
```

### Choosing an approach

- Use the GeoPandas-based approach for simplicity and exploratory workflows.
- Use the ADBC-based approach for large datasets or production pipelines
  where performance and memory efficiency are critical.
