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
  - `geoalchemy2`
  - `psycopg2-binary`
  - `adbc-driver-postgresql`

### Optional: Installing dependencies in a Jupyter environment

If you are running this notebook interactively, you can install the required
dependencies using:

```bash
pip install geopandas sqlalchemy geoalchemy2 psycopg2-binary adbc-driver-postgresql
```


## PostGIS Setup

This tutorial assumes a running PostgreSQL instance with PostGIS enabled.

For development and testing, the SedonaDB repository provides a PostGIS
Docker container that can be started with:

```bash
docker compose up postgis --detach
```


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



We'll use `create_engine()` to access PostGIS via SQLAlchemy.


```python
engine = create_engine("postgresql+psycopg2://postgres:password@127.0.0.1:5432")
```

## PostGIS → SedonaDB using GeoPandas

This approach reads a PostGIS table into a GeoPandas DataFrame and then converts it into a SedonaDB DataFrame.



```python
import geopandas as gpd
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
      name: utf8<LargeUtf8>
      geometry: geometry<Wkb(epsg:4326)>



## High-performance PostGIS integration using ADBC

Apache Arrow Database Connectivity (ADBC) enables efficient, zero-copy data transfer between databases and analytical engines. This approach is especially useful when working with large tables or when minimizing memory overhead is important.

By using `adbc_ingest()` and `fetch_arrow()`, this approach avoids row-wise iteration and intermediate Pandas DataFrames, making it well suited for large datasets and performance-critical pipelines.

First, we'll open the connection using ADBC:


```python
import adbc_driver_postgresql.dbapi

conn = adbc_driver_postgresql.dbapi.connect(
    "postgresql://postgres:password@127.0.0.1:5432/postgres"
)
```

To write the data from SedonaDB, we'll first ingest the table as a temporary table with geometry columns as WKB. This approach leverages ADBC's optimized Postgres ingest path.


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

Next, we'll create the table using a SELECT query that populates the geometry column.


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

To read data, we'll use the features of `create_data_frame()` that allows us to ingest any Arrow reader as a SedonaDB data frame. Next, we'll collect it while the cursor is still open using `to_memtable()`.


```python
with conn.cursor() as cur:
    cur.execute("""
        SELECT "OBJECTID", ST_AsBinary(geometry) AS geom_wkb
        FROM ns_water_point
    """)

    sd.create_data_frame(cur.fetch_arrow()).to_view("postgis_result", overwrite=True)

    df = sd.sql("""
        SELECT  "OBJECTID", ST_GeomFromWKB(geom_wkb) AS geometry
        FROM postgis_result
    """).to_memtable()
```

After the dataframwe has been collected, we can interact with it even after the cursor has been closed.


```python
df.head(5).show()
```

    ┌──────────┬──────────────────────────────────────────────────────────────────┐
    │ OBJECTID ┆                             geometry                             │
    │   int64  ┆                             geometry                             │
    ╞══════════╪══════════════════════════════════════════════════════════════════╡
    │     1055 ┆ POINT Z(258976.3273 4820275.6807 -0.5)                           │
    ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │     1023 ┆ POINT Z(258340.72730000038 4819923.080700001 0.6000000000058208) │
    ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │     1021 ┆ POINT Z(258338.4263000004 4819908.080700001 0.5)                 │
    ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │      985 ┆ POINT Z(258526.62729999982 4819583.580700001 0)                  │
    ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │      994 ┆ POINT Z(258498.92729999963 4819652.080700001 1.8999999999941792) │
    └──────────┴──────────────────────────────────────────────────────────────────┘


### Choosing an approach

- Use the GeoPandas-based approach for simplicity and exploratory workflows.
- Use the ADBC-based approach for large datasets or production pipelines
  where performance and memory efficiency are critical.

