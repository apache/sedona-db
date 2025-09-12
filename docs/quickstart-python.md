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

# Python Quickstart

SedonaDB for Python can be installed from PyPI:

```shell
pip install "apache-sedona[db]"
```
!!!tip "Run this tutorial as in interactive notebook"
    You can also download the `.ipynb` version of this file from the [SedonaDB GitHub](https://github.com/apache/sedona-db/blob/main/docs/quickstart-python.ipynb)and run it as an interactive notebook.

## Import SedonaDB

To get started, import the library and connect to a new session. You can run SQL queries directly on the session object.

```python
import sedona.db

sd = sedona.db.connect()
sd.sql("SELECT ST_Point(0, 1) as geom").show()
```

**Output:**

```sh
┌────────────┐
│    geom    │
│     wkb    │
╞════════════╡
│ POINT(0 1) │
└────────────┘
```

## Spatial Join Example

A common use case is performing a spatial join.
In this example, we'll find the country that each city belongs to by checking if the city's point geometry intersects with a country's polygon geometry.

### Load Datasets

First, load the cities and countries parquet files from their URLs into SedonaDB DataFrames.

```python
cities_url = "https://raw.githubusercontent.com/geoarrow/geoarrow-data/v0.2.0/natural-earth/files/natural-earth_cities_geo.parquet"
countries_url = "https://raw.githubusercontent.com/geoarrow/geoarrow-data/v0.2.0/natural-earth/files/natural-earth_countries_geo.parquet"

cities = sd.read_parquet(cities_url)
countries = sd.read_parquet(countries_url)
```

### Register Views

To query these DataFrames using SQL, they must be registered as temporary views in the session.

```python
cities.to_view("cities")
countries.to_view("countries")
```

### Run the Join Query

Now you can run a SQL query using `ST_Intersects` to join the two views.

```python
# Join the cities and countries tables
sd.sql("""
    SELECT
        cities.name AS city,
        countries.name AS country,
        countries.continent
    FROM cities
    JOIN countries
    WHERE ST_Intersects(cities.geometry, countries.geometry)
""").show()
```

**Output:**

```
┌───────────────┬─────────────────────────────┬───────────────┐
│     city      ┆           country           ┆   continent   │
│    utf8view   ┆           utf8view          ┆    utf8view   │
╞═══════════════╪═════════════════════════════╪═══════════════╡
│ Suva          ┆ Fiji                        ┆ Oceania       │
├───────────────┼─────────────────────────────┼───────────────┤
│ Dodoma        ┆ United Republic of Tanzania ┆ Africa        │
├───────────────┼─────────────────────────────┼───────────────┤
│ Dar es Salaam ┆ United Republic of Tanzania ┆ Africa        │
├───────────────┼─────────────────────────────┼───────────────┤
│ Bir Lehlou    ┆ Western Sahara              ┆ Africa        │
...
└───────────────┴─────────────────────────────┴───────────────┘
```

## Creating a DataFrame Manually

You can also create a SedonaDB DataFrame from scratch using SQL `VALUES` clauses and geometry functions like `ST_GeomFromWkt`.

```python
df = sd.sql("""
    SELECT * FROM (VALUES
        ('one', ST_GeomFromWkt('POINT(1 2)')),
        ('two', ST_GeomFromWkt('POLYGON((-74.0 40.7, -74.0 40.8, -73.9 40.8, -73.9 40.7, -74.0 40.7))')),
        ('three', ST_GeomFromWkt('LINESTRING(-74.0060 40.7128, -73.9352 40.7306, -73.8561 40.8484)')))
    AS t(val, point)
""")

# Verify the object type
type(df)
```

**Output:**

```
sedonadb.dataframe.DataFrame
```

Once created, you can register it as a view and run further spatial operations on it.

```python
df.to_view("fun_table")
sd.sql("SELECT *, ST_Centroid(point) AS centroid FROM fun_table").show()
```

**Output:**

```
┌───────┬─────────────────────────────────────────────┬────────────────────────────────────────────┐
│  val  ┆                    point                    ┆                  centroid                  │
│  utf8 ┆                     wkb                     ┆                     wkb                    │
╞═══════╪═════════════════════════════════════════════╪════════════════════════════════════════════╡
│ one   ┆ POINT(1 2)                                  ┆ POINT(1 2)                                 │
├───────┼─────────────────────────────────────────────┼────────────────────────────────────────────┤
│ two   ┆ POLYGON((-74 40.7,-74 40.8,-73.9 40.8,-73.… ┆ POINT(-73.95000000000002 40.75)            │
├───────┼─────────────────────────────────────────────┼────────────────────────────────────────────┤
│ three ┆ LINESTRING(-74.006 40.7128,-73.9352 40.730… ┆ POINT(-73.92111155675562 40.7664673976246… │
└───────┴─────────────────────────────────────────────┴────────────────────────────────────────────┘
```
