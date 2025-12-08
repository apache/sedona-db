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

# SedonaDB + Iceberg

This page demonstrates how to store spatial data in Iceberg tables and how to query them with SedonaDB.

You will learn how to create an Iceberg table with a well-known text (WKT) or well-known binary (WKB) column in an Iceberg table and some of the advantages of storing geometric data in Iceberg.

Make sure to run `pip install pyiceberg` to install the required dependencies for this notebook.

Let’s start by loading the required dependencies and saving a spatial dataset in an Iceberg table.


```python
from pyiceberg.catalog import load_catalog
import sedona.db
import pyarrow as pa
import os
```

## Create an Iceberg table with geometric data

Start by creating the Iceberg warehouse:


```python
os.makedirs("/tmp/warehouse", exist_ok=True)
```

Now set up the catalog:


```python
warehouse_path = "/tmp/warehouse"
catalog = load_catalog(
    "default",
    **{
        "type": "sql",
        "uri": f"sqlite:///{warehouse_path}/pyiceberg_catalog.db",
        "warehouse": f"file://{warehouse_path}",
    },
)
```

Use SedonaDB to read a Parquet file containing country data into a DataFrame.


```python
sd = sedona.db.connect()

countries = sd.read_parquet(
    "https://raw.githubusercontent.com/geoarrow/geoarrow-data/v0.2.0/natural-earth/files/natural-earth_countries_geo.parquet"
)
```

Convert all the columns to be plain strings because Iceberg doesn’t support geometry columns yet:


```python
countries.to_view("countries", True)
df = sd.sql("""
    select
        ARROW_CAST(name, 'Utf8') as name,
        ARROW_CAST(continent, 'Utf8') as continent,
        ST_AsText(geometry) as geometry_wkt
    from countries
""")
```

The explicit casting with `ARROW_CAST` is necessary because PyIceberg doesn't support string views.

Check out the schema of the DataFrame:


```python
df.schema
```




    SedonaSchema with 3 fields:
      name: utf8<Utf8>
      continent: utf8<Utf8>
      geometry_wkt: utf8<Utf8>



Now create a new Iceberg table:


```python
from pyiceberg.exceptions import NamespaceAlreadyExistsError

try:
    catalog.create_namespace("default")
except NamespaceAlreadyExistsError:
    pass

if catalog.table_exists("default.countries"):
    catalog.drop_table("default.countries")

table = catalog.create_table(
    "default.countries",
    schema=pa.schema(df.schema),
)
```

Append the DataFrame to the table:


```python
table.append(df.to_arrow_table())
```

Now let’s see how to read the data with SedonaDB.

## Read the Iceberg table into SedonaDB via Arrow

Here’s how to read an Iceberg table into a SedonaDB DataFrame:


```python
table = catalog.load_table("default.countries")
arrow_table = table.scan().to_arrow()
df = sd.create_data_frame(arrow_table)
```

The Iceberg table is first exposed as an arrow table and then read into a SedonaDB DataFrame.

Now view the contents of the SedonaDB DataFrame:


```python
df.to_view("my_table", True)
res = sd.sql("""
SELECT
  name,
  continent,
  ST_GeomFromWKT(geometry_wkt) as geom
from my_table
""")
res.show(3)
```

    ┌─────────────────────────────┬───────────┬────────────────────────────────────────────────────────┐
    │             name            ┆ continent ┆                          geom                          │
    │             utf8            ┆    utf8   ┆                        geometry                        │
    ╞═════════════════════════════╪═══════════╪════════════════════════════════════════════════════════╡
    │ Fiji                        ┆ Oceania   ┆ MULTIPOLYGON(((180 -16.067132663642447,180 -16.555216… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ United Republic of Tanzania ┆ Africa    ┆ POLYGON((33.90371119710453 -0.9500000000000001,34.072… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Western Sahara              ┆ Africa    ┆ POLYGON((-8.665589565454809 27.656425889592356,-8.665… │
    └─────────────────────────────┴───────────┴────────────────────────────────────────────────────────┘


You can see that the geom column contains the geometry type, which enables spatial analysis of the data.

The geometry data is stored as WKT, which isn't as efficient as WKB.  The example that follows demonstrates how to store WKB in Iceberg tables.

## Future Iceberg geography/geometry work

Iceberg added support for geometry and geography columns in the v3 spec.

The Iceberg v3 implementation has not been released yet, and it the v3 spec hasn't started in Iceberg Rust.  Here is [the open issue](https://github.com/apache/iceberg-rust/issues/1884) to add geo support to Iceberg Rust.

It’s best to manually persist the bbox information of files in your Iceberg table if you’re storing geometric data as WKT or WKB.

## Create an Iceberg table with WKB and bbox

Let’s see how to create an Iceberg table with a WKB and bbox columns to allow for faster spatial analyses.

Start by creating the cities DataFrame.


```python
cities = sd.read_parquet(
    "https://raw.githubusercontent.com/geoarrow/geoarrow-data/v0.2.0/natural-earth/files/natural-earth_cities_geo.parquet"
)
cities.to_view("cities", True)
```

Now write the DataFrame to an Iceberg table with bbox columns:


```python
df = sd.sql("""
select
    ARROW_CAST(name, 'Utf8') as name,
    ARROW_CAST(ST_AsBinary(geometry), 'Binary') as geometry_wkb,
    ST_XMin(geometry) as bbox_xmin,
    ST_YMin(geometry) as bbox_ymin,
    ST_XMax(geometry) as bbox_xmax,
    ST_YMax(geometry) as bbox_ymax
from cities
""")
```


```python
if catalog.table_exists("default.cities"):
    catalog.drop_table("default.cities")

table = catalog.create_table(
    "default.cities",
    schema=pa.schema(df.schema),
)
```


```python
table.append(df.to_arrow_table())
```

Load the `cities` table into a DataFrame with SedonaDB.


```python
table = catalog.load_table("default.cities")
arrow_table = table.scan().to_arrow()
df = sd.create_data_frame(arrow_table)
df.show()
```

    ┌──────────────┬───────────────────────────┬─────────────┬─────────────┬─────────────┬─────────────┐
    │     name     ┆        geometry_wkb       ┆  bbox_xmin  ┆  bbox_ymin  ┆  bbox_xmax  ┆  bbox_ymax  │
    │     utf8     ┆           binary          ┆   float64   ┆   float64   ┆   float64   ┆   float64   │
    ╞══════════════╪═══════════════════════════╪═════════════╪═════════════╪═════════════╪═════════════╡
    │ Vatican City ┆ 010100000054e57b4622e828… ┆  12.4533865 ┆  41.9032822 ┆  12.4533865 ┆  41.9032822 │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ San Marino   ┆ 0101000000dcb122b42fe228… ┆  12.4417702 ┆  43.9360958 ┆  12.4417702 ┆  43.9360958 │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Vaduz        ┆ 01010000006dae9ae7880823… ┆   9.5166695 ┆  47.1337238 ┆   9.5166695 ┆  47.1337238 │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Lobamba      ┆ 01010000007bcb8b0233333f… ┆  31.1999971 ┆ -26.4666675 ┆  31.1999971 ┆ -26.4666675 │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Luxembourg   ┆ 0101000000c08d39741f8518… ┆   6.1300028 ┆  49.6116604 ┆   6.1300028 ┆  49.6116604 │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Palikir      ┆ 0101000000b237e796ccc463… ┆ 158.1499743 ┆   6.9166437 ┆ 158.1499743 ┆   6.9166437 │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Majuro       ┆ 010100000027ef2df6286c65… ┆ 171.3800002 ┆   7.1030043 ┆ 171.3800002 ┆   7.1030043 │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Funafuti     ┆ 0101000000be28e6c5ee6666… ┆ 179.2166471 ┆   -8.516652 ┆ 179.2166471 ┆   -8.516652 │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Melekeok     ┆ 0101000000749b70af0cd460… ┆ 134.6265485 ┆   7.4873962 ┆ 134.6265485 ┆   7.4873962 │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Bir Lehlou   ┆ 0101000000f4d3c963174e23… ┆  -9.6525222 ┆  26.1191667 ┆  -9.6525222 ┆  26.1191667 │
    └──────────────┴───────────────────────────┴─────────────┴─────────────┴─────────────┴─────────────┘



```python
arrow_table.schema
```




    name: string
    geometry_wkb: binary
    bbox_xmin: double
    bbox_ymin: double
    bbox_xmax: double
    bbox_ymax: double



Read the Iceberg table and filter it to only include cities in the eastern half of North America.


```python
from pyiceberg.expressions import And, GreaterThanOrEqual, LessThanOrEqual

filter_expr = And(
    GreaterThanOrEqual("bbox_xmax", -97.0),
    LessThanOrEqual("bbox_xmin", -67.0),
    GreaterThanOrEqual("bbox_ymax", 25.0),
    LessThanOrEqual("bbox_ymin", 50.0),
)

arrow_table = table.scan(row_filter=filter_expr).to_arrow()
```


```python
df = sd.create_data_frame(arrow_table)

df.to_view("us_east_cities", True)
sd.sql("select name, ST_GeomFromWKB(geometry_wkb) as geom from us_east_cities").show()
```

    ┌──────────────────┬──────────────────────────────────────────────┐
    │       name       ┆                     geom                     │
    │       utf8       ┆                   geometry                   │
    ╞══════════════════╪══════════════════════════════════════════════╡
    │ Ottawa           ┆ POINT(-75.7019612 45.4186427)                │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Nassau           ┆ POINT(-77.3500438 25.0833901)                │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Houston          ┆ POINT(-95.34843625672217 29.741272831862542) │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Miami            ┆ POINT(-80.2260519 25.7895566)                │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Atlanta          ┆ POINT(-84.36764186571386 33.73945728378348)  │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Chicago          ┆ POINT(-87.63523655322338 41.847961283364114) │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Toronto          ┆ POINT(-79.38945855491194 43.66464454743429)  │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Washington, D.C. ┆ POINT(-77.0113644 38.9014952)                │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ New York         ┆ POINT(-73.99571754361698 40.72156174972766)  │
    └──────────────────┴──────────────────────────────────────────────┘
