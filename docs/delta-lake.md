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

# SedonaDB + Delta Lake

This page shows how to read and write Delta Lake tables with SedonaDB.

Make sure you run `pip install deltalake` to run the code snippets below.


```python
from deltalake import write_deltalake, DeltaTable
import sedona.db
import pyarrow.compute as pc

sd = sedona.db.connect()
```

Read in a GeoParquet dataset into a SedonaDB DataFrame.


```python
countries = sd.read_parquet(
    "https://raw.githubusercontent.com/geoarrow/geoarrow-data/v0.2.0/natural-earth/files/natural-earth_countries_geo.parquet"
)
```

## Create a Delta Lake table with WKT

Now write the DataFrame to a Delta Lake table.  Notice that you must convert the geometry column to Well-Known Text (WKT) or Well-Known Binary (WKB) before writing to the Delta table.

Delta Lake does not support geometry columns.

This example shows how to create a Delta table with a WKT column.  The section that follows shows how to create a Delta table with a WKB column.


```python
countries.to_view("countries", True)
df = sd.sql(
    "select name, continent, ST_AsText(geometry) as geometry_wkt from countries"
)
table_path = "/tmp/delta_with_wkt"
write_deltalake(table_path, df.to_pandas(), mode="overwrite")
```

## Read Delta table into SedonaDB

Now read the Delta table back into a SedonaDB DataFrame.


```python
dt = DeltaTable(table_path)
arrow_table = dt.to_pyarrow_table()
df = sd.create_data_frame(arrow_table)
df.show()
```

    ┌─────────────────────────────┬───────────────┬────────────────────────────────────────────────────┐
    │             name            ┆   continent   ┆                    geometry_wkt                    │
    │             utf8            ┆      utf8     ┆                        utf8                        │
    ╞═════════════════════════════╪═══════════════╪════════════════════════════════════════════════════╡
    │ Fiji                        ┆ Oceania       ┆ MULTIPOLYGON(((180 -16.067132663642447,180 -16.55… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ United Republic of Tanzania ┆ Africa        ┆ POLYGON((33.90371119710453 -0.9500000000000001,34… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Western Sahara              ┆ Africa        ┆ POLYGON((-8.665589565454809 27.656425889592356,-8… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Canada                      ┆ North America ┆ MULTIPOLYGON(((-122.84000000000003 49.00000000000… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ United States of America    ┆ North America ┆ MULTIPOLYGON(((-122.84000000000003 49.00000000000… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Kazakhstan                  ┆ Asia          ┆ POLYGON((87.35997033076265 49.21498078062912,86.5… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Uzbekistan                  ┆ Asia          ┆ POLYGON((55.96819135928291 41.30864166926936,55.9… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Papua New Guinea            ┆ Oceania       ┆ MULTIPOLYGON(((141.00021040259185 -2.600151055515… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Indonesia                   ┆ Asia          ┆ MULTIPOLYGON(((141.00021040259185 -2.600151055515… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Argentina                   ┆ South America ┆ MULTIPOLYGON(((-68.63401022758323 -52.63637045887… │
    └─────────────────────────────┴───────────────┴────────────────────────────────────────────────────┘


Notice that the `geometry_wkt` column is `utf8`.  It's not a geometry column.

Let's convert the `geometry_wkt` column to a geometry column.


```python
df.to_view("my_table", True)
res = sd.sql("""
SELECT
  name,
  continent,
  ST_GeomFromWKT(geometry_wkt) as geom
from my_table
""")
res.show()
```

    ┌─────────────────────────────┬───────────────┬────────────────────────────────────────────────────┐
    │             name            ┆   continent   ┆                        geom                        │
    │             utf8            ┆      utf8     ┆                      geometry                      │
    ╞═════════════════════════════╪═══════════════╪════════════════════════════════════════════════════╡
    │ Fiji                        ┆ Oceania       ┆ MULTIPOLYGON(((180 -16.067132663642447,180 -16.55… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ United Republic of Tanzania ┆ Africa        ┆ POLYGON((33.90371119710453 -0.9500000000000001,34… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Western Sahara              ┆ Africa        ┆ POLYGON((-8.665589565454809 27.656425889592356,-8… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Canada                      ┆ North America ┆ MULTIPOLYGON(((-122.84000000000003 49.00000000000… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ United States of America    ┆ North America ┆ MULTIPOLYGON(((-122.84000000000003 49.00000000000… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Kazakhstan                  ┆ Asia          ┆ POLYGON((87.35997033076265 49.21498078062912,86.5… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Uzbekistan                  ┆ Asia          ┆ POLYGON((55.96819135928291 41.30864166926936,55.9… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Papua New Guinea            ┆ Oceania       ┆ MULTIPOLYGON(((141.00021040259185 -2.600151055515… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Indonesia                   ┆ Asia          ┆ MULTIPOLYGON(((141.00021040259185 -2.600151055515… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Argentina                   ┆ South America ┆ MULTIPOLYGON(((-68.63401022758323 -52.63637045887… │
    └─────────────────────────────┴───────────────┴────────────────────────────────────────────────────┘


Confirm the schema of the DataFrame.


```python
res.schema
```




    SedonaSchema with 3 fields:
      name: utf8<Utf8>
      continent: utf8<Utf8>
      geom: geometry<Wkb>



## Filter countries in a particular geographic region

Now, let's grab some countries in the western portion of South America using a polygon region.

SedonaDB can run these types of queries on geometric data.


```python
res = sd.sql("""
SELECT
  name,
  continent,
  ST_GeomFromWKT(geometry_wkt) as geom
FROM my_table
WHERE ST_Intersects(
  ST_GeomFromWKT(geometry_wkt),
  ST_GeomFromWKT('POLYGON((-81 5, -75 5, -75 -56, -81 -56, -81 5))')
)
""")
res.show()
```

    ┌──────────┬───────────────┬───────────────────────────────────────────────────────────────────────┐
    │   name   ┆   continent   ┆                                  geom                                 │
    │   utf8   ┆      utf8     ┆                                geometry                               │
    ╞══════════╪═══════════════╪═══════════════════════════════════════════════════════════════════════╡
    │ Chile    ┆ South America ┆ MULTIPOLYGON(((-68.63401022758323 -52.63637045887449,-68.63335000000… │
    ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Peru     ┆ South America ┆ POLYGON((-69.89363521999663 -4.2981869441943275,-70.7947688463023 -4… │
    ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Colombia ┆ South America ┆ POLYGON((-66.87632585312258 1.253360500489336,-67.0650481838525 1.13… │
    ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Ecuador  ┆ South America ┆ POLYGON((-75.37322323271385 -0.1520317521204504,-75.23372270374195 -… │
    └──────────┴───────────────┴───────────────────────────────────────────────────────────────────────┘


## Create a Delta Lake table with WKB

You can also create a Delta table with WKB.

WKB is binary, can be compressed more effectively than WKT, and results in smaller file sizes.

The following example shows how to store the cities dataset in a Delta table with the geometry data stored as WKB.

It also demonstrates how to add a `bbox` column to the Delta table, enabling more efficient filtering.


```python
cities = sd.read_parquet(
    "https://raw.githubusercontent.com/geoarrow/geoarrow-data/v0.2.0/natural-earth/files/natural-earth_cities_geo.parquet"
)
cities.to_view("cities", True)
```


```python
df = sd.sql("""
select
    name,
    ST_AsBinary(geometry) as geometry_wkb,
    STRUCT(
        ST_XMin(geometry) as xmin,
        ST_YMin(geometry) as ymin,
        ST_XMax(geometry) as xmax,
        ST_YMax(geometry) as ymax
    ) as bbox
from cities
""")
table_path = "/tmp/delta_with_wkb"
write_deltalake(table_path, df.to_pandas(), mode="overwrite", schema_mode="overwrite")
```

Read the Delta table and filter it to only include cities in the eastern half of North America.


```python
dt = DeltaTable(table_path)
dataset = dt.to_pyarrow_dataset()
filter_expr = (
    (pc.field("bbox", "xmax") >= -97.0) &
    (pc.field("bbox", "xmin") <= -67.0) &
    (pc.field("bbox", "ymax") >= 25.0) &
    (pc.field("bbox", "ymin") <= 50.0)
)
filtered_table = dataset.to_table(filter=filter_expr)
```


```python
df = sd.create_data_frame(filtered_table.to_pandas())
```


```python
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
