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

# Process Vector Data with SedonaDB

Process vector data using SedonaDB. You will learn to create DataFrames, run spatial queries, and manage file I/O. Let's begin by connecting to SedonaDB.

Let's start by establishing a SedonaDB connection.

## Establish SedonaDB connection

Here's how to create the SedonaDB connection:


```python
import sedona.db

sd = sedona.db.connect()
```

Now, let's see how to create SedonaDB dataframes.

## Create SedonaDB DataFrame

**Manually creating SedonaDB DataFrame**

Here's how to manually create a SedonaDB DataFrame:


```python
df = sd.sql("""
SELECT * FROM (VALUES
    ('one', ST_GeomFromWkt('POINT(1 2)')),
    ('two', ST_GeomFromWkt('POLYGON((-74.0 40.7, -74.0 40.8, -73.9 40.8, -73.9 40.7, -74.0 40.7))')),
    ('three', ST_GeomFromWkt('LINESTRING(-74.0060 40.7128, -73.9352 40.7306, -73.8561 40.8484)')))
AS t(val, point)""")
```

Check the type of the DataFrame.


```python
type(df)
```




    sedonadb.dataframe.DataFrame



**Create SedonaDB DataFrame from files in S3**

For most production applications, you will create SedonaDB DataFrames by reading data from a file.  Let's see how to read GeoParquet files in AWS S3 into a SedonaDB DataFrame.


```python
sd.read_parquet(
    "s3://overturemaps-us-west-2/release/2025-08-20.0/theme=divisions/type=division_area/",
    options={"aws.skip_signature": True, "aws.region": "us-west-2"},
).to_view("division_area")
```

Now, let's run some spatial queries.

### Read from GeoPandas DataFrame

This section shows how to convert a GeoPandas DataFrame into a SedonaDB DataFrame.

Start by reading a FlatGeoBuf file into a GeoPandas DataFrame:


```python
import geopandas as gpd

path = "https://raw.githubusercontent.com/geoarrow/geoarrow-data/v0.2.0/natural-earth/files/natural-earth_cities.fgb"
gdf = gpd.read_file(path)
```

Now convert the GeoPandas DataFrame to a SedonaDB DataFrame and view three rows of content:


```python
df = sd.create_data_frame(gdf)
df.show(3)
```

    ┌──────────────┬──────────────────────────────┐
    │     name     ┆           geometry           │
    │     utf8     ┆           geometry           │
    ╞══════════════╪══════════════════════════════╡
    │ Vatican City ┆ POINT(12.4533865 41.9032822) │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ San Marino   ┆ POINT(12.4417702 43.9360958) │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Vaduz        ┆ POINT(9.5166695 47.1337238)  │
    └──────────────┴──────────────────────────────┘


## Spatial queries

Let's see how to run spatial operations like filtering, joins, and clustering algorithms.

### Spatial filtering

Let's run a spatial filtering operation to fetch all the objects in the following polygon:


```python
nova_scotia_bbox_wkt = (
    "POLYGON((-66.5 43.4, -66.5 47.1, -59.8 47.1, -59.8 43.4, -66.5 43.4))"
)

ns = sd.sql(f"""
SELECT country, region, geometry
FROM division_area
WHERE ST_Intersects(geometry, ST_SetSRID(ST_GeomFromText('{nova_scotia_bbox_wkt}'), 4326))
""")

ns.show(3)
```

    ┌──────────┬──────────┬────────────────────────────────────────────────────────────────────────────┐
    │  country ┆  region  ┆                                  geometry                                  │
    │ utf8view ┆ utf8view ┆                                  geometry                                  │
    ╞══════════╪══════════╪════════════════════════════════════════════════════════════════════════════╡
    │ CA       ┆ CA-NS    ┆ POLYGON((-66.0528452 43.4531336,-66.0883401 43.3978188,-65.9647654 43.361… │
    ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ CA       ┆ CA-NS    ┆ POLYGON((-66.0222822 43.5166842,-66.0252286 43.5100071,-66.0528452 43.453… │
    ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ CA       ┆ CA-NS    ┆ POLYGON((-65.7451389 43.5336263,-65.7450818 43.5347004,-65.7449545 43.535… │
    └──────────┴──────────┴────────────────────────────────────────────────────────────────────────────┘


You can see it only includes the divisions in the Nova Scotia area.

### K-nearest neighbors (KNN) joins

Create `restaurants` and `customers` views so we can demonstrate the KNN join functionality.


```python
df = sd.sql("""
SELECT name, ST_Point(lng, lat) AS location
FROM (VALUES
    (101, -74.0, 40.7, 'Pizza Palace'),
    (102, -73.99, 40.69, 'Burger Barn'),
    (103, -74.02, 40.72, 'Taco Town'),
    (104, -73.98, 40.75, 'Sushi Spot'),
    (105, -74.05, 40.68, 'Deli Direct')
) AS t(id, lng, lat, name)
""")
sd.sql("drop view if exists restaurants")
df.to_view("restaurants")

df = sd.sql("""
SELECT name, ST_Point(lng, lat) AS location
FROM (VALUES
    (1, -74.0, 40.7, 'Alice'),
    (2, -73.9, 40.8, 'Bob'),
    (3, -74.1, 40.6, 'Carol')
) AS t(id, lng, lat, name)
""")
sd.sql("drop view if exists customers")
df.to_view("customers")
```


```python
df.show()
```

    ┌───────┬───────────────────┐
    │  name ┆      location     │
    │  utf8 ┆      geometry     │
    ╞═══════╪═══════════════════╡
    │ Alice ┆ POINT(-74 40.7)   │
    ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Bob   ┆ POINT(-73.9 40.8) │
    ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Carol ┆ POINT(-74.1 40.6) │
    └───────┴───────────────────┘


Perform a KNN join to identify the two restaurants that are nearest to each customer:


```python
sd.sql("""
SELECT
    c.name AS customer,
    r.name AS restaurant
FROM customers c, restaurants r
WHERE ST_KNN(c.location, r.location, 2, false)
ORDER BY c.name, r.name;
""").show()
```

    ┌──────────┬──────────────┐
    │ customer ┆  restaurant  │
    │   utf8   ┆     utf8     │
    ╞══════════╪══════════════╡
    │ Alice    ┆ Burger Barn  │
    ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Alice    ┆ Pizza Palace │
    ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Bob      ┆ Pizza Palace │
    ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Bob      ┆ Sushi Spot   │
    ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Carol    ┆ Deli Direct  │
    ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Carol    ┆ Pizza Palace │
    └──────────┴──────────────┘


Notice how each customer has two rows - one for each of the two closest restaurants.
