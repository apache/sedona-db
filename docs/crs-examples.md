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

# Coordinate Reference System (CRS) Examples

This example demonstrates how one table with an EPSG 4326 CRS cannot be joined with another table that uses EPSG 3857.


```python
import sedonadb

sd = sedonadb.connect()
```

Read a table with a geometry column that uses EPSG 4326.

Note how SedonaDB reads the CRS specified in the Parquet file.


```python
countries = sd.read_parquet(
    "https://raw.githubusercontent.com/geoarrow/geoarrow-data/v0.2.0/natural-earth/files/natural-earth_countries_geo.parquet"
)
```


```python
countries.schema
```




    SedonaSchema with 3 fields:
      name: Utf8View
      continent: Utf8View
      geometry: wkb_view <epsg:4326>




```python
cities = sd.sql("""
SELECT city, ST_SetSRID(ST_GeomFromText(wkt), 3857) AS geometry FROM (VALUES
    ('New York', 'POINT(-8238310.24 4969803.34)'),
    ('Los Angeles', 'POINT(-13153204.78 4037636.04)'),
    ('Chicago', 'POINT(-9757148.04 5138517.44)'))
AS t(city, wkt)""")
```


```python
cities.schema
```




    SedonaSchema with 2 fields:
      city: Utf8
      geometry: wkb <epsg:3857>




```python
cities.to_view("cities", overwrite=True)
countries.to_view("countries", overwrite=True)
```

## Join with mismatched CRSs

The cities and countries tables have different CRSs.

The cities table uses EPSG:3857 and the countries table uses EPSG:4326.

Let's confirm that the code errors out if we try to join the mismatched tables.


```python
# join doesn't work when CRSs don't match
sd.sql("""
select * from cities
join countries
where ST_Intersects(cities.geometry, countries.geometry)
""").show()
```


    ---------------------------------------------------------------------------

    SedonaError                               Traceback (most recent call last)

    Cell In[7], line 6
          1 # join doesn't work when CRSs don't match
          2 sd.sql("""
          3 select * from cities
          4 join countries
          5 where ST_Intersects(cities.geometry, countries.geometry)
    ----> 6 """).show()


    File /opt/miniconda3/lib/python3.12/site-packages/sedonadb/dataframe.py:297, in DataFrame.show(self, limit, width, ascii)
        272 """Print the first limit rows to the console
        273 
        274 Args:
       (...)
        294 
        295 """
        296 width = _out_width(width)
    --> 297 print(self._impl.show(self._ctx, limit, width, ascii), end="")


    SedonaError: type_coercion
    caused by
    Error during planning: Mismatched CRS arguments: epsg:3857 vs epsg:4326
    Use ST_Transform() or ST_SetSRID() to ensure arguments are compatible.


## Convert CRS and then join

Let's convert the cities table to use EPSG:4326 and then perform the join with the two tables once they have matching CRSs.


```python
# update cities to use 4326
cities = sd.sql("""
SELECT city, ST_Transform(geometry, 'EPSG:4326') as geometry
FROM cities
""")
```


```python
cities.schema
```




    SedonaSchema with 2 fields:
      city: Utf8
      geometry: wkb <ogc:crs84>




```python
cities.to_view("cities", overwrite=True)
```


```python
# join works when CRSs match
sd.sql("""
select * from cities
join countries
where ST_Intersects(cities.geometry, countries.geometry)
""").show()
```

    ┌─────────────┬──────────────────────┬──────────────────────┬───────────────┬──────────────────────┐
    │     city    ┆       geometry       ┆         name         ┆   continent   ┆       geometry       │
    │     utf8    ┆       geometry       ┆       utf8view       ┆    utf8view   ┆       geometry       │
    ╞═════════════╪══════════════════════╪══════════════════════╪═══════════════╪══════════════════════╡
    │ New York    ┆ POINT(-74.006000039… ┆ United States of Am… ┆ North America ┆ MULTIPOLYGON(((-122… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Los Angeles ┆ POINT(-118.15724889… ┆ United States of Am… ┆ North America ┆ MULTIPOLYGON(((-122… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Chicago     ┆ POINT(-87.649952137… ┆ United States of Am… ┆ North America ┆ MULTIPOLYGON(((-122… │
    └─────────────┴──────────────────────┴──────────────────────┴───────────────┴──────────────────────┘


## Example #2: Joining two tables with different CRSs

This example shows how to join a `vermont` table with an EPSG 32618 CRS with a `buildings` table that uses an EPSG 4326 CRS.

The example highlights the following features:

1. SedonaDB reads the CRS stored in the files
2. SedonaDB protects you from accidentally joining files with mismatched CRSs
3. It's easy to convert a GeoPandas DataFrame => a SedonaDB DataFrame and maintain the CRS


```python
import geopandas as gpd

path = "https://raw.githubusercontent.com/geoarrow/geoarrow-data/v0.2.0/example-crs/files/example-crs_vermont-utm.fgb"
gdf = gpd.read_file(path)
```


```python
vermont = sd.create_data_frame(gdf)
```


```python
vermont.schema
```




    SedonaSchema with 1 field:
      geometry: wkb <epsg:32618>




```python
buildings = sd.read_parquet(
    "https://github.com/geoarrow/geoarrow-data/releases/download/v0.2.0/microsoft-buildings_point_geo.parquet"
)
```


```python
buildings.show(3)
```

    ┌─────────────────────────────────┐
    │             geometry            │
    │             geometry            │
    ╞═════════════════════════════════╡
    │ POINT(-77.10109681 42.53495524) │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ POINT(-77.10048552 42.53695011) │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ POINT(-77.10096508 42.53681338) │
    └─────────────────────────────────┘



```python
buildings.schema
```




    SedonaSchema with 1 field:
      geometry: wkb_view <ogc:crs84>




```python
buildings.count()
```




    129735970




```python
buildings.to_view("buildings", overwrite=True)
vermont.to_view("vermont", overwrite=True)
```


```python
sd.sql("""
select count(*) from buildings
join vermont
where ST_Intersects(buildings.geometry, vermont.geometry)
""").show()
```


    ---------------------------------------------------------------------------

    SedonaError                               Traceback (most recent call last)

    Cell In[12], line 5
          1 sd.sql("""
          2 select count(*) from buildings
          3 join vermont
          4 where ST_Intersects(buildings.geometry, vermont.geometry)
    ----> 5 """).show()


    File /opt/miniconda3/lib/python3.12/site-packages/sedonadb/dataframe.py:297, in DataFrame.show(self, limit, width, ascii)
        272 """Print the first limit rows to the console
        273 
        274 Args:
       (...)
        294 
        295 """
        296 width = _out_width(width)
    --> 297 print(self._impl.show(self._ctx, limit, width, ascii), end="")


    SedonaError: type_coercion
    caused by
    Error during planning: Mismatched CRS arguments: ogc:crs84 vs epsg:32618
    Use ST_Transform() or ST_SetSRID() to ensure arguments are compatible.



```python
sd.sql("""
select count(*) from buildings
join vermont
where ST_Intersects(buildings.geometry, ST_Transform(vermont.geometry, 'EPSG:4326'))
""").show()
```

    ┌──────────┐
    │ count(*) │
    │   int64  │
    ╞══════════╡
    │   361856 │
    └──────────┘

