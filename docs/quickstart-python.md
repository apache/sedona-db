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

First, install SedonaDB:



```shell
pip install "apache-sedona[db]"
```

If the import and connection are successful, your installation is complete.


```python
import sedona.db

sd = sedona.db.connect()
sd.sql("SELECT ST_Point(0, 1) as geom").show()
```

    ┌────────────┐
    │    geom    │
    │  geometry  │
    ╞════════════╡
    │ POINT(0 1) │
    └────────────┘


## Point in polygon join


```python
cities = sd.read_parquet(
    "https://raw.githubusercontent.com/geoarrow/geoarrow-data/v0.2.0/natural-earth/files/natural-earth_cities_geo.parquet"
)
```


```python
cities.show()
```

    ┌──────────────┬───────────────────────────────┐
    │     name     ┆            geometry           │
    │     utf8     ┆            geometry           │
    ╞══════════════╪═══════════════════════════════╡
    │ Vatican City ┆ POINT(12.4533865 41.9032822)  │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ San Marino   ┆ POINT(12.4417702 43.9360958)  │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Vaduz        ┆ POINT(9.5166695 47.1337238)   │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Lobamba      ┆ POINT(31.1999971 -26.4666675) │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Luxembourg   ┆ POINT(6.1300028 49.6116604)   │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Palikir      ┆ POINT(158.1499743 6.9166437)  │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Majuro       ┆ POINT(171.3800002 7.1030043)  │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Funafuti     ┆ POINT(179.2166471 -8.516652)  │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Melekeok     ┆ POINT(134.6265485 7.4873962)  │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Bir Lehlou   ┆ POINT(-9.6525222 26.1191667)  │
    └──────────────┴───────────────────────────────┘



```python
countries = sd.read_parquet(
    "https://raw.githubusercontent.com/geoarrow/geoarrow-data/v0.2.0/natural-earth/files/natural-earth_countries_geo.parquet"
)
```


```python
countries.show()
```

    ┌─────────────────────────────┬───────────────┬────────────────────────────────────────────────────┐
    │             name            ┆   continent   ┆                      geometry                      │
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



```python
cities.to_view("cities")
countries.to_view("countries")
```


```python
# join the cities and countries tables
sd.sql("""
select * from cities
join countries
where ST_Intersects(cities.geometry, countries.geometry)
""").show()
```

    ┌───────────────┬──────────────────────┬─────────────────────┬───────────────┬─────────────────────┐
    │      name     ┆       geometry       ┆         name        ┆   continent   ┆       geometry      │
    │      utf8     ┆       geometry       ┆         utf8        ┆      utf8     ┆       geometry      │
    ╞═══════════════╪══════════════════════╪═════════════════════╪═══════════════╪═════════════════════╡
    │ Suva          ┆ POINT(178.4417073 -… ┆ Fiji                ┆ Oceania       ┆ MULTIPOLYGON(((180… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Dodoma        ┆ POINT(35.7500036 -6… ┆ United Republic of… ┆ Africa        ┆ POLYGON((33.903711… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Dar es Salaam ┆ POINT(39.266396 -6.… ┆ United Republic of… ┆ Africa        ┆ POLYGON((33.903711… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Bir Lehlou    ┆ POINT(-9.6525222 26… ┆ Western Sahara      ┆ Africa        ┆ POLYGON((-8.665589… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Ottawa        ┆ POINT(-75.7019612 4… ┆ Canada              ┆ North America ┆ MULTIPOLYGON(((-12… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Vancouver     ┆ POINT(-123.1235901 … ┆ Canada              ┆ North America ┆ MULTIPOLYGON(((-12… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Toronto       ┆ POINT(-79.389458554… ┆ Canada              ┆ North America ┆ MULTIPOLYGON(((-12… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ San Francisco ┆ POINT(-122.39959956… ┆ United States of A… ┆ North America ┆ MULTIPOLYGON(((-12… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Denver        ┆ POINT(-104.9859618 … ┆ United States of A… ┆ North America ┆ MULTIPOLYGON(((-12… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Houston       ┆ POINT(-95.348436256… ┆ United States of A… ┆ North America ┆ MULTIPOLYGON(((-12… │
    └───────────────┴──────────────────────┴─────────────────────┴───────────────┴─────────────────────┘


## Manually create SedonaDB DataFrames

Let's create a DataFrame with one string column and one geometry column to show some of the functionality of the SedonaDB Python interface.


```python
df = sd.sql("""
    SELECT * FROM (VALUES
        ('one', ST_GeomFromWkt('POINT(1 2)')),
        ('two', ST_GeomFromWkt('POLYGON((
            -74.0 40.7, -74.0 40.8, -73.9 40.8,
            -73.9 40.7, -74.0 40.7
        ))')),
        ('three', ST_GeomFromWkt('LINESTRING(
            -74.0060 40.7128, -73.9352 40.7306,
            -73.8561 40.8484
        )'))
    ) AS t(val, point)
""")
```


```python
df.show()
```

    ┌───────┬──────────────────────────────────────────────────────────────────────────────────────────┐
    │  val  ┆                                           point                                          │
    │  utf8 ┆                                          binary                                          │
    ╞═══════╪══════════════════════════════════════════════════════════════════════════════════════════╡
    │ one   ┆ 0101000000000000000000f03f0000000000000040                                               │
    ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ two   ┆ 0103000000010000000500000000000000008052c09a9999999959444000000000008052c06666666666664… │
    ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ three ┆ 010200000003000000aaf1d24d628052c05e4bc8073d5b444007ce1951da7b52c0933a014d845d4440c286a… │
    └───────┴──────────────────────────────────────────────────────────────────────────────────────────┘


Verify that this object is a SedonaDB DataFrame.


```python
type(df)
```




    sedonadb.dataframe.DataFrame



Expose the DataFrame as a view and run a SQL operation on the geometry data.


```python
df.to_view("fun_table")
```


```python
sd.sql("DESCRIBE fun_table").show()
```

    ┌─────────────┬───────────┬─────────────┐
    │ column_name ┆ data_type ┆ is_nullable │
    │     utf8    ┆    utf8   ┆     utf8    │
    ╞═════════════╪═══════════╪═════════════╡
    │ val         ┆ Utf8      ┆ YES         │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ point       ┆ Binary    ┆ YES         │
    └─────────────┴───────────┴─────────────┘



```python
sd.sql("""
    SELECT *, ST_Centroid(ST_GeomFromWKB(point)) as centroid
    FROM fun_table
""").show()
```

    ┌───────┬─────────────────────────────────────────────┬────────────────────────────────────────────┐
    │  val  ┆                    point                    ┆                  centroid                  │
    │  utf8 ┆                    binary                   ┆                  geometry                  │
    ╞═══════╪═════════════════════════════════════════════╪════════════════════════════════════════════╡
    │ one   ┆ 0101000000000000000000f03f0000000000000040  ┆ POINT(1 2)                                 │
    ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ two   ┆ 0103000000010000000500000000000000008052c0… ┆ POINT(-73.95 40.75)                        │
    ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ three ┆ 010200000003000000aaf1d24d628052c05e4bc807… ┆ POINT(-73.92111155675562 40.7664673976246… │
    └───────┴─────────────────────────────────────────────┴────────────────────────────────────────────┘
