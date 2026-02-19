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

# SedonaDB Overture Examples

> Note: Before running this notebook, ensure that you have installed SedonaDB: `pip install "apache-sedona[db]"`

This notebook demonstrates how to query and analyze the [Overture Maps](https://overturemaps.org/) dataset using SedonaDB.

The notebook explains how to:

* Load Overture data for the `buildings` and `divisions` themes directly from S3.
* Perform spatial queries to find features within a specific geographic area.
* Optimize subsequent query performance by caching a subset of data in memory.


```python
import sedona.db

sd = sedona.db.connect()
```

## Overture divisions

Let's load a table! Like any local or remote collection of Parquet files, we can use `sd.read_parquet()`. This is a lazy operation, fetching only metadata required to calculate a table schema. To reduce the number of times this needs to happen (and make the resulting DataFrame easier to reference in SQL), we use `.to_view()`.

> Overture removes old releases. See [this page](https://docs.overturemaps.org/release-calendar/#current-release) to see the latest version number and replace the relevant portion of the URL below.


```python
sd.read_parquet(
    "s3://overturemaps-us-west-2/release/2026-02-18.0/theme=divisions/type=division_area/",
    options={"aws.skip_signature": True, "aws.region": "us-west-2"}
).to_view("divisions")
```

We can preview the first few rows using `.show()`. Because this is a lazy operation and we've already cached the schema using `.to_view()`, this only takes a few seconds.


```python
sd.view("divisions").show(5)
```

    ┌───────────────┬───────────────┬──────────────┬─────────┬───┬────────┬─────────────┬──────────────┐
    │       id      ┆    geometry   ┆     bbox     ┆ country ┆ … ┆ region ┆ admin_level ┆  division_id │
    │      utf8     ┆    geometry   ┆    struct    ┆   utf8  ┆   ┆  utf8  ┆    int32    ┆     utf8     │
    ╞═══════════════╪═══════════════╪══════════════╪═════════╪═══╪════════╪═════════════╪══════════════╡
    │ a5c573c4-022… ┆ POLYGON((-49… ┆ {xmin: -49.… ┆ BR      ┆ … ┆ BR-PR  ┆             ┆ 388a8056-ee… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ cf523f8c-c26… ┆ POLYGON((-49… ┆ {xmin: -49.… ┆ BR      ┆ … ┆ BR-PR  ┆             ┆ 068ef37e-3b… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 8ace3d06-b8a… ┆ POLYGON((-49… ┆ {xmin: -49.… ┆ BR      ┆ … ┆ BR-PR  ┆             ┆ 7238aeb3-b8… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ b26d2cba-b54… ┆ POLYGON((-49… ┆ {xmin: -49.… ┆ BR      ┆ … ┆ BR-PR  ┆             ┆ 3c2dc8fc-79… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 20103725-17c… ┆ POLYGON((-49… ┆ {xmin: -49.… ┆ BR      ┆ … ┆ BR-PR  ┆             ┆ 45037e82-de… │
    └───────────────┴───────────────┴──────────────┴─────────┴───┴────────┴─────────────┴──────────────┘


The default view of the data hides some columns to ensure the entire output can be shown. To look at all the columns with type details, use `.schema`:


```python
sd.view("divisions").schema
```




    SedonaSchema with 14 fields:
      id: utf8<Utf8View>
      geometry: geometry<WkbView(ogc:crs84)>
      bbox: struct<Struct("xmin": Float32, "xmax": Float32, "ymin": Float32, "ymax": Float32)>
      country: utf8<Utf8View>
      version: int32<Int32>
      sources: list<List(Struct("property": Utf8, "dataset": Utf8, "license": Utf8, "record_id": Utf8, "update_time": Utf8, "confidence": Float64, "between": List(Float64, field: 'element')), field: 'element')>
      subtype: utf8<Utf8View>
      class: utf8<Utf8View>
      names: struct<Struct("primary": Utf8, "common": Map("key_value": non-null Struct("key": non-null Utf8, "value": Utf8), unsorted), "rules": List(Struct("variant": Utf8, "language": Utf8, "perspectives": Struct("mode": Utf8, "countries": List(Utf8, field: 'element')), "value": Utf8, "between": List(Float64, field: 'element'), "side": Utf8), field: 'element'))>
      is_land: boolean<Boolean>
      is_territorial: boolean<Boolean>
      region: utf8<Utf8View>
      admin_level: int32<Int32>
      division_id: utf8<Utf8View>



Overture data makes heavy use of nested types. These can be indexed into or expanded using SQL:


```python
sd.sql("SELECT names.primary AS name, geometry FROM divisions WHERE region = 'CA-NS'").show(5)
```

    ┌────────────────────────────────────┬─────────────────────────────────────────────────────────────┐
    │                name                ┆                           geometry                          │
    │                utf8                ┆                           geometry                          │
    ╞════════════════════════════════════╪═════════════════════════════════════════════════════════════╡
    │ Sable Island National Park Reserve ┆ POLYGON((-60.178333 43.9824655,-60.1785682 43.9825425,-60.… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Sable Island                       ┆ POLYGON((-59.7744732 44.2254616,-59.7928902 44.2173253,-59… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Halifax Regional Municipality      ┆ MULTIPOLYGON(((-59.7321078 44.2390248,-59.7502166 44.23385… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ West Liscomb                       ┆ POLYGON((-62.0615594 45.0023306,-62.0621839 45.0024475,-62… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Marie Joseph                       ┆ POLYGON((-61.9911914 44.95646,-61.9912383 44.9579526,-61.9… │
    └────────────────────────────────────┴─────────────────────────────────────────────────────────────┘


Like all remote tables, it is worth resolving a query into a concrete local table to avoid fetching unnecessary data on repeated queries. The `.to_memtable()` method can be used to resolve a remote table into memory (great for small results); `.to_parquet()` can be used to resolve a remote table to disk (great for medium to large results).


```python
sd.sql(
    "SELECT names.primary AS name, geometry FROM divisions WHERE region = 'CA-NS'"
).to_memtable().to_view("divisions_ns")

sd.view("divisions_ns").show(5)
```

    ┌────────────────────────────────────┬─────────────────────────────────────────────────────────────┐
    │                name                ┆                           geometry                          │
    │                utf8                ┆                           geometry                          │
    ╞════════════════════════════════════╪═════════════════════════════════════════════════════════════╡
    │ Sable Island National Park Reserve ┆ POLYGON((-60.178333 43.9824655,-60.1785682 43.9825425,-60.… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Sable Island                       ┆ POLYGON((-59.7744732 44.2254616,-59.7928902 44.2173253,-59… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Halifax Regional Municipality      ┆ MULTIPOLYGON(((-59.7321078 44.2390248,-59.7502166 44.23385… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ West Liscomb                       ┆ POLYGON((-62.0615594 45.0023306,-62.0621839 45.0024475,-62… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Marie Joseph                       ┆ POLYGON((-61.9911914 44.95646,-61.9912383 44.9579526,-61.9… │
    └────────────────────────────────────┴─────────────────────────────────────────────────────────────┘


Importantly, Overture data is distributed using GeoParquet 1.1, for which SedonaDB has built in support! This means that spatial queries (e.g., `ST_Intersects()`) tend to execute quickly against overture. In this case, the spatial query for Nova Scotia is ~5x faster than the text-based region query.


```python
import shapely

ns_bbox_wkb = shapely.box(-66.5, 43.4, -59.8, 47.1).wkb

sd.sql(
    """
    SELECT names.primary AS name, geometry
    FROM divisions
    WHERE ST_Contains(ST_GeomFromWKB($wkb, 4326), geometry)
    """,
    params={"wkb": ns_bbox_wkb}
).to_memtable().to_view("divisions_ns", overwrite=True)

sd.view("divisions_ns").show(5)
```

    ┌───────────────────┬──────────────────────────────────────────────────────────────────────────────┐
    │        name       ┆                                   geometry                                   │
    │        utf8       ┆                                   geometry                                   │
    ╞═══════════════════╪══════════════════════════════════════════════════════════════════════════════╡
    │ Maces Bay         ┆ POLYGON((-66.4491254 45.1265729,-66.4577261 45.126933,-66.4591563 45.126991… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Gooseberry Island ┆ POLYGON((-66.2598821 45.1380421,-66.2599962 45.1381233,-66.2600591 45.13828… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Musquash Parish   ┆ POLYGON((-66.4595418 45.2215004,-66.4595406 45.221468,-66.4595396 45.221391… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Dipper Harbour    ┆ POLYGON((-66.3755086 45.118812,-66.4089711 45.1488327,-66.4284252 45.138119… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Chance Harbour    ┆ POLYGON((-66.4089711 45.1488327,-66.3755086 45.118812,-66.3541725 45.105991… │
    └───────────────────┴──────────────────────────────────────────────────────────────────────────────┘


## Overture buildings table

The [Overture buildings table](https://docs.overturemaps.org/guides/buildings/) is one of the largest tables provided by the Overture Maps Foundation. The workflow is similar to the division table or any remote table; however, the buildings table presents several unique challeneges.

First, the metadata size for all files in the buildings table is very large. SedonaDB caches remote metadata to avoid repeated download; however, the default cache size is too small. For repeated queries against the buildings table, ensure that the cache size is increased and/or `.to_view()` is used to cache the schema.

> Overture removes old releases. See [this page](https://docs.overturemaps.org/release-calendar/#current-release) to see the latest version number and replace the relevant portion of the URL below.


```python
sd.sql("SET datafusion.runtime.metadata_cache_limit = '900M'").execute()

sd.read_parquet(
    "s3://overturemaps-us-west-2/release/2026-02-18.0/theme=buildings/type=building/",
    options={"aws.skip_signature": True, "aws.region": "us-west-2"}
).to_view("buildings")
```

Like all SedonaDB DataFrames, viewing a schema or previewing the first few rows are lazy and usually fast unless a query contains large aggregations or joins.


```python
sd.view("buildings").show(5)
```

    ┌──────────────────────────────────────┬─────────────────────────────────────────┬───┬─────────────┐
    │                  id                  ┆                 geometry                ┆ … ┆ roof_height │
    │                 utf8                 ┆                 geometry                ┆   ┆   float64   │
    ╞══════════════════════════════════════╪═════════════════════════════════════════╪═══╪═════════════╡
    │ ab23f7ee-4c05-4246-a016-8260ce58a916 ┆ POLYGON((-67.589523 -39.0908362,-67.58… ┆ … ┆             │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 58356258-2e80-48fc-aacf-d81fcf74074c ┆ POLYGON((-67.5896327 -39.0907868,-67.5… ┆ … ┆             │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ b50595a8-cddb-44dd-bdbf-7bbe1e858ae0 ┆ POLYGON((-67.5897117 -39.0908483,-67.5… ┆ … ┆             │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ cbabe2df-f49a-4e9f-9cbe-c527a4b3b9f1 ┆ POLYGON((-67.5898768 -39.0907073,-67.5… ┆ … ┆             │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ bcd6984b-8da4-4dfe-9212-be2b02a24b67 ┆ POLYGON((-67.5901879 -39.0908288,-67.5… ┆ … ┆             │
    └──────────────────────────────────────┴─────────────────────────────────────────┴───┴─────────────┘


Some operations like `.count()` use summary statistics and execute quickly even for large remote tables:


```python
sd.view("buildings").count()
```




    2541282557



Overture buildings has a number of attributes on which we can filter. For long-running queries it may be convenient to cache a result locally using `.to_memtable()` or `.to_parquet()` before inspecting using other tools; however like all Overture tables it is optimized for spatial queries and these are usually not expensive for small areas.

For example, we can find all of the buildings in New York City taller than 20 meters:


```python
nyc_bbox_wkt = (
    "POLYGON((-74.2591 40.4774, -74.2591 40.9176, -73.7004 40.9176, "
    "-73.7004 40.4774, -74.2591 40.4774))"
)

sd.sql(
    """
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
        AND ST_Intersects(
            geometry,
            ST_GeomFromText($1, 4326)
        )
    LIMIT 5;
    """,
    params=(nyc_bbox_wkt,)
).to_memtable().to_view("buildings_nyc")

sd.view("buildings_nyc").show(5)
```

    ┌─────────────────────────┬────────────────────┬────────────┬────────────┬─────────────────────────┐
    │            id           ┆       height       ┆ num_floors ┆ roof_shape ┆         centroid        │
    │           utf8          ┆       float64      ┆    int32   ┆    utf8    ┆         geometry        │
    ╞═════════════════════════╪════════════════════╪════════════╪════════════╪═════════════════════════╡
    │ aa8e3a73-c72c-4f1a-b6e… ┆  20.38205909729004 ┆            ┆            ┆ POINT(-74.187673580307… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ efe7616b-7f7e-464c-9ce… ┆  26.18361473083496 ┆            ┆            ┆ POINT(-74.189040982134… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ b3f734a1-325b-4e8c-b1d… ┆ 27.025876998901367 ┆            ┆            ┆ POINT(-74.2558161 40.8… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 45d88655-e2f4-4a08-926… ┆ 25.485210418701172 ┆            ┆            ┆ POINT(-74.182252194444… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 31e8353c-7d5b-4b20-94e… ┆ 21.294815063476562 ┆            ┆            ┆ POINT(-74.197113787905… │
    └─────────────────────────┴────────────────────┴────────────┴────────────┴─────────────────────────┘
