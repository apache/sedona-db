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
    "s3://overturemaps-us-west-2/release/2026-09-23.1/theme=divisions/type=division_area/",
    options={"aws.skip_signature": True, "aws.region": "us-west-2"},
).to_view("divisions")
```

We can preview the first few rows using `.show()`. Because this is a lazy operation and we've already cached the schema using `.to_view()`, this only takes a few seconds.


```python
sd.view("divisions").show(5)
```

    ┌──────────────┬──────────────┬─────────┬──────────────┬───┬──────────────┬─────────┬──────────────┐
    │      id      ┆   geometry   ┆ country ┆    sources   ┆ … ┆  division_id ┆ version ┆     bbox     │
    │     utf8     ┆   geometry   ┆   utf8  ┆     list     ┆   ┆     utf8     ┆  int32  ┆    struct    │
    ╞══════════════╪══════════════╪═════════╪══════════════╪═══╪══════════════╪═════════╪══════════════╡
    │ e2be6aeb-80… ┆ MULTIPOLYGO… ┆ CU      ┆ [{property:… ┆ … ┆ 571dfc68-c4… ┆       6 ┆ {xmin: -83.… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 15f9cf35-b4… ┆ POLYGON((-8… ┆ CU      ┆ [{property:… ┆ … ┆ b0cca183-05… ┆       1 ┆ {xmin: -82.… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 73d74f48-97… ┆ POLYGON((-8… ┆ CU      ┆ [{property:… ┆ … ┆ b267d0d1-b7… ┆       5 ┆ {xmin: -82.… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 9b53b7a8-13… ┆ POLYGON((-8… ┆ CU      ┆ [{property:… ┆ … ┆ c821a5eb-fd… ┆       1 ┆ {xmin: -82.… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 5a4c6b4f-c0… ┆ POLYGON((-8… ┆ CU      ┆ [{property:… ┆ … ┆ f1c33ec6-ae… ┆       1 ┆ {xmin: -82.… │
    └──────────────┴──────────────┴─────────┴──────────────┴───┴──────────────┴─────────┴──────────────┘


The default view of the data hides some columns to ensure the entire output can be shown. To look at all the columns with type details, use `.schema`:


```python
sd.view("divisions").schema
```




    SedonaSchema with 14 fields:
      id: utf8<Utf8View>
      geometry: geometry<WkbView(ogc:crs84)>
      country: utf8<Utf8View>
      sources: list<List(Struct("property": Utf8, "dataset": Utf8, "license": Utf8, "record_id": Utf8, "update_time": Utf8, "confidence": Float64, "between": List(Float64, field: 'element'), "provider": Utf8, "resource": Utf8, "version": Utf8), field: 'element')>
      subtype: utf8<Utf8View>
      admin_level: int32<Int32>
      class: utf8<Utf8View>
      names: struct<Struct("primary": Utf8, "common": Map("key_value": non-null Struct("key": non-null Utf8, "value": Utf8), unsorted), "rules": List(Struct("variant": Utf8, "language": Utf8, "perspectives": Struct("mode": Utf8, "countries": List(Utf8, field: 'element')), "value": Utf8, "between": List(Float64, field: 'element'), "side": Utf8), field: 'element'))>
      is_land: boolean<Boolean>
      is_territorial: boolean<Boolean>
      region: utf8<Utf8View>
      division_id: utf8<Utf8View>
      version: int32<Int32>
      bbox: struct<Struct("xmin": Float64, "xmax": Float64, "ymin": Float64, "ymax": Float64)>



Overture data makes heavy use of nested types. These can be indexed into or expanded using SQL:


```python
sd.sql(
    "SELECT names.primary AS name, geometry FROM divisions WHERE region = 'CA-NS'"
).show(5)
```

    ┌─────────────────┬────────────────────────────────────────────────────────────────────────────────┐
    │       name      ┆                                    geometry                                    │
    │       utf8      ┆                                    geometry                                    │
    ╞═════════════════╪════════════════════════════════════════════════════════════════════════════════╡
    │ Canso           ┆ POLYGON((-60.9939805 45.3523129,-60.9940236 45.3534661,-60.9940748 45.3546192… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Pondville South ┆ POLYGON((-60.951331 45.5363495,-60.9515096 45.5366057,-60.9516036 45.536732,-… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Boudreauville   ┆ POLYGON((-60.9694031 45.5087873,-60.9689416 45.5064224,-60.9739396 45.5059255… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ River Bourgeois ┆ POLYGON((-61.0152528 45.6714737,-61.0152967 45.6715587,-61.0260109 45.6685926… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Petit-de-Grat   ┆ POLYGON((-60.9519968 45.5242318,-60.9524737 45.5246381,-60.9526837 45.5249956… │
    └─────────────────┴────────────────────────────────────────────────────────────────────────────────┘


Like all remote tables, it is worth resolving a query into a concrete local table to avoid fetching unnecessary data on repeated queries. The `.to_memtable()` method can be used to resolve a remote table into memory (great for small results); `.to_parquet()` can be used to resolve a remote table to disk (great for medium to large results).


```python
sd.sql(
    "SELECT names.primary AS name, geometry FROM divisions WHERE region = 'CA-NS'"
).to_memtable().to_view("divisions_ns")

sd.view("divisions_ns").show(5)
```

    ┌─────────────────┬────────────────────────────────────────────────────────────────────────────────┐
    │       name      ┆                                    geometry                                    │
    │       utf8      ┆                                    geometry                                    │
    ╞═════════════════╪════════════════════════════════════════════════════════════════════════════════╡
    │ Seal Island     ┆ POLYGON((-65.9481828 43.4398466,-66.0528452 43.4531336,-66.0883401 43.3978188… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Mud Island      ┆ POLYGON((-66.0216384 43.518589,-66.0222822 43.5166842,-66.0252286 43.5100071,… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Tusket Islands  ┆ POLYGON((-66.0341854 43.5998264,-66.033595 43.5985202,-66.0330826 43.5966887,… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Lower Wedgeport ┆ POLYGON((-65.9817164 43.7321666,-65.9825263 43.7327018,-65.9878682 43.7318659… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Chebogue Point  ┆ POLYGON((-66.1264091 43.6848494,-66.1078926 43.6952532,-66.0932243 43.7029995… │
    └─────────────────┴────────────────────────────────────────────────────────────────────────────────┘


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
    params={"wkb": ns_bbox_wkb},
).to_memtable().to_view("divisions_ns", overwrite=True)

sd.view("divisions_ns").show(5)
```

    ┌─────────────────┬────────────────────────────────────────────────────────────────────────────────┐
    │       name      ┆                                    geometry                                    │
    │       utf8      ┆                                    geometry                                    │
    ╞═════════════════╪════════════════════════════════════════════════════════════════════════════════╡
    │ Mud Island      ┆ POLYGON((-66.0216384 43.518589,-66.0222822 43.5166842,-66.0252286 43.5100071,… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Tusket Islands  ┆ POLYGON((-66.0341854 43.5998264,-66.033595 43.5985202,-66.0330826 43.5966887,… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Lower Wedgeport ┆ POLYGON((-65.9817164 43.7321666,-65.9825263 43.7327018,-65.9878682 43.7318659… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Chebogue Point  ┆ POLYGON((-66.1264091 43.6848494,-66.1078926 43.6952532,-66.0932243 43.7029995… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Comeaus Hill    ┆ POLYGON((-66.0396689 43.699741,-66.0437404 43.6954361,-66.0459004 43.6935827,… │
    └─────────────────┴────────────────────────────────────────────────────────────────────────────────┘


## Overture buildings table

The [Overture buildings table](https://docs.overturemaps.org/guides/buildings/) is one of the largest tables provided by the Overture Maps Foundation. The workflow is similar to the division table or any remote table; however, the buildings table presents several unique challenges.

First, the metadata size for all files in the buildings table is very large. SedonaDB caches remote metadata to avoid repeated download; however, the default cache size is too small. For repeated queries against the buildings table, ensure that the cache size is increased to at least 900 MB and/or `.to_view()` is used to cache the schema. The cache lives as long as the session...use `sd = sedona.db.connect()` or reset the cache size to a smaller value to clear the cache.

> Overture removes old releases. See [this page](https://docs.overturemaps.org/release-calendar/#current-release) to see the latest version number and replace the relevant portion of the URL below.


```python
sd.sql("SET datafusion.runtime.metadata_cache_limit = '900M'").execute()

sd.read_parquet(
    "s3://overturemaps-us-west-2/release/2026-09-23.1/theme=buildings/type=building/",
    options={"aws.skip_signature": True, "aws.region": "us-west-2"},
).to_view("buildings")
```

Like all SedonaDB DataFrames, viewing a schema or previewing the first few rows are lazy and usually fast unless a query contains large aggregations or joins.


```python
sd.view("buildings").show(5)
```

    ┌──────────────────────────────────────┬───┬───────────────────────────────────────────────────────┐
    │                  id                  ┆ … ┆                          bbox                         │
    │                 utf8                 ┆   ┆                         struct                        │
    ╞══════════════════════════════════════╪═══╪═══════════════════════════════════════════════════════╡
    │ 57663251-bddc-4327-8381-21d0b1387aad ┆ … ┆ {xmin: -72.21996307373047, xmax: -72.21990203857422,… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 60edbc3a-0bef-4ad1-9526-a78e4929d65b ┆ … ┆ {xmin: -72.22010803222656, xmax: -72.2200698852539, … │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ fc81281b-fa92-40fd-8e6a-a7d0af66f658 ┆ … ┆ {xmin: -72.22064971923828, xmax: -72.2205810546875, … │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ ca6e790a-9bd8-4a88-91cf-6d4a8e973931 ┆ … ┆ {xmin: -72.2198715209961, xmax: -72.21979522705078, … │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 7ad78e04-753c-487e-a322-d5fd965832dd ┆ … ┆ {xmin: -72.21929931640625, xmax: -72.21922302246094,… │
    └──────────────────────────────────────┴───┴───────────────────────────────────────────────────────┘


Some operations like `.count()` use summary statistics and execute quickly even for large remote tables:


```python
sd.view("buildings").count()
```




    2533842612



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
    params=(nyc_bbox_wkt,),
).to_memtable().to_view("buildings_nyc")

sd.view("buildings_nyc").show(5)
```

    ┌───────────────────────────────┬─────────┬────────────┬────────────┬──────────────────────────────┐
    │               id              ┆  height ┆ num_floors ┆ roof_shape ┆           centroid           │
    │              utf8             ┆ float64 ┆    int32   ┆    utf8    ┆           geometry           │
    ╞═══════════════════════════════╪═════════╪════════════╪════════════╪══════════════════════════════╡
    │ baa975bc-05cd-4b61-8601-ee4c… ┆    21.4 ┆            ┆            ┆ POINT(-73.90407354738332 40… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 0c13ed5f-7910-40a9-b0b2-bdcc… ┆    25.4 ┆            ┆            ┆ POINT(-73.8991184967379 40.… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ d5850743-2733-4d65-ad0d-176e… ┆    21.2 ┆            ┆            ┆ POINT(-73.89783640847207 40… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 2c2cd855-e561-4bea-8a06-0a57… ┆    20.2 ┆            ┆            ┆ POINT(-73.89800666906386 40… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 2ed122f6-0f6e-48b5-acab-8b85… ┆    20.2 ┆            ┆            ┆ POINT(-73.8978977796945 40.… │
    └───────────────────────────────┴─────────┴────────────┴────────────┴──────────────────────────────┘
