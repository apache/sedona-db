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

> Note: Before running this notebook, ensure that you have installed SedonaDB: `pip install "sedona[db]"`

This notebook demonstrates how to query and analyze the [Overture Maps](https://overturemaps.org/) dataset using SedonaDB.

The notebook explains how to:
* Load Overture data for the `buildings` and `divisions` themes directly from S3.
* Perform spatial queries to find features within a specific geographic area.
* Optimize subsequent query performance by caching a subset of data in memory.


```python
import sedona.db
import os

os.environ["AWS_SKIP_SIGNATURE"] = "true"
os.environ["AWS_DEFAULT_REGION"] = "us-west-2"

sd = sedona.db.connect()
```

## Overture buildings table


```python
df = sd.read_parquet(
    "s3://overturemaps-us-west-2/release/2025-08-20.0/theme=buildings/type=building/"
)
```


```python
df.limit(10).show()
```

    ┌──────────────────────────────────────┬─────────────────────────────────────────┬───┬─────────────┐
    │                  id                  ┆                 geometry                ┆ … ┆ roof_height │
    │                 utf8                 ┆                 geometry                ┆   ┆   float64   │
    ╞══════════════════════════════════════╪═════════════════════════════════════════╪═══╪═════════════╡
    │ 76045c54-bd50-415b-9ac8-35cfc22fe80f ┆ POLYGON((-105.7936939 25.7460051,-105.… ┆ … ┆             │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ f1eb3f3e-c9b9-40af-b83e-7f2c09ef8e43 ┆ POLYGON((-105.7971397 25.7517374,-105.… ┆ … ┆             │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ f6019f66-3137-4e62-9d84-e4f4febe2f6d ┆ POLYGON((-105.7928137 25.7493614,-105.… ┆ … ┆             │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 37b96ac3-0093-47d4-a898-263f6550ef14 ┆ POLYGON((-105.7857365 25.7465883,-105.… ┆ … ┆             │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ e686ab8d-7062-4c0a-a8da-32ab751f7212 ┆ POLYGON((-105.7579541 25.7150521,-105.… ┆ … ┆             │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 767a736b-ac77-4427-89f0-3f9544b1e9e9 ┆ POLYGON((-105.7576387 25.7149905,-105.… ┆ … ┆             │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ db3a2233-a7cb-425a-ac75-ad83cf868a32 ┆ POLYGON((-105.7589831 25.71648,-105.75… ┆ … ┆             │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 53866308-afc7-4ff1-88f6-b2af571238e3 ┆ POLYGON((-105.76268 25.7197151,-105.76… ┆ … ┆             │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 31d8f64e-b7f9-4b26-8c2a-2fd205fed995 ┆ POLYGON((-105.7626968 25.7197994,-105.… ┆ … ┆             │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 72ec6ed0-d33c-4e64-8c8f-28e7a6ac6310 ┆ POLYGON((-105.7578861 25.7190388,-105.… ┆ … ┆             │
    └──────────────────────────────────────┴─────────────────────────────────────────┴───┴─────────────┘



```python
df.to_view("buildings")
```


```python
# the buildings table is large and contains billions of rows
sd.sql("""
SELECT
    COUNT(*)
FROM
    buildings
""").show()
```

    ┌────────────┐
    │  count(*)  │
    │    int64   │
    ╞════════════╡
    │ 2539170484 │
    └────────────┘



```python
# check out the schema of the buildings table to see what it contains
df.schema
```




    SedonaSchema with 24 fields:
      id: utf8<Utf8View>
      geometry: geometry<WkbView(ogc:crs84)>
      bbox: struct<Struct(xmin Float32, xmax Float32, ymin Float32, ymax Float32)>
      version: int32<Int32>
      sources: list<List(Field { name: "element", data_type: Struct([Field { name: "property", data_type: Utf8, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }, Field { name: "dataset", data_type: Utf8, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }, Field { name: "record_id", data_type: Utf8, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }, Field { name: "update_time", data_type: Utf8, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }, Field { name: "confidence", data_type: Float64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }, Field { name: "between", data_type: List(Field { name: "element", data_type: Float64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }]), nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} })>
      level: int32<Int32>
      subtype: utf8<Utf8View>
      class: utf8<Utf8View>
      height: float64<Float64>
      names: struct<Struct(primary Utf8, common Map(Field { name: "key_value", data_type: Struct([Field { name: "key", data_type: Utf8, nullable: false, dict_id: 0, dict_is_ordered: false, metadata: {} }, Field { name: "value", data_type: Utf8, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }]), nullable: false, dict_id: 0, dict_is_ordered: false, metadata: {} }, false), rules List(Field { name: "element", data_type: Struct([Field { name: "variant", data_type: Utf8, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }, Field { name: "language", data_type: Utf8, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }, Field { name: "perspectives", data_type: Struct([Field { name: "mode", data_type: Utf8, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }, Field { name: "countries", data_type: List(Field { name: "element", data_type: Utf8, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }]), nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }, Field { name: "value", data_type: Utf8, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }, Field { name: "between", data_type: List(Field { name: "element", data_type: Float64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }, Field { name: "side", data_type: Utf8, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }]), nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }))>
      has_parts: boolean<Boolean>
      is_underground: boolean<Boolean>
      num_floors: int32<Int32>
      num_floors_underground: int32<Int32>
      min_height: float64<Float64>
      min_floor: int32<Int32>
      facade_color: utf8<Utf8View>
      facade_material: utf8<Utf8View>
      roof_material: utf8<Utf8View>
      roof_shape: utf8<Utf8View>
      roof_direction: float64<Float64>
      roof_orientation: utf8<Utf8View>
      roof_color: utf8<Utf8View>
      roof_height: float64<Float64>




```python
# find all the buildings in New York City that are taller than 20 meters
nyc_bbox_wkt = (
    "POLYGON((-74.2591 40.4774, -74.2591 40.9176, -73.7004 40.9176, "
    "-73.7004 40.4774, -74.2591 40.4774))"
)
sd.sql(f"""
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
        ST_SetSRID(ST_GeomFromText('{nyc_bbox_wkt}'), 4326)
    )
LIMIT 5;
""").show()
```

    ┌─────────────────────────┬────────────────────┬────────────┬────────────┬─────────────────────────┐
    │            id           ┆       height       ┆ num_floors ┆ roof_shape ┆         centroid        │
    │           utf8          ┆       float64      ┆    int32   ┆    utf8    ┆         geometry        │
    ╞═════════════════════════╪════════════════════╪════════════╪════════════╪═════════════════════════╡
    │ 1b9040c2-2e79-4f56-aba… ┆               22.4 ┆            ┆            ┆ POINT(-74.230407502993… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 1b5e1cd2-d697-489e-892… ┆               21.5 ┆            ┆            ┆ POINT(-74.231451103592… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ c1afdf78-bf84-4b8f-ae1… ┆               20.9 ┆            ┆            ┆ POINT(-74.232593032240… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 88f36399-b09f-491b-bb6… ┆               24.5 ┆            ┆            ┆ POINT(-74.231878209597… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ df37a283-f5bd-4822-a05… ┆ 24.154542922973633 ┆            ┆            ┆ POINT(-74.241910239840… │
    └─────────────────────────┴────────────────────┴────────────┴────────────┴─────────────────────────┘


## Overture divisions table


```python
df = sd.read_parquet(
    "s3://overturemaps-us-west-2/release/2025-08-20.0/theme=divisions/type=division_area/"
)
```


```python
# inspect a few rows of the data
df.show(10)
```

    ┌─────────────────┬────────────────┬────────────────┬───┬────────────────┬────────┬────────────────┐
    │        id       ┆    geometry    ┆      bbox      ┆ … ┆ is_territorial ┆ region ┆   division_id  │
    │       utf8      ┆    geometry    ┆     struct     ┆   ┆     boolean    ┆  utf8  ┆      utf8      │
    ╞═════════════════╪════════════════╪════════════════╪═══╪════════════════╪════════╪════════════════╡
    │ 6dd2a4d4-542f-… ┆ POLYGON((-82.… ┆ {xmin: -82.38… ┆ … ┆ true           ┆ PA-4   ┆ 4b010ede-3285… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 5eef9108-6811-… ┆ POLYGON((-82.… ┆ {xmin: -82.38… ┆ … ┆ true           ┆ PA-4   ┆ 4aa05d3a-0300… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ fe123780-bf85-… ┆ POLYGON((-82.… ┆ {xmin: -82.38… ┆ … ┆ true           ┆ PA-4   ┆ 63c6570b-70fd… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ c0f0c861-a3bc-… ┆ POLYGON((-82.… ┆ {xmin: -82.38… ┆ … ┆ true           ┆ PA-4   ┆ c3fc4642-f97b… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ a57076cb-c74d-… ┆ POLYGON((-82.… ┆ {xmin: -82.38… ┆ … ┆ true           ┆ PA-4   ┆ 5d206247-f4b2… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ d86411e8-02a1-… ┆ POLYGON((-82.… ┆ {xmin: -82.40… ┆ … ┆ true           ┆ PA-4   ┆ c5564623-b42e… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ d13f593f-7fde-… ┆ POLYGON((-82.… ┆ {xmin: -82.39… ┆ … ┆ true           ┆ PA-4   ┆ b02aa2a3-3889… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 3598138e-788a-… ┆ POLYGON((-82.… ┆ {xmin: -82.38… ┆ … ┆ true           ┆ PA-4   ┆ 6700edfc-96e5… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ a5cecb4f-d6d1-… ┆ POLYGON((-82.… ┆ {xmin: -82.38… ┆ … ┆ true           ┆ PA-4   ┆ 9f24a80b-355d… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ a4d912fc-69d9-… ┆ POLYGON((-82.… ┆ {xmin: -82.37… ┆ … ┆ true           ┆ PA-4   ┆ 65786eba-d8fd… │
    └─────────────────┴────────────────┴────────────────┴───┴────────────────┴────────┴────────────────┘



```python
df.to_view("division_area")
```


```python
sd.sql("""
SELECT
    COUNT(*)
FROM division_area
""").show()
```

    ┌──────────┐
    │ count(*) │
    │   int64  │
    ╞══════════╡
    │  1035749 │
    └──────────┘



```python
df.schema
```




    SedonaSchema with 13 fields:
      id: utf8<Utf8View>
      geometry: geometry<WkbView(ogc:crs84)>
      bbox: struct<Struct(xmin Float32, xmax Float32, ymin Float32, ymax Float32)>
      country: utf8<Utf8View>
      version: int32<Int32>
      sources: list<List(Field { name: "element", data_type: Struct([Field { name: "property", data_type: Utf8, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }, Field { name: "dataset", data_type: Utf8, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }, Field { name: "record_id", data_type: Utf8, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }, Field { name: "update_time", data_type: Utf8, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }, Field { name: "confidence", data_type: Float64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }, Field { name: "between", data_type: List(Field { name: "element", data_type: Float64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }]), nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} })>
      subtype: utf8<Utf8View>
      class: utf8<Utf8View>
      names: struct<Struct(primary Utf8, common Map(Field { name: "key_value", data_type: Struct([Field { name: "key", data_type: Utf8, nullable: false, dict_id: 0, dict_is_ordered: false, metadata: {} }, Field { name: "value", data_type: Utf8, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }]), nullable: false, dict_id: 0, dict_is_ordered: false, metadata: {} }, false), rules List(Field { name: "element", data_type: Struct([Field { name: "variant", data_type: Utf8, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }, Field { name: "language", data_type: Utf8, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }, Field { name: "perspectives", data_type: Struct([Field { name: "mode", data_type: Utf8, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }, Field { name: "countries", data_type: List(Field { name: "element", data_type: Utf8, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }]), nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }, Field { name: "value", data_type: Utf8, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }, Field { name: "between", data_type: List(Field { name: "element", data_type: Float64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }, Field { name: "side", data_type: Utf8, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }]), nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }))>
      is_land: boolean<Boolean>
      is_territorial: boolean<Boolean>
      region: utf8<Utf8View>
      division_id: utf8<Utf8View>




```python
# get all the divisions in Nova Scotia and save them in memory with to_memtable()
nova_scotia_bbox_wkt = (
    "POLYGON((-66.5 43.4, -66.5 47.1, -59.8 47.1, -59.8 43.4, -66.5 43.4))"
)
ns = sd.sql(f"""
SELECT
    country, region, names, geometry
FROM division_area
WHERE
    ST_Intersects(
        geometry,
        ST_SetSRID(ST_GeomFromText('{nova_scotia_bbox_wkt}'), 4326)
    )
""").to_memtable()
```


```python
ns.to_view("ns_divisions")
```


```python
df = sd.sql("""
SELECT UNNEST(names), geometry
FROM ns_divisions
WHERE region = 'CA-NS'
""")
```


```python
%%time
# this executes quickly because the Nova Scotia data was persisted in memory with `to_memtable()`
df.show(2)
```

    ┌────────────────────────┬────────────────────────┬────────────────────────┬───────────────────────┐
    │ __unnest_placeholder(n ┆ __unnest_placeholder(n ┆ __unnest_placeholder(n ┆        geometry       │
    │ s_divisions.names).pr… ┆ s_divisions.names).co… ┆ s_divisions.names).ru… ┆        geometry       │
    ╞════════════════════════╪════════════════════════╪════════════════════════╪═══════════════════════╡
    │ Seal Island            ┆                        ┆                        ┆ POLYGON((-66.0528452… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Mud Island             ┆                        ┆                        ┆ POLYGON((-66.0222822… │
    └────────────────────────┴────────────────────────┴────────────────────────┴───────────────────────┘
    CPU times: user 1.2 ms, sys: 844 μs, total: 2.04 ms
    Wall time: 2.03 ms
