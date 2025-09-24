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
    │ 97a89436-5295-417e-bde1-2dbf8fe10700 ┆ POLYGON((-107.0377042 28.8102782,-107.… ┆ … ┆             │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ e09fcdeb-65ab-443c-bc59-4f9acf32af17 ┆ POLYGON((-107.0372022 28.8103812,-107.… ┆ … ┆             │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ cb06e75f-66a0-40ca-9029-e1a24dff1434 ┆ POLYGON((-107.0372848 28.8104923,-107.… ┆ … ┆             │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 5b8eaaca-5628-41d2-9e74-8c4c1205ed54 ┆ POLYGON((-107.0371986 28.8104859,-107.… ┆ … ┆             │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 1a4fba29-6160-4a98-b8f7-03395fd6466a ┆ POLYGON((-107.0373331 28.8105281,-107.… ┆ … ┆             │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 9de2d803-ab58-46e1-9a35-7423f8d0d121 ┆ POLYGON((-107.0378928 28.8112268,-107.… ┆ … ┆             │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 1704c72b-6f2b-445b-b9a3-36fdd0c74233 ┆ POLYGON((-107.0376099 28.8112085,-107.… ┆ … ┆             │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ b30bf41d-479f-4ee5-bd45-d4e17ffd8875 ┆ POLYGON((-107.0377094 28.8115554,-107.… ┆ … ┆             │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ c88928d4-b165-4335-9bd8-ccc5c9f79cee ┆ POLYGON((-107.0374986 28.8111847,-107.… ┆ … ┆             │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ ca1000b0-6af3-47fc-bdf2-61e8b1b716fc ┆ POLYGON((-107.0370972 28.8115247,-107.… ┆ … ┆             │
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
    │ cd743a89-9507-… ┆ POLYGON((111.… ┆ {xmin: 111.40… ┆ … ┆ true           ┆        ┆ e08d9394-cbc8… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ eb9f314e-1184-… ┆ POLYGON((112.… ┆ {xmin: 112.15… ┆ … ┆ true           ┆        ┆ aa2561c6-a578… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 89fbee90-bca0-… ┆ MULTIPOLYGON(… ┆ {xmin: 112.15… ┆ … ┆ true           ┆        ┆ 1cf6b19e-756f… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 758c0e1e-67aa-… ┆ POLYGON((112.… ┆ {xmin: 112.27… ┆ … ┆ true           ┆        ┆ 505f563f-7aa3… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ e5d4c707-66d7-… ┆ POLYGON((109.… ┆ {xmin: 109.64… ┆ … ┆ true           ┆ CN-HI  ┆ 86939aee-1fbd… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 26015e14-62ae-… ┆ POLYGON((109.… ┆ {xmin: 109.66… ┆ … ┆ true           ┆ CN-HI  ┆ ed052bae-e26f… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ eb42fd17-ad1e-… ┆ POLYGON((109.… ┆ {xmin: 109.83… ┆ … ┆ true           ┆ CN-HI  ┆ 201eeb03-4f5a… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ d63e1220-a308-… ┆ POLYGON((109.… ┆ {xmin: 109.90… ┆ … ┆ true           ┆ CN-HI  ┆ 71e30c32-2bc1… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ e62cef5e-6768-… ┆ POLYGON((109.… ┆ {xmin: 109.74… ┆ … ┆ true           ┆ CN-HI  ┆ 24478621-6802… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ c6fcbed1-ad8d-… ┆ POLYGON((110.… ┆ {xmin: 110.00… ┆ … ┆ true           ┆ CN-HI  ┆ ac0f434d-1e10… │
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
    CPU times: user 1.31 ms, sys: 1.02 ms, total: 2.33 ms
    Wall time: 2.54 ms


## Visualize the results with lonboard


```python
import lonboard

lonboard.viz(df)
```




    Map(basemap_style=<CartoBasemap.DarkMatter: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json'…



![Lonboard NS](image/lonboard_ns.png)
