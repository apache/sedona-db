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

This notebook demonstrates how to query and analyze the [Overture Maps](https://overturemaps.org/) dataset using SedonaDB.  See [this page](https://docs.overturemaps.org/release-calendar/) to get the latest version of the Overture data.

The notebook explains how to:

* Load Overture data for the `buildings` and `divisions` themes directly from S3.
* Perform spatial queries to find features within a specific geographic area.
* Optimize subsequent query performance by caching a subset of data in memory.


```python
%pip install lonboard
```

    Requirement already satisfied: lonboard in /opt/miniconda3/lib/python3.12/site-packages (0.12.1)
    Requirement already satisfied: anywidget~=0.9.0 in /opt/miniconda3/lib/python3.12/site-packages (from lonboard) (0.9.18)
    Requirement already satisfied: arro3-compute>=0.4.1 in /opt/miniconda3/lib/python3.12/site-packages (from lonboard) (0.6.3)
    Requirement already satisfied: arro3-core>=0.4.1 in /opt/miniconda3/lib/python3.12/site-packages (from lonboard) (0.6.3)
    Requirement already satisfied: arro3-io>=0.4.1 in /opt/miniconda3/lib/python3.12/site-packages (from lonboard) (0.6.3)
    Requirement already satisfied: geoarrow-rust-core>=0.5.2 in /opt/miniconda3/lib/python3.12/site-packages (from lonboard) (0.5.2)
    Requirement already satisfied: ipywidgets>=7.6.0 in /opt/miniconda3/lib/python3.12/site-packages (from lonboard) (8.1.7)
    Requirement already satisfied: numpy>=1.14 in /opt/miniconda3/lib/python3.12/site-packages (from lonboard) (2.3.3)
    Requirement already satisfied: pyproj>=3.3 in /opt/miniconda3/lib/python3.12/site-packages (from lonboard) (3.7.2)
    Requirement already satisfied: traitlets>=5.7.1 in /opt/miniconda3/lib/python3.12/site-packages (from lonboard) (5.14.3)
    Requirement already satisfied: psygnal>=0.8.1 in /opt/miniconda3/lib/python3.12/site-packages (from anywidget~=0.9.0->lonboard) (0.14.1)
    Requirement already satisfied: typing-extensions>=4.2.0 in /opt/miniconda3/lib/python3.12/site-packages (from anywidget~=0.9.0->lonboard) (4.15.0)
    Requirement already satisfied: comm>=0.1.3 in /opt/miniconda3/lib/python3.12/site-packages (from ipywidgets>=7.6.0->lonboard) (0.2.3)
    Requirement already satisfied: ipython>=6.1.0 in /opt/miniconda3/lib/python3.12/site-packages (from ipywidgets>=7.6.0->lonboard) (9.5.0)
    Requirement already satisfied: widgetsnbextension~=4.0.14 in /opt/miniconda3/lib/python3.12/site-packages (from ipywidgets>=7.6.0->lonboard) (4.0.14)
    Requirement already satisfied: jupyterlab_widgets~=3.0.15 in /opt/miniconda3/lib/python3.12/site-packages (from ipywidgets>=7.6.0->lonboard) (3.0.15)
    Requirement already satisfied: certifi in /opt/miniconda3/lib/python3.12/site-packages (from pyproj>=3.3->lonboard) (2025.8.3)
    Requirement already satisfied: decorator in /opt/miniconda3/lib/python3.12/site-packages (from ipython>=6.1.0->ipywidgets>=7.6.0->lonboard) (5.2.1)
    Requirement already satisfied: ipython-pygments-lexers in /opt/miniconda3/lib/python3.12/site-packages (from ipython>=6.1.0->ipywidgets>=7.6.0->lonboard) (1.1.1)
    Requirement already satisfied: jedi>=0.16 in /opt/miniconda3/lib/python3.12/site-packages (from ipython>=6.1.0->ipywidgets>=7.6.0->lonboard) (0.19.2)
    Requirement already satisfied: matplotlib-inline in /opt/miniconda3/lib/python3.12/site-packages (from ipython>=6.1.0->ipywidgets>=7.6.0->lonboard) (0.1.7)
    Requirement already satisfied: pexpect>4.3 in /opt/miniconda3/lib/python3.12/site-packages (from ipython>=6.1.0->ipywidgets>=7.6.0->lonboard) (4.9.0)
    Requirement already satisfied: prompt_toolkit<3.1.0,>=3.0.41 in /opt/miniconda3/lib/python3.12/site-packages (from ipython>=6.1.0->ipywidgets>=7.6.0->lonboard) (3.0.52)
    Requirement already satisfied: pygments>=2.4.0 in /opt/miniconda3/lib/python3.12/site-packages (from ipython>=6.1.0->ipywidgets>=7.6.0->lonboard) (2.19.2)
    Requirement already satisfied: stack_data in /opt/miniconda3/lib/python3.12/site-packages (from ipython>=6.1.0->ipywidgets>=7.6.0->lonboard) (0.6.3)
    Requirement already satisfied: parso<0.9.0,>=0.8.4 in /opt/miniconda3/lib/python3.12/site-packages (from jedi>=0.16->ipython>=6.1.0->ipywidgets>=7.6.0->lonboard) (0.8.5)
    Requirement already satisfied: ptyprocess>=0.5 in /opt/miniconda3/lib/python3.12/site-packages (from pexpect>4.3->ipython>=6.1.0->ipywidgets>=7.6.0->lonboard) (0.7.0)
    Requirement already satisfied: wcwidth in /opt/miniconda3/lib/python3.12/site-packages (from prompt_toolkit<3.1.0,>=3.0.41->ipython>=6.1.0->ipywidgets>=7.6.0->lonboard) (0.2.14)
    Requirement already satisfied: executing>=1.2.0 in /opt/miniconda3/lib/python3.12/site-packages (from stack_data->ipython>=6.1.0->ipywidgets>=7.6.0->lonboard) (2.2.1)
    Requirement already satisfied: asttokens>=2.1.0 in /opt/miniconda3/lib/python3.12/site-packages (from stack_data->ipython>=6.1.0->ipywidgets>=7.6.0->lonboard) (3.0.0)
    Requirement already satisfied: pure-eval in /opt/miniconda3/lib/python3.12/site-packages (from stack_data->ipython>=6.1.0->ipywidgets>=7.6.0->lonboard) (0.2.3)
    Note: you may need to restart the kernel to use updated packages.



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
    "s3://overturemaps-us-west-2/release/2025-11-19.0/theme=buildings/type=building/"
)
```


```python
df.limit(10).show()
```

    ┌──────────────────────────────────────┬─────────────────────────────────────────┬───┬─────────────┐
    │                  id                  ┆                 geometry                ┆ … ┆ roof_height │
    │                 utf8                 ┆                 geometry                ┆   ┆   float64   │
    ╞══════════════════════════════════════╪═════════════════════════════════════════╪═══╪═════════════╡
    │ 85b47da4-1b8d-4132-ac6c-d8dc14fab4b8 ┆ POLYGON((-6.4292972 54.8290034,-6.4291… ┆ … ┆             │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ ec12e345-d44d-4e40-8e08-e1e6e68d4d17 ┆ POLYGON((-6.430836 54.8299412,-6.43095… ┆ … ┆             │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 285f9ff9-2d6d-409c-b214-74992c8d7e7d ┆ POLYGON((-6.4311579 54.8300247,-6.4313… ┆ … ┆             │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ abedfc7c-e5fd-4a29-931e-da77b610d02d ┆ POLYGON((-6.4321833 54.8294427,-6.4322… ┆ … ┆             │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ a203a2c6-e130-4979-a7d5-8a059c6f31fd ┆ POLYGON((-6.4300627 54.829276,-6.43006… ┆ … ┆             │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 1d14caf6-b12d-486e-87dd-feef82fba9a7 ┆ POLYGON((-6.4301786 54.8281533,-6.4299… ┆ … ┆             │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 4b1e67cf-7355-439b-9a31-46a50f3ee227 ┆ POLYGON((-6.4298614 54.8278977,-6.4299… ┆ … ┆             │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 06de994e-efd4-4a1c-8a20-b4e883904cb2 ┆ POLYGON((-6.4296383 54.827599,-6.42956… ┆ … ┆             │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ ea0b2ea6-7c52-4395-9baa-bc023c7d3166 ┆ POLYGON((-6.4296844 54.8277379,-6.4296… ┆ … ┆             │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 49f022ef-5574-4613-ae54-af139666fde3 ┆ POLYGON((-6.4296843 54.8278169,-6.4296… ┆ … ┆             │
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
    │ 2541497985 │
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
      sources: list<List(Field { name: "element", data_type: Struct([Field { name: "property", data_type: Utf8, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }, Field { name: "dataset", data_type: Utf8, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }, Field { name: "license", data_type: Utf8, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }, Field { name: "record_id", data_type: Utf8, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }, Field { name: "update_time", data_type: Utf8, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }, Field { name: "confidence", data_type: Float64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }, Field { name: "between", data_type: List(Field { name: "element", data_type: Float64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }]), nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} })>
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
        ST_GeomFromText('{nyc_bbox_wkt}', 4326)
    )
LIMIT 5;
""").show()
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


## Overture divisions table


```python
df = sd.read_parquet(
    "s3://overturemaps-us-west-2/release/2025-11-19.0/theme=divisions/type=division_area/"
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
    │ 3665c36d-d3a9-… ┆ POLYGON((12.5… ┆ {xmin: 12.455… ┆ … ┆ true           ┆ IT-34  ┆ f05aa29f-151f… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 18a69439-a1da-… ┆ POLYGON((12.5… ┆ {xmin: 12.596… ┆ … ┆ true           ┆ IT-36  ┆ ae00d58c-6e67… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 7d0f6d37-bb55-… ┆ POLYGON((12.6… ┆ {xmin: 12.567… ┆ … ┆ true           ┆ IT-36  ┆ bdfc82ca-5f23… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 3f480ff6-6361-… ┆ POLYGON((12.5… ┆ {xmin: 12.549… ┆ … ┆ true           ┆ IT-36  ┆ 1c750104-4470… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 31c3ab5e-eb6f-… ┆ POLYGON((12.6… ┆ {xmin: 12.612… ┆ … ┆ true           ┆ IT-34  ┆ d90804ee-19a4… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 308517e6-64b4-… ┆ POLYGON((12.5… ┆ {xmin: 12.589… ┆ … ┆ true           ┆ IT-34  ┆ aabd71e9-4d98… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 646e5b1f-b76a-… ┆ POLYGON((12.5… ┆ {xmin: 12.485… ┆ … ┆ true           ┆ IT-34  ┆ 502c1c4e-fc19… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ f2809a49-1082-… ┆ POLYGON((12.5… ┆ {xmin: 12.538… ┆ … ┆ true           ┆ IT-34  ┆ 8b446eed-00ad… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 72b27245-c7fd-… ┆ POLYGON((12.5… ┆ {xmin: 12.501… ┆ … ┆ true           ┆ IT-34  ┆ 1d535e1f-d19e… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 815855d9-05d0-… ┆ POLYGON((12.4… ┆ {xmin: 12.371… ┆ … ┆ true           ┆ IT-34  ┆ 5aa91354-9e8c… │
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
    │  1052542 │
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
      sources: list<List(Field { name: "element", data_type: Struct([Field { name: "property", data_type: Utf8, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }, Field { name: "dataset", data_type: Utf8, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }, Field { name: "license", data_type: Utf8, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }, Field { name: "record_id", data_type: Utf8, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }, Field { name: "update_time", data_type: Utf8, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }, Field { name: "confidence", data_type: Float64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }, Field { name: "between", data_type: List(Field { name: "element", data_type: Float64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }]), nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} })>
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
        ST_GeomFromText('{nova_scotia_bbox_wkt}', 4326)
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
    │ Apple River            ┆                        ┆                        ┆ POLYGON((-64.7260681… │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ Allen Hill             ┆                        ┆                        ┆ POLYGON((-64.6956656… │
    └────────────────────────┴────────────────────────┴────────────────────────┴───────────────────────┘
    CPU times: user 1.25 ms, sys: 805 μs, total: 2.05 ms
    Wall time: 1.42 ms


## Visualize the results with lonboard


```python
import lonboard

lonboard.viz(df)
```




    Map(basemap_style=<CartoBasemap.DarkMatter: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json'…



![Lonboard NS](image/lonboard_ns.png)
