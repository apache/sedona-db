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

# SedonaDB Python

This package is a Python interface to [SedonaDB](https://sedona.apache.org/sedonadb).

## Installation

SedonaDB for Python can be installed from [PyPI](https://pypi.org):

```shell
pip install "apache-sedona[db]"
```

## Example

```python
import sedona.db

sd = sedona.db.connect()
sd.options.interactive = True

url = "https://raw.githubusercontent.com/geoarrow/geoarrow-data/v0.2.0/natural-earth/files/natural-earth_cities_geo.parquet"
sd.read_parquet(url).head(5)
```
