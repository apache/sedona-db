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

# Using SedonaDB from Rust

This example shows how to use the `sedona` crate alongside `datafusion` in a Rust
Project to run a basic query.

```shell
git clone https://github.com/apache/sedona-db.git
cd sedona-db/examples/sedona-rust
cargo run
```

```
+-------------+----------------------------------------------+
|     name    |                   geometry                   |
+-------------+----------------------------------------------+
| Abidjan     | POINT(-4.020206835187587 5.3231260722445715) |
| Abu Dhabi   | POINT(54.3665934 24.4666836)                 |
| Abuja       | POINT(7.489505042885861 9.054620406360845)   |
| Accra       | POINT(-0.2186616 5.5519805)                  |
| Addis Ababa | POINT(38.6980586 9.0352562)                  |
+-------------+----------------------------------------------+
```
