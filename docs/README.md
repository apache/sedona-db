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

# SedonaDB Documentation

SedonaDB's documentation is powered by [mkdocs.org](https://www.mkdocs.org) and [mkdocs-material](https://squidfunk.github.io/mkdocs-material/reference/). To build the documentation locally, clone the repo, install the Python requirements, and use the `mkdocs` command-line tool to build or serve the documentation.

```shell
git clone https://github.com/apache/sedona-db.git && cd sedona-db

# OPTIONAL: build the doc for the latest dev version of sedona-db
pip install -e "python/sedonadb/[test]" -vv

pip install -r docs/requirements.txt
```

* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.
