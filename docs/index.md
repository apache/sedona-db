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

# SedonaDB

SedonaDB is a high-performance, dependency-free geospatial compute engine.

=== "SQL"

	```sql
	SELECT ST_Point(0, 1) as geom
	```

=== "Python"

	```python
	import seonda.db

	sd = sedona.db.connect()
	sd.sql("SELECT ST_Point(0, 1) as geom")
	```

=== "Rust"

	```Rust
	use datafusion::prelude::*
	use sedona::context{SedonaContext, SedonaDataFrame};

	let ctx = SedonaContext::new_local_interactive().await?;
        let batches = ctx
            .sql("SELECT ST_Point(0, 1) as geom")
            .await?
            .show_sedona(&cts, None, Default::default())
            .await?;
	```

=== "R"

	```r
	library(sedonadb)

        sd_sql("SELECT ST_Point(0, 1) as geom")
	```

## Key features

* **Blazing fast**: SedonaDB runs on a single machine, optimized for geospatial workflows.
* SedonaDB is a **dependency-free**, **small binary** that is only XX KB.
* Supports **various file formats**, including GeoJSON, Shapefile, GeoParquet, CSV, and PostGIS.
* Exposes **several language APIs,** including SQL, Python, Rust, and R.
* **Portable**: Easy to run on the command line, locally or in the cloud with AWS Sagemaker, AWS Lambda, Azure Functions, Azure Machine Learning, or Google Colab.
* **Extensible**: You can extend SedonaDB to build your own geospatial compute engine custom for your needs.
* **Open source**: Apache Sedona is an open-source project managed according to the Apache Software Foundation's guidelines.

## Installation

Here’s how to install SedonaDB with various build tools:

=== "pip"

	```bash
	pip install "apache-sedona[db]"
	```

=== "Rust"

	```rust
	cargo add sedona
	```

=== "R"

	```bash
	install.packages("sedonadb", repos = "https://community.r-multiverse.org")
	```

## SedonaDB example with vector data

TODO

## Have questions?

Feel free to start a GitHub Discussion or join the Discord community to ask the developers any questions you may have.

We look forward to collaborating with you!
