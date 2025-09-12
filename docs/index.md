---
hide:
  - navigation

title: Introducing SedonaDB
---

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

SedonaDB is a high-performance, dependency-free geospatial compute engine designed for single-node processing, making it ideal for smaller datasets on local machines or cloud instances.

The initial `0.1` release supports a core set of vector operations, with comprehensive vector and raster computation capabilities planned for the near future.

## Run a query in SQL, Python, or Rust

SedonaDB offers a flexible query interface in SQL, Python, or Rust.

Engineered for speed, SedonaDB provides blazing-fast geospatial processing on a single machine. This makes it perfect for the rapid analysis of smaller datasets, whether you're working locally or on a cloud server. While the initial release focuses on core vector operations, a full suite of vector and raster computations is on the roadmap.

For massive, distributed workloads, you can leverage the power of SedonaSpark,
SedonaFlink, or SedonaSnow.

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

SedonaDB has several advantages:

* **Blazing-Fast Performance:** Built in Rust to process massive geospatial datasets with exceptional speed.
* **Unified Geospatial Toolkit:** Access a comprehensive suite of functions for both vector and raster data in a single, powerful library.
* **Seamless Ecosystem Integration:** Built on Apache Arrow for smooth interoperability with popular data science libraries like GeoPandas, DuckDB, and Polars.
* **Flexible APIs:** Effortlessly switch between Python and SQL interfaces to match your preferred workflow and skillset.
* **Guaranteed CRS Propagation:** Automatically manages coordinate reference systems (CRS) to ensure spatial accuracy and prevent common errors.
* **Broad File Format Support:** Work with a wide range of both modern and legacy geospatial file formats like geoparquet.
* **Highly Extensible:** Easily customize and extend the library's functionality to meet your project's unique requirements.

## Installation

Here's how to install SedonaDB with various build tools:

=== "pip"

	```bash
	pip install "apache-sedona[db]"
	```

=== "R"

	```bash
	install.packages("sedonadb", repos = "https://community.r-multiverse.org")
	```

## Have questions?

Start a [GitHub Discussion ](https://github.com/apache/sedona-db/issues)or join the [Discord community](https://discord.com/invite/9A3k5dEBsY) and ask the developers any questions you may have.

We look forward to collaborating with you!
