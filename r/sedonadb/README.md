
<!---
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at
&#10;    http://www.apache.org/licenses/LICENSE-2.0
&#10;  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

# sedonadb

<!-- badges: start -->

[![R-multiverse
status](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fcommunity.r-multiverse.org%2Fapi%2Fpackages%2Fsedonadb&query=%24.Version&label=r-multiverse)](https://community.r-multiverse.org/sedonadb)
<!-- badges: end -->

The goal of sedonadb is to provide an R interface to [Apache
SedonaDB](https://sedona.apache.org/sedonadb). SedonaDB provides a
[DataFusion](https://datafusion.apache.org)-powered single-node engine
with a wide range of spatial capabilities built in, including spatial
SQL with high function coverage and GeoParquet IO.

## Installation

sedonadb can be installed from
[R-multiverse](https://community.r-multiverse.org/):

``` r
install.packages("sedonadb", repos = "https://community.r-multiverse.org")
```

You can install the development version of sedonadb from
[GitHub](https://github.com/) with:

``` r
pak::pkg_install("apache/sedona-db/r/sedonadb")
```

Installing a development version of sedonadb requires a [Rust
compiler](https://rustup.rs) and a GEOS system dependency (e.g.,
`brew install geos` or `apt-get install libgeos-dev`). Install
instructions for these dependencies on other platforms can be found on
the [sf package homepage](https://r-spatial.github.io/sf).

## Example

You can use SedonaDB to read (Geo)Parquet files from anywhere. Filters
are used to reduce the amount of data downloaded based on GeoParquet
and/or Parquet metadata:

``` r
library(sedonadb)
#> Warning: package 'sedonadb' was built under R version 4.5.2

url <- "https://github.com/geoarrow/geoarrow-data/releases/download/v0.2.0/microsoft-buildings_point_geo.parquet"
sd_read_parquet(url) |> sd_to_view("buildings", overwrite = TRUE)

filter <- "POLYGON ((-73.4341 44.0087, -73.4341 43.7981, -73.2531 43.7981, -73.2531 43.8889, -73.1531 43.8889, -73.1531 44.0087, -73.4341 44.0087))"

sd_sql(glue::glue("
  SELECT * FROM buildings
  WHERE ST_Intersects(ST_SetSRID(ST_GeomFromText('{filter}'), 4326), geometry)
")) |> sd_preview()
#> ┌─────────────────────────────────┐
#> │             geometry            │
#> │             geometry            │
#> ╞═════════════════════════════════╡
#> │ POINT(-73.29533522 44.00847556) │
#> ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
#> │ POINT(-73.29092778 44.00421331) │
#> ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
#> │ POINT(-73.277808 43.998823)     │
#> ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
#> │ POINT(-73.277524 44.004619)     │
#> ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
#> │ POINT(-73.2774573 44.0044719)   │
#> ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
#> │ POINT(-73.27775838 44.0046742)  │
#> └─────────────────────────────────┘
#> Preview of up to 6 row(s)
```

Conversion to and from sf are supported:

``` r
library(sf)
#> Linking to GEOS 3.12.1, GDAL 3.8.4, PROJ 9.4.0; sf_use_s2() is TRUE

nc <- sf::read_sf(system.file("shape/nc.shp", package = "sf"))
nc |> sd_to_view("nc", overwrite = TRUE)

sd_sql("SELECT * FROM nc")
#> ┌─────────┬───────────┬─────────┬───┬─────────┬─────────┬────────────────────────────────────────┐
#> │   AREA  ┆ PERIMETER ┆  CNTY_  ┆ … ┆  SID79  ┆ NWBIR79 ┆                geometry                │
#> │ float64 ┆  float64  ┆ float64 ┆   ┆ float64 ┆ float64 ┆                geometry                │
#> ╞═════════╪═══════════╪═════════╪═══╪═════════╪═════════╪════════════════════════════════════════╡
#> │   0.114 ┆     1.442 ┆  1825.0 ┆ … ┆     0.0 ┆    19.0 ┆ MULTIPOLYGON(((-81.4727554321289 36.2… │
#> ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
#> │   0.061 ┆     1.231 ┆  1827.0 ┆ … ┆     3.0 ┆    12.0 ┆ MULTIPOLYGON(((-81.2398910522461 36.3… │
#> ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
#> │   0.143 ┆      1.63 ┆  1828.0 ┆ … ┆     6.0 ┆   260.0 ┆ MULTIPOLYGON(((-80.45634460449219 36.… │
#> ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
#> │    0.07 ┆     2.968 ┆  1831.0 ┆ … ┆     2.0 ┆   145.0 ┆ MULTIPOLYGON(((-76.00897216796875 36.… │
#> ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
#> │   0.153 ┆     2.206 ┆  1832.0 ┆ … ┆     3.0 ┆  1197.0 ┆ MULTIPOLYGON(((-77.21766662597656 36.… │
#> ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
#> │   0.097 ┆      1.67 ┆  1833.0 ┆ … ┆     5.0 ┆  1237.0 ┆ MULTIPOLYGON(((-76.74506378173828 36.… │
#> └─────────┴───────────┴─────────┴───┴─────────┴─────────┴────────────────────────────────────────┘
#> Preview of up to 6 row(s)
```

``` r
sd_sql("SELECT ST_Buffer(geometry, 0.1) as geometry FROM nc") |>
  st_as_sf() |>
  plot()
```

<img src="man/figures/README-unnamed-chunk-3-1.png" width="100%" />
