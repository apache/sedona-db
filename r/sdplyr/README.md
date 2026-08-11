
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

# sdplyr

<!-- badges: start -->

[![R-universe
status](https://apache.r-universe.dev/badges/sdplyr)](https://apache.r-universe.dev/sdplyr)
<!-- badges: end -->

The goal of sdplyr is to provide a [dplyr](https://dplyr.tidyverse.org/)
interface to [Apache SedonaDB](https://sedona.apache.org/sedonadb). It
lets you use familiar dplyr verbs and the spatial functions documented
by [sedonafns](../sedonafns) with SedonaDB-backed lazy data frames.

## Installation

sdplyr can be installed from
[R-universe](https://apache.r-universe.dev/sdplyr):

``` r
install.packages(
  "sdplyr",
  repos = c("https://apache.r-universe.dev", "https://cloud.r-project.org")
)
```

You can install the development version of sdplyr from
[GitHub](https://github.com/) with:

``` r
pak::pkg_install("apache/sedona-db/r/sdplyr")
```

Installing a development version of sdplyr requires a [Rust
compiler](https://rustup.rs) and a GEOS system dependency (e.g.,
`brew install geos` or `apt-get install libgeos-dev`). Install
instructions for these dependencies on other platforms can be found on
the [sf package homepage](https://r-spatial.github.io/sf).

## Example

Read GeoParquet files into lazy SedonaDB data frames and use dplyr verbs
to perform a spatial join:

``` r
library(sdplyr)
#> ── Attaching sdplyr packages ────────────────────────────────────────────────────────────────────────────────────────── 0.4.0.9000 ──
#> ✔ sedonadb  0.4.0.9000
#> ✔ sedonafns 0.4.0.9000
#> ✔ dplyr     1.2.0

cities_url <- "https://raw.githubusercontent.com/geoarrow/geoarrow-data/v0.2.0/natural-earth/files/natural-earth_cities.parquet"
countries_url <- "https://raw.githubusercontent.com/geoarrow/geoarrow-data/v0.2.0/natural-earth/files/natural-earth_countries.parquet"

cities <- sd_read_parquet(cities_url)
countries <- sd_read_parquet(countries_url)

cities |>
  inner_join(
    countries,
    by = sd_join_intersects()
  ) |>
  filter(continent != "North America") |>
  select(
    city = name.x,
    country = name.y,
    continent
  ) |>
  arrange(country) |>
  head(10)
#> <sedonab_dataframe: NA x 3>
#> ┌──────────────┬─────────────┬───────────────┐
#> │     city     ┆   country   ┆   continent   │
#> │     utf8     ┆     utf8    ┆      utf8     │
#> ╞══════════════╪═════════════╪═══════════════╡
#> │ Kabul        ┆ Afghanistan ┆ Asia          │
#> ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
#> │ Tirana       ┆ Albania     ┆ Europe        │
#> ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
#> │ Algiers      ┆ Algeria     ┆ Africa        │
#> ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
#> │ Luanda       ┆ Angola      ┆ Africa        │
#> ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
#> │ Buenos Aires ┆ Argentina   ┆ South America │
#> ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
#> │ Yerevan      ┆ Armenia     ┆ Asia          │
#> └──────────────┴─────────────┴───────────────┘
#> Preview of up to 6 row(s)
```

Spatial predicates can also compare a geometry column with an R spatial
object. For example, this query finds cities inside a longitude-latitude
bounding box:

``` r
cities |>
  filter(
    sd_intersects(
      geometry,
      wk::rct(-80, 40, -60, 60, wk::wk_crs_longlat())
    )
  )
#> <sedonab_dataframe: NA x 2>
#> ┌──────────┬─────────────────────────────────────────────┐
#> │   name   ┆                   geometry                  │
#> │   utf8   ┆                   geometry                  │
#> ╞══════════╪═════════════════════════════════════════════╡
#> │ Ottawa   ┆ POINT(-75.7019612 45.4186427)               │
#> ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
#> │ Toronto  ┆ POINT(-79.38945855491194 43.66464454743429) │
#> ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
#> │ New York ┆ POINT(-73.99571754361698 40.72156174972766) │
#> └──────────┴─────────────────────────────────────────────┘
#> Preview of up to 6 row(s)
```
