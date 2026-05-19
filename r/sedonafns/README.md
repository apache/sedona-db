
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

# sedonafns

<!-- badges: start -->

<!-- badges: end -->

The goal of sedonafns is to provide function definitions and
documentation to enhance the user experience writing SedonaDB code in
practice.

## Installation

You can install the development version of sedonafns from
[GitHub](https://github.com/) with:

``` r
# install.packages("pak")
pak::pak("apache/sedona-db")
```

## Example

You can use `sd_xxx()` functions in SedonaDB pipelines and benefit from
autocomplete and inline documentation.

``` r
library(sedonadb)
library(sedonafns)

sd_read_sf(system.file("shape/nc.shp", package = "sf")) |>
  sd_transmute(
    NAME,
    area = sd_area(wkb_geometry)
  )
#> <sedonab_dataframe: NA x 2>
#> ┌─────────────┬─────────────────────┐
#> │     NAME    ┆         area        │
#> │     utf8    ┆       float64       │
#> ╞═════════════╪═════════════════════╡
#> │ Ashe        ┆ 0.11428350451751612 │
#> ├╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
#> │ Alleghany   ┆ 0.06139975567930378 │
#> ├╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
#> │ Surry       ┆  0.1430162843025755 │
#> ├╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
#> │ Currituck   ┆ 0.06977097556227818 │
#> ├╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
#> │ Northampton ┆ 0.15275930054485798 │
#> ├╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
#> │ Hertford    ┆ 0.09715755874640308 │
#> └─────────────┴─────────────────────┘
#> Preview of up to 6 row(s)
```
