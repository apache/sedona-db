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

# Spatial Joins

You can perform spatial joins using standard SQL `JOIN` syntax. The join condition is defined in the `ON` clause using a spatial function that specifies the relationship between the geometries of the two tables.

## General Spatial Join

Use functions like `ST_Contains`, `ST_Intersects`, or `ST_Within` to join tables based on their spatial relationship.

### Example

Assign a country to each city by checking which country polygon contains each city point.

```sql
SELECT
    cities.name as city,
    countries.name as country
FROM
    cities
INNER JOIN
    countries
    ON ST_Contains(countries.geometry, cities.geometry)
```

## K-Nearest Neighbor (KNN) Join

Use the specialized `ST_KNN` function to find the *k* nearest neighbors from one table for each geometry in another. This is useful for proximity analysis.

### Example

For each city, find the 5 other closest cities.

```sql
SELECT
    cities_l.name AS city,
    cities_r.name AS nearest_neighbor
FROM
    cities AS cities_l
INNER JOIN
    cities AS cities_r
    ON ST_KNN(cities_l.geometry, cities_r.geometry, 5, false)
```

### KNN Join Caveats

#### Filter Pushdown Behavior

In KNN joins, the **query side** is the first geometry argument to `ST_KNN`, and the **object side** is the second argument. For each query-side row, the join finds the *k* nearest object-side rows.

The optimizer automatically pushes **query-side** filters below the KNN join for inner joins. This is safe because filtering query rows before the join only reduces the number of probe points — each remaining query point still gets its full KNN search against all objects.

**Object-side** filters are never pushed below the KNN join automatically, because doing so would change which candidates are considered for the KNN search, altering the results. All object-side `WHERE` clause predicates are evaluated after the K nearest neighbor candidates have been selected.

For example, in the following query, `r.rating > 4.0` is applied *after* finding the 3 nearest restaurants for each hotel — it does not reduce the set of candidate restaurants before the KNN search:

```sql
SELECT
    h.name AS hotel,
    r.name AS restaurant,
    r.rating
FROM
    hotels AS h
INNER JOIN
    restaurants AS r
    ON ST_KNN(h.geometry, r.geometry, 3, false)
WHERE
    r.rating > 4.0
```

This means the result may contain fewer than 3 restaurants per hotel if some of the nearest neighbors do not pass the filter.

However, a query-side filter like `h.stars >= 4` is automatically pushed below the join:

```sql
SELECT h.name AS hotel, r.name AS restaurant, r.rating
FROM
    hotels AS h
INNER JOIN
    restaurants AS r
    ON ST_KNN(h.geometry, r.geometry, 3, false)
WHERE
    h.stars >= 4
```

The physical plan shows the filter *below* the join, inside the query-side input:

```
SpatialJoinExec: join_type=Inner, on=ST_KNN(geometry, geometry, 3, false)
  FilterExec: stars >= 4
    ...hotels...        ← only luxury hotels are scanned
  ...restaurants...
```

Only hotels with `stars >= 4` are used as query points, and the 3 nearest restaurants are found for each of those luxury hotels.

#### Pre-Filtering the Object Side

To filter the object side *before* the KNN search (e.g., only consider high-rated restaurants), use a subquery or CTE so the filter is applied before the join sees the data:

```sql
SELECT h.name AS hotel, r.name AS restaurant, r.rating
FROM
    hotels AS h
INNER JOIN
    (SELECT * FROM restaurants WHERE rating > 4.0) AS r
    ON ST_KNN(h.geometry, r.geometry, 3, false)
```

Or equivalently, using a CTE:

```sql
WITH high_rated AS (
    SELECT * FROM restaurants WHERE rating > 4.0
)
SELECT h.name AS hotel, r.name AS restaurant, r.rating
FROM
    hotels AS h
INNER JOIN
    high_rated AS r
    ON ST_KNN(h.geometry, r.geometry, 3, false)
```

This answers "What are the 3 nearest *high-rated* restaurants to each hotel?" because the KNN search only considers restaurants with rating > 4.0.

#### ST_KNN Predicate Precedence

When `ST_KNN` is combined with other predicates via `AND`, `ST_KNN` always takes precedence. It is extracted first to determine the KNN candidates, and the remaining predicates are applied as post-filters on the join output.

For example, the following two queries produce the same results:

```sql
-- ST_KNN in ON clause combined with another predicate via AND
SELECT h.name AS hotel, r.name AS restaurant
FROM hotels AS h
JOIN restaurants AS r
    ON ST_KNN(h.geometry, r.geometry, 3, false) AND r.rating > 4.0

-- Equivalent: ST_KNN in ON clause, other predicate in WHERE
SELECT h.name AS hotel, r.name AS restaurant
FROM hotels AS h
JOIN restaurants AS r
    ON r.rating > 4.0 AND ST_KNN(h.geometry, r.geometry, 3, false)
```

In both cases, `ST_KNN` determines the 3 nearest restaurants first, then `r.rating > 4.0` filters the results.
