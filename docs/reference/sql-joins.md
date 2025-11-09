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

You can perform spatial joins using standard SQL `INNER JOIN` syntax. The join condition is defined in the `ON` clause using a spatial function that specifies the relationship between the geometries of the two tables.

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

## Optimization Barrier

Use the `barrier` function to prevent filter pushdown and control predicate evaluation order in complex spatial joins. This function creates an optimization barrier by evaluating boolean expressions at runtime.

The `barrier` function takes a boolean expression as a string, followed by pairs of variable names and their values that will be substituted into the expression:

```sql
barrier(expression, var_name1, var_value1, var_name2, var_value2, ...)
```

The placement of filters relative to KNN joins changes the semantic meaning of the query:

- **Filter before KNN**: First filters the data, then finds K nearest neighbors from the filtered subset. This answers "What are the K nearest high-rated restaurants?"
- **Filter after KNN**: First finds K nearest neighbors from all data, then filters those results. This answers "Of the K nearest restaurants, which ones are high-rated?"

### Example

Find the 3 nearest restaurants for each luxury hotel, and then filter the results to only show pairs where the restaurant is also high-rated.

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
    barrier('rating > 4.0 AND stars >= 4',
            'rating', r.rating,
            'stars', h.stars)
```

With the barrier function, this query first finds the 3 nearest restaurants to each hotel (regardless of rating), then filters to keep only those pairs where the restaurant has rating > 4.0 and the hotel has stars >= 4. Without the barrier, an optimizer might push the filters down, changing the query to first filter for high-rated restaurants and luxury hotels, then find the 3 nearest among those filtered sets.
