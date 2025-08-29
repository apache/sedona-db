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

# SQL API Reference

SedonaDB SQL is a derivative of [DataFusion SQL](https://datafusion.apache.org/user-guide/sql/index.html)
with support for additional functions, data types, and file formats built in to SQL syntax.

See the [Apache Sedona SQL documentation](https://sedona.apache.org/latest/api/sql/Overview/) for
additional function documentation and examples.

Here are the markdown files for each SQL function found in the provided `.rs` files.

## ST_Analyze_Aggr

### Description

Return the statistics of geometries for the input geometry.

### Format

`ST_Analyze_Aggr (A: Geometry)`

### Arguments

  * **geom**: Input geometry or geography.

### SQL Example

```sql
SELECT ST_Analyze_Aggr(ST_GeomFromText('MULTIPOINT(1.1 101.1,2.1 102.1,3.1 103.1,4.1 104.1,5.1 105.1,6.1 106.1,7.1 107.1,8.1 108.1,9.1 109.1,10.1 110.1)'))
```

## ST_AsText

### Description

Return the Well-Known Text string representation of a geometry or geography.

### Format

`ST_AsText (A: Geometry)`

### Arguments

  * **geom**: Input geometry or geography.

### SQL Example

```sql
SELECT ST_AsText(ST_Point(1.0, 2.0))
```

## ST_Contains

### Description

Return true if geomA contains geomB.

### Format

`ST_Contains (A: Geometry, B: Geometry)`

### Arguments

  * **geomA**: Input geometry or geography.
  * **geomB**: Input geometry or geography.

### SQL Example

```sql
SELECT ST_Contains(ST_Point(0.25 0.25), ST_GeomFromText('POLYGON ((0 0, 1 0, 0 1, 0 0))')) AS val
```

## ST_CoveredBy

### Description

Return true if geomA is covered by geomB.

### Format

`ST_CoveredBy (A: Geometry, B: Geometry)`

### Arguments

  * **geomA**: Input geometry or geography.
  * **geomB**: Input geometry or geography.

### SQL Example

```sql
SELECT ST_CoveredBy(ST_Point(0.25 0.25), ST_GeomFromText('POLYGON ((0 0, 1 0, 0 1, 0 0))')) AS val
```

## ST_Covers

### Description

Return true if geomA covers geomB.

### Format

`ST_Covers (A: Geometry, B: Geometry)`

### Arguments

  * **geomA**: Input geometry or geography.
  * **geomB**: Input geometry or geography.

### SQL Example

```sql
SELECT ST_Covers(ST_Point(0.25 0.25), ST_GeomFromText('POLYGON ((0 0, 1 0, 0 1, 0 0))')) AS val
```

## ST_Disjoint

### Description

Return true if geomA is disjoint from geomB.

### Format

`ST_Disjoint (A: Geometry, B: Geometry)`

### Arguments

  * **geomA**: Input geometry or geography.
  * **geomB**: Input geometry or geography.

### SQL Example

```sql
SELECT ST_Disjoint(ST_Point(0.25 0.25), ST_GeomFromText('POLYGON ((0 0, 1 0, 0 1, 0 0))')) AS val
```

## ST_Equals

### Description

Return true if geomA equals geomB.

### Format

`ST_Equals (A: Geometry, B: Geometry)`

### Arguments

  * **geomA**: Input geometry or geography.
  * **geomB**: Input geometry or geography.

### SQL Example

```sql
SELECT ST_Equals(ST_Point(0.25 0.25), ST_GeomFromText('POLYGON ((0 0, 1 0, 0 1, 0 0))')) AS val
```

## ST_GeometryType

### Description

Return the type of a geometry.

### Format

`ST_GeometryType (A: Geometry)`

### Arguments

  * **geom**: Input geometry.

### SQL Example

```sql
SELECT ST_GeometryType(ST_GeomFromWKT('POLYGON ((0 0, 1 0, 0 1, 0 0))'))
```

## ST_HasM

### Description

Return true if the geometry has a M dimension.

### Format

`ST_HasM (A: Geometry)`

### Arguments

  * **geom**: Input geometry.

### SQL Example

```sql
SELECT ST_HasM(ST_GeomFromWKT('POLYGON ((0 0, 1 0, 0 1, 0 0))'))
```

## ST_HasZ

### Description

Return true if the geometry has a Z dimension.

### Format

`ST_HasZ (A: Geometry)`

### Arguments

  * **geom**: Input geometry.

### SQL Example

```sql
SELECT ST_HasZ(ST_GeomFromWKT('POLYGON ((0 0, 1 0, 0 1, 0 0))'))
```

## ST_Intersects

### Description

Return true if geomA intersects geomB.

### Format

`ST_Intersects (A: Geometry, B: Geometry)`

### Arguments

  * **geomA**: Input geometry or geography.
  * **geomB**: Input geometry or geography.

### SQL Example

```sql
SELECT ST_Intersects(ST_Point(0.25 0.25), ST_GeomFromText('POLYGON ((0 0, 1 0, 0 1, 0 0))')) AS val
```

## ST_KNN

### Description

Return true if geomA finds k nearest neighbors from geomB.

### Format

`ST_KNN (A: Geometry, B: Geometry, k: Integer, use_spheroid: Boolean)`

### Arguments

  * **geomA**: Query geometry or geography.
  * **geomB**: Object geometry or geography.
  * **k**: Number of nearest neighbors to find.
  * **use_spheroid**: Use spheroid distance calculation.

### SQL Example

```sql
SELECT * FROM table1 a JOIN table2 b ON ST_KNN(a.geom, b.geom, 5, false)
```

## ST_M

### Description

Return the M component of a point geometry or geography.

### Format

`ST_M(A: Point)`

### Arguments

  * **geom**: Input geometry or geography.

### SQL Example

```sql
SELECT ST_M(ST_Point(1.0, 2.0))
```

## ST_Touches

### Description

Return true if geomA touches geomB.

### Format

`ST_Touches (A: Geometry, B: Geometry)`

### Arguments

  * **geomA**: Input geometry or geography.
  * **geomB**: Input geometry or geography.

### SQL Example

```sql
SELECT ST_Touches(ST_Point(0.25 0.25), ST_GeomFromText('POLYGON ((0 0, 1 0, 0 1, 0 0))')) AS val
```

## ST_Within

### Description

Return true if geomA is fully contained by geomB.

### Format

`ST_Within (A: Geometry, B: Geometry)`

### Arguments

  * **geomA**: Input geometry or geography.
  * **geomB**: Input geometry or geography.

### SQL Example

```sql
SELECT ST_Within(ST_Point(0.25 0.25), ST_GeomFromText('POLYGON ((0 0, 1 0, 0 1, 0 0))')) AS val
```

## ST_X

### Description

Return the X component of a point geometry or geography.

### Format

`ST_X(A: Point)`

### Arguments

  * **geom**: Input geometry or geography.

### SQL Example

```sql
SELECT ST_X(ST_Point(1.0, 2.0))
```


## ST_Y

### Description

Return the Y component of a point geometry or geography.

### Format

`ST_Y(A: Point)`

### Arguments

  * **geom**: Input geometry or geography.

### SQL Example

```sql
SELECT ST_Y(ST_Point(1.0, 2.0))
```

## ST_Z

### Description

Return the Z component of a point geometry or geography.

### Format

`ST_Z(A: Point)`

### Arguments

  * **geom**: Input geometry or geography.

### SQL Example

```sql
SELECT ST_Z(ST_Point(1.0, 2.0))
```

## ST_Distance

### Description

Calculates the distance between geomA and geomB.

### Format

`ST_Distance (A: Geometry, B: Geometry)`

### Arguments

  * **geomA**: Input geometry or geography.
  * **geomB**: Input geometry or geography.

### SQL Example

```sql
SELECT ST_Distance(ST_GeomFromText('POLYGON ((10 10, 11 10, 10 11, 10 10))'), ST_GeomFromText('POLYGON ((0 0, 1 0, 0 1, 0 0))')) AS val
```

## ST_DistanceSphere

### Description

Calculates the spherical distance between geomA and geomB.

### Format

`ST_DistanceSphere (A: Geometry, B: Geometry)`

### Arguments

  * **geomA**: Input geometry or geography.
  * **geomB**: Input geometry or geography.

### SQL Example

```sql
SELECT ST_DistanceSphere(ST_GeomFromText('POLYGON ((10 10, 11 10, 10 11, 10 10))'), ST_GeomFromText('POLYGON ((0 0, 1 0, 0 1, 0 0))')) AS val
```

## ST_DistanceSpheroid

### Description

Calculates the spheroidal (ellipsoidal) distance between geomA and geomB.

### Format

`ST_DistanceSpheroid (A: Geometry, B: Geometry)`

### Arguments

  * **geomA**: Input geometry or geography.
  * **geomB**: Input geometry or geography.

### SQL Example

```sql
SELECT ST_DistanceSpheroid(ST_GeomFromText('POLYGON ((10 10, 11 10, 10 11, 10 10))'), ST_GeomFromText('POLYGON ((0 0, 1 0, 0 1, 0 0))')) AS val
```

## ST_MaxDistance

### Description

Calculates the maximum distance between geomA and geomB.

### Format

`ST_MaxDistance (A: Geometry, B: Geometry)`

### Arguments

  * **geomA**: Input geometry or geography.
  * **geomB**: Input geometry or geography.

### SQL Example

```sql
SELECT ST_MaxDistance(ST_GeomFromText('POLYGON ((10 10, 11 10, 10 11, 10 10))'), ST_GeomFromText('POLYGON ((0 0, 1 0, 0 1, 0 0))')) AS val
```

## ST_HausdorffDistance

### Description

Calculates the Hausdorff distance between geomA and geomB.

### Format

`ST_HausdorffDistance (A: Geometry, B: Geometry)`

### Arguments

  * **geomA**: Input geometry or geography.
  * **geomB**: Input geometry or geography.

### SQL Example

```sql
SELECT ST_HausdorffDistance(ST_GeomFromText('POLYGON ((10 10, 11 10, 10 11, 10 10))'), ST_GeomFromText('POLYGON ((0 0, 1 0, 0 1, 0 0))')) AS val
```

## ST_FrechetDistance

### Description

Calculates the Frechet distance between geomA and geomB.

### Format

`ST_FrechetDistance (A: Geometry, B: Geometry)`

### Arguments

  * **geomA**: Input geometry or geography.
  * **geomB**: Input geometry or geography.

### SQL Example

```sql
SELECT ST_FrechetDistance(ST_GeomFromText('POLYGON ((10 10, 11 10, 10 11, 10 10))'), ST_GeomFromText('POLYGON ((0 0, 1 0, 0 1, 0 0))')) AS val
```

## ST_Intersection

### Description

Computes the intersection between geomA and geomB.

### Format

`ST_Intersection (A: Geometry, B: Geometry)`

### Arguments

  * **geomA**: Input geometry or geography.
  * **geomB**: Input geometry or geography.

### SQL Example

```sql
SELECT ST_Intersection(ST_GeomFromText('POLYGON ((1 1, 11 1, 1 11, 0 0))'), ST_GeomFromText('POLYGON ((0 0, 10 0, 0 10, 0 0))')) AS val
```

## ST_Union

### Description

Computes the union between geomA and geomB.

### Format

`ST_Union (A: Geometry, B: Geometry)`

### Arguments

  * **geomA**: Input geometry or geography.
  * **geomB**: Input geometry or geography.

### SQL Example

```sql
SELECT ST_Union(ST_GeomFromText('POLYGON ((1 1, 11 1, 1 11, 0 0))'), ST_GeomFromText('POLYGON ((0 0, 10 0, 0 10, 0 0))')) AS val
```

## ST_Difference

### Description

Computes the difference between geomA and geomB.

### Format

`ST_Difference (A: Geometry, B: Geometry)`

### Arguments

  * **geomA**: Input geometry or geography.
  * **geomB**: Input geometry or geography.

### SQL Example

```sql
SELECT ST_Difference(ST_GeomFromText('POLYGON ((1 1, 11 1, 1 11, 0 0))'), ST_GeomFromText('POLYGON ((0 0, 10 0, 0 10, 0 0))')) AS val
```

## ST_SymDifference

### Description

Computes the symmetric difference between geomA and geomB.

### Format

`ST_SymDifference (A: Geometry, B: Geometry)`

### Arguments

  * **geomA**: Input geometry or geography.
  * **geomB**: Input geometry or geography.

### SQL Example

```sql
SELECT ST_SymDifference(ST_GeomFromText('POLYGON ((1 1, 11 1, 1 11, 0 0))'), ST_GeomFromText('POLYGON ((0 0, 10 0, 0 10, 0 0))')) AS val
```
