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

The following SQL functions are available for SedonaDB.

You can query data directly from files and URLs by treating them like database tables. This feature supports formats like **Parquet**, **CSV**, and **JSON**.

To query a file, place its path or URL in single quotes within the `FROM` clause.

```python
# Query a remote Parquet file directly
"SELECT * FROM 'https://raw.githubusercontent.com/geoarrow/geoarrow-data/v0.2.0/natural-earth/files/natural-earth_cities_geo.parquet'").show()
```

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

## ST_Area

### Description

Return the area of a geometry.

### Format

`ST_Area (A: Geometry)`

### Arguments

  * **geom**: Input geometry.

### SQL Example

```sql
SELECT ST_Area(ST_GeomFromWKT('POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))'));
```

## ST_AsBinary

### Description

Return the Well-Known Binary representation of a geometry or geography. This function also has the alias **ST_AsWKB**.

### Format

`ST_AsBinary (A: Geometry)`

### Arguments

  * **geom**: Input geometry or geography.

### SQL Example

```sql
SELECT ST_AsBinary(ST_Point(1.0, 2.0));
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

## ST_Buffer

### Description

Returns a geometry that represents all points whose distance from the input geometry is less than or equal to a specified distance.

### Format

`ST_Buffer (A: Geometry, distance: Double)`

### Arguments

  * **geom**: Input geometry.
  * **distance**: Radius of the buffer.

### SQL Example

```sql
SELECT ST_Buffer(ST_GeomFromText('POLYGON ((10 10, 11 10, 10 11, 10 10))'), 1.0);
```

## ST_Centroid

### Description

Returns the centroid of geom.

### Format

`ST_Centroid (A: Geometry)`

### Arguments

  * **geom**: Input geometry.

### SQL Example

```sql
SELECT ST_AsText(ST_Centroid(ST_GeomFromWKT('POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))')));
```

## ST_Collect

### Description

Aggregates a set of geometries into a single `GeometryCollection`, `MultiPoint`, `MultiLineString`, or `MultiPolygon`. If all input geometries are of the same type (e.g., all points), it creates a multi-geometry of that type. If the geometries are of mixed types, it returns a `GeometryCollection`.

### Format

`ST_Collect (geom: Geometry)`

### Arguments

  * **geom**: The input geometry or geography to be collected.

### SQL Example

```sql
SELECT ST_Collect(ST_GeomFromWKT('MULTIPOINT (0 1, 10 11)'))
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

## ST_Dimension

### Description

Return the dimension of the geometry.

### Format

`ST_Dimension (A: Geometry)`

### Arguments

  * **geom**: Input geometry.

### SQL Example

```sql
SELECT ST_Dimension(ST_GeomFromWKT('POLYGON ((0 0, 1 0, 0 1, 0 0))'));
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

## ST_DWithin

### Description

Returns true if two geometries are within a specified distance of each other.

### Format

`ST_DWithin (A: Geometry, B: Geometry, distance: Double)`

### Arguments

  * **geomA**: Input geometry or geography.
  * **geomB**: Input geometry or geography.
  * **distance**: Distance in units of the geometry's coordinate system.

### SQL Example

```sql
SELECT ST_DWithin(ST_Point(0.25, 0.25), ST_GeomFromText('POLYGON ((0 0, 1 0, 0 1, 0 0))'), 0.5);
```

## ST_Envelope

### Description

Returns the bounding box (envelope) of a geometry as a new geometry. The resulting geometry represents the minimum bounding rectangle that encloses the input geometry. Depending on the input, the output can be a `Point`, `LineString`, or `Polygon`.

### Format

`ST_Envelope (A: Geometry)`

### Arguments

  * **geom**: The input geometry.

### SQL Example

```sql
SELECT ST_Envelope(ST_Point(1.0, 2.0))
```

## ST_Envelope_Aggr

### Description

An aggregate function that returns the collective bounding box (envelope) of a set of geometries.

### Format

`ST_Envelope_Aggr (geom: Geometry)`

### Arguments

  * **geom**: A column of geometries to be aggregated.

### SQL Example

```sql
-- Create a table with geometries and calculate the aggregate envelope
WITH shapes(geom) AS (
    VALUES (ST_GeomFromWKT('POINT (0 1)')),
           (ST_GeomFromWKT('POINT (10 11)'))
)
SELECT ST_AsText(ST_Envelope_Aggr(geom)) FROM shapes;
-- Returns: POLYGON ((0 1, 0 11, 10 11, 10 1, 0 1))
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

## ST_FlipCoordinates

### Description

Returns a new geometry with the X and Y coordinates of each vertex swapped. This is useful for correcting geometries that have been created with longitude and latitude in the wrong order.

### Format

`ST_FlipCoordinates (A: geometry)`

### Arguments

  * **geom**: The input geometry whose coordinates will be flipped.

### SQL Example

```sql
SELECT ST_FlipCoordinates(df.geometry)
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

## ST_GeomFromWKB

### Description

Construct a Geometry from WKB.

### Format

`ST_GeomFromWKB (Wkb: Binary)`

### Arguments

  * **WKB**: binary: Well-known binary representation of the geometry.

### SQL Example

```sql
-- Creates a POINT(1 2) geometry from its WKB representation
SELECT ST_AsText(ST_GeomFromWKB(FROM_HEX('0101000000000000000000F03F0000000000000040')));
```

## ST_GeomFromWKT

### Description

Construct a Geometry from WKT. This function also has the alias **ST_GeomFromText**.

### Format

`ST_GeomFromWKT (Wkt: String)`

### Arguments

  * **WKT**: string: Well-known text representation of the geometry.

### SQL Example

```sql
SELECT ST_AsText(ST_GeomFromWKT('POINT (30 10)'));
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

## ST_Intersection_Aggr

### Description

An aggregate function that returns the geometric intersection of all geometries in a set.

### Format

`ST_Intersection_Aggr (geom: Geometry)`

### Arguments

  * **geom**: A column of geometries to be aggregated.

### SQL Example

```sql
-- Create a table with overlapping polygons and find their common intersection
WITH shapes(geom) AS (
    VALUES (ST_GeomFromWKT('POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))')),
           (ST_GeomFromWKT('POLYGON((1 1, 3 1, 3 3, 1 3, 1 1))'))
)
SELECT ST_AsText(ST_Intersection_Aggr(geom)) FROM shapes;
-- Returns: POLYGON ((1 1, 1 2, 2 2, 2 1, 1 1))
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

## ST_IsEmpty

### Description

Return true if the geometry is empty.

### Format

`ST_IsEmpty (A: Geometry)`

### Arguments

  * **geom**: Input geometry.

### SQL Example

```sql
SELECT ST_IsEmpty(ST_GeomFromWKT('POLYGON EMPTY'));
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

## ST_Length

### Description

Returns the length of geom. This function only supports LineString, MultiLineString, and GeometryCollections containing linear geometries. Use ST_Perimeter for polygons.

### Format

`ST_Length (A: Geometry)`

### Arguments

  * **geom**: geometry: Input geometry.

### SQL Example

```sql
SELECT ST_Length(ST_GeomFromWKT('LINESTRING(0 0, 10 0)'));
```

## ST_M

### Description

Returns the M (measure) coordinate of a `Point` geometry. If the geometry does not have an M value, it returns `NULL`.

### Format

`ST_M (A: Point)`

### Arguments

  * **geom**: The input point geometry or geography.

### SQL Example

```sql
SELECT ST_M(ST_Point(1.0, 2.0))
```

## ST_MakeLine

### Description

Creates a `LineString` from two or more input `Point`, `MultiPoint`, or `LineString` geometries. The function connects the input geometries in the order they are provided to form a single continuous line.

### Format

`ST_MakeLine (g1: Geometry or Geography, g2: Geometry or Geography)`

### Arguments

  * **g1**: The first `Point`, `MultiPoint`, or `LineString` geometry or geography.
  * **g2**: The second `Point`, `MultiPoint`, or `LineString` geometry or geography.

### SQL Example

```sql
SELECT ST_MakeLine(ST_Point(0, 1), ST_Point(2, 3)) as geom
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

## ST_MMax

### Description

Returns the maximum M (measure) value from a geometry's bounding box.

### Format

`ST_MMax (A: Geometry)`

### Arguments

  * **geom**: The input geometry.

### SQL Example

```sql
SELECT ST_MMax(ST_GeomFromWKT('POLYGON ((0 0, 1 0, 0 1, 0 0))'))
```

## ST_MMin

### Description

Returns the minimum **M-coordinate** (measure) of a geometry's bounding box.

### Format

`ST_MMin (A: Geometry)`

### Arguments

  * **geom**: Input geometry.

### SQL Example

```sql
SELECT ST_MMin(ST_GeomFromWKT('LINESTRING ZM (1 2 3 4, 5 6 7 8)'));
-- Returns: 4
```

## ST_Perimeter

### Description

This function calculates the 2D perimeter of a given geometry. It supports Polygon, MultiPolygon, and GeometryCollection geometries (as long as the GeometryCollection contains polygonal geometries). For other types, it returns 0. To measure lines, use ST_Length.

To get the perimeter in meters, set **use_spheroid** to true. This calculates the geodesic perimeter using the WGS84 spheroid. When using use_spheroid, the **lenient** parameter defaults to true, assuming the geometry uses EPSG:4326. To throw an exception instead, set lenient to false.

### Format

`ST_Perimeter(geom: Geometry)`
`ST_Perimeter(geom: Geometry, use_spheroid: Boolean)`
`ST_Perimeter(geom: Geometry, use_spheroid: Boolean, lenient: Boolean = True)`

### Arguments

  * **geom**: Input geometry.
  * **use_spheroid**: If true, calculates the geodesic perimeter using the WGS84 spheroid. Defaults to false.
  * **lenient**: If true, assumes the geometry uses EPSG:4326 when use_spheroid is true. Defaults to true.

### SQL Example

```sql
SELECT ST_Perimeter(ST_GeomFromWKT('POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))'));
```

## ST_Point

### Description

Construct a Point Geometry from X and Y.

### Format

`ST_Point (x: Double, y: Double)`

### Arguments

  * **x**: X value.
  * **y**: Y value.

### SQL Example

```sql
SELECT ST_AsText(ST_Point(-74.0060, 40.7128));
```

## ST_PointM

### Description

Constructs a `Point` with an M (measure) coordinate from X, Y, and M values.

### Format

`ST_PointM (x: Double, y: Double, m: Double)`

### Arguments

  * **x**: The X-coordinate value.
  * **y**: The Y-coordinate value.
  * **m**: The M-coordinate (measure) value.

### SQL Example

```sql
SELECT ST_PointM(-64.36, 45.09, 50.0)
```

## ST_PointZ

### Description

Constructs a `Point` with a Z (elevation) coordinate from X, Y, and Z values.

### Format

`ST_PointZ (x: Double, y: Double, z: Double)`

### Arguments

  * **x**: The X-coordinate value.
  * **y**: The Y-coordinate value.
  * **z**: The Z-coordinate (elevation) value.

### SQL Example

```sql
SELECT ST_PointZ(-64.36, 45.09, 100.0)
```

## ST_PointZM

### Description

Constructs a `Point` with both Z (elevation) and M (measure) coordinates from X, Y, Z, and M values.

### Format

`ST_PointZM (x: Double, y: Double, z: Double, m: Double)`

### Arguments

  * **x**: The X-coordinate value.
  * **y**: The Y-coordinate value.
  * **z**: The Z-coordinate (elevation) value.
  * **m**: The M-coordinate (measure) value.

### SQL Example

```sql
SELECT ST_PointZM(-64.36, 45.09, 100.0, 50.0)
```

## ST_SetSRID

### Description

Sets the spatial reference system identifier (SRID) of a geometry. This only changes the metadata; it does not transform the coordinates.

### Format

`ST_SetSRID (geom: Geometry, srid: Integer)`

### Arguments

  * **geom**: Input geometry or geography.
  * **srid**: EPSG code to set (e.g., 4326).

### SQL Example

```sql
SELECT ST_SetSRID(ST_GeomFromWKT('POINT (-64.363049 45.091501)'), 4326);
```

## ST_SRID

### Description

Returns the Spatial Reference System Identifier (SRID) of a geometry. If the geometry does not have an SRID, it returns 0.

### Format

`ST_SRID (geom: Geometry)`

### Arguments

  * **geom**: The input geometry or geography.

### SQL Example

```sql
SELECT ST_SRID(polygon)
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

## ST_Transform

### Description

Transforms the coordinates of a geometry from a source Coordinate Reference System (CRS) to a target CRS.

If the source CRS is not specified, it will be read from the geometry's metadata. Sedona ensures that coordinates are handled in longitude/latitude order for geographic CRS transformations.

### Format

`ST_Transform (A: Geometry, TargetCRS: String)`
`ST_Transform (A: Geometry, SourceCRS: String, TargetCRS: String)`

### Arguments

  * **geom**: Input geometry or geography.
  * **source_crs**: The source CRS code (e.g., 'EPSG:4326').
  * **target_crs**: The target CRS code to transform into.
  * **lenient**: A boolean that, if true, assumes the source is EPSG:4326 if not specified. Defaults to true.

### SQL Example

```sql
-- Transform a WGS84 polygon to UTM zone 49N
SELECT ST_Transform(ST_SetSRID(ST_GeomFromWkt('POLYGON((170 50,170 72,-130 72,-130 50,170 50))'), 4326), 'EPSG:32649');
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

## ST_Union_Aggr

### Description

An aggregate function that returns the geometric union of all geometries in a set.

### Format

`ST_Union_Aggr (geom: Geometry)`

### Arguments

  * **geom**: A column of geometries to be aggregated.

### SQL Example

```sql
-- Create a table with two separate polygons and unite them into a single multipolygon
WITH shapes(geom) AS (
    VALUES (ST_GeomFromWKT('POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))')),
           (ST_GeomFromWKT('POLYGON ((2 2, 3 2, 3 3, 2 3, 2 2))'))
)
SELECT ST_AsText(ST_Union_Aggr(geom)) FROM shapes;
-- Returns: MULTIPOLYGON (((2 2, 3 2, 3 3, 2 3, 2 2)), ((0 0, 1 0, 1 1, 0 1, 0 0)))
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

## ST_XMax

### Description

Returns the maximum **X-coordinate** of a geometry's bounding box.

### Format

`ST_XMax (A: Geometry)`

### Arguments

  * **geom**: Input geometry.

### SQL Example

```sql
SELECT ST_XMax(ST_GeomFromWKT('LINESTRING(1 5, 10 15)'));
-- Returns: 10
```

## ST_XMin

### Description

Returns the minimum **X-coordinate** of a geometry's bounding box.

### Format

`ST_XMin (A: Geometry)`

### Arguments

  * **geom**: Input geometry.

### SQL Example

```sql
SELECT ST_XMin(ST_GeomFromWKT('LINESTRING(1 5, 10 15)'));
-- Returns: 1
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

## ST_YMax

### Description

Returns the maximum **Y-coordinate** of a geometry's bounding box.

### Format

`ST_YMax (A: Geometry)`

### Arguments

  * **geom**: Input geometry.

### SQL Example

```sql
SELECT ST_YMax(ST_GeomFromWKT('LINESTRING(1 5, 10 15)'));
-- Returns: 15
```

## ST_YMin

### Description

Returns the minimum **Y-coordinate** of a geometry's bounding box.

### Format

`ST_YMin (A: Geometry)`

### Arguments

  * **geom**: Input geometry.

### SQL Example

```sql
SELECT ST_YMin(ST_GeomFromWKT('LINESTRING(1 5, 10 15)'));
-- Returns: 5
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

## ST_ZMax

### Description

Returns the maximum **Z-coordinate** of a geometry's bounding box.

### Format

`ST_ZMax (A: Geometry)`

### Arguments

  * **geom**: Input geometry.

### SQL Example

```sql
SELECT ST_ZMax(ST_GeomFromWKT('LINESTRING ZM (1 2 3 4, 5 6 7 8)'));
-- Returns: 7
```

## ST_ZMin

### Description

Returns the minimum **Z-coordinate** of a geometry's bounding box.

### Format

`ST_ZMin (A: Geometry)`

### Arguments

  * **geom**: Input geometry.

### SQL Example

```sql
SELECT ST_ZMin(ST_GeomFromWKT('LINESTRING ZM (1 2 3 4, 5 6 7 8)'));
-- Returns: 3
```
