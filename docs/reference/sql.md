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

## ST_Analyze_Agg

#### Description

Return the statistics of geometries for the input geometry.

Since: v0.1.

#### Format

`ST_Analyze_Agg (A: Geometry)`

#### Arguments

  * **geom**: Input geometry or geography.

#### SQL Example

```sql
SELECT ST_Analyze_Agg(ST_GeomFromText('MULTIPOINT(1.1 101.1,2.1 102.1,3.1 103.1,4.1 104.1,5.1 105.1,6.1 106.1,7.1 107.1,8.1 108.1,9.1 109.1,10.1 110.1)'));
```

## ST_Area

#### Description

Return the area of a geometry.

Since: v0.1.

#### Format

`ST_Area (A: Geometry)`

#### Arguments

  * **geom**: Input geometry.

#### SQL Example

```sql
SELECT ST_Area(ST_GeomFromWKT('POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))'));
```

## ST_AsBinary

#### Description

Return the Well-Known Binary representation of a geometry or geography. This function also has the alias **ST_AsWKB**.

Since: v0.1.

#### Format

`ST_AsBinary (A: Geometry)`

#### Arguments

  * **geom**: Input geometry or geography.

#### SQL Example

```sql
SELECT ST_AsBinary(ST_Point(1.0, 2.0));
```

## ST_AsText

#### Description

Return the Well-Known Text string representation of a geometry or geography.

Since: v0.1.

#### Format

`ST_AsText (A: Geometry)`

#### Arguments

  * **geom**: Input geometry or geography.

#### SQL Example

```sql
SELECT ST_AsText(ST_Point(1.0, 2.0));
```

## ST_Azimuth

Introduction: Returns Azimuth for two given points in radians null otherwise.

Format: ST_Azimuth(pointA: Point, pointB: Point)

Since: v0.2.

```sql
SELECT ST_Azimuth(ST_POINT(0.0, 25.0), ST_POINT(0.0, 0.0));
```

## ST_Boundary

Returns the closure of the combinatorial boundary of this Geometry.

Format: `ST_Boundary(geom: Geometry)`

Since: v0.2.

```sql
SELECT ST_Boundary(ST_GeomFromWKT('POLYGON((1 1,0 0, -1 1, 1 1))'));
```

## ST_Buffer

#### Description

Returns a geometry that represents all points whose distance from the input geometry is less than or equal to a specified distance.

Since: v0.1.

Changed in version v0.2: Support buffer parameters argument

#### Format

`ST_Buffer (A: Geometry, distance: Double)`

#### Arguments

  * **geom**: Input geometry.
  * **distance**: Radius of the buffer.

#### SQL Example

```sql
SELECT ST_Buffer(ST_GeomFromText('POLYGON ((10 10, 11 10, 10 11, 10 10))'), 1.0);
```

## ST_Centroid

#### Description

Returns the centroid of geom.

Since: v0.1.

#### Format

`ST_Centroid (A: Geometry)`

#### Arguments

  * **geom**: Input geometry.

#### SQL Example

```sql
SELECT ST_AsText(ST_Centroid(ST_GeomFromWKT('POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))')));
```

## ST_ClosestPoint

Returns the 2-dimensional point on geom1 that is closest to geom2. This is the first point of the shortest line between the geometries. If using 3D geometries, the Z coordinates will be ignored. If you have a 3D Geometry, you may prefer to use ST_3DClosestPoint. It will throw an exception indicates illegal argument if one of the params is an empty geometry.

Format: `ST_ClosestPoint(g1: Geometry, g2: Geometry)`

Since: v0.2.

```sql
SELECT ST_AsText(
  ST_ClosestPoint(
    ST_GeogFromText('POINT(-118.4 34.0)'),  -- Santa Monica
    ST_GeogFromText('LINESTRING(-118.5 34.1, -118.3 33.9, -118.2 33.8)')  -- LA coastline
  )
) As ptwkt;
```

## ST_Collect_Agg

ST_Collect_Agg is an aggregate function that combines multiple geometries from a set of rows into a single collection.

```sql
SELECT ST_Collect_Agg(geom) as collected_points
FROM (
    SELECT ST_Point(-122.4194, 37.7749) as geom  -- San Francisco
    UNION ALL
    SELECT ST_Point(-118.2437, 34.0522)           -- Los Angeles
    UNION ALL
    SELECT ST_Point(-122.6765, 45.5231)           -- Portland
) as cities;
```

## ST_Contains

#### Description

Return true if geomA contains geomB.

Since: v0.1.

#### Format

`ST_Contains (A: Geometry, B: Geometry)`

#### Arguments

  * **geomA**: Input geometry or geography.
  * **geomB**: Input geometry or geography.

#### SQL Example

```sql
SELECT ST_Contains(
    ST_Point(0.25, 0.25),
    ST_GeomFromText('POLYGON ((0 0, 1 0, 0 1, 0 0))')
) AS val;
```

## ST_ConvexHull

Return the Convex Hull of polygon A

Format: `ST_ConvexHull (A: Geometry)`

Since: v0.2.

```sql
SELECT ST_ConvexHull(ST_GeomFromText('POLYGON((175 150, 20 40, 50 60, 125 100, 175 150))'));
```

## ST_CoveredBy

#### Description

Return true if geomA is covered by geomB.

Since: v0.1.

#### Format

`ST_CoveredBy (A: Geometry, B: Geometry)`

#### Arguments

  * **geomA**: Input geometry or geography.
  * **geomB**: Input geometry or geography.

#### SQL Example

```sql
SELECT ST_CoveredBy(ST_Point(0.25, 0.25), ST_GeomFromText('POLYGON ((0 0, 1 0, 0 1, 0 0))'));
```

## ST_Covers

#### Description

Return true if geomA covers geomB.

Since: v0.1.

#### Format

`ST_Covers (A: Geometry, B: Geometry)`

#### Arguments

  * **geomA**: Input geometry or geography.
  * **geomB**: Input geometry or geography.

#### SQL Example

```sql
SELECT ST_Covers(ST_Point(0.25, 0.25), ST_GeomFromText('POLYGON ((0 0, 1 0, 0 1, 0 0))')) AS val;
```

## ST_Crosses

Introduction: Return true if A crosses B

Format: `ST_Crosses (A: Geometry, B: Geometry)`

Since: v0.2.

```sql
SELECT ST_Crosses(ST_GeomFromWKT('POLYGON((1 1, 4 1, 4 4, 1 4, 1 1))'),ST_GeomFromWKT('POLYGON((2 2, 5 2, 5 5, 2 5, 2 2))'));
```

## ST_CRS

ST_CRS returns the Coordinate Reference System (CRS) metadata associated with a geometry or geography object.

```sql
SELECT ST_CRS(ST_Point(0.25, 0.25, 4326)) as crs_info;
```

## ST_Difference

#### Description

Computes the difference between geomA and geomB.

Since: v0.1.

#### Format

`ST_Difference (A: Geometry, B: Geometry)`

#### Arguments

  * **geomA**: Input geometry or geography.
  * **geomB**: Input geometry or geography.

#### SQL Example

```sql
sd.sql("""
SELECT ST_Difference(
    ST_GeomFromText('POLYGON ((1 1, 11 1, 11 11, 1 11, 1 1))'),
    ST_GeomFromText('POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))')
) AS val;
""").show()
```

## ST_Dimension

#### Description

Return the dimension of the geometry.

Since: v0.1.

#### Format

`ST_Dimension (A: Geometry)`

#### Arguments

  * **geom**: Input geometry.

#### SQL Example

```sql
SELECT ST_Dimension(ST_GeomFromWKT('POLYGON ((0 0, 1 0, 0 1, 0 0))'));
```

## ST_Disjoint

#### Description

Return true if geomA is disjoint from geomB.

Since: v0.1.

#### Format

`ST_Disjoint (A: Geometry, B: Geometry)`

#### Arguments

  * **geomA**: Input geometry or geography.
  * **geomB**: Input geometry or geography.

#### SQL Example

```sql
SELECT ST_Disjoint(ST_Point(0.25, 0.25), ST_GeomFromText('POLYGON ((0 0, 1 0, 0 1, 0 0))')) AS val;
```

## ST_Dump

It expands the geometries. If the geometry is simple (Point, Polygon Linestring etc.) it returns the geometry itself, if the geometry is collection or multi it returns record for each of collection components.

Format: ST_Dump(geom: Geometry)

Since: v0.2.

```sql
SELECT ST_Dump(ST_GeomFromText('MULTIPOINT ((10 40), (40 30), (20 20), (30 10))'));
```

## ST_DWithin

#### Description

Returns true if two geometries are within a specified distance of each other.

Since: v0.1.

#### Format

`ST_DWithin (A: Geometry, B: Geometry, distance: Double)`

#### Arguments

  * **geomA**: Input geometry or geography.
  * **geomB**: Input geometry or geography.
  * **distance**: Distance in units of the geometry's coordinate system.

#### SQL Example

```sql
SELECT ST_DWithin(
  ST_Point(0.25, 0.25),
  ST_GeomFromText('POLYGON ((0 0, 1 0, 0 1, 0 0))'),
  0.5
);
```

## ST_EndPoint

Returns last point of given linestring.

Format: `ST_EndPoint(geom: Geometry)`

Since: v0.2.

```sql
SELECT ST_EndPoint(ST_GeomFromText('LINESTRING(100 150,50 60, 70 80, 160 170)'));
```

## ST_Envelope

#### Description

Returns the bounding box (envelope) of a geometry as a new geometry. The resulting geometry represents the minimum bounding rectangle that encloses the input geometry. Depending on the input, the output can be a `Point`, `LineString`, or `Polygon`.

Since: v0.1.

#### Format

`ST_Envelope (A: Geometry)`

#### Arguments

  * **geom**: The input geometry.

#### SQL Example

```sql
SELECT ST_Envelope(ST_Point(1.0, 2.0));
```

## ST_Envelope_Agg

#### Description

An aggregate function that returns the collective bounding box (envelope) of a set of geometries.

Since: v0.1.

#### Format

`ST_Envelope_Agg (geom: Geometry)`

#### Arguments

  * **geom**: A column of geometries to be aggregated.

#### SQL Example

```sql
SELECT ST_Envelope_Agg(ST_GeomFromText('POLYGON ((0 0, 1 0, 0 1, 0 0))'))
```

## ST_Equals

#### Description

Return true if geomA equals geomB.

Since: v0.1.

#### Format

`ST_Equals (A: Geometry, B: Geometry)`

#### Arguments

  * **geomA**: Input geometry or geography.
  * **geomB**: Input geometry or geography.

#### SQL Example

```sql
SELECT ST_Equals(
  ST_Point(0.25, 0.25),
  ST_GeomFromText('POLYGON ((0 0, 1 0, 0 1, 0 0))')
);
```

## ST_FlipCoordinates

#### Description

Returns a new geometry with the X and Y coordinates of each vertex swapped. This is useful for correcting geometries that have been created with longitude and latitude in the wrong order.

Since: v0.1.

#### Format

`ST_FlipCoordinates (A: geometry)`

#### Arguments

  * **geom**: The input geometry whose coordinates will be flipped.

#### SQL Example

```sql
SELECT ST_FlipCoordinates(
  ST_Point(1, 2)
);
```

## ST_GeogPoint

ST_GeogPoint creates a geography POINT from given longitude and latitude coordinates.

```sql
SELECT ST_GeogPoint(10, 10), ST_GeogPoint(0, 0);
```

## ST_GeogFromWKB

Construct a Geography from WKB Binary.

Format: `ST_GeogFromWKB (Wkb: Binary)`

Since: v0.2.

```sql
SELECT ST_GeogFromWKB(decode('010200000002000000000000000084d600c0000000000080b5d6bf00000060e1eff7bf00000080075de5bf', 'hex'));
```

## ST_GeogFromWKT

Construct a Geography from WKT. If SRID is not set, it defaults to 0 (unknown).

Format: `ST_GeogFromWKT (Wkt: String)`

or

`ST_GeogFromWKT (Wkt: String, srid: Integer)`

Since: v0.2.

```sql
SELECT ST_GeogFromWKT('LINESTRING (1 2, 3 4, 5 6)');
```

## ST_GeometryN

Return the 0-based Nth geometry if the geometry is a GEOMETRYCOLLECTION, (MULTI)POINT, (MULTI)LINESTRING, MULTICURVE or (MULTI)POLYGON. Otherwise, return null

Format: ST_GeometryN(geom: Geometry, n: Integer)

Since: v0.2.

```sql
SELECT ST_GeometryN(ST_GeomFromText('MULTIPOINT((1 2), (3 4), (5 6), (8 9))'), 1);
```

## ST_GeomFromWKB

#### Description

Construct a Geometry from WKB.

Since: v0.1.

#### Format

`ST_GeomFromWKB (Wkb: Binary)`

#### Arguments

  * **WKB**: binary: Well-known binary representation of the geometry.

#### SQL Example

```sql
-- Creates a POINT(1 2) geometry from its WKB representation
SELECT ST_AsText(
    ST_GeomFromWKB(decode('0101000000000000000000F03F0000000000000040', 'hex'))
);
```

## ST_GeomFromWKT

#### Description

Construct a Geometry from WKT. This function also has the aliases **ST_GeomFromText** and ** ST_GeometryFromText**

Since: v0.1.

#### Format

`ST_GeomFromWKT (Wkt: String)`

#### Arguments

  * **WKT**: string: Well-known text representation of the geometry.

#### SQL Example

```sql
SELECT ST_AsText(ST_GeomFromWKT('POINT (30 10)'));
```

## ST_GeometryType

#### Description

Return the type of a geometry.

Since: v0.1.

#### Format

`ST_GeometryType (A: Geometry)`

#### Arguments

  * **geom**: Input geometry.

#### SQL Example

```sql
SELECT ST_GeometryType(ST_GeomFromWKT('POLYGON ((0 0, 1 0, 0 1, 0 0))'));
```

## ST_HasM

#### Description

Return true if the geometry has a M dimension.

Since: v0.1.

#### Format

`ST_HasM (A: Geometry)`

#### Arguments

  * **geom**: Input geometry.

#### SQL Example

```sql
SELECT ST_HasM(ST_GeomFromWKT('POLYGON ((0 0, 1 0, 0 1, 0 0))'));
```

## ST_HasZ

#### Description

Return true if the geometry has a Z dimension.

Since: v0.1.

#### Format

`ST_HasZ (A: Geometry)`

#### Arguments

  * **geom**: Input geometry.

#### SQL Example

```sql
SELECT ST_HasZ(ST_GeomFromWKT('POLYGON ((0 0, 1 0, 0 1, 0 0))'));
```

## ST_Intersection

#### Description

Computes the intersection between geomA and geomB.

Since: v0.1.

#### Format

`ST_Intersection (A: Geometry, B: Geometry)`

#### Arguments

  * **geomA**: Input geometry or geography.
  * **geomB**: Input geometry or geography.

#### SQL Example

```sql
SELECT ST_Intersection(
    ST_GeomFromText('POLYGON ((1 1, 11 1, 11 11, 1 11, 1 1))'),
    ST_GeomFromText('POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))')
) AS val;
```

## ST_Intersection_Agg

#### Description

An aggregate function that returns the geometric intersection of all geometries in a set.

Since: v0.1.

#### Format

`ST_Intersection_Agg (geom: Geometry)`

#### Arguments

  * **geom**: A column of geometries to be aggregated.

#### SQL Example

```sql
-- Create a table with overlapping polygons and find their common intersection
WITH shapes(geom) AS (
    VALUES (ST_GeomFromWKT('POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))')),
           (ST_GeomFromWKT('POLYGON((1 1, 3 1, 3 3, 1 3, 1 1))'))
)
SELECT ST_AsText(ST_Intersection_Agg(geom)) FROM shapes;
-- Returns: POLYGON ((1 1, 1 2, 2 2, 2 1, 1 1))
```

## ST_Intersects

#### Description

Return true if geomA intersects geomB.

Since: v0.1.

#### Format

`ST_Intersects (A: Geometry, B: Geometry)`

#### Arguments

  * **geomA**: Input geometry or geography.
  * **geomB**: Input geometry or geography.

#### SQL Example

```sql
SELECT ST_Intersects(ST_Point(0.25, 0.25), ST_GeomFromText('POLYGON ((0 0, 1 0, 0 1, 0 0))')) AS val;
```

## ST_IsClosed

RETURNS true if the LINESTRING start and end point are the same.

Format: `ST_IsClosed(geom: Geometry)`

Since: v0.2.

```sql
SELECT ST_IsClosed(ST_GeomFromText('LINESTRING(0 0, 1 1, 1 0)'));
```

## ST_IsCollection

Returns TRUE if the geometry type of the input is a geometry collection type. Collection types are the following:

```
GEOMETRYCOLLECTION
MULTI{POINT, POLYGON, LINESTRING}
```

Format: `ST_IsCollection(geom: Geometry)`

Since: v0.2.

```sql
SELECT ST_IsCollection(ST_GeomFromText('MULTIPOINT(0 0), (6 6)'));
```

```sql
SELECT ST_IsCollection(ST_GeomFromText('POINT(5 5)'));
```

## ST_IsEmpty

#### Description

Return true if the geometry is empty.

Since: v0.1.

#### Format

`ST_IsEmpty (A: Geometry)`

#### Arguments

  * **geom**: Input geometry.

#### SQL Example

```sql
SELECT ST_IsEmpty(ST_GeomFromWKT('POLYGON EMPTY'));
```

## ST_IsRing

RETURN true if LINESTRING is ST_IsClosed and ST_IsSimple.

Format: `ST_IsRing(geom: Geometry)`

Since: v0.2.

```sql
SELECT ST_IsRing(ST_GeomFromText('LINESTRING(0 0, 0 1, 1 1, 1 0, 0 0)'));
```

## ST_IsSimple

Test if geometry's only self-intersections are at boundary points.

Format: `ST_IsSimple (A: Geometry)`

Since: v0.2.

```sql
SELECT ST_IsSimple(ST_GeomFromWKT('POLYGON((1 1, 3 1, 3 3, 1 3, 1 1))'));
```

## ST_IsValid

ST_IsValid checks whether a geometry meets the rules of a valid spatial object according to the OGC standard.

```sql
SELECT ST_IsValid(
    ST_GeomFromText('POLYGON ((0 0, 2 0, 2 2, 1 1, 0 2, 0 0))')
) AS is_valid;
```

## ST_IsValidReason

ST_IsValidReason returns a text explanation describing why a geometry is invalid.

```sql
SELECT ST_IsValidReason(
    ST_GeomFromText('POLYGON((0 0, 2 0, 2 2, 1 1, 0 2, 0 0))')
) AS reason;
```

## ST_KNN

#### Description

Return true if geomA finds k nearest neighbors from geomB.

Since: v0.1.

#### Format

`ST_KNN (A: Geometry, B: Geometry, k: Integer, use_spheroid: Boolean)`

#### Arguments

  * **geomA**: Query geometry or geography.
  * **geomB**: Object geometry or geography.
  * **k**: Number of nearest neighbors to find.
  * **use_spheroid**: Use spheroid distance calculation.

#### SQL Example

```sql
SELECT * FROM table1 a JOIN table2 b ON ST_KNN(a.geom, b.geom, 5, false);
```

## ST_Length

#### Description

Returns the length of geom. This function only supports LineString, MultiLineString, and GeometryCollections containing linear geometries. Use ST_Perimeter for polygons.

Since: v0.1.

#### Format

`ST_Length (A: Geometry)`

#### Arguments

  * **geom**: geometry: Input geometry.

#### SQL Example

```sql
SELECT ST_Length(ST_GeomFromWKT('LINESTRING(0 0, 10 0)'));
```

## ST_LineInterpolatePoint

Returns a point interpolated along a line. First argument must be a LINESTRING. Second argument is a Double between 0 and 1 representing fraction of total linestring length the point has to be located.

Format: `ST_LineInterpolatePoint (geom: Geometry, fraction: Double)`

Since: v0.2.

```sql
SELECT ST_LineInterpolatePoint(ST_GeomFromWKT('LINESTRING(25 50, 100 125, 150 190)'), 0.2);
```

## ST_LineLocatePoint

Returns a double between 0 and 1, representing the location of the closest point on the LineString as a fraction of its total length. The first argument must be a LINESTRING, and the second argument is a POINT geometry.

Format: `ST_LineLocatePoint(linestring: Geometry, point: Geometry)`

Since: v0.2.

```sql
SELECT ST_LineLocatePoint(
    ST_GeogFromText('LINESTRING(0 0, 1 1, 2 2)'),
    ST_GeogFromText('POINT(0 2)')
);
```

## ST_M

#### Description

Returns the M (measure) coordinate of a `Point` geometry. If the geometry does not have an M value, it returns `NULL`.

Since: v0.1.

#### Format

`ST_M (A: Point)`

#### Arguments

  * **geom**: The input point geometry or geography.

#### SQL Example

```sql
SELECT ST_M(ST_Point(1.0, 2.0));
```

## ST_MakeLine

#### Description

Creates a `LineString` from two or more input `Point`, `MultiPoint`, or `LineString` geometries. The function connects the input geometries in the order they are provided to form a single continuous line.

Since: v0.1.

#### Format

`ST_MakeLine (g1: Geometry or Geography, g2: Geometry or Geography)`

#### Arguments

  * **g1**: The first `Point`, `MultiPoint`, or `LineString` geometry or geography.
  * **g2**: The second `Point`, `MultiPoint`, or `LineString` geometry or geography.

#### SQL Example

```sql
SELECT ST_MakeLine(ST_Point(0, 1), ST_Point(2, 3)) as geom;
```

## ST_MakeValid

Given an invalid geometry, create a valid representation of the geometry.

Collapsed geometries are either converted to empty (keepCollapsed=false) or a valid geometry of lower dimension (keepCollapsed=true). Default is keepCollapsed=false.

Format:

```
ST_MakeValid (A: Geometry)
ST_MakeValid (A: Geometry, keepCollapsed: Boolean)
```

Since: v0.2.

```sql
SELECT ST_MakeValid(ST_GeomFromWKT('LINESTRING(1 1, 1 1)'));
```

## ST_MaxDistance

#### Description

Calculates the maximum distance between geomA and geomB.

Since: v0.1.

#### Format

`ST_MaxDistance (A: Geometry, B: Geometry)`

#### Arguments

  * **geomA**: Input geometry or geography.
  * **geomB**: Input geometry or geography.

#### SQL Example

```sql
SELECT ST_MaxDistance(
    ST_GeogFromText('POLYGON ((10 10, 11 10, 10 11, 10 10))'),
    ST_GeogFromText('POLYGON ((0 0, 1 0, 0 1, 0 0))')
) AS val;
```

## ST_MinimumClearance

The minimum clearance is a metric that quantifies a geometry's tolerance to changes in coordinate precision or vertex positions. It represents the maximum distance by which vertices can be adjusted without introducing invalidity to the geometry's structure. A larger minimum clearance value indicates greater robustness against such perturbations.

For a geometry with a minimum clearance of x, the following conditions hold:

No two distinct vertices are separated by a distance less than x.

No vertex lies within a distance x from any line segment it is not an endpoint of.

For geometries with no definable minimum clearance, such as single Point geometries or MultiPoint geometries where all points occupy the same location, the function returns Double.MAX_VALUE.

Format: `ST_MinimumClearance(geometry: Geometry)`

Since: v0.2.

```sql
SELECT ST_MinimumClearance(
  ST_GeomFromWKT('POLYGON ((65 18, 62 16, 64.5 16, 62 14, 65 14, 65 18))')
);
```

## ST_MinimumClearanceLine

This function returns a two-point LineString geometry representing the minimum clearance distance of the input geometry. If the input geometry does not have a defined minimum clearance, such as for single Points or coincident MultiPoints, an empty LineString geometry is returned instead.

Format: `ST_MinimumClearanceLine(geometry: Geometry)`

Since: v0.2.

```sql
SELECT ST_MinimumClearanceLine(
  ST_GeomFromWKT('POLYGON ((65 18, 62 16, 64.5 16, 62 14, 65 14, 65 18))')
);
```

## ST_MMax

#### Description

Returns the maximum M (measure) value from a geometry's bounding box.

Since: v0.1.

#### Format

`ST_MMax (A: Geometry)`

#### Arguments

  * **geom**: The input geometry.

#### SQL Example

```sql
SELECT ST_MMax(ST_GeomFromWKT('POLYGON ((0 0, 1 0, 0 1, 0 0))'));
```

## ST_MMin

#### Description

Returns the minimum **M-coordinate** (measure) of a geometry's bounding box.

Since: v0.1.

#### Format

`ST_MMin (A: Geometry)`

#### Arguments

  * **geom**: Input geometry.

#### SQL Example

```sql
SELECT ST_MMin(ST_GeomFromWKT('LINESTRING ZM (1 2 3 4, 5 6 7 8)'));
-- Returns: 4
```

## ST_NPoints

Return points of the geometry

Format: `ST_NPoints (A: Geometry)`

Since: v0.2.

```sql
SELECT ST_NPoints(ST_GeomFromText('LINESTRING(77.29 29.07,77.42 29.26,77.27 29.31,77.29 29.07)'));
```

## ST_NumGeometries

Returns the number of Geometries. If geometry is a GEOMETRYCOLLECTION (or MULTI*) return the number of geometries, for single geometries will return 1.

Format: `ST_NumGeometries (A: Geometry)`

Since: v0.2.

```sql
SELECT ST_NumGeometries(ST_GeomFromWKT('LINESTRING (-29 -27, -30 -29.7, -45 -33)'));
```

## ST_Overlaps

Return true if A overlaps B

Format: `ST_Overlaps (A: Geometry, B: Geometry)`

Since: v0.2.

```sql
SELECT ST_Overlaps(ST_GeomFromWKT('POLYGON((2.5 2.5, 2.5 4.5, 4.5 4.5, 4.5 2.5, 2.5 2.5))'), ST_GeomFromWKT('POLYGON((4 4, 4 6, 6 6, 6 4, 4 4))'));
```

## ST_Perimeter

#### Description

This function calculates the 2D perimeter of a given geometry. It supports Polygon, MultiPolygon, and GeometryCollection geometries (as long as the GeometryCollection contains polygonal geometries). For other types, it returns 0. To measure lines, use ST_Length.

To get the perimeter in meters, set **use_spheroid** to true. This calculates the geodesic perimeter using the WGS84 spheroid. When using use_spheroid, the **lenient** parameter defaults to true, assuming the geometry uses EPSG:4326. To throw an exception instead, set lenient to false.

Since: v0.1.

#### Format

`ST_Perimeter(geom: Geometry)`
`ST_Perimeter(geom: Geometry, use_spheroid: Boolean)`
`ST_Perimeter(geom: Geometry, use_spheroid: Boolean, lenient: Boolean = True)`

#### Arguments

  * **geom**: Input geometry.
  * **use_spheroid**: If true, calculates the geodesic perimeter using the WGS84 spheroid. Defaults to false.
  * **lenient**: If true, assumes the geometry uses EPSG:4326 when use_spheroid is true. Defaults to true.

#### SQL Example

```sql
SELECT ST_Perimeter(ST_GeomFromWKT('POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))'));
```

## ST_Point

#### Description

Construct a Point Geometry from X and Y.

Since: v0.1.

#### Format

`ST_Point (x: Double, y: Double)`

#### Arguments

  * **x**: X value.
  * **y**: Y value.

#### SQL Example

```sql
SELECT ST_AsText(ST_Point(-74.0060, 40.7128));
```

## ST_PointM

#### Description

Constructs a `Point` with an M (measure) coordinate from X, Y, and M values.

Since: v0.1.

#### Format

`ST_PointM (x: Double, y: Double, m: Double)`

#### Arguments

  * **x**: The X-coordinate value.
  * **y**: The Y-coordinate value.
  * **m**: The M-coordinate (measure) value.

#### SQL Example

```sql
SELECT ST_PointM(-64.36, 45.09, 50.0);
```

## ST_PointN

Return the Nth point in a single linestring or circular linestring in the geometry. Negative values are counted backwards from the end of the LineString, so that -1 is the last point. Returns NULL if there is no linestring in the geometry.

Format: `ST_PointN(geom: Geometry, n: Integer)`

Since: v0.2.

```sql
SELECT ST_PointN(ST_GeomFromText('LINESTRING(0 0, 1 2, 2 4, 3 6)'), 2);
```

## ST_Points

Returns a MultiPoint geometry consisting of all the coordinates of the input geometry. It preserves duplicate points as well as M and Z coordinates.

Format: `ST_Points(geom: Geometry)`

Since: v0.2.

```sql
SELECT ST_AsText(ST_Points(ST_GeomFromText('LINESTRING (2 4, 3 3, 4 2, 7 3)')));
```

## ST_PointZ

#### Description

Constructs a `Point` with a Z (elevation) coordinate from X, Y, and Z values.

Since: v0.1.

#### Format

`ST_PointZ (x: Double, y: Double, z: Double)`

#### Arguments

  * **x**: The X-coordinate value.
  * **y**: The Y-coordinate value.
  * **z**: The Z-coordinate (elevation) value.

#### SQL Example

```sql
SELECT ST_PointZ(-64.36, 45.09, 100.0);
```

## ST_PointZM

#### Description

Constructs a `Point` with both Z (elevation) and M (measure) coordinates from X, Y, Z, and M values.

Since: v0.1.

#### Format

`ST_PointZM (x: Double, y: Double, z: Double, m: Double)`

#### Arguments

  * **x**: The X-coordinate value.
  * **y**: The Y-coordinate value.
  * **z**: The Z-coordinate (elevation) value.
  * **m**: The M-coordinate (measure) value.

#### SQL Example

```sql
SELECT ST_PointZM(-64.36, 45.09, 100.0, 50.0);
```

## ST_Polygonize

Generates a GeometryCollection composed of polygons that are formed from the linework of an input GeometryCollection. When the input does not contain any linework that forms a polygon, the function will return an empty GeometryCollection.

Note that ST_Polygonize function assumes that the input geometries form a valid and simple closed linestring that can be turned into a polygon. If the input geometries are not noded or do not form such linestrings, the resulting GeometryCollection may be empty or may not contain the expected polygons.

Format: `ST_Polygonize(geom: Geometry)`

Since: v0.2.

```sql
SELECT ST_AsText(ST_Polygonize(ST_GeomFromText('GEOMETRYCOLLECTION (LINESTRING (2 0, 2 1, 2 2), LINESTRING (2 2, 2 3, 2 4), LINESTRING (0 2, 1 2, 2 2), LINESTRING (2 2, 3 2, 4 2), LINESTRING (0 2, 1 3, 2 4), LINESTRING (2 4, 3 3, 4 2))')));
```

## ST_Polygonize_Agg

ST_Polygonize_Agg is an aggregate function that combines a set of linestrings into polygons by finding closed rings formed by the input geometries.

```sql
SELECT ST_Polygonize_Agg(geom) AS polygons
FROM (
    SELECT ST_GeomFromText('LINESTRING(0 0, 0 1)') AS geom
    UNION ALL
    SELECT ST_GeomFromText('LINESTRING(0 1, 1 1)')
    UNION ALL
    SELECT ST_GeomFromText('LINESTRING(1 1, 1 0)')
    UNION ALL
    SELECT ST_GeomFromText('LINESTRING(1 0, 0 0)')
) AS edges;
```

## ST_Reverse

Return the geometry with vertex order reversed

Format: `ST_Reverse (A: Geometry)`

Since: v0.2.

```sql
SELECT ST_Reverse(ST_GeomFromWKT('LINESTRING(0 0, 1 2, 2 4, 3 6)'));
```

## ST_SetCRS

ST_SetCRS sets or changes the Coordinate Reference System (CRS) identifier of a geometry without transforming its coordinates.

```python
import pyproj
crs = pyproj.CRS("EPSG:3857")

sd.sql(f"""
SELECT ST_SetCRS(ST_GeomFromText('LINESTRING(0 1, 2 3)'), '{crs.to_json()}') as geom;
""").show()
```

## ST_Simplify

This function simplifies the input geometry by applying the Douglas-Peucker algorithm.

Note: The simplification may not preserve topology, potentially producing invalid geometries. Use ST_SimplifyPreserveTopology to retain valid topology after simplification.

Format: `ST_Simplify(geom: Geometry, tolerance: Double)`

Since: v0.2.

```sql
SELECT ST_Simplify(ST_Buffer(ST_GeomFromWKT('POINT (0 2)'), 10), 1);
```

## ST_SimplifyPreserveTopology

Simplifies a geometry and ensures that the result is a valid geometry having the same dimension and number of components as the input, and with the components having the same topological relationship.

Since: v0.2.

Format: `ST_SimplifyPreserveTopology (A: Geometry, distanceTolerance: Double)`

```sql
SELECT ST_SimplifyPreserveTopology(ST_GeomFromText('POLYGON((8 25, 28 22, 28 20, 15 11, 33 3, 56 30, 46 33,46 34, 47 44, 35 36, 45 33, 43 19, 29 21, 29 22,35 26, 24 39, 8 25))'), 10);
```

## ST_Snap

Snaps the vertices and segments of the input geometry to reference geometry within the specified tolerance distance. The tolerance parameter controls the maximum snap distance.

If the minimum distance between the geometries exceeds the tolerance, the input geometry is returned unmodified. Adjusting the tolerance value allows tuning which vertices should snap to the reference and which remain untouched.

Since: v0.2.

Format: `ST_Snap(input: Geometry, reference: Geometry, tolerance: double)`

```sql
SELECT
    ST_Snap(poly, line, ST_Distance(poly, line) * 1.01) AS polySnapped FROM (
        SELECT ST_GeomFromWKT('POLYGON ((236877.58 -6.61, 236878.29 -8.35, 236879.98 -8.33, 236879.72 -7.63, 236880.35 -6.62, 236877.58 -6.61), (236878.45 -7.01, 236878.43 -7.52, 236879.29 -7.50, 236878.63 -7.22, 236878.76 -6.89, 236878.45 -7.01))') as poly,
            ST_GeomFromWKT('LINESTRING (236880.53 -8.22, 236881.15 -7.68, 236880.69 -6.81)') as line
);
```

## ST_SetSRID

#### Description

Sets the spatial reference system identifier (SRID) of a geometry. This only changes the metadata; it does not transform the coordinates.

Since: v0.1.

#### Format

`ST_SetSRID (geom: Geometry, srid: Integer)`

#### Arguments

  * **geom**: Input geometry or geography.
  * **srid**: EPSG code to set (e.g., 4326).

#### SQL Example

```sql
SELECT ST_SetSRID(ST_GeomFromWKT('POINT (-64.363049 45.091501)'), 4326);
```

## ST_SRID

#### Description

Returns the Spatial Reference System Identifier (SRID) of a geometry. If the geometry does not have an SRID, it returns 0.

Since: v0.1.

#### Format

`ST_SRID (geom: Geometry)`

#### Arguments

  * **geom**: The input geometry or geography.

#### SQL Example

```sql
SELECT ST_SRID(polygon)
```

## ST_StartPoint

Returns first point of given linestring.

Format: `ST_StartPoint(geom: Geometry)`

Since: v0.2.

```sql
SELECT ST_StartPoint(ST_GeomFromText('LINESTRING(100 150,50 60, 70 80, 160 170)'));
```

## ST_SymDifference

#### Description

Computes the symmetric difference between geomA and geomB.

Since: v0.1.

#### Format

`ST_SymDifference (A: Geometry, B: Geometry)`

#### Arguments

  * **geomA**: Input geometry or geography.
  * **geomB**: Input geometry or geography.

#### SQL Example

```sql
SELECT ST_SymDifference(
    ST_GeomFromText('POLYGON ((1 1, 11 1, 11 11, 1 11, 1 1))'),
    ST_GeomFromText('POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))')
) AS val;
```

## ST_Touches

#### Description

Return true if geomA touches geomB.

Since: v0.1.

#### Format

`ST_Touches (A: Geometry, B: Geometry)`

#### Arguments

  * **geomA**: Input geometry or geography.
  * **geomB**: Input geometry or geography.

#### SQL Example

```sql
SELECT ST_Touches(ST_Point(0.25, 0.25), ST_GeomFromText('POLYGON ((0 0, 1 0, 0 1, 0 0))')) AS val;
```

## ST_Transform

#### Description

Transforms the coordinates of a geometry from a source Coordinate Reference System (CRS) to a target CRS.

If the source CRS is not specified, it will be read from the geometry's metadata. Sedona ensures that coordinates are handled in longitude/latitude order for geographic CRS transformations.

Since: v0.1.

#### Format

`ST_Transform (A: Geometry, TargetCRS: String)`
`ST_Transform (A: Geometry, SourceCRS: String, TargetCRS: String)`

#### Arguments

  * **geom**: Input geometry or geography.
  * **source_crs**: The source CRS code (e.g., 'EPSG:4326').
  * **target_crs**: The target CRS code to transform into.
  * **lenient**: A boolean that, if true, assumes the source is EPSG:4326 if not specified. Defaults to true.

#### SQL Example

```sql
-- Transform a WGS84 polygon to UTM zone 49N
SELECT ST_Transform(ST_SetSRID(ST_GeomFromWkt('POLYGON((170 50,170 72,-130 72,-130 50,170 50))'), 4326), 'EPSG:32649');
```

## ST_Translate

Returns the input geometry with its X, Y and Z coordinates (if present in the geometry) translated by deltaX, deltaY and deltaZ (if specified)

If the geometry is 2D, and a deltaZ parameter is specified, no change is done to the Z coordinate of the geometry and the resultant geometry is also 2D.

If the geometry is empty, no change is done to it. If the given geometry contains sub-geometries (GEOMETRY COLLECTION, MULTI POLYGON/LINE/POINT), all underlying geometries are individually translated.

Format: `ST_Translate(geometry: Geometry, deltaX: Double, deltaY: Double, deltaZ: Double)`

Since: v0.2.

```sql
SELECT ST_Translate(ST_GeomFromText('POINT(-71.01 42.37)'), 1, 2);
```

## vST_UnaryUnion

This variant of ST_Union operates on a single geometry input. The input geometry can be a simple Geometry type, a MultiGeometry, or a GeometryCollection. The function calculates the geometric union across all components and elements within the provided geometry object.

Format: `ST_UnaryUnion(geometry: Geometry)`

Since: v0.2.

```sql
SELECT ST_UnaryUnion(ST_GeomFromWKT('MULTIPOLYGON(((0 10,0 30,20 30,20 10,0 10)),((10 0,10 20,30 20,30 0,10 0)))'));
```

## ST_Union

#### Description

Computes the union between geomA and geomB.

Since: v0.1.

#### Format

`ST_Union (A: Geometry, B: Geometry)`

#### Arguments

  * **geomA**: Input geometry or geography.
  * **geomB**: Input geometry or geography.

#### SQL Example

```sql
SELECT ST_Union(
    ST_GeomFromText('POLYGON ((1 1, 11 1, 11 11, 1 11, 1 1))'),
    ST_GeomFromText('POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))')
) AS val;
```

## ST_Union_Agg

#### Description

An aggregate function that returns the geometric union of all geometries in a set.

Since: v0.1.

#### Format

`ST_Union_Agg (geom: Geometry)`

#### Arguments

  * **geom**: A column of geometries to be aggregated.

#### SQL Example

```sql
SELECT ST_AsText(ST_Union_Agg(geom))
FROM (
    SELECT ST_GeomFromWKT('POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))') AS geom
    UNION ALL
    SELECT ST_GeomFromWKT('POLYGON ((2 2, 3 2, 3 3, 2 3, 2 2))')
) AS shapes;
```

## ST_Within

#### Description

Return true if geomA is fully contained by geomB.

Since: v0.1.

#### Format

`ST_Within (A: Geometry, B: Geometry)`

#### Arguments

  * **geomA**: Input geometry or geography.
  * **geomB**: Input geometry or geography.

#### SQL Example

```sql
SELECT ST_Within(ST_Point(0.25, 0.25), ST_GeomFromText('POLYGON ((0 0, 1 0, 0 1, 0 0))')) AS val;
```

## ST_X

#### Description

Return the X component of a point geometry or geography.

Since: v0.1.

#### Format

`ST_X(A: Point)`

#### Arguments

  * **geom**: Input geometry or geography.

#### SQL Example

```sql
SELECT ST_X(ST_Point(1.0, 2.0));
```

## ST_XMax

#### Description

Returns the maximum **X-coordinate** of a geometry's bounding box.

Since: v0.1.

#### Format

`ST_XMax (A: Geometry)`

#### Arguments

  * **geom**: Input geometry.

#### SQL Example

```sql
SELECT ST_XMax(ST_GeomFromWKT('LINESTRING(1 5, 10 15)'));
-- Returns: 10
```

## ST_XMin

#### Description

Returns the minimum **X-coordinate** of a geometry's bounding box.

Since: v0.1.

#### Format

`ST_XMin (A: Geometry)`

#### Arguments

  * **geom**: Input geometry.

#### SQL Example

```sql
SELECT ST_XMin(ST_GeomFromWKT('LINESTRING(1 5, 10 15)'));
-- Returns: 1
```

## ST_Y

#### Description

Return the Y component of a point geometry or geography.

Since: v0.1.

#### Format

`ST_Y(A: Point)`

#### Arguments

  * **geom**: Input geometry or geography.

#### SQL Example

```sql
SELECT ST_Y(ST_Point(1.0, 2.0));
```

## ST_YMax

#### Description

Returns the maximum **Y-coordinate** of a geometry's bounding box.

Since: v0.1.

#### Format

`ST_YMax (A: Geometry)`

#### Arguments

  * **geom**: Input geometry.

#### SQL Example

```sql
SELECT ST_YMax(ST_GeomFromWKT('LINESTRING(1 5, 10 15)'));
-- Returns: 15
```

## ST_YMin

#### Description

Returns the minimum **Y-coordinate** of a geometry's bounding box.

Since: v0.1.

#### Format

`ST_YMin (A: Geometry)`

#### Arguments

  * **geom**: Input geometry.

#### SQL Example

```sql
SELECT ST_YMin(ST_GeomFromWKT('LINESTRING(1 5, 10 15)'));
-- Returns: 5
```

## ST_Z

#### Description

Return the Z component of a point geometry or geography.

Since: v0.1.

#### Format

`ST_Z(A: Point)`

#### Arguments

  * **geom**: Input geometry or geography.

#### SQL Example

```sql
SELECT ST_Z(ST_Point(1.0, 2.0));
```

## ST_ZMax

#### Description

Returns the maximum **Z-coordinate** of a geometry's bounding box.

Since: v0.1.

#### Format

`ST_ZMax (A: Geometry)`

#### Arguments

  * **geom**: Input geometry.

#### SQL Example

```sql
SELECT ST_ZMax(ST_GeomFromWKT('LINESTRING ZM (1 2 3 4, 5 6 7 8)'));
-- Returns: 7.0
```

## ST_ZMin

#### Description

Returns the minimum **Z-coordinate** of a geometry's bounding box.

Since: v0.1.

#### Format

`ST_ZMin (A: Geometry)`

#### Arguments

  * **geom**: Input geometry.

#### SQL Example

```sql
SELECT ST_ZMin(ST_GeomFromWKT('LINESTRING ZM (1 2 3 4, 5 6 7 8)'));
-- Returns: 3
```

## ST_Zmflag

Returns a code indicating the Z and M coordinate dimensions present in the input geometry.

Values are: 0 = 2D, 1 = 3D-M, 2 = 3D-Z, 3 = 4D.

Format: `ST_Zmflag(geom: Geometry)`

Since: v0.2.

```sql
SELECT ST_Zmflag(
  ST_GeomFromWKT('LINESTRING Z(1 2 3, 4 5 6)')
);
```
