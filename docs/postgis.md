# SedonaDB + PostGIS

This page demonstrates how to read PostGIS tables into SedonaDB DataFrames.

You need to install these Python packages to run this notebook:

* `psycopg2-binary`
* `sqlalchemy`

Let's start by creating a PostGIS table that SedonaDB can read.

Here's how to create the `my_places` table:

```sql
CREATE TABLE my_places (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    geom GEOMETRY(Point, 4326)
);
```

Now add some data to the table:

```sql
INSERT INTO my_places (name, geom) VALUES
    ('New York', ST_SetSRID(ST_MakePoint(-74.006, 40.7128), 4326)),
    ('Los Angeles', ST_SetSRID(ST_MakePoint(-118.2437, 34.0522), 4326)),
    ('Chicago', ST_SetSRID(ST_MakePoint(-87.6298, 41.8781), 4326));
```

View the content of the table to make sure it was created correctly:

```
SELECT id, name, ST_AsText(geom) FROM my_places;

 id |    name     |        st_astext         
----+-------------+--------------------------
  1 | New York    | POINT(-74.006 40.7128)
  2 | Los Angeles | POINT(-118.2437 34.0522)
  3 | Chicago     | POINT(-87.6298 41.8781)
```

Let's start by reading the PostGIS table into a GeoPandas DataFrame.  The GeoPandas DataFrame can then easily be converted to a SedonaDB DataFrame.


```python
import geopandas as gpd
from sqlalchemy import create_engine
import sedona.db
```


```python
# you need to replace your username and database name in the following string
engine = create_engine('postgresql://matthewpowers@localhost:5432/matthewpowers')
```


```python
gdf = gpd.read_postgis('SELECT * FROM my_places', engine, geom_col='geom')
```


```python
print(gdf)
```

       id         name                       geom
    0   1     New York    POINT (-74.006 40.7128)
    1   2  Los Angeles  POINT (-118.2437 34.0522)
    2   3      Chicago   POINT (-87.6298 41.8781)


## Read PostGIS table into SedonaDB DataFrame


```python
sd = sedona.db.connect()
```


```python
# read the GeoPandas DataFrame to a SedonaDB DataFrame

df = sd.create_data_frame(gdf)
```


```python
df.show()
```

    ┌───────┬─────────────┬──────────────────────────┐
    │   id  ┆     name    ┆           geom           │
    │ int64 ┆     utf8    ┆         geometry         │
    ╞═══════╪═════════════╪══════════════════════════╡
    │     1 ┆ New York    ┆ POINT(-74.006 40.7128)   │
    ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │     2 ┆ Los Angeles ┆ POINT(-118.2437 34.0522) │
    ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │     3 ┆ Chicago     ┆ POINT(-87.6298 41.8781)  │
    └───────┴─────────────┴──────────────────────────┘



```python
# confirm that the SedonaDB DataFrame retains the CRS defined in the PostGIS table

df.schema
```




    SedonaSchema with 3 fields:
      id: int64<Int64>
      name: utf8<Utf8>
      geom: geometry<Wkb(epsg:4326)>


