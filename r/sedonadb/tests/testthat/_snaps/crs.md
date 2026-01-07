# sd_parse_crs works for GeoArrow metadata with EPSG

    Code
      sedonadb:::sd_parse_crs(meta)
    Output
      $authority_code
      [1] "EPSG:5070"
      
      $srid
      [1] 5070
      
      $name
      [1] "NAD83 / Conus Albers"
      
      $proj_string
      [1] "{\"id\":{\"authority\":\"EPSG\",\"code\":5070},\"name\":\"NAD83 / Conus Albers\"}"
      

# sd_parse_crs works for Engineering CRS (no EPSG ID)

    Code
      sedonadb:::sd_parse_crs(meta)
    Output
      $authority_code
      NULL
      
      $srid
      NULL
      
      $name
      [1] "Construction Site Local Grid"
      
      $proj_string
      [1] "{\"coordinate_system\":{\"axis\":[{\"abbreviation\":\"N\",\"direction\":\"north\",\"name\":\"Northing\",\"unit\":\"metre\"},{\"abbreviation\":\"E\",\"direction\":\"east\",\"name\":\"Easting\",\"unit\":\"metre\"}],\"subtype\":\"Cartesian\"},\"datum\":{\"name\":\"Local Datum\",\"type\":\"EngineeringDatum\"},\"name\":\"Construction Site Local Grid\",\"type\":\"EngineeringCRS\"}"
      

# sd_parse_crs returns NULL if crs field is missing

    Code
      sedonadb:::sd_parse_crs("{\"something_else\": 123}")
    Output
      NULL

---

    Code
      sedonadb:::sd_parse_crs("{}")
    Output
      NULL

# sd_parse_crs handles invalid JSON gracefully

    Code
      sedonadb:::sd_parse_crs("invalid json")
    Condition
      Error:
      ! Failed to parse metadata JSON: expected value at line 1 column 1

# sd_parse_crs works with plain strings if that's what's in 'crs'

    Code
      sedonadb:::sd_parse_crs(meta)
    Output
      $authority_code
      [1] "OGC:CRS84"
      
      $srid
      [1] 4326
      
      $name
      NULL
      
      $proj_string
      [1] "OGC:CRS84"
      

# print.sedonadb_dataframe shows CRS info for geometry column with EPSG

    Code
      print(df, n = 0)
    Output
      # A sedonadb_dataframe: ? x 1
      # Geometry: geom (CRS: ogc:crs84)
      +----------+
      |   geom   |
      | geometry |
      +----------+
      +----------+
      Preview of up to 0 row(s)

# print.sedonadb_dataframe shows CRS info with different SRID

    Code
      print(df, n = 0)
    Output
      # A sedonadb_dataframe: ? x 1
      # Geometry: geom (CRS: epsg:5070)
      +----------+
      |   geom   |
      | geometry |
      +----------+
      +----------+
      Preview of up to 0 row(s)

# print.sedonadb_dataframe shows multiple geometry columns with CRS

    Code
      print(df, n = 0)
    Output
      # A sedonadb_dataframe: ? x 2
      # Geometry: geom1 (CRS: ogc:crs84), geom2 (CRS: epsg:5070)
      +----------+----------+
      |   geom1  |   geom2  |
      | geometry | geometry |
      +----------+----------+
      +----------+----------+
      Preview of up to 0 row(s)

# print.sedonadb_dataframe handles geometry without explicit CRS

    Code
      print(df, n = 0)
    Output
      # A sedonadb_dataframe: ? x 1
      # Geometry: geom
      +----------+
      |   geom   |
      | geometry |
      +----------+
      +----------+
      Preview of up to 0 row(s)

# print.sedonadb_dataframe respects width parameter for geometry line

    Code
      print(df, n = 0, width = 60)
    Output
      # A sedonadb_dataframe: ? x 2
      # Geometry: very_long_geometry_column_name_1 (CRS: ogc:cr...
      +-----------------------------+----------------------------+
      | very_long_geometry_column_n | very_long_geometry_column_ |
      |           ame_1...          |          name_2...         |
      +-----------------------------+----------------------------+
      +-----------------------------+----------------------------+
      Preview of up to 0 row(s)

# sd_parse_crs handles empty string

    Code
      sedonadb:::sd_parse_crs("")
    Condition
      Error:
      ! Failed to parse metadata JSON: EOF while parsing a value at line 1 column 0

# sd_parse_crs handles CRS with only name, no ID

    Code
      sedonadb:::sd_parse_crs(meta)
    Output
      $authority_code
      NULL
      
      $srid
      NULL
      
      $name
      [1] "Custom Geographic CRS"
      
      $proj_string
      [1] "{\"name\":\"Custom Geographic CRS\",\"type\":\"GeographicCRS\"}"
      

# sd_parse_crs handles OGC:CRS84

    Code
      sedonadb:::sd_parse_crs(meta)
    Output
      $authority_code
      [1] "OGC:CRS84"
      
      $srid
      [1] 4326
      
      $name
      NULL
      
      $proj_string
      [1] "OGC:CRS84"
      

