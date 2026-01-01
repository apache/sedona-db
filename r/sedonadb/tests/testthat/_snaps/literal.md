# literals with Arrow extension metadata can be converted to literals

    Code
      as_sedonadb_literal(wk::as_wkb("POINT (0 1)"))
    Output
      <SedonaDBExpr>
      Binary("1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,63") FieldMetadata { inner: {"ARROW:extension:metadata": "{}", "ARROW:extension:name": "geoarrow.wkb"} }

