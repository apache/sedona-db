// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
use datafusion_common::{DataFusionError, Result};
use sedona_functions::executor::{GenericExecutor, GeometryFactory};

use crate::wkb_to_geos::GEOSWkbFactory;

/// A [GenericExecutor] that iterates over [geos::Geometry]
pub type GeosExecutor<'a, 'b> = GenericExecutor<'a, 'b, GeosGeometryFactory, GeosGeometryFactory>;

/// [GeometryFactory] implementation for iterating over [geos::Geometry]
#[derive(Default)]
pub struct GeosGeometryFactory {
    inner: GEOSWkbFactory,
}

impl GeometryFactory for GeosGeometryFactory {
    type Geom<'a> = geos::Geometry;

    fn try_from_wkb<'a>(&self, wkb_bytes: &'a [u8]) -> Result<Self::Geom<'a>> {
        let wkb =
            wkb::reader::read_wkb(wkb_bytes).map_err(|e| DataFusionError::External(Box::new(e)))?;
        self.inner
            .create(&wkb)
            .map_err(|e| DataFusionError::External(Box::new(e)))
    }
}

#[cfg(test)]
mod test {
    use datafusion_expr::ColumnarValue;
    use geos::Geom;
    use sedona_schema::datatypes::WKB_GEOMETRY;
    use sedona_testing::create::create_array_storage;

    use super::*;

    #[test]
    fn test_executor() {
        let items = vec![
            Some("POINT (0 1)"),
            Some("LINESTRING (1 2, 3 4)"),
            Some("POLYGON ((0 0, 1 0, 0 1, 0 0))"),
            Some("MULTIPOINT ((1 2), (3 4))"),
            Some("MULTILINESTRING ((1 2, 3 4))"),
            Some("MULTIPOLYGON (((0 0, 1 0, 0 1, 0 0)))"),
            Some("GEOMETRYCOLLECTION (POINT (1 2))"),
            None,
        ];
        let args = vec![ColumnarValue::Array(create_array_storage(
            &items,
            &WKB_GEOMETRY,
        ))];

        let expected_items = items
            .iter()
            .map(|item| item.map(|item| item.to_string()))
            .collect::<Vec<_>>();

        let mut actual_items = Vec::new();
        let executor = GeosExecutor::new(&[WKB_GEOMETRY], &args);
        executor
            .execute_wkb_void(|geo| {
                actual_items.push(geo.map(|geo| geo.to_wkt().unwrap()));
                Ok(())
            })
            .unwrap();
        assert_eq!(actual_items, expected_items);
    }
}
