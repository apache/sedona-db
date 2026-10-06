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

//! Bounds-checked conversion of verified FlatGeobuf geometry arrays to ISO WKB.
//!
//! FlatBuffers verification checks vector storage, not coordinate/part semantics.
//! In particular the released crate's geo-traits converter assumes nonempty
//! points and matching XY/Z/M lengths. Check these contracts before indexing.
use datafusion_common::{Result, exec_datafusion_err, exec_err};
use flatgeobuf::{Geometry, GeometryType};

pub(crate) fn to_wkb(g: Geometry<'_>, kind: GeometryType, z: bool, m: bool) -> Result<Vec<u8>> {
    let mut out = vec![];
    write(&mut out, g, kind, z, m, 0)?;
    Ok(out)
}
fn u32_value(out: &mut Vec<u8>, n: usize) -> Result<()> {
    let n = u32::try_from(n).map_err(|_| exec_datafusion_err!("FlatGeobuf WKB count overflow"))?;
    out.extend_from_slice(&n.to_le_bytes());
    Ok(())
}
fn header(out: &mut Vec<u8>, kind: GeometryType, z: bool, m: bool) {
    out.push(1);
    let code = kind.0 as u32 + u32::from(z) * 1000 + u32::from(m) * 2000;
    out.extend_from_slice(&code.to_le_bytes());
}
fn coord(out: &mut Vec<u8>, g: Geometry<'_>, i: usize, z: bool, m: bool) {
    let xy = g.xy().unwrap();
    out.extend_from_slice(&xy.get(i * 2).to_le_bytes());
    out.extend_from_slice(&xy.get(i * 2 + 1).to_le_bytes());
    if z {
        out.extend_from_slice(&g.z().unwrap().get(i).to_le_bytes());
    }
    if m {
        out.extend_from_slice(&g.m().unwrap().get(i).to_le_bytes());
    }
}
fn coords(
    out: &mut Vec<u8>,
    g: Geometry<'_>,
    lo: usize,
    hi: usize,
    z: bool,
    m: bool,
) -> Result<()> {
    u32_value(out, hi - lo)?;
    for i in lo..hi {
        coord(out, g, i, z, m);
    }
    Ok(())
}
fn write(
    out: &mut Vec<u8>,
    g: Geometry<'_>,
    kind: GeometryType,
    z: bool,
    m: bool,
    depth: usize,
) -> Result<()> {
    if depth > 64 {
        return exec_err!("FlatGeobuf geometry nesting exceeds 64 levels");
    }
    let kind = if kind == GeometryType::Unknown {
        g.type_()
    } else {
        kind
    };
    if kind.0 == 0 || kind.0 > 7 {
        return exec_err!("Unsupported FlatGeobuf geometry type: {kind:?}");
    }
    let xy_len = g.xy().map_or(0, |v| v.len());
    if !xy_len.is_multiple_of(2) {
        return exec_err!("FlatGeobuf XY array must contain coordinate pairs");
    }
    let n = xy_len / 2;
    if (z && n > 0 && g.z().map_or(0, |v| v.len()) != n)
        || (m && n > 0 && g.m().map_or(0, |v| v.len()) != n)
    {
        return exec_err!("FlatGeobuf coordinate dimension lengths differ");
    }
    if (!z && g.z().is_some_and(|v| !v.is_empty())) || (!m && g.m().is_some_and(|v| !v.is_empty()))
    {
        return exec_err!("FlatGeobuf geometry dimensions differ from header");
    }
    header(out, kind, z, m);
    match kind {
        GeometryType::Point => {
            if n > 1 {
                return exec_err!("FlatGeobuf Point must have at most one coordinate");
            }
            if n == 0 {
                for _ in 0..2 + usize::from(z) + usize::from(m) {
                    out.extend_from_slice(&f64::NAN.to_le_bytes());
                }
            } else {
                coord(out, g, 0, z, m);
            }
        }
        GeometryType::LineString => coords(out, g, 0, n, z, m)?,
        GeometryType::MultiPoint => {
            u32_value(out, n)?;
            for i in 0..n {
                header(out, GeometryType::Point, z, m);
                coord(out, g, i, z, m);
            }
        }
        GeometryType::Polygon | GeometryType::MultiLineString => {
            let ends = if let Some(ends) = g.ends() {
                ends.iter().map(|v| v as usize).collect::<Vec<_>>()
            } else if n > 0 {
                vec![n]
            } else {
                vec![]
            };
            let mut lo = 0;
            for hi in &ends {
                if *hi < lo || *hi > n {
                    return exec_err!("Invalid FlatGeobuf ring/line boundary");
                }
                lo = *hi;
            }
            if lo != n {
                return exec_err!("FlatGeobuf ring/line boundaries omit coordinates");
            }
            u32_value(out, ends.len())?;
            lo = 0;
            for hi in ends {
                if kind == GeometryType::MultiLineString {
                    header(out, GeometryType::LineString, z, m);
                }
                coords(out, g, lo, hi, z, m)?;
                lo = hi;
            }
        }
        GeometryType::MultiPolygon | GeometryType::GeometryCollection => {
            if n > 0 {
                return exec_err!("FlatGeobuf multipart geometry has unexpected coordinates");
            }
            let parts = g.parts();
            u32_value(out, parts.map_or(0, |p| p.len()))?;
            if let Some(parts) = parts {
                for part in parts {
                    write(
                        out,
                        part,
                        if kind == GeometryType::MultiPolygon {
                            GeometryType::Polygon
                        } else {
                            GeometryType::Unknown
                        },
                        z,
                        m,
                        depth + 1,
                    )?;
                }
            }
        }
        _ => unreachable!(),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use flatgeobuf::{Feature, FeatureArgs, GeometryArgs, size_prefixed_root_as_feature};
    use std::str::FromStr;
    #[test]
    fn core_types_match_independent_wkb_writer() {
        let square = [0., 0., 1., 0., 1., 1., 0., 1., 0., 0.];
        for (kind, xy, ends, wkt) in [
            (GeometryType::Point, &[1., 2.][..], None, "POINT (1 2)"),
            (GeometryType::Point, &[][..], None, "POINT EMPTY"),
            (
                GeometryType::LineString,
                &[0., 0., 1., 1.][..],
                None,
                "LINESTRING (0 0,1 1)",
            ),
            (GeometryType::LineString, &[][..], None, "LINESTRING EMPTY"),
            (
                GeometryType::Polygon,
                &square[..],
                None,
                "POLYGON ((0 0,1 0,1 1,0 1,0 0))",
            ),
            (GeometryType::Polygon, &[][..], None, "POLYGON EMPTY"),
            (
                GeometryType::MultiPoint,
                &[0., 0., 1., 1.][..],
                None,
                "MULTIPOINT ((0 0),(1 1))",
            ),
            (GeometryType::MultiPoint, &[][..], None, "MULTIPOINT EMPTY"),
            (
                GeometryType::MultiLineString,
                &[0., 0., 1., 1., 2., 2., 3., 3.][..],
                Some(&[2u32, 4][..]),
                "MULTILINESTRING ((0 0,1 1),(2 2,3 3))",
            ),
            (
                GeometryType::MultiLineString,
                &[][..],
                None,
                "MULTILINESTRING EMPTY",
            ),
            (
                GeometryType::MultiPolygon,
                &square[..],
                None,
                "MULTIPOLYGON (((0 0,1 0,1 1,0 1,0 0)))",
            ),
            (
                GeometryType::GeometryCollection,
                &[1., 2.][..],
                None,
                "GEOMETRYCOLLECTION (POINT (1 2),LINESTRING EMPTY)",
            ),
        ] {
            let mut b = flatbuffers::FlatBufferBuilder::new();
            let xy = b.create_vector(xy);
            let ends = ends.map(|v| b.create_vector(v));
            let child = flatgeobuf::Geometry::create(
                &mut b,
                &GeometryArgs {
                    type_: if kind == GeometryType::MultiPolygon {
                        GeometryType::Polygon
                    } else if kind == GeometryType::GeometryCollection {
                        GeometryType::Point
                    } else {
                        kind
                    },
                    xy: Some(xy),
                    ends,
                    ..Default::default()
                },
            );
            let g =
                if kind == GeometryType::MultiPolygon || kind == GeometryType::GeometryCollection {
                    let mut children = vec![child];
                    if kind == GeometryType::GeometryCollection {
                        children.push(flatgeobuf::Geometry::create(
                            &mut b,
                            &GeometryArgs {
                                type_: GeometryType::LineString,
                                ..Default::default()
                            },
                        ));
                    }
                    let parts = b.create_vector(&children);
                    flatgeobuf::Geometry::create(
                        &mut b,
                        &GeometryArgs {
                            parts: Some(parts),
                            ..Default::default()
                        },
                    )
                } else {
                    child
                };
            let feat = Feature::create(
                &mut b,
                &FeatureArgs {
                    geometry: Some(g),
                    ..Default::default()
                },
            );
            b.finish_size_prefixed(feat, None);
            let g = size_prefixed_root_as_feature(b.finished_data())
                .unwrap()
                .geometry()
                .unwrap();
            let got = to_wkb(g, kind, false, false).unwrap();
            let mut expected = vec![];
            wkb::writer::write_geometry(
                &mut expected,
                &wkt::Wkt::<f64>::from_str(wkt).unwrap(),
                &Default::default(),
            )
            .unwrap();
            assert_eq!(got, expected, "{kind:?}");
            wkb::reader::read_wkb(&got).unwrap();
        }
    }
}
