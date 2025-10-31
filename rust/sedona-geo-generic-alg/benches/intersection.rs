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
use criterion::{criterion_group, criterion_main, Criterion};
use geo::Triangle;
use geo_traits::to_geo::ToGeoGeometry;
use geo_types::Geometry;
use sedona_geo_generic_alg::MultiPolygon;
use sedona_geo_generic_alg::{intersects::Intersects, Centroid};

#[path = "utils/wkb_util.rs"]
mod wkb_util;

fn multi_polygon_intersection(c: &mut Criterion) {
    let plot_polygons: MultiPolygon = sedona_testing::fixtures::nl_plots_wgs84();
    let zone_polygons: MultiPolygon = sedona_testing::fixtures::nl_zones();
    let plot_geoms: Vec<Geometry> = plot_polygons.into_iter().map(|p| p.into()).collect();
    let zone_geoms: Vec<Geometry> = zone_polygons.into_iter().map(|p| p.into()).collect();

    c.bench_function("MultiPolygon intersects", |bencher| {
        bencher.iter(|| {
            let mut intersects = 0;
            let mut non_intersects = 0;

            for a in &plot_geoms {
                for b in &zone_geoms {
                    if criterion::black_box(b.intersects(a)) {
                        intersects += 1;
                    } else {
                        non_intersects += 1;
                    }
                }
            }

            assert_eq!(intersects, 974);
            assert_eq!(non_intersects, 27782);
        });
    });

    c.bench_function("MultiPolygon intersects 2", |bencher| {
        bencher.iter(|| {
            let mut intersects = 0;
            let mut non_intersects = 0;

            for a in &plot_geoms {
                for b in &zone_geoms {
                    if criterion::black_box(a.intersects(b)) {
                        intersects += 1;
                    } else {
                        non_intersects += 1;
                    }
                }
            }

            assert_eq!(intersects, 974);
            assert_eq!(non_intersects, 27782);
        });
    });

    c.bench_function("MultiPolygon intersects geo", |bencher| {
        bencher.iter(|| {
            let mut intersects = 0;
            let mut non_intersects = 0;

            for a in &plot_geoms {
                for b in &zone_geoms {
                    if criterion::black_box(geo::Intersects::intersects(b, a)) {
                        intersects += 1;
                    } else {
                        non_intersects += 1;
                    }
                }
            }

            assert_eq!(intersects, 974);
            assert_eq!(non_intersects, 27782);
        });
    });

    c.bench_function("MultiPolygon intersects geo 2", |bencher| {
        bencher.iter(|| {
            let mut intersects = 0;
            let mut non_intersects = 0;

            for a in &plot_geoms {
                for b in &zone_geoms {
                    if criterion::black_box(geo::Intersects::intersects(a, b)) {
                        intersects += 1;
                    } else {
                        non_intersects += 1;
                    }
                }
            }

            assert_eq!(intersects, 974);
            assert_eq!(non_intersects, 27782);
        });
    });
}

fn multi_polygon_intersection_wkb(c: &mut Criterion) {
    let plot_polygons: MultiPolygon = sedona_testing::fixtures::nl_plots_wgs84();
    let zone_polygons: MultiPolygon = sedona_testing::fixtures::nl_zones();

    // Convert intersected polygons to WKB
    let mut plot_polygon_wkbs = Vec::new();
    let mut zone_polygon_wkbs = Vec::new();
    for plot_polygon in plot_polygons {
        plot_polygon_wkbs.push(wkb_util::geo_to_wkb(plot_polygon));
    }
    for zone_polygon in zone_polygons {
        zone_polygon_wkbs.push(wkb_util::geo_to_wkb(zone_polygon));
    }

    c.bench_function("MultiPolygon intersects wkb", |bencher| {
        bencher.iter(|| {
            let mut intersects = 0;
            let mut non_intersects = 0;

            for a in &plot_polygon_wkbs {
                for b in &zone_polygon_wkbs {
                    let a_geom = wkb::reader::read_wkb(a).unwrap(); // Skip padding
                    let b_geom = wkb::reader::read_wkb(b).unwrap(); // Skip padding
                    if criterion::black_box(b_geom.intersects(&a_geom)) {
                        intersects += 1;
                    } else {
                        non_intersects += 1;
                    }
                }
            }

            assert_eq!(intersects, 974);
            assert_eq!(non_intersects, 27782);
        });
    });
}

fn multi_polygon_intersection_wkb_aligned(c: &mut Criterion) {
    let plot_polygons: MultiPolygon = sedona_testing::fixtures::nl_plots_wgs84();
    let zone_polygons: MultiPolygon = sedona_testing::fixtures::nl_zones();

    // Convert intersected polygons to WKB
    let mut plot_polygon_wkbs = Vec::new();
    let mut zone_polygon_wkbs = Vec::new();
    for plot_polygon in plot_polygons {
        let mut wkb = vec![0, 0, 0]; // Add 3-byte padding
        wkb.extend_from_slice(&wkb_util::geo_to_wkb(plot_polygon));
        plot_polygon_wkbs.push(wkb);
    }
    for zone_polygon in zone_polygons {
        let mut wkb = vec![0, 0, 0]; // Add 3-byte padding
        wkb.extend_from_slice(&wkb_util::geo_to_wkb(zone_polygon));
        zone_polygon_wkbs.push(wkb);
    }

    c.bench_function("MultiPolygon intersects wkb aligned", |bencher| {
        bencher.iter(|| {
            let mut intersects = 0;
            let mut non_intersects = 0;

            for a in &plot_polygon_wkbs {
                for b in &zone_polygon_wkbs {
                    let a_geom = wkb::reader::read_wkb(&a[3..]).unwrap(); // Skip padding
                    let b_geom = wkb::reader::read_wkb(&b[3..]).unwrap(); // Skip padding
                    if criterion::black_box(b_geom.intersects(&a_geom)) {
                        intersects += 1;
                    } else {
                        non_intersects += 1;
                    }
                }
            }

            assert_eq!(intersects, 974);
            assert_eq!(non_intersects, 27782);
        });
    });
}

fn multi_polygon_intersection_wkb_conv(c: &mut Criterion) {
    let plot_polygons: MultiPolygon = sedona_testing::fixtures::nl_plots_wgs84();
    let zone_polygons: MultiPolygon = sedona_testing::fixtures::nl_zones();

    // Convert intersected polygons to WKB
    let mut plot_polygon_wkbs = Vec::new();
    let mut zone_polygon_wkbs = Vec::new();
    for plot_polygon in plot_polygons {
        plot_polygon_wkbs.push(wkb_util::geo_to_wkb(plot_polygon));
    }
    for zone_polygon in zone_polygons {
        zone_polygon_wkbs.push(wkb_util::geo_to_wkb(zone_polygon));
    }

    c.bench_function("MultiPolygon intersects wkb conv", |bencher| {
        bencher.iter(|| {
            let mut intersects = 0;
            let mut non_intersects = 0;

            for a in &plot_polygon_wkbs {
                for b in &zone_polygon_wkbs {
                    let a_geom = wkb::reader::read_wkb(a).unwrap();
                    let b_geom = wkb::reader::read_wkb(b).unwrap();
                    let a_geom = a_geom.to_geometry();
                    let b_geom = b_geom.to_geometry();
                    if criterion::black_box(b_geom.intersects(&a_geom)) {
                        intersects += 1;
                    } else {
                        non_intersects += 1;
                    }
                }
            }

            assert_eq!(intersects, 974);
            assert_eq!(non_intersects, 27782);
        });
    });
}

fn point_polygon_intersection(c: &mut Criterion) {
    let plot_polygons: MultiPolygon = sedona_testing::fixtures::nl_plots_wgs84();
    let zone_polygons: MultiPolygon = sedona_testing::fixtures::nl_zones();
    let plot_geoms: Vec<Geometry> = plot_polygons
        .into_iter()
        .map(|p| {
            let centroid = p.centroid().unwrap();
            centroid.into()
        })
        .collect();
    let zone_geoms: Vec<Geometry> = zone_polygons.into_iter().map(|p| p.into()).collect();

    c.bench_function("Point polygon intersects", |bencher| {
        bencher.iter(|| {
            for a in &plot_geoms {
                for b in &zone_geoms {
                    criterion::black_box(b.intersects(a));
                }
            }
        });
    });

    c.bench_function("Point polygon intersects geo", |bencher| {
        bencher.iter(|| {
            for a in &plot_geoms {
                for b in &zone_geoms {
                    criterion::black_box(geo::Intersects::intersects(b, a));
                }
            }
        });
    });
}

fn point_polygon_intersection_wkb(c: &mut Criterion) {
    let plot_polygons: MultiPolygon = sedona_testing::fixtures::nl_plots_wgs84();
    let zone_polygons: MultiPolygon = sedona_testing::fixtures::nl_zones();

    // Convert intersected polygons to WKB
    let mut plot_centroid_wkbs = Vec::new();
    let mut zone_polygon_wkbs = Vec::new();
    for plot_polygon in plot_polygons {
        let centroid = plot_polygon.centroid().unwrap();
        plot_centroid_wkbs.push(wkb_util::geo_to_wkb(centroid));
    }
    for zone_polygon in zone_polygons {
        zone_polygon_wkbs.push(wkb_util::geo_to_wkb(zone_polygon));
    }

    c.bench_function("Point polygon intersects wkb", |bencher| {
        bencher.iter(|| {
            for a in &plot_centroid_wkbs {
                for b in &zone_polygon_wkbs {
                    let a_geom = wkb::reader::read_wkb(a).unwrap();
                    let b_geom = wkb::reader::read_wkb(b).unwrap();
                    criterion::black_box(b_geom.intersects(&a_geom));
                }
            }
        });
    });
}

fn point_polygon_intersection_wkb_conv(c: &mut Criterion) {
    let plot_polygons: MultiPolygon = sedona_testing::fixtures::nl_plots_wgs84();
    let zone_polygons: MultiPolygon = sedona_testing::fixtures::nl_zones();

    // Convert intersected polygons to WKB
    let mut plot_centroid_wkbs = Vec::new();
    let mut zone_polygon_wkbs = Vec::new();
    for plot_polygon in plot_polygons {
        let centroid = plot_polygon.centroid().unwrap();
        plot_centroid_wkbs.push(wkb_util::geo_to_wkb(centroid));
    }
    for zone_polygon in zone_polygons {
        zone_polygon_wkbs.push(wkb_util::geo_to_wkb(zone_polygon));
    }

    c.bench_function("Point polygon intersects wkb conv", |bencher| {
        bencher.iter(|| {
            for a in &plot_centroid_wkbs {
                for b in &zone_polygon_wkbs {
                    let a_geom = wkb::reader::read_wkb(a).unwrap();
                    let b_geom = wkb::reader::read_wkb(b).unwrap();
                    let a_geom = a_geom.to_geometry();
                    let b_geom = b_geom.to_geometry();
                    criterion::black_box(b_geom.intersects(&a_geom));
                }
            }
        });
    });
}

fn rect_intersection(c: &mut Criterion) {
    use sedona_geo_generic_alg::algorithm::BoundingRect;
    use sedona_geo_generic_alg::Rect;
    let plot_bbox: Vec<Rect> = sedona_testing::fixtures::nl_plots_wgs84()
        .iter()
        .map(|plot| plot.bounding_rect().unwrap())
        .collect();
    let zone_bbox: Vec<Rect> = sedona_testing::fixtures::nl_zones()
        .iter()
        .map(|plot| plot.bounding_rect().unwrap())
        .collect();

    c.bench_function("Rect intersects", |bencher| {
        bencher.iter(|| {
            let mut intersects = 0;
            let mut non_intersects = 0;

            for a in &plot_bbox {
                for b in &zone_bbox {
                    if criterion::black_box(a.intersects(b)) {
                        intersects += 1;
                    } else {
                        non_intersects += 1;
                    }
                }
            }

            assert_eq!(intersects, 3054);
            assert_eq!(non_intersects, 25702);
        });
    });
}

fn point_rect_intersection(c: &mut Criterion) {
    use sedona_geo_generic_alg::algorithm::{BoundingRect, Centroid};
    use sedona_geo_generic_alg::geometry::{Point, Rect};
    let plot_centroids: Vec<Point> = sedona_testing::fixtures::nl_plots_wgs84()
        .iter()
        .map(|plot| plot.centroid().unwrap())
        .collect();
    let zone_bbox: Vec<Rect> = sedona_testing::fixtures::nl_zones()
        .iter()
        .map(|plot| plot.bounding_rect().unwrap())
        .collect();

    c.bench_function("Point intersects rect", |bencher| {
        bencher.iter(|| {
            let mut intersects = 0;
            let mut non_intersects = 0;

            for a in &plot_centroids {
                for b in &zone_bbox {
                    if criterion::black_box(a.intersects(b)) {
                        intersects += 1;
                    } else {
                        non_intersects += 1;
                    }
                }
            }

            assert_eq!(intersects, 2246);
            assert_eq!(non_intersects, 26510);
        });
    });
}

fn point_triangle_intersection(c: &mut Criterion) {
    use geo::algorithm::TriangulateEarcut;
    use sedona_geo_generic_alg::{Point, Triangle};
    let plot_centroids: Vec<Point> = sedona_testing::fixtures::nl_plots_wgs84()
        .iter()
        .map(|plot| plot.centroid().unwrap())
        .collect();
    let zone_triangles: Vec<Triangle> = sedona_testing::fixtures::nl_zones()
        .iter()
        .flat_map(|plot| plot.earcut_triangles_iter())
        .collect();

    c.bench_function("Point intersects triangle", |bencher| {
        bencher.iter(|| {
            let mut intersects = 0;
            let mut non_intersects = 0;

            for a in &plot_centroids {
                for b in &zone_triangles {
                    if criterion::black_box(a.intersects(b)) {
                        intersects += 1;
                    } else {
                        non_intersects += 1;
                    }
                }
            }

            assert_eq!(intersects, 533);
            assert_eq!(non_intersects, 5450151);
        });
    });

    c.bench_function("Triangle intersects point", |bencher| {
        let triangle = Triangle::from([(0., 0.), (10., 0.), (5., 10.)]);
        let point = Point::new(5., 5.);

        bencher.iter(|| {
            assert!(criterion::black_box(&triangle).intersects(criterion::black_box(&point)));
        });
    });

    c.bench_function("Triangle intersects point on edge", |bencher| {
        let triangle = Triangle::from([(0., 0.), (10., 0.), (6., 10.)]);
        let point = Point::new(3., 5.);

        bencher.iter(|| {
            assert!(criterion::black_box(&triangle).intersects(criterion::black_box(&point)));
        });
    });
}

fn linestring_polygon_intersection(c: &mut Criterion) {
    use geo::{coord, line_string, LineString, Polygon, Rect};
    c.bench_function("LineString above Polygon", |bencher| {
        let ls = line_string![
            coord! {x:0., y:1.},
            coord! {x:5., y:6.},
            coord! {x:10., y:1.}
        ];
        let poly = Polygon::new(
            line_string![
                coord! {x:0., y:0.},
                coord! {x:5., y:4.},
                coord! {x:10., y:0.}
            ],
            vec![],
        );

        bencher.iter(|| {
            assert!(!criterion::black_box(&ls).intersects(criterion::black_box(&poly)));
        });
    });
    c.bench_function("LineString above Triangle", |bencher| {
        let ls = line_string![
            coord! {x:0., y:1.},
            coord! {x:5., y:6.},
            coord! {x:10., y:1.}
        ];
        let poly = Triangle::new(
            coord! {x:0., y:0.},
            coord! {x:5., y:4.},
            coord! {x:10., y:0.},
        );

        bencher.iter(|| {
            assert!(!criterion::black_box(&ls).intersects(criterion::black_box(&poly)));
        });
    });
    c.bench_function("LineString around Rectangle", |bencher| {
        let ls = line_string![
            coord! {x:-1., y:-1.},
            coord! {x:-1., y:11.},
            coord! {x:11., y:11.}
        ];
        let poly = Rect::new(coord! {x:0., y:0.}, coord! {x:10., y:10.});

        bencher.iter(|| {
            assert!(!criterion::black_box(&ls).intersects(criterion::black_box(&poly)));
        });
    });

    c.bench_function("long disjoint ", |bencher| {
        let ls = LineString::from_iter((0..1000).map(|x| coord! {x:x as f64, y:x as f64}));
        let ln = (0..1000).map(|x| coord! {x:x as f64, y:(x-1) as f64});
        let k = vec![coord! {x:-5. ,y:-5. }].into_iter();
        let ext = ln.chain(k);

        let poly = Polygon::new(LineString::from_iter(ext), vec![]);

        bencher.iter(|| {
            assert!(!criterion::black_box(&ls).intersects(criterion::black_box(&poly)));
        });
    });

    c.bench_function("ls within poly ", |bencher| {
        let ls = line_string![
            coord! {x:1., y:1.},
            coord! {x:5., y:6.},
            coord! {x:9., y:1.}
        ];

        let poly: Polygon = Rect::new(coord! {x:0., y:0.}, coord! {x:10., y:10.}).into();

        bencher.iter(|| {
            assert!(criterion::black_box(&ls).intersects(criterion::black_box(&poly)));
        });
    });
    c.bench_function("ls within rect ", |bencher| {
        let ls = line_string![
            coord! {x:1., y:1.},
            coord! {x:5., y:6.},
            coord! {x:9., y:1.}
        ];

        let poly = Rect::new(coord! {x:0., y:0.}, coord! {x:10., y:10.});

        bencher.iter(|| {
            assert!(criterion::black_box(&ls).intersects(criterion::black_box(&poly)));
        });
    });
}

criterion_group! {
    name = bench_multi_polygons;
    config = Criterion::default().sample_size(10);
    targets = multi_polygon_intersection
}
criterion_group! {
    name = bench_multi_polygons_wkb;
    config = Criterion::default().sample_size(10);
    targets = multi_polygon_intersection_wkb
}
criterion_group! {
    name = bench_multi_polygons_wkb_aligned;
    config = Criterion::default().sample_size(10);
    targets = multi_polygon_intersection_wkb_aligned
}
criterion_group! {
    name = bench_multi_polygons_wkb_conv;
    config = Criterion::default().sample_size(10);
    targets = multi_polygon_intersection_wkb_conv
}

criterion_group!(bench_rects, rect_intersection);
criterion_group! {
    name = bench_point_rect;
    config = Criterion::default().sample_size(50);
    targets = point_rect_intersection
}
criterion_group! {
    name = bench_point_triangle;
    config = Criterion::default().sample_size(50);
    targets = point_triangle_intersection
}

criterion_group! {
    name = bench_point_polygon;
    config = Criterion::default().sample_size(50);
    targets = point_polygon_intersection
}
criterion_group! {
    name = bench_point_polygon_wkb;
    config = Criterion::default().sample_size(50);
    targets = point_polygon_intersection_wkb
}
criterion_group! {
    name = bench_point_polygon_wkb_conv;
    config = Criterion::default().sample_size(50);
    targets = point_polygon_intersection_wkb_conv
}

criterion_group! { bench_linestring_poly,linestring_polygon_intersection}

criterion_main!(
    bench_multi_polygons,
    bench_multi_polygons_wkb,
    bench_multi_polygons_wkb_aligned,
    bench_multi_polygons_wkb_conv,
    bench_rects,
    bench_linestring_poly,
    bench_point_rect,
    bench_point_triangle,
    bench_point_polygon,
    bench_point_polygon_wkb,
    bench_point_polygon_wkb_conv
);
