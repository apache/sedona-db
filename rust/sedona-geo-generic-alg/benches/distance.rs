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
use geo::{Distance as GeoDistance, Euclidean};
use sedona_geo_generic_alg::algorithm::line_measures::DistanceExt;
use sedona_geo_generic_alg::{coord, LineString, MultiPolygon, Point, Polygon};

#[path = "utils/wkb_util.rs"]
mod wkb_util;

// Helper function to create complex polygons with many vertices for stress testing
fn create_complex_polygon(
    center_x: f64,
    center_y: f64,
    radius: f64,
    num_vertices: usize,
) -> Polygon<f64> {
    let mut vertices = Vec::with_capacity(num_vertices + 1);

    for i in 0..num_vertices {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / num_vertices as f64;
        // Add some variation to make it non-regular
        let r = radius * (1.0 + 0.1 * (i as f64 * 0.3).sin());
        let x = center_x + r * angle.cos();
        let y = center_y + r * angle.sin();
        vertices.push(coord!(x: x, y: y));
    }

    // Close the polygon
    vertices.push(vertices[0]);

    Polygon::new(LineString::from(vertices), vec![])
}

// Helper function to create multipolygons for testing iteration overhead
fn create_multipolygon(num_polygons: usize) -> MultiPolygon<f64> {
    let mut polygons = Vec::with_capacity(num_polygons);

    for i in 0..num_polygons {
        let offset = i as f64 * 50.0;
        let poly = Polygon::new(
            LineString::from(vec![
                coord!(x: offset, y: offset),
                coord!(x: offset + 30.0, y: offset),
                coord!(x: offset + 30.0, y: offset + 30.0),
                coord!(x: offset, y: offset + 30.0),
                coord!(x: offset, y: offset),
            ]),
            vec![],
        );
        polygons.push(poly);
    }

    MultiPolygon::new(polygons)
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("distance_point_to_point", |bencher| {
        let p1 = Point::new(0.0, 0.0);
        let p2 = Point::new(100.0, 100.0);

        bencher.iter(|| {
            criterion::black_box(criterion::black_box(&p1).distance_ext(criterion::black_box(&p2)));
        });
    });

    c.bench_function("distance_linestring_to_linestring", |bencher| {
        let ls1 = sedona_testing::fixtures::norway_main::<f64>();
        let ls2 = LineString::from(vec![
            coord!(x: 100.0, y: 100.0),
            coord!(x: 200.0, y: 200.0),
            coord!(x: 300.0, y: 300.0),
        ]);

        bencher.iter(|| {
            criterion::black_box(
                criterion::black_box(&ls1).distance_ext(criterion::black_box(&ls2)),
            );
        });
    });

    c.bench_function("distance_polygon_to_polygon", |bencher| {
        let poly1 = Polygon::new(
            LineString::from(vec![
                coord!(x: 0.0, y: 0.0),
                coord!(x: 100.0, y: 0.0),
                coord!(x: 100.0, y: 100.0),
                coord!(x: 0.0, y: 100.0),
                coord!(x: 0.0, y: 0.0),
            ]),
            vec![],
        );
        let poly2 = Polygon::new(
            LineString::from(vec![
                coord!(x: 200.0, y: 200.0),
                coord!(x: 300.0, y: 200.0),
                coord!(x: 300.0, y: 300.0),
                coord!(x: 200.0, y: 300.0),
                coord!(x: 200.0, y: 200.0),
            ]),
            vec![],
        );

        bencher.iter(|| {
            criterion::black_box(
                criterion::black_box(&poly1).distance_ext(criterion::black_box(&poly2)),
            );
        });
    });

    c.bench_function("distance_wkb_point_to_point", |bencher| {
        let p1 = Point::new(0.0, 0.0);
        let p2 = Point::new(100.0, 100.0);
        let wkb_bytes1 = wkb_util::geo_to_wkb(p1);
        let wkb_bytes2 = wkb_util::geo_to_wkb(p2);

        bencher.iter(|| {
            let wkb_geom1 = wkb::reader::read_wkb(&wkb_bytes1).unwrap();
            let wkb_geom2 = wkb::reader::read_wkb(&wkb_bytes2).unwrap();
            criterion::black_box(wkb_geom1.distance_ext(&wkb_geom2));
        });
    });

    c.bench_function("distance_wkb_linestring_to_linestring", |bencher| {
        let ls1 = sedona_testing::fixtures::norway_main::<f64>();
        let ls2 = LineString::from(vec![
            coord!(x: 100.0, y: 100.0),
            coord!(x: 200.0, y: 200.0),
            coord!(x: 300.0, y: 300.0),
        ]);
        let wkb_bytes1 = wkb_util::geo_to_wkb(ls1);
        let wkb_bytes2 = wkb_util::geo_to_wkb(ls2);

        bencher.iter(|| {
            let wkb_geom1 = wkb::reader::read_wkb(&wkb_bytes1).unwrap();
            let wkb_geom2 = wkb::reader::read_wkb(&wkb_bytes2).unwrap();
            criterion::black_box(wkb_geom1.distance_ext(&wkb_geom2));
        });
    });

    c.bench_function("distance_multipolygon_to_multipolygon", |bencher| {
        let poly1 = Polygon::new(
            LineString::from(vec![
                coord!(x: 0.0, y: 0.0),
                coord!(x: 50.0, y: 0.0),
                coord!(x: 50.0, y: 50.0),
                coord!(x: 0.0, y: 50.0),
                coord!(x: 0.0, y: 0.0),
            ]),
            vec![],
        );
        let poly2 = Polygon::new(
            LineString::from(vec![
                coord!(x: 60.0, y: 60.0),
                coord!(x: 110.0, y: 60.0),
                coord!(x: 110.0, y: 110.0),
                coord!(x: 60.0, y: 110.0),
                coord!(x: 60.0, y: 60.0),
            ]),
            vec![],
        );
        let mp1 = MultiPolygon::new(vec![poly1.clone(), poly1]);
        let mp2 = MultiPolygon::new(vec![poly2.clone(), poly2]);

        bencher.iter(|| {
            criterion::black_box(
                criterion::black_box(&mp1).distance_ext(criterion::black_box(&mp2)),
            );
        });
    });

    c.bench_function("distance_concrete_point_to_point", |bencher| {
        let p1 = Point::new(0.0, 0.0);
        let p2 = Point::new(100.0, 100.0);

        bencher.iter(|| {
            criterion::black_box(
                Euclidean.distance(criterion::black_box(p1), criterion::black_box(p2)),
            );
        });
    });

    c.bench_function("distance_concrete_linestring_to_linestring", |bencher| {
        let ls1 = sedona_testing::fixtures::norway_main::<f64>();
        let ls2 = LineString::from(vec![
            coord!(x: 100.0, y: 100.0),
            coord!(x: 200.0, y: 200.0),
            coord!(x: 300.0, y: 300.0),
        ]);

        bencher.iter(|| {
            criterion::black_box(
                geo::Euclidean.distance(criterion::black_box(&ls1), criterion::black_box(&ls2)),
            );
        });
    });

    c.bench_function("distance_cross_type_point_to_linestring", |bencher| {
        let point = Point::new(50.0, 50.0);
        let linestring = LineString::from(vec![
            coord!(x: 0.0, y: 0.0),
            coord!(x: 100.0, y: 100.0),
            coord!(x: 200.0, y: 0.0),
        ]);

        bencher.iter(|| {
            criterion::black_box(Euclidean.distance(
                criterion::black_box(&point),
                criterion::black_box(&linestring),
            ));
        });
    });

    c.bench_function("distance_cross_type_linestring_to_polygon", |bencher| {
        let linestring =
            LineString::from(vec![coord!(x: -50.0, y: 50.0), coord!(x: 150.0, y: 50.0)]);
        let polygon = Polygon::new(
            LineString::from(vec![
                coord!(x: 0.0, y: 0.0),
                coord!(x: 100.0, y: 0.0),
                coord!(x: 100.0, y: 100.0),
                coord!(x: 0.0, y: 100.0),
                coord!(x: 0.0, y: 0.0),
            ]),
            vec![],
        );

        bencher.iter(|| {
            criterion::black_box(Euclidean.distance(
                criterion::black_box(&linestring),
                criterion::black_box(&polygon),
            ));
        });
    });

    c.bench_function("distance_cross_type_point_to_polygon", |bencher| {
        let point = Point::new(150.0, 50.0);
        let polygon = Polygon::new(
            LineString::from(vec![
                coord!(x: 0.0, y: 0.0),
                coord!(x: 100.0, y: 0.0),
                coord!(x: 100.0, y: 100.0),
                coord!(x: 0.0, y: 100.0),
                coord!(x: 0.0, y: 0.0),
            ]),
            vec![],
        );

        bencher.iter(|| {
            criterion::black_box(
                Euclidean.distance(criterion::black_box(&point), criterion::black_box(&polygon)),
            );
        });
    });

    // ┌────────────────────────────────────────────────────────────┐
    // │ Targeted Performance Benchmarks: Generic vs Concrete      │
    // └────────────────────────────────────────────────────────────┘

    c.bench_function(
        "generic_vs_concrete_polygon_containment_simple",
        |bencher| {
            // Simple polygon-to-polygon distance (no holes, no containment)
            let poly1 = Polygon::new(
                LineString::from(vec![
                    coord!(x: 0.0, y: 0.0),
                    coord!(x: 50.0, y: 0.0),
                    coord!(x: 50.0, y: 50.0),
                    coord!(x: 0.0, y: 50.0),
                    coord!(x: 0.0, y: 0.0),
                ]),
                vec![],
            );
            let poly2 = Polygon::new(
                LineString::from(vec![
                    coord!(x: 100.0, y: 100.0),
                    coord!(x: 150.0, y: 100.0),
                    coord!(x: 150.0, y: 150.0),
                    coord!(x: 100.0, y: 150.0),
                    coord!(x: 100.0, y: 100.0),
                ]),
                vec![],
            );

            bencher.iter(|| {
                // Generic implementation
                criterion::black_box(
                    criterion::black_box(&poly1).distance_ext(criterion::black_box(&poly2)),
                );
            });
        },
    );

    c.bench_function(
        "concrete_vs_generic_polygon_containment_simple",
        |bencher| {
            let poly1 = Polygon::new(
                LineString::from(vec![
                    coord!(x: 0.0, y: 0.0),
                    coord!(x: 50.0, y: 0.0),
                    coord!(x: 50.0, y: 50.0),
                    coord!(x: 0.0, y: 50.0),
                    coord!(x: 0.0, y: 0.0),
                ]),
                vec![],
            );
            let poly2 = Polygon::new(
                LineString::from(vec![
                    coord!(x: 100.0, y: 100.0),
                    coord!(x: 150.0, y: 100.0),
                    coord!(x: 150.0, y: 150.0),
                    coord!(x: 100.0, y: 150.0),
                    coord!(x: 100.0, y: 100.0),
                ]),
                vec![],
            );

            bencher.iter(|| {
                // Concrete implementation
                criterion::black_box(
                    Euclidean.distance(criterion::black_box(&poly1), criterion::black_box(&poly2)),
                );
            });
        },
    );

    c.bench_function("generic_polygon_with_holes_distance", |bencher| {
        // Polygon with holes - this triggers the containment check and temporary object creation
        let outer = LineString::from(vec![
            coord!(x: 0.0, y: 0.0),
            coord!(x: 200.0, y: 0.0),
            coord!(x: 200.0, y: 200.0),
            coord!(x: 0.0, y: 200.0),
            coord!(x: 0.0, y: 0.0),
        ]);
        let hole = LineString::from(vec![
            coord!(x: 50.0, y: 50.0),
            coord!(x: 150.0, y: 50.0),
            coord!(x: 150.0, y: 150.0),
            coord!(x: 50.0, y: 150.0),
            coord!(x: 50.0, y: 50.0),
        ]);
        let poly_with_hole = Polygon::new(outer, vec![hole]);

        // Small polygon that might be inside the hole (triggers containment logic)
        let small_poly = Polygon::new(
            LineString::from(vec![
                coord!(x: 75.0, y: 75.0),
                coord!(x: 125.0, y: 75.0),
                coord!(x: 125.0, y: 125.0),
                coord!(x: 75.0, y: 125.0),
                coord!(x: 75.0, y: 75.0),
            ]),
            vec![],
        );

        bencher.iter(|| {
            criterion::black_box(
                criterion::black_box(&poly_with_hole)
                    .distance_ext(criterion::black_box(&small_poly)),
            );
        });
    });

    c.bench_function("concrete_polygon_with_holes_distance", |bencher| {
        let outer = LineString::from(vec![
            coord!(x: 0.0, y: 0.0),
            coord!(x: 200.0, y: 0.0),
            coord!(x: 200.0, y: 200.0),
            coord!(x: 0.0, y: 200.0),
            coord!(x: 0.0, y: 0.0),
        ]);
        let hole = LineString::from(vec![
            coord!(x: 50.0, y: 50.0),
            coord!(x: 150.0, y: 50.0),
            coord!(x: 150.0, y: 150.0),
            coord!(x: 50.0, y: 150.0),
            coord!(x: 50.0, y: 50.0),
        ]);
        let poly_with_hole = Polygon::new(outer, vec![hole]);
        let small_poly = Polygon::new(
            LineString::from(vec![
                coord!(x: 75.0, y: 75.0),
                coord!(x: 125.0, y: 75.0),
                coord!(x: 125.0, y: 125.0),
                coord!(x: 75.0, y: 125.0),
                coord!(x: 75.0, y: 75.0),
            ]),
            vec![],
        );

        bencher.iter(|| {
            criterion::black_box(Euclidean.distance(
                criterion::black_box(&poly_with_hole),
                criterion::black_box(&small_poly),
            ));
        });
    });

    c.bench_function("generic_complex_polygon_distance", |bencher| {
        // Complex polygons with many vertices - stress test for temporary object creation
        let complex_poly1 = create_complex_polygon(0.0, 0.0, 50.0, 20); // 20 vertices
        let complex_poly2 = create_complex_polygon(100.0, 100.0, 30.0, 15); // 15 vertices

        bencher.iter(|| {
            criterion::black_box(
                criterion::black_box(&complex_poly1)
                    .distance_ext(criterion::black_box(&complex_poly2)),
            );
        });
    });

    c.bench_function("concrete_complex_polygon_distance", |bencher| {
        let complex_poly1 = create_complex_polygon(0.0, 0.0, 50.0, 20);
        let complex_poly2 = create_complex_polygon(100.0, 100.0, 30.0, 15);

        bencher.iter(|| {
            criterion::black_box(Euclidean.distance(
                criterion::black_box(&complex_poly1),
                criterion::black_box(&complex_poly2),
            ));
        });
    });

    c.bench_function("generic_linestring_to_polygon_intersecting", |bencher| {
        // LineString that intersects polygon - tests early exit performance
        let polygon = Polygon::new(
            LineString::from(vec![
                coord!(x: 0.0, y: 0.0),
                coord!(x: 100.0, y: 0.0),
                coord!(x: 100.0, y: 100.0),
                coord!(x: 0.0, y: 100.0),
                coord!(x: 0.0, y: 0.0),
            ]),
            vec![],
        );
        let intersecting_linestring = LineString::from(vec![
            coord!(x: -50.0, y: 50.0),
            coord!(x: 150.0, y: 50.0), // Crosses through the polygon
        ]);

        bencher.iter(|| {
            criterion::black_box(
                criterion::black_box(&intersecting_linestring)
                    .distance_ext(criterion::black_box(&polygon)),
            );
        });
    });

    c.bench_function("concrete_linestring_to_polygon_intersecting", |bencher| {
        let polygon = Polygon::new(
            LineString::from(vec![
                coord!(x: 0.0, y: 0.0),
                coord!(x: 100.0, y: 0.0),
                coord!(x: 100.0, y: 100.0),
                coord!(x: 0.0, y: 100.0),
                coord!(x: 0.0, y: 0.0),
            ]),
            vec![],
        );
        let intersecting_linestring =
            LineString::from(vec![coord!(x: -50.0, y: 50.0), coord!(x: 150.0, y: 50.0)]);

        bencher.iter(|| {
            criterion::black_box(Euclidean.distance(
                criterion::black_box(&intersecting_linestring),
                criterion::black_box(&polygon),
            ));
        });
    });

    c.bench_function("generic_multipolygon_distance_overhead", |bencher| {
        // Test multipolygon distance to measure iterator and temporary object overhead
        let mp1 = create_multipolygon(5); // 5 polygons
        let mp2 = create_multipolygon(3); // 3 polygons

        bencher.iter(|| {
            criterion::black_box(
                criterion::black_box(&mp1).distance_ext(criterion::black_box(&mp2)),
            );
        });
    });

    c.bench_function("concrete_multipolygon_distance_overhead", |bencher| {
        let mp1 = create_multipolygon(5);
        let mp2 = create_multipolygon(3);

        bencher.iter(|| {
            criterion::black_box(
                Euclidean.distance(criterion::black_box(&mp1), criterion::black_box(&mp2)),
            );
        });
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
