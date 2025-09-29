use criterion::{criterion_group, criterion_main, Criterion};
use geo_generic_alg::Area;
use geo_generic_alg::Polygon;
use geo_traits::to_geo::ToGeoGeometry;

#[path = "utils/wkb_util.rs"]
mod wkb_util;

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("area_generic_f32", |bencher| {
        let norway = geo_test_fixtures::norway_main::<f32>();
        let polygon = Polygon::new(norway, vec![]);

        bencher.iter(|| {
            criterion::black_box(criterion::black_box(&polygon).signed_area());
        });
    });

    c.bench_function("area_generic", |bencher| {
        let norway = geo_test_fixtures::norway_main::<f64>();
        let polygon = Polygon::new(norway, vec![]);

        bencher.iter(|| {
            criterion::black_box(criterion::black_box(&polygon).signed_area());
        });
    });

    c.bench_function("area_geo_f32", |bencher| {
        let norway = geo_test_fixtures::norway_main::<f32>();
        let polygon = Polygon::new(norway, vec![]);

        bencher.iter(|| {
            criterion::black_box(geo::Area::signed_area(criterion::black_box(&polygon)));
        });
    });

    c.bench_function("area_geo", |bencher| {
        let norway = geo_test_fixtures::norway_main::<f64>();
        let polygon = Polygon::new(norway, vec![]);

        bencher.iter(|| {
            criterion::black_box(geo::Area::signed_area(criterion::black_box(&polygon)));
        });
    });

    c.bench_function("area_wkb", |bencher| {
        let norway = geo_test_fixtures::norway_main::<f64>();
        let polygon = Polygon::new(norway, vec![]);
        let wkb_bytes = wkb_util::geo_to_wkb(polygon);

        bencher.iter(|| {
            let wkb_geom = wkb::reader::read_wkb(&wkb_bytes).unwrap();
            criterion::black_box(wkb_geom.signed_area());
        });
    });

    c.bench_function("area_wkb_convert", |bencher| {
        let norway = geo_test_fixtures::norway_main::<f64>();
        let polygon = Polygon::new(norway, vec![]);
        let wkb_bytes = wkb_util::geo_to_wkb(polygon);

        bencher.iter(|| {
            let wkb_geom = wkb::reader::read_wkb(&wkb_bytes).unwrap();
            let geom = wkb_geom.to_geometry();
            criterion::black_box(geom.signed_area());
        });
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
