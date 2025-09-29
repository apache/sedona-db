use criterion::{criterion_group, criterion_main};
use geo::{LineString, Point};
use sedona_geos::wkb_to_geos::GEOSWkbFactory;
use wkb::Endianness;

fn generate_wkb_linestring(num_points: usize, endianness: Endianness) -> Vec<u8> {
    let mut points = Vec::new();
    for i in 0..num_points {
        points.push(Point::new(i as f64, i as f64));
    }
    let linestring = LineString::from(points);
    let mut buffer = Vec::new();
    wkb::writer::write_geometry(
        &mut buffer,
        &linestring,
        &wkb::writer::WriteOptions { endianness },
    )
    .unwrap();
    buffer
}

fn bench_parse(c: &mut criterion::Criterion) {
    for num_points in [4, 10, 100, 500, 1000] {
        for endianness in [Endianness::BigEndian, Endianness::LittleEndian] {
            let wkb_buf = generate_wkb_linestring(num_points, endianness);
            let wkb = wkb::reader::read_wkb(&wkb_buf).unwrap();
            let endianness_name: &str = match endianness {
                Endianness::BigEndian => "big endian",
                Endianness::LittleEndian => "little endian",
            };

            c.bench_function(
                &format!(
                    "convert linestring containing {num_points} points using to_geos ({endianness_name})"
                ),
                |b| {
                    let factory = GEOSWkbFactory::new();
                    b.iter(|| {
                        let g = factory.create(&wkb).unwrap();
                        criterion::black_box(g);
                    });
                },
            );

            c.bench_function(
                &format!(
                    "convert linestring containing {num_points} points using geos wkb parser ({endianness_name})"
                ),
                |b| {
                    b.iter(|| {
                        let g = geos::Geometry::new_from_wkb(wkb.buf()).unwrap();
                        criterion::black_box(g);
                    });
                },
            );
        }
    }
}

criterion_group!(benches, bench_parse);
criterion_main!(benches);
