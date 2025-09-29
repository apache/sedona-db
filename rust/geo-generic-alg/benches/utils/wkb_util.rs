pub fn geo_to_wkb<G>(geo: G) -> Vec<u8>
where
    G: Into<geo::Geometry>,
{
    let geom = geo.into();
    let mut out: Vec<u8> = vec![];
    wkb::writer::write_geometry(&mut out, &geom, &wkb::writer::WriteOptions::default()).unwrap();
    out
}
