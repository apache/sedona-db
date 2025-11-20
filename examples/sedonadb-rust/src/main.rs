use datafusion::{common::Result, prelude::*};
use sedona::context::{SedonaContext, SedonaDataFrame};

#[tokio::main]
async fn main() -> Result<()> {
    let ctx = SedonaContext::new_local_interactive().await?;
    let url = "https://raw.githubusercontent.com/geoarrow/geoarrow-data/v0.2.0/natural-earth/files/natural-earth_cities_geo.parquet";
    let df = ctx.read_parquet(url, Default::default()).await?;
    let output = df
        .sort_by(vec![col("name")])?
        .show_sedona(&ctx, Some(5), Default::default())
        .await?;
    println!("{output}");
    Ok(())
}
