use datafusion::assert_batches_eq;
use sedona::context::SedonaContext;

#[tokio::test]
async fn select_one_returns_single_row() {
    let ctx = SedonaContext::new();

    let df = ctx
        .sql("SELECT 1 AS one")
        .await
        .expect("SQL execution should succeed");

    let batches = df
        .collect()
        .await
        .expect("Collecting results should succeed");

    assert_batches_eq!(
        ["+-----+", "| one |", "+-----+", "|   1 |", "+-----+",],
        &batches
    );
}
