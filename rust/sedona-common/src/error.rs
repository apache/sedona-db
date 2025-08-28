/// Macro to create Sedona Internal Error that avoids the misleading error message from
/// DataFusionError::Internal.
#[macro_export]
macro_rules! sedona_internal_err {
    ($($args:expr),*) => {{
        let msg = std::format!(
            "SedonaDB internal error: {}{}.\nThis issue was likely caused by a bug in SedonaDB's code. \
            Please help us to resolve this by filing a bug report in our issue tracker: \
            https://github.com/apache/sedona-db/issues",
            std::format!($($args),*),
            datafusion_common::DataFusionError::get_back_trace(),
        );
        // We avoid using Internal to avoid the message suggesting it's internal to DataFusion
        Err(datafusion_common::DataFusionError::External(msg.into()))
    }};
}

#[cfg(test)]
mod tests {
    use datafusion_common::DataFusionError;

    #[test]
    fn test_sedona_internal_err() {
        let result: Result<(), DataFusionError> = sedona_internal_err!("Test error: {}", "details");
        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_string = err.to_string();
        assert!(err_string.contains("SedonaDB internal error: Test error: details"));

        // Ensure the message doesn't contain the 'DataFusion'-specific messages
        assert!(!err_string.contains("DataFusion's code"));
        assert!(!err_string.contains("https://github.com/apache/datafusion/issues"));
    }
}
