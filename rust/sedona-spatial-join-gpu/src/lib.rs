// Module declarations
mod evaluated_batch;
mod operand_evaluator;

mod build_index;

pub mod config;
pub mod exec;
mod index;
pub mod spatial_predicate;
pub mod stream;
pub mod utils;

// Re-exports for convenience
pub use config::GpuSpatialJoinConfig;
pub use datafusion::logical_expr::JoinType;
pub use exec::GpuSpatialJoinExec;
pub use stream::GpuSpatialJoinStream;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("GPU initialization error: {0}")]
    GpuInit(String),

    #[error("DataFusion error: {0}")]
    DataFusion(#[from] datafusion::error::DataFusionError),

    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    #[error("GPU spatial operation error: {0}")]
    GpuSpatial(String),
}

pub type Result<T> = std::result::Result<T, Error>;
