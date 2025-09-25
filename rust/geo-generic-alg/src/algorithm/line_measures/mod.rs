mod distance;
pub use distance::{Distance, DistanceExt};

mod length;
pub use length::LengthMeasurableExt;

pub mod metric_spaces;
pub use metric_spaces::Euclidean;
