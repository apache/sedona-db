/// Calculate the minimum distance between two geometries.
pub trait Distance<F, Origin, Destination> {
    /// Note that not all implementations support all geometry combinations, but at least `Point` to `Point`
    /// is supported.
    /// See [specific implementations](#implementers) for details.
    ///
    /// # Units
    ///
    /// - `origin`, `destination`: geometry where the units of x/y depend on the trait implementation.
    /// - returns: depends on the trait implementation.
    fn distance(&self, origin: Origin, destination: Destination) -> F;
}

// Re-export the DistanceExt trait from the refactored euclidean metric space
pub use super::metric_spaces::euclidean::DistanceExt;
