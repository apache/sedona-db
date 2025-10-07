mod distance;
mod utils;
pub use distance::DistanceExt;
use geo_types::{Coord, CoordFloat, Point};

use crate::line_measures::distance::Distance;

/// Operations on the [Euclidean plane] measure distance with the pythagorean formula -
/// what you'd measure with a ruler.
///
/// If you have lon/lat points, use the [`Haversine`], [`Geodesic`], or other [metric spaces] -
/// Euclidean methods will give nonsense results.
///
/// If you wish to use Euclidean operations with lon/lat, the coordinates must first be transformed
/// using the [`Transform::transform`](crate::Transform::transform) / [`Transform::transform_crs_to_crs`](crate::Transform::transform_crs_to_crs) methods or their
/// immutable variants. Use of these requires the proj feature
///
/// [Euclidean plane]: https://en.wikipedia.org/wiki/Euclidean_plane
/// [`Transform`]: crate::Transform
/// [`Haversine`]: super::Haversine
/// [`Geodesic`]: super::Geodesic
/// [metric spaces]: super
pub struct Euclidean;

// ┌───────────────────────────┐
// │ Implementations for Coord │
// └───────────────────────────┘

impl<F: CoordFloat> Distance<F, Coord<F>, Coord<F>> for Euclidean {
    fn distance(&self, origin: Coord<F>, destination: Coord<F>) -> F {
        let delta = origin - destination;
        delta.x.hypot(delta.y)
    }
}

// ┌───────────────────────────┐
// │ Implementations for Point │
// └───────────────────────────┘

/// Calculate the Euclidean distance (a.k.a. pythagorean distance) between two Points
impl<F: CoordFloat> Distance<F, Point<F>, Point<F>> for Euclidean {
    /// Calculate the Euclidean distance (a.k.a. pythagorean distance) between two Points
    ///
    /// # Units
    /// - `origin`, `destination`: Point where the units of x/y represent non-angular units
    ///   — e.g. meters or miles, not lon/lat. For lon/lat points, use the
    ///   [`Haversine`] or [`Geodesic`] [metric spaces].
    /// - returns: distance in the same units as the `origin` and `destination` points
    ///
    /// # Example
    /// ```
    /// use geo::{Euclidean, Distance};
    /// use geo::Point;
    /// // web mercator
    /// let new_york_city = Point::new(-8238310.24, 4942194.78);
    /// // web mercator
    /// let london = Point::new(-14226.63, 6678077.70);
    /// let distance: f64 = Euclidean.distance(new_york_city, london);
    ///
    /// assert_eq!(
    ///     8_405_286., // meters in web mercator
    ///     distance.round()
    /// );
    /// ```
    ///
    /// [`Haversine`]: crate::line_measures::metric_spaces::Haversine
    /// [`Geodesic`]: crate::line_measures::metric_spaces::Geodesic
    /// [metric spaces]: crate::line_measures::metric_spaces
    fn distance(&self, origin: Point<F>, destination: Point<F>) -> F {
        self.distance(origin.0, destination.0)
    }
}

impl<F: CoordFloat> Distance<F, &Point<F>, &Point<F>> for Euclidean {
    fn distance(&self, origin: &Point<F>, destination: &Point<F>) -> F {
        self.distance(*origin, *destination)
    }
}
