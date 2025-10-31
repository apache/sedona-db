// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
use std::fmt::Display;
use std::sync::OnceLock;

use crate::{error::TgError, tg_bindgen::*};

/// Index types available from tg
///
/// These index types are used to inform the internal index used in geometries
/// to accelerate repeated operations.
#[derive(Debug, Clone, Copy, Default)]
pub enum IndexType {
    /// Do not build an index
    Unindexed,
    /// Use the statically set default
    #[default]
    Default,
    /// Use natural indexing
    Natural,
    /// Use y-stripes indexing
    YStripes,
}

impl IndexType {
    fn to_tg(self) -> tg_index {
        match self {
            IndexType::Unindexed => tg_index_TG_NONE,
            IndexType::Default => tg_index_TG_DEFAULT,
            IndexType::Natural => tg_index_TG_NATURAL,
            IndexType::YStripes => tg_index_TG_YSTRIPES,
        }
    }
}

/// tg Geometry Class
///
/// The Geom is a generic geometry variant like a GeometryTrait, GEOSGeometry,
/// or Wkb. This class wraps a pinned, heap-allocated tg_geom.
pub struct Geom {
    inner: *mut tg_geom,
}

/// Safety: tg is thread-safe, so we mark it as Send + Sync.
unsafe impl Send for Geom {}
unsafe impl Sync for Geom {}

impl Drop for Geom {
    fn drop(&mut self) {
        if self.inner.is_null() {
            return;
        }

        unsafe { tg_geom_free(self.inner) };
    }
}

impl Geom {
    /// Create a [Geom] from well-known text
    pub fn parse_wkt(wkt: &str, index: IndexType) -> Result<Self, TgError> {
        unsafe {
            let inner = tg_parse_wktn_ix(wkt.as_ptr() as _, wkt.len(), index.to_tg());
            Self::try_new(inner)
        }
    }

    /// Create a [Geom] from well-known binary
    pub fn parse_wkb(wkb: &[u8], index: IndexType) -> Result<Self, TgError> {
        unsafe {
            let inner = tg_parse_wkb_ix(wkb.as_ptr(), wkb.len(), index.to_tg());
            Self::try_new(inner)
        }
    }

    unsafe fn try_new(inner: *mut tg_geom) -> Result<Self, TgError> {
        if inner.is_null() {
            return Err(TgError::Memory);
        }

        let geom = Self { inner };
        let err = tg_geom_error(geom.inner);
        if err.is_null() {
            return Ok(geom);
        }

        let c_str = std::ffi::CStr::from_ptr(err);
        Err(TgError::Invalid(c_str.to_string_lossy().into_owned()))
    }

    /// Write this [Geom] as well-known binary into out
    ///
    /// Callers must check the return size to ensure that all bytes were written.
    /// Use [Self::to_wkb] for a slower but more convenient way to obtain the result.
    pub fn write_wkb(&self, out: &mut [u8]) -> usize {
        unsafe { tg_geom_wkb(self.inner, out.as_mut_ptr(), out.len()) }
    }

    /// Write this [Geom] as well-known text into out
    ///
    /// Callers must check the return size to ensure that all bytes were written.
    /// Use [Self::to_wkt] for a slower but more convenient way to obtain the result.
    pub fn write_wkt(&self, out: &mut [u8]) -> usize {
        unsafe { tg_geom_wkt(self.inner, out.as_mut_ptr() as _, out.len()) }
    }

    /// Write this [Geom] to well-known binary as a [Vec]
    #[allow(clippy::uninit_vec)]
    pub fn to_wkb(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let size_required = self.write_wkb(&mut []);
        out.reserve(size_required);
        unsafe {
            out.set_len(size_required);
            self.write_wkb(&mut out);
        }
        out
    }

    /// Write this [Geom] to well-known binary as a [String]
    pub fn to_wkt(&self) -> String {
        let mut out = String::new();
        let size_required = self.write_wkt(&mut []);
        out.reserve(size_required + 1);
        unsafe {
            out.as_mut_vec().set_len(size_required + 1);
            self.write_wkt(out.as_mut_vec());
            out.as_mut_vec().set_len(size_required);
        }
        out
    }

    pub fn memsize(&self) -> usize {
        unsafe { tg_geom_memsize(self.inner) }
    }
}

impl Display for Geom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_wkt())
    }
}

/// Trait to make operations generic over binary predicates
pub trait BinaryPredicate: std::fmt::Debug + Default {
    fn evaluate(lhs: &Geom, rhs: &Geom) -> bool;
}

/// Check for topologicall equal geometries
#[derive(Debug, Default)]
pub struct Equals {}
impl BinaryPredicate for Equals {
    fn evaluate(lhs: &Geom, rhs: &Geom) -> bool {
        unsafe { tg_geom_equals(lhs.inner, rhs.inner) }
    }
}

/// Check if two geometries intersect
#[derive(Debug, Default)]
pub struct Intersects {}
impl BinaryPredicate for Intersects {
    fn evaluate(lhs: &Geom, rhs: &Geom) -> bool {
        unsafe { tg_geom_intersects(lhs.inner, rhs.inner) }
    }
}

/// Check if two geometries do not intersect
#[derive(Debug, Default)]
pub struct Disjoint {}
impl BinaryPredicate for Disjoint {
    fn evaluate(lhs: &Geom, rhs: &Geom) -> bool {
        unsafe { tg_geom_disjoint(lhs.inner, rhs.inner) }
    }
}

/// Check if the left geometry contains the right
#[derive(Debug, Default)]
pub struct Contains {}
impl BinaryPredicate for Contains {
    fn evaluate(lhs: &Geom, rhs: &Geom) -> bool {
        unsafe { tg_geom_contains(lhs.inner, rhs.inner) }
    }
}

/// Check if the right geometry contains the left
#[derive(Debug, Default)]
pub struct Within {}
impl BinaryPredicate for Within {
    fn evaluate(lhs: &Geom, rhs: &Geom) -> bool {
        unsafe { tg_geom_within(lhs.inner, rhs.inner) }
    }
}

/// Check if the left geometry covers the right geometry
#[derive(Debug, Default)]
pub struct Covers {}
impl BinaryPredicate for Covers {
    fn evaluate(lhs: &Geom, rhs: &Geom) -> bool {
        unsafe { tg_geom_covers(lhs.inner, rhs.inner) }
    }
}

/// Check if the right geometry covers the left geometry
#[derive(Debug, Default)]
pub struct CoveredBy {}
impl BinaryPredicate for CoveredBy {
    fn evaluate(lhs: &Geom, rhs: &Geom) -> bool {
        unsafe { tg_geom_coveredby(lhs.inner, rhs.inner) }
    }
}

/// Check if the geometries touch but do not otherwise intersect
#[derive(Debug, Default)]
pub struct Touches {}
impl BinaryPredicate for Touches {
    fn evaluate(lhs: &Geom, rhs: &Geom) -> bool {
        unsafe { tg_geom_touches(lhs.inner, rhs.inner) }
    }
}

/// Allocator functions for tg
pub type TgMallocFn = unsafe extern "C" fn(usize) -> *mut std::os::raw::c_void;
pub type TgReallocFn =
    unsafe extern "C" fn(*mut std::os::raw::c_void, usize) -> *mut std::os::raw::c_void;
pub type TgFreeFn = unsafe extern "C" fn(*mut std::os::raw::c_void);

/// Set the allocator for tg. This can only be called once.
///
/// # Safety
/// This function must be called prior to calling any other tg functions.
///
/// # Arguments
/// * `malloc` - The malloc function to use
/// * `realloc` - The realloc function to use
/// * `free` - The free function to use
///
/// # Returns
/// * `Ok(())` if the allocator was set successfully
/// * `Err(TgError)` if an allocator has already been set
pub unsafe fn set_allocator(
    malloc: TgMallocFn,
    realloc: TgReallocFn,
    free: TgFreeFn,
) -> Result<(), TgError> {
    static ALLOCATOR_SET: OnceLock<()> = OnceLock::new();

    if ALLOCATOR_SET.set(()).is_ok() {
        tg_env_set_allocator(Some(malloc), Some(realloc), Some(free));
        Ok(())
    } else {
        Err(TgError::Invalid("Allocator already set".to_string()))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn geom() {
        let geom = Geom::parse_wkt("POINT (0 1)", Default::default()).unwrap();
        assert_eq!(geom.to_wkt(), "POINT(0 1)");

        let wkb = geom.to_wkb();
        assert!(!wkb.is_empty());
        let geom2 = Geom::parse_wkb(&wkb, Default::default()).unwrap();
        assert_eq!(geom2.to_wkt(), "POINT(0 1)");

        assert!(geom.memsize() > 0);
    }

    fn test_predicate<Op: BinaryPredicate>(lhs: &str, rhs: &str, expected: bool) {
        let lhs_geom = Geom::parse_wkt(lhs, Default::default()).unwrap();
        let rhs_geom = Geom::parse_wkt(rhs, Default::default()).unwrap();
        if Op::evaluate(&lhs_geom, &rhs_geom) != expected {
            panic!(
                "Expected {lhs_geom} {:?} {rhs_geom} == {expected}",
                Op::default()
            )
        }
    }

    #[test]
    fn predicates() {
        test_predicate::<Equals>("POINT (0 1)", "POINT (0 1)", true);
        test_predicate::<Equals>("POINT (0 1)", "POINT (0 0)", false);

        test_predicate::<Intersects>("POINT (0 1)", "POINT (0 1)", true);
        test_predicate::<Intersects>("POINT (0 1)", "POINT (0 0)", false);

        test_predicate::<Disjoint>("POINT (0 1)", "POINT (0 1)", false);
        test_predicate::<Disjoint>("POINT (0 1)", "POINT (0 0)", true);

        test_predicate::<Contains>("POLYGON ((0 0, 5 0, 0 5, 0 0))", "POINT (1 1)", true);
        test_predicate::<Contains>("POLYGON ((0 0, 5 0, 0 5, 0 0))", "POINT (-1 -1)", false);

        test_predicate::<Within>("POINT (1 1)", "POLYGON ((0 0, 5 0, 0 5, 0 0))", true);
        test_predicate::<Within>("POINT (-1 -1)", "POLYGON ((0 0, 5 0, 0 5, 0 0))", false);

        test_predicate::<Covers>("POLYGON ((0 0, 5 0, 0 5, 0 0))", "POINT (1 1)", true);
        test_predicate::<Covers>("POLYGON ((0 0, 5 0, 0 5, 0 0))", "POINT (-1 -1)", false);

        test_predicate::<CoveredBy>("POINT (1 1)", "POLYGON ((0 0, 5 0, 0 5, 0 0))", true);
        test_predicate::<CoveredBy>("POINT (-1 -1)", "POLYGON ((0 0, 5 0, 0 5, 0 0))", false);

        test_predicate::<Touches>("POLYGON ((0 0, 5 0, 0 5, 0 0))", "POINT (0 0)", true);
        test_predicate::<Touches>("POLYGON ((0 0, 5 0, 0 5, 0 0))", "POINT (1 1)", false);
    }

    #[test]
    fn intersects() {
        let wkt1 = "POLYGON(\
            (-118.2936175 34.026516,-118.2935834 34.026516,-118.2935834 34.0264634,\
            -118.2935834 34.0263308,-118.2935834 34.0262071,-118.2935834 34.0260835,\
            -118.2935834 34.0259598,-118.2935835 34.02592,-118.2934886 34.02592,\
            -118.2934886 34.0259073,-118.2934259 34.0259073,-118.2934259 34.02592,\
            -118.2933447 34.0259205,-118.2933443 34.0260829,-118.2933442 34.0262066,\
            -118.2933442 34.0263303,-118.2933442 34.0264637,-118.2933442 34.0264907,\
            -118.2933259 34.0264907,-118.2933259 34.0265669,-118.2933035 34.0265669,\
            -118.2931745 34.0265669,-118.2931745 34.0265216,-118.2931693 34.0265191,\
            -118.2931876 34.0263539,-118.2931384 34.0263608,-118.2929819 34.0263827,\
            -118.2929795 34.0264157,-118.292977 34.0266764,-118.2929762 34.026758,\
            -118.2929659 34.026758,-118.2929658 34.0267856,-118.2931085 34.0267856,\
            -118.2931085 34.0267629,-118.2931722 34.0267629,-118.2931722 34.0267683,\
            -118.293303 34.0267683,-118.2934675 34.0267684,-118.2934675 34.0267859,\
            -118.2934684 34.0267859,-118.2935959 34.0267859,-118.2935959 34.0267078,\
            -118.2936175 34.0267078,-118.2936175 34.0265897,-118.2936175 34.026516))";
        let wkt2 = "POINT(-118.2934684 34.0267859)";
        let geom1 = Geom::parse_wkt(wkt1, IndexType::Default).unwrap();
        let geom2 = Geom::parse_wkt(wkt2, IndexType::Default).unwrap();
        let intersects = Intersects::evaluate(&geom1, &geom2);
        assert!(intersects);

        let geom1 = Geom::parse_wkt(wkt1, IndexType::YStripes).unwrap();
        let geom2 = Geom::parse_wkt(wkt2, IndexType::Default).unwrap();
        let intersects = Intersects::evaluate(&geom1, &geom2);
        assert!(intersects);
    }

    #[test]
    fn intersects_touch_top_edge() {
        let mut pts = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        for i in (0..100).rev() {
            let y = i as f64 / 100.0;
            pts.push((0.0, y));
        }

        let wkt = format!(
            "POLYGON(({}))",
            pts.iter()
                .map(|(x, y)| format!("{x} {y}"))
                .collect::<Vec<_>>()
                .join(",")
        );

        let geom = Geom::parse_wkt(&wkt, IndexType::Default).unwrap();
        let geom2 = Geom::parse_wkt("POINT (0.5 1)", IndexType::Default).unwrap();
        let intersects = Intersects::evaluate(&geom, &geom2);
        assert!(intersects);

        // This used to be problematic: https://github.com/tidwall/tg/issues/16
        let geom = Geom::parse_wkt(&wkt, IndexType::YStripes).unwrap();
        let geom2 = Geom::parse_wkt("POINT (0.5 1)", IndexType::Default).unwrap();
        let intersects = Intersects::evaluate(&geom, &geom2);
        assert!(intersects);
    }
}
