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

//! View entries — per-axis slice / broadcast / permutation specs for an
//! N-D raster band. The [`ViewEntries`] newtype owns a `Vec<ViewEntry>`
//! and is the entry point for all view-machinery operations
//! (validation, identity check, visible-shape derivation, composition).

use arrow_schema::ArrowError;

/// One per-dimension entry of a band's logical view. Describes how a
/// visible axis maps onto an axis of the underlying source buffer.
///
/// - `source_axis`: index into the band's `source_shape` that this visible
///   axis reads from. Across a band's full view, `source_axis` values must
///   form a permutation of `0..ndim` — axis-dropping and axis-introducing
///   views are not supported today.
/// - `start`: starting index along the source axis (in elements, not bytes).
/// - `step`: stride between consecutive visible elements along the source
///   axis. `step == 0` means broadcast (the same source element is
///   exposed `steps` times); negative `step` means reverse iteration.
/// - `steps`: number of visible elements along this axis. `steps == 0` is
///   allowed (empty axis).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ViewEntry {
    pub source_axis: i64,
    pub start: i64,
    pub step: i64,
    pub steps: i64,
}

/// A band's full view, one [`ViewEntry`] per visible axis.
///
/// Use [`ViewEntries::new`] to wrap an existing `Vec<ViewEntry>` or
/// [`ViewEntries::identity_for_shape`] to build the canonical
/// no-op view over a given source shape. Operations on the view live
/// as methods on this type — `validate`, `visible_shape`,
/// `is_identity`, `compose`. The newtype gives Vec<ViewEntry> a name
/// and a single place to attach helpers and tests.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ViewEntries(Vec<ViewEntry>);

impl ViewEntries {
    /// Wrap a pre-built vector of entries. The view is not validated
    /// here — call [`Self::validate`] before relying on it, or use
    /// [`Self::try_new`] to wrap and validate in one step.
    pub fn new(inner: Vec<ViewEntry>) -> Self {
        Self(inner)
    }

    /// Wrap a pre-built vector of entries and validate it against
    /// `source_shape` in one step — [`Self::new`] followed by
    /// [`Self::validate`].
    pub fn try_new(inner: Vec<ViewEntry>, source_shape: &[i64]) -> Result<Self, ArrowError> {
        let entries = Self::new(inner);
        entries.validate(source_shape)?;
        Ok(entries)
    }

    /// Build the canonical identity view over `source_shape`:
    /// `[(source_axis=k, start=0, step=1, steps=source_shape[k])]` for
    /// each k. Always self-validates.
    pub fn identity_for_shape(source_shape: &[i64]) -> Self {
        Self(
            source_shape
                .iter()
                .enumerate()
                .map(|(k, &s)| ViewEntry {
                    source_axis: k as i64,
                    start: 0,
                    step: 1,
                    steps: s,
                })
                .collect(),
        )
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn as_slice(&self) -> &[ViewEntry] {
        &self.0
    }

    pub fn iter(&self) -> std::slice::Iter<'_, ViewEntry> {
        self.0.iter()
    }

    /// Visible shape derived from `[v.steps for v in self]`. `validate`
    /// guarantees `steps >= 0`, so callers can treat the result as
    /// non-negative after a successful validation.
    pub fn visible_shape(&self) -> Vec<i64> {
        self.0.iter().map(|v| v.steps).collect()
    }

    /// True iff this view is the canonical identity over a C-order
    /// source buffer: every visible axis `k` is a no-op (no
    /// permutation, no slice offset, no stride/reverse, full coverage
    /// of the corresponding source axis).
    ///
    /// Identity views are always contiguous, so the reader borrows the
    /// underlying `data` column directly through `NdBuffer::as_contiguous()`
    /// with no allocation or copy. Non-identity views may still be
    /// contiguous (e.g. an outer-axis slice); strided ones are rejected by
    /// `as_contiguous()` and must be repacked via an explicit plan node.
    pub fn is_identity(&self, source_shape: &[i64]) -> bool {
        if self.0.len() != source_shape.len() {
            return false;
        }
        self.0.iter().enumerate().all(|(k, v)| {
            v.source_axis == k as i64 && v.start == 0 && v.step == 1 && v.steps == source_shape[k]
        })
    }

    /// Validate against a band's `source_shape`. `Ok(())` iff:
    ///
    /// - `self.len() == source_shape.len()`.
    /// - `source_axis` values across `self` form a permutation of
    ///   `0..source_shape.len()` (no axis duplicated, none missing).
    /// - Every `source_shape[k] >= 0`.
    /// - Every `steps >= 0`.
    /// - When `steps > 0`: `start ∈ [0, source_shape[source_axis])`,
    ///   and when `step != 0` the last addressed element
    ///   `start + (steps - 1) * step` is also in that range.
    pub fn validate(&self, source_shape: &[i64]) -> Result<(), ArrowError> {
        let ndim = source_shape.len();
        if self.0.len() != ndim {
            return Err(ArrowError::InvalidArgumentError(format!(
                "view length ({}) must equal source_shape length ({ndim})",
                self.0.len()
            )));
        }
        let mut seen = vec![false; ndim];
        for (k, v) in self.0.iter().enumerate() {
            if v.source_axis < 0 || (v.source_axis as usize) >= ndim {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "view[{k}].source_axis = {} is out of range [0, {ndim})",
                    v.source_axis
                )));
            }
            let sa = v.source_axis as usize;
            if seen[sa] {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "view source_axis values must be a permutation of 0..{ndim}; \
                     axis {sa} appears more than once"
                )));
            }
            seen[sa] = true;

            if v.steps < 0 {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "view[{k}].steps = {} must be >= 0",
                    v.steps
                )));
            }
            if source_shape[sa] < 0 {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "source_shape[{sa}] = {} must be >= 0",
                    source_shape[sa]
                )));
            }
            if v.steps > 0 {
                let s = source_shape[sa];
                if v.start < 0 || v.start >= s {
                    return Err(ArrowError::InvalidArgumentError(format!(
                        "view[{k}].start = {} is out of range [0, {s}) for source axis {sa}",
                        v.start
                    )));
                }
                if v.step != 0 {
                    // Checked arithmetic: a malicious or corrupted view can't
                    // wrap (steps-1)*step or start+… into an in-range value and
                    // bypass the bound check. Any overflow is a validation
                    // error.
                    let last = (v.steps - 1)
                        .checked_mul(v.step)
                        .and_then(|d| v.start.checked_add(d))
                        .ok_or_else(|| {
                            ArrowError::InvalidArgumentError(format!(
                                "view[{k}] last-element index overflows i64 for \
                                 start={}, step={}, steps={} on source axis {sa}",
                                v.start, v.step, v.steps
                            ))
                        })?;
                    if last < 0 || last >= s {
                        return Err(ArrowError::InvalidArgumentError(format!(
                            "view[{k}] addresses element {last} which is out of range \
                             [0, {s}) for source axis {sa}"
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// Compose `next` (a view spec over `self`'s visible axes) with
    /// `self` (a view spec over a band's `source_shape`), producing a
    /// single view over the same `source_shape` that represents the
    /// result of viewing through `next`.
    ///
    /// `next[k].source_axis` indexes into `self`'s visible axes
    /// (`0..self.len()`), NOT into the underlying source axes. This
    /// lets callers (e.g. lazy slicing) describe the new view in terms
    /// of what they see, without knowing the input's internal layout.
    ///
    /// Math: for each output axis `k`, walking through `self` resolves
    /// the source axis and translates start/step:
    ///   - `out.source_axis  = self[next.source_axis].source_axis`
    ///   - `out.start        = self.start + next.start * self.step`
    ///   - `out.step         = next.step * self.step`
    ///   - `out.steps        = next.steps`
    ///
    /// Uses checked arithmetic at every step. The caller must
    /// [`validate`] the result against the source shape before use.
    ///
    /// [`validate`]: Self::validate
    pub fn compose(&self, next: &Self) -> Result<Self, ArrowError> {
        let mut out = Vec::with_capacity(next.0.len());
        for (k, next_entry) in next.0.iter().enumerate() {
            if next_entry.source_axis < 0 || (next_entry.source_axis as usize) >= self.0.len() {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "compose: next[{k}].source_axis ({}) is out of range \
                     for input with {} visible axes",
                    next_entry.source_axis,
                    self.0.len()
                )));
            }
            let input = &self.0[next_entry.source_axis as usize];
            let step = next_entry.step.checked_mul(input.step).ok_or_else(|| {
                ArrowError::InvalidArgumentError(format!(
                    "compose: step product overflows i64 at axis {k} \
                     (next.step={}, input.step={})",
                    next_entry.step, input.step
                ))
            })?;
            let start_offset = next_entry.start.checked_mul(input.step).ok_or_else(|| {
                ArrowError::InvalidArgumentError(format!(
                    "compose: next.start * input.step overflows i64 at axis {k} \
                     (next.start={}, input.step={})",
                    next_entry.start, input.step
                ))
            })?;
            let start = input.start.checked_add(start_offset).ok_or_else(|| {
                ArrowError::InvalidArgumentError(format!(
                    "compose: composed start overflows i64 at axis {k} \
                     (input.start={}, offset={})",
                    input.start, start_offset
                ))
            })?;
            out.push(ViewEntry {
                source_axis: input.source_axis,
                start,
                step,
                steps: next_entry.steps,
            });
        }
        Ok(Self(out))
    }
}

impl From<Vec<ViewEntry>> for ViewEntries {
    fn from(inner: Vec<ViewEntry>) -> Self {
        Self(inner)
    }
}

impl AsRef<[ViewEntry]> for ViewEntries {
    fn as_ref(&self) -> &[ViewEntry] {
        &self.0
    }
}

impl std::ops::Index<usize> for ViewEntries {
    type Output = ViewEntry;
    fn index(&self, i: usize) -> &ViewEntry {
        &self.0[i]
    }
}

impl<'a> IntoIterator for &'a ViewEntries {
    type Item = &'a ViewEntry;
    type IntoIter = std::slice::Iter<'a, ViewEntry>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

/// Build a [`ViewEntries`] from numpy-style slice syntax. Each `a:b`
/// becomes a `ViewEntry { source_axis: k, start: a, step: 1, steps: b - a }`
/// on the next axis. Step ≠ 1 isn't supported by the macro — use
/// the struct constructor directly for that.
///
/// ```ignore
/// let v = view_entries![0:4, 1:5];
/// // → axis 0: start=0, step=1, steps=4
/// // → axis 1: start=1, step=1, steps=4
/// ```
#[macro_export]
macro_rules! view_entries {
    ($($start:literal : $stop:literal),* $(,)?) => {
        $crate::view_entries::ViewEntries::new({
            #[allow(unused_mut)]
            let mut entries: Vec<$crate::view_entries::ViewEntry> = Vec::new();
            // Pre-increment so the last write into `axis` is always read by
            // the following push — no dangling assignment on the final entry,
            // and no warning on single-entry uses.
            #[allow(unused_mut)]
            let mut axis: i64 = -1;
            $(
                axis += 1;
                entries.push($crate::view_entries::ViewEntry {
                    source_axis: axis,
                    start: $start,
                    step: 1,
                    steps: ($stop as i64) - ($start as i64),
                });
            )*
            entries
        })
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ve(source_axis: i64, start: i64, step: i64, steps: i64) -> ViewEntry {
        ViewEntry {
            source_axis,
            start,
            step,
            steps,
        }
    }

    fn entries(v: &[ViewEntry]) -> ViewEntries {
        ViewEntries::new(v.to_vec())
    }

    // ---- macro ----

    #[test]
    fn macro_expands_python_style_slice_to_view_entries() {
        let v = view_entries![0:4, 1:5];
        assert_eq!(v.len(), 2);
        assert_eq!(v[0], ve(0, 0, 1, 4));
        assert_eq!(v[1], ve(1, 1, 1, 4));
    }

    // ---- visible_shape ----

    #[test]
    fn visible_shape_pulls_steps_from_each_entry() {
        let v = entries(&[ve(0, 0, 1, 4), ve(1, 0, 1, 5)]);
        assert_eq!(v.visible_shape(), vec![4, 5]);
    }

    // ---- identity_for_shape ----

    #[test]
    fn identity_for_shape_round_trips_through_is_identity() {
        let v = ViewEntries::identity_for_shape(&[4, 5, 6]);
        assert!(v.is_identity(&[4, 5, 6]));
    }

    // ---- validate ----

    #[test]
    fn validate_accepts_identity() {
        let v = entries(&[ve(0, 0, 1, 4), ve(1, 0, 1, 5)]);
        v.validate(&[4, 5]).unwrap();
    }

    #[test]
    fn validate_rejects_length_mismatch() {
        let v = entries(&[ve(0, 0, 1, 4)]);
        let err = v.validate(&[4, 5]).unwrap_err();
        assert!(err.to_string().contains("must equal"), "got {err}");
    }

    #[test]
    fn validate_rejects_negative_source_axis() {
        let v = entries(&[ve(-1, 0, 1, 4)]);
        let err = v.validate(&[4]).unwrap_err();
        assert!(err.to_string().contains("source_axis"), "got {err}");
    }

    #[test]
    fn validate_rejects_oob_source_axis() {
        let v = entries(&[ve(2, 0, 1, 4)]);
        let err = v.validate(&[4]).unwrap_err();
        assert!(err.to_string().contains("source_axis"), "got {err}");
    }

    #[test]
    fn validate_rejects_duplicate_source_axis() {
        let v = entries(&[ve(0, 0, 1, 2), ve(0, 0, 1, 2)]);
        let err = v.validate(&[2, 3]).unwrap_err();
        assert!(err.to_string().contains("permutation"), "got {err}");
    }

    #[test]
    fn validate_rejects_negative_steps() {
        let v = entries(&[ve(0, 0, 1, -1)]);
        let err = v.validate(&[4]).unwrap_err();
        assert!(err.to_string().contains("steps"), "got {err}");
    }

    #[test]
    fn validate_rejects_negative_start() {
        let v = entries(&[ve(0, -1, 1, 1)]);
        let err = v.validate(&[4]).unwrap_err();
        assert!(err.to_string().contains("start"), "got {err}");
    }

    #[test]
    fn validate_rejects_start_at_source_size() {
        let v = entries(&[ve(0, 4, 1, 1)]);
        let err = v.validate(&[4]).unwrap_err();
        assert!(err.to_string().contains("start"), "got {err}");
    }

    #[test]
    fn validate_rejects_negative_step_underrun() {
        // start=0, step=-1, steps=2 addresses element 0 then -1 → underrun.
        let v = entries(&[ve(0, 0, -1, 2)]);
        let err = v.validate(&[4]).unwrap_err();
        assert!(err.to_string().contains("out of range"), "got {err}");
    }

    #[test]
    fn validate_accepts_negative_step_full_reverse() {
        // start=3, step=-1, steps=4 addresses 3,2,1,0 — all in range.
        let v = entries(&[ve(0, 3, -1, 4)]);
        v.validate(&[4]).unwrap();
    }

    #[test]
    fn validate_accepts_steps_zero_with_unconstrained_start() {
        let v = entries(&[ve(0, 999, 1, 0)]);
        v.validate(&[4]).unwrap();
    }

    #[test]
    fn validate_steps_one_only_checks_start() {
        let v = entries(&[ve(0, 3, 999, 1)]);
        v.validate(&[4]).unwrap();
    }

    #[test]
    fn validate_step_zero_broadcast_within_bounds() {
        let v_ok = entries(&[ve(0, 3, 0, 100)]);
        v_ok.validate(&[4]).unwrap();
        let v_bad = entries(&[ve(0, 4, 0, 1)]);
        let err = v_bad.validate(&[4]).unwrap_err();
        assert!(err.to_string().contains("start"), "got {err}");
    }

    #[test]
    fn validate_permutation_with_slice_ok() {
        let v = entries(&[ve(1, 0, 1, 3), ve(0, 1, 1, 1)]);
        v.validate(&[2, 3]).unwrap();
    }

    #[test]
    fn validate_rejects_i64_overflow_in_last_element() {
        let v = entries(&[ve(0, 10, i64::MAX, 3)]);
        let err = v.validate(&[100]).unwrap_err();
        assert!(err.to_string().contains("overflow"), "got: {err}");
    }

    #[test]
    fn validate_rejects_i64_overflow_in_start_plus_offset() {
        let v = entries(&[ve(0, 2, 1, i64::MAX)]);
        let err = v.validate(&[100]).unwrap_err();
        assert!(err.to_string().contains("overflow"), "got: {err}");
    }

    // ---- is_identity ----

    #[test]
    fn is_identity_accepts_canonical_identity() {
        let v = entries(&[ve(0, 0, 1, 4), ve(1, 0, 1, 5)]);
        assert!(v.is_identity(&[4, 5]));
    }

    #[test]
    fn is_identity_rejects_axis_permutation_even_with_identity_step() {
        // start/step/steps all look identity-shaped, but source_axis is
        // permuted. Indistinguishable from identity in start/step terms —
        // the source_axis check is the only thing that catches it.
        let v = entries(&[ve(1, 0, 1, 5), ve(0, 0, 1, 4)]);
        assert!(!v.is_identity(&[4, 5]));
    }

    #[test]
    fn is_identity_rejects_partial_axis_coverage() {
        let v = entries(&[ve(0, 0, 1, 3), ve(1, 0, 1, 5)]);
        assert!(!v.is_identity(&[4, 5]));
    }

    #[test]
    fn is_identity_rejects_non_zero_start() {
        let v = entries(&[ve(0, 1, 1, 3), ve(1, 0, 1, 5)]);
        assert!(!v.is_identity(&[4, 5]));
    }

    #[test]
    fn is_identity_rejects_non_unit_step() {
        let v = entries(&[ve(0, 0, 2, 2), ve(1, 0, 1, 5)]);
        assert!(!v.is_identity(&[4, 5]));
    }

    #[test]
    fn is_identity_rejects_length_mismatch() {
        let v = entries(&[ve(0, 0, 1, 4)]);
        assert!(!v.is_identity(&[4, 5]));
    }

    // ---- compose ----

    #[test]
    fn compose_identity_input_returns_next_unchanged() {
        let identity = entries(&[ve(0, 0, 1, 8)]);
        let next = entries(&[ve(0, 2, 3, 2)]);
        let composed = identity.compose(&next).unwrap();
        assert_eq!(composed.len(), 1);
        assert_eq!(composed[0], ve(0, 2, 3, 2));
    }

    #[test]
    fn compose_slice_of_slice_collapses_into_one_view() {
        // Input view samples source[0..8] at indices 1, 3, 5.
        // Next view samples that visible region at index 1 only.
        // Composed: samples source axis 0 at index 3 (= 1 + 1*2), step=2, steps=1.
        let input = entries(&[ve(0, 1, 2, 3)]);
        let next = entries(&[ve(0, 1, 1, 1)]);
        let composed = input.compose(&next).unwrap();
        assert_eq!(composed.as_slice(), &[ve(0, 3, 2, 1)]);
    }

    #[test]
    fn compose_preserves_source_axis_through_permutation() {
        // Input transposes axes: visible 0 = source 1, visible 1 = source 0.
        // Next picks axis 1 (which is source axis 0), so the composed
        // view's sole entry must have source_axis = 0.
        let input = entries(&[ve(1, 0, 1, 5), ve(0, 0, 1, 4)]);
        let next = entries(&[ve(1, 2, 1, 2)]);
        let composed = input.compose(&next).unwrap();
        assert_eq!(composed.len(), 1);
        assert_eq!(composed[0].source_axis, 0);
        assert_eq!(composed[0].start, 2);
        assert_eq!(composed[0].step, 1);
        assert_eq!(composed[0].steps, 2);
    }

    #[test]
    fn compose_step_multiplies() {
        // Input step=3, next step=2 → composed step=6.
        let input = entries(&[ve(0, 0, 3, 5)]);
        let next = entries(&[ve(0, 1, 2, 2)]);
        let composed = input.compose(&next).unwrap();
        assert_eq!(composed.as_slice(), &[ve(0, 3, 6, 2)]);
    }

    #[test]
    fn compose_rejects_next_source_axis_out_of_range() {
        let input = entries(&[ve(0, 0, 1, 4)]);
        let next = entries(&[ve(2, 0, 1, 1)]);
        let err = input.compose(&next).unwrap_err();
        assert!(err.to_string().contains("out of range"), "got {err}");
    }

    #[test]
    fn compose_rejects_step_product_overflow() {
        let input = entries(&[ve(0, 0, i64::MAX, 2)]);
        let next = entries(&[ve(0, 0, 2, 1)]);
        let err = input.compose(&next).unwrap_err();
        assert!(
            err.to_string().contains("step product overflows"),
            "got {err}"
        );
    }
}
