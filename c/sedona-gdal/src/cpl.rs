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

//! Ported (and contains copied code) from georust/gdal:
//! <https://github.com/georust/gdal/blob/v0.19.0/src/cpl.rs>.
//! Original code is licensed under MIT.
//!
//! GDAL Common Portability Library Functions.
//!
//! Provides [`CslStringList`], a pure-Rust implementation of GDAL's null-terminated
//! string list (`char **papszStrList`), compatible with the georust/gdal API surface.

use std::ffi::{c_char, CString};
use std::fmt::{Debug, Display, Formatter};
use std::ptr;

use crate::errors::{GdalError, Result};

/// A null-terminated array of null-terminated C strings (`char **papszStrList`).
///
/// This data structure is used throughout GDAL to pass `KEY=VALUE`-formatted options
/// to various functions.
///
/// This is a pure Rust implementation that mirrors the API of georust/gdal's
/// `CslStringList`. Memory is managed entirely in Rust — no GDAL `CSL*` functions
/// are called for list management. This should be fine as long as GDAL does not
/// take ownership of the string lists and free them using `CSLDestroy`.
///
/// # Example
///
/// There are a number of ways to populate a [`CslStringList`]:
///
/// ```rust
/// use sedona_gdal::cpl::{CslStringList, CslStringListEntry};
///
/// let mut sl1 = CslStringList::new();
/// sl1.set_name_value("NUM_THREADS", "ALL_CPUS").unwrap();
/// sl1.set_name_value("COMPRESS", "LZW").unwrap();
/// sl1.add_string("MAGIC_FLAG").unwrap();
///
/// let sl2 = CslStringList::try_from_iter(["NUM_THREADS=ALL_CPUS", "COMPRESS=LZW", "MAGIC_FLAG"]).unwrap();
///
/// assert_eq!(sl1.to_string(), sl2.to_string());
/// ```
pub struct CslStringList {
    /// Owned strings.
    strings: Vec<CString>,
    /// Null-terminated pointer array into `strings`, rebuilt on every mutation.
    /// Invariant: `ptrs.len() == strings.len() + 1` and `ptrs.last() == Some(&null_mut())`.
    ptrs: Vec<*mut c_char>,
}

// Safety: CslStringList is Send + Sync because:
// - `strings` (Vec<CString>) is Send + Sync.
// - `ptrs` contains pointers derived from `strings` (stable heap-allocated CString data).
//   They are only used for read-only FFI calls.
unsafe impl Send for CslStringList {}
unsafe impl Sync for CslStringList {}

impl CslStringList {
    /// Creates an empty GDAL string list.
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    /// Create an empty GDAL string list with given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        let mut ptrs = Vec::with_capacity(capacity + 1);
        ptrs.push(ptr::null_mut());
        Self {
            strings: Vec::with_capacity(capacity),
            ptrs,
        }
    }

    /// Rebuilds the null-terminated pointer array from `self.strings`.
    ///
    /// Must be called after every mutation to `self.strings`.
    /// This is O(n) but n is always small (option lists are typically < 20 entries).
    ///
    /// Safety argument: `CString` stores its data on the heap. Moving a `CString`
    /// (as happens during `Vec` reallocation) does not invalidate the heap pointer
    /// returned by `CString::as_ptr()`. Therefore pointers stored in `self.ptrs`
    /// remain valid as long as the corresponding `CString` in `self.strings` is alive.
    fn rebuild_ptrs(&mut self) {
        self.ptrs.clear();
        for s in &self.strings {
            self.ptrs.push(s.as_ptr() as *mut c_char);
        }
        self.ptrs.push(ptr::null_mut());
    }

    /// Check that the given `name` is a valid [`CslStringList`] key.
    ///
    /// Per [GDAL documentation](https://gdal.org/api/cpl.html#_CPPv415CSLSetNameValuePPcPKcPKc),
    /// a key cannot have non-alphanumeric characters in it (underscores are allowed).
    ///
    /// Returns `Err(GdalError::BadArgument)` on invalid name, `Ok(())` otherwise.
    fn check_valid_name(name: &str) -> Result<()> {
        if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            Err(GdalError::BadArgument(format!(
                "Invalid characters in name: '{name}'"
            )))
        } else {
            Ok(())
        }
    }

    /// Check that the given `value` is a valid [`CslStringList`] value.
    ///
    /// Per [GDAL documentation](https://gdal.org/api/cpl.html#_CPPv415CSLSetNameValuePPcPKcPKc),
    /// a value cannot have newline characters in it.
    ///
    /// Returns `Err(GdalError::BadArgument)` on invalid value, `Ok(())` otherwise.
    fn check_valid_value(value: &str) -> Result<()> {
        if value.contains(['\n', '\r']) {
            Err(GdalError::BadArgument(format!(
                "Invalid characters in value: '{value}'"
            )))
        } else {
            Ok(())
        }
    }

    /// Assigns `value` to the key `name` without checking for pre-existing assignments.
    ///
    /// Returns `Ok(())` on success, or `Err(GdalError::BadArgument)`
    /// if `name` has non-alphanumeric characters or `value` has newline characters.
    ///
    /// See: [`CSLAddNameValue`](https://gdal.org/api/cpl.html#_CPPv415CSLAddNameValuePPcPKcPKc)
    /// for details.
    pub fn add_name_value(&mut self, name: &str, value: &str) -> Result<()> {
        Self::check_valid_name(name)?;
        Self::check_valid_value(value)?;
        let entry = CString::new(format!("{name}={value}"))?;
        self.strings.push(entry);
        self.rebuild_ptrs();
        Ok(())
    }

    /// Assigns `value` to the key `name`, overwriting any existing assignment to `name`.
    ///
    /// Name lookup is case-insensitive, matching GDAL's `CSLSetNameValue` behavior.
    ///
    /// Returns `Ok(())` on success, or `Err(GdalError::BadArgument)`
    /// if `name` has non-alphanumeric characters or `value` has newline characters.
    ///
    /// See: [`CSLSetNameValue`](https://gdal.org/api/cpl.html#_CPPv415CSLSetNameValuePPcPKcPKc)
    /// for details.
    pub fn set_name_value(&mut self, name: &str, value: &str) -> Result<()> {
        Self::check_valid_name(name)?;
        Self::check_valid_value(value)?;
        let existing = self.strings.iter().position(|s| {
            s.to_str().is_ok_and(|v| {
                v.split_once('=')
                    .is_some_and(|(k, _)| k.eq_ignore_ascii_case(name))
            })
        });
        let new_entry = CString::new(format!("{name}={value}"))?;
        if let Some(idx) = existing {
            self.strings[idx] = new_entry;
        } else {
            self.strings.push(new_entry);
        }
        self.rebuild_ptrs();
        Ok(())
    }

    /// Adds a copy of the string slice `value` to the list.
    ///
    /// Returns `Ok(())` on success, `Err(GdalError::FfiNulError)` if `value` cannot be
    /// converted to a C string (e.g. `value` contains a `0` byte).
    ///
    /// See: [`CSLAddString`](https://gdal.org/api/cpl.html#_CPPv412CSLAddStringPPcPKc)
    pub fn add_string(&mut self, value: &str) -> Result<()> {
        let v = CString::new(value)?;
        self.strings.push(v);
        self.rebuild_ptrs();
        Ok(())
    }

    /// Adds the contents of a [`CslStringListEntry`] to `self`.
    ///
    /// Returns `Err(GdalError::BadArgument)` if entry doesn't meet entry restrictions as
    /// described by [`CslStringListEntry`].
    pub fn add_entry(&mut self, entry: &CslStringListEntry) -> Result<()> {
        match entry {
            CslStringListEntry::Flag(f) => self.add_string(f),
            CslStringListEntry::Pair { name, value } => self.add_name_value(name, value),
        }
    }

    /// Looks up the value corresponding to `name` (case-insensitive).
    ///
    /// See [`CSLFetchNameValue`](https://gdal.org/doxygen/cpl__string_8h.html#a4f23675f8b6f015ed23d9928048361a1)
    /// for details.
    pub fn fetch_name_value(&self, name: &str) -> Option<String> {
        for s in &self.strings {
            if let Ok(v) = s.to_str() {
                if let Some((k, val)) = v.split_once('=') {
                    if k.eq_ignore_ascii_case(name) {
                        return Some(val.to_string());
                    }
                }
            }
        }
        None
    }

    /// Perform a case **insensitive** search for the given string.
    ///
    /// Returns `Some(usize)` of value index position, or `None` if not found.
    ///
    /// See: [`CSLFindString`](https://gdal.org/api/cpl.html#_CPPv413CSLFindString12CSLConstListPKc)
    /// for details.
    pub fn find_string(&self, value: &str) -> Option<usize> {
        self.strings
            .iter()
            .position(|s| s.to_str().is_ok_and(|v| v.eq_ignore_ascii_case(value)))
    }

    /// Perform a case sensitive search for the given string.
    ///
    /// Returns `Some(usize)` of value index position, or `None` if not found.
    pub fn find_string_case_sensitive(&self, value: &str) -> Option<usize> {
        self.strings.iter().position(|s| s.to_str() == Ok(value))
    }

    /// Perform a case sensitive partial string search indicated by `fragment`.
    ///
    /// Returns `Some(usize)` of value index position, or `None` if not found.
    ///
    /// See: [`CSLPartialFindString`](https://gdal.org/api/cpl.html#_CPPv420CSLPartialFindString12CSLConstListPKc)
    /// for details.
    pub fn partial_find_string(&self, fragment: &str) -> Option<usize> {
        self.strings
            .iter()
            .position(|s| s.to_str().is_ok_and(|v| v.contains(fragment)))
    }

    /// Fetch the [`CslStringListEntry`] for the entry at the given index.
    ///
    /// Returns `None` if index is out of bounds, `Some(entry)` otherwise.
    pub fn get_field(&self, index: usize) -> Option<CslStringListEntry> {
        self.strings
            .get(index)
            .and_then(|s| s.to_str().ok())
            .map(CslStringListEntry::from)
    }

    /// Determine the number of entries in the list.
    ///
    /// See: [`CSLCount`](https://gdal.org/api/cpl.html#_CPPv48CSLCount12CSLConstList) for details.
    pub fn len(&self) -> usize {
        self.strings.len()
    }

    /// Determine if the list has any values.
    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }

    /// Get an iterator over the entries of the list.
    pub fn iter(&self) -> CslStringListIterator<'_> {
        CslStringListIterator { list: self, idx: 0 }
    }

    /// Get the raw null-terminated `char**` pointer for passing to GDAL functions.
    ///
    /// The returned pointer is valid as long as `self` is alive and not mutated.
    /// An empty list returns a pointer to `[null]`, which is a valid empty CSL.
    pub fn as_ptr(&self) -> *mut *mut c_char {
        self.ptrs.as_ptr() as *mut *mut c_char
    }

    /// Truncate the list to at most `len` entries.
    pub fn truncate(&mut self, len: usize) {
        self.strings.truncate(len);
        self.rebuild_ptrs();
    }

    /// Extend the list from an iterator, rolling back to the original size on error.
    pub fn try_extend<T, I>(&mut self, iter: I) -> Result<()>
    where
        I: IntoIterator<Item = T>,
        T: Into<CslStringListEntry>,
    {
        let original_len = self.len();
        for item in iter {
            let entry = item.into();
            if let Err(err) = self.add_entry(&entry) {
                self.truncate(original_len);
                return Err(err);
            }
        }
        Ok(())
    }

    /// Construct a `CslStringList` from a fallible iterator of entries.
    pub fn try_from_iter<T, I>(iter: I) -> Result<Self>
    where
        I: IntoIterator<Item = T>,
        T: Into<CslStringListEntry>,
    {
        let mut list = Self::new();
        list.try_extend(iter)?;
        Ok(list)
    }
}

impl Default for CslStringList {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for CslStringList {
    fn clone(&self) -> Self {
        let strings = self.strings.clone();
        let mut result = Self {
            strings,
            ptrs: Vec::new(),
        };
        result.rebuild_ptrs();
        result
    }
}

impl Debug for CslStringList {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut b = f.debug_tuple("CslStringList");
        for e in self.iter() {
            b.field(&e.to_string());
        }
        b.finish()
    }
}

impl Display for CslStringList {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for e in self.iter() {
            f.write_fmt(format_args!("{e}\n"))?;
        }
        Ok(())
    }
}

impl<'a> IntoIterator for &'a CslStringList {
    type Item = CslStringListEntry;
    type IntoIter = CslStringListIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Represents an entry in a [`CslStringList`].
///
/// An entry is either a single token ([`Flag`](Self::Flag)), or a `name=value`
/// assignment ([`Pair`](Self::Pair)).
///
/// Note: When constructed directly, assumes string values do not contain newline characters
/// nor the null `\0` character. If these conditions are violated, the provided values will
/// be ignored.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CslStringListEntry {
    /// A single token entry.
    Flag(String),
    /// A `name=value` pair entry.
    Pair { name: String, value: String },
}

impl CslStringListEntry {
    /// Create a new [`Self::Flag`] entry.
    pub fn new_flag(flag: &str) -> Self {
        CslStringListEntry::Flag(flag.to_owned())
    }

    /// Create a new [`Self::Pair`] entry.
    pub fn new_pair(name: &str, value: &str) -> Self {
        CslStringListEntry::Pair {
            name: name.to_owned(),
            value: value.to_owned(),
        }
    }
}

impl From<&str> for CslStringListEntry {
    fn from(value: &str) -> Self {
        value.to_owned().into()
    }
}

impl From<(&str, &str)> for CslStringListEntry {
    fn from((key, value): (&str, &str)) -> Self {
        Self::new_pair(key, value)
    }
}

impl From<String> for CslStringListEntry {
    fn from(value: String) -> Self {
        match value.split_once('=') {
            Some((name, value)) => Self::new_pair(name, value),
            None => Self::new_flag(&value),
        }
    }
}

impl From<(String, String)> for CslStringListEntry {
    fn from((name, value): (String, String)) -> Self {
        Self::Pair { name, value }
    }
}

impl Display for CslStringListEntry {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            CslStringListEntry::Flag(s) => f.write_str(s),
            CslStringListEntry::Pair { name, value } => f.write_fmt(format_args!("{name}={value}")),
        }
    }
}

/// State for iterator over [`CslStringList`] entries.
pub struct CslStringListIterator<'a> {
    list: &'a CslStringList,
    idx: usize,
}

impl Iterator for CslStringListIterator<'_> {
    type Item = CslStringListEntry;

    fn next(&mut self) -> Option<Self::Item> {
        let entry = self.list.strings.get(self.idx)?;
        self.idx += 1;
        Some(entry.to_string_lossy().into_owned().into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::Result;

    fn fixture() -> Result<CslStringList> {
        let mut l = CslStringList::new();
        l.set_name_value("ONE", "1")?;
        l.set_name_value("TWO", "2")?;
        l.set_name_value("THREE", "3")?;
        l.add_string("SOME_FLAG")?;
        Ok(l)
    }

    #[test]
    fn construct() -> Result<()> {
        let mut sl1 = CslStringList::new();
        sl1.set_name_value("NUM_THREADS", "ALL_CPUS").unwrap();
        sl1.set_name_value("COMPRESS", "LZW").unwrap();
        sl1.add_string("MAGIC_FLAG").unwrap();

        let sl2 =
            CslStringList::try_from_iter(["NUM_THREADS=ALL_CPUS", "COMPRESS=LZW", "MAGIC_FLAG"])?;
        let sl3 = CslStringList::try_from_iter([
            CslStringListEntry::from(("NUM_THREADS", "ALL_CPUS")),
            CslStringListEntry::from(("COMPRESS", "LZW")),
            CslStringListEntry::from("MAGIC_FLAG"),
        ])?;

        assert_eq!(sl1.to_string(), sl2.to_string());
        assert_eq!(sl2.to_string(), sl3.to_string());

        Ok(())
    }

    #[test]
    fn basic_list() -> Result<()> {
        let l = fixture()?;
        assert!(matches!(l.fetch_name_value("ONE"), Some(s) if s == *"1"));
        assert!(matches!(l.fetch_name_value("THREE"), Some(s) if s == *"3"));
        assert!(l.fetch_name_value("FOO").is_none());

        Ok(())
    }

    #[test]
    fn has_length() -> Result<()> {
        let l = fixture()?;
        assert_eq!(l.len(), 4);

        Ok(())
    }

    #[test]
    fn can_be_empty() -> Result<()> {
        let l = CslStringList::new();
        assert!(l.is_empty());

        let l = fixture()?;
        assert!(!l.is_empty());

        Ok(())
    }

    #[test]
    fn has_iterator() -> Result<()> {
        let f = fixture()?;
        let mut it = f.iter();
        assert_eq!(it.next(), Some(("ONE", "1").into()));
        assert_eq!(it.next(), Some(("TWO", "2").into()));
        assert_eq!(it.next(), Some(("THREE", "3").into()));
        assert_eq!(it.next(), Some("SOME_FLAG".into()));
        assert_eq!(it.next(), None);
        assert_eq!(it.next(), None);
        Ok(())
    }

    #[test]
    fn invalid_name_value() -> Result<()> {
        let mut l = fixture()?;
        assert!(l.set_name_value("l==t", "2").is_err());
        assert!(l.set_name_value("foo", "2\n4\r5").is_err());

        Ok(())
    }

    #[test]
    fn add_vs_set() -> Result<()> {
        let mut f = CslStringList::new();
        f.add_name_value("ONE", "1")?;
        f.add_name_value("ONE", "2")?;
        let s = f.to_string();
        assert!(s.contains("ONE") && s.contains('1') && s.contains('2'));

        let mut f = CslStringList::new();
        f.set_name_value("ONE", "1")?;
        f.set_name_value("ONE", "2")?;
        let s = f.to_string();
        assert!(s.contains("ONE") && !s.contains('1') && s.contains('2'));

        Ok(())
    }

    #[test]
    fn try_from_iter_constructs_list() -> Result<()> {
        let l = CslStringList::try_from_iter(["ONE=1", "TWO=2"])?;
        assert!(matches!(l.fetch_name_value("ONE"), Some(s) if s == *"1"));
        assert!(matches!(l.fetch_name_value("TWO"), Some(s) if s == *"2"));

        Ok(())
    }

    #[test]
    fn try_from_iter_rejects_invalid_entry() {
        let result = CslStringList::try_from_iter([CslStringListEntry::from(("bad-name", "1"))]);
        assert!(matches!(result, Err(GdalError::BadArgument(_))));
    }

    #[test]
    fn try_extend_is_transactional() -> Result<()> {
        let mut list = CslStringList::try_from_iter([("ONE", "1"), ("TWO", "2")])?;
        let before = list.clone();

        let result = list.try_extend([
            CslStringListEntry::from(("THREE", "3")),
            CslStringListEntry::from(("bad-name", "4")),
        ]);

        assert!(matches!(result, Err(GdalError::BadArgument(_))));
        assert_eq!(list.to_string(), before.to_string());

        Ok(())
    }

    #[test]
    fn truncate_preserves_ptr_invariant() -> Result<()> {
        let mut list = CslStringList::try_from_iter(["A", "B", "C"])?;
        list.truncate(1);

        assert_eq!(list.len(), 1);
        assert_eq!(list.get_field(0), Some(CslStringListEntry::from("A")));
        assert_eq!(list.get_field(1), None);

        let ptr = list.as_ptr();
        unsafe {
            assert!(!(*ptr).is_null());
            assert!((*ptr.add(1)).is_null());
        }

        Ok(())
    }

    #[test]
    fn debug_fmt() -> Result<()> {
        let l = fixture()?;
        let s = format!("{l:?}");
        assert!(s.contains("ONE=1"));
        assert!(s.contains("TWO=2"));
        assert!(s.contains("THREE=3"));
        assert!(s.contains("SOME_FLAG"));

        Ok(())
    }

    #[test]
    fn can_add_strings() -> Result<()> {
        let mut l = CslStringList::new();
        assert!(l.is_empty());
        l.add_string("-abc")?;
        l.add_string("-d_ef")?;
        l.add_string("A")?;
        l.add_string("B")?;
        assert_eq!(l.len(), 4);

        Ok(())
    }

    #[test]
    fn find_string() -> Result<()> {
        let f = fixture()?;
        assert_eq!(f.find_string("NON_FLAG"), None);
        assert_eq!(f.find_string("SOME_FLAG"), Some(3));
        assert_eq!(f.find_string("ONE=1"), Some(0));
        assert_eq!(f.find_string("one=1"), Some(0));
        assert_eq!(f.find_string("TWO="), None);
        Ok(())
    }

    #[test]
    fn find_string_case_sensitive() -> Result<()> {
        let f = fixture()?;
        assert_eq!(f.find_string_case_sensitive("ONE=1"), Some(0));
        assert_eq!(f.find_string_case_sensitive("one=1"), None);
        assert_eq!(f.find_string_case_sensitive("SOME_FLAG"), Some(3));
        Ok(())
    }

    #[test]
    fn partial_find_string() -> Result<()> {
        let f = fixture()?;
        assert_eq!(f.partial_find_string("ONE=1"), Some(0));
        assert_eq!(f.partial_find_string("ONE="), Some(0));
        assert_eq!(f.partial_find_string("=1"), Some(0));
        assert_eq!(f.partial_find_string("1"), Some(0));
        assert_eq!(f.partial_find_string("THREE="), Some(2));
        assert_eq!(f.partial_find_string("THREE"), Some(2));
        assert_eq!(f.partial_find_string("three"), None);
        Ok(())
    }

    #[test]
    fn as_ptr_is_null_terminated() {
        let mut l = CslStringList::new();
        l.add_string("A").unwrap();
        l.add_string("B").unwrap();
        let ptr = l.as_ptr();
        unsafe {
            // First entry
            assert!(!(*ptr).is_null());
            // Second entry
            assert!(!(*ptr.add(1)).is_null());
            // Null terminator
            assert!((*ptr.add(2)).is_null());
        }
    }

    #[test]
    fn clone_is_independent() -> Result<()> {
        let f = fixture()?;
        let mut g = f.clone();
        g.set_name_value("ONE", "999")?;
        // Original is unchanged.
        assert_eq!(f.fetch_name_value("ONE"), Some("1".into()));
        assert_eq!(g.fetch_name_value("ONE"), Some("999".into()));
        Ok(())
    }
}
