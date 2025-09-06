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
// Example functions

use std::ffi::c_void;

use savvy::savvy;

use savvy::{IntegerSexp, OwnedIntegerSexp, OwnedStringSexp, StringSexp};

use savvy::NotAvailableValue;
use savvy_ffi::R_NilValue;
use sedona_adbc::AdbcSedonadbDriverInit;

mod context;
mod dataframe;
mod error;
mod runtime;

#[savvy]
fn sedonadb_adbc_init_func() -> savvy::Result<savvy::Sexp> {
    let driver_init_void = AdbcSedonadbDriverInit as *mut c_void;

    unsafe {
        Ok(savvy::Sexp(savvy_ffi::R_MakeExternalPtr(
            driver_init_void,
            R_NilValue,
            R_NilValue,
        )))
    }
}

/// Convert Input To Upper-Case
///
/// @param x A character vector.
/// @returns A character vector with upper case version of the input.
/// @export
#[savvy]
fn to_upper(x: StringSexp) -> savvy::Result<savvy::Sexp> {
    let mut out = OwnedStringSexp::new(x.len())?;

    for (i, e) in x.iter().enumerate() {
        if e.is_na() {
            out.set_na(i)?;
            continue;
        }

        let e_upper = e.to_uppercase();
        out.set_elt(i, &e_upper)?;
    }

    Ok(out.into())
}

/// Multiply Input By Another Input
///
/// @param x An integer vector.
/// @param y An integer to multiply.
/// @returns An integer vector with values multiplied by `y`.
/// @export
#[savvy]
fn int_times_int(x: IntegerSexp, y: i32) -> savvy::Result<savvy::Sexp> {
    let mut out = OwnedIntegerSexp::new(x.len())?;

    for (i, e) in x.iter().enumerate() {
        if e.is_na() {
            out.set_na(i)?;
        } else {
            out[i] = e * y;
        }
    }

    Ok(out.into())
}

#[savvy]
struct Person {
    pub name: String,
}

/// A person with a name
///
/// @export
#[savvy]
impl Person {
    fn new() -> Self {
        Self {
            name: "".to_string(),
        }
    }

    fn set_name(&mut self, name: &str) -> savvy::Result<()> {
        self.name = name.to_string();
        Ok(())
    }

    fn name(&self) -> savvy::Result<savvy::Sexp> {
        let mut out = OwnedStringSexp::new(1)?;
        out.set_elt(0, &self.name)?;
        Ok(out.into())
    }

    fn associated_function() -> savvy::Result<savvy::Sexp> {
        let mut out = OwnedStringSexp::new(1)?;
        out.set_elt(0, "associated_function")?;
        Ok(out.into())
    }
}

// This test is run by `cargo test`. You can put tests that don't need a real
// R session here.
#[cfg(test)]
mod test1 {
    #[test]
    fn test_person() {
        let mut p = super::Person::new();
        p.set_name("foo").expect("set_name() must succeed");
        assert_eq!(&p.name, "foo");
    }
}
