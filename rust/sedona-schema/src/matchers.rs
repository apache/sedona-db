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

use std::{fmt::Debug, iter::zip, sync::Arc};

use arrow_schema::DataType;
use datafusion_common::{plan_err, Result};
use sedona_common::sedona_internal_err;
use sedona_geometry::types::Edges;

use crate::datatypes::{SedonaType, RASTER, WKB_GEOGRAPHY, WKB_GEOMETRY};

/// Helper to match arguments and compute return types
#[derive(Debug)]
pub struct ArgMatcher {
    matchers: Vec<Arc<dyn TypeMatcher + Send + Sync>>,
    out_type: SedonaType,
}

impl ArgMatcher {
    /// Create a new ArgMatcher
    pub fn new(matchers: Vec<Arc<dyn TypeMatcher + Send + Sync>>, out_type: SedonaType) -> Self {
        Self { matchers, out_type }
    }

    /// Calculate a return type given input types
    ///
    /// Returns Some(physical_type) if this kernel applies to the input types or
    /// None otherwise. This function also checks that all input arguments have
    /// compatible CRSes and if so, applies the CRS to the output type.
    pub fn match_args(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        if !self.matches(args) {
            return Ok(None);
        }

        let geometry_arg_crses = args
            .iter()
            .filter(|arg_type| IsGeometryOrGeography {}.match_type(arg_type))
            .map(|arg_type| arg_type.crs().clone())
            .collect::<Vec<_>>();

        if geometry_arg_crses.is_empty() {
            return Ok(Some(self.out_type.clone()));
        }

        let out_crs = geometry_arg_crses[0].clone();
        for this_crs in geometry_arg_crses.into_iter().skip(1) {
            if out_crs != this_crs {
                let hint = "Use ST_Transform() or ST_SetSRID() to ensure arguments are compatible.";

                return match (out_crs, this_crs) {
                    (None, Some(rhs_crs)) => {
                        plan_err!("Mismatched CRS arguments: None vs {rhs_crs}\n{hint}")
                    }
                    (Some(lhs_crs), None) => {
                        plan_err!("Mismatched CRS arguments: {lhs_crs} vs None\n{hint}")
                    }
                    (Some(lhs_crs), Some(rhs_crs)) => {
                        plan_err!("Mismatched CRS arguments: {lhs_crs} vs {rhs_crs}\n{hint}")
                    }
                    _ => sedona_internal_err!("None vs. None should be considered equal"),
                };
            }
        }

        match &self.out_type {
            SedonaType::Wkb(edges, _) => Ok(Some(SedonaType::Wkb(*edges, out_crs))),
            SedonaType::WkbView(edges, _) => Ok(Some(SedonaType::WkbView(*edges, out_crs))),
            _ => Ok(Some(self.out_type.clone())),
        }
    }

    /// Check for an input type match
    ///
    /// Returns true if args applies to the input types.
    pub fn matches(&self, args: &[SedonaType]) -> bool {
        if args.len() > self.matchers.len() {
            return false;
        }

        let matcher_iter = self.matchers.iter();
        let mut arg_iter = args.iter().peekable();

        for matcher in matcher_iter {
            if let Some(arg) = arg_iter.peek() {
                if arg == &&SedonaType::Arrow(DataType::Null) || matcher.match_type(arg) {
                    arg_iter.next(); // Consume the argument
                    continue; // Move to the next matcher
                } else if matcher.optional() {
                    continue; // Skip the optional matcher
                } else {
                    return false; // Non-optional matcher failed
                }
            } else if matcher.optional() {
                continue; // Skip remaining optional matchers
            } else {
                return false; // Non-optional matcher failed with no arguments left
            }
        }

        // Ensure all arguments are consumed
        arg_iter.next().is_none()
    }

    /// Calls each [TypeMatcher]'s `type_if_null()`
    ///
    /// This method errors if one or more matchers does not have an
    /// unambiguous castable-from-null storage type. It is provided
    /// as a utility for generic kernel implementations that rely on
    /// the matcher to sanitize input that may contain literal nulls.
    pub fn types_if_null(&self, args: &[SedonaType]) -> Result<Vec<SedonaType>> {
        let mut out = Vec::new();
        for (arg, matcher) in zip(args, &self.matchers) {
            if let SedonaType::Arrow(DataType::Null) = arg {
                if let Some(type_if_null) = matcher.type_if_null() {
                    out.push(type_if_null);
                } else {
                    return sedona_internal_err!(
                        "Matcher {matcher:?} does not provide type_if_null()"
                    );
                }
            } else {
                out.push(arg.clone());
            }
        }

        Ok(out)
    }

    /// Matches any argument
    pub fn is_any() -> Arc<dyn TypeMatcher + Send + Sync> {
        Arc::new(IsAny {})
    }

    /// Matches the given Arrow type using PartialEq
    pub fn is_arrow(data_type: DataType) -> Arc<dyn TypeMatcher + Send + Sync> {
        Self::is_exact(SedonaType::Arrow(data_type))
    }

    /// Matches the given [SedonaType] using PartialEq
    pub fn is_exact(exact_type: SedonaType) -> Arc<dyn TypeMatcher + Send + Sync> {
        Arc::new(IsExact { exact_type })
    }

    /// Matches any geography or geometry argument without considering Crs
    pub fn is_geometry_or_geography() -> Arc<dyn TypeMatcher + Send + Sync> {
        Arc::new(IsGeometryOrGeography {})
    }

    /// Matches any geometry argument without considering Crs
    pub fn is_geometry() -> Arc<dyn TypeMatcher + Send + Sync> {
        Arc::new(IsGeometry {})
    }

    /// Matches any geography argument without considering Crs
    pub fn is_geography() -> Arc<dyn TypeMatcher + Send + Sync> {
        Arc::new(IsGeography {})
    }

    /// Matches any argument that is an item-level Crs type
    pub fn is_item_crs() -> Arc<dyn TypeMatcher + Send + Sync> {
        Arc::new(IsItemCrs {})
    }

    /// Matches any raster argument
    pub fn is_raster() -> Arc<dyn TypeMatcher + Send + Sync> {
        Self::is_exact(RASTER)
    }

    /// Matches any [`SedonaType::Extension`] with the given extension name.
    ///
    /// Prototype convenience for user-defined types (see
    /// `crate::extension_type_registry`) -- lets a kernel declare "I accept
    /// the Tensor type" the same way `is_raster()` declares "I accept
    /// Raster", without `SedonaType` needing a dedicated variant per type.
    pub fn is_extension(extension_name: &'static str) -> Arc<dyn TypeMatcher + Send + Sync> {
        Arc::new(IsExtension { extension_name })
    }

    /// Matches a null argument
    pub fn is_null() -> Arc<dyn TypeMatcher + Send + Sync> {
        Arc::new(IsNull {})
    }

    /// Matches any numeric argument
    pub fn is_numeric() -> Arc<dyn TypeMatcher + Send + Sync> {
        Arc::new(IsNumeric {})
    }

    /// Matches any integer argument
    pub fn is_integer() -> Arc<dyn TypeMatcher + Send + Sync> {
        Arc::new(IsInteger {})
    }

    /// Matches a list whose element type satisfies `item_matcher`, covering
    /// `List`/`LargeList`/`ListView`/`LargeListView`/`FixedSizeList`. For
    /// example, `is_list_of(is_integer())` matches Spark's `Array[Int]`.
    pub fn is_list_of(
        item_matcher: Arc<dyn TypeMatcher + Send + Sync>,
    ) -> Arc<dyn TypeMatcher + Send + Sync> {
        Arc::new(IsListOf { item_matcher })
    }

    /// Matches any string argument
    pub fn is_string() -> Arc<dyn TypeMatcher + Send + Sync> {
        Arc::new(IsString {})
    }

    /// Matches any binary argument
    pub fn is_binary() -> Arc<dyn TypeMatcher + Send + Sync> {
        Arc::new(IsBinary {})
    }

    /// Matches any boolean argument
    pub fn is_boolean() -> Arc<dyn TypeMatcher + Send + Sync> {
        Arc::new(IsBoolean {})
    }

    /// Matches any argument that is optional
    pub fn optional(
        matcher: Arc<dyn TypeMatcher + Send + Sync>,
    ) -> Arc<dyn TypeMatcher + Send + Sync> {
        Arc::new(OptionalMatcher { inner: matcher })
    }

    /// Matches if any of the given matchers match
    pub fn or(
        matchers: Vec<Arc<dyn TypeMatcher + Send + Sync>>,
    ) -> Arc<dyn TypeMatcher + Send + Sync> {
        Arc::new(OrMatcher { matchers })
    }
}

/// A TypeMatcher is a predicate on a [SedonaType]
///
/// TypeMatchers are the building blocks of an [ArgMatcher] that
/// represent a single argument. This is a generalization of the
/// DataFusion [Signature] which does not currently consider
/// extension types and/or how extension arrays might be casted
/// to conform to a function with a given signature.
pub trait TypeMatcher: Debug {
    /// Returns true if this matcher matches a type
    fn match_type(&self, arg: &SedonaType) -> bool;

    /// If this argument is optional, return true
    fn optional(&self) -> bool {
        false
    }

    /// Return the type to which an argument should be casted,
    /// if applicable. This can be used to generalize null handling
    /// or casting.
    fn type_if_null(&self) -> Option<SedonaType> {
        None
    }
}

#[derive(Debug)]
struct IsAny;

impl TypeMatcher for IsAny {
    fn match_type(&self, _arg: &SedonaType) -> bool {
        true
    }
}

#[derive(Debug)]
struct IsExact {
    exact_type: SedonaType,
}

impl TypeMatcher for IsExact {
    fn match_type(&self, arg: &SedonaType) -> bool {
        self.exact_type.match_signature(arg)
    }

    fn type_if_null(&self) -> Option<SedonaType> {
        Some(self.exact_type.clone())
    }
}

#[derive(Debug)]
struct OptionalMatcher {
    inner: Arc<dyn TypeMatcher + Send + Sync>,
}

impl TypeMatcher for OptionalMatcher {
    fn match_type(&self, arg: &SedonaType) -> bool {
        self.inner.match_type(arg)
    }

    fn optional(&self) -> bool {
        true
    }

    fn type_if_null(&self) -> Option<SedonaType> {
        self.inner.type_if_null()
    }
}

#[derive(Debug)]
struct OrMatcher {
    matchers: Vec<Arc<dyn TypeMatcher + Send + Sync>>,
}

impl TypeMatcher for OrMatcher {
    fn match_type(&self, arg: &SedonaType) -> bool {
        self.matchers.iter().any(|m| m.match_type(arg))
    }

    fn type_if_null(&self) -> Option<SedonaType> {
        None
    }
}

#[derive(Debug)]
struct IsGeometryOrGeography {}

impl TypeMatcher for IsGeometryOrGeography {
    fn match_type(&self, arg: &SedonaType) -> bool {
        matches!(arg, SedonaType::Wkb(_, _) | SedonaType::WkbView(_, _))
    }
}

#[derive(Debug)]
struct IsGeometry {}

impl TypeMatcher for IsGeometry {
    fn match_type(&self, arg: &SedonaType) -> bool {
        match arg {
            SedonaType::Wkb(edges, _) | SedonaType::WkbView(edges, _) => {
                matches!(edges, Edges::Planar)
            }
            _ => false,
        }
    }

    fn type_if_null(&self) -> Option<SedonaType> {
        Some(WKB_GEOMETRY)
    }
}

#[derive(Debug)]
struct IsGeography {}

impl TypeMatcher for IsGeography {
    fn match_type(&self, arg: &SedonaType) -> bool {
        match arg {
            SedonaType::Wkb(edges, _) | SedonaType::WkbView(edges, _) => {
                matches!(edges, Edges::Spherical)
            }
            _ => false,
        }
    }

    fn type_if_null(&self) -> Option<SedonaType> {
        Some(WKB_GEOGRAPHY)
    }
}

#[derive(Debug)]
struct IsItemCrs {}

impl TypeMatcher for IsItemCrs {
    fn match_type(&self, arg: &SedonaType) -> bool {
        arg.is_item_crs()
    }
}

#[derive(Debug)]
struct IsNumeric {}

impl TypeMatcher for IsNumeric {
    fn match_type(&self, arg: &SedonaType) -> bool {
        match arg {
            SedonaType::Arrow(data_type) => data_type.is_numeric(),
            _ => false,
        }
    }

    fn type_if_null(&self) -> Option<SedonaType> {
        Some(SedonaType::Arrow(DataType::Float64))
    }
}

#[derive(Debug)]
struct IsInteger {}

impl TypeMatcher for IsInteger {
    fn match_type(&self, arg: &SedonaType) -> bool {
        match arg {
            SedonaType::Arrow(data_type) => data_type.is_integer(),
            _ => false,
        }
    }

    fn type_if_null(&self) -> Option<SedonaType> {
        Some(SedonaType::Arrow(DataType::Int64))
    }
}

#[derive(Debug)]
struct IsListOf {
    item_matcher: Arc<dyn TypeMatcher + Send + Sync>,
}

impl TypeMatcher for IsListOf {
    fn match_type(&self, arg: &SedonaType) -> bool {
        let SedonaType::Arrow(data_type) = arg else {
            return false;
        };
        let item_field = match data_type {
            DataType::List(field)
            | DataType::LargeList(field)
            | DataType::ListView(field)
            | DataType::LargeListView(field)
            | DataType::FixedSizeList(field, _) => field,
            _ => return false,
        };
        self.item_matcher
            .match_type(&SedonaType::Arrow(item_field.data_type().clone()))
    }

    fn type_if_null(&self) -> Option<SedonaType> {
        let item_field = self
            .item_matcher
            .type_if_null()?
            .to_storage_field("item", true)
            .ok()?;
        Some(SedonaType::Arrow(DataType::List(Arc::new(item_field))))
    }
}

#[derive(Debug)]
struct IsString {}

impl TypeMatcher for IsString {
    fn match_type(&self, arg: &SedonaType) -> bool {
        match arg {
            SedonaType::Arrow(data_type) => {
                matches!(
                    data_type,
                    DataType::Utf8 | DataType::Utf8View | DataType::LargeUtf8
                )
            }
            _ => false,
        }
    }

    fn type_if_null(&self) -> Option<SedonaType> {
        Some(SedonaType::Arrow(DataType::Utf8))
    }
}

#[derive(Debug)]
struct IsBinary {}

impl TypeMatcher for IsBinary {
    fn match_type(&self, arg: &SedonaType) -> bool {
        match arg {
            SedonaType::Arrow(data_type) => {
                matches!(data_type, DataType::Binary | DataType::BinaryView)
            }
            _ => false,
        }
    }

    fn type_if_null(&self) -> Option<SedonaType> {
        Some(SedonaType::Arrow(DataType::Binary))
    }
}

#[derive(Debug)]
struct IsBoolean {}

impl TypeMatcher for IsBoolean {
    fn match_type(&self, arg: &SedonaType) -> bool {
        match arg {
            SedonaType::Arrow(data_type) => {
                matches!(data_type, DataType::Boolean)
            }
            _ => false,
        }
    }

    fn type_if_null(&self) -> Option<SedonaType> {
        Some(SedonaType::Arrow(DataType::Boolean))
    }
}

#[derive(Debug)]
struct IsNull {}
impl TypeMatcher for IsNull {
    fn match_type(&self, arg: &SedonaType) -> bool {
        matches!(arg, SedonaType::Arrow(DataType::Null))
    }
}

#[derive(Debug)]
struct IsExtension {
    extension_name: &'static str,
}

impl TypeMatcher for IsExtension {
    fn match_type(&self, arg: &SedonaType) -> bool {
        matches!(arg, SedonaType::Extension(ext) if ext.extension_name() == self.extension_name)
    }
}

#[cfg(test)]
mod tests {
    use arrow_schema::Field;
    use std::any::Any;
    use std::sync::LazyLock;

    use crate::datatypes::{WKB_GEOGRAPHY, WKB_GEOMETRY};
    use crate::extension_type_registry::SedonaExtensionType;

    use super::*;

    /// A toy "Tensor"-shaped extension type -- just enough to prove a
    /// function signature can be declared against `SedonaType::Extension`
    /// and dispatched the same way `is_raster()`/`is_geometry()` are today.
    #[derive(Debug, PartialEq)]
    struct ToyTensorType {
        dtype: &'static str,
    }

    static TOY_TENSOR_STORAGE_TYPE: LazyLock<DataType> =
        LazyLock::new(|| DataType::Struct(vec![Field::new("dtype", DataType::Utf8, false)].into()));

    impl SedonaExtensionType for ToyTensorType {
        fn extension_name(&self) -> &'static str {
            "sedona.test.toy_tensor"
        }

        fn storage_type(&self) -> &DataType {
            &TOY_TENSOR_STORAGE_TYPE
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn dyn_eq(&self, other: &dyn SedonaExtensionType) -> bool {
            other.as_any().downcast_ref::<Self>() == Some(self)
        }
    }

    fn toy_tensor(dtype: &'static str) -> SedonaType {
        SedonaType::Extension(Arc::new(ToyTensorType { dtype }))
    }

    #[test]
    fn is_extension_matches_the_named_extension_type_only() {
        let matcher = ArgMatcher::is_extension("sedona.test.toy_tensor");
        assert!(matcher.match_type(&toy_tensor("float64")));
        assert!(!matcher.match_type(&toy_tensor_with_other_name()));
        assert!(!matcher.match_type(&RASTER));
        assert!(!matcher.match_type(&WKB_GEOMETRY));
        assert!(!matcher.match_type(&SedonaType::Arrow(DataType::Int32)));

        fn toy_tensor_with_other_name() -> SedonaType {
            #[derive(Debug, PartialEq)]
            struct OtherType;
            impl SedonaExtensionType for OtherType {
                fn extension_name(&self) -> &'static str {
                    "sedona.test.other"
                }
                fn storage_type(&self) -> &DataType {
                    &TOY_TENSOR_STORAGE_TYPE
                }
                fn as_any(&self) -> &dyn Any {
                    self
                }
                fn dyn_eq(&self, other: &dyn SedonaExtensionType) -> bool {
                    other.as_any().downcast_ref::<Self>() == Some(self)
                }
            }
            SedonaType::Extension(Arc::new(OtherType))
        }
    }

    /// The end-to-end proof this is meant to unblock: a two-argument
    /// signature (`TN_Add(Tensor, Tensor)`-shaped) declared with
    /// `ArgMatcher::is_extension`, matched and given a return type, exactly
    /// like a real kernel would use `is_raster()`/`is_geometry()` today --
    /// with zero changes to the core `SedonaType` enum needed for this new
    /// type to participate.
    #[test]
    fn extension_type_flows_through_a_two_arg_signature_like_a_real_kernel() {
        let signature = ArgMatcher::new(
            vec![
                ArgMatcher::is_extension("sedona.test.toy_tensor"),
                ArgMatcher::is_extension("sedona.test.toy_tensor"),
            ],
            toy_tensor("float64"),
        );

        assert!(signature.matches(&[toy_tensor("float64"), toy_tensor("float64")]));
        // Wrong type in either position: no match, same as a real kernel's
        // signature correctly declining a Raster/Wkb argument.
        assert!(!signature.matches(&[toy_tensor("float64"), RASTER]));
        assert!(!signature.matches(&[RASTER, toy_tensor("float64")]));

        let resolved = signature
            .match_args(&[toy_tensor("float64"), toy_tensor("float64")])
            .unwrap();
        assert_eq!(resolved, Some(toy_tensor("float64")));
    }

    #[test]
    fn extension_type_equality_and_display_work_without_a_core_variant_match() {
        let a = toy_tensor("float64");
        let b = toy_tensor("float64");
        let c = toy_tensor("int32");
        assert!(a.match_signature(&b));
        assert!(!a.match_signature(&c));
        assert_eq!(a.to_string(), "sedona.test.toy_tensor");
        assert_eq!(a.logical_type_name(), "sedona.test.toy_tensor");
    }

    #[test]
    fn matchers() {
        assert!(ArgMatcher::is_arrow(DataType::Null).match_type(&SedonaType::Arrow(DataType::Null)));

        assert!(ArgMatcher::is_geometry_or_geography().match_type(&WKB_GEOMETRY));
        assert!(ArgMatcher::is_geometry_or_geography().match_type(&WKB_GEOGRAPHY));
        assert!(!ArgMatcher::is_geometry_or_geography()
            .match_type(&SedonaType::Arrow(DataType::Binary)));
        assert_eq!(ArgMatcher::is_geometry_or_geography().type_if_null(), None);

        assert!(ArgMatcher::is_geometry().match_type(&WKB_GEOMETRY));
        assert!(!ArgMatcher::is_geometry().match_type(&WKB_GEOGRAPHY));
        assert_eq!(ArgMatcher::is_geometry().type_if_null(), Some(WKB_GEOMETRY));

        assert!(ArgMatcher::is_geography().match_type(&WKB_GEOGRAPHY));
        assert!(!ArgMatcher::is_geography().match_type(&WKB_GEOMETRY));
        assert_eq!(
            ArgMatcher::is_geography().type_if_null(),
            Some(WKB_GEOGRAPHY)
        );

        assert!(ArgMatcher::is_numeric().match_type(&SedonaType::Arrow(DataType::Int32)));
        assert!(ArgMatcher::is_numeric().match_type(&SedonaType::Arrow(DataType::Float64)));
        assert_eq!(
            ArgMatcher::is_numeric().type_if_null(),
            Some(SedonaType::Arrow(DataType::Float64))
        );

        assert!(ArgMatcher::is_integer().match_type(&SedonaType::Arrow(DataType::UInt32)));
        assert!(ArgMatcher::is_integer().match_type(&SedonaType::Arrow(DataType::Int32)));
        assert!(!ArgMatcher::is_integer().match_type(&SedonaType::Arrow(DataType::Float64)));

        let list_of = |item: DataType| {
            SedonaType::Arrow(DataType::List(Arc::new(Field::new("item", item, true))))
        };
        let is_integer_list = || ArgMatcher::is_list_of(ArgMatcher::is_integer());
        assert!(is_integer_list().match_type(&list_of(DataType::Int32)));
        assert!(is_integer_list().match_type(&list_of(DataType::Int64)));
        assert!(
            is_integer_list().match_type(&SedonaType::Arrow(DataType::LargeList(Arc::new(
                Field::new("item", DataType::Int32, true)
            ))))
        );
        assert!(
            is_integer_list().match_type(&SedonaType::Arrow(DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Int32, true)),
                2
            )))
        );
        // A list of a non-integer element, and a bare integer, must not match.
        assert!(!is_integer_list().match_type(&list_of(DataType::Float64)));
        assert!(!is_integer_list().match_type(&SedonaType::Arrow(DataType::Int32)));
        assert_eq!(
            is_integer_list().type_if_null(),
            Some(list_of(DataType::Int64))
        );
        // The item matcher is honored: a list of strings matches is_list_of(is_string).
        assert!(
            ArgMatcher::is_list_of(ArgMatcher::is_string()).match_type(&list_of(DataType::Utf8))
        );
        assert!(
            !ArgMatcher::is_list_of(ArgMatcher::is_string()).match_type(&list_of(DataType::Int32))
        );

        assert!(ArgMatcher::is_string().match_type(&SedonaType::Arrow(DataType::Utf8)));
        assert!(ArgMatcher::is_string().match_type(&SedonaType::Arrow(DataType::Utf8View)));
        assert!(ArgMatcher::is_string().match_type(&SedonaType::Arrow(DataType::LargeUtf8)));
        assert!(!ArgMatcher::is_string().match_type(&SedonaType::Arrow(DataType::Binary)));
        assert_eq!(
            ArgMatcher::is_string().type_if_null(),
            Some(SedonaType::Arrow(DataType::Utf8))
        );

        assert!(ArgMatcher::is_binary().match_type(&SedonaType::Arrow(DataType::Binary)));
        assert!(ArgMatcher::is_binary().match_type(&SedonaType::Arrow(DataType::BinaryView)));
        assert!(!ArgMatcher::is_binary().match_type(&SedonaType::Arrow(DataType::Utf8)));
        assert_eq!(
            ArgMatcher::is_binary().type_if_null(),
            Some(SedonaType::Arrow(DataType::Binary))
        );

        assert!(ArgMatcher::is_boolean().match_type(&SedonaType::Arrow(DataType::Boolean)));
        assert!(!ArgMatcher::is_boolean().match_type(&SedonaType::Arrow(DataType::Int32)));

        assert!(ArgMatcher::is_null().match_type(&SedonaType::Arrow(DataType::Null)));
        assert!(!ArgMatcher::is_null().match_type(&SedonaType::Arrow(DataType::Int32)));
        assert_eq!(
            ArgMatcher::is_boolean().type_if_null(),
            Some(SedonaType::Arrow(DataType::Boolean))
        );

        assert!(ArgMatcher::is_raster().match_type(&RASTER));
        assert!(!ArgMatcher::is_raster().match_type(&SedonaType::Arrow(DataType::Int32)));
        assert!(!ArgMatcher::is_raster().match_type(&WKB_GEOMETRY));
    }

    #[test]
    fn optional_matcher() {
        let matcher = ArgMatcher::new(
            vec![
                ArgMatcher::is_geometry(),
                ArgMatcher::optional(ArgMatcher::is_boolean()),
                ArgMatcher::optional(ArgMatcher::is_numeric()),
            ],
            SedonaType::Arrow(DataType::Null),
        );

        // Match with all args present and matching
        assert!(matcher.matches(&[
            WKB_GEOMETRY,
            SedonaType::Arrow(DataType::Boolean),
            SedonaType::Arrow(DataType::Int32)
        ]));

        // Match when first argument present, second is None
        assert!(matcher.matches(&[WKB_GEOMETRY]));

        // Match when skip an optional arg
        assert!(matcher.matches(&[WKB_GEOMETRY, SedonaType::Arrow(DataType::Int32)]));

        // No match when first is None, second is present
        assert!(!matcher.matches(&[SedonaType::Arrow(DataType::Boolean)]));

        // No match when second argument is incorrect type
        assert!(!matcher.matches(&[WKB_GEOMETRY, WKB_GEOMETRY]));

        // No match when first argument is incorrect type
        assert!(!matcher.matches(&[
            SedonaType::Arrow(DataType::Boolean),
            SedonaType::Arrow(DataType::Boolean)
        ]));

        // No match when too many arguments
        assert!(!matcher.matches(&[
            WKB_GEOGRAPHY,
            SedonaType::Arrow(DataType::Boolean),
            SedonaType::Arrow(DataType::Int32),
            SedonaType::Arrow(DataType::Int32)
        ]));
    }

    #[test]
    fn or_matcher() {
        let matcher = ArgMatcher::new(
            vec![
                ArgMatcher::is_geometry(),
                ArgMatcher::or(vec![ArgMatcher::is_boolean(), ArgMatcher::is_numeric()]),
            ],
            SedonaType::Arrow(DataType::Null),
        );

        // Matches first arg
        assert!(matcher.matches(&[WKB_GEOMETRY, SedonaType::Arrow(DataType::Boolean),]));

        // Matches second arg
        assert!(matcher.matches(&[WKB_GEOMETRY, SedonaType::Arrow(DataType::Int32)]));

        // No match when second arg is incorrect type
        assert!(!matcher.matches(&[WKB_GEOMETRY, WKB_GEOMETRY]));

        // No match when first arg is incorrect type
        assert!(!matcher.matches(&[
            SedonaType::Arrow(DataType::Boolean),
            SedonaType::Arrow(DataType::Boolean)
        ]));

        // Return type if null
        assert_eq!(
            ArgMatcher::or(vec![ArgMatcher::is_boolean(), ArgMatcher::is_numeric()]).type_if_null(),
            None
        );
    }

    #[test]
    fn arg_matcher_matches_null() {
        for type_matcher in [
            ArgMatcher::is_arrow(DataType::Null),
            ArgMatcher::is_arrow(DataType::Float32),
            ArgMatcher::is_geometry_or_geography(),
            ArgMatcher::is_geometry(),
            ArgMatcher::is_geography(),
            ArgMatcher::is_numeric(),
            ArgMatcher::is_string(),
            ArgMatcher::is_binary(),
            ArgMatcher::is_boolean(),
            ArgMatcher::optional(ArgMatcher::is_numeric()),
        ] {
            let matcher = ArgMatcher::new(vec![type_matcher], SedonaType::Arrow(DataType::Null));
            assert!(matcher.matches(&[SedonaType::Arrow(DataType::Null)]));
        }
    }
}
