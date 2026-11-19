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

//! Out-db chunk references and codec pipelines.
//!
//! An out-db band is a regular grid of chunks over its `source_shape`. Each
//! element of the band's `chunks` list names one object, or a byte range of
//! one, and the band's `codecs` — a Zarr v3 codec pipeline as JSON, shared by
//! every chunk of the band — says how the referenced bytes decode into one
//! chunk of `chunk_shape` pixels. The chunk fields are those of the Iceberg V4
//! `file` type (`uri`, `offset`, `size`, `content_type`, `checksum`,
//! `inline`); `inline` is always null, since in-db pixels live in the band's
//! `data`.
//!
//! Loaders dispatch on the pipeline's codec *names*
//! ([`AsyncRasterLoader::supports`](crate::raster_loader::AsyncRasterLoader::supports)).
//! Formats that are not described chunk by chunk stay readable through a
//! *container codec*: a single codec whose input is one whole object and whose
//! output is the band's raw pixels. [`GDAL_CODEC`] is the one defined here;
//! its configuration selects the source band (and optionally the driver and
//! subdataset) that used to ride in a `#band=N` URI fragment.

use arrow_array::{Array, BinaryViewArray, Int64Array, StringViewArray, StructArray};
use arrow_buffer::NullBuffer;
use datafusion_common::cast::{as_binary_view_array, as_int64_array, as_string_view_array};
use sedona_schema::raster::chunk_indices;
use serde_json::{Map, Value, json};

use crate::error::RasterError;

/// Name of the container codec that decodes one whole object with GDAL.
///
/// Configuration: `{"band": N}` selects the 1-based source band (required);
/// `"driver"` (a GDAL driver short name) skips format probing; `"subdataset"`
/// names the variable of a container format (NetCDF, HDF5, GRIB);
/// `"open_options"` are passed to GDAL unchanged. A band on this codec has one
/// chunk whose `chunk_shape` equals its `source_shape`.
pub const GDAL_CODEC: &str = "sedona.gdal";

/// One element of a band's `chunks` list, borrowed from its Arrow arrays.
///
/// `uri` names the object. `offset` and `size` select a byte range of it;
/// `offset` requires `size`, and a chunk without `offset` is the whole object.
/// `checksum` is `ETAG:<etag>` of the object the chunk was anchored against,
/// so a changed object fails the read instead of serving stale offsets.
/// `content_type` is carried but unused: `codecs` decides decoding. `inline`
/// is not exposed because it is always null; [`Chunks::inline_is_set`] lets a
/// reader reject a row that sets it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ChunkRef<'a> {
    pub uri: Option<&'a str>,
    pub offset: Option<i64>,
    pub size: Option<i64>,
    pub content_type: Option<&'a str>,
    pub checksum: Option<&'a str>,
}

impl<'a> ChunkRef<'a> {
    /// A chunk that is one whole object.
    pub fn whole(uri: &'a str) -> Self {
        Self {
            uri: Some(uri),
            ..Default::default()
        }
    }

    /// A chunk that is the `size` bytes of `uri` starting at `offset`.
    pub fn range(uri: &'a str, offset: i64, size: i64) -> Self {
        Self {
            uri: Some(uri),
            offset: Some(offset),
            size: Some(size),
            ..Default::default()
        }
    }

    /// The URI, or an error naming the problem: a non-null chunk must point
    /// somewhere.
    pub fn require_uri(&self) -> Result<&'a str, RasterError> {
        self.uri
            .ok_or_else(|| RasterError::Invalid("chunk has no uri".to_string()))
    }

    /// Check one non-null chunk: it has a `uri`, `offset` comes with `size`,
    /// and neither is negative.
    pub fn validate(&self) -> Result<(), RasterError> {
        self.require_uri()?;
        match (self.offset, self.size) {
            (Some(_), None) => {
                return Err(RasterError::Invalid(
                    "chunk offset requires size".to_string(),
                ));
            }
            (Some(offset), _) if offset < 0 => {
                return Err(RasterError::Invalid(format!(
                    "chunk offset must not be negative, got {offset}"
                )));
            }
            (_, Some(size)) if size < 0 => {
                return Err(RasterError::Invalid(format!(
                    "chunk size must not be negative, got {size}"
                )));
            }
            _ => {}
        }
        Ok(())
    }

    /// True when the chunk is a whole object rather than a byte range.
    pub fn is_whole_object(&self) -> bool {
        self.offset.is_none()
    }
}

/// A band's `chunks` list, borrowed from the flattened chunk arrays.
///
/// Zero-copy: `get` reads straight from the Arrow buffers. Constructed by the
/// array-backed band and raster accessors; other `BandRef` implementations
/// return `None` from [`crate::traits::BandRef::chunks`].
#[derive(Debug, Clone, Copy)]
pub struct Chunks<'a> {
    uri: &'a StringViewArray,
    offset: &'a Int64Array,
    size: &'a Int64Array,
    content_type: &'a StringViewArray,
    checksum: &'a StringViewArray,
    inline: &'a BinaryViewArray,
    /// Element validity of the chunk struct (null entry = missing chunk).
    nulls: Option<&'a NullBuffer>,
    start: usize,
    len: usize,
}

/// The chunk struct's child arrays, downcast once per [`StructArray`].
#[derive(Debug, Clone, Copy)]
pub struct ChunkColumns<'a> {
    pub uri: &'a StringViewArray,
    pub offset: &'a Int64Array,
    pub size: &'a Int64Array,
    pub content_type: &'a StringViewArray,
    pub checksum: &'a StringViewArray,
    pub inline: &'a BinaryViewArray,
    pub nulls: Option<&'a NullBuffer>,
}

impl<'a> ChunkColumns<'a> {
    /// Downcast the children of a chunk `StructArray` (the values of a band's
    /// `chunks` list).
    pub fn try_new(chunks: &'a StructArray) -> Result<Self, RasterError> {
        if chunks.fields().len() != chunk_indices::FIELD_COUNT {
            return Err(RasterError::Invalid(
                "Unexpected column count for chunk array".to_string(),
            ));
        }
        Ok(Self {
            uri: as_string_view_array(chunks.column(chunk_indices::URI))?,
            offset: as_int64_array(chunks.column(chunk_indices::OFFSET))?,
            size: as_int64_array(chunks.column(chunk_indices::SIZE))?,
            content_type: as_string_view_array(chunks.column(chunk_indices::CONTENT_TYPE))?,
            checksum: as_string_view_array(chunks.column(chunk_indices::CHECKSUM))?,
            inline: as_binary_view_array(chunks.column(chunk_indices::INLINE))?,
            nulls: chunks.nulls(),
        })
    }

    /// The `len` chunks starting at flattened row `start`.
    pub fn slice(&self, start: usize, len: usize) -> Chunks<'a> {
        Chunks {
            uri: self.uri,
            offset: self.offset,
            size: self.size,
            content_type: self.content_type,
            checksum: self.checksum,
            inline: self.inline,
            nulls: self.nulls,
            start,
            len,
        }
    }
}

impl<'a> Chunks<'a> {
    /// Number of grid positions, null (missing) entries included.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// True when entry `i` is a missing chunk.
    pub fn is_null(&self, i: usize) -> bool {
        assert!(i < self.len, "chunk index {i} out of range ({})", self.len);
        self.nulls
            .is_some_and(|nulls| nulls.is_null(self.start + i))
    }

    /// True when entry `i` sets `inline`, which no writer does; a reader
    /// fails such a row rather than keep two sources for a band's pixels.
    pub fn inline_is_set(&self, i: usize) -> bool {
        assert!(i < self.len, "chunk index {i} out of range ({})", self.len);
        !self.is_null(i) && self.inline.is_valid(self.start + i)
    }

    /// Entry `i`, or `None` for a missing chunk. Panics when `i >= len()`.
    pub fn get(&self, i: usize) -> Option<ChunkRef<'a>> {
        if self.is_null(i) {
            return None;
        }
        let row = self.start + i;
        let str_at = |array: &'a StringViewArray| -> Option<&'a str> {
            if array.is_null(row) {
                None
            } else {
                Some(array.value(row))
            }
        };
        let int_at = |array: &'a Int64Array| -> Option<i64> {
            if array.is_null(row) {
                None
            } else {
                Some(array.value(row))
            }
        };
        Some(ChunkRef {
            uri: str_at(self.uri),
            offset: int_at(self.offset),
            size: int_at(self.size),
            content_type: str_at(self.content_type),
            checksum: str_at(self.checksum),
        })
    }

    /// Every entry in grid order; `None` marks a missing chunk.
    pub fn iter(&self) -> impl Iterator<Item = Option<ChunkRef<'a>>> + '_ {
        (0..self.len).map(move |i| self.get(i))
    }

    /// Every entry in grid order, owned. [`ChunkRef`] borrows from the
    /// arrays, so this allocates only the `Vec`.
    pub fn to_vec(&self) -> Vec<Option<ChunkRef<'a>>> {
        self.iter().collect()
    }

    /// The one chunk of a band whose codec decodes a whole object: the list
    /// has exactly one non-null entry that is a whole object. Container
    /// codecs such as [`GDAL_CODEC`] read bands of this shape only.
    pub fn single_whole_object(&self) -> Result<ChunkRef<'a>, RasterError> {
        if self.len != 1 {
            return Err(RasterError::Invalid(format!(
                "expected a single-chunk band, got {} chunks",
                self.len
            )));
        }
        let chunk = self
            .get(0)
            .ok_or_else(|| RasterError::Invalid("the band's only chunk is missing".to_string()))?;
        if !chunk.is_whole_object() {
            return Err(RasterError::Invalid(
                "expected a whole-object chunk, got a byte range".to_string(),
            ));
        }
        Ok(chunk)
    }
}

/// One codec of a pipeline: a registered or URI name plus its configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct CodecSpec {
    pub name: String,
    pub configuration: Option<Map<String, Value>>,
}

impl CodecSpec {
    pub fn new(name: impl Into<String>, configuration: Option<Map<String, Value>>) -> Self {
        Self {
            name: name.into(),
            configuration,
        }
    }

    /// The codec as a JSON object with its keys in sorted order. `serde_json`
    /// keeps insertion order when any crate in the build enables its
    /// `preserve_order` feature (zarrs does) and sorts otherwise, so keys are
    /// inserted sorted to make the stored string the same either way.
    fn to_value(&self) -> Value {
        let mut object = Map::new();
        if let Some(configuration) = &self.configuration {
            let mut entries: Vec<(&String, &Value)> = configuration.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            let mut sorted = Map::new();
            for (key, value) in entries {
                sorted.insert(key.clone(), value.clone());
            }
            object.insert("configuration".to_string(), Value::Object(sorted));
        }
        object.insert("name".to_string(), Value::String(self.name.clone()));
        Value::Object(object)
    }
}

/// A band's `codecs`: a Zarr v3 codec pipeline, parsed from its JSON form.
///
/// Only the shape is checked here (an array of objects with a string `name`
/// and an optional object `configuration`). Whether the names are supported
/// is each loader's call.
#[derive(Debug, Clone, PartialEq)]
pub struct CodecPipeline {
    codecs: Vec<CodecSpec>,
}

impl CodecPipeline {
    pub fn new(codecs: Vec<CodecSpec>) -> Self {
        Self { codecs }
    }

    /// Parse the JSON form.
    pub fn parse(json: &str) -> Result<Self, RasterError> {
        let value: Value = serde_json::from_str(json)
            .map_err(|e| RasterError::Invalid(format!("codecs is not valid JSON: {e}")))?;
        let Value::Array(items) = value else {
            return Err(RasterError::Invalid(
                "codecs must be a JSON array of codec objects".to_string(),
            ));
        };
        let mut codecs = Vec::with_capacity(items.len());
        for item in items {
            let Value::Object(object) = item else {
                return Err(RasterError::Invalid(
                    "each codec must be a JSON object".to_string(),
                ));
            };
            let name = match object.get("name") {
                Some(Value::String(name)) if !name.is_empty() => name.clone(),
                _ => {
                    return Err(RasterError::Invalid(
                        "each codec needs a non-empty string \"name\"".to_string(),
                    ));
                }
            };
            let configuration = match object.get("configuration") {
                None | Some(Value::Null) => None,
                Some(Value::Object(configuration)) => Some(configuration.clone()),
                Some(_) => {
                    return Err(RasterError::Invalid(format!(
                        "codec {name}: \"configuration\" must be a JSON object"
                    )));
                }
            };
            codecs.push(CodecSpec {
                name,
                configuration,
            });
        }
        Ok(Self { codecs })
    }

    pub fn codecs(&self) -> &[CodecSpec] {
        &self.codecs
    }

    /// The codec names in pipeline order; what loaders dispatch on.
    pub fn names(&self) -> Vec<&str> {
        self.codecs.iter().map(|c| c.name.as_str()).collect()
    }

    /// The pipeline's one codec, when it has exactly one.
    pub fn single(&self) -> Option<&CodecSpec> {
        match self.codecs.as_slice() {
            [codec] => Some(codec),
            _ => None,
        }
    }

    /// The JSON form, as stored in the band.
    pub fn to_json(&self) -> String {
        Value::Array(self.codecs.iter().map(CodecSpec::to_value).collect()).to_string()
    }
}

/// Configuration of the [`GDAL_CODEC`] container codec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GdalCodec {
    /// 1-based source band within the object.
    pub band: u32,
    /// GDAL driver short name (`"GTiff"`), when known. Null lets GDAL probe.
    pub driver: Option<String>,
    /// The variable of a container format (NetCDF, HDF5, GRIB) to open.
    pub subdataset: Option<String>,
    /// GDAL open options, passed through unchanged.
    pub open_options: Vec<String>,
}

impl GdalCodec {
    /// The codec for source band `band` with GDAL probing the format.
    pub fn new(band: u32) -> Self {
        Self {
            band,
            driver: None,
            subdataset: None,
            open_options: Vec::new(),
        }
    }

    pub fn with_driver(mut self, driver: impl Into<String>) -> Self {
        self.driver = Some(driver.into());
        self
    }

    /// This codec as a one-entry pipeline.
    pub fn to_pipeline(&self) -> CodecPipeline {
        let mut configuration = Map::new();
        configuration.insert("band".to_string(), json!(self.band));
        if let Some(driver) = &self.driver {
            configuration.insert("driver".to_string(), Value::String(driver.clone()));
        }
        if let Some(subdataset) = &self.subdataset {
            configuration.insert("subdataset".to_string(), Value::String(subdataset.clone()));
        }
        if !self.open_options.is_empty() {
            configuration.insert("open_options".to_string(), json!(self.open_options));
        }
        CodecPipeline::new(vec![CodecSpec::new(GDAL_CODEC, Some(configuration))])
    }

    /// The JSON a band stores in `codecs` for this codec.
    pub fn to_json(&self) -> String {
        self.to_pipeline().to_json()
    }

    /// Read this codec out of a pipeline. `Ok(None)` when the pipeline is not
    /// a single [`GDAL_CODEC`]; an error when it is but its configuration is
    /// malformed.
    pub fn from_pipeline(pipeline: &CodecPipeline) -> Result<Option<Self>, RasterError> {
        let Some(codec) = pipeline.single() else {
            return Ok(None);
        };
        if codec.name != GDAL_CODEC {
            return Ok(None);
        }
        let configuration = codec.configuration.as_ref().ok_or_else(|| {
            RasterError::Invalid(format!("{GDAL_CODEC} needs a configuration with \"band\""))
        })?;
        let band = match configuration.get("band") {
            Some(Value::Number(n)) => n
                .as_u64()
                .and_then(|b| u32::try_from(b).ok())
                .filter(|&b| b >= 1)
                .ok_or_else(|| {
                    RasterError::Invalid(format!(
                        "{GDAL_CODEC}: \"band\" must be a positive integer in 1..=u32::MAX, got {n}"
                    ))
                })?,
            _ => {
                return Err(RasterError::Invalid(format!(
                    "{GDAL_CODEC}: \"band\" must be a positive integer"
                )));
            }
        };
        let optional_string = |key: &str| -> Result<Option<String>, RasterError> {
            match configuration.get(key) {
                None | Some(Value::Null) => Ok(None),
                Some(Value::String(s)) => Ok(Some(s.clone())),
                Some(_) => Err(RasterError::Invalid(format!(
                    "{GDAL_CODEC}: \"{key}\" must be a string"
                ))),
            }
        };
        let open_options = match configuration.get("open_options") {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::Array(items)) => items
                .iter()
                .map(|item| match item {
                    Value::String(s) => Ok(s.clone()),
                    _ => Err(RasterError::Invalid(format!(
                        "{GDAL_CODEC}: \"open_options\" must be an array of strings"
                    ))),
                })
                .collect::<Result<_, _>>()?,
            Some(_) => {
                return Err(RasterError::Invalid(format!(
                    "{GDAL_CODEC}: \"open_options\" must be an array of strings"
                )));
            }
        };
        Ok(Some(Self {
            band,
            driver: optional_string("driver")?,
            subdataset: optional_string("subdataset")?,
            open_options,
        }))
    }

    /// [`Self::from_pipeline`] on the JSON form.
    pub fn parse(json: &str) -> Result<Option<Self>, RasterError> {
        Self::from_pipeline(&CodecPipeline::parse(json)?)
    }
}

/// Number of chunks in a regular grid of `chunk_shape` over `source_shape`:
/// `Π ceil(source_shape[d] / chunk_shape[d])`. Errors on a non-positive
/// chunk extent or on overflow.
pub fn chunk_grid_size(source_shape: &[i64], chunk_shape: &[i64]) -> Result<usize, RasterError> {
    if source_shape.len() != chunk_shape.len() {
        return Err(RasterError::Invalid(format!(
            "chunk_shape has {} entries but source_shape has {}",
            chunk_shape.len(),
            source_shape.len()
        )));
    }
    let mut total: usize = 1;
    for (&extent, &chunk) in source_shape.iter().zip(chunk_shape) {
        if chunk <= 0 {
            return Err(RasterError::Invalid(format!(
                "chunk_shape entries must be positive, got {chunk}"
            )));
        }
        if extent < 0 {
            return Err(RasterError::Invalid(format!(
                "source_shape entries must not be negative, got {extent}"
            )));
        }
        // Both are non-negative here, so the unsigned ceiling division is exact.
        let along = usize::try_from((extent as u64).div_ceil(chunk as u64))
            .map_err(|_| RasterError::Invalid("chunk grid does not fit in usize".to_string()))?;
        total = total
            .checked_mul(along)
            .ok_or_else(|| RasterError::Invalid("chunk grid size overflows usize".to_string()))?;
    }
    Ok(total)
}

/// Check a band's out-db fields together, as a writer must leave them:
///
/// - no `chunks`: `chunk_shape` and `codecs` are null too (an in-db band);
/// - `chunks` present: `chunk_shape` has one positive entry per dimension,
///   `codecs` parses as a pipeline, the list has exactly one entry per grid
///   position, and every non-null entry passes [`ChunkRef::validate`].
pub fn validate_band_chunks(
    source_shape: &[i64],
    chunk_shape: Option<&[i64]>,
    codecs: Option<&str>,
    chunks: Option<&[Option<ChunkRef<'_>>]>,
) -> Result<(), RasterError> {
    let Some(chunks) = chunks else {
        if chunk_shape.is_some() {
            return Err(RasterError::Invalid(
                "chunk_shape is set on a band without chunks".to_string(),
            ));
        }
        if codecs.is_some() {
            return Err(RasterError::Invalid(
                "codecs is set on a band without chunks".to_string(),
            ));
        }
        return Ok(());
    };
    let chunk_shape = chunk_shape.ok_or_else(|| {
        RasterError::Invalid("a band with chunks needs a chunk_shape".to_string())
    })?;
    let codecs = codecs
        .ok_or_else(|| RasterError::Invalid("a band with chunks needs codecs".to_string()))?;
    CodecPipeline::parse(codecs)?;
    let grid_size = chunk_grid_size(source_shape, chunk_shape)?;
    if chunks.len() != grid_size {
        return Err(RasterError::Invalid(format!(
            "chunks has {} entries but the chunk grid has {grid_size} positions \
             (source_shape {source_shape:?}, chunk_shape {chunk_shape:?})",
            chunks.len()
        )));
    }
    for (i, chunk) in chunks.iter().enumerate() {
        if let Some(chunk) = chunk {
            chunk
                .validate()
                .map_err(|e| RasterError::Invalid(format!("chunk {i}: {e}")))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gdal_codec_round_trips_through_json() {
        // The exact string matters: it is what a band stores, and Python's
        // `gdal_codecs()` produces the same bytes. Keys are sorted whether or
        // not serde_json's `preserve_order` feature is on in this build.
        let codec = GdalCodec::new(3).with_driver("GTiff");
        let json = codec.to_json();
        assert_eq!(
            json,
            r#"[{"configuration":{"band":3,"driver":"GTiff"},"name":"sedona.gdal"}]"#
        );
        assert_eq!(GdalCodec::parse(&json).unwrap(), Some(codec));

        // A configuration parsed from unsorted input is written back sorted.
        let unsorted = r#"[{"name": "sedona.gdal", "configuration": {"subdataset": "tas", "band": 1, "open_options": ["A=1"], "driver": "netCDF"}}]"#;
        assert_eq!(
            CodecPipeline::parse(unsorted).unwrap().to_json(),
            r#"[{"configuration":{"band":1,"driver":"netCDF","open_options":["A=1"],"subdataset":"tas"},"name":"sedona.gdal"}]"#
        );

        // Band only, as `RS_FromPath` writes for an unknown driver.
        let plain = GdalCodec::new(1);
        assert_eq!(GdalCodec::parse(&plain.to_json()).unwrap(), Some(plain));
    }

    #[test]
    fn gdal_codec_is_none_for_other_pipelines() {
        let zstd =
            r#"[{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "zstd"}]"#;
        assert_eq!(GdalCodec::parse(zstd).unwrap(), None);
        let other = r#"[{"name": "sedona.zarr", "configuration": {"array": "a"}}]"#;
        assert_eq!(GdalCodec::parse(other).unwrap(), None);
    }

    #[test]
    fn gdal_codec_rejects_bad_band() {
        for config in [
            r#"[{"name": "sedona.gdal"}]"#,
            r#"[{"name": "sedona.gdal", "configuration": {}}]"#,
            r#"[{"name": "sedona.gdal", "configuration": {"band": 0}}]"#,
            r#"[{"name": "sedona.gdal", "configuration": {"band": -2}}]"#,
            r#"[{"name": "sedona.gdal", "configuration": {"band": "1"}}]"#,
            r#"[{"name": "sedona.gdal", "configuration": {"band": 4294967296}}]"#,
        ] {
            let err = GdalCodec::parse(config).unwrap_err().to_string();
            assert!(err.contains("band"), "{config}: {err}");
        }
    }

    #[test]
    fn pipeline_parse_checks_shape() {
        let pipeline = CodecPipeline::parse(
            r#"[{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "zstd"}]"#,
        )
        .unwrap();
        assert_eq!(pipeline.names(), ["bytes", "zstd"]);
        assert!(pipeline.single().is_none());
        assert_eq!(
            pipeline.to_json(),
            r#"[{"configuration":{"endian":"little"},"name":"bytes"},{"name":"zstd"}]"#
        );

        for bad in [
            "not json",
            r#"{"name": "bytes"}"#,
            r#"["bytes"]"#,
            r#"[{"configuration": {}}]"#,
            r#"[{"name": ""}]"#,
            r#"[{"name": "bytes", "configuration": "little"}]"#,
        ] {
            assert!(CodecPipeline::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn chunk_grid_size_rounds_up_partial_edges() {
        assert_eq!(chunk_grid_size(&[10, 10], &[10, 10]).unwrap(), 1);
        assert_eq!(chunk_grid_size(&[10, 10], &[4, 4]).unwrap(), 9);
        assert_eq!(
            chunk_grid_size(&[3, 1024, 1024], &[1, 512, 512]).unwrap(),
            12
        );
        assert_eq!(chunk_grid_size(&[0, 4], &[2, 2]).unwrap(), 0);
        assert!(chunk_grid_size(&[4, 4], &[0, 2]).is_err());
        assert!(chunk_grid_size(&[4, 4], &[2]).is_err());
        assert!(chunk_grid_size(&[-1, 4], &[2, 2]).is_err());
    }

    #[test]
    fn chunk_ref_validate() {
        assert!(ChunkRef::whole("s3://b/a.tif").validate().is_ok());
        assert!(ChunkRef::range("s3://b/a.tif", 0, 10).validate().is_ok());
        assert!(ChunkRef::default().validate().is_err());
        let offset_without_size = ChunkRef {
            uri: Some("s3://b/a.tif"),
            offset: Some(4),
            ..Default::default()
        };
        assert!(offset_without_size.validate().is_err());
        assert!(ChunkRef::range("s3://b/a.tif", -1, 10).validate().is_err());
        assert!(ChunkRef::range("s3://b/a.tif", 1, -10).validate().is_err());
    }

    #[test]
    fn validate_band_chunks_rules() {
        let gdal = GdalCodec::new(1).to_json();
        let one = [Some(ChunkRef::whole("s3://b/a.tif"))];

        // In-db: nothing set.
        validate_band_chunks(&[4, 4], None, None, None).unwrap();
        // Stray fields on an in-db band.
        assert!(validate_band_chunks(&[4, 4], Some(&[4, 4]), None, None).is_err());
        assert!(validate_band_chunks(&[4, 4], None, Some(&gdal), None).is_err());

        // Out-db, one whole-object chunk.
        validate_band_chunks(&[4, 4], Some(&[4, 4]), Some(&gdal), Some(&one)).unwrap();
        // Missing pieces.
        assert!(validate_band_chunks(&[4, 4], None, Some(&gdal), Some(&one)).is_err());
        assert!(validate_band_chunks(&[4, 4], Some(&[4, 4]), None, Some(&one)).is_err());
        assert!(validate_band_chunks(&[4, 4], Some(&[4, 4]), Some("nope"), Some(&one)).is_err());
        // Grid mismatch: 2×2 chunks over 4×4 need four entries.
        assert!(validate_band_chunks(&[4, 4], Some(&[2, 2]), Some(&gdal), Some(&one)).is_err());
        let four = [
            Some(ChunkRef::range("s3://b/a.tif", 0, 16)),
            None,
            Some(ChunkRef::range("s3://b/a.tif", 32, 16)),
            Some(ChunkRef::range("s3://b/a.tif", 48, 16)),
        ];
        validate_band_chunks(&[4, 4], Some(&[2, 2]), Some(&gdal), Some(&four)).unwrap();
        // A bad entry names its position.
        let bad = [Some(ChunkRef::default())];
        let err = validate_band_chunks(&[4, 4], Some(&[4, 4]), Some(&gdal), Some(&bad))
            .unwrap_err()
            .to_string();
        assert!(err.contains("chunk 0"), "{err}");
    }
}
