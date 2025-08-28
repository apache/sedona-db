use std::fs::File;

use arrow_array::{ArrayRef, RecordBatchReader};
use datafusion_common::{DataFusionError, Result};
use parquet::arrow::arrow_reader::ParquetRecordBatchReader;
use sedona_common::sedona_internal_err;
use sedona_schema::datatypes::SedonaType;

use crate::data::test_geoparquet;

/// Options for test file readers
#[derive(Debug, Clone)]
pub struct TestReadOptions {
    /// Type to use for geometry when reading test files
    pub sedona_type: SedonaType,

    /// Chunk size to output when reading test files
    pub chunk_size: usize,

    /// Approximate number of rows
    ///
    /// This number is approximate and the actual number will be obtained
    /// by either truncating the input or cycling through batches until
    /// at least this number is reached. If omitted, the entire test file
    /// will be read.
    pub output_size: Option<usize>,
}

impl TestReadOptions {
    /// Create new options with defaults
    pub fn new(sedona_type: SedonaType) -> Self {
        TestReadOptions {
            sedona_type,
            chunk_size: 8192,
            output_size: None,
        }
    }

    /// Apply a target output size to these options
    pub fn with_output_size(self, output_size: usize) -> Self {
        TestReadOptions {
            sedona_type: self.sedona_type,
            chunk_size: self.chunk_size,
            output_size: Some(output_size),
        }
    }
}

/// Read a geoarrow-data file's geometry column
///
/// This function is intended for reading data for benchmarks and tests
pub fn read_geoarrow_data_geometry(
    group: &str,
    name: &str,
    options: &TestReadOptions,
) -> Result<Vec<ArrayRef>> {
    let path = test_geoparquet(group, name)?;
    let file = File::open(path).map_err(DataFusionError::IoError)?;
    let reader = ParquetRecordBatchReader::try_new(file, options.chunk_size)
        .map_err(|e| DataFusionError::External(Box::new(e)))?;

    if reader.schema().fields().is_empty() {
        return sedona_internal_err!("Unexpected schema: zero columns");
    }

    // True for all geoarrow-data files
    let geometry_index = reader.schema().fields().len() - 1;
    let raw_arrays = reader
        .map(|batch| -> Result<ArrayRef> {
            let array = batch?.column(geometry_index).clone();
            // We may need something more sophisticated to support non-wkb geometry types
            // This covers WKB and WKB_VIEW
            let array_casted = arrow_cast::cast(&array, options.sedona_type.storage_type())?;
            options.sedona_type.wrap_array(&array_casted)
        })
        .collect::<Result<Vec<_>>>()?;

    apply_output_size(raw_arrays, options)
}

fn apply_output_size(arrays: Vec<ArrayRef>, options: &TestReadOptions) -> Result<Vec<ArrayRef>> {
    if let Some(output_size) = options.output_size {
        let mut out = Vec::new();
        let mut i = 0;
        let mut out_size = 0;
        while out_size < output_size {
            let array = &arrays[i % arrays.len()];
            out_size += array.len();
            i += 1;
            out.push(array.clone());
        }

        Ok(out)
    } else {
        Ok(arrays)
    }
}

#[cfg(test)]
mod test {
    use sedona_schema::datatypes::WKB_GEOMETRY;

    use super::*;

    #[test]
    fn read() {
        let batches =
            read_geoarrow_data_geometry("example", "geometry", &TestReadOptions::new(WKB_GEOMETRY))
                .unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 9);
        assert!(batches[0].data_type().is_nested());

        let options = TestReadOptions::new(WKB_GEOMETRY).with_output_size(100);
        let batches = read_geoarrow_data_geometry("example", "geometry", &options).unwrap();
        assert_eq!(batches.len(), 12);
    }
}
