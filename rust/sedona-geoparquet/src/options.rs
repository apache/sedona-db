use datafusion::config::TableParquetOptions;

/// [TableParquetOptions] wrapper with GeoParquet-specific options
#[derive(Debug, Default, Clone)]
pub struct TableGeoParquetOptions {
    /// Inner [TableParquetOptions]
    pub inner: TableParquetOptions,
    /// [GeoParquetVersion] to use when writing GeoParquet files
    pub geoparquet_version: GeoParquetVersion,
}

impl From<TableParquetOptions> for TableGeoParquetOptions {
    fn from(value: TableParquetOptions) -> Self {
        Self {
            inner: value,
            geoparquet_version: GeoParquetVersion::default(),
        }
    }
}

/// The GeoParquet Version to write for output with spatial columns
#[derive(Debug, Clone, Copy)]
pub enum GeoParquetVersion {
    /// Write GeoParquet 1.0 metadata
    ///
    /// GeoParquet 1.0 has the widest support among readers and writers; however
    /// it does not include row-group level statistics.
    V1_0,

    /// Write GeoParquet 1.1 metadata and optional bounding box column
    ///
    /// A bbox column will be included for any column where the Parquet options would
    /// have otherwise written statistics (which it will by default).
    /// This option may be more computationally expensive; however, will result in
    /// row-group level statistics that some readers (e.g., SedonaDB) can use to prune
    /// row groups on read.
    V1_1,

    /// Write GeoParquet 2.0
    ///
    /// The GeoParquet 2.0 options is identical to GeoParquet 1.0 except the underlying storage
    /// of spatial columns is Parquet native geometry, where the Parquet writer will include
    /// native statistics according to the underlying Parquet options. Some readers
    /// (e.g., SedonaDB) can use these statistics to prune row groups on read.
    V2_0,

    /// Do not write GeoParquet metadata
    ///
    /// This option suppresses GeoParquet metadata; however, spatial types will be written as
    /// Parquet native Geometry/Geography when this is supported by the underlying writer.
    Omitted,
}

impl Default for GeoParquetVersion {
    fn default() -> Self {
        Self::V1_0
    }
}
